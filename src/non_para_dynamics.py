import torch
import torch.nn as nn
import itertools
import numpy as np
from torch.optim import Adam
from src.utilities import dm2vec
from src.internal_ham_learning_model import LearnableHamiltonian
from typing import Optional, Tuple, Union, List, Dict, Any, Callable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class swish(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class FC(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        weight_decay: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(
            torch.Tensor(ensemble_size, in_features, out_features)
        )
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])


class EnsembleModel(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        reward_size,
        ensemble_size,
        hidden_size=200,
        learning_rate=1e-3,
        use_decay=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.nn1 = FC(
            state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025
        )
        self.nn2 = FC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = FC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = FC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay

        self.output_dim = state_size + reward_size
        # Add variance output
        self.nn5 = FC(
            hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001
        )

        self.max_logvar = nn.Parameter(
            (torch.ones((1, self.output_dim)).float() / 2), requires_grad=False
        )
        self.min_logvar = nn.Parameter(
            (-torch.ones((1, self.output_dim)).float() * 10), requires_grad=False
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = swish()
        self.softplus1 = nn.Softplus()
        self.softplus2 = nn.Softplus()

    def forward(self, x, ret_log_var=False):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        mean = nn5_output[:, :, : self.output_dim]

        logvar = self.max_logvar - nn.functional.softplus(
            self.max_logvar - nn5_output[:, :, self.output_dim :]
        )
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.0
        for m in self.children():
            if isinstance(m, FC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.0
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(
                torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1
            )
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel:
    def __init__(
        self,
        network_size,
        elite_size,
        state_size,
        action_size,
        reward_size=1,
        hidden_size=200,
        use_decay=False,
    ):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.ensemble_model = EnsembleModel(
            state_size,
            action_size,
            reward_size,
            network_size,
            hidden_size,
            use_decay=use_decay,
        )
        self.scaler = StandardScaler()

    def train(
        self,
        inputs,
        labels,
        batch_size=256,
        holdout_ratio=0.0,
        max_epochs_since_update=5,
    ):

        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_inputs = torch.from_numpy(holdout_inputs).float()
        holdout_labels = torch.from_numpy(holdout_labels).float()
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])
        i = 0

        for epoch in itertools.count():

            train_idx = np.vstack(
                [
                    np.random.permutation(train_inputs.shape[0])
                    for _ in range(self.network_size)
                ]
            )

            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos : start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float()
                train_label = torch.from_numpy(train_labels[idx]).float()
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(
                    holdout_inputs, ret_log_var=True
                )
                _, holdout_mse_losses = self.ensemble_model.loss(
                    holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False
                )
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[: self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, batch_size=1024, factored=True):
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(
                inputs[i : min(i + batch_size, inputs.shape[0])]
            ).float()
            b_mean, b_var = self.ensemble_model(
                input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False
            )
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(
                torch.square(ensemble_mean - mean[None, :, :]), dim=0
            )
            return mean, var

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]
        ## [ num_networks, batch_size ]
        log_prob = (
            -1
            / 2
            * (
                k * np.log(2 * np.pi)
                + np.log(variances).sum(-1)
                + (np.power(x - means, 2) / variances).sum(-1)
            )
        )
        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)
        ## [ batch_size ]
        log_prob = np.log(prob)
        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False, debug_model=None):
        "probabilistic rollout w.r.t. the parametric model dynamics distribution"
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)

        ensemble_model_means, ensemble_model_vars = self.predict(inputs)

        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = (
                ensemble_model_means
                + np.random.normal(size=ensemble_model_means.shape)
                * ensemble_model_stds
            )

        num_models, batch_size, _ = ensemble_model_means.shape
        model_idxes = np.random.choice(self.elite_model_idxes, size=batch_size)

        batch_idxes = np.arange(batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        log_prob, dev = self._get_logprob(
            samples, ensemble_model_means, ensemble_model_vars
        )

        rewards, next_obs = samples[:, :1], samples[:, 1:]

        batch_size = model_means.shape[0]

        return_means = np.concatenate((model_means[:, :1], model_means[:, 1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:, :1], model_stds[:, 1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            # terminals = terminals[0]

        info = {
            "mean": return_means,
            "std": return_stds,
            "log_prob": log_prob,
            "dev": dev,
        }
        return next_obs, rewards, info


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(
                cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t
            )
        return t

    if type(m) == nn.Linear or isinstance(m, FC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


def get_model_learning_data(
    inputs,
    labels,
    holdout_ratio,
    network_size,
    timesteps,
    learn_time_dep_ham=False,
    num_actions=2,
):
    num_holdout = int(inputs.shape[0] * holdout_ratio)
    permutation = np.random.permutation(inputs.shape[0])
    inputs, labels = inputs[permutation], labels[permutation]
    train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
    holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

    (
        holdout_input_actions,
        holdout_input_states,
        holdout_label_states,
        holdout_label_rewards,
        rank,
    ) = make_holdout_dataset(holdout_inputs, holdout_labels, network_size, num_actions)
    if learn_time_dep_ham:
        timesteps = timesteps[permutation]
        train_timesteps, holdout_timesteps = (
            timesteps[num_holdout:],
            timesteps[:num_holdout],
        )
        holdout_timesteps = torch.from_numpy(holdout_timesteps).long()
    else:
        train_timesteps, holdout_timesteps = None, None
    return (
        train_inputs,
        train_labels,
        holdout_input_actions,
        holdout_input_states,
        holdout_label_states,
        holdout_label_rewards,
        rank,
        train_timesteps,
        holdout_timesteps,
    )


def make_holdout_dataset(inputs, labels, network_size, num_actions=2):
    permutation = np.random.permutation(inputs.shape[0])
    inputs, labels = inputs[permutation], labels[permutation]

    holdout_inputs, holdout_labels = inputs, labels
    holdout_inputs = torch.from_numpy(holdout_inputs).float()
    holdout_labels = torch.from_numpy(holdout_labels).float()
    # (batch, states+actions)
    holdout_inputs = holdout_inputs[None, :, :].repeat([network_size, 1, 1])
    holdout_input_states, holdout_input_actions = (
        holdout_inputs[..., :-num_actions],
        holdout_inputs[..., -num_actions:],
    )
    rank = int(np.sqrt(holdout_input_states.shape[-1] // 2))
    holdout_input_states = holdout_input_states.reshape(
        holdout_inputs.shape[0], -1, 2 * rank, rank
    )

    # (batch, rewards+deltas)
    holdout_labels = holdout_labels[None, :, :].repeat([network_size, 1, 1])
    holdout_label_rewards, holdout_label_states = (
        holdout_labels[..., 0],
        holdout_labels[..., 1:],
    )
    holdout_label_states = holdout_label_states.reshape(
        holdout_labels.shape[0], -1, 2 * rank, rank
    )

    return (
        holdout_input_actions,
        holdout_input_states,
        holdout_label_states,
        holdout_label_rewards,
        rank,
    )


class StandardScaler:
    def __init__(self):
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu


class LearnableHamEnsembleDynamicsModel:
    def __init__(
        self,
        network_size,
        elite_size,
        debug_transmon=None,
        qubits=None,
        final_time=None,
        num_timesteps=None,
        debug_model=False,
        imperfection_delta=None,
        target=None,
        tname=None,
        ftol=0.1,
        diff_thru_model=None,
        use_ruthless_delta: bool = True,
        lind=None,
        learn_diss=None,
        decay_params=None,
        learn_diss_coeffs_only=None,
        learn_H_cont=None,
        override_ham_init=None,
        learn_time_dep_ham=None,
        trl=None,
        System=None,
    ):
        self.diff_thru_model = diff_thru_model
        self.verbose = True
        self.network_size = network_size
        self.elite_size = elite_size
        self.debug_mode = debug_model
        self.debug_transmon = debug_transmon
        self.learn_diss = learn_diss
        self.lind = lind
        self.learn_H_cont = learn_H_cont
        self.learn_time_dep_ham = learn_time_dep_ham
        self.HModel = LearnableHamiltonian(
            network_size,
            ansatz_A=not learn_H_cont,
            debug_transmon=debug_transmon,
            final_time=final_time,
            num_timesteps=num_timesteps,
            debug=debug_model,
            target=target,
            imperfection_delta=imperfection_delta,
            tname=tname,
            ftol=ftol,
            use_ruthless_delta=use_ruthless_delta,
            learn_diss=learn_diss,
            decay_params=decay_params,
            learn_diss_coeffs_only=learn_diss_coeffs_only,
            override_ham_init=override_ham_init,
            learn_time_dep_ham=learn_time_dep_ham,
            qubits=qubits,
            trl=trl,
            System=System,
        )

        self.optimizer = Adam(self.HModel.parameters(), lr=5e-3)
        self.elite_model_idxes = []

    def step(self, states, actions, debug_model=False, timesteps=None):
        "probabilistic or deterministic? latter is simpler so start with that"
        "requres a call to `predict` and returns something the policy can interact with i.e. next states and rewards"

        rank = int(np.sqrt(states.shape[-1] // 2))
        states = states.reshape(-1, 2 * rank, rank)
        states = torch.as_tensor(states, dtype=torch.float32)
        timesteps = None
        if self.learn_time_dep_ham:
            timesteps = torch.from_numpy(timesteps).long()
        if not self.diff_thru_model:
            actions = torch.as_tensor(actions, dtype=torch.float32)
        next_states, rewards = self.HModel.predict_prop(
            actions, states, lind=self.lind, timesteps=timesteps
        )
        if not self.diff_thru_model:
            next_states, rewards = (
                next_states.detach().numpy(),
                rewards.detach().numpy(),
            )
            next_states = next_states.reshape(
                next_states.shape[0], next_states.shape[1], -1
            )

        else:
            next_states = next_states.reshape(
                next_states.shape[0], next_states.shape[1], -1
            )  # , rewards, None
        num_models, batch_size, _ = next_states.shape

        if debug_model:
            # always pick the first model as they are all identically perfect or the model is imperfect and we wish to study its behavior w.r.t. imperfections
            model_idxes = np.zeros(batch_size, dtype=np.int64)
        else:
            model_idxes = np.random.choice(self.elite_model_idxes, size=batch_size)
        batch_idxes = np.arange(batch_size)

        next_states = next_states[model_idxes, batch_idxes]
        rewards = rewards[model_idxes, batch_idxes]
        # this is where you would make the next states probabilistic
        return next_states, rewards, None

    def mse_loss(self, pred_rewards, rewards):
        try:
            if rewards.shape != pred_rewards.shape:
                rewards = rewards.reshape(*pred_rewards.shape)
            mse_loss_ensemble = (pred_rewards - rewards) ** 2
            if len(mse_loss_ensemble.shape) > 2:
                mse_loss_ensemble = mse_loss_ensemble.view(self.network_size, -1).sum(
                    axis=-1
                )
            else:
                mse_loss_ensemble = mse_loss_ensemble.sum(axis=-1)
            return mse_loss_ensemble.sum(), mse_loss_ensemble
        except:
            raise AssertionError(
                "rewards and pred_rewards are not the same shape inside `mse_loss`"
            )

    def call_backward(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        batch_size: Optional[int] = 256,
        holdout_ratio: Optional[float] = 0.2,
        max_epochs_since_update: Optional[int] = 5,
        epochs: Optional[int] = 20,
        data_scaling_exp: Optional[bool] = False,
        custom_holdouts: Optional[torch.Tensor] = None,
        use_full_dataset: Optional[bool] = False,
        timesteps: Optional[torch.IntTensor] = None,
        improv_threshold: Optional[float] = 0.01,
    ) -> Tuple[List, List, List]:
        """
        _summary_

        Parameters
        ----------
        inputs : np.ndarray
            _description_
        labels : np.ndarray
            _description_
        batch_size : Optional[int], optional
            _description_, by default 256
        holdout_ratio : Optional[float], optional
            _description_, by default 0.2
        max_epochs_since_update : Optional[int], optional
            _description_, by default 5
        epochs : Optional[int], optional
            _description_, by default 20
        data_scaling_exp : Optional[bool], optional
            _description_, by default False
        custom_holdouts : Optional[torch.Tensor], optional
            _description_, by default None
        use_full_dataset : Optional[bool], optional
            _description_, by default False
        timesteps : Optional[torch.IntTensor], optional
            _description_, by default None
        improv_threshold : Optional[float], optional
            _description_, by default 0.01

        Returns
        -------
        Tuple[List, List, List]
            _description_
        """
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}
        num_actions = len(self.debug_transmon.reduced_cont_basis)

        (
            train_inputs,
            train_labels,
            holdout_input_actions,
            holdout_input_states,
            holdout_label_states,
            holdout_label_rewards,
            rank,
            train_timesteps,
            holdout_timesteps,
        ) = get_model_learning_data(
            inputs,
            labels,
            holdout_ratio,
            self.network_size,
            timesteps,
            learn_time_dep_ham=self.learn_time_dep_ham,
            num_actions=num_actions,
        )

        if type(custom_holdouts) != type(None):
            (
                holdout_input_actions,
                holdout_input_states,
                holdout_label_states,
                holdout_label_rewards,
                rank,
            ) = custom_holdouts
            train_size = 1600
            if use_full_dataset:
                train_size = train_inputs.shape[0]
        else:
            train_size = train_inputs.shape[0]

        def get_coarse_supers(train_input_states, shots=10000000):
            tls_choi_real, tls_choi_imag = self.HModel.convert_to_choi(
                train_input_states
            )
            diag_sums = torch.einsum("...ii", tls_choi_real)
            tls_choi_real = torch.einsum(
                "nabc, na -> nabc", tls_choi_real, 1 / diag_sums
            )
            tls_choi_imag = torch.einsum(
                "nabc, na -> nabc", tls_choi_imag, 1 / diag_sums
            )
            bvecs = self.HModel.convert_to_bloch(
                A=(tls_choi_real, tls_choi_imag), dont_convert_to_choi=True
            )
            probs = 0.5 * (bvecs + 1)
            # sample 1 giant multinomial vector for these tests...
            nprobs = probs.ravel()
            probs_norm = nprobs.sum(axis=-1).detach().numpy()
            coarse = (
                np.random.multinomial(
                    shots * probs.shape[0] * probs.shape[1], nprobs / probs_norm
                )
                / (shots * probs.shape[0] * probs.shape[1])
                * probs_norm
            )
            cprobs = torch.as_tensor(coarse.reshape(probs.shape), dtype=torch.float32)
            cbvecs = 2 * cprobs - 1
            new_chois_real, new_chois_imag = self.HModel.bloch_to_choi(cbvecs)
            new_chois_real = torch.einsum("nabc, na -> nabc", new_chois_real, diag_sums)
            new_chois_imag = torch.einsum("nabc, na -> nabc", new_chois_imag, diag_sums)
            new_supers = self.HModel.convert_to_choi(
                (new_chois_real, new_chois_imag), True
            )
            train_input_states = torch.cat(new_supers, dim=-2).detach()
            return train_input_states

        train_losses, holdout_losses, other_losses, train_timestep = [], [], [], None
        for epoch in range(epochs):
            if type(custom_holdouts) == type(None) or train_inputs.shape[0] >= 1600:
                train_idx = np.vstack(
                    [
                        np.random.permutation(train_inputs.shape[0])
                        for _ in range(self.network_size)
                    ]
                )

            else:
                train_idx = np.vstack(
                    [
                        np.random.randint(low=0, high=train_inputs.shape[0], size=1600)
                        for _ in range(self.network_size)
                    ]
                )

            for start_pos in range(0, train_size, batch_size):
                idx = train_idx[:, start_pos : start_pos + batch_size]
                train_input = torch.from_numpy((train_inputs[idx])).float()
                train_input_states, train_input_actions = (
                    train_input[..., :-num_actions],
                    train_input[..., -num_actions:],
                )
                train_label = torch.from_numpy(train_labels[idx]).float()
                train_label_rewards, train_label_states = (
                    train_label[..., 0],
                    train_label[..., 1:],
                )
                train_input_states = train_input_states.reshape(
                    train_input.shape[0], -1, 2 * rank, rank
                )
                train_label_states = train_label_states.reshape(
                    train_label.shape[0], -1, 2 * rank, rank
                )
                if self.learn_time_dep_ham:
                    train_timestep = torch.from_numpy(train_timesteps[idx]).long()

                next_states, rewards = self.HModel.predict_prop(
                    train_input_actions,
                    train_input_states,
                    lind=self.lind,
                    heun_eps=5e-4,
                    learn_diss=self.learn_diss,
                    timesteps=train_timestep,
                )

                loss = ((next_states - train_label_states) ** 2).sum()

                if not self.debug_mode:
                    self.call_backward(loss)
                # print(loss)
            train_losses.append(
                np.array(loss.detach().numpy(), dtype=np.float64).tolist()
            )

            with torch.no_grad():
                next_states, rewards = self.HModel.predict_prop(
                    holdout_input_actions,
                    holdout_input_states,
                    lind=self.lind,
                    heun_eps=5e-4,
                    learn_diss=self.learn_diss,
                    timesteps=holdout_timesteps,
                )
                _, holdout_mse_losses = self.mse_loss(next_states, holdout_label_states)
                # holdout_mse_losses = ((next_states-holdout_label_states)**2).sum(axis=-1)
                holdout_mse_losses = np.array(
                    holdout_mse_losses.detach().cpu().numpy(), dtype=np.float64
                )
                new_break_loss = np.max(holdout_mse_losses)
                holdout_losses.append(new_break_loss)
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[: self.elite_size].tolist()
                # INSIGHT: nice idea to break training on current dataset
                #          if holdout losses are increasing!
                if not data_scaling_exp:
                    break_train = self._save_best(
                        epoch, holdout_mse_losses, improv_threshold
                    )
                    if break_train or new_break_loss < 1e-4:
                        return train_losses, holdout_losses, other_losses
            other_loss = {}
            diff = self.HModel.ham_params.detach().numpy() - dm2vec(
                self.debug_transmon.transmon_sys_ham
            )
            diff *= diff
            mean_diff = np.real(
                np.mean(diff, axis=-1)
            )  # completely real, just a code-induced sanitization
            other_loss["ham"] = np.array(np.real(mean_diff), dtype=np.float64).tolist()
            if self.verbose:
                print(
                    "epoch: {}, holdout mse losses: {}".format(
                        epoch, holdout_mse_losses
                    )
                )
                print("ham params error from truth", mean_diff)
                if self.learn_diss:
                    true_d = np.concatenate(
                        (np.real(self.HModel.true_ops), np.imag(self.HModel.true_ops))
                    )
                    a, b = self.HModel.diss_params_r, self.HModel.diss_params_i
                    est_d = a.detach().numpy(), b.detach().numpy()
                    est_d = np.concatenate(est_d)
                    diss_loss = ((true_d - est_d) ** 2).mean()
                    other_loss["diss"] = np.array(
                        np.real(diss_loss), dtype=np.float64
                    ).tolist()
                    print("lind error from true", diss_loss)
                self.learn_H_cont = True
                if self.learn_H_cont:

                    learnt_contHs = self.HModel.control_ham_ansatz.detach().numpy()
                    true_contHs = dm2vec(self.debug_transmon.reduced_cont_basis)
                    learnt_h_cont_loss = ((learnt_contHs - true_contHs) ** 2).mean()
                    other_loss["Hcont"] = np.array(
                        np.real(learnt_h_cont_loss), dtype=np.float64
                    ).tolist()
                    print(learnt_contHs)
                    print("contH error from true", learnt_h_cont_loss)
            other_losses.append(other_loss)
        return train_losses, holdout_losses, other_losses

    def _save_best(self, epoch, holdout_losses, improv_threshold=0.01):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > improv_threshold:
                self._snapshots[i] = (epoch, current)
                updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self):
        "maybe related to predict prop?"
        raise NotImplementedError
