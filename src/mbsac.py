from src.ppo import PPO
from src.basic_transmon import Transmon
from src.sac.sac import SAC
import numpy as np
from src.non_para_dynamics import (
    EnsembleDynamicsModel,
    LearnableHamEnsembleDynamicsModel,
)
from tqdm import tqdm
from src.baseclass import CNOT, TOFFOLI, hadamard, rotation
import torch
import pickle
import os
from src.env_sampler import ReplayMemory, EnvSampler
from src.str_to_qsys import str_to_qsys
from typing import Optional, List, Tuple, Any

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class act_dims:
    def __init__(self, low, high):
        self.low = low
        self.high = high


def illegal_boolean_combinations(use_model, debug_model, use_ansatz_model):
    if use_model and debug_model and not use_ansatz_model:
        raise ValueError(
            f"Illegal combination detected! use_model:{use_model}, debug_model:{debug_model}, use_ansatz_model{use_ansatz_model}"
        )


class MBSAC(PPO):
    def __init__(
        self,
        rollout_schedule: Optional[List] = [20, 150, 1, 1],
        use_decay: Optional[bool] = False,
        rollout_batch_size: Optional[int] = 9000,
        model_retain_epochs: Optional[int] = 1,
        model_train_freq: Optional[int] = 250,
        hidden: int = 256,
        replay_size: int = 1000000,
        max_path_length: Optional[int] = 1000,
        network_size: Optional[int] = 7,
        elite_size: Optional[int] = 5,
        automatic_entropy_tuning: Optional[bool] = True,
        rollout_max_length: Optional[int] = 15,
        rollout_max_epoch: Optional[int] = 50,
        real_ratio: Optional[float] = 0.05,
        lr: float = 0.0003,
        use_ansatz_model: float = True,
        use_model: bool = True,
        debug_model: bool = False,
        const_rollout_length: Optional[int] = 5,
        model_train_iterations: Optional[int] = 10,
        imperfection_delta: float = 0.1,
        imperfection_ftol: Optional[float] = 0.1,
        diff_thru_model: Optional[bool] = False,
        reset_rate_epochs: Optional[int] = 5,
        counter_primacy_bias: Optional[bool] = False,
        use_ruthless_delta: Optional[bool] = True,
        use_shots_to_recon_state: Optional[bool] = False,
        M: int = int(pow(10, 6)),
        static_lind: bool = False,
        learn_diss: bool = False,
        learn_diss_coeffs_only: bool = False,
        learn_H_cont: Optional[bool] = False,
        buffer_size: Optional[int] = 2000,
        override_ham_init: Optional[bool] = False,
        learn_time_dep_ham: Optional[bool] = False,
        trl: Optional[int] = 2,
        System: Optional[str] = "transmon",
        qubits: Optional[int] = 2,
        tname: Optional[str] = "cnot",
        bmin: Optional[float] = -10,
        bmax: Optional[float] = 10,
        epoch_checkpoint_rate: Optional[int] = 1,
        learn_ham_improv_threshold: Optional[int] = 0.01,
        allow_respawn: Optional[bool] = False,
        *args: Optional[Any],
        **kwargs: Optional[Any],
    ) -> None:
        """
        The model based soft actor critic algorithm consists of two flavors:
        1. Learnable-Hamiltonian Model-Based Soft Actor Critic (LH-MBSAC).
           This is the focus of the paper.
        2. An ensemble neural network based SAC that was used in https://arxiv.org/abs/1906.08253
           and adapted from the original repo https://github.com/Xingyu-Lin/mbpo_pytorch/

        Based on our experiments with both, we found 1. to be much more sample efficient
        and effective for the gate control problems presented in the paper. So that will
        be the focus our repo too. However, the ensemble NN functionality is still provided
        in case it might be relevant to someone.

        Coming back to the LH-MBSAC, we equip the model-fee off-policy algorithm
        with a Learnable ODE to exploit a strong and correct inductive bias for
        the dynamics of the system in order to sample-efficiently learn an effective
        model of the controllable system.

        Some features of this module are inherited from `PPO` and both are children
        of `BaseOpt` that stores common cost and evolution functions.
        This class is the center of all the code in this repo and the main contribution
        of the paper. So all the pieces fit together here.
        There are quite a few parameters but most of them of the SAC specific ones can
        be ignored as their values were obtained through hyperparameter tuning.


        Parameters
        ----------
        rollout_schedule : Optional[List], optional
            Number of model k-branch rollouts or consecutive MDP transitions
            made by the agent with the model, by default [20, 150, 1, 1]
        use_decay : Optional[bool], optional
            Use a decaying loss function in the NN-based model, by default False
        rollout_batch_size : Optional[int], optional
            Size of the training data or MDP instances to perform a k-branch rollout
            on, by default 9000
        model_retain_epochs : Optional[int], optional
            For how many consecutive epochs should the model be used, by default 1
        model_train_freq : Optional[int], optional
            How many iterations should elapse before the model is trained, by default 250
        hidden : int, optional
            Linear layer hidden dimension size, by default 256
        replay_size : int, optional
            Size of the replay buffer, by default 1000000
        max_path_length : Optional[int], optional
            Maximum length of the MDP transition trajectory.
            Deprecated or not useful in the LH-MBSAC context, by default 1000
        network_size : Optional[int], optional
            Ensemble size if multiple models are evolved at once, by default 7
            Deprecated if LH-MBSAC is used. Then the number of models is by default 1.
        elite_size : Optional[int], optional
            Part of PETS, subset size of models to use to make
            model predictions, by default 5
        automatic_entropy_tuning : Optional[bool], optional
            use autodiff to train the temperature parameter in SAC, by default True
        rollout_max_length : Optional[int], optional
            length k of the k-branch rollout performed by the model, by default 15
        rollout_max_epoch : Optional[int], optional
            maximum number of epochs to train the model, by default 50
        real_ratio : Optional[float], optional
            ratio of real to synthetic data to maintain by the MBRL algo, by default 0.05
        lr : float, optional
            learning rate for the model, by default 0.0003
        use_ansatz_model : float, optional
            flag for using LH-MBSAC instead of the NN-based MBPO with PETS, by default True
        use_model : bool, optional
            flag to switch between model based SAC and model-free SAC, by default True
        debug_model : bool, optional
            use the environment model as the LH-MBSAC model for debugging, by default False
        const_rollout_length : Optional[int], optional
            rollout length that isn't affected by the rollout schedule, by default 5
        model_train_iterations : Optional[int], optional
            number of gradient descent steps to train the model, by default 10
        imperfection_delta : float, optional
            Distance of the guess Hamiltonian to the truth in terms of
            mean squared error of the Pauli basis coefficients, by default 0.1
        imperfection_ftol : Optional[float], optional
            Deprecated/uninteresting. equivalent of `imperfection_delta` but
            in terms of average fidelity deficit between the true annd guess
            Hamiltonian, by default 0.1
        diff_thru_model : Optional[bool], optional
            flag to try autodiffing all timesteps instead of just 1.
            Experimental, by default False
        reset_rate_epochs : Optional[int], optional
            number of epochs to go before re-initializing the policy and value functions
            to address cases when the model gets stuck, by default 5
        counter_primacy_bias : Optional[bool], optional
            flag to turn on resetting the policy and value functions, by default False
        use_ruthless_delta : Optional[bool], optional
            perturb all the coefficients of the ansatz Hamiltonian in the model
            when initializing with some `imperfection_delta` distance
            from truth, by default True
        use_shots_to_recon_state : Optional[bool], optional
            add shot noise to the state observations, by default False
        M : int, optional
            number of shots per observable used to estimate the Choi state
            representation of the gate that the policy sees, by default int(pow(10, 6))
        static_lind : bool, optional
            add dissipatory dynamics to the ODE simulation, by default False
        learn_diss : bool, optional
            learn the disspatory dynamics as part of the model, by default False
        learn_diss_coeffs_only : bool, optional
            Flag to learn only the scalar coefficients of the dissipation operators.
            The alternative is to learn the entire matrix with equally many coefficients
            for each entry in the dissipation operator, by default False
        learn_H_cont : Optional[bool], optional
            flag to start learning the time-dependent control Hamiltonian
            in the system, by default False
        buffer_size : Optional[int], optional
            Environment buffer size or number of samples randomly observed
            during the initial exploration stage. See Algorithm 2 in the
            paper, by default 2000
        override_ham_init : Optional[bool], optional
            Whether to override system Hamiltonian initialization with saved random/or
            other values, by default False
        learn_time_dep_ham : Optional[bool], optional
            Start learning the time dependent Hamiltonian using a ZOH like method.
            Under construction, by default False
        trl : Optional[int], optional
            Hilbert space truncation level for the qu(`trl`)it, by default 2
        System : Optional[str], optional
            System architecture or Hamiltonian to try, by default "transmon"
        qubits : Optional[int], optional
            Number of qubits in the system, by default 2
        tname : Optional[str], optional
            Name of thet target gate. Needs to be specified uniquely as it is used in checkpoint files
            and data logs and one could risk overriding old saved files, by default "cnot"
        bmin : Optional[float], optional
            min. control pulse amplitude, by default -10
        bmax : Optional[float], optional
            max. control pulse amplitude, by default 10
        epoch_checkpoint_rate : Optional[int], optional
            save frequency of various logging-worthy quantities including e.g. 100 best control pulses found so far,
             model weights or learnt parameters etc., by default 1
        learn_ham_improv_threshold : Optional[int], optional
            When to stop training the model during the training phase,
            This step is performed after each gradient descent iteration to avoid getting stuck
            in sub-optimal training updates. The holdout loss is used for this comparison. by default 0.01
        allow_respawn : Optional[bool], optional
            Keep updating a checkpoint of the state of the program so that training can just pick right
            off the last checkpoint instead of starting again from scratch, by default False
        """

        illegal_boolean_combinations(use_model, debug_model, use_ansatz_model)

        super().__init__(
            *args,
            use_shots_to_recon_state=use_shots_to_recon_state,
            M=M,
            trl=trl,
            qsys=str_to_qsys(System)[0],
            qubits=qubits,
            tname=tname,
            bmin=bmin,
            bmax=bmax,
            epoch_checkpoint_rate=epoch_checkpoint_rate,
            buffer_size=buffer_size,
            **kwargs,
        )
        self.static_lind = static_lind
        self.learn_diss = learn_diss
        self.learn_time_dep_ham = learn_time_dep_ham
        self.diff_thru_model = diff_thru_model
        self.counter_primacy_bias = counter_primacy_bias
        self.primacy_bias_reset_rate = reset_rate_epochs
        self.model_train_iterations = model_train_iterations
        self.const_rollout_length = const_rollout_length
        self.use_model = use_model
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.rollout_schedule = rollout_schedule
        self.use_decay = use_decay
        self.use_ansatz_model = use_ansatz_model
        self.debug_model = debug_model
        self.network_size = network_size
        self.elite_size = elite_size
        self.replay_size = replay_size
        self.min_pool_size = buffer_size
        self.rollout_max_length = rollout_max_length
        self.rollout_max_epoch = rollout_max_epoch
        self.max_path_length = max_path_length
        self.real_ratio = real_ratio
        self.improv_threshold = learn_ham_improv_threshold
        self.allow_respawn = allow_respawn
        self.env_model = None
        # torch.manual_seed(3) # seeds 3 (9k 0.97f), 4 (10k, 0.95f)

        if self.use_model and not self.use_ansatz_model:
            self.env_model = EnsembleDynamicsModel(
                network_size,
                elite_size,
                self.obs_dim * self.obs_dim * 2,
                self.qubits,
                use_decay=use_decay,
            )
        elif self.use_model and self.use_ansatz_model:
            if self.debug_model:
                network_size = 1
                elite_size = 1
            if imperfection_delta != 0:
                use_ruthless_delta = True
                network_size = 1
                elite_size = 1
            network_size = 1
            elite_size = 1
            self.env_model = LearnableHamEnsembleDynamicsModel(
                network_size,
                elite_size,
                debug_transmon=self.qsys,
                qubits=qubits,
                final_time=self.final_time,
                num_timesteps=self.num_timesteps,
                debug_model=debug_model,
                imperfection_delta=imperfection_delta,
                target=self.target,
                tname=self.tname,
                ftol=imperfection_ftol,
                diff_thru_model=diff_thru_model,
                use_ruthless_delta=use_ruthless_delta,
                lind=self.use_shots_to_recon_state,
                learn_diss=self.learn_diss,
                decay_params=(self.qsys.decay_1[0], self.qsys.decay_2[0]),
                learn_diss_coeffs_only=learn_diss_coeffs_only,
                learn_H_cont=learn_H_cont,
                override_ham_init=override_ham_init,
                learn_time_dep_ham=learn_time_dep_ham,
                trl=trl,
                System=System,
            )

        self.buffer_size = buffer_size

        self.rollout_batch_size = rollout_batch_size
        self.model_retain_epochs = model_retain_epochs
        self.model_train_freq = (
            model_train_freq if self.buffer_size > 1000 else self.buffer_size // 4
        )
        self.hidden = [hidden]

        ## new sac edits starts here
        self.args = dict(
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            policy="Gaussian",
            target_update_interval=1,
            automatic_entropy_tuning=automatic_entropy_tuning,
            hidden_size=hidden,
            lr=lr,
            act_dims=act_dims(-1 * self.Bmax, self.Bmax),
            min_pool_size=self.min_pool_size,
            rollout_min_epoch=20,
            real_ratio=real_ratio,
            rollout_min_length=1,
            rollout_max_length=rollout_max_length,
            rollout_batch_size=rollout_batch_size,
            train_every_n_steps=1,
            max_train_repeat_per_step=5,
            num_train_repeat=20,
            policy_train_batch_size=256,
            init_exploration_steps=self.buffer_size,
            rollout_max_epoch=rollout_max_epoch,
        )

        self.agent = SAC(
            self.obs_dim * self.obs_dim * 2, self.num_controls_per_timestep, self.args
        )
        self.env_pool = ReplayMemory(replay_size)
        self.rollouts_per_epoch = (
            self.rollout_batch_size * self.buffer_size / self.model_train_freq
        )
        model_steps_per_epoch = int(1 * self.rollouts_per_epoch)
        self.new_pool_size = self.model_retain_epochs * model_steps_per_epoch
        self.model_pool = ReplayMemory(self.new_pool_size)
        self.env_sampler = EnvSampler(self, max_path_length=max_path_length)
        self.verbose = False
        self.ep_ret = 0
        self.ep_len = 0
        self.funcalls = 0
        self.running_controllers = {}
        self.max_fid_seen = 0
        self.max_fid_est = 0
        self.max_ep_ret = 0

    def resize_model_pool(self, rollout_length, model_pool):
        model_steps_per_epoch = int(rollout_length * self.rollouts_per_epoch)
        new_pool_size = self.model_retain_epochs * model_steps_per_epoch
        sample_all = model_pool.return_all()
        new_model_pool = ReplayMemory(new_pool_size)
        new_model_pool.push_batch(sample_all)
        return new_model_pool

    def step(self, action):
        "env step function for the Env sampler -> returns next obs, reward, terminal, None for info"
        if self.use_shots_to_recon_state:
            next_obs, reward, next_obs_est, reward_est = self.take_step(action)
            terminal = False  # Using the CIRCULAR MDP assumption
            return next_obs, reward, next_obs_est, reward_est, terminal, None
        else:
            next_obs, reward = self.take_step(action)
            terminal = False  # Using the CIRCULAR MDP assumption
            return next_obs, reward, terminal, None

    def exploration_before_start(self):
        "cache the init exploration stage"
        if not self.use_shots_to_recon_state:
            if self.buffer_size == 2000:
                ep_fname = "explore_pools/init_expl_mbsac_env_pool_lind_False"
            else:
                ep_fname = f"explore_pools/init_expl_mbsac_env_pool_lind_False_bs_{self.buffer_size}"
        else:
            logshots = int(np.ceil(np.log(self.M) / np.log(10)))
            ep_fname = f"explore_pools/init_expl_mbsac_env_pool_lind_{self.use_shots_to_recon_state}_shots_10e{logshots}"
            if self.static_lind:
                ep_fname += "_static_lind"
            if self.buffer_size != 2000:
                ep_fname += f"_bs_{self.buffer_size}"
        if not type(self.qsys) == type(Transmon()):
            ep_fname += "_" + self.qsys.name()
        ep_fname += ".pkl"
        if not os.path.exists(ep_fname):
            for t in range(self.args["init_exploration_steps"]):
                if not self.use_shots_to_recon_state:
                    (
                        cur_state,
                        action,
                        next_state,
                        reward,
                        done,
                        _,
                    ) = self.env_sampler.sample(
                        self.agent, timestep=t, num_timesteps=self.num_timesteps
                    )
                    self.env_pool.push_timestep(t % self.num_timesteps)
                    self.env_pool.push(cur_state, action, reward, next_state, done)
                    self.log_stuff(reward)
                else:
                    (
                        cur_state,
                        cur_state_est,
                        action,
                        next_state,
                        reward,
                        next_state_est,
                        reward_est,
                        done,
                        _,
                    ) = self.env_sampler.sample(
                        self.agent, timestep=t, num_timesteps=self.num_timesteps
                    )
                    self.env_pool.push_timestep(t % self.num_timesteps)
                    self.env_pool.push(
                        cur_state_est, action, reward_est, next_state_est, done
                    )
                    self.log_stuff(reward_est, reward)

            if self.use_shots_to_recon_state:
                pickle.dump(
                    (
                        self.env_pool,
                        self.running_controllers,
                        self.max_fid_seen,
                        self.max_fid_est,
                        self.max_ep_ret,
                        self.funcalls,
                    ),
                    open(ep_fname, "wb"),
                )
            else:
                pickle.dump(
                    (
                        self.env_pool,
                        self.running_controllers,
                        self.max_fid_seen,
                        self.max_ep_ret,
                        self.funcalls,
                    ),
                    open(ep_fname, "wb"),
                )
        else:
            if self.use_shots_to_recon_state:
                (
                    self.env_pool,
                    self.running_controllers,
                    self.max_fid_seen,
                    self.max_fid_est,
                    self.max_ep_ret,
                    self.funcalls,
                ) = pickle.load(open(ep_fname, "rb"))
            else:
                (
                    self.env_pool,
                    self.running_controllers,
                    self.max_fid_seen,
                    self.max_ep_ret,
                    self.funcalls,
                ) = pickle.load(open(ep_fname, "rb"))

    def log_stuff(self, reward, reward_true=None):
        self.ep_ret += reward
        self.ep_len += 1
        self.funcalls += 1
        if self.use_shots_to_recon_state:
            self.max_fid_seen = max(self.max_fid_seen, reward_true)
            self.max_fid_est = max(self.max_fid_est, reward)
            self.max_ep_ret = max(self.ep_ret, self.max_ep_ret)

            if self.verbose:
                print(
                    f"max_fid_obtained(t/e): {round(self.max_fid_seen,4), round(self.max_fid_est,4)},"
                    + f" fid(t/e): {round(reward_true,4),round(reward,4)}"
                )
                print(f"func calls {self.funcalls}", f"max ep_ret: {self.max_ep_ret}")
        else:
            self.max_fid_seen = max(self.max_fid_seen, reward)
            self.max_ep_ret = max(self.ep_ret, self.max_ep_ret)

            if self.verbose:
                print(f"max_fid_obtained: {self.max_fid_seen}, fid: {reward}")
                print(f"func calls {self.funcalls}", f"max ep_ret: {self.max_ep_ret}")

        # if max_fid_seen > self.fid_threshold:
        #     return
        if self.save_topc:
            l = len(self.running_controllers.keys())
            if l < self.save_topc:
                self.running_controllers[reward] = self.control_sequence_for_epoch[
                    : self.timestep
                ].tolist()
                # print("running_list: \n", self.running_controllers)
            else:
                # itopop=self.find_min_fid_index(self.running_controllers) # time to pop this ###
                itopop = min(list(self.running_controllers.keys()))
                self.running_controllers.pop(itopop)
                self.running_controllers[reward] = self.control_sequence_for_epoch[
                    : self.timestep
                ].tolist()  # maintain const size list
                if self.testing:
                    f = list(self.running_controllers.keys())[0]
                    c = np.array(self.running_controllers[f])
                    assert np.allclose(float(f), 1 - self.infidelity_Lind(c)), print(
                        float(f), self.infidelity_Lind(c)
                    )
                    if self.verbose:
                        print("running_list: \n", self.running_controllers.keys())

    def set_rollout_length(self, epoch_step):
        rollout_length = min(
            max(
                self.args["rollout_min_length"]
                + (epoch_step - self.args["rollout_min_epoch"])
                / (self.args["rollout_max_length"] - self.args["rollout_min_epoch"])
                * (self.args["rollout_max_epoch"] - self.args["rollout_min_length"]),
                self.args["rollout_min_length"],
            ),
            self.args["rollout_max_epoch"],
        )
        return int(rollout_length)

    def get_sup_learning_ingredients(self):
        state, action, reward, next_state, done, timesteps = self.env_pool.sample(
            len(self.env_pool), get_timesteps=self.learn_time_dep_ham
        )
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        if self.use_ansatz_model:
            labels = np.concatenate(
                (np.reshape(reward, (reward.shape[0], -1)), next_state), axis=-1
            )
        else:
            labels = np.concatenate(
                (np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1
            )
        return inputs, labels, timesteps

    def train_predict_model(self):
        # Get all samples from environment
        inputs, labels, timesteps = self.get_sup_learning_ingredients()
        train_losses, holdout_losses, other_losses = self.env_model.train(
            inputs,
            labels,
            batch_size=128,
            holdout_ratio=0.2,
            timesteps=timesteps,
            improv_threshold=self.improv_threshold,
        )
        # record the ham parameters for further analysis
        if self.use_ansatz_model:
            self.record["ham_params"] = (
                self.env_model.HModel.ham_params.detach().numpy().tolist()
            )
        elif self.use_model:
            d = self.env_model.ensemble_model.state_dict()
            for key in d:
                d[key] = d[key].numpy().tolist()
            self.record["nn_params"] = d
        self.record["train_losses"] = train_losses
        self.record["holdout_losses"] = holdout_losses
        self.record["other_losses"] = other_losses

    def rollout_model(self, rollout_length, total_step=None, train_policy_steps=None):
        """_summary_

        Parameters
        ----------
        rollout_length : _type_
            _description_
        total_step : _type_, optional
            _description_, by default None
        train_policy_steps : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        (
            state,
            action,
            reward,
            next_state,
            done,
            timesteps,
        ) = self.env_pool.sample_all_batch(
            self.args["rollout_batch_size"], get_timesteps=True
        )

        for i in range(rollout_length):
            if self.diff_thru_model:
                for _ in range(10):
                    idx = np.random.randint(low=0, high=len(state), size=1000)
                    states = state[idx]
                    for _ in range(20):
                        action = self.agent.select_action(
                            states, diff_thru_model=self.diff_thru_model
                        )

                        _, rewards, _ = self.env_model.step(
                            states, action, debug_model=self.debug_model
                        )
                        self.agent.policy_optim.zero_grad()
                        # self.env_model.optimizer.zero_grad()
                        loss = (1 - rewards).sum()
                        loss.backward()
                        self.agent.policy_optim.step()
                        # self.env_model.optimizer.step()
                        print(loss)
            action = self.agent.select_action(
                state, diff_thru_model=self.diff_thru_model
            )

            next_states, rewards, _ = self.env_model.step(
                state, action, debug_model=self.debug_model, timesteps=timesteps
            )
            if self.diff_thru_model:
                next_states, rewards = (
                    next_states.detach().numpy(),
                    rewards.detach().numpy(),
                )
                action = action.detach().numpy()
            # if np.isnan(next_states).sum() != 0 or np.isnan(rewards).sum() != 0:
            #     print("nan found")
            #     break
            if self.verbose:
                print("model rollout iteration", i)
            # dummy for now
            terminals = np.zeros_like(rewards, dtype=bool)
            # drop out terminal states from the rollout batch (TODO::checking if the NN model needs this too)
            if self.use_ansatz_model:
                etmask = ~(
                    (timesteps + 1) >= self.num_timesteps
                )  # ended trajectory mask for filtering
                state, action, rewards, next_states, terminals, timesteps = (
                    state[etmask],
                    action[etmask],
                    rewards[etmask],
                    next_states[etmask],
                    terminals[etmask],
                    timesteps[etmask],
                )

            self.model_pool.push_batch(
                [
                    (state[j], action[j], rewards[j], next_states[j], terminals[j])
                    for j in range(state.shape[0])
                ]
            )
            state = next_states
            timesteps += 1
            if self.debug_model:
                for _ in range(self.model_train_iterations):
                    x = self.train_policy(total_step, train_policy_steps)
                train_policy_steps = x  #

        return train_policy_steps
        # except Exception as e:
        #     print(e)

    def train_policy_repeats(self, total_step, train_step):
        if total_step % self.args["train_every_n_steps"] > 0:
            return 0
        # for experimental reasons we will not worry about this
        # if train_step > self.args["max_train_repeat_per_step"] * total_step:
        #     return 0
        if self.verbose:
            print("inside policy train: tot, train", total_step, train_step)
        for i in range(self.args["num_train_repeat"]):
            env_batch_size = int(
                self.args["policy_train_batch_size"] * self.args["real_ratio"]
            )
            model_batch_size = self.args["policy_train_batch_size"] - env_batch_size

            (
                env_state,
                env_action,
                env_reward,
                env_next_state,
                env_done,
                _,
            ) = self.env_pool.sample(int(env_batch_size))

            if model_batch_size > 0 and len(self.model_pool) > 0:
                (
                    model_state,
                    model_action,
                    model_reward,
                    model_next_state,
                    model_done,
                ) = self.model_pool.sample_all_batch(int(model_batch_size))
                if len(model_reward.shape) == 1:
                    model_reward = model_reward.reshape(-1, 1)
                if len(model_done.shape) == 1:
                    model_done = model_done.reshape(-1, 1)

                (
                    batch_state,
                    batch_action,
                    batch_reward,
                    batch_next_state,
                    batch_done,
                ) = (
                    np.concatenate((env_state, model_state), axis=0),
                    np.concatenate((env_action, model_action), axis=0),
                    np.concatenate(
                        (
                            np.reshape(env_reward, (env_reward.shape[0], -1)),
                            model_reward,
                        ),
                        axis=0,
                    ),
                    np.concatenate((env_next_state, model_next_state), axis=0),
                    np.concatenate(
                        (np.reshape(env_done, (env_done.shape[0], -1)), model_done),
                        axis=0,
                    ),
                )
            else:
                (
                    batch_state,
                    batch_action,
                    batch_reward,
                    batch_next_state,
                    batch_done,
                ) = (env_state, env_action, env_reward, env_next_state, env_done)

            batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
            batch_done = (~batch_done).astype(int)
            self.agent.update_parameters(
                (batch_state, batch_action, batch_reward, batch_next_state, batch_done),
                self.args["policy_train_batch_size"],
                i,
            )

        return self.args["num_train_repeat"]

    def train_policy(self, train_policy_steps, total_step):
        if len(self.env_pool) > self.args["min_pool_size"]:
            if self.verbose:
                print("training policy")
            train_policy_steps += self.train_policy_repeats(
                total_step, train_policy_steps
            )
        return train_policy_steps

    def reset_sac(self):
        "only triggers if `counter_primacy_bias` is set to True"
        """primacy bias reset from https://proceedings.mlr.press/v162/nikishin22a.html"""
        """Does not actually work. The problem is definitely more about state visitation 
           and neural network parameter init than anticipated.
           Need to restart the replay buffer too"""
        if not self.max_fid_seen > 0.9:
            if (
                self.counter_primacy_bias
                and (self.funcalls + 1)
                % (self.primacy_bias_reset_rate * self.buffer_size)
                == 0
            ):
                # reset the entire agent
                if self.verbose:
                    print(f"resetting SAC")
                self.agent = SAC(
                    self.obs_dim * self.obs_dim * 2, self.qubits, self.args
                )

    def respawn_dump(self, epoch_step, total_step, rollout_length, train_policy_steps):
        """
        Store the current state of `MBSAC` instance that can be loaded again on the fly
        to continue training.
        """
        if self.use_shots_to_recon_state:
            pickle.dump(
                (
                    epoch_step,
                    self.env_model,
                    self.agent,
                    self.env_pool,
                    self.model_pool,
                    self.ep_ret,
                    self.ep_len,
                    total_step,
                    rollout_length,
                    self.max_fid_seen,
                    self.running_controllers,
                    rollout_length,
                    train_policy_steps,
                    self.max_fid_est,
                    self.max_ep_ret,
                    self.funcalls,
                ),
                open(f"checkpoints/{self.experiment_name}.pkl", "wb"),
            )
        else:
            pickle.dump(
                (
                    epoch_step,
                    self.env_model,
                    self.agent,
                    self.env_pool,
                    self.model_pool,
                    self.ep_ret,
                    self.ep_len,
                    total_step,
                    rollout_length,
                    self.max_fid_seen,
                    self.running_controllers,
                    rollout_length,
                    train_policy_steps,
                    self.max_ep_ret,
                    self.funcalls,
                ),
                open(f"checkpoints/{self.experiment_name}.pkl", "wb"),
            )

    def first_run(self):
        self.exploration_before_start()
        total_step = 0
        rollout_length = 1
        max_fid_seen = 0
        epoch_step = 0
        return total_step, rollout_length, max_fid_seen, epoch_step

    def run(self, verbose=False, optimizer=None):
        self.verbose = verbose
        if not self.allow_respawn:
            total_step, rollout_length, max_fid_seen, epoch_step = self.first_run()
            pbar = tqdm(total=self.epochs // self.epoch_checkpoint_rate, leave=False)
        else:
            pbar = tqdm(total=self.epochs // self.epoch_checkpoint_rate, leave=False)
            if self.use_shots_to_recon_state:
                try:
                    (
                        epoch_step,
                        self.env_model,
                        self.agent,
                        self.env_pool,
                        self.model_pool,
                        self.ep_ret,
                        self.ep_len,
                        total_step,
                        rollout_length,
                        self.max_fid_seen,
                        self.running_controllers,
                        rollout_length,
                        train_policy_steps,
                        self.max_fid_est,
                        self.max_ep_ret,
                        self.funcalls,
                    ) = pickle.load(
                        open(f"checkpoints/{self.experiment_name}.pkl", "rb")
                    )
                # This error exception is hack to ascertain whether the algo is being run for the first time without
                # requiring an extra flag that a user must specify
                except FileNotFoundError:
                    print("First time running this experiment")
                    (
                        total_step,
                        rollout_length,
                        max_fid_seen,
                        epoch_step,
                    ) = self.first_run()
                    pass
            else:
                try:
                    (
                        epoch_step,
                        self.env_model,
                        self.agent,
                        self.env_pool,
                        self.model_pool,
                        self.ep_ret,
                        self.ep_len,
                        total_step,
                        rollout_length,
                        self.max_fid_seen,
                        self.running_controllers,
                        rollout_length,
                        train_policy_steps,
                        self.max_ep_ret,
                        self.funcalls,
                    ) = pickle.load(
                        open(f"checkpoints/{self.experiment_name}.pkl", "rb")
                    )

                except FileNotFoundError:
                    print("First time running this experiment")
                    (
                        total_step,
                        rollout_length,
                        max_fid_seen,
                        epoch_step,
                    ) = self.first_run()
                    pass
            if epoch_step > 0:
                pbar.update(epoch_step)

        # hyp tuning edit
        if optimizer is not None:
            try:
                (
                    self.agent.alpha_optim,
                    self.agent.critic_optim,
                    self.agent.policy_optim,
                ) = optimizer
            except ValueError:
                self.agent.critic_optim, self.agent.policy_optim = optimizer
            else:
                raise Exception("Unexpected packing in the SAC optimizer tuple found")

        while epoch_step < self.epochs:
            self.ep_ret = 0
            self.ep_len = 0
            start_step = total_step
            train_policy_steps = 0

            while True:
                cur_step = total_step - start_step

                if (
                    cur_step >= self.buffer_size
                    and len(self.env_pool) > self.args["min_pool_size"]
                ):
                    break
                ################################# ENABLE MODEL ##############################
                if self.use_model or self.use_ansatz_model:
                    if (
                        cur_step > 0
                        and cur_step % self.model_train_freq == 0
                        and self.args["real_ratio"] < 1.0
                    ):
                        if self.verbose:
                            print("training model")
                        if not self.debug_model:
                            self.train_predict_model()
                        if self.use_ansatz_model:
                            new_rollout_length = (
                                self.const_rollout_length
                            )  # FIX ROLLOUT LENGTH
                        else:
                            new_rollout_length = self.set_rollout_length(epoch_step)
                        if rollout_length != new_rollout_length:
                            rollout_length = new_rollout_length
                            self.model_pool = self.resize_model_pool(
                                rollout_length, self.model_pool
                            )

                        train_policy_steps = self.rollout_model(
                            rollout_length, total_step, train_policy_steps
                        )
                ################################# ENABLE MODEL ##############################
                self.reset_sac()
                if not self.use_shots_to_recon_state:
                    (
                        cur_state,
                        action,
                        next_state,
                        reward,
                        done,
                        _,
                    ) = self.env_sampler.sample(
                        self.agent, timestep=cur_step, num_timesteps=self.num_timesteps
                    )
                    print(action)
                    max_fid_seen = self.max_fid_seen
                    self.env_pool.push_timestep(cur_step % self.num_timesteps)
                    self.env_pool.push(cur_state, action, reward, next_state, done)
                    self.log_stuff(reward)
                else:
                    (
                        cur_state,
                        cur_state_est,
                        action,
                        next_state,
                        reward,
                        next_state_est,
                        reward_est,
                        done,
                        _,
                    ) = self.env_sampler.sample(
                        self.agent, timestep=cur_step, num_timesteps=self.num_timesteps
                    )
                    max_fid_seen = self.max_fid_seen
                    self.env_pool.push_timestep(cur_step % self.num_timesteps)
                    self.env_pool.push(
                        cur_state_est, action, reward_est, next_state_est, done
                    )
                    self.log_stuff(reward_est, reward)
                train_policy_steps = self.train_policy(train_policy_steps, total_step)
                total_step += 1
            ###########################  logging more stuff ################################
            # save model parameters
            if (epoch_step + 1) % self.epoch_checkpoint_rate == 0:
                # timelog=get_timelog() # save using epochs instead
                self.record["controllers"] = self.running_controllers
                self.record["funccalls"] = self.funcalls
                self.record["max_fid_seen"] = self.max_fid_seen
                if self.experiment_name:
                    pbar.update(1)
                    self.save_ac(
                        f"{self.experiment_name}__{epoch_step+1}__"
                    )  # TODO save model
                    self.save_record(f"{self.experiment_name}__{epoch_step+1}__")
                    ########################     respawn functionality     ##########################
                    if self.allow_respawn:
                        self.respawn_dump(
                            epoch_step, total_step, rollout_length, train_policy_steps
                        )
            self.ep_ret, self.ep_len = 0, 0
            epoch_step += 1

        max_fid_seen = self.max_fid_seen
        return max_fid_seen

    def save_ac(self, experiment_name: str):
        checkpoint = {
            "policy_state_dict": self.agent.policy.state_dict(),
            "vf_state_dict": self.agent.critic.state_dict(),
        }
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(checkpoint, f"models/model_{experiment_name}.pth")


if __name__ == "__main__":
    from src.ppo import add_args

    args = add_args()
    # print(args.const_rollout_length, isinstance(args.const_rollout_length, int))
    System, target, tname, num_qubits, trl, bmin, bmax, hidden = str_to_qsys(
        args.system
    )
    print("running delta", args.imperfection_delta)
    print("buffer size is ", args.buffer_size)
    trial = MBSAC(
        num_timesteps=20,
        final_time=20,
        epochs=args.epochs,
        dissipate=args.dissipate,
        decay_1=args.decay_1,
        decay_2=args.decay_2,
        saved_model_path=args.load_saved_model,
        experiment_name=args.experiment_name,
        target=target,
        use_model=args.use_rl_model,
        use_ansatz_model=args.use_ham_model,
        debug_model=args.debug_model,
        model_train_iterations=args.model_train_iterations,
        const_rollout_length=args.const_rollout_length,
        imperfection_delta=args.imperfection_delta,
        use_ruthless_delta=args.use_ruthless_delta,
        ham_noise_level=args.ham_noise_level,
        use_shots_to_recon_state=args.use_shots,
        static_lind=args.static_lind,
        learn_diss=args.learn_diss,
        learn_diss_coeffs_only=args.learn_diss_coeffs_only,
        override_ham_init=args.use_totally_random_ham,
        System=args.system,
        trl=trl,
        qubits=num_qubits,
        tname=tname,
        bmin=bmin,
        bmax=bmax,
        buffer_size=args.buffer_size,
        learn_ham_improv_threshold=args.improv_thres,
        hidden=hidden,
        allow_respawn=args.respawn,
    ).run(verbose=args.verbose)
