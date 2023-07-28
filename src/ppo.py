"""
ppo code adapted from the spinningup repository
"""
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from src.baseclass import CNOT, QSys
from src.baseopt import BaseOpt
from src.utilities import add_args
from torch.optim import Adam
import json
import time as tt
from src.utilities import super_pre, super_post, super_op, dm2vec, vec2dm
from scipy.linalg import expm
import os
import warnings
from tqdm import tqdm
from typing import List
from src.utilities import get_pert_ham_ruthless, get_pauli_basis_matrices
from src.gjcummings import GJC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes: list, activation, output_activation=nn.Identity):
    "multilayer perceptron"
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.

        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1
        )  # Last axis sum needed for Torch Normal distribution


# from play.n_ode import NeuralODE
# from torchdiffeq import odeint_adjoint as odeint
class MLPCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_sizes,
        activation,
        use_learned_fourier=False,
        concat_Fourier=False,
        fourier_std=0.1,
        target=None,
        dissipate=False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.use_learned_fourier = use_learned_fourier
        self.concat_fourier = concat_Fourier
        self.target = target
        self.dissipate = dissipate
        self.target_view = int(np.sqrt(self.obs_dim))
        self.norm = self.target.shape[-1] * self.target.shape[-1]
        if self.use_learned_fourier:
            self.B = nn.Parameter(
                torch.normal(
                    torch.zeros(obs_dim, hidden_sizes[0] // 2),
                    torch.full((obs_dim, hidden_sizes[0] // 2), fourier_std),
                )
            )
            self.B.requires_grad = True

        if self.concat_fourier:
            obs_dim += hidden_sizes[0]
        elif self.use_learned_fourier:
            obs_dim = hidden_sizes[0]
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        # self.lstm = nn.LSTM(1, obs_dim, batch_first=True)
        # self.final = nn.Linear(obs_dim, 1)
        # self.ft = nn.Tanh()

    def cost_function(self, obs):
        obs = obs.view(self.target_view, -1)
        if not self.dissipate:
            return torch.trace(obs.conj().T @ self.target) ** 2 / self.norm
        else:
            raise NotImplementedError

    def forward(self, obs):
        # t_ = torch.ones_like(obs)*t
        # obs = torch.cat([t_, obs], -1)
        if self.use_learned_fourier:
            proj = (2 * np.pi) * torch.matmul(obs, self.B)
            ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
            if self.concat_fourier:
                obs = torch.cat([obs, ff], dim=-1)
            else:
                obs = ff
        o = self.v_net(obs)
        out = torch.squeeze(o, -1)  # + self.cost_function(obs)
        # o, _ = self.lstm(o.view(-1,obs.shape[-1], 1)) # (Batch, seq_len, inp_dim)
        # return torch.squeeze(self.ft(self.final(o.squeeze(-1))).squeeze(-1).sum(axis=-1), -1) # Critical to ensure v has right shape.
        return out


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes=(100, 100),
        activation=nn.Tanh,
        use_learned_fourier=False,
        concat_fourier=False,
        fourier_std=0.1,
        target=None,
        dissipate=False,
    ):
        super().__init__()
        if not dissipate:
            target = torch.as_tensor(target, dtype=torch.float32)
        else:
            target = torch.as_tensor(super_op(target), dtype=torch.float32)
        # policy function
        self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)

        # value function
        self.v = MLPCritic(
            obs_dim,
            hidden_sizes,
            activation,
            use_learned_fourier=use_learned_fourier,
            fourier_std=fourier_std,
            concat_Fourier=concat_fourier,
            target=target,
            dissipate=dissipate,
        )
        self.final_act = nn.Tanh()
        self.final = nn.Linear(obs_dim, 1)

    def step(self, obs, t=None):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            # v = odeint(self.v, obs, t)
            # v = self.final_act(self.final(v))
            # v = v.sum()
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size=10000, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, only_env_buffer=False):

        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        # print(self.adv_buf, len(self.adv_buf))
        # breakpoint()
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class PPO(BaseOpt):
    def __init__(
        self,
        *args,
        init_state=None,
        bmin=-10,
        bmax=10,
        repeats=100,
        fid_threshold=0.98,
        timestep_res=0.5,
        epochs=10000,
        rollouts=4000,
        save=False,
        timeout=1800,
        verbose=False,
        ham_noisy=False,
        draws=10,
        adaptive=False,
        adp_tol=0.05,
        testing=False,
        save_topc: int = 100,
        train_pi_iters=250,
        train_v_iters=250,
        clip_ratio=0.2,
        lam=0.97,
        gamma=0.99,
        pi_lr=3e-3,
        vf_lr=1e-3,
        use_fixed_ham: bool = False,
        opt_train_size: int = 100,
        records_update_rate: float = None,
        hidden=100,
        layers=3,
        saved_model_path: str = None,
        experiment_name: str = None,
        epoch_checkpoint_rate: int = 2,
        buffer_size=3000,
        M: int = int(pow(10, 14)),
        use_learned_fourier: bool = False,
        fourier_std=0.1,
        concat_fourier=False,
        add_miss=False,
        use_shots_to_recon_state=False,
        ham_noise_level: float = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        #### hyperparameters to be optimized
        self.use_shots_to_recon_state = use_shots_to_recon_state
        if self.use_shots_to_recon_state:
            self.dissipate = True
        self.M = M
        self.lam = lam
        self.gamma = gamma
        self.ham_noise_level = ham_noise_level
        self.buffer_size = buffer_size

        self.save_topc = save_topc
        self.experiment_name = experiment_name
        self.use_learned_fourier = use_learned_fourier
        self.add_miss = add_miss
        self.fourier_std = fourier_std
        self.concat_fourier = concat_fourier
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        from copy import deepcopy

        self.new_qsys = deepcopy(self.qsys)

        self.Tmin = 0
        self.Bmin = bmin
        self.Bmax = bmax
        self.repeats = repeats
        self.timestep_res = timestep_res
        self.draws = draws
        self.ham_noisy = ham_noisy
        self.verbose = verbose
        self.timeout = timeout
        self.adaptive = adaptive
        self.adp_func_calls_increment = self.draws
        self.adp_var_tol = adp_tol
        self.use_fixed_ham = use_fixed_ham
        self.train_size = opt_train_size
        self.clip_ratio = clip_ratio
        self.train_pi_iters = int(train_pi_iters)
        self.train_v_iters = int(train_v_iters)
        self.epoch_checkpoint_rate = epoch_checkpoint_rate

        self.num_controls_per_timestep = len(self.qsys.reduced_cont_basis)

        if init_state is None:
            self._init_state(init_state)

        self.hidden = [hidden] * layers
        if self != "MBPPO()":
            self.ac = MLPActorCritic(
                self.obs_dim * self.obs_dim * 2,
                self.num_controls_per_timestep,
                hidden_sizes=self.hidden,
                use_learned_fourier=use_learned_fourier,
                fourier_std=fourier_std,
                concat_fourier=concat_fourier,
                target=self.target,
                dissipate=self.dissipate,
            )

            self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
            self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
            self.buf = PPOBuffer(
                self.obs_dim * self.obs_dim * 2,
                self.qubits,
                self.buffer_size,
                self.gamma,
                self.lam,
            )

        if saved_model_path:
            self.load_saved_ac_model(saved_model_path)
            self.ac.train()

        self.state = self.init_state.copy()
        # Create actor-critic module

        self.epochs = epochs
        self.rollouts = rollouts
        self.repeats = repeats
        self.fid_threshold = fid_threshold
        self.total_rewards = []

        # self.filename = self.filename_generator()
        self.save = save
        self.testing = testing
        self.record = {}
        self.record_update_rate = records_update_rate  # every 1e5 function calls
        self.update_counter = 0

        self.control_sequence_for_epoch = np.zeros(
            (self.num_timesteps, self.num_controls_per_timestep)
        )
        assert (
            self.buffer_size % self.num_timesteps == 0
        ), "can't have incomplete control pulse runs"
        self.pauli_basis_2n = get_pauli_basis_matrices(2 * self.qubits, trl=self.trl)
        self.static_diss_term = self.get_col_op_accumulant()
        target = QSys.super_to_choi(super_op(self.target))
        self.target_vec = dm2vec(target / target.trace(), self.pauli_basis_2n)

    def __repr__(self):
        return "PPO()"

    def _init_state(self, init_state):
        if init_state:
            self.init_state = init_state.copy()
        else:
            if not self.dissipate:
                self.obs_dim = pow(self.trl, self.qubits)
            else:
                self.obs_dim = self.rank * self.rank  # density matrix dimensions

            # self.init_state = np.zeros(self.obs_dim)
            # self.init_state[0] = 1.
            self.init_state = np.eye(self.obs_dim, dtype=np.complex128).ravel()

    def reset(self):
        # reset the state and return a reward too?
        self.state = self.init_state.copy()
        self.control_sequence_for_epoch = np.zeros(
            (self.num_timesteps, self.num_controls_per_timestep)
        )
        self.timestep = 0
        # return np.zeros(self.qubits)
        return np.concatenate([np.real(self.state), np.imag(self.state)])

    def fuzzify(self, Ch, shots):
        "estimate the choi state using M shots"
        assert Ch.shape[-1] == int(pow(self.trl, self.qubits * 2))
        # return Ch
        trace_scale = Ch.trace()
        Ch /= trace_scale
        # print(Ch)
        try:
            pauli_exps = np.real(dm2vec(Ch, self.pauli_basis_2n))
            pauli_probs = 0.5 * (1 + pauli_exps)
            scale = pauli_probs.sum(axis=-1)
            # sum(p) = 1 for multinomial sampling
            pauli_probs /= scale
            ss_pauli_probs = np.random.multinomial(shots, pauli_probs) * scale / shots
            exp_pauli_exps = 2 * ss_pauli_probs - 1
            fuzzy_reconstructed = trace_scale * vec2dm(
                exp_pauli_exps, self.pauli_basis_2n
            )

        except ValueError as e:
            print(e)
            breakpoint()
            raise e
        return fuzzy_reconstructed

    def take_entire_trajectory_step(self, actions: np.ndarray):
        assert (
            actions.shape == (self.num_timesteps, self.num_controls_per_timestep),
            f"expected shape {(self.num_timesteps, self.num_controls_per_timestep)}"
            + f"but got {actions.shape}",
        )
        for action in actions:
            obs, reward = self.take_step(action)
        return obs, reward  # terminal

    def take_step(self, action: List):
        action = (
            action % np.sign(action) * self.Bmax
            if (np.abs(action) > self.Bmax).any()
            else action
        )
        if self.ham_noise_level > 0:
            nHam = get_pert_ham_ruthless(
                self.qsys, gns_strength_std=self.ham_noise_level
            )
            self.qsys.transmon_sys_ham = nHam
            Ham = self.qsys.hamiltonian_full(action)
        else:
            Ham = self.qsys.hamiltonian_full(action)
        self.state = self.state.reshape(self.obs_dim, self.obs_dim)
        if not self.dissipate:
            self.state = expm(-1j * self.dt * Ham) @ self.state
            reward = self.cost_function(self.state, self.target)
            if self.ham_noise_level > 0:
                self.true_reward = self.cost_function(
                    expm(-1j * self.dt * self.new_qsys.hamiltonian_full(action))
                    @ self.state
                )

            # reward = self.rho_fid(self.state @ self.state.conj().T, self.target@ self.init_state @self.target.conj().T, 2)
        else:
            # dL/dt = H L
            lind_op = -1j * (super_pre(Ham) - super_post(Ham))
            if self.static_lind:
                lind_op += self.static_diss_term
            # # else:
            # lind_op += self.get_col_op_accumulant()
            self.state = expm(lind_op * self.dt) @ self.state

            if self.use_shots_to_recon_state:
                true_state = QSys.super_to_choi(self.state.copy())
                self.true_reward = self.qsys.pauli_vec_norm_reward(
                    dm2vec(true_state / true_state.trace(), self.pauli_basis_2n),
                    self.target_vec,
                )  # self.qsys.superoper_average_fidelity(self.state, super_op(self.target))
                self.recon_state = self.fuzzify(true_state, shots=self.M)
                reward = self.qsys.pauli_vec_norm_reward(
                    dm2vec(
                        self.recon_state / self.recon_state.trace(), self.pauli_basis_2n
                    ),
                    self.target_vec,
                )  # self.qsys.superoper_average_fidelity(self.recon_state, super_op(self.target))
                self.recon_state = QSys.super_to_choi(
                    self.recon_state
                )  # convert back to super
                # assert np.allclose(self.recon_state-self.state,0, atol=1e-1), breakpoint()
                # from copy import deepcopy
                # self.recon_state = deepcopy(self.state)
                self.recon_state = self.recon_state.ravel()
                self.recon_state = np.concatenate(
                    [np.real(self.recon_state), np.imag(self.recon_state)]
                )
            # reward = self.rho_fid(vec2mat(state), self.target@vec2mat(init)@self.target.conj().T, 2)

        self.control_sequence_for_epoch[self.timestep] = action
        if self.verbose:
            c = self.control_sequence_for_epoch[: self.timestep + 1]
            if not self.use_shots_to_recon_state:
                # internal test
                if not self.dissipate and not self.ham_noise_level > 0:
                    assert np.allclose(self.infidelity_uni(c), 1 - reward)
                else:
                    assert np.allclose(self.infidelity_Lind(c), 1 - reward)

        self.timestep += 1
        o = self.state.ravel()
        o = np.concatenate([np.real(o), np.imag(o)])
        if self.use_shots_to_recon_state:
            return o, self.true_reward, self.recon_state, reward
        else:
            return o, reward  ## more specific case
        # return action, reward   ### completely non-informative

    def save_record(self, experiment_name):
        if not os.path.exists("conts_dicts"):
            os.mkdir("conts_dicts")
        file_path = os.path.join("conts_dicts", f"conts_{experiment_name}.json")
        json.dump(self.record, open(file_path, "w"))

    def read_record(self, experiment_name: str):
        file_path = os.path.join("conts_dicts", f"conts_{experiment_name}.json")
        return json.load(open(file_path))

    def filename_generator(self):
        raise NotImplementedError

    def infidelity(self, c):
        raise NotImplementedError

    def load_saved_ac_model(self, experiment_name: str):
        file_path = os.path.join("models", f"model_{experiment_name}.pth")
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            self.ac.load_state_dict(checkpoint["model_state_dict"])
            self.pi_optimizer.load_state_dict(checkpoint["pi_optimizer_state_dict"])
            self.vf_optimizer.load_state_dict(checkpoint["vf_optimizer_state_dict"])
        else:
            warnings.warn("couldn't load model defaulting to random weight init...")

    def save_ac(self, experiment_name: str):
        checkpoint = {
            "model_state_dict": self.ac.state_dict(),
            "pi_optimizer_state_dict": self.pi_optimizer.state_dict(),
            "vf_optimizer_state_dict": self.vf_optimizer.state_dict(),
            "buffer": self.buf.obs_buf.tolist(),
        }
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(checkpoint, f"models/model_{experiment_name}.pth")

    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data["obs"], data["ret"]
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def update(
        self, pi_optimizer, vf_optimizer, target_kl=0.01, only_from_env_buffer=False
    ):

        data = self.buf.get(only_from_env_buffer)

        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info["kl"]
            # print(f"kl = {kl}")
            # raise AssertionError("cbp")
            if kl > 1.5 * target_kl:
                # print('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            # print(loss_pi)
            # mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        # Value function learning
        iters = range(self.train_v_iters)
        for i in iters:
            vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            # print(loss_v)
            # mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

    def run(
        self,
        seed=0,
        max_ep_len=1000,
        target_kl=0.01,
        logger_kwargs=dict(),
        save_freq=10,
        optimizer=None,
    ):
        self.true_max_fid = 0
        self.recon_state = None

        if self.testing:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Set up optimizers for policy and value function
        if optimizer is None:
            pi_optimizer = self.pi_optimizer
            vf_optimizer = self.vf_optimizer
        else:
            pi_optimizer = optimizer[0]
            vf_optimizer = optimizer[1]

        # Prepare for interaction with environment

        o, ep_ret, ep_len = self.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        max_fid_seen = 0
        true_fid = 0
        funcalls = 0
        max_ep_ret = 0
        running_controllers = {}
        pbar = tqdm(total=self.epochs // self.epoch_checkpoint_rate, leave=False)
        alpha = 0.2
        for epoch in range(self.epochs):
            for t in range(self.buffer_size):
                # print(torch.as_tensor(o, dtype=torch.float32))
                times = torch.as_tensor(
                    [self.timestep * self.dt, (self.timestep + 1) * self.dt],
                    dtype=torch.float32,
                )

                a, v, logp = self.ac.step(
                    torch.as_tensor(o, dtype=torch.float32), times
                )
                # a = a[:len(a)//2] + 1j*a[len(a)//2:]
                # next_o, r = self.step(a+o) # changes in the observation

                next_o, r = self.take_step(a)
                # v = r + self.gamma*v

                ep_ret += r
                ep_len += 1
                funcalls += 1

                if max_fid_seen >= r:
                    max_fid_seen = max(max_fid_seen, r)
                    self.true_max_fid = self.infidelity_uni(c)

                if self.use_shots_to_recon_state:
                    self.true_max_fid = max(self.true_max_fid, self.true_reward)
                max_ep_ret = max(ep_ret, max_ep_ret)

                if self.verbose:
                    if not self.use_shots_to_recon_state:
                        print(f"max_fid_obtained: {max_fid_seen}, fid: {r}")
                    elif self.ham_noise_level > 0:
                        print(
                            f"max_fid_obtained, true: {max_fid_seen, self.true_max_fid}, fid: {r}"
                        )
                    else:
                        print(
                            f"max_fid_fuzzy, max_true_fid: {max_fid_seen, self.true_max_fid}, fid: {r}"
                        )
                    print(f"func calls {funcalls}", f"max ep_ret: {max_ep_ret}")

                # print(max_fid_seen, r)

                # save and log
                if self.use_shots_to_recon_state:
                    self.buf.store(self.recon_state, a, r, v, logp)
                else:
                    self.buf.store(o, a, r, v, logp)
                # Update obs (critical!)
                o = next_o

                terminal = (t + 1) % self.num_timesteps == 0
                epoch_ended = None  # t+1==self.buffer_size  # currently not needed

                # if max_fid_seen > self.fid_threshold:
                #     return
                if self.save_topc:
                    l = len(running_controllers.keys())
                    if l < self.save_topc:
                        running_controllers[r] = self.control_sequence_for_epoch[
                            : self.timestep
                        ].tolist()
                        # print("running_list: \n", running_controllers)
                    else:
                        # itopop=self.find_min_fid_index(running_controllers) # time to pop this ###
                        itopop = min(list(running_controllers.keys()))
                        running_controllers.pop(itopop)
                        running_controllers[r] = self.control_sequence_for_epoch[
                            : self.timestep
                        ].tolist()  # maintain const size list
                        if self.testing:
                            f = list(running_controllers.keys())[0]
                            c = np.array(running_controllers[f])
                            assert np.allclose(
                                float(f), 1 - self.infidelity_Lind(c)
                            ), print(float(f), self.infidelity_Lind(c))
                            if self.verbose:
                                print("running_list: \n", running_controllers.keys())

                if terminal or epoch_ended:
                    # if epoch_ended and not(terminal):
                    # print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.ac.step(
                            torch.as_tensor(o.ravel(), dtype=torch.float32)
                        )
                    else:
                        v = 0
                    self.buf.finish_path(v)
                    # if terminal:
                    #     # only save EpRet / EpLen if trajectory finished
                    #     # logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, ep_ret, ep_len = self.reset(), 0, 0

            # Perform PPO update!
            # if funcalls > self.buffer_size:
            self.update(pi_optimizer, vf_optimizer, target_kl)

            # save model parameters
            if (epoch + 1) % self.epoch_checkpoint_rate == 0:
                # timelog=get_timelog() # save using epochs instead
                self.record["controllers"] = running_controllers
                self.record["funccalls"] = funcalls
                self.record["max_fid_seen"] = max_fid_seen
                if self.experiment_name:
                    pbar.update(1)
                    self.save_ac(f"{self.experiment_name}__{epoch+1}__")
                    self.save_record(f"{self.experiment_name}__{epoch+1}__")
        pbar.close()
        return max_fid_seen


def cleanup(keywords=["test"]):
    import glob

    mpaths = glob.glob("models/*")
    cpaths = glob.glob("conts_dicts/*")
    for mpath in mpaths:
        for keyword in keywords:
            if keyword in mpath:
                os.remove(mpath)
    for cpath in cpaths:
        for keyword in keywords:
            if keyword in cpath:
                os.remove(cpath)


CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

if __name__ == "__main__":
    args = add_args()

    trial = PPO(
        target=CNOT,
        trl=2,
        num_timesteps=20,
        verbose=args.verbose,
        epochs=args.epochs,
        dissipate=args.dissipate,
        decay_1=args.decay_1,
        decay_2=args.decay_2,
        saved_model_path=args.load_saved_model,
        experiment_name=args.experiment_name,
        buffer_size=args.buffer_size,
        use_learned_fourier=args.use_learned_fourier,
    )

    trial.run()
    # perform cleanup
    # cleanup()
