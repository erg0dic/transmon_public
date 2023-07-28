import numpy as np
import torch
from src.baseclass import CNOT, QSys
from src.utilities import add_args
import os
from tqdm import tqdm
from src.non_para_dynamics import EnsembleDynamicsModel
from src.ppo import PPO, PPOBuffer
from operator import itemgetter
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnvSampler:
    def __init__(self, env: PPO, max_path_length=1000):
        self.env = env
        self.path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        # self.path_rewards = []
        self.sum_reward = 0

    def sample(
        self,
        agent,
        eval_t=False,
        timestep=None,
        num_timesteps=None,
        set_actions_to_zero=False,
    ):
        
        if self.current_state is None:
            self.current_state = self.env.reset()
        ## modify for use_shots_to_recon_state
        cur_state = self.current_state
        if self.env.use_shots_to_recon_state:
            cur_state_ = cur_state.copy()
            cur_state_ = (
                cur_state_[: len(cur_state_) // 2]
                + 1j * cur_state_[len(cur_state_) // 2 :]
            )
            cur_state_ = cur_state_.reshape(self.env.obs_dim, -1)
            cur_state_est = QSys.super_to_choi(
                self.env.fuzzify(QSys.super_to_choi(cur_state_), shots=self.env.M)
            )
            cur_state_est = cur_state_est.ravel()
            cur_state_est = np.concatenate(
                [np.real(cur_state_est), np.imag(cur_state_est)]
            )
            action = agent.select_action(cur_state_est, eval_t)
        else:
            action = agent.select_action(cur_state, eval_t)
        if set_actions_to_zero:
            action = np.zeros_like(action)
            action[1] = 1
        if self.env.use_shots_to_recon_state:
            (
                next_state,
                reward,
                next_state_est,
                reward_est,
                terminal,
                info,
            ) = self.env.step(action)
        else:
            next_state, reward, terminal, info = self.env.step(action)
        self.path_length += 1
        # self.sum_reward += reward if not self.env.use_shots_to_recon_state else reward_est
        ending = (timestep + 1) % num_timesteps == 0
        if ending or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            # self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
            self.env.reset()
        else:
            self.current_state = next_state
        if self.env.use_shots_to_recon_state:
            return (
                cur_state,
                cur_state_est,
                action,
                next_state,
                reward,
                next_state_est,
                reward_est,
                terminal,
                info,
            )
        else:
            return cur_state, action, next_state, reward, terminal, info


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.time_idxes_buffer = []
        self.position = 0

    def add_to_buffer(self, *args, buffer=None):
        if type(buffer) == type(None):
            raise AssertionError("You must provide a buffer for this method.")
        if len(buffer) < self.capacity:
            buffer.append(None)
        buffer[self.position] = args

    def push(self, state, action, reward, next_state, done):
        self.add_to_buffer(state, action, reward, next_state, done, buffer=self.buffer)
        self.position = (self.position + 1) % self.capacity

    def push_timestep(self, timestep):
        self.add_to_buffer(timestep, buffer=self.time_idxes_buffer)

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[
                : len(self.buffer) - self.position
            ]
            self.buffer[: len(batch) - len(self.buffer) + self.position] = batch[
                len(self.buffer) - self.position :
            ]
            self.position = len(batch) - len(self.buffer) + self.position

    def sample(self, batch_size, get_timesteps=False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        idxes = random.sample(
            range(len(self.buffer)), int(batch_size)
        )  # unique samples without replacement
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        if not get_timesteps:
            return state, action, reward, next_state, done, None
        else:
            timesteps = list(itemgetter(*idxes)(self.time_idxes_buffer))
            timesteps = np.stack(timesteps).ravel()
            return state, action, reward, next_state, done, timesteps

    def sample_all_batch(self, batch_size, get_timesteps=False):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        if not get_timesteps:
            return state, action, reward, next_state, done
        else:
            timesteps = list(itemgetter(*idxes)(self.time_idxes_buffer))
            timesteps = np.stack(timesteps).ravel()
            return state, action, reward, next_state, done, timesteps

    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)


class MBPO_Buffer(PPOBuffer):
    def __init__(self, *args, env_buffer_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_buffer_size = env_buffer_size
        self.next_obs_buf = np.zeros_like(self.obs_buf)

    def get_rand_sample_indices(self, size, all, max_id=None):
        id_max = max_id if not all else self.max_size
        # size = (size-1)%self.max_size+1
        sample_indices = np.random.randint(id_max, size=size)
        return sample_indices

    def sample(self, size, all=False):
        sample_indices = self.get_rand_sample_indices(size, max_id=self.ptr, all=all)

        return (
            self.obs_buf[sample_indices],
            self.act_buf[sample_indices],
            self.rew_buf[sample_indices],
            self.next_obs_buf[sample_indices],
        )

    def store(self, obs, act, rew, val, logp, next_obs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.next_obs_buf[self.ptr] = next_obs
        self.ptr += 1

    def push_batch(self, obs, act, rew, values, next_obs, advantages=None):
        "push stuff to the buffer in chunks"
        ptr_shift = len(obs)
        rew = rew.ravel()
        # roll around the buffer after exceeding capacity
        if ptr_shift + self.ptr <= self.max_size:
            roll_around = 0
            fit = ptr_shift + self.ptr
            self.obs_buf[self.ptr : fit] = obs
            self.act_buf[self.ptr : fit] = act
            self.rew_buf[self.ptr : fit] = rew
            self.val_buf[self.ptr : fit] = values
            self.next_obs_buf[self.ptr : fit] = next_obs
            if type(advantages) != type(None):
                self.adv_buf[self.ptr : fit] = advantages
            self.ptr += ptr_shift
        else:
            fit = self.max_size - self.ptr
            roll_around = abs(ptr_shift - fit)
            self.obs_buf[self.ptr :] = obs[:fit]
            self.act_buf[self.ptr :] = act[:fit]
            self.rew_buf[self.ptr :] = rew[:fit]
            self.val_buf[self.ptr :] = values[:fit]
            self.next_obs_buf[self.ptr :] = next_obs[:fit]

            if roll_around != 0:
                self.obs_buf[:roll_around] = obs[fit:]
                self.act_buf[:roll_around] = act[fit:]
                self.rew_buf[:roll_around] = rew[fit:]
                self.val_buf[:roll_around] = values[fit:]
                self.next_obs_buf[:roll_around] = next_obs[fit:]
                self.ptr = roll_around
            else:
                self.ptr = ptr_shift

    def get(self, only_env_buffer=False):
        "Right now, training on the most recent `epoch_buffer_size` data"
        if only_env_buffer:
            path_slice = slice(-self.epoch_buffer_size, None)
        else:
            path_slice = self.get_rand_sample_indices(
                self.epoch_buffer_size, all=False, max_id=self.ptr
            )
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = (
            self.adv_buf[path_slice].mean(),
            self.adv_buf[path_slice].std(),
        )
        self.adv_buf = (self.adv_buf[path_slice] - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf[path_slice],
            act=self.act_buf[path_slice],
            ret=self.ret_buf[path_slice],
            adv=self.adv_buf,
            logp=self.logp_buf[path_slice],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def return_all(self):
        return (
            self.obs_buf,
            self.act_buf,
            self.rew_buf,
            self.val_buf,
            self.next_obs_buf,
            self.adv_buf,
        )
