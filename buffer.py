import torch
import numpy as np
from typing import Dict
from torch import Tensor
from tools import array, mpi


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by PPO agents interacting with the environment, and using Generalized
    Advantage Estimation (GAE-Lambda) for calculating the advantages of state-action pairs.
    """

    def __init__(self, state_size: int, action_size: int, size: int, gamma: float = 0.99, lam: float = 0.95):
        self.state_buf = np.zeros(array.combined_shape(size, state_size), dtype=np.float32)
        self.action_buf = np.zeros(array.combined_shape(size, action_size), dtype=np.float32)
        self.advantage_buf = np.zeros(size, dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.rewards_to_go_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr = 0    # Number of current steps
        self.path_start_idx, self.max_size = 0, size

    def store(self, state: np.ndarray, action: np.ndarray, reward: float, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store

        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: float = 0):
        """
        Call this at the end of a trajectory, or when one gets cut off by an epoch ending. This looks back in the buffer
        to where the trajectory started, and uses rewards and value estimates from the whole trajectory to compute
        advantage estimates with GAE-Lambda, as well as compute the rewards-to-go for each state, to use as
        the targets for the value function. The "last_val" argument should be 0 if the trajectory ended because the
        agent reached a terminal state (died), and otherwise should be V(s_T), the value function estimated for the
        last state. This allows us to bootstrap the reward-to-go calculation to account for timesteps beyond the
        arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.reward_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.advantage_buf[path_slice] = array.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.rewards_to_go_buf[path_slice] = array.discount_cumsum(rewards, self.gamma)[:-1]

        # Reset start path pointer
        self.path_start_idx = self.ptr

    def get(self) -> Dict[str, Tensor]:
        """
        Call this at the end of an epoch to get all of the data from the buffer, with advantages appropriately
        normalized (shifted to have mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get

        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi.mpi_statistics_scalar(self.advantage_buf)
        self.advantage_buf = (self.advantage_buf - adv_mean) / adv_std
        data = dict(
            states=self.state_buf,
            actions=self.action_buf,
            ret=self.rewards_to_go_buf,
            advantages=self.advantage_buf,
            logp=self.logp_buf
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
