import time
import torch
import numpy as np
from torch import Tensor
from tools import mpi, nets
from config import settings
from buffer import PPOBuffer
from torch.optim import Adam
from collections import deque
from agents import ActorCritic
from tools.log import EpochLogger
from unityagents import UnityEnvironment
from typing import Dict, Callable, Tuple


class PPO:
    def __init__(
            self,
            env_fn: Callable[[int], Tuple[UnityEnvironment, str, int, int, Tuple[float]]],
            seed: int,
            steps_per_epoch: int,
            epochs: int,
            gamma: float,
            clip_ratio: float,
            policy_lr: float,
            value_lr: float,
            train_policy_iters: int,
            train_value_iters: int,
            lam: float,
            max_ep_len: int,
            target_kl: float,
            logger_kwargs: Dict[str, str],
            save_freq: int
    ):
        """
        Proximal Policy Optimization (by clipping),
        with early stopping based on approximate KL
        Args:
            env_fn : A function which creates a copy of the environment. The environment must satisfy the OpenAI Gym API
            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object you provided to PPO.
            seed: Seed for random number generators.
            steps_per_epoch: Number of steps of interaction (state-action pairs) for the agent and the environment in
                             each epoch.
            epochs: Number of epochs of interaction (equivalent to number of policy updates) to perform.
            gamma: Discount factor. (Always between 0 and 1.)
            clip_ratio: Hyperparameter for clipping in the policy objective. Roughly: how far can the new policy go from
                        the old policy while still profiting (improving the objective function)? The new policy can
                        still go farther than the clip_ratio says, but it doesn't help on the objective anymore.
                        (Usually small, 0.1 to 0.3.) Typically denoted by :math:`\epsilon`.
            policy_lr: Learning rate for policy optimizer.
            value_lr: Learning rate for value function optimizer.
            train_policy_iters: Maximum number of gradient descent steps to take on policy loss per epoch.
                                (Early stopping may cause optimizer to take fewer than this.)
            train_value_iters: Number of gradient descent steps to take on value function per epoch.
            lam: Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)
            max_ep_len: Maximum length of trajectory / episode / rollout.
            target_kl: Roughly what KL divergence we think is appropriate between new and old policies after an update.
                       This will get used for early stopping. (Usually small, 0.01 or 0.05.)
            logger_kwargs: Keyword args for EpochLogger.
            save_freq: How often (in terms of gap between epochs) to save the current policy and value function.
        """

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        mpi.setup_pytorch_for_mpi()

        # Set up logger and save configuration
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(settings.as_dict())

        # Random seed
        seed += 10000 * mpi.proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Instantiate environment
        self.env, self.brain_name, state_size, action_size, state = env_fn(seed)

        # Create actor-critic module
        self.ac = ActorCritic(
            state_size=state_size,
            action_size=action_size,
            seed=seed,
            layers=settings.ActorCritic.layers,
            hidden_nodes=settings.ActorCritic.hidden_nodes,
            activation=settings.ActorCritic.activation
        )

        # Sync params across processes
        mpi.sync_params(self.ac)

        # Count variables
        var_counts = tuple(nets.count_vars(module) for module in [self.ac.pi, self.ac.v])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        # Set up experience buffer
        self.steps_per_epoch = steps_per_epoch
        self.local_steps_per_epoch = int(self.steps_per_epoch / mpi.num_procs())
        self.buf = PPOBuffer(state_size, action_size, self.local_steps_per_epoch, gamma, lam)

        # Set up optimizers for policy and value function
        self.policy_optimizer = Adam(self.ac.pi.parameters(), lr=policy_lr)
        self.value_optimizer = Adam(self.ac.v.parameters(), lr=value_lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

        # Save properties.
        self.clip_ratio = clip_ratio
        self.train_policy_iters = train_policy_iters
        self.train_value_iters = train_value_iters
        self.target_kl = target_kl
        self.epochs = epochs
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        self.last_rewards = deque(maxlen=settings.PPO.epochs_mean_rewards)

    def train(self):

        # Prepare for interaction with environment
        start_time = time.time()
        state, ep_ret, ep_len = self.env.reset(train_mode=True)[self.brain_name].vector_observations[0], 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.epochs):
            for t in range(self.local_steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(state, dtype=torch.float32))

                brain_info = self.env.step(a)[self.brain_name]
                next_state = brain_info.vector_observations[0]
                r = brain_info.rewards[0]
                d = brain_info.local_done[0]
                ep_ret += r
                ep_len += 1

                # save and log
                self.buf.store(state, a, r, v, logp)
                self.logger.store(VVals=v)

                # Update obs (critical!)
                state = next_state

                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.local_steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not terminal:
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(state, dtype=torch.float32))
                    else:
                        v = 0
                    self.buf.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                        self.last_rewards.append(ep_ret)
                    state, ep_ret, ep_len = self.env.reset(True)[self.brain_name].vector_observations[0], 0, 0

            avg_last_rewards = mpi.mpi_avg(sum(self.last_rewards)/len(self.last_rewards))
            if (avg_last_rewards > settings.PPO.env_solved_at) and (epoch >= settings.PPO.epochs_mean_rewards):
                print("Environmente solved!")
                self.logger.save_state({}, epoch)
                break

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
                self.logger.save_state({}, None)

            # Perform PPO update!
            self.update()

            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular(f'AvgRewardsLast{settings.PPO.epochs_mean_rewards}Ep', avg_last_rewards)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('TotalEnvInteracts', (epoch+1) * self.steps_per_epoch)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossPi', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
            self.logger.log_tabular('Entropy', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            self.logger.log_tabular('ClipFrac', average_only=True)
            self.logger.log_tabular('StopIter', average_only=True)
            self.logger.log_tabular('Time', time.time()-start_time)
            self.logger.dump_tabular()

        print("training finished")
        if mpi.proc_id() == 0:
            self.logger.plot_rewards()
        self.env.close()

    def compute_loss_policy(self, data: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """ Set up function for computing PPO policy loss"""

        states, actions, advantages, logp_old = data['states'], data['actions'], data['advantages'], data['logp']

        # Policy loss
        pi, logp = self.ac.pi(states, actions)
        ratio = torch.exp(logp - logp_old)
        print(f"compute_loss_policy\nratio shape: {ratio.shape}\nvalues:\n{ratio}")
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        loss_pi = -(torch.min(ratio * advantages, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, data: Dict[str, Tensor]) -> Tensor:
        """Set up function for computing value loss"""

        states, rewards_to_go = data['states'], data['ret']
        return ((self.ac.v(states) - rewards_to_go)**2).mean()

    def update(self):
        data = self.buf.get()

        policy_loss_old, pi_info_old = self.compute_loss_policy(data)
        policy_loss_old = policy_loss_old.item()
        value_loss_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_policy_iters):
            self.policy_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_policy(data)
            kl = mpi.mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            mpi.mpi_avg_grads(self.ac.pi)    # average grads across MPI processes
            self.policy_optimizer.step()

        self.logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_value_iters):
            self.value_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi.mpi_avg_grads(self.ac.v)    # average grads across MPI processes
            self.value_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(
            LossPi=policy_loss_old, LossV=value_loss_old,
            KL=kl, Entropy=ent, ClipFrac=cf,
            DeltaLossPi=(loss_pi.item() - policy_loss_old),
            DeltaLossV=(loss_v.item() - value_loss_old)
        )