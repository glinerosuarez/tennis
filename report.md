[//]: # (Image References)

[image1]: output/2021-06-18_tennis/2021-06-18_19-51-28-tennis_s100/scores.png "Plot of rewards"

# Continuous control Report

## Learning Algorithm
In this project I used two independent ActorCritic agents, each agent is an implementation of the 
[*PPO Actor-Critic style*](https://arxiv.org/pdf/1707.06347.pdf) algorithm with fixed-length trajectory segments. 
Each iteration, each of N (parallel) actors collect T timesteps of data. The surrogate loss is constructed on these NT 
timesteps of data, and optimized with Adam optimizer, for K epochs. The pseudocode is described below:

###Algorithm:
- `for iteration= 1,2,3,... do`
    - `for actor=1,2,3,..., N do`
        - `Run policy pi_old in environment for T timesteps`
        - `Compute advantage estimates A1, ..., AT`
- `end for`
- `Optimize surrogate L wrt theta, with K epochs and minibatch size M < NT`
- `theta_old <- theta`

### Hyperparameters:

- `gamma = 0.99`; discount factor. (Always between 0 and 1.)
- `lam = 0.97`; lambda for [GAE-Lambda](https://arxiv.org/pdf/1506.02438.pdf). (Always between 0 and 1, close to 1.)
- `steps_per_epoch = 4000`; number of steps of interaction (state-action pairs) for the agents and the environment in
  each epoch.
- `clip_ratio = 0.2`; for clipping in the policy objective. 
- `target_kl = 0.01`; KL divergence we think is appropriate between new and old policies after an update. 
  Used for early stopping.
- `epochs = 2000`; number of epochs of interaction (equivalent to number of policy updates) to perform. The algorithm 
  could stop before if it reaches an average score of `env_solved_at`over the last `epochs_mean_rewards` epochs. For the 
  solution, the environment was solved in 379 epochs.
- `max_ep_len = 1000`; maximum length of trajectory / episode / rollout.
- `epochs_mean_rewards = 100`; Number of epochs to compute mean rewards.
- `env_solved_at = 0.5`; the environment is considered solved if the agent gets this score averaged over 
  `epochs_mean_rewards` epochs.
- `train_policy_iters = 80`; maximum number of gradient descent steps to take on policy loss per epoch.
  (Early stopping may cause optimizer to take fewer than this.)
- `train_value_iters = 80`; maximum number of gradient descent steps to take on value function loss per epoch.
  (Early stopping may cause optimizer to take fewer than this.)

### Actor-Critic Agent Neural Network Architecture

There are two agents with identical structures, each with two networks, the policy network:
- Layer 1: feed forward with 24 inputs(state size) and 64 nodes with ReLU activations.
- Layer 2: feed forward with 64 inputs and 64 nodes with ReLU activations.
- Layer 3: feed forward with 64 inputs and 2(action size) nodes with tanh activations.
  The learning rate is set to `0.0003`

  and the value functions network:
- Layer 1: feed forward with 24 inputs(state size) and 64 nodes with ReLU activations.
- Layer 2: feed forward with 64 inputs and 64 nodes with ReLU activations.
- Layer 3: feed forward with 64 inputs and 1 nodes without activation.
  The learning rate is set to `0.0003`

### Plot of Rewards
This agent was able to solve the environment with the selected hyper parameter at 379 episodes:
![Plot of rewards][image1]

### Ideas for Future Work

1. Fine-tune hyper parameters, to get better performance with less training.
2. Train with more parallel environments.
3. Try other algorithms such as DDPG or TRPO.
4. Implement self-play.


  
  

