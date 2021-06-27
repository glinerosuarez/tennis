import random
import numpy as np
from typing import Dict

from pathlib import Path

import torch
from unityagents import BrainInfo

from agents import MultiActorCritic
from ppo import PPO
from tools import mpi, log
from config import settings
from tools.env import init_env, init_tennis_env
from argparse import ArgumentParser


def play_tennis_randomly() -> None:
    """
    Play in the Tennis environment with two agents that choose actions at random
    """
    
    # Init environment.
    env, brain_name, num_agents, state_size, action_size, _ = init_env(
        settings.env_file, False, random.randint(0, 1000), random.randint(0, 1000)
    )

    for _ in range(5):
        scores = np.zeros(num_agents)                               # initialize the score
        while True:
            action = np.random.randn(num_agents, action_size)       # select an action
            action = np.clip(action, -1, 1)                         # all actions between -1 and 1
            env_info = env.step(action)[brain_name]                 # send the action to the environment
            rewards = env_info.rewards                              # get the reward for each agent
            done = env_info.local_done                              # see if episode has finished
            scores += rewards                                       # update the score
            if np.any(done):                                        # exit loop if episode finished for any of the agents
                break

        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

    env.close()


def train(exp_name: str) -> None:
    """
    Implement PPO algorithm to train two ActorCritic agents to play in the tennis environment.
    :param exp_name: Name of the experiment.
    """

    # Run parallel code with MPI
    mpi.mpi_fork(settings.cores)

    # Get logging kwargs
    logger_kwargs: Dict[str, str] = log.setup_logger_kwargs(exp_name, settings.seed, settings.out_dir, True)

    ppo: PPO = PPO(
        env_fn=init_tennis_env,
        seed=settings.seed,
        steps_per_epoch=settings.PPO.steps_per_epoch,
        epochs=settings.PPO.epochs,
        gamma=settings.PPO.gamma,
        clip_ratio=settings.PPO.clip_ratio,
        policy_lr=settings.ActorCritic.policy_lr,
        value_lr=settings.ActorCritic.value_lr,
        train_policy_iters=settings.ActorCritic.train_policy_iters,
        train_value_iters=settings.ActorCritic.train_value_iters,
        lam=settings.PPO.lam,
        max_ep_len=settings.PPO.max_ep_len,
        target_kl=settings.PPO.target_kl,
        logger_kwargs=logger_kwargs,
        save_freq=settings.save_freq
    )

    ppo.train()


def play_tennis(path: str, epochs: int) -> None:
    """
    Take a two trained agents to run an episode of the Tennis environment.
    :param epochs: epochs checkpoint to select model.
    :param path: path to the directory that contains the saved model.
    """
    model_path = Path()/path/'pyt_save'/f'model{epochs}.pt'  # Path to saved model file
    state_dicts = torch.load(model_path)                     # Load Python dict with state_dicts for ActorCritic agent

    policy_state_dict1 = state_dicts['policy_state_dict1']
    value_state_dict1 = state_dicts['value_state_dict1']
    policy_state_dict2 = state_dicts['policy_state_dict2']
    value_state_dict2 = state_dicts['value_state_dict2']

    # Init environment
    seed = random.randint(0, 1000)
    env, brain_name, num_agents, state_size, action_size, states = init_env(
        settings.env_file, False, random.randint(0, 1000), seed
    )

    # Init agent
    agents = MultiActorCritic(state_size, action_size, seed, settings.ActorCritic.hidden_nodes)

    # Update state dicts
    agents.agent1.pi.load_state_dict(policy_state_dict1)
    agents.agent1.v.load_state_dict(value_state_dict1)
    agents.agent2.pi.load_state_dict(policy_state_dict2)
    agents.agent2.v.load_state_dict(value_state_dict2)

    # Run environment
    agents.agent1.pi.eval()
    agents.agent1.v.eval()
    agents.agent2.pi.eval()
    agents.agent2.v.eval()

    score = 0.0
    while True:
        # Get actions for each agent
        a1 = agents.agent1.act(torch.as_tensor(states[0], dtype=torch.float32))
        a2 = agents.agent2.act(torch.as_tensor(states[1], dtype=torch.float32))
        actions = np.stack([a1, a2])

        # Perform chosen actions in the env and get next states
        brain_info: BrainInfo = env.step(actions)[brain_name]
        next_states = brain_info.vector_observations

        # Get rewards
        r1 = brain_info.rewards[0]
        r2 = brain_info.rewards[1]

        d = any(brain_info.local_done)

        states = next_states
        score += max([r1, r2])

        if d:
            break

    env.close()

    print("Score: {}".format(score))


def main():
    # Get arguments
    parser = ArgumentParser()
    parser.add_argument("-n", "--exp_name", type=str, default="tennis")
    parser.add_argument("-e", "--epochs", type=str, default="")
    parser.add_argument(
        "-t",
        "--train",
        help="Use PPO algorithm to train an ActorCritic agent that solves the Reacher environment",
        action="store_true"
    )
    parser.add_argument(
        "-p",
        "--pretrained",
        help="Receives a path to the folder that contains a trained agent to run an epoch in the Reacher environment",
        type=str
    )
    parser.add_argument(
        "-r",
        "--random",
        help="Use a agent that chooses actions at random to run an epoch in the Reacher environment",
        action="store_true"
    )
    args = parser.parse_args()

    if args.train:
        train(args.exp_name)
    elif args.random:
        play_tennis_randomly()
    else:
        play_tennis(args.pretrained, args.epochs)


if __name__ == "__main__":
    main()
