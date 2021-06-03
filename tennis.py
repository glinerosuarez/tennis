import random
import numpy as np
from argparse import ArgumentParser

from config import settings
from tools.env import init_env


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


def main():
    # Get arguments
    parser = ArgumentParser()
    parser.add_argument("-n", "--exp_name", type=str, default="reach-ppo")
    parser.add_argument(
        "-t",
        "--train",
        help="Use PPO algorithm to train an ActorCritic agent that solves the Reacher environment",
        action="store_true"
    )
    parser.add_argument(
        "-c",
        "--continuous_control",
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
        #train(args.exp_name)
        pass
    elif args.random:
        play_tennis_randomly()
    else:
        #smart_cc(args.continuous_control)
        pass


if __name__ == "__main__":
    main()
