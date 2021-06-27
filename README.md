[//]: # (Image References)

[image1]: tennis-agents.gif "Trained Agents"
[image2]: random-tennis.gif "Random Agent"

### Introduction

In this project, we will train an agent to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#reacher)
environment implementing the [*PPO Actor-Critic style*](https://arxiv.org/pdf/1707.06347.pdf) algorithm with two agents. 
The code is based on [OpenAI's spinning repo](https://github.com/openai/spinningup).

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it 
receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward 
of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each 
agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or 
away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 
consecutive episodes, after taking the maximum over both agents). Specifically, after each episode, we add up the 
rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially 
different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode. The 
environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Getting Started

1. Download the environment from one of the links below. In this repo I am using the Mac OSX environment, you only need
   to select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

   (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)
   if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

   (_For AWS_) If you'd like to train the agent on AWS (and have not enabled a [virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), 
   then please use this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to 
   obtain the "headless" version of the environment. You will **not** be able to watch the agent without enabling a virtual 
   screen, but you will be able to train the agent. (_To watch the agent, you should follow the instructions to [enable a 
   virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), 
   and then download the environment for the **Linux** operating system above._)

2. Set `env_file` in the `settings.toml` file to the path where your environment file is stored, I'm using the
   `environment` dir.

3. Install the dependencies:
    - cd to the navigation directory.
    - activate your virtualenv.
    - run: `pip install -r requirements.txt` in your shell.

### Instructions

There are three options currently available to run `tennis.py`:

1. Run `python tennis.py -r` to run 5 episodes of the tennis environment with two agents that select actions randomly.

   ![Random Agent][image2]


2. Run `python tennis.py -t` to train two agents to solve the Tennis environment while collecting experience
   in it.
   Hyperparameters are specified in `settings.toml`, feel free to tune them to see if you can get better results! Also,
   you can change `save_freq` which controls how often the agent weights are stored in the output dir.


3. Run `python tennis.py -p` and pass the path to the dir where the agents' weights are stored and the number of epochs
   they were trained like this: `python tennis.py -p output/2021-05-30_tennis/2021-05-30_09-13-25-tennis_s24 --epochs
   200` to use those pretrained agents to explore the Tennis environment.