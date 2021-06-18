from typing import Tuple

from unityagents import BrainParameters, UnityEnvironment, BrainInfo


def init_env(
        env_file: str,
        train_mode: bool,
        worker_id: int,
        seed: int,
) -> Tuple[UnityEnvironment, str, int, int, int, Tuple[float]]:
    """initialize UnityEnvironment"""

    env: UnityEnvironment = UnityEnvironment(file_name=env_file, worker_id=worker_id, seed=seed)

    # Environments contain brains which are responsible for deciding the actions of their associated agents.
    # Here we check for the first brain available, and set it as the default brain we will be controlling from Python.
    brain_name: str = env.brain_names[0]
    brain: BrainParameters = env.brains[brain_name]

    action_size: int = brain.vector_action_space_size   # number of actions

    env_info: BrainInfo = env.reset(train_mode=train_mode)[brain_name]

    num_agents: int = len(env_info.agents)              # number of agents interacting with the environment

    states = env_info.vector_observations               # initial state
    state_size: int = states.shape[1]                   # get the current stat

    return env, brain_name, num_agents, state_size, action_size, states
