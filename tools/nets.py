import numpy as np
from enum import Enum
from collections import OrderedDict
from torch.nn import Module, Sequential, Linear, Tanh, ReLU


class Activation(Enum):

    @classmethod
    def get_values(cls):
        return [a.value for a in cls.__members__.values()]

    TANH = "tanh"
    RELU = "relu"


def build_actor_model(state_size: int, hidden_nodes: int, o_dim: int) -> Sequential:
    """Build an Actor MLP.
    Params
    ======
        state_size:     Size of the states
        hidden_nodes:   Number of nodes per hidden layer
        o_dim:          output dimension
    """
    return Sequential(OrderedDict([
        ("Layer1", Linear(state_size, hidden_nodes)),
        ("Activation1", ReLU()),
        ("Layer2", Linear(hidden_nodes, hidden_nodes)),
        ("Activation2", ReLU()),
        ("Layer3", Linear(hidden_nodes, o_dim)),
        ("Activation3", Tanh()),
    ]))


def build_critic_model(state_size: int, hidden_nodes: int) -> Sequential:
    """Build a Critic MLP.
    Params
    ======
        state_size:     Size of the states
        hidden_nodes:   Number of nodes per hidden layer
    """
    return Sequential(OrderedDict([
        ("Layer1", Linear(state_size, hidden_nodes)),
        ("Activation1", ReLU()),
        ("Layer2", Linear(hidden_nodes, hidden_nodes)),
        ("Activation2", ReLU()),
        ("Layer3", Linear(hidden_nodes, 1)),
    ]))


def build_model(state_size: int, n_layers: int, hidden_nodes: int, activation: Activation, o_dim: int) -> Sequential:
    """Build a MLP.
    Params
    ======
        state_size:     Size of the states
        n_layers:       Number of hidden layers
        hidden_nodes:   Number of nodes per hidden layer
        activation:     Type of activations
        o_dim:          output dimension
    """
    layers: OrderedDict[str, Module] = OrderedDict()

    for i in range(n_layers):
        if i == 0:
            layers["Linear1"] = Linear(state_size, hidden_nodes)
            layers["Activation1"] = Tanh() if activation == Activation.TANH else ReLU()
        elif i == (n_layers - 1):
            layers[f"Linear{n_layers}"] = Linear(hidden_nodes, o_dim)
            layers[f"Activation{n_layers}"] = Tanh() if activation == Activation.TANH else ReLU()
        else:
            layers[f"Linear{i + 1}"] = Linear(hidden_nodes, hidden_nodes)
            layers[f"Activation{i + 1}"] = Tanh() if activation == Activation.TANH else ReLU()

    return Sequential(layers)


def count_vars(module: Module) -> int:
    return sum([np.prod(p.shape) for p in module.parameters()])
