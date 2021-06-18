import torch
import numpy as np
from torch import Tensor
from typing import Optional, Tuple
from torch.distributions.normal import Normal
from tools.nets import Activation, build_model
from torch.nn import Parameter, Sequential, Module


class Actor(Module):
    def __init__(self, state_size: int, action_size: int, layers: int, hidden_nodes: int, activation: Activation):
        super(Actor, self).__init__()

        self.log_std: Parameter = Parameter(-0.5 * torch.ones(action_size, dtype=torch.float32))
        self.mu_net: Sequential = build_model(
            state_size=state_size,
            n_layers=layers,
            hidden_nodes=hidden_nodes,
            activation=activation,
            o_dim=action_size
        )

    def _distribution(self, states: Tensor) -> Normal:
        mu: Tensor = self.mu_net(states)
        std: Tensor = torch.exp(self.log_std)
        return Normal(mu, std)

    def log_prob_from_dist(self, prob_dist: Normal, act):
        return prob_dist.log_prob(act).sum(axis=-1)

    def forward(self, states: Tensor, act: Optional[Tensor] = None) -> Tuple[Normal, Optional[Tensor]]:
        """
        Accept a batch of observations and optionally a batch of actions. Produce action distributions for given
        observations, and optionally compute the log likelihood of given actions under and those distributions.
        return:
        ===========  ================  ===========================================================================
        Symbol       Shape             Description
        ===========  ================  ===========================================================================
        ``pi``       N/A               | Torch Distribution object, containing a batch of distributions describing
                                       | the policy for the provided observations.
        ``logp_a``   (batch,)          | Optional (only returned if batch of actions is given). Tensor containing
                                       | the log probability, according to the policy, of the provided actions.
                                       | If actions not given, will contain ``None``.
        ===========  ================  ===========================================================================
        """

        prob_dist = self._distribution(states)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob_from_dist(prob_dist, act)
        return prob_dist, logp_a


class Critic(Module):
    def __init__(self, state_size: int, layers: int, hidden_nodes: int, activation: Activation):
        super(Critic, self).__init__()

        self.v_net: Sequential = build_model(
            state_size=state_size,
            n_layers=layers,
            hidden_nodes=hidden_nodes,
            activation=activation,
            o_dim=1
        )

    def forward(self, obs: Tensor) -> Tensor:
        """
        Accepts a batch of observations and return:
        ===========  ================  ======================================
        Symbol       Shape             Description
        ===========  ================  ======================================
        ``v``        (batch,)          | Tensor containing the value estimates for the provided observations.
                                       | (Critical: make sure to flatten this!)
        ===========  ================  ======================================
        """
        return torch.squeeze(self.v_net(obs), -1)


class ActorCritic(Module):
    """ A PyTorch Module with a ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` module. """

    def __init__(
            self, state_size: int, action_size: int, seed: int, layers: int, hidden_nodes: int, activation: Activation
    ):
        """Initialize parameters and build model.
        Params
        ======
            state_size: Dimension of each state
            action_size: Dimension of each action
            seed: Random seed
        """

        super(ActorCritic, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.pi = Actor(state_size, action_size, layers, hidden_nodes, activation)
        self.v = Critic(state_size, layers, hidden_nodes, activation)

    def step(self, obs: Tensor) -> Tuple[np.array, np.array, np.array]:
        """
        The ``step`` method accepts a batch of observations and return:
        ===========  ====================      ===================================
        Symbol       Shape                     Description
        ===========  ====================      ===================================
        ``a``        (batch, action_size)      | Numpy array of actions for each observation.
        ``v``        (batch,)                  | Numpy array of value estimates for the provided observations.
        ``logp_a``   (batch,)                  | Numpy array of log probs for the actions in ``a``.
        ===========  ====================      ===================================
        """

        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi.log_prob_from_dist(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        """ Behaves the same as ``step`` but only returns ``a``. """
        return self.step(obs)[0]


class MultiActorCritic:
    def __init__(
            self, state_size: int, action_size: int, seed: int, layers: int, hidden_nodes: int, activation: Activation
    ):
        """Initialize parameters and build model for two ActorCritic Agents.
        Params
        ======
            state_size: Dimension of each state
            action_size: Dimension of each action
            seed: Random seed
        """

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.agent1 = ActorCritic(
            state_size=state_size,
            action_size=action_size,
            seed=seed,
            layers=layers,
            hidden_nodes=hidden_nodes,
            activation=activation
        )

        self.agent2 = ActorCritic(
            state_size=state_size,
            action_size=action_size,
            seed=seed,
            layers=layers,
            hidden_nodes=hidden_nodes,
            activation=activation
        )

