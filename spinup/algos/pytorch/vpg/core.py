import numpy as np
import scipy.signal
try:
    from gymnasium.spaces import Box, Discrete
except ImportError:
    from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        # isinstance() is a Python function that checks if an object is of a certain class.
        # For example, isinstance(5, int) is True.
        # Box is the Gym class for continuous action spaces (e.g., joint torques, steering angles).
        if isinstance(action_space, Box):
            # Example: HalfCheetah-v2 The agent needs to control the torques of 6 different joints. 
            # The action space is a vector of 6 floating-point numbers.action_space.shape would be (6,).
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            # Example: CartPole-v1 The agent can only take one of two actions: "push cart to the left" (action 0) or "push cart to the right" (action 1).
            # action_space.n would be 2.
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            # pi is now a torch.distributions.Normal object. 
            # This object represents 6 independent Gaussian (normal) distributions. 
            # For example, the first distribution has a mean of 0.1 and a std of 0.5, the second has a mean of -0.2 and a std of 0.5, and so on.
            pi = self.pi._distribution(obs)
            # It draws one random sample from each of the 6 Gaussian distributions.
            a = pi.sample()
            # calculates the log-probability of having sampled that specific action a from the distribution pi.(How likely this action is taken under this policy distribution?)
            # The .sum(axis=-1) then adds these 6 log-probabilities together to get a single scalar value representing the total log-probability of the entire action vector a.
            # The result, logp_a, is a single-element tensor, for example, tensor(-4.71).
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            # The result, v, is a single-element tensor, for example, tensor(25.5).
            # The Critic estimates the value v, before the agent takes action a. 
            # Seems redundant, but its entire purpose is to help the "Actor" (the policy) make better decisions.
            v = self.v(obs)
            # a.numpy(): array([0.32, -0.51, 1.1, -0.25, -0.1, 0.6])
            # v.numpy(): array(25.5)
            # logp_a.numpy(): array(-4.71)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]