"""
  @Time : 2021/11/26 15:08 
  @Author : Ziqi Wang
  @File : ac_model.py 
"""

import torch
from torch import nn
from torch.distributions import Normal

# Refers to https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianMLPActor(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -32

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = mlp([obs_dim, *hidden_sizes], activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, SquashedGaussianMLPActor.LOG_STD_MIN, SquashedGaussianMLPActor.LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= torch.sum(torch.log(1 - torch.tanh(pi_action) ** 2 + 1e-6), dim=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU, m=1):
        super().__init__()
        # obs_dim, act_dim = obs_space.shape[0], act_space.shape[0]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [m], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return q # Critical to ensure q has right shape.
