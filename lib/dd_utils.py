# based on the book Deep Reinforcement Learning Hands On, Second Edition by Maxim Lapan

import ptan
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()
        # nothing complicated here - 2 hidden layers
        # Tanh activation at the end to squash it to -1, 1
        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()
        # the obs_net "pre-processes" the observations
        # before they are mixed with actions and processed further into the single Q value
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU()
        )

        # the "pre-processed" observations are mixed with actions and processed into a single Q value
        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, a):
        # 2 steps - pre-process and then mix and generate Q
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

class D4PGCritic(nn.Module):
    def __init__(self, obs_size, act_size, n_atoms, v_min, v_max):
        super(D4PGCritic, self).__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU()
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            # main difference with DDPG - instead of a single Q value, it's now returning N_ATOMS
            # referred to the paper as C51
            nn.Linear(300, n_atoms)
        )

        delta = (v_max - v_min) / (n_atoms - 1)
        # this is a torch buffer with reward supports
        # it will be used to get from the probability to single mean Q-value
        self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))

    def forward(self, x, a):
        obs = self.obs_net(x)
        # combine the actions with pre-processed observations
        return self.out_net(torch.cat([obs, a], dim=1))

    def distr_to_q(self, distr):
        # this converts the distribution to the Q value
        # we use softmax to get probabilities from the logits
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)

# we need a special "stateful" type of agent to be able to calcuate the OU
class AgentDDPG(ptan.agent.BaseAgent):
    # this params are defaulted to values from the Continuous Control with Deep Reinforcement Learning paper
    def __init__(self, net, device, ou_enabled=True, ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2,ou_espilon=1.0, clip_actions=False):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_espilon
        self.clip_actions = clip_actions

    # this is needed by the base class but we will init the state only in __call__
    def initial_state(self):
        return None

    # the core method - converts observations (+ internal state of agent) into actions
    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        # add exploration noise by applying the OU process
        if self.ou_enabled:
            new_a_states = []
            # iterate over agent states and observations and update the OU process value
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)

                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)

        else:
            new_a_states = agent_states

        if self.clip_actions:
            # I believe this is only for py-bullet
            actions = np.clip(actions, -1, 1)

        return actions, new_a_states

# the agent is much simpler because it doesn't have any state
class AgentD4PG(ptan.agent.BaseAgent):
    def __init__(self, net, device, epsilon=0.3, clip_actions=False):
        self.net = net
        self.device = device
        self.epsilon = epsilon
        self.clip_actions = clip_actions

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        # add Gaussian noise scaled by the epsilon value
        actions += self.epsilon * np.random.normal(size=actions.shape)

        if self.clip_actions:
            actions = np.clip(actions, -1, 1)

        return actions, agent_states


def unpack_batch(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)

    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = ptan.agent.float32_preprocessor(actions).to(device)
    rewards_v = ptan.agent.float32_preprocessor(rewards).to(device)
    last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.tensor(dones, dtype=torch.bool, device=device)

    return states_v, actions_v, rewards_v, dones_t, last_states_v