import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import Dict, List, Tuple

class BaseRLAgent:
    """
    Base class for RL agents that wrap an LLM or any other policy model.
    Provides interfaces for training steps, policy actions, and updating neural networks.
    """

    def act(self, state) -> int:
        """
        Decide on an action given the current state.
        """
        raise NotImplementedError

    def update_network(self, transitions) -> None:
        """
        Perform a training update for the agent.
        """
        raise NotImplementedError

    def set_eval_mode(self):
        """
        Switch the agent to evaluation mode.
        """
        raise NotImplementedError

    def set_train_mode(self):
        """
        Switch the agent to training mode.
        """
        raise NotImplementedError

###############################################################################
# SIMPLE POLICY OPTIMIZATION
###############################################################################

class SimplePolicyAgent(BaseRLAgent):
    """
    Implements a very naive policy gradient approach where we collect rollouts
    and perform a gradient ascent step on the log-probabilities weighted by returns.
    This is for demonstration and not recommended for production.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128, lr: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def act(self, state):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy_network(s).squeeze(0)
        action_dist = torch.distributions.Categorical(probs=probs)
        action = action_dist.sample()
        return action.item()

    def update_network(self, transitions):
        """
        transitions: list of (state, action, reward, next_state, done)
        We'll do a simple REINFORCE: sum of log-probs * discounted returns
        """
        self.set_train_mode()
        returns = 0
        loss = 0
        gamma = 0.99

        for (state, action, reward, _, _) in reversed(transitions):
            returns = reward + gamma * returns
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = self.policy_network(s).squeeze(0)
            action_dist = torch.distributions.Categorical(probs=probs)
            log_prob = action_dist.log_prob(torch.tensor(action))
            loss = loss - log_prob * returns

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def set_eval_mode(self):
        self.policy_network.eval()

    def set_train_mode(self):
        self.policy_network.train()

###############################################################################
# PROXIMAL POLICY OPTIMIZATION (PPO)
###############################################################################

class PPOAgent(BaseRLAgent):
    """
    A simplified PPO implementation, excluding many advanced features like GAE-lambda or advantage normalization.
    Demonstrates clipped objective for stable improvements.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        lr: float = 1e-3,
        clip_epsilon: float = 0.2
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_epsilon = clip_epsilon

        # Policy & old policy
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.old_policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.old_policy_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def act(self, state):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy_network(s).squeeze(0)
        action_dist = torch.distributions.Categorical(probs=probs)
        action = action_dist.sample()
        return action.item()

    def update_network(self, transitions):
        """
        transitions: list of (state, action, reward, next_state, done)
        We'll do a simplified advantage (reward - baseline=0).
        Then apply PPO clip objective.
        """
        self.set_train_mode()

        # Copy current net to old net
        self.old_policy_network.load_state_dict(self.policy_network.state_dict())

        all_states = []
        all_actions = []
        all_returns = []

        returns = 0
        gamma = 0.99

        for (state, action, reward, _, _) in reversed(transitions):
            returns = reward + gamma * returns
            all_states.insert(0, state)
            all_actions.insert(0, action)
            all_returns.insert(0, returns)

        states_t = torch.tensor(all_states, dtype=torch.float32)
        actions_t = torch.tensor(all_actions, dtype=torch.long)
        returns_t = torch.tensor(all_returns, dtype=torch.float32)

        # Current policy
        current_probs = self.policy_network(states_t)
        current_dist = torch.distributions.Categorical(probs=current_probs)
        current_log_probs = current_dist.log_prob(actions_t)

        # Old policy
        old_probs = self.old_policy_network(states_t)
        old_dist = torch.distributions.Categorical(probs=old_probs)
        old_log_probs = old_dist.log_prob(actions_t).detach()

        ratio = torch.exp(current_log_probs - old_log_probs)
        # Advantage ~ returns_t (no baseline for simplicity)
        advantage = returns_t

        # Clipped objective
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        loss_1 = ratio * advantage
        loss_2 = clipped_ratio * advantage
        loss = -torch.min(loss_1, loss_2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def set_eval_mode(self):
        self.policy_network.eval()

    def set_train_mode(self):
        self.policy_network.train()

###############################################################################
# TRUST REGION POLICY OPTIMIZATION (TRPO)
###############################################################################

class TRPOAgent(BaseRLAgent):
    """
    A placeholder TRPO agent. Real TRPO uses conjugate gradient to solve a constrained
    optimization problem. This is just a stub illustrating how we'd structure it.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        lr: float = 1e-3,
        max_kl: float = 0.01
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.max_kl = max_kl

    def act(self, state):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy_network(s).squeeze(0)
        action_dist = torch.distributions.Categorical(probs=probs)
        action = action_dist.sample()
        return action.item()

    def update_network(self, transitions):
        """
        With actual TRPO, we would do:
          1) Compute advantage
          2) Compute policy gradient with conjugate gradient, ensuring KL divergence < max_kl
          3) Update policy parameters
        This function is a placeholder to illustrate structure only.
        """
        self.set_train_mode()
        # Collect states, actions, rewards
        returns = 0
        gamma = 0.99
        all_states = []
        all_actions = []
        all_returns = []
        for (state, action, reward, _, _) in reversed(transitions):
            returns = reward + gamma * returns
            all_states.insert(0, state)
            all_actions.insert(0, action)
            all_returns.insert(0, returns)
        # Pseudocode for TRPO:
        #  advantage = all_returns - baseline
        #  grad = compute_policy_gradient(self.policy_network, states, actions, advantage)
        #  stepdir = conjugate_gradient(grad, fisher_vector_product, self.max_kl)
        #  line_search_update(self.policy_network, stepdir)
        pass

    def set_eval_mode(self):
        self.policy_network.eval()

    def set_train_mode(self):
        self.policy_network.train()
