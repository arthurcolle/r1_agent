import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import Dict, List, Tuple

class AdvancedDQNAgent:
    """
    A more advanced RL-based approach that uses a Deep Q-Network (DQN) with a
    multi-layer perceptron to approximate the Q-function. This allows better
    generalization to unseen states and more sophisticated decision-making for
    offloading or resource management tasks.

    In this demo code, we illustrate:
    - Neural network to process state
    - Epsilon-greedy policy
    - Experience replay approach
    - Target network for stable training
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        replay_buffer_size: int = 10000,
        batch_size: int = 32
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Build primary network and target network
        self.model = self._build_network()
        self.target_model = self._build_network()
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer for experience replay
        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size

    def _build_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_dim)
        )

    def act(self, state: List[float]) -> int:
        """
        Epsilon-greedy action selection.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.model(s)
                return q_values.argmax(dim=1).item()

    def store_experience(
        self,
        state: List[float],
        action: int,
        reward: float,
        next_state: List[float],
        done: bool
    ):
        """
        Add new experience tuple to the replay buffer.
        """
        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update_network(self) -> None:
        """
        Sample a batch from the replay buffer and train the DQN.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Current Q values
        q_values = self.model(states_t).gather(1, actions_t)

        # Next Q values (from target network)
        with torch.no_grad():
            next_q_values = self.target_model(next_states_t).max(dim=1, keepdim=True)[0]

        # Calculate target Q
        target_q = rewards_t + (1 - dones_t) * self.gamma * next_q_values

        # Compute loss
        loss = self.loss_fn(q_values, target_q)

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self) -> None:
        """
        Periodically sync parameters from the main network to the target network.
        """
        self.target_model.load_state_dict(self.model.state_dict())
