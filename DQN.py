"""
DQN architecture 
- double DQN
- noisy net
- experience replay buffer
- Dueling networks
- 2 FC (128, 128)
- 2 Noisy layers (32, 32)

- input : states    (43)
- output : Q-values (36)
- actions are selected implicitly for having max Q-values
- action selecting dependent to remaining inventory
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Define the Rainbow DQN architecture

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

class DuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):    # input : 43, output : 36
        super(DuelingDQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions

        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
            
        )

        self.advantage = nn.Sequential(
            NoisyLinear(128, 32),
            nn.ReLU(),
            NoisyLinear(32, num_actions)
        )

        self.value = nn.Sequential(
            NoisyLinear(128, 32),
            nn.ReLU(),
            NoisyLinear(32, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        batch_transposed = zip(*batch)
        samples = [np.array(elem) for elem in batch_transposed]
        return samples

    def __len__(self):
        return len(self.buffer)


# Create the Double DQN agent

# state_size : 43, action_size : 36, batch_size : 16, gamma : 0.9, replay_capacity : 600 * 360, epsilon : 0.3, epsilon_decay : 0.8

class DoubleDQNAgent:
    def __init__(self, state_size, action_size, batch_size, gamma, replay_capacity, epsilon, epsilon_decay):    
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Create two Dueling DQN networks: one for the target network and one for the policy network
        self.policy_net = DuelingDQN(state_size, action_size)
        self.target_net = DuelingDQN(state_size, action_size)
        self.update_target_net()

        # Define the loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_net.parameters())

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    """
    state = [
            self.elapsed_time,
            self.inventory,
            self._flatten_orderbook(),
            self.bid_ask_spread,
        ]
    """

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            if state[1] == 0:       # inventory == 0
                return 0
            else:
                state = torch.from_numpy(state).float().unsqueeze(0)
                q_values = self.policy_net(state)
                action = q_values.argmax().item()
        return action

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        # Compute Q-values for the current state using the policy network
        q_values = self.policy_net(state).gather(1, action)

        # Compute Q-values for the next state using the target network
        next_q_values = self.target_net(next_state).detach().max(1)[0].unsqueeze(1)

        # Compute the target Q-values
        target_q_values = reward + self.gamma * next_q_values * (1 - done)

        # Compute the loss
        loss = self.criterion(q_values, target_q_values)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.update_target_net()

        # Decay epsilon
        self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
