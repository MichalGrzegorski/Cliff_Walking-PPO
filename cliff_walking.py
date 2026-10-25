import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

ENV_NAME = "CliffWalking-v1"
LEARNING_RATE = 0.002
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
ENTROPY_COEF = 0.05
MAX_EPISODES = 1000
MAX_STEPS = 200


class ActorCritic(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(num_states, 64)

        self.actor = nn.Linear(64, num_actions)

        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value


def state_to_tensor(state, num_states=48):
    return F.one_hot(torch.tensor(state), num_classes=num_states).float()


def train_ppo():
    pass


if __name__ == "__main__":
    train_ppo()
