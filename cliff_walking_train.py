import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

ENV_NAME = "CliffWalking-v1"
LEARNING_RATE = 0.001
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 6
ENTROPY_COEF = 0.04
MAX_EPISODES = 10000
MAX_STEPS = 100


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


def state_to_tensor(state, num_states):
    return F.one_hot(torch.tensor(state), num_classes=num_states).float()


def train_ppo():
    env = gym.make(ENV_NAME)
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    model = ActorCritic(num_states, num_actions)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()

        log_probs = []
        values = []
        rewards = []
        masks = []
        states = []
        actions = []

        # Faze 1 - collect data
        for step in range(MAX_STEPS):
            state_tensor = state_to_tensor(state, num_states)
            states.append(state_tensor)

            action_probs, state_value = model(state_tensor)

            dist = Categorical(action_probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            done = terminated or truncated

            log_probs.append(dist.log_prob(action))
            values.append(state_value)
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            masks.append(torch.tensor([1 - int(done)], dtype=torch.float32))
            actions.append(action)

            state = next_state
            if done:
                break

        # Faze 2 calculate advanates
        next_state_tensor = state_to_tensor(state, num_states)
        _, next_value = model(next_state_tensor)

        returns = []
        R = next_value.detach()

        for i in reversed(range(len(rewards))):
            R = rewards[i] + GAMMA * R * masks[i]
            returns.insert(0, R)

        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        log_probs_old = torch.stack(log_probs).detach()
        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Faze 3 ppo weights update
        for _ in range(K_EPOCHS):

            action_probs_new, values_new = model(states_tensor)
            dist_new = Categorical(action_probs_new)

            log_probs_new = dist_new.log_prob(actions_tensor)
            entropy = dist_new.entropy()

            ratio = torch.exp(log_probs_new - log_probs_old)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(values_new.squeeze(-1), returns)
            entropy_loss = -entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % 50 == 0:
            print(
                f"Episode {episode:4d} | Steps: {step+1:3d} | Rewards sum: {sum(rewards).item():.1f} | Loss: {loss}"
            )

    torch.save(model.state_dict(), "ppo_cliffwalking.pth")
    print("\nModel saved\n")
    env.close()


if __name__ == "__main__":
    train_ppo()
