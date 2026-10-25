import time

import gymnasium as gym
import torch

from cliff_walking_train import ENV_NAME, MAX_STEPS, ActorCritic, state_to_tensor


def test_model():
    env = gym.make(ENV_NAME, render_mode="human")

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    model = ActorCritic(num_states, num_actions)
    model.load_state_dict(torch.load("ppo_cliffwalking.pth"))
    model.eval()

    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print("Eval start...")

    while not done:
        steps += 1
        state_tensor = state_to_tensor(state, num_states)

        with torch.no_grad():
            action_probs, _ = model(state_tensor)

        action = torch.argmax(action_probs).item()

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated or steps == MAX_STEPS
        total_reward += reward

        print(f"Step: {steps}")
        time.sleep(0.1)

    print(f"End of test. Steps: {steps} | Score (Rewards sum): {total_reward}")

    env.close()


if __name__ == "__main__":
    test_model()
