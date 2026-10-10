import time

import gymnasium as gym

# Oryginalny renderer Gymnasium
env = gym.make("FrozenLake-v1", render_mode="human")


state, info = env.reset()

try:
    for step in range(10):

        # Losowy ruch agenta
        action = env.action_space.sample()

        # Wykonaj ruch i wyświetl animację
        state, reward, terminated, truncated, info = env.step(action)

        # Zakończenie epizodu i powrót do startu
        if terminated or truncated:
            time.sleep(1)
            state, info = env.reset()

finally:
    env.close()
