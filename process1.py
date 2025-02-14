import gymnasium as gym
from gymnasium.envs.registration import register
from blackjackenv import BlackjackEnv  # Import your modified Blackjack environment


# Create the environment
register(
    id="Blackjack-Custom-v1",  # Unique environment ID
    entry_point="blackjackenv:BlackjackEnv",  # Path to your env class
    kwargs={"natural": True, "sab": False},  # Default settings
    max_episode_steps=100,  # Optional, adjust if needed
)

env = gym.make("Blackjack-Custom-v1", natural=True, render_mode="human")

print("-------")
print("New Hand")
print("--------")
obs, info = env.reset()
print(f"Initial Hand: {obs}, Info: {info}")
hands = 0

while hands < 10:
    action = 1 if  obs[0] < 16 else 0  # Simple strategy: hit if sum < 16, else stick
    obs, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, New Obs: {obs}, Reward: {reward}, Info: {info}")
    if done or truncated:
        hands += 1
        print("-------")
        print("New Hand")
        print("--------")
        obs, info = env.reset()
        print(f"Initial Hand: {obs}, Info: {info}")

env.close()