import gymnasium as gym
from gymnasium.envs.registration import register
from blackjackenv import BlackjackEnv  # Import your modified Blackjack environment


# Create and register custom environment
register(
    id="Blackjack-Custom-v1",  # Unique environment ID
    entry_point="blackjackenv:BlackjackEnv",  # Path to your env class
    kwargs={"natural": True, "sab": False},  # Default settings
    max_episode_steps=100,  # Optional, adjust if needed
)
env = gym.make("Blackjack-Custom-v1", natural=True, render_mode="human")


# Deal number of hands to play, initial bet, and total balance
hands = 100
total = 100
bet = 10

# Deal first hand
print("-------")
print("New Hand")
print("--------")
obs, info = env.reset()
print(f"Initial Hand: {obs}, Info: {info}")


# Play Hands
while hands:
    # Simple strategy: Adjusted by RL based on count, cards, and dealer's card
    action = 1 if  obs[0] < 16 else 0

    # Take action and observe new state and reward
    obs, reward, done, truncated, info = env.step(action)

    # Update total balance
    total += reward * bet

    # Print results from action
    print(f"Action: {action}, New Obs: {obs}, Reward: {reward}, Total Balance: {total}, Info: {info}")
    
    # Check if hand is over or more actions can be taken
    if done or truncated:
        hands -= 1

        # Playing around with adjusted betting strategy (Also RL based)
        if total < 100:
            bet *= 2

        # Deal New Hand
        print("-------")
        print("New Hand")
        print("--------")
        obs, info = env.reset()
        print(f"Initial Hand: {obs}, Info: {info}")

# Close the environment
env.close()