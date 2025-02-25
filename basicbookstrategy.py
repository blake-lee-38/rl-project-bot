import numpy as np
import random
import gymnasium as gym
from blackjackenv import BlackjackEnv

# Register and create the environment
gym.register(id="Blackjack-Custom-v2", entry_point="blackjackenv:BlackjackEnv")
env = gym.make("Blackjack-Custom-v2")

# Initialize Q-table and Returns dictionary
Q = {}
Returns = {}
policy = {}

# Possible states
for player_sum in range(4, 22):  # Player sum (4-21)
    for dealer_card in range(1, 11):  # Dealer's upcard (1-10)
        for usable_ace in [True, False]:  # Usable ace or not
            Q[(player_sum, dealer_card, usable_ace)] = [0.0, 0.0]  # Q-values for [Stand, Hit]
            Returns[(player_sum, dealer_card, usable_ace)] = {0: [], 1: []}
            policy[(player_sum, dealer_card, usable_ace)] = random.choice([0, 1])  # Initialize randomly

def first_visit_mc(num_episodes=100000, gamma=0.5, epsilon=0.1):
    """ First-Visit Monte Carlo Control for Blackjack """
    for episode in range(num_episodes):
        episode_memory = []  # Store (state, action, reward) for this episode
        obs, _ = env.reset()
        done = False

        while not done:
            state = obs  # (player_sum, dealer_card, usable_ace)
            
            # ϵ-Greedy Action Selection
            if random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1])  # Random action
            else:
                action = policy[state]  # Greedy action

            obs, reward, done, _, _ = env.step(action)
            episode_memory.append((state, action, reward))

        # Compute Returns (First-Visit)
        G = 0
        visited_states = set()

        for t in reversed(range(len(episode_memory))):  # Work backwards
            state, action, reward = episode_memory[t]
            G = gamma * G + reward  # Update return
            
            # First-visit check
            if (state, action) not in episode_memory[:t]:
                Returns[state][action].append(G)
                Q[state][action] = np.mean(Returns[state][action])  # Update Q-value
                policy[state] = np.argmax(Q[state])  # Update policy to best action
                visited_states.add((state, action))

        # Decay epsilon over time to favor exploitation
        epsilon = max(0.01, 1 - episode / (num_episodes / 2))  # Slower decay


    print("Training complete!")

# Train the model
first_visit_mc(500000)

# Evaluate learned policy
def evaluate_policy(num_hands=1000):
    wins, losses, draws = 0, 0, 0
    for _ in range(num_hands):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy[obs]  # Use learned policy
            obs, reward, done, _, _ = env.step(action)
        
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    print(f"Results: Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Win Rate: {wins / num_hands:.2%}")

evaluate_policy(10000)

def book_policy(obs):
    """ Basic strategy for Blackjack """
    player_sum, dealer_card, usable_ace = obs
    if usable_ace:
        if player_sum >= 19:
            return 0
        if player_sum == 18 and dealer_card not in [9, 10, 1]:
            return 0
        else:
            return 1
    else:
        if player_sum >= 17:
            return 0
        elif player_sum >= 13 and dealer_card in [2, 3, 4, 5, 6]:
            return 0
        elif player_sum == 12 and dealer_card in [4, 5, 6]:
            return 0
        else:
            return 1

num_deviations = 0
# Test Learned Strategy against Book
for key, value in policy.items():
    if policy[key] != book_policy(key) and not (key[2] and key[0] < 12):
        print(f"Discrepancy Detected! State: {key}, Learned Action: {policy[key]}, Book Action: {book_policy(key)}")
        num_deviations += 1
        
print(f"Total Discrepancies: {num_deviations}")
        

# Close the environment
env.close()
