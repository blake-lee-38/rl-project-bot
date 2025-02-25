import random
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from blackjackenv import BlackjackEnv

# Create and register custom environment
register(
    id="Blackjack-Custom-v1",  # Unique environment ID
    entry_point="blackjackenv:BlackjackEnv",  # Path to your env class
    kwargs={"natural": True, "sab": False},  # Default settings
    max_episode_steps=100,  # Optional, adjust if needed
)
env = gym.make("Blackjack-Custom-v1", natural=True)

# Initialize Q-table and Returns dictionary
Q = {}
Returns = {}
policy = {}
betting_Q = {}
betting_Returns = {}
betting_policy = {}

# Betting levels
bet_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Betting options based on bankroll

# Define Blackjack book strategy (assuming dealer always hits on 16)
def blackjack_book_strategy(player_sum, dealer_upcard, usable_ace):
    if player_sum >= 17:
        return 0  # Stand
    if player_sum <= 11:
        return 1  # Hit
    if usable_ace:
        if player_sum == 12 and dealer_upcard in [4, 5, 6]:
            return 0  # Stand
        return 1  # Hit
    if 12 <= player_sum <= 16:
        if dealer_upcard in [2, 3, 4, 5, 6]:
            return 0  # Stand
        return 1  # Hit
    return 1  # Default to hit

def get_count_bin(count):
    """ Convert count to a bin representation """
    if count < -9:
        return 'VLow'
    elif count < -4:
        return 'Low'
    elif count > 9:
        return 'VHigh'
    elif count > 4:
        return 'High'
    else:
        return 'Neutral'

def get_state(obs, count):
    """ Convert observation to a state representation """
    player_sum, dealer_upcard, usable_ace = obs
    count_bin = get_count_bin(count)
    return (player_sum, dealer_upcard, usable_ace, count_bin)

# Initialize state-action value table
def initialize_q():
    for player_sum in range(4, 22):  # Player sum (4-21)
        for dealer_card in range(1, 11):  # Dealer's upcard (1-10)
            for usable_ace in [True, False]:  # Usable ace or not
                for count in ['VLow', 'Low', 'Neutral', 'High', 'VHigh']:  # Count range (-30 to 30)
                    Q[(player_sum, dealer_card, usable_ace, count)] = np.random.uniform(-0.1, 0.1, size=2)  # Stand, Hit
                    Returns[(player_sum, dealer_card, usable_ace, count)] = {0: [], 1: []}
                    policy[(player_sum, dealer_card, usable_ace, count)] = blackjack_book_strategy(player_sum, dealer_card, usable_ace)  # Default to hit

                    # Initialize betting strategy Q-table
                    betting_Q[count] = [0] * len(bet_levels)
                    betting_Returns[count] = {b: [] for b in range(len(bet_levels))}
                    betting_policy[count] = 4

initialize_q()
def first_visit_mc(num_episodes=5000, gamma=1.0, epsilon=0.05):
    """ Monte Carlo First-Visit Control Algorithm for both playing and betting decisions """
    global policy, betting_policy
    count = 0
    for episode in range(num_episodes):
        epsilon = max(0.01, 1 - episode / (num_episodes / 2))

        # Select Bet Before Seeing Hand
        orig_count_bin = get_count_bin(count)
        count_bin = orig_count_bin
        # Select betting strategy using epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            bet_idx = random.choice(range(len(bet_levels)))
        else:
            bet_idx = betting_policy[orig_count_bin]
        bet = bet_levels[bet_idx]


        print(f"Episode {episode + 1}/{num_episodes}")
        episode_memory = []  # Store (state, action, reward) for this episode
        obs, info = env.reset()
        count = int(info['count'])  # Track the deck count
        
        
        done = False
        while not done:
            state = get_state(obs, count)
            
            # ϵ-Greedy Action Selection for in-hand decisions
            if state[0] < 12:
                action = 1
            elif state[0] >= 17:
                action = 0
            elif random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1])  # Stand or Hit
            else:
                action = policy[state]
            
            obs, reward, done, _, info = env.step(action)
            episode_memory.append((state, action, reward, bet_idx, count_bin))
            count = int(info['count'])
            count_bin = get_count_bin(count)
        
        
        # Compute Returns (First-Visit Only)
        G1 = 0
        G2 = 0
        visited_betting_states = set()
        for t in reversed(range(len(episode_memory))):  # Work backwards
            state, action, reward, bet_idx, count_bin = episode_memory[t]
            G1 = gamma * G1 + reward  # Apply betting factor to returns
            G2 = gamma * G2 + reward * bet
            # Every-visit check for in-hand policy
            Returns[state][action].append(G1)
            Q[state][action] = np.mean(Returns[state][action])
            policy[state] = np.argmax(Q[state])
        

        betting_Returns[orig_count_bin][bet_idx].append(G2)
        betting_Q[orig_count_bin][bet_idx] = np.mean(betting_Returns[orig_count_bin][bet_idx])
        betting_policy[orig_count_bin] = np.argmax(betting_Q[orig_count_bin])
        visited_betting_states.add(orig_count_bin)
    
    print("Training completed!")

num_deviations = 0
def play_blackjack(num_hands=100):
    global num_deviations
    total_money = 1000  # Initial bankroll
    wins = 0
    count = 0
    count_bin = get_count_bin(count)
    for _ in range(num_hands):
        bet_idx = betting_policy[count_bin]
        bet = bet_levels[bet_idx]

        obs, info = env.reset()
        count = info['count']
        count_bin = get_count_bin(count)
        
        # Select bet using trained betting policy
        
        
        done = False
        while not done:
            state = get_state(obs, count)
            if state[0] < 12:
                p1action = 1
            elif state[0] >= 17:
                p1action = 0
            else:
                p1action = policy[state]  # Use learned policy
            book_action = blackjack_book_strategy(state[0], state[1], state[2])
            if p1action != book_action:
                #print(f"Discrepancy Detected! State: {state}, Action: {p1action}, Book Action: {book_action}")
                num_deviations += 1

            obs, reward, done, _, info = env.step(p1action)
            if done:
                total_money += reward * bet
                if reward > 0:
                    wins += 1
            count = info['count']
    
    return (total_money, wins)

def play_blackjack2(num_hands=100):
    global num_deviations
    total_money = 1000  # Initial bankroll
    wins = 0
    for _ in range(num_hands):
        obs, info = env.reset()
        count = info['count']
        count_bin = get_count_bin(count)
        bet = bet_levels[betting_policy[count_bin]]
        
        done = False
        while not done:
            state = get_state(obs, count)  # Use learned policy
            book_action = blackjack_book_strategy(state[0], state[1], state[2])

            obs, reward, done, _, info = env.step(book_action)
            if done:
                total_money += reward * bet
                if reward > 0:
                    wins += 1
            count = info['count']
    
    return (total_money, wins)

first_visit_mc(100000)

profit = 0
loss = 0
win = 0
outperform = 0
for i in range(100):
    total, wins1 = play_blackjack(100)
    total2, wins2 = play_blackjack2(100)
    if total > total2:
        profit += 1
    else:
        loss += 1

    if wins1 > wins2:
        outperform += 1

    if total > 1000:
        win += 1


print("Number of deviations:", num_deviations)
print(f"Profit: {profit}, Loss: {loss}", "Win:", win, "Outperform:", outperform)

profit = 0
loss = 0
win = 0
outperform = 0
for i in range(100):
    total, wins1 = play_blackjack(100)
    total2, wins2 = play_blackjack2(100)
    if total > total2:
        profit += 1
    else:
        loss += 1

    if wins1 > wins2:
        outperform += 1

    if total > 1000:
        win += 1


print("Number of deviations:", num_deviations)
print(f"Profit: {profit}, Loss: {loss}", "Win:", win, "Outperform:", outperform)

profit = 0
loss = 0
win = 0
outperform = 0
for i in range(100):
    total, wins1 = play_blackjack(100)
    total2, wins2 = play_blackjack2(100)
    if total > total2:
        profit += 1
    else:
        loss += 1

    if wins1 > wins2:
        outperform += 1

    if total > 1000:
        win += 1


print("Number of deviations:", num_deviations)
print(f"Profit: {profit}, Loss: {loss}", "Win:", win, "Outperform:", outperform)

print(betting_policy)


# Close the environment
env.close()
