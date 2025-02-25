import random
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from blackjackenv import BlackjackEnv
import matplotlib.pyplot as plt


# Create and register custom environment
register(
    id="Blackjack-Custom-v2",  # Unique environment ID
    entry_point="blackjackenv:BlackjackEnv",  # Path to your env class
    kwargs={"natural": True, "sab": False},  # Default settings
    max_episode_steps=100,  # Optional, adjust if needed
)
env = gym.make("Blackjack-Custom-v2", natural=True)

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
                for count in ['VLow', 'Low', 'Neutral', 'High', 'VHigh']:  # Count range (5 Buckets)
                    Q[(player_sum, dealer_card, usable_ace, count)] = [0.0, 0.0]  # Stand, Hit
                    Returns[(player_sum, dealer_card, usable_ace, count)] = {0: [], 1: []}
                    policy[(player_sum, dealer_card, usable_ace, count)] = blackjack_book_strategy(player_sum, dealer_card, usable_ace)  # Default to hit

                    # Initialize betting strategy Q-table
                    betting_Q[count] = [0] * len(bet_levels)
                    betting_Returns[count] = {b: [] for b in range(len(bet_levels))}
                    betting_policy[count] = 4

initialize_q()
def first_visit_mc(num_episodes=5000, gamma=0.8, epsilon=0.01):
    """ Monte Carlo First-Visit Control Algorithm for both playing and betting decisions """
    global policy, betting_policy
    count = 0
    for episode in range(num_episodes):
        epsilon = max(0.01, 1 - episode / (num_episodes / 4))

        # Select Bet Before Seeing Hand
        orig_count_bin = get_count_bin(count)
        count_bin = orig_count_bin
        # Select betting strategy using epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            bet_idx = random.choice(range(len(bet_levels)))
        else:
            bet_idx = betting_policy[orig_count_bin]
        bet = bet_levels[bet_idx]


        #print(f"Episode {episode + 1}/{num_episodes}")
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
            G2 = gamma * G2 + reward * bet  # Apply betting factor to returns
            # First-visit check for in-hand policy
            if (state, action) not in episode_memory[:t]:
                Returns[state][action].append(G1)
                Q[state][action] = np.mean(Returns[state][action])
                policy[state] = np.argmax(Q[state])
        

        betting_Returns[orig_count_bin][bet_idx].append(G2)
        betting_Q[orig_count_bin][bet_idx] = np.mean(betting_Returns[orig_count_bin][bet_idx])
        betting_policy[orig_count_bin] = np.argmax(betting_Q[orig_count_bin])
        visited_betting_states.add(orig_count_bin)
    
    print("Training completed!")

num_deviations = 0
def play_blackjack(num_hands=100, action=True, betting=True):
    global num_deviations
    total_money = 1000  # Initial bankroll
    wins = 0
    count = 0
    count_bin = get_count_bin(count)
    for _ in range(num_hands):
        bet = 0
        if betting:
            bet_idx = betting_policy[count_bin]
            bet = bet_levels[bet_idx]
        else:
            bet = 50

        obs, info = env.reset()
        count = info['count']
        count_bin = get_count_bin(count)        
        
        done = False
        while not done:
            state = get_state(obs, count)
            if state[0] < 12:
                p1action = 1
            elif state[0] >= 17:
                p1action = 0
            else:
                if action:
                    p1action = policy[state]  # Use learned policy
                else:
                    p1action = blackjack_book_strategy(state[0], state[1], state[2])

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

first_visit_mc(50000)

for i in range(1):
    model_best = 0
    bet_best = 0
    book_best = 0
    model_profit = 0
    bet_profit = 0
    book_profit = 0
    win = 0
    outperform = 0
    mw = []
    mm = []
    bew = []
    bem = []
    bw = []
    bm = []
    for i in range(100):
        env.reset(options={"full_reset": True})
        model_total, model_wins = play_blackjack(1000, action=True, betting=True)
        env.reset(options={"full_reset": True})
        bet_total, bet_wins = play_blackjack(1000, action=False, betting=True)
        env.reset(options={"full_reset": True})
        book_total, book_wins = play_blackjack(1000, action=False, betting=False)
        
        if max(model_total, bet_total, book_total) == model_total:
            model_profit += 1
        elif max(model_total, bet_total, book_total) == bet_total:
            bet_profit += 1
        else:
            book_profit += 1
        

        if max(model_wins, bet_wins, book_wins) == model_wins:
            model_best += 1
        elif max(model_wins, bet_wins, book_wins) == bet_wins:
            bet_best += 1
        else:
            book_best += 1
        
        print("------Win Percentage------")
        print("Model:", model_wins / 100, "Betting:", bet_wins / 100, "Book:", book_wins / 100)
        mw.append(model_wins / 1000)
        bew.append(bet_wins / 1000)
        bw.append(book_wins / 1000)
        print("------Ending Balance------")
        print("Model:", model_total, "Betting:", bet_total, "Book:", book_total)
        mm.append(model_total)
        bem.append(bet_total)
        bm.append(book_total)
        print("------")

    print("------Win Percentage Over 100,000 Hands------")
    print("Model:", sum(mw) / 100, "Betting:", sum(bew) / 100, "Book:", sum(bw) / 100)
    print("------Average Balance Over 100,000 Hands------")
    print("Model:", sum(mm) / 10, "Betting:", sum(bem) / 10, "Book:", sum(bm) / 10)

'''
    print("Number of deviations:", num_deviations)
    print("------Total Money Performance------")
    print("Model:", model_profit, "Betting:", bet_profit, "Book:", book_profit)
    print("Overall Best: ", max(model_profit, bet_profit, book_profit))
    print("------Wins Performance------")
    print("Model:", model_best, "Betting:", bet_best, "Book:", book_best)
    print("Overall Best: ", max(model_best, bet_best, book_best))
    print("------")
'''
print(betting_policy)


# Close the environment
env.close()


plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), mw, marker="o", linestyle="-", label="Book Strategy")
# plt.plot(range(1, 11), bew, marker="s", linestyle="-", label="Betting Strategy")
plt.plot(range(1, 101), bw, marker="^", linestyle="-", label="Model Strategy")
plt.xlabel("Episode Number")
plt.ylabel("Win Percentage")
plt.title("Win Rate Trends Over Episodes")
plt.legend()
plt.grid()
plt.show()