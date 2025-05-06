#!/usr/bin/env python3
"""
train_td_agent.py

This script trains a TD(λ) agent (using eligibility traces with a SARSA(λ)-style update and card counting) using our AdvancedBlackjackEnv.
  - After training stops, the final trained agent is evaluated over 1,000 new episodes (each with 100 rounds)
    using its own learned playing policy (for both betting and playing phases).
  - The evaluation reports the average money won (or lost) per episode as well as the percentage of episodes
    that were profitable. Only the final evaluation summary is printed.
  - Eligibility traces are used for updating Q-values with the update rule:
      δ = r_shaped + γ * Q(s', a') − Q(s, a)
      e(s, a) ← e(s, a) + 1
      Q(s, a) ← Q(s, a) + α · δ · e(s, a)   and   e(s, a) ← γλ · e(s, a)
"""

import random
import numpy as np
import threading
import time
import tkinter as tk
from collections import defaultdict
import os
from datetime import datetime
from PIL import ImageGrab  # Requires Pillow
import argparse

from advanced_blackjack_env import AdvancedBlackjackEnv

# --- Evaluation Function (Money Metrics for Agent Only) ---
def evaluate_money(num_episodes=1000):
    """
    Runs evaluation for num_episodes (each with 100 rounds) and computes the total reward per episode.
    For phase 0 the learned betting policy is used and for phase 1 the agent's learned playing policy is used.
    Prints only the final summary results:
      - Average reward per episode.
      - Percentage of episodes that were profitable.
      
    Returns a tuple: (avg_reward, profitable_count, non_profitable_count)
    """
    env = AdvancedBlackjackEnv(render_mode=None, natural=True, initial_bankroll=1000, max_rounds=args.max_rounds)
    total_rewards = []
    profitable_count = 0
    for i in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            phase = obs["phase"]
            if phase == 0:
                state = discretize_state(obs)
                action = choose_action(state, phase, EPSILON_BET, i, obs)
            else:
                state = discretize_state(obs)
                action = choose_action(state, phase, EPSILON_PLAY, i, obs)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
        if episode_reward > 0:
            profitable_count += 1
    avg_reward = sum(total_rewards) / num_episodes
    print(f"\nEvaluation (Agent Policy) over {num_episodes} episodes:")
    print(f"Average Reward per Episode: {avg_reward:.2f}")
    print(f"Profitable Episodes: {profitable_count} out of {num_episodes} ({profitable_count/num_episodes*100:.1f}%)")
    return avg_reward, profitable_count, num_episodes - profitable_count

# --- Global UI Elements and Flags ---
header_label = None
left_stats_label = None
right_stats_label = None
canvas_chart = None
canvas_epsilon = None
canvas_action_table = None
canvas_betting_table = None
varied_bets_enabled = False
varied_bets_button = None
auto_enable_label = None
epsilon_label = None

# Global variable to hold evaluation metrics (for the agent)
evaluation_metrics = None

def create_ui(args):
    global header_label, left_stats_label, right_stats_label, epsilon_label
    global canvas_chart, canvas_epsilon, canvas_action_table, canvas_betting_table
    global varied_bets_button, auto_enable_label
    top_frame = tk.Frame(stats_frame)
    top_frame.grid(row=0, column=0, sticky="ew", pady=(0,10))
    top_frame.grid_columnconfigure(0, weight=1)
    top_frame.grid_columnconfigure(1, weight=1)
    episode_frame = tk.Frame(top_frame)
    episode_frame.grid(row=0, column=0, sticky="w")
    header_label = tk.Label(episode_frame, text="Episode: 0", font=("Helvetica",12,"bold"), justify=tk.LEFT)
    header_label.pack(anchor="w")
    if args.episodes:
        max_ep_label = tk.Label(episode_frame, text=f"Max Episodes: {args.episodes:,}", font=("Helvetica",10), justify=tk.LEFT)
        max_ep_label.pack(anchor="w")
    control_frame = tk.Frame(top_frame)
    control_frame.grid(row=0, column=1, sticky="e")
    def toggle_pause():
        if pause_event.is_set():
            pause_event.clear()
            pause_button.config(text="Pause")
        else:
            pause_event.set()
            pause_button.config(text="Start")
    pause_button = tk.Button(control_frame, text="Pause", command=toggle_pause)
    pause_button.pack(side=tk.LEFT, padx=5)
    def stop_training():
        stop_event.set()
        save_screenshot_and_summary()
        print("Running money evaluation (1,000 episodes)...")
        global evaluation_metrics
        evaluation_metrics = evaluate_money(1000)
        root.quit()
    stop_button = tk.Button(control_frame, text="Stop", command=stop_training)
    stop_button.pack(side=tk.LEFT, padx=5)
    epsilon_frame = tk.Frame(top_frame)
    epsilon_frame.grid(row=1, column=0, sticky="w", pady=(5,0))
    epsilon_label = tk.Label(epsilon_frame, text="Play ε: 1.000  |  Bet ε: 0.000", font=("Helvetica",10))
    epsilon_label.pack(anchor="w")
    spacer_frame = tk.Frame(top_frame)
    spacer_frame.grid(row=1, column=1)
    stats_labels_frame = tk.Frame(stats_frame)
    stats_labels_frame.grid(row=1, column=0, sticky="w", pady=(0,10))
    left_stats_label = tk.Label(stats_labels_frame, text="", font=("Helvetica",10), justify=tk.LEFT)
    left_stats_label.grid(row=0, column=0, padx=20)
    right_stats_label = tk.Label(stats_labels_frame, text="", font=("Helvetica",10), justify=tk.LEFT)
    right_stats_label.grid(row=0, column=1, padx=20)
    canvas_chart = tk.Canvas(charts_frame, width=760, height=300, bg="white")
    canvas_chart.pack(pady=10)
    canvas_epsilon = tk.Canvas(charts_frame, width=760, height=300, bg="white")
    canvas_epsilon.pack(pady=10)
    canvas_action_table = tk.Canvas(right_frame, width=760, height=400, bg="white")
    canvas_action_table.pack(pady=10)
    betting_frame = tk.Frame(right_frame)
    betting_frame.pack(pady=10, padx=10)
    varied_bets_button = tk.Button(betting_frame, text="Enable Varied Betting", command=enable_varied_bets, width=30, height=2)
    varied_bets_button.pack(pady=(20,5))
    auto_enable_label = tk.Label(betting_frame, text=f"Will auto-enable at episode {args.varied_bets_at:,}", font=("Helvetica",9))
    auto_enable_label.pack(pady=(0,20))
    canvas_betting_table = tk.Canvas(betting_frame, width=760, height=400, bg="white")
    canvas_betting_table.pack(pady=10)

# ----- State Discretization and Action Selection Helpers -----
def discretize_state(obs: dict) -> tuple:
    if obs["phase"] == 0:
        bankroll_disc = int(obs["bankroll"] // 10)
        count = obs.get("card_count", 0)
        count_bin = int(np.clip(count, -5, 5))
        return ("bet", bankroll_disc, count_bin)
    else:
        player_sum = obs["hand_sum"]
        dealer_str = obs["dealer_card"]
        if dealer_str is not None:
            rank = dealer_str[1]
            if rank in ['T','J','Q','K']:
                dealer_val = 10
            elif rank == 'A':
                dealer_val = 1
            else:
                dealer_val = int(rank)
        else:
            dealer_val = 0
        usable = int(obs["usable_ace"])
        count = obs.get("card_count", 0)
        count_bin = int(np.clip(count, -10, 10))
        return (player_sum, dealer_val, usable, count_bin)

def possible_actions(phase: int, obs: dict = None):
    if phase == 0:
        return list(range(10))
    else:
        actions = [0,1]
        if obs is not None:
            if obs.get("can_double", False):
                actions.append(2)
            if obs.get("can_split", False):
                actions.append(3)
        return actions

ALPHA = 0.1
GAMMA = 0.95
EPSILON_PLAY = 1.0
EPSILON_BET = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99975

Q_play = defaultdict(lambda: np.zeros(4))
Q_bet = defaultdict(lambda: np.zeros(10))

def get_q(state: tuple, phase: int):
    if phase == 0:
        if state not in Q_bet:
            Q_bet[state] = np.zeros(10)
        return Q_bet[state]
    else:
        if state not in Q_play:
            Q_play[state] = np.zeros(4)
        return Q_play[state]

def choose_action(state: tuple, phase: int, epsilon: float, episode: int, obs: dict):
    global args
    if phase == 0:
        if episode < args.varied_bets_at and not varied_bets_enabled:
            return 4
        else:
            if random.random() < epsilon:
                count = state[2]
                if count > 2:
                    return random.randint(5,9)
                elif count < -2:
                    return random.randint(0,4)
                else:
                    return random.randint(0,9)
            else:
                q_vals = get_q(state, phase)
                noise = np.random.normal(0,0.1,size=len(q_vals))
                return int(np.argmax(q_vals+noise))
    else:
        allowed = possible_actions(phase, obs)
        if random.random() < epsilon:
            return random.choice(allowed)
        else:
            q_vals = get_q(state, phase)
            q_vals_allowed = [q_vals[a] if a in allowed else -float('inf') for a in range(4)]
            return int(np.argmax(q_vals_allowed))

def get_print_interval(episode):
    if episode < 2500:
        return 100
    elif episode < 10000:
        return 500
    else:
        return 1000

pause_event = threading.Event()
stop_event = threading.Event()

script_dir = os.path.dirname(os.path.abspath(__file__))
results_folder = os.path.join(script_dir, "results")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

root = tk.Tk()
root.title("TD(λ) Training Progress")

window_width = 1650
window_height = 900

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width - window_width) // 2
y = (screen_height - int(window_height*1.1)) // 2

root.geometry(f"{window_width}x{window_height}+{x}+{y}")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

left_frame = tk.Frame(main_frame, width=800)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

right_frame = tk.Frame(main_frame, width=800)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

stats_frame = tk.Frame(left_frame)
stats_frame.pack(fill=tk.X, padx=10, pady=10)

charts_frame = tk.Frame(left_frame)
charts_frame.pack(fill=tk.BOTH, expand=True)

ui_data = {
    "episode": 0,
    "ep_reward": 0.0,
    "total_avg": 0.0,
    "recent_avg": 0.0,
    "epsilon_play": EPSILON_PLAY,
    "epsilon_bet": EPSILON_BET,
    "best_reward": -float('inf'),
    "wins": 0,
    "losses": 0,
    "ties": 0,
    "episodes": [],
    "total_avgs": [],
    "recent_avgs": [],
    "epsilons_play": [],
    "epsilons_bet": [],
    "best_rewards": [],
    "varied_bets_episode": None,
    "last_recent_wins": 0,
    "last_recent_losses": 0,
    "last_recent_win_ratio": 0
}
recent_rewards = []

def update_text():
    global epsilon_label
    header_text = (f"Episode: {ui_data['episode']}")
    epsilon_text = (f"Play ε: {ui_data['epsilon_play']:.3f}  |  Bet ε: {ui_data['epsilon_bet']:.3f}")
    left_text = (f"Total Moving Avg: {ui_data['total_avg']:.2f}\n"
                 f"Recent Avg: {ui_data['recent_avg']:.2f}\n"
                 f"Best Reward: {ui_data['best_reward']:.2f}")
    lifetime_win_ratio = ((ui_data["wins"]/(ui_data["wins"]+ui_data["losses"]))*100) if (ui_data["wins"]+ui_data["losses"])>0 else 0
    right_text = (f"Lifetime Wins: {ui_data['wins']}  Losses: {ui_data['losses']}\n"
                  f"Lifetime Win Ratio: {lifetime_win_ratio:.1f}%\n"
                  f"Recent Wins: {ui_data['last_recent_wins']}  Losses: {ui_data['last_recent_losses']}\n"
                  f"Recent Win Ratio: {ui_data['last_recent_win_ratio']:.0f}%")
    
    header_label.config(text=header_text)
    epsilon_label.config(text=epsilon_text)
    left_stats_label.config(text=left_text)
    right_stats_label.config(text=right_text)

def draw_grid(canvas, x_min, x_max, y_min, y_max, margin=40, num_grid=10):
    canvas.delete("grid")
    w = int(canvas["width"])
    h = int(canvas["height"])
    x_range = x_max - x_min
    y_range = y_max - y_min
    for i in range(num_grid+1):
        if x_range:
            data_x = x_min + i*x_range/num_grid
            x = margin + (data_x-x_min)/x_range*(w-2*margin)
        else:
            data_x = x_min; x = margin
        canvas.create_line(x, margin, x, h-margin, fill="lightgray", tags="grid")
        canvas.create_text(x, h-margin+15, text=str(round(data_x)), fill="gray", font=("Helvetica",8), tags="grid")
    for i in range(num_grid+1):
        if y_range:
            data_y = y_min + i*y_range/num_grid
            y = h-margin - ((data_y-y_min)/y_range)*(h-2*margin)
        else:
            data_y = y_min; y = h-margin
        color = "black" if abs(data_y)<1e-6 else "lightgray"
        canvas.create_line(margin,y, w-margin,y, fill=color, tags="grid")
        canvas.create_text(margin-15, y, text=str(round(data_y,1)), fill="gray", font=("Helvetica",8), tags="grid")

def draw_chart(canvas, episodes, series1, series2, label1, label2):
    canvas.delete("all")
    w = int(canvas["width"])
    h = int(canvas["height"])
    margin = 40
    if not episodes: return
    points1 = []
    points2 = []
    for i in range(len(episodes)-1):
        ep_start = episodes[i]
        ep_end = episodes[i+1]
        val1_start = series1[i]
        val1_end = series1[i+1]
        val2_start = series2[i]
        val2_end = series2[i+1]
        for j in range(10):
            t = j/10
            ep = ep_start + (ep_end-ep_start)*t
            val1 = val1_start + (val1_end-val1_start)*t
            val2 = val2_start + (val2_end-val2_start)*t
            points1.append((ep,val1))
            points2.append((ep,val2))
    if episodes:
        points1.append((episodes[-1], series1[-1]))
        points2.append((episodes[-1], series2[-1]))
    x_min = episodes[0] if episodes else 0
    x_max = episodes[-1] if episodes else 1
    y_min_data = min(min(series1), min(series2))
    y_max_data = max(max(series1), max(series2))
    y_range = max(100, max(abs(y_min_data), abs(y_max_data)))
    y_min = -y_range; y_max = y_range
    draw_grid(canvas, x_min, x_max, y_min, y_max, margin=margin, num_grid=10)
    def scale_x(x):
        return margin + (x-x_min)/(x_max-x_min)*(w-2*margin) if x_max!=x_min else margin
    def scale_y(y):
        return h-margin - ((y-y_min)/(y_max-y_min))*(h-2*margin) if y_max!=y_min else h/2
    for i in range(1, len(points1)):
        x1 = scale_x(points1[i-1][0]); y1 = scale_y(points1[i-1][1])
        x2 = scale_x(points1[i][0]); y2 = scale_y(points1[i][1])
        canvas.create_line(x1,y1,x2,y2, fill="blue", width=2)
    for i in range(1, len(points2)):
        x1 = scale_x(points2[i-1][0]); y1 = scale_y(points2[i-1][1])
        x2 = scale_x(points2[i][0]); y2 = scale_y(points2[i][1])
        canvas.create_line(x1,y1,x2,y2, fill="orange", width=2)
    canvas.create_text(margin+20, margin-10, text=label1, fill="blue", font=("Helvetica",10))
    canvas.create_text(margin+150, margin-10, text=label2, fill="orange", font=("Helvetica",10))
    if ui_data["varied_bets_episode"] is not None:
        x = scale_x(ui_data["varied_bets_episode"])
        canvas.create_line(x, margin, x, h-margin, dash=(5,5), fill="red", width=1)
        canvas.create_text(x+5, margin+15, text="Varied Bets", fill="red", font=("Helvetica",8), anchor="w")
    canvas.create_text(w/2, 20, text="Training Progress Over Time", font=("Helvetica",12,"bold"))

def draw_epsilon_chart(canvas, episodes, epsilons_play, epsilons_bet):
    canvas.delete("all")
    w = int(canvas["width"])
    h = int(canvas["height"])
    margin = 40
    if not episodes: return
    title_y = 20
    canvas.create_text(w/2, title_y, text="Exploration Rates (ε) Decay", font=("Helvetica",12,"bold"))
    eps_text = f"Play ε: {epsilons_play[-1]:.3f}  |  Bet ε: {epsilons_bet[-1]:.3f}"
    canvas.create_text(w/2, title_y+20, text=eps_text, font=("Helvetica",10))
    points_play = []
    points_bet = []
    for i in range(len(episodes)-1):
        ep_start = episodes[i]; ep_end = episodes[i+1]
        eps_play_start = epsilons_play[i]; eps_play_end = epsilons_play[i+1]
        eps_bet_start = epsilons_bet[i]; eps_bet_end = epsilons_bet[i+1]
        for j in range(10):
            t = j/10
            ep = ep_start+(ep_end-ep_start)*t
            eps_play = eps_play_start+(eps_play_end-eps_play_start)*t
            eps_bet = eps_bet_start+(eps_bet_end-eps_bet_start)*t
            points_play.append((ep,eps_play))
            points_bet.append((ep,eps_bet))
    points_play.append((episodes[-1], epsilons_play[-1]))
    points_bet.append((episodes[-1], epsilons_bet[-1]))
    x_min = episodes[0]
    x_max = episodes[-1]
    y_min = 0; y_max = 1
    draw_grid(canvas, x_min, x_max, y_min, y_max, margin=margin, num_grid=10)
    def scale_x(x):
        return margin+(x-x_min)/(x_max-x_min)*(w-2*margin) if x_max!=x_min else margin
    def scale_y(y):
        return h-margin-((y-y_min)/(y_max-y_min))*(h-2*margin) if y_max!=y_min else h/2
    for i in range(1, len(points_play)):
        x1 = scale_x(points_play[i-1][0]); y1 = scale_y(points_play[i-1][1])
        x2 = scale_x(points_play[i][0]); y2 = scale_y(points_play[i][1])
        canvas.create_line(x1,y1,x2,y2, fill="green", width=2)
    for i in range(1, len(points_bet)):
        x1 = scale_x(points_bet[i-1][0]); y1 = scale_y(points_bet[i-1][1])
        x2 = scale_x(points_bet[i][0]); y2 = scale_y(points_bet[i][1])
        canvas.create_line(x1,y1,x2,y2, fill="blue", width=2)
    legend_x = w-margin-10
    legend_y = margin+10
    canvas.create_line(legend_x-40, legend_y, legend_x-20, legend_y, fill="green", width=2)
    canvas.create_text(legend_x-10, legend_y, text="Play ε", fill="green", font=("Helvetica",10), anchor="w")
    canvas.create_line(legend_x-40, legend_y+20, legend_x-20, legend_y+20, fill="blue", width=2)
    canvas.create_text(legend_x-10, legend_y+20, text="Bet ε", fill="blue", font=("Helvetica",10), anchor="w")
    canvas.create_text(w/2, 20, text="Exploration Rates (ε) Decay", font=("Helvetica",12,"bold"))

def get_color_from_probability(prob):
    g = b = int(255*(1-prob))
    g = max(0, min(255,g))
    b = max(0, min(255,b))
    return f"#{255:02x}{g:02x}{b:02x}"

def draw_action_probability_table(canvas, aggregated_action_data):
    canvas.delete("all")
    w = int(canvas["width"])
    h = int(canvas["height"])
    margin_left = 60
    margin_top = 120
    margin_right = 20
    margin_bottom = 40
    canvas.create_text(w/2, margin_top-60, text="Hand Sum vs Action Probability (Recent Episodes)", font=("Helvetica",12,"bold"))
    sorted_hand_sums = list(range(4,22))
    num_cols = max(1, len(sorted_hand_sums))
    num_rows = 4
    cell_width = (w-margin_left-margin_right)/num_cols
    cell_height = (h-margin_top-margin_bottom)/num_rows
    action_labels = ["Hit", "Stand", "x2", "Split"]
    for row, label in enumerate(action_labels):
        y = margin_top+row*cell_height+cell_height/2
        canvas.create_text(margin_left-10, y, text=label, font=("Helvetica",10), anchor="e")
    for col, hand_sum in enumerate(sorted_hand_sums):
        counts = aggregated_action_data.get(hand_sum, [0,0,0,0])
        total = sum(counts)
        for row in range(num_rows):
            prob = counts[row]/total if total>0 else 0
            color = get_color_from_probability(prob)
            x0 = margin_left+col*cell_width
            y0 = margin_top+row*cell_height
            x1 = x0+cell_width
            y1 = y0+cell_height
            canvas.create_rectangle(x0,y0,x1,y1, fill=color, outline="black")
            canvas.create_text((x0+x1)/2, (y0+y1)/2, text=f"{round(prob*100)}%", font=("Helvetica",8))
    for col, hand_sum in enumerate(sorted_hand_sums):
        x = margin_left+col*cell_width+cell_width/2
        canvas.create_text(x, margin_top-15, text=str(hand_sum), font=("Helvetica",10))
    
def draw_betting_distribution_table(canvas, recent_betting_data):
    canvas.delete("all")
    w = int(canvas["width"])
    h = int(canvas["height"])
    margin_left = 60
    margin_top = 120
    margin_right = 20
    margin_bottom = 40
    canvas.create_text(w/2, margin_top-60, text="Card Count vs Bet Size Distribution (Recent Episodes)", font=("Helvetica",12,"bold"))
    count_range = range(-5,6)
    bet_sizes = range(1,11)
    num_rows = len(count_range)
    num_cols = len(bet_sizes)
    cell_width = (w-margin_left-margin_right)/num_cols
    cell_height = (h-margin_top-margin_bottom)/num_rows
    for row, count in enumerate(count_range):
        for col, bet_size in enumerate(bet_sizes):
            action = bet_size-1
            prob = recent_betting_data.get((count,action),0)
            color = get_color_from_probability(prob)
            x0 = margin_left+col*cell_width
            y0 = margin_top+row*cell_height
            x1 = x0+cell_width
            y1 = y0+cell_height
            canvas.create_rectangle(x0,y0,x1,y1, fill=color, outline="black")
            canvas.create_text((x0+x1)/2, (y0+y1)/2, text=f"{round(prob*100)}%", font=("Helvetica",8))
    for col, bet_size in enumerate(bet_sizes):
        x = margin_left+col*cell_width+cell_width/2
        canvas.create_text(x, margin_top-15, text=str(bet_size), font=("Helvetica",10))
    for row, count in enumerate(count_range):
        y = margin_top+row*cell_height+cell_height/2
        canvas.create_text(margin_left-20, y, text=str(count), font=("Helvetica",10))

def update_ui_elements(aggregated_action_data=None, betting_distribution=None, force_update=False):
    update_text()
    draw_chart(canvas_chart, ui_data["episodes"], ui_data["total_avgs"], ui_data["recent_avgs"], "Total Moving Avg", "Recent Avg")
    draw_epsilon_chart(canvas_epsilon, ui_data["episodes"], ui_data["epsilons_play"], ui_data["epsilons_bet"])
    if aggregated_action_data is not None:
        draw_action_probability_table(canvas_action_table, aggregated_action_data)
    if varied_bets_enabled and betting_distribution is not None:
        draw_betting_distribution_table(canvas_betting_table, betting_distribution)
    if force_update:
        root.update_idletasks()
        root.update()
        time.sleep(0.5)

def shape_reward(reward, phase, state=None, action=None):
    if phase == 0:
        if reward > 0:
            return reward*2.0
        elif reward < 0:
            return reward*1.5
        return reward
    else:
        shaped_r = reward
        if state is not None and action is not None:
            player_sum = state[0]
            if action == 0:
                if player_sum >= 18:
                    shaped_r -= 1.0
            elif action == 1:
                if player_sum <= 11:
                    shaped_r -= 1.0
        if shaped_r > 0:
            return shaped_r*2.0
        elif shaped_r < 0:
            return shaped_r*0.5
        return shaped_r

def parse_args():
    parser = argparse.ArgumentParser(description='Train Q-Learning Agent for Blackjack')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor (default: 0.95)')
    parser.add_argument('--epsilon-decay', type=float, default=0.99975, help='Epsilon decay rate (default: 0.99975)')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes to train (default: None, runs indefinitely)')
    parser.add_argument('--no-ui', action='store_true', help='Disable UI and run in console mode')
    parser.add_argument('--quiet', action='store_true', help='Disable console output')
    parser.add_argument('--max-rounds', type=int, default=100, help='Number of rounds per episode (default: 100)')
    parser.add_argument('--varied-bets-at', type=int, default=25000, help='Episode number to enable varied bets (default: 25000)')
    # TD(λ) parameter:
    parser.add_argument('--lambda', type=float, dest='lmbda', default=0.9, help='Eligibility trace decay rate (default: 0.9)')
    return parser.parse_args()

def train_agent(args):
    global EPSILON_PLAY, EPSILON_BET, varied_bets_enabled, evaluation_metrics
    global action_stats_per_episode, betting_stats_per_episode
    env = AdvancedBlackjackEnv(render_mode=None, natural=True, initial_bankroll=1000, max_rounds=args.max_rounds)
    best_reward = -float('inf')
    total_moving_avg = 0.0
    global recent_rewards
    recent_rewards.clear()
    episode_counter = 0
    action_stats_per_episode = []
    betting_stats_per_episode = []
    
    EPSILON_PLAY = 1.0
    EPSILON_BET = 0.0
    varied_bets_enabled = False
    
    # TD(λ) training loop with eligibility traces:
    while not stop_event.is_set():
        if args.episodes and episode_counter >= args.episodes:
            stop_event.set()
            save_screenshot_and_summary()
            break
        while pause_event.is_set() and not stop_event.is_set():
            time.sleep(0.1)
        obs, _ = env.reset()
        E_bet = defaultdict(lambda: np.zeros(10))
        E_play = defaultdict(lambda: np.zeros(4))
        total_reward = 0
        done = False
        episode_action_stats = {}
        episode_betting_stats = []
        while not done:
            phase = obs["phase"]
            state = discretize_state(obs)
            epsilon = EPSILON_BET if phase==0 else EPSILON_PLAY
            action = choose_action(state, phase, epsilon, episode_counter, obs)
            next_obs, reward, done, _, info = env.step(action)
            if phase == 1:
                hand_sum = state[0]
                if hand_sum not in episode_action_stats:
                    episode_action_stats[hand_sum] = [0,0,0,0]
                episode_action_stats[hand_sum][action] += 1
            elif phase == 0:
                count = state[2]
                episode_betting_stats.append((count, action))
            total_reward += reward
            shaped_r = shape_reward(reward, phase, state, action)
            if not done and next_obs["phase"] == phase:
                next_state = discretize_state(next_obs)
                next_action = choose_action(next_state, phase, epsilon, episode_counter, next_obs)
                td_target = shaped_r + GAMMA * get_q(next_state, phase)[next_action]
            else:
                td_target = shaped_r
            delta = td_target - get_q(state, phase)[action]
            if phase == 0:
                E_bet[state][action] += 1
                for s in list(E_bet.keys()):
                    Q_bet[s] += ALPHA * delta * E_bet[s]
                    E_bet[s] *= GAMMA * args.lmbda
            else:
                E_play[state][action] += 1
                for s in list(E_play.keys()):
                    Q_play[s] += ALPHA * delta * E_play[s]
                    E_play[s] *= GAMMA * args.lmbda
            obs = next_obs
        episode_counter += 1
        recent_rewards.append(total_reward)
        EPSILON_PLAY = max(EPSILON_PLAY * EPSILON_DECAY, EPSILON_MIN)
        if varied_bets_enabled:
            EPSILON_BET = max(EPSILON_BET * EPSILON_DECAY, EPSILON_MIN)
        elif episode_counter >= args.varied_bets_at:
            EPSILON_BET = 1.0
            varied_bets_enabled = True
            ui_data["varied_bets_episode"] = episode_counter
            if not args.no_ui:
                varied_bets_button.pack_forget()
                auto_enable_label.pack_forget()
                canvas_betting_table.pack(pady=10)
            if not args.quiet:
                print(f"\nEnabling varied bets at episode {episode_counter}")
        action_stats_per_episode.append(episode_action_stats)
        betting_stats_per_episode.append(episode_betting_stats)
        if total_reward > best_reward:
            best_reward = total_reward
        if episode_counter == 1:
            total_moving_avg = total_reward
        else:
            total_moving_avg = total_moving_avg * 0.999 + total_reward * 0.001
        recent_avg = sum(recent_rewards) / len(recent_rewards)
        if total_reward > 0:
            ui_data["wins"] += 1
        elif total_reward < 0:
            ui_data["losses"] += 1
        else:
            ui_data["ties"] += 1
        recent_wins = sum(1 for r in recent_rewards if r > 0)
        recent_losses = sum(1 for r in recent_rewards if r < 0)
        recent_total = recent_wins + recent_losses
        recent_win_ratio = (recent_wins / recent_total * 100) if recent_total > 0 else 0
        ui_data["last_recent_wins"] = recent_wins
        ui_data["last_recent_losses"] = recent_losses
        ui_data["last_recent_win_ratio"] = recent_win_ratio
        interval = get_print_interval(episode_counter)
        if episode_counter % interval == 0:
            ui_data["episode"] = episode_counter
            ui_data["ep_reward"] = total_reward
            ui_data["total_avg"] = total_moving_avg
            ui_data["recent_avg"] = recent_avg
            ui_data["epsilon_play"] = EPSILON_PLAY
            ui_data["epsilon_bet"] = EPSILON_BET
            ui_data["best_reward"] = best_reward
            ui_data["episodes"].append(episode_counter)
            ui_data["total_avgs"].append(total_moving_avg)
            ui_data["recent_avgs"].append(recent_avg)
            ui_data["epsilons_play"].append(EPSILON_PLAY)
            ui_data["epsilons_bet"].append(EPSILON_BET)
            ui_data["best_rewards"].append(best_reward)
            aggregated_action_data = {}
            for ep_stats in action_stats_per_episode[-interval:]:
                for hand_sum, counts in ep_stats.items():
                    if hand_sum not in aggregated_action_data:
                        aggregated_action_data[hand_sum] = [0,0,0,0]
                    for i in range(4):
                        aggregated_action_data[hand_sum][i] += counts[i]
            recent_betting_data = defaultdict(int)
            total_decisions = defaultdict(int)
            for stats in betting_stats_per_episode[-interval:]:
                for count, bet in stats:
                    recent_betting_data[(count,bet)] += 1
                    total_decisions[count] += 1
            betting_distribution = {}
            for (count,bet), freq in recent_betting_data.items():
                if total_decisions[count] > 0:
                    betting_distribution[(count,bet)] = freq/total_decisions[count]
            if not args.no_ui:
                update_ui_elements(aggregated_action_data, betting_distribution)
            if not args.quiet:
                current_epsilon = EPSILON_BET if phase == 0 else EPSILON_PLAY
                print(f"Episode {episode_counter:,} | Total Avg: {total_moving_avg:.2f} | Recent Avg: {recent_avg:.2f} | Epsilon: {current_epsilon:.3f} | Best: {best_reward:.2f}")
            recent_rewards.clear()
        if not args.quiet:
            print(f"Completed episode {episode_counter:,}")
    if not args.quiet:
        print("Training complete. Evaluating final agent using money metrics...")
    print("\nNow evaluating money over 1,000 episodes:")
    agent_money = evaluate_money(1000)
    global evaluation_metrics
    evaluation_metrics = {"agent": agent_money}
    root.quit()

def save_screenshot_and_summary():
    global action_stats_per_episode, betting_stats_per_episode, evaluation_metrics
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    try:
        interval = get_print_interval(ui_data['episode'])
        aggregated_action_data = {}
        if action_stats_per_episode:
            for ep_stats in action_stats_per_episode[-interval:]:
                for hand_sum, counts in ep_stats.items():
                    if hand_sum not in aggregated_action_data:
                        aggregated_action_data[hand_sum] = [0,0,0,0]
                    for i in range(4):
                        aggregated_action_data[hand_sum][i] += counts[i]
        recent_betting_data = defaultdict(int)
        total_decisions = defaultdict(int)
        if betting_stats_per_episode:
            for stats in betting_stats_per_episode[-interval:]:
                for count, bet in stats:
                    recent_betting_data[(count,bet)] += 1
                    total_decisions[count] += 1
        betting_distribution = {}
        for (count,bet), freq in recent_betting_data.items():
            if total_decisions[count] > 0:
                betting_distribution[(count,bet)] = freq/total_decisions[count]
        for _ in range(4):
            update_ui_elements(aggregated_action_data, betting_distribution, force_update=True)
            root.update_idletasks()
            root.update()
            time.sleep(0.5)
        update_ui_elements(aggregated_action_data, betting_distribution, force_update=True)
        root.update_idletasks()
        root.update()
        time.sleep(1.0)
        x = root.winfo_rootx()
        y = root.winfo_rooty()
        w = root.winfo_width()
        h = root.winfo_height()
        bbox = (x, y, x+w, y+h)
        screenshot = ImageGrab.grab(bbox)
        now = datetime.now()
        date_str = now.strftime("%m-%d")
        time_str = now.strftime("%H-%M")
        base_filename = f"{date_str}_{time_str}_{ui_data['episode']}_{ui_data['recent_avg']:.2f}"
        screenshot_filename = os.path.join(results_folder, f"{base_filename}.png")
        screenshot.save(screenshot_filename)
        print(f"Saved screenshot: {screenshot_filename}")
        summary_filename = os.path.join(results_folder, f"{base_filename}.txt")
        with open(summary_filename, "w") as f:
            f.write("=== Training Session Parameters ===\n")
            f.write(f"Date: {date_str}\n")
            f.write(f"Time: {time_str}\n")
            f.write(f"Alpha (Learning Rate): {ALPHA}\n")
            f.write(f"Gamma (Discount Factor): {GAMMA}\n")
            f.write(f"Epsilon Decay Rate: {EPSILON_DECAY}\n")
            f.write(f"Final Play Epsilon: {EPSILON_PLAY:.6f}\n")
            f.write(f"Final Bet Epsilon: {EPSILON_BET:.6f}\n")
            f.write(f"Initial Bankroll: {1000}\n")
            f.write(f"Rounds per Episode: {args.max_rounds}\n")
            f.write(f"Total Episodes: {ui_data['episode']:,}\n")
            f.write(f"Varied Bets Start: {args.varied_bets_at:,}\n\n")
            f.write("=== Performance Metrics ===\n")
            f.write(f"Total Moving Average: {ui_data['total_avg']:.2f}\n")
            f.write(f"Recent Average: {ui_data['recent_avg']:.2f}\n")
            f.write(f"Best Reward: {ui_data['best_reward']:.2f}\n")
            lifetime_win_ratio = (ui_data["wins"]/(ui_data["wins"]+ui_data["losses"]))*100
            f.write(f"Lifetime Wins: {ui_data['wins']:,}\n")
            f.write(f"Lifetime Losses: {ui_data['losses']:,}\n")
            f.write(f"Lifetime Ties: {ui_data['ties']:,}\n")
            f.write(f"Lifetime Win Ratio: {lifetime_win_ratio:.1f}%\n\n")
            f.write("=== Recent Performance (Last Update Window) ===\n")
            f.write(f"Recent Wins: {ui_data['last_recent_wins']}\n")
            f.write(f"Recent Losses: {ui_data['last_recent_losses']}\n")
            f.write(f"Recent Win Ratio: {ui_data['last_recent_win_ratio']:.1f}%\n\n")
            f.write("=== Evaluation (Money) ===\n")
            if evaluation_metrics is not None:
                agent_avg, agent_prof, agent_non = evaluation_metrics["agent"]
                f.write("Agent Policy Evaluation:\n")
                f.write(f"  Average Reward per Episode: {agent_avg:.2f}\n")
                f.write(f"  Profitable Episodes: {agent_prof} ({agent_prof/1000*100:.1f}%)\n")
            else:
                f.write("No evaluation data available.\n\n")
            f.write("=== Betting Strategy ===\n")
            if ui_data["varied_bets_episode"] is not None:
                f.write(f"Varied Bets Enabled at Episode: {ui_data['varied_bets_episode']:,}\n")
                f.write(f"Episodes with Varied Bets: {ui_data['episode']-ui_data['varied_bets_episode']:,}\n")
            else:
                f.write("Varied Bets: Not Enabled\n")
        print(f"Saved summary: {summary_filename}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def enable_varied_bets():
    global EPSILON_BET, varied_bets_enabled, args
    if not varied_bets_enabled:
        EPSILON_BET = 1.0
        varied_bets_enabled = True
        ui_data["varied_bets_episode"] = ui_data["episode"]
        varied_bets_button.pack_forget()
        auto_enable_label.pack_forget()
        canvas_betting_table.pack(pady=10)
        if not args.quiet:
            print(f"\nManually enabling varied bets at episode {ui_data['episode']}")

def main():
    global args
    args = parse_args()
    global ALPHA, GAMMA, EPSILON_DECAY
    ALPHA = args.alpha
    GAMMA = args.gamma
    EPSILON_DECAY = args.epsilon_decay
    if args.no_ui:
        training_thread = threading.Thread(target=train_agent, args=(args,), daemon=True)
        training_thread.start()
        training_thread.join()
    else:
        create_ui(args)
        training_thread = threading.Thread(target=train_agent, args=(args,), daemon=True)
        training_thread.start()
        root.mainloop()

if __name__ == "__main__":
    main()