#!/usr/bin/env python3
"""
train_qlearning_agent_tkinter.py

This script trains a Q-Learning agent (with card counting) using our AdvancedBlackjackEnv.
Key updates:
  - The training loop runs indefinitely until manually stopped.
  - A Pause/Play button (in the top-left section) pauses/resumes training.
  - A Stop button (in the top-left section) stops training, saves a screenshot and final summary log
    (with filenames formatted as {mm-dd}_{hh-mm}_{episode}_{recent_avg}), and then closes the program.
  - The top stats area is split into three side-by-side sections:
      • Section 1 (Header): Displays the episode number and control buttons.
      • Section 2 (Stats): Displays Total Moving Avg, Recent Avg, Epsilon, and Best Reward.
      • Section 3 (Wins): Displays Lifetime win/loss stats and recent win ratio.
  - Console output and UI updates occur only at intervals:
      • Every 100 episodes until 2500,
      • Every 500 episodes from 2500 to 10000,
      • Every 1000 episodes thereafter.
  - When training stops (via the Stop button or window close), a screenshot and summary file are saved in the "results" folder.

--- Additional Reward Shaping Suggestions ---
# 1. Provide intermediate rewards during the playing phase.
# 2. Penalize unsafe actions by subtracting a small reward.
# 3. Use multi-objective rewards considering decision efficiency.
# 4. Experiment with alternative RL methods.
# 5. Incorporate additional state features or finer state discretization.
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
import argparse  # Add this at the top with other imports

from advanced_blackjack_env import AdvancedBlackjackEnv

# Add these at the top level with other global variables
header_label = None
left_stats_label = None
right_stats_label = None
canvas_chart = None
canvas_epsilon = None
canvas_action_table = None
canvas_betting_table = None
varied_bets_enabled = False  # Global flag for varied bets status
varied_bets_button = None
auto_enable_label = None
epsilon_label = None

def create_ui(args):
    global header_label, left_stats_label, right_stats_label, epsilon_label
    global canvas_chart, canvas_epsilon, canvas_action_table, canvas_betting_table
    global varied_bets_button, auto_enable_label
    
    # Create top section with 2x2 grid
    top_frame = tk.Frame(stats_frame)
    top_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    top_frame.grid_columnconfigure(0, weight=1)
    top_frame.grid_columnconfigure(1, weight=1)

    # Top Left: Episode Counter
    episode_frame = tk.Frame(top_frame)
    episode_frame.grid(row=0, column=0, sticky="w")
    header_label = tk.Label(episode_frame, text="Episode: 0", 
                           font=("Helvetica", 12, "bold"), justify=tk.LEFT)
    header_label.pack(anchor="w")
    
    # Add max episodes label if set
    if args.episodes:
        max_ep_label = tk.Label(episode_frame, 
                               text=f"Max Episodes: {args.episodes:,}", 
                               font=("Helvetica", 10), 
                               justify=tk.LEFT)
        max_ep_label.pack(anchor="w")

    # Top Right: Control Buttons
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
        root.quit()
    stop_button = tk.Button(control_frame, text="Stop", command=stop_training)
    stop_button.pack(side=tk.LEFT, padx=5)

    # Bottom Left: Epsilon Values
    epsilon_frame = tk.Frame(top_frame)
    epsilon_frame.grid(row=1, column=0, sticky="w", pady=(5,0))
    epsilon_label = tk.Label(epsilon_frame, 
                           text="Play ε: 1.000  |  Bet ε: 0.000",
                           font=("Helvetica", 10))
    epsilon_label.pack(anchor="w")

    # Bottom Right: Empty for now (symmetry)
    spacer_frame = tk.Frame(top_frame)
    spacer_frame.grid(row=1, column=1)

    # Create stats labels in their own frame below
    stats_labels_frame = tk.Frame(stats_frame)
    stats_labels_frame.grid(row=1, column=0, sticky="w", pady=(0,10))
    
    left_stats_label = tk.Label(stats_labels_frame, text="", font=("Helvetica", 10), justify=tk.LEFT)
    left_stats_label.grid(row=0, column=0, padx=20)
    
    right_stats_label = tk.Label(stats_labels_frame, text="", font=("Helvetica", 10), justify=tk.LEFT)
    right_stats_label.grid(row=0, column=1, padx=20)

    # Create canvas elements for charts
    canvas_chart = tk.Canvas(charts_frame, width=760, height=300, bg="white")
    canvas_chart.pack(pady=10)
    
    canvas_epsilon = tk.Canvas(charts_frame, width=760, height=300, bg="white")
    canvas_epsilon.pack(pady=10)
    
    canvas_action_table = tk.Canvas(right_frame, width=760, height=400, bg="white")
    canvas_action_table.pack(pady=10)

    # Create a frame for the betting section
    betting_frame = tk.Frame(right_frame)
    betting_frame.pack(pady=10, padx=10)

    # Create the varied bets button (initially visible)
    varied_bets_button = tk.Button(betting_frame, text="Enable Varied Betting", 
                                  command=enable_varied_bets,
                                  width=30, height=2)
    varied_bets_button.pack(pady=(20,5))  # Reduced bottom padding

    # Add auto-enable info label
    auto_enable_label = tk.Label(betting_frame, 
                                text=f"Will auto-enable at episode {args.varied_bets_at:,}", 
                                font=("Helvetica", 9))
    auto_enable_label.pack(pady=(0,20))  # Added top and bottom padding

    # Create the betting distribution table
    canvas_betting_table = tk.Canvas(betting_frame, width=760, height=400, bg="white")
    canvas_betting_table.pack(pady=10)

# ----- State Discretization and Action Selection Helpers -----
def discretize_state(obs: dict) -> tuple:
    if obs["phase"] == 0:
        bankroll_disc = int(obs["bankroll"] // 10)
        count = obs.get("card_count", 0)  # Get the card count for betting phase
        count_bin = int(np.clip(count, -5, 5))  # Clip count to [-5, 5] range
        return ("bet", bankroll_disc, count_bin)  # Include count in betting state
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
        actions = [0, 1]  # Hit and Stick always allowed.
        if obs is not None:
            if obs.get("can_double", False):
                actions.append(2)
            if obs.get("can_split", False):
                actions.append(3)
        return actions

# Learning parameters
ALPHA = 0.1             # Learning rate
GAMMA = 0.95            # Discount factor
EPSILON_PLAY = 1.0      # Initial epsilon for playing decisions
EPSILON_BET = 1.0       # Initial epsilon for betting decisions
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99975  # Same decay rate for both epsilons

# Separate Q-tables for betting and playing
Q_play = defaultdict(lambda: np.zeros(4))   # For playing phase; 4 actions
Q_bet = defaultdict(lambda: np.zeros(10))   # For betting phase; 10 bet sizes

def get_q(state: tuple, phase: int):
    if phase == 0:  # Betting phase
        if state not in Q_bet:
            Q_bet[state] = np.zeros(10)
        return Q_bet[state]
    else:  # Playing phase
        if state not in Q_play:
            Q_play[state] = np.zeros(4)
        return Q_play[state]

def choose_action(state: tuple, phase: int, epsilon: float, episode: int, obs: dict):
    global args
    if phase == 0:  # Betting phase
        if episode < args.varied_bets_at and not varied_bets_enabled:
            return 4  # Always bet 5 until varied_bets_at
        else:
            # Use epsilon-greedy for betting decisions
            if random.random() < EPSILON_BET:
                return random.randint(0, 9)  # Random bet size 0-9 (maps to 1-10)
            else:
                q_vals = get_q(state, phase)
                return int(np.argmax(q_vals))
    else:  # Playing phase
        allowed = possible_actions(phase, obs)
        if random.random() < epsilon:
            return random.choice(allowed)
        else:
            q_vals = get_q(state, phase)
            q_vals_allowed = [q_vals[a] if a in allowed else -float('inf') for a in range(4)]
            return int(np.argmax(q_vals_allowed))

# Determine printing (and UI update) interval based on episode count.
def get_print_interval(episode):
    if episode < 2500:
        return 100
    elif episode < 10000:
        return 500
    else:
        return 1000

# ----- Global Control Flags and Results Folder Setup -----
pause_event = threading.Event()  # When set, training is paused.
stop_event = threading.Event()   # When set, training should stop.

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
results_folder = os.path.join(script_dir, "results")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# ----- Tkinter UI Setup -----
root = tk.Tk()
root.title("Q-Learning Training Progress")

# Set window size and center it
window_width = 1650
window_height = 900  # Reduced from 950

# Get screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate position for center of screen
x = (screen_width - window_width) // 2
y = (screen_height - int(window_height*1.1) ) // 2

# Set the window size and position
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Main frame for charts and tables
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Left frame for charts
left_frame = tk.Frame(main_frame, width=800)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Right frame for tables
right_frame = tk.Frame(main_frame, width=800)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Move existing stats frame to left frame
stats_frame = tk.Frame(left_frame)
stats_frame.pack(fill=tk.X, padx=10, pady=10)

# Move existing charts frame to left frame
charts_frame = tk.Frame(left_frame)
charts_frame.pack(fill=tk.BOTH, expand=True)

# Global UI data.
ui_data = {
    "episode": 0,
    "ep_reward": 0.0,
    "total_avg": 0.0,
    "recent_avg": 0.0,  # Renamed from last100_avg to recent_avg
    "epsilon_play": EPSILON_PLAY,
    "epsilon_bet": EPSILON_BET,
    "best_reward": -float('inf'),
    "wins": 0,         # Lifetime wins
    "losses": 0,       # Lifetime losses
    "ties": 0,
    "episodes": [],
    "total_avgs": [],
    "recent_avgs": [],  # Renamed accordingly
    "epsilons_play": [],
    "epsilons_bet": [],
    "best_rewards": [],
    "varied_bets_episode": None  # Add this field
}
recent_rewards = []  # This will track episodes since the last UI update.

def update_text():
    global epsilon_label  # Add global declaration
    
    header_text = (f"Episode: {ui_data['episode']}")
    epsilon_text = (f"Play ε: {ui_data['epsilon_play']:.3f}  |  Bet ε: {ui_data['epsilon_bet']:.3f}")
    left_text = (f"Total Moving Avg: {ui_data['total_avg']:.2f}\n"
                 f"Recent Avg: {ui_data['recent_avg']:.2f}\n"
                 f"Best Reward: {ui_data['best_reward']:.2f}")
    lifetime_win_ratio = ((ui_data["wins"] / (ui_data["wins"] + ui_data["losses"]) * 100)
                         if (ui_data["wins"] + ui_data["losses"]) > 0 else 0)
    recent_wins = sum(1 for r in recent_rewards if r > 0)
    recent_losses = sum(1 for r in recent_rewards if r < 0)
    recent_total = recent_wins + recent_losses
    recent_win_ratio = (recent_wins / recent_total * 100) if recent_total > 0 else 0
    right_text = (f"Lifetime Wins: {ui_data['wins']}  Losses: {ui_data['losses']}\n"
                  f"Lifetime Win Ratio: {lifetime_win_ratio:.1f}%\n"
                  f"Recent Wins: {recent_wins}  Losses: {recent_losses}\n"
                  f"Recent Win Ratio: {recent_win_ratio:.0f}%")
    
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
    for i in range(num_grid + 1):
        if x_range != 0:
            data_x = x_min + i * x_range / num_grid
            x = margin + (data_x - x_min) / x_range * (w - 2 * margin)
        else:
            data_x = x_min
            x = margin
        canvas.create_line(x, margin, x, h - margin, fill="lightgray", tags="grid")
        canvas.create_text(x, h - margin + 15, text=str(round(data_x)), fill="gray", font=("Helvetica", 8), tags="grid")
    for i in range(num_grid + 1):
        if y_range != 0:
            data_y = y_min + i * y_range / num_grid
            y = h - margin - ((data_y - y_min) / y_range) * (h - 2 * margin)
        else:
            data_y = y_min
            y = h - margin
        color = "black" if abs(data_y) < 1e-6 else "lightgray"
        canvas.create_line(margin, y, w - margin, y, fill=color, tags="grid")
        canvas.create_text(margin - 15, y, text=str(round(data_y,1)), fill="gray", font=("Helvetica", 8), tags="grid")

def draw_chart(canvas, episodes, series1, series2, label1, label2):
    canvas.delete("all")
    w = int(canvas["width"])
    h = int(canvas["height"])
    margin = 40
    
    if not episodes:
        return
        
    # Create 10 interpolated points between each update interval
    points1 = []
    points2 = []
    for i in range(len(episodes)-1):
        ep_start = episodes[i]
        ep_end = episodes[i+1]
        val1_start = series1[i]
        val1_end = series1[i+1]
        val2_start = series2[i]
        val2_end = series2[i+1]
        
        # Create 10 points for this interval
        for j in range(10):
            t = j / 10
            ep = ep_start + (ep_end - ep_start) * t
            val1 = val1_start + (val1_end - val1_start) * t
            val2 = val2_start + (val2_end - val2_start) * t
            points1.append((ep, val1))
            points2.append((ep, val2))
    
    # Add the final point
    if episodes:
        points1.append((episodes[-1], series1[-1]))
        points2.append((episodes[-1], series2[-1]))
    
    x_min = episodes[0] if episodes else 0
    x_max = episodes[-1] if episodes else 1
    y_min_data = min(min(series1), min(series2))
    y_max_data = max(max(series1), max(series2))
    y_range = max(100, max(abs(y_min_data), abs(y_max_data)))
    y_min = -y_range
    y_max = y_range
    
    draw_grid(canvas, x_min, x_max, y_min, y_max, margin=margin, num_grid=10)
    
    def scale_x(x):
        return margin + (x - x_min) / (x_max - x_min) * (w - 2 * margin) if x_max != x_min else margin
    def scale_y(y):
        return h - margin - ((y - y_min) / (y_max - y_min)) * (h - 2 * margin) if y_max != y_min else h/2
    
    # Draw interpolated lines for series1 (blue)
    for i in range(1, len(points1)):
        x1 = scale_x(points1[i-1][0])
        y1 = scale_y(points1[i-1][1])
        x2 = scale_x(points1[i][0])
        y2 = scale_y(points1[i][1])
        canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)
    
    # Draw interpolated lines for series2 (orange)
    for i in range(1, len(points2)):
        x1 = scale_x(points2[i-1][0])
        y1 = scale_y(points2[i-1][1])
        x2 = scale_x(points2[i][0])
        y2 = scale_y(points2[i][1])
        canvas.create_line(x1, y1, x2, y2, fill="orange", width=2)
    
    canvas.create_text(margin + 20, margin - 10, text=label1, fill="blue", font=("Helvetica", 10))
    canvas.create_text(margin + 150, margin - 10, text=label2, fill="orange", font=("Helvetica", 10))

    # Draw vertical line for when varied bets were enabled
    if ui_data["varied_bets_episode"] is not None:
        x = scale_x(ui_data["varied_bets_episode"])
        canvas.create_line(x, margin, x, h - margin, dash=(5,5), fill="red", width=1)
        canvas.create_text(x + 5, margin + 15, text="Varied Bets", fill="red", 
                         font=("Helvetica", 8), anchor="w")

    # Draw title
    canvas.create_text(w/2, 20, text="Training Progress Over Time", 
                      font=("Helvetica", 12, "bold"))

def draw_epsilon_chart(canvas, episodes, epsilons_play, epsilons_bet):
    canvas.delete("all")
    w = int(canvas["width"])
    h = int(canvas["height"])
    margin = 40
    if not episodes:
        return

    # Draw title and epsilon values at the top
    title_y = 20
    canvas.create_text(w/2, title_y, 
                      text="Exploration Rates (ε) Decay", 
                      font=("Helvetica", 12, "bold"))
    
    # Add current epsilon values at top
    eps_text = f"Play ε: {epsilons_play[-1]:.3f}  |  Bet ε: {epsilons_bet[-1]:.3f}"
    canvas.create_text(w/2, title_y + 20, 
                      text=eps_text,
                      font=("Helvetica", 10))
    
    # Create interpolated points for both epsilon series
    points_play = []
    points_bet = []
    for i in range(len(episodes)-1):
        ep_start = episodes[i]
        ep_end = episodes[i+1]
        eps_play_start = epsilons_play[i]
        eps_play_end = epsilons_play[i+1]
        eps_bet_start = epsilons_bet[i]
        eps_bet_end = epsilons_bet[i+1]
        
        # Create 10 points for this interval
        for j in range(10):
            t = j / 10
            ep = ep_start + (ep_end - ep_start) * t
            eps_play = eps_play_start + (eps_play_end - eps_play_start) * t
            eps_bet = eps_bet_start + (eps_bet_end - eps_bet_start) * t
            points_play.append((ep, eps_play))
            points_bet.append((ep, eps_bet))
    
    # Add the final points
    points_play.append((episodes[-1], epsilons_play[-1]))
    points_bet.append((episodes[-1], epsilons_bet[-1]))
    
    x_min = episodes[0]
    x_max = episodes[-1]
    y_min = 0
    y_max = 1
    draw_grid(canvas, x_min, x_max, y_min, y_max, margin=margin, num_grid=10)
    
    def scale_x(x):
        return margin + (x - x_min) / (x_max - x_min) * (w - 2 * margin) if x_max != x_min else margin
    def scale_y(y):
        return h - margin - ((y - y_min) / (y_max - y_min)) * (h - 2 * margin) if y_max != y_min else h/2
    
    # Draw interpolated lines for play epsilon (green)
    for i in range(1, len(points_play)):
        x1 = scale_x(points_play[i-1][0])
        y1 = scale_y(points_play[i-1][1])
        x2 = scale_x(points_play[i][0])
        y2 = scale_y(points_play[i][1])
        canvas.create_line(x1, y1, x2, y2, fill="green", width=2)
    
    # Draw interpolated lines for bet epsilon (blue)
    for i in range(1, len(points_bet)):
        x1 = scale_x(points_bet[i-1][0])
        y1 = scale_y(points_bet[i-1][1])
        x2 = scale_x(points_bet[i][0])
        y2 = scale_y(points_bet[i][1])
        canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)
    
    # Draw legend
    legend_x = w - margin - 10
    legend_y = margin + 10
    canvas.create_line(legend_x - 40, legend_y, legend_x - 20, legend_y, fill="green", width=2)
    canvas.create_text(legend_x - 10, legend_y, text="Play ε", fill="green", 
                      font=("Helvetica", 10), anchor="w")
    canvas.create_line(legend_x - 40, legend_y + 20, legend_x - 20, legend_y + 20, 
                      fill="blue", width=2)
    canvas.create_text(legend_x - 10, legend_y + 20, text="Bet ε", fill="blue", 
                      font=("Helvetica", 10), anchor="w")

    # Draw title
    canvas.create_text(w/2, 20, text="Exploration Rates (ε) Decay", 
                      font=("Helvetica", 12, "bold"))

def get_color_from_probability(prob):
    g = b = int(255 * (1 - prob))
    # Ensure values are between 0 and 255
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return f"#{255:02x}{g:02x}{b:02x}"

def draw_action_probability_table(canvas, aggregated_action_data):
    canvas.delete("all")
    w = int(canvas["width"])
    h = int(canvas["height"])
    margin_left = 60
    margin_top = 120
    margin_right = 20
    margin_bottom = 40

    # Draw title
    canvas.create_text(w/2, margin_top - 60,
                      text="Hand Sum vs Action Probability (Recent Episodes)", 
                      font=("Helvetica", 12, "bold"))

    # Check if data is empty or has no hand sums
    if not aggregated_action_data:
        canvas.create_text(w/2, h/2,
                          text="Collecting data...", 
                          font=("Helvetica", 10))
        return

    sorted_hand_sums = sorted(aggregated_action_data.keys())
    if not sorted_hand_sums:  # If no hand sums in data
        canvas.create_text(w/2, h/2,
                          text="Collecting data...", 
                          font=("Helvetica", 10))
        return

    num_cols = max(1, len(sorted_hand_sums))  # Ensure at least 1 column
    num_rows = 4  # Hit, Stand, Double, Split

    cell_width = (w - margin_left - margin_right) / num_cols
    cell_height = (h - margin_top - margin_bottom) / num_rows

    # Draw action labels on the left
    action_labels = ["Hit", "Stand", "x2", "Split"]
    for row, label in enumerate(action_labels):
        y = margin_top + row * cell_height + cell_height/2
        canvas.create_text(margin_left - 10, y, text=label, 
                          font=("Helvetica", 10), anchor="e")

    # Draw cells
    for col, hand_sum in enumerate(sorted_hand_sums):
        counts = aggregated_action_data[hand_sum]
        total = sum(counts)
        for row in range(num_rows):
            prob = counts[row] / total if total > 0 else 0
            color = get_color_from_probability(prob)
            x0 = margin_left + col * cell_width
            y0 = margin_top + row * cell_height
            x1 = x0 + cell_width
            y1 = y0 + cell_height
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
            canvas.create_text((x0+x1)/2, (y0+y1)/2, text=f"{round(prob*100)}%", font=("Helvetica", 8))

    # Draw hand sum labels at the top
    for col, hand_sum in enumerate(sorted_hand_sums):
        x = margin_left + col * cell_width + cell_width/2
        canvas.create_text(x, margin_top - 15, text=str(hand_sum), font=("Helvetica", 10))

def draw_betting_distribution_table(canvas, recent_betting_data):
    canvas.delete("all")
    w = int(canvas["width"])
    h = int(canvas["height"])
    margin_left = 60
    margin_top = 120
    margin_right = 20
    margin_bottom = 40

    # Define count range and bet sizes
    count_range = range(-5, 6)  # -5 to +5
    bet_sizes = range(1, 11)    # 1 to 10 (inclusive)
    
    num_rows = len(count_range)
    num_cols = len(bet_sizes)

    cell_width = (w - margin_left - margin_right) / num_cols
    cell_height = (h - margin_top - margin_bottom) / num_rows

    # Draw title
    canvas.create_text(w/2, margin_top - 60,
                      text="Card Count vs Bet Size Distribution (Recent Episodes)", 
                      font=("Helvetica", 12, "bold"))

    # Draw cells
    for row, count in enumerate(count_range):
        for col, bet_size in enumerate(bet_sizes):
            # Convert from 1-based bet size to 0-based action index
            action = bet_size - 1
            prob = recent_betting_data.get((count, action), 0)
            color = get_color_from_probability(prob)
            x0 = margin_left + col * cell_width
            y0 = margin_top + row * cell_height
            x1 = x0 + cell_width
            y1 = y0 + cell_height
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
            canvas.create_text((x0+x1)/2, (y0+y1)/2, text=f"{round(prob*100)}%", font=("Helvetica", 8))

    # Draw labels
    for col, bet_size in enumerate(bet_sizes):
        x = margin_left + col * cell_width + cell_width/2
        canvas.create_text(x, margin_top - 15, text=str(bet_size), font=("Helvetica", 10))
    
    for row, count in enumerate(count_range):
        y = margin_top + row * cell_height + cell_height/2
        canvas.create_text(margin_left - 20, y, text=str(count), font=("Helvetica", 10))

def update_ui_elements(aggregated_action_data=None, betting_distribution=None, force_update=False):
    """
    Unified function to update all UI elements.
    Set force_update=True to ensure all elements update even during screenshot.
    """
    # Update text elements
    update_text()
    
    # Update charts
    draw_chart(canvas_chart, ui_data["episodes"], ui_data["total_avgs"], ui_data["recent_avgs"],
               "Total Moving Avg", "Recent Avg")
    draw_epsilon_chart(canvas_epsilon, ui_data["episodes"], ui_data["epsilons_play"], ui_data["epsilons_bet"])
    
    # Update tables
    if aggregated_action_data is not None:
        draw_action_probability_table(canvas_action_table, aggregated_action_data)
    if varied_bets_enabled and betting_distribution is not None:
        draw_betting_distribution_table(canvas_betting_table, betting_distribution)
    
    # Force immediate update if requested
    if force_update:
        root.update_idletasks()
        root.update()
        time.sleep(0.5)  # Wait for render

def shape_reward(reward):
    if reward > 0:
        return reward * 2.0
    elif reward < 0:
        return reward * 0.5
    else:
        return reward

def parse_args():
    parser = argparse.ArgumentParser(description='Train Q-Learning Agent for Blackjack')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.95,
                       help='Discount factor (default: 0.95)')
    parser.add_argument('--epsilon-decay', type=float, default=0.99975,
                       help='Epsilon decay rate (default: 0.99975)')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes to train (default: None, runs indefinitely)')
    parser.add_argument('--no-ui', action='store_true',
                       help='Disable UI and run in console mode')
    parser.add_argument('--quiet', action='store_true',
                       help='Disable console output')
    parser.add_argument('--max-rounds', type=int, default=100,
                       help='Number of rounds per episode (default: 100)')
    parser.add_argument('--varied-bets-at', type=int, default=25000,
                       help='Episode number to enable varied bets (default: 25000)')
    return parser.parse_args()

def train_agent(args):
    global EPSILON_PLAY, EPSILON_BET, varied_bets_enabled
    global action_stats_per_episode, betting_stats_per_episode  # Add global declaration
    
    env = AdvancedBlackjackEnv(render_mode=None, natural=True, initial_bankroll=1000, max_rounds=args.max_rounds)
    best_reward = -float('inf')
    total_moving_avg = 0.0
    global recent_rewards
    recent_rewards.clear()
    episode_counter = 0
    action_stats_per_episode = []  # Initialize here
    betting_stats_per_episode = []  # Initialize here
    
    # Reset epsilons at start of training
    EPSILON_PLAY = 1.0
    EPSILON_BET = 0.0
    varied_bets_enabled = False
    
    while not stop_event.is_set():
        if args.episodes and episode_counter >= args.episodes:
            stop_event.set()
            if not args.no_ui:
                save_screenshot_and_summary()
            break
            
        while pause_event.is_set() and not stop_event.is_set():
            time.sleep(0.1)
        
        obs, _ = env.reset()
        total_reward = 0
        done = False
        episode_action_stats = {}
        episode_betting_stats = []
        
        # Update epsilons with same decay rate
        EPSILON_PLAY = max(EPSILON_PLAY * EPSILON_DECAY, EPSILON_MIN)
        if varied_bets_enabled:  # Decay bet epsilon once betting is enabled
            EPSILON_BET = max(EPSILON_BET * EPSILON_DECAY, EPSILON_MIN)
        elif episode_counter >= args.varied_bets_at:  # Check for automatic trigger
            EPSILON_BET = 1.0  # Reset to 1.0 when varied bets are first enabled
            varied_bets_enabled = True
            if not args.quiet:
                print(f"\nEnabling varied bets at episode {episode_counter}")

        while not done:
            phase = obs["phase"]
            state = discretize_state(obs)
            
            # Use appropriate epsilon based on phase
            epsilon = EPSILON_BET if phase == 0 else EPSILON_PLAY
            
            action = choose_action(state, phase, epsilon, episode_counter, obs)
            next_obs, reward, done, _, info = env.step(action)
            
            # Record action statistics
            if phase == 1:  # Playing phase
                hand_sum = state[0]  # Get player's hand sum
                if hand_sum not in episode_action_stats:
                    episode_action_stats[hand_sum] = [0, 0, 0, 0]
                episode_action_stats[hand_sum][action] += 1
            elif phase == 0:  # Betting phase
                count = state[2]  # Get card count from betting state tuple
                episode_betting_stats.append((count, action))
            
            total_reward += reward
            next_state = discretize_state(next_obs)
            q_current = get_q(state, phase)
            if not done:
                next_phase = next_obs["phase"]
                q_next = get_q(next_state, next_phase)
                max_next = np.max(q_next)
            else:
                max_next = 0
            shaped_r = shape_reward(reward)
            q_current[action] = q_current[action] + ALPHA * (shaped_r + GAMMA * max_next - q_current[action])
            obs = next_obs
        
        action_stats_per_episode.append(episode_action_stats)
        betting_stats_per_episode.append(episode_betting_stats)
        episode_counter += 1
        recent_rewards.append(total_reward)
        # Use all episodes since last UI update.
        recent_avg = np.mean(recent_rewards) if recent_rewards else 0.0
        total_moving_avg = (total_moving_avg * (episode_counter - 1) + total_reward) / episode_counter
        best_reward = max(best_reward, total_reward)
        
        if total_reward > 0:
            ui_data["wins"] += 1
        elif total_reward < 0:
            ui_data["losses"] += 1
        else:
            ui_data["ties"] += 1
        
        interval = get_print_interval(episode_counter)
        if episode_counter % interval == 0:
            # Update UI data
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

            # Calculate aggregated data
            aggregated_action_data = {}
            for ep_stats in action_stats_per_episode[-interval:]:
                for hand_sum, counts in ep_stats.items():
                    if hand_sum not in aggregated_action_data:
                        aggregated_action_data[hand_sum] = [0, 0, 0, 0]
                    for i in range(4):
                        aggregated_action_data[hand_sum][i] += counts[i]
            
            # Calculate betting distribution
            recent_betting_data = defaultdict(int)
            total_decisions = defaultdict(int)
            for stats in betting_stats_per_episode[-interval:]:
                for count, bet in stats:
                    recent_betting_data[(count, bet)] += 1
                    total_decisions[count] += 1
            
            betting_distribution = {}
            for (count, bet), freq in recent_betting_data.items():
                if total_decisions[count] > 0:
                    betting_distribution[(count, bet)] = freq / total_decisions[count]
            
            if not args.no_ui:
                update_ui_elements(aggregated_action_data, betting_distribution)
            
            if not args.quiet:
                print(f"Episode {episode_counter:,} | Total Avg: {total_moving_avg:.2f} | "
                      f"Recent Avg: {recent_avg:.2f} | Epsilon: {EPSILON_BET if phase == 0 else EPSILON_PLAY:.3f} | "
                      f"Best: {best_reward:.2f}")
            
            recent_rewards.clear()
        
        # Check for automatic varied bets enable
        if not varied_bets_enabled and episode_counter >= args.varied_bets_at:
            EPSILON_BET = 1.0  # Reset to 1.0 when varied bets are first enabled
            varied_bets_enabled = True
            ui_data["varied_bets_episode"] = episode_counter
            if not args.no_ui:
                # Hide button and auto-enable label
                varied_bets_button.pack_forget()
                auto_enable_label.pack_forget()
                # Show betting table
                canvas_betting_table.pack(pady=10)
            if not args.quiet:
                print(f"\nEnabling varied bets at episode {episode_counter}")
    
    if not args.quiet:
        print("Training complete.")
    root.quit()

def save_screenshot_and_summary():
    global action_stats_per_episode, betting_stats_per_episode
    
    # Create results folder if it doesn't exist
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    try:
        # Calculate aggregated action data from recent episodes
        interval = get_print_interval(ui_data['episode'])
        aggregated_action_data = {}
        
        if action_stats_per_episode:
            for ep_stats in action_stats_per_episode[-interval:]:
                for hand_sum, counts in ep_stats.items():
                    if hand_sum not in aggregated_action_data:
                        aggregated_action_data[hand_sum] = [0, 0, 0, 0]
                    for i in range(4):
                        aggregated_action_data[hand_sum][i] += counts[i]
        
        # Collect betting distribution data
        recent_betting_data = defaultdict(int)
        total_decisions = defaultdict(int)
        
        if betting_stats_per_episode:
            for stats in betting_stats_per_episode[-interval:]:
                for count, bet in stats:
                    recent_betting_data[(count, bet)] += 1
                    total_decisions[count] += 1
        
        # Convert frequencies to probabilities
        betting_distribution = {}
        for (count, bet), freq in recent_betting_data.items():
            if total_decisions[count] > 0:
                betting_distribution[(count, bet)] = freq / total_decisions[count]

        # Multiple UI updates to ensure everything is rendered
        for _ in range(4):
            # Update UI with force_update=True for screenshot
            update_ui_elements(aggregated_action_data, betting_distribution, force_update=True)
            root.update_idletasks()
            root.update()
            time.sleep(0.5)  # Short wait between updates
        
        # Final update and longer wait
        update_ui_elements(aggregated_action_data, betting_distribution, force_update=True)
        root.update_idletasks()
        root.update()
        time.sleep(1.0)  # Longer wait before screenshot
        
        # Take the screenshot
        x = root.winfo_rootx()
        y = root.winfo_rooty()
        w = root.winfo_width()
        h = root.winfo_height()
        bbox = (x, y, x + w, y + h)
        screenshot = ImageGrab.grab(bbox)
        
        # Save screenshot and summary
        now = datetime.now()
        date_str = now.strftime("%m-%d")
        time_str = now.strftime("%H-%M")
        base_filename = f"{date_str}_{time_str}_{ui_data['episode']}_{ui_data['recent_avg']:.2f}"
        
        screenshot_filename = os.path.join(results_folder, f"{base_filename}.png")
        screenshot.save(screenshot_filename)
        print(f"Saved screenshot: {screenshot_filename}")
        
        summary_filename = os.path.join(results_folder, f"{base_filename}.txt")
        with open(summary_filename, "w") as f:
            # Session Parameters
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

            # Performance Metrics
            f.write("=== Performance Metrics ===\n")
            f.write(f"Total Moving Average: {ui_data['total_avg']:.2f}\n")
            f.write(f"Recent Average: {ui_data['recent_avg']:.2f}\n")
            f.write(f"Best Reward: {ui_data['best_reward']:.2f}\n")
            lifetime_win_ratio = (ui_data["wins"] / (ui_data["wins"] + ui_data["losses"]) ) * 100
            f.write(f"Lifetime Wins: {ui_data['wins']:,}\n")
            f.write(f"Lifetime Losses: {ui_data['losses']:,}\n")
            f.write(f"Lifetime Ties: {ui_data['ties']:,}\n")
            f.write(f"Lifetime Win Ratio: {lifetime_win_ratio:.1f}%\n\n")

            # Recent Performance
            recent_wins = sum(1 for r in recent_rewards if r > 0)
            recent_losses = sum(1 for r in recent_rewards if r < 0)
            recent_total = recent_wins + recent_losses
            recent_win_ratio = (recent_wins / recent_total * 100) if recent_total > 0 else 0
            f.write("=== Recent Performance ===\n")
            f.write(f"Recent Wins: {recent_wins:,}\n")
            f.write(f"Recent Losses: {recent_losses:,}\n")
            f.write(f"Recent Win Ratio: {recent_win_ratio:.1f}%\n\n")

            # Varied Bets Information
            f.write("=== Betting Strategy ===\n")
            if ui_data["varied_bets_episode"] is not None:
                f.write(f"Varied Bets Enabled at Episode: {ui_data['varied_bets_episode']:,}\n")
                f.write(f"Episodes with Varied Bets: {ui_data['episode'] - ui_data['varied_bets_episode']:,}\n")
            else:
                f.write("Varied Bets: Not Enabled\n")

        print(f"Saved summary: {summary_filename}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def enable_varied_bets():
    global EPSILON_BET, varied_bets_enabled, args
    if not varied_bets_enabled:
        EPSILON_BET = 1.0  # Set to 1.0 when manually enabled
        varied_bets_enabled = True
        ui_data["varied_bets_episode"] = ui_data["episode"]
        # Hide button and auto-enable label
        varied_bets_button.pack_forget()
        auto_enable_label.pack_forget()
        # Show betting table
        canvas_betting_table.pack(pady=10)
        if not args.quiet:
            print(f"\nManually enabling varied bets at episode {ui_data['episode']}")

def main():
    global args  # Add this line
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