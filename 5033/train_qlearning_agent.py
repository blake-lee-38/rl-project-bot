#!/usr/bin/env python3
"""
train_qlearning_agent_tkinter.py

This script trains a Q-Learning agent (with card counting) using our AdvancedBlackjackEnv.
Key updates:
  - The training loop runs indefinitely until manually stopped.
  - A Pause/Play button (in the top-left section) pauses/resumes training.
  - A Stop button (in the top-left section) stops training, saves a screenshot and final summary log
    (with filenames formatted as {mm-dd}_{hh-mm}_{episode}_{last100_avg}), and then closes the program.
  - The top stats area is split into three side-by-side sections:
      • Section 1 (Header): Displays the episode number and control buttons.
      • Section 2 (Stats): Displays Total Moving Avg, Last 100 Avg, Epsilon, and Best Reward.
      • Section 3 (Wins): Displays global win/loss stats and last 100 win ratio.
  - Console output and UI updates occur only at intervals:
      • Every 100 episodes until 2500,
      • Every 500 episodes from 2500 to 10000,
      • Every 1000 episodes from 10000 to 25000, etc.
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

# Import the environment using its proper name.
from advanced_blackjack_env import AdvancedBlackjackEnv

# ----- State Discretization and Action Selection Helpers -----
def discretize_state(obs: dict) -> tuple:
    if obs["phase"] == 0:
        bankroll_disc = int(obs["bankroll"] // 10)
        return ("bet", bankroll_disc)
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

ALPHA = 0.1         # Learning rate
GAMMA = 0.95        # Discount factor
EPSILON = 1.0       # Initial epsilon
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99975

Q = defaultdict(lambda: np.zeros(10))  # For betting phase; playing phase uses 4 actions.

def get_q(state: tuple, phase: int):
    if phase == 0:
        if state not in Q:
            Q[state] = np.zeros(10)
        return Q[state]
    else:
        if state not in Q:
            Q[state] = np.zeros(4)
        return Q[state]

def choose_action(state: tuple, phase: int, epsilon: float, episode: int, obs: dict):
    if phase == 0:
        return 9  # Always bet 10.
    else:
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
        return 1000  # Cap at 1000 episodes max interval

# ----- Global Control Flags and Results Folder Setup -----
pause_event = threading.Event()  # When set, training is paused.
stop_event = threading.Event()   # When set, training should stop.

results_folder = "results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# ----- Tkinter UI Setup -----
root = tk.Tk()
root.title("Q-Learning Training Progress")
# Position the window at (20,20) with a size of 800x950.
root.geometry("800x950+20+20")

# Top stats frame with 3 side-by-side sections.
stats_frame = tk.Frame(root)
stats_frame.pack(fill=tk.X, padx=10, pady=10)

# Section 1: Header (Episode number and control buttons)
header_frame = tk.Frame(stats_frame)
header_frame.grid(row=0, column=0, sticky="w")
header_label = tk.Label(header_frame, text="", font=("Helvetica", 12, "bold"), justify=tk.LEFT)
header_label.pack(anchor="w")
control_buttons_frame = tk.Frame(header_frame)
control_buttons_frame.pack(anchor="w", pady=5)
def toggle_pause():
    if pause_event.is_set():
        pause_event.clear()
        pause_button.config(text="Pause")
    else:
        pause_event.set()
        pause_button.config(text="Play")
pause_button = tk.Button(control_buttons_frame, text="Pause", command=toggle_pause)
pause_button.pack(side=tk.LEFT, padx=5)
def stop_training():
    stop_event.set()
    save_screenshot_and_summary()
    root.quit()
stop_button = tk.Button(control_buttons_frame, text="Stop", command=stop_training)
stop_button.pack(side=tk.LEFT, padx=5)

# Section 2: Left Stats (Performance metrics)
left_stats_frame = tk.Frame(stats_frame)
left_stats_frame.grid(row=0, column=1, padx=20, sticky="w")
left_stats_label = tk.Label(left_stats_frame, text="", font=("Helvetica", 12), justify=tk.LEFT)
left_stats_label.pack(anchor="w")

# Section 3: Right Stats (Win/Loss metrics)
right_stats_frame = tk.Frame(stats_frame)
right_stats_frame.grid(row=0, column=2, sticky="w")
right_stats_label = tk.Label(right_stats_frame, text="", font=("Helvetica", 12), justify=tk.LEFT)
right_stats_label.pack(anchor="w")

# Main frame for charts.
charts_frame = tk.Frame(root)
charts_frame.pack(fill=tk.BOTH, expand=True)

# Upper frame for charts.
upper_frame = tk.Frame(charts_frame)
upper_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
canvas_chart = tk.Canvas(upper_frame, width=760, height=300, bg="white")
canvas_chart.pack(pady=10)
canvas_epsilon = tk.Canvas(upper_frame, width=760, height=150, bg="white")
canvas_epsilon.pack(pady=10)

# Lower frame for the action probability table.
lower_frame = tk.Frame(charts_frame)
lower_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
canvas_action_table = tk.Canvas(lower_frame, width=760, height=300, bg="white")
canvas_action_table.pack(pady=10)

# Global UI data.
ui_data = {
    "episode": 0,
    "ep_reward": 0.0,
    "total_avg": 0.0,
    "last100_avg": 0.0,
    "epsilon": EPSILON,
    "best_reward": -float('inf'),
    "wins": 0,
    "losses": 0,
    "ties": 0,
    "episodes": [],
    "total_avgs": [],
    "last100_avgs": [],
    "epsilons": [],
    "best_rewards": []
}
recent_rewards = []  # To track recent episode outcomes (for last 100 metrics).

def update_text():
    header_text = f"Episode: {ui_data['episode']}"
    left_text = (f"Total Moving Avg: {ui_data['total_avg']:.2f}\n"
                 f"Last 100 Avg: {ui_data['last100_avg']:.2f}\n"
                 f"Epsilon: {ui_data['epsilon']:.3f}\n"
                 f"Best Reward: {ui_data['best_reward']:.2f}")
    global_win_ratio = (ui_data["wins"] / (ui_data["wins"] + ui_data["losses"]) * 100
                        if (ui_data["wins"] + ui_data["losses"]) > 0 else 0)
    last100_wins = sum(1 for r in recent_rewards if r > 0)
    last100_losses = sum(1 for r in recent_rewards if r < 0)
    last100_total = last100_wins + last100_losses
    last100_win_ratio = (last100_wins / last100_total * 100) if last100_total > 0 else 0
    right_text = (f"Global Wins: {ui_data['wins']}  Losses: {ui_data['losses']}\n"
                  f"Global Win Ratio: {global_win_ratio:.1f}%\n"
                  f"Last 100 Wins: {last100_wins}  Losses: {last100_losses}\n"
                  f"Last 100 Win Ratio: {last100_win_ratio:.0f}%")
    
    header_label.config(text=header_text)
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
    x_min = min(episodes)
    x_max = max(episodes)
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
    for i in range(1, len(episodes)):
        x1 = scale_x(episodes[i-1])
        y1 = scale_y(series1[i-1])
        x2 = scale_x(episodes[i])
        y2 = scale_y(series1[i])
        canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)
    for i in range(1, len(episodes)):
        x1 = scale_x(episodes[i-1])
        y1 = scale_y(series2[i-1])
        x2 = scale_x(episodes[i])
        y2 = scale_y(series2[i])
        canvas.create_line(x1, y1, x2, y2, fill="orange", width=2)
    canvas.create_text(margin + 20, margin - 10, text=label1, fill="blue", font=("Helvetica", 10))
    canvas.create_text(margin + 150, margin - 10, text=label2, fill="orange", font=("Helvetica", 10))

def draw_epsilon_chart(canvas, episodes, epsilons):
    canvas.delete("all")
    w = int(canvas["width"])
    h = int(canvas["height"])
    margin = 40
    if not episodes:
        return
    x_min = episodes[0]
    x_max = episodes[-1]
    y_min = 0
    y_max = 1
    draw_grid(canvas, x_min, x_max, y_min, y_max, margin=margin, num_grid=10)
    
    def scale_x(x):
        return margin + (x - x_min) / (x_max - x_min) * (w - 2 * margin) if x_max != x_min else margin
    def scale_y(y):
        return h - margin - ((y - y_min) / (y_max - y_min)) * (h - 2 * margin) if y_max != y_min else h/2
    x1 = scale_x(episodes[0])
    y1 = scale_y(epsilons[0])
    x2 = scale_x(episodes[-1])
    y2 = scale_y(epsilons[-1])
    canvas.create_line(x1, y1, x2, y2, fill="green", width=2)
    for mark in [0, 0.5, 1]:
        y = scale_y(mark)
        canvas.create_line(margin, y, w - margin, y, fill="gray", dash=(2, 2))
        canvas.create_text(margin - 20, y, text=str(mark), fill="gray", font=("Helvetica", 8))

def get_color_from_probability(prob):
    g = b = int(255 * (1 - prob))
    return f"#{255:02x}{g:02x}{b:02x}"

def draw_action_probability_table(canvas, aggregated_action_data):
    canvas.delete("all")
    w = int(canvas["width"])
    h = int(canvas["height"])
    margin_left = 60
    margin_top = 40
    margin_right = 20
    margin_bottom = 40

    sorted_hand_sums = sorted(aggregated_action_data.keys())
    num_cols = len(sorted_hand_sums)
    num_rows = 4

    cell_width = (w - margin_left - margin_right) / num_cols
    cell_height = (h - margin_top - margin_bottom) / num_rows

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
    for col, hand_sum in enumerate(sorted_hand_sums):
        x = margin_left + col * cell_width + cell_width/2
        canvas.create_text(x, margin_top - 15, text=str(hand_sum), font=("Helvetica", 10))
    canvas.create_text((w+margin_left-margin_right)/2, 15, text="Hand Sum vs Action Probability (Last 100 Episodes)", font=("Helvetica", 12, "bold"))
    action_labels = ["Hit", "Stick", "Double Down", "Split"]
    for row, label in enumerate(action_labels):
        y = margin_top + row * cell_height + cell_height/2
        canvas.create_text(margin_left - 30, y, text=label, font=("Helvetica", 10))

def update_tk_ui(aggregated_action_data):
    update_text()
    draw_chart(canvas_chart, ui_data["episodes"], ui_data["total_avgs"], ui_data["last100_avgs"],
               "Total Moving Avg", "Last 100 Avg")
    draw_epsilon_chart(canvas_epsilon, ui_data["episodes"], ui_data["epsilons"])
    draw_action_probability_table(canvas_action_table, aggregated_action_data)
    root.update_idletasks()

def shape_reward(reward):
    if reward > 0:
        return reward * 2.0
    elif reward < 0:
        return reward * 0.5
    else:
        return reward

def train_agent():
    global EPSILON
    env = AdvancedBlackjackEnv(render_mode=None, natural=True, initial_bankroll=1000, max_rounds=100)
    best_reward = -float('inf')
    total_moving_avg = 0.0
    global recent_rewards
    recent_rewards.clear()
    last_update_episode = 0
    episode_counter = 0
    action_stats_per_episode = []
    
    # Run indefinitely until stop_event is set.
    while not stop_event.is_set():
        while pause_event.is_set() and not stop_event.is_set():
            time.sleep(0.1)
        
        obs, _ = env.reset()
        total_reward = 0
        done = False
        episode_action_stats = {}
        
        while not done:
            phase = obs["phase"]
            state = discretize_state(obs)
            if phase == 0:
                action = 9
            else:
                action = choose_action(state, phase, EPSILON, episode_counter, obs)
                hand_sum = obs["hand_sum"]
                if hand_sum not in episode_action_stats:
                    episode_action_stats[hand_sum] = [0, 0, 0, 0]
                episode_action_stats[hand_sum][action] += 1
            next_obs, reward, done, _, info = env.step(action)
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
        episode_counter += 1
        recent_rewards.append(total_reward)
        window_size = 100 if episode_counter >= 100 else episode_counter
        if len(recent_rewards) > window_size:
            recent_rewards = recent_rewards[-window_size:]
        total_moving_avg = (total_moving_avg * (episode_counter - 1) + total_reward) / episode_counter
        last100_avg = np.mean(recent_rewards)
        best_reward = max(best_reward, total_reward)
        
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
        
        if total_reward > 0:
            ui_data["wins"] += 1
        elif total_reward < 0:
            ui_data["losses"] += 1
        else:
            ui_data["ties"] += 1
        
        interval = get_print_interval(episode_counter)
        if episode_counter % interval == 0:
            ui_data["episode"] = episode_counter
            ui_data["ep_reward"] = total_reward
            ui_data["total_avg"] = total_moving_avg
            ui_data["last100_avg"] = last100_avg
            ui_data["epsilon"] = EPSILON
            ui_data["best_reward"] = best_reward
            ui_data["episodes"].append(episode_counter)
            ui_data["total_avgs"].append(total_moving_avg)
            ui_data["last100_avgs"].append(last100_avg)
            ui_data["epsilons"].append(EPSILON)
            ui_data["best_rewards"].append(best_reward)
            aggregated_action_data = {}
            for ep_stats in action_stats_per_episode[-100:]:
                for hand_sum, counts in ep_stats.items():
                    if hand_sum not in aggregated_action_data:
                        aggregated_action_data[hand_sum] = [0, 0, 0, 0]
                    for i in range(4):
                        aggregated_action_data[hand_sum][i] += counts[i]
            update_tk_ui(aggregated_action_data)
            print(f"Episode {episode_counter} | Reward: {total_reward:.2f} | Total Avg: {total_moving_avg:.2f} | "
                  f"Last100 Avg: {last100_avg:.2f} | Epsilon: {EPSILON:.3f} | Best: {best_reward:.2f}")
    
    print("Training complete.")
    root.quit()

def save_screenshot_and_summary():
    root.update()
    x = root.winfo_rootx()
    y = root.winfo_rooty()
    w = root.winfo_width()
    h = root.winfo_height()
    bbox = (x, y, x + w, y + h)
    screenshot = ImageGrab.grab(bbox)
    now = datetime.now()
    date_str = now.strftime("%m-%d")
    time_str = now.strftime("%H-%M")
    screenshot_filename = os.path.join(results_folder, f"{date_str}_{time_str}_{ui_data['episode']}_{ui_data['last100_avg']:.2f}.png")
    screenshot.save(screenshot_filename)
    print(f"Saved screenshot: {screenshot_filename}")
    
    summary_filename = os.path.join(results_folder, f"{date_str}_{time_str}_{ui_data['episode']}_{ui_data['last100_avg']:.2f}.txt")
    with open(summary_filename, "w") as f:
        f.write(f"Final Episode: {ui_data['episode']}\n")
        f.write(f"Total Moving Avg: {ui_data['total_avg']:.2f}\n")
        f.write(f"Last 100 Avg: {ui_data['last100_avg']:.2f}\n")
        f.write(f"Epsilon: {ui_data['epsilon']:.3f}\n")
        f.write(f"Best Reward: {ui_data['best_reward']:.2f}\n")
        f.write(f"Global Wins: {ui_data['wins']}  Losses: {ui_data['losses']}\n")
    print(f"Saved final summary: {summary_filename}")

def main():
    training_thread = threading.Thread(target=train_agent, daemon=True)
    training_thread.start()
    root.mainloop()

if __name__ == "__main__":
    main()