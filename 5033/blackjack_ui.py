#!/usr/bin/env python3
"""

Blackjack UI

This script creates a graphical user interface (UI) for the Advanced Blackjack Environment.

It allows a player to:
  - In the betting phase: press number keys (1–0) to place a bet (1–10).
  - In the playing phase: use keys:
         H: Hit | S: Stick | D: Double Down | P: Split

During a round's resolution, the dealer's full hand is shown along with a win/loss/tie indicator next to each player hand.

The current active player hand is highlighted with a yellow border.

"""

import pygame
import sys
from advanced_blackjack_env import AdvancedBlackjackEnv

# Initialize pygame.
pygame.init()

# Screen dimensions.
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Blackjack")

# Colors.
WHITE = (255, 255, 255)
GREEN = (7, 99, 36)
YELLOW = (255, 255, 0)

# Load custom font.
try:
    font_path = "font/Minecraft.ttf"
    font = pygame.font.Font(font_path, 24)
    big_font = pygame.font.Font(font_path, 36)
except Exception as e:
    font = pygame.font.SysFont("Arial", 24)
    big_font = pygame.font.SysFont("Arial", 36)

# Card image dimensions.
CARD_IMG_HEIGHT = SCREEN_HEIGHT // 4
CARD_IMG_WIDTH = int(CARD_IMG_HEIGHT * 142 / 197)
CARD_SPACING = 10

def load_card_image(card_str: str) -> pygame.Surface:
    """
    Load a card image from the 'img' folder.
    card_str is expected to be in the format "H2", "CA", etc.
    If the image is not found, a placeholder is returned.
    """
    image_path = f"img/{card_str}.png"
    try:
        image = pygame.image.load(image_path)
        image = pygame.transform.scale(image, (CARD_IMG_WIDTH, CARD_IMG_HEIGHT))
    except Exception as e:
        image = pygame.Surface((CARD_IMG_WIDTH, CARD_IMG_HEIGHT))
        image.fill(WHITE)
    return image

def draw_text(surface, text, pos, font, color=WHITE):
    """Helper function to draw text on the surface."""
    text_obj = font.render(text, True, color)
    surface.blit(text_obj, pos)

def draw_ui(env: AdvancedBlackjackEnv, screen, round_result=None, outcomes=None):
    """Render the current game state using images and custom fonts."""
    screen.fill(GREEN)
    
    # Draw round and bankroll.
    draw_text(screen, f"Round: {env.current_round+1}/{env.max_rounds}", (20, 20), font)
    draw_text(screen, f"Bankroll: {env.bankroll}", (20, 50), font)
    
    obs = env._get_obs()
    
    if obs["phase"] == 0:
        draw_text(screen, "Betting Phase: Press number key (1-0) to bet (1-10)", (20, 100), font)
    else:
        draw_text(screen, "Playing Phase", (20, 100), font)
        # Draw all player hands.
        y_offset = 150
        for idx, hand in enumerate(env.player_hands):
            hand_strs = [card[0] + card[1] for card in hand["cards"]]
            hand_sum = env.sum_hand(hand["cards"])
            x_offset = 20
            hand_info = f"Hand {idx+1} (Bet: {hand['bet']}): Sum = {hand_sum}"
            draw_text(screen, hand_info, (x_offset, y_offset - 30), font)
            
            # Calculate width of the hand image block.
            num_cards = len(hand_strs)
            width = num_cards * CARD_IMG_WIDTH + (num_cards - 1) * CARD_SPACING
            # If this is the current hand, draw a yellow rectangle around it.
            if obs["phase"] == 1 and idx == env.current_hand_index:
                rect = pygame.Rect(x_offset - 5, y_offset - 5, width + 10, CARD_IMG_HEIGHT + 10)
                pygame.draw.rect(screen, YELLOW, rect, 3)
            
            # Draw each card image.
            for i, card_str in enumerate(hand_strs):
                card_img = load_card_image(card_str)
                screen.blit(card_img, (x_offset + i * (CARD_IMG_WIDTH + CARD_SPACING), y_offset))
            
            # If round outcomes are available, show win/loss/tie indicator.
            if outcomes is not None and idx < len(outcomes):
                outcome_text = outcomes[idx]
                draw_text(screen, outcome_text, (x_offset + width + 10, y_offset + CARD_IMG_HEIGHT // 2 - 12), font, color=YELLOW)
            y_offset += CARD_IMG_HEIGHT + 40
        
        # Draw dealer's visible card.
        if env.dealer:
            dealer_str = env.dealer[0][0] + env.dealer[0][1]
            dealer_img = load_card_image(dealer_str)
            draw_text(screen, "Dealer's Visible Card:", (20, y_offset), font)
            screen.blit(dealer_img, (20, y_offset + 30))
        
        # If round result is available, display dealer's full hand and overall round result.
        if round_result is not None:
            y_offset += CARD_IMG_HEIGHT + 80
            dealer_hand_str = " ".join([card[0] + card[1] for card in env.dealer])
            dealer_sum = env.sum_hand(env.dealer)
            draw_text(screen, f"Dealer's Hand: {dealer_hand_str} (Sum: {dealer_sum})", (20, y_offset), font)
            draw_text(screen, f"Overall Round Result: {round_result}", (20, y_offset + 40), big_font)
    
    pygame.display.flip()

def compute_outcomes(env: AdvancedBlackjackEnv) -> list:
    """
    Compute outcome (Win, Loss, or Tie) for each player hand based on dealer's hand.
    Uses the same logic as the environment.
    """
    outcomes = []
    dealer_bust = env.is_bust(env.dealer)
    dealer_score = env.sum_hand(env.dealer) if not dealer_bust else 0
    for hand in env.player_hands:
        if env.is_bust(hand["cards"]):
            outcomes.append("Loss")
        else:
            player_score = env.sum_hand(hand["cards"])
            if dealer_bust:
                outcomes.append("Win")
            else:
                if player_score > dealer_score:
                    outcomes.append("Win")
                elif player_score < dealer_score:
                    outcomes.append("Loss")
                else:
                    outcomes.append("Tie")
    return outcomes

def main():
    # Set round cap to 20.
    env = AdvancedBlackjackEnv(render_mode="human", natural=True, initial_bankroll=1000, max_rounds=20)
    obs, info = env.reset()
    clock = pygame.time.Clock()
    running = True
    round_result = None  # Holds overall round result.
    outcomes = None      # List of outcomes per hand.
    
    while running:
        draw_ui(env, screen, round_result, outcomes)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                env.close()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if obs["phase"] == 0:
                    # Betting phase: use number keys.
                    if event.key in [pygame.K_1, pygame.K_KP1]:
                        action = 0
                    elif event.key in [pygame.K_2, pygame.K_KP2]:
                        action = 1
                    elif event.key in [pygame.K_3, pygame.K_KP3]:
                        action = 2
                    elif event.key in [pygame.K_4, pygame.K_KP4]:
                        action = 3
                    elif event.key in [pygame.K_5, pygame.K_KP5]:
                        action = 4
                    elif event.key in [pygame.K_6, pygame.K_KP6]:
                        action = 5
                    elif event.key in [pygame.K_7, pygame.K_KP7]:
                        action = 6
                    elif event.key in [pygame.K_8, pygame.K_KP8]:
                        action = 7
                    elif event.key in [pygame.K_9, pygame.K_KP9]:
                        action = 8
                    elif event.key in [pygame.K_0, pygame.K_KP0]:
                        action = 9
                    else:
                        action = None
                    if action is not None:
                        obs, reward, done, truncated, info = env.step(action)
                        if "round_reward" in info:
                            round_result = info["round_reward"]
                            outcomes = compute_outcomes(env)
                            draw_ui(env, screen, round_result, outcomes)
                            pygame.display.flip()
                            pygame.time.wait(3000)  # Pause 3 seconds.
                            round_result = None
                            outcomes = None
                        if done:
                            running = False
                else:
                    # Playing phase: use H, S, D, P.
                    if event.key == pygame.K_h:
                        action = 0
                    elif event.key == pygame.K_s:
                        action = 1
                    elif event.key == pygame.K_d:
                        action = 2
                    elif event.key == pygame.K_p:
                        action = 3
                    else:
                        action = None
                    if action is not None:
                        obs, reward, done, truncated, info = env.step(action)
                        if "round_reward" in info:
                            round_result = info["round_reward"]
                            outcomes = compute_outcomes(env)
                            draw_ui(env, screen, round_result, outcomes)
                            pygame.display.flip()
                            pygame.time.wait(3000)  # Pause 3 seconds.
                            round_result = None
                            outcomes = None
                        if done:
                            running = False
        clock.tick(30)

if __name__ == "__main__":
    main()