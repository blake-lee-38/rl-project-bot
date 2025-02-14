import os
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

class BlackjackEnv(gym.Env):
    """
    Modified Blackjack environment with a finite 6-deck shoe and cumulative rewards.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
        )

        self.natural = natural
        self.sab = sab
        self.render_mode = render_mode
        self.count = 0
        
        # 6-deck shoe setup
        self.single_deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        self.num_decks = 6
        self.shoe = []
        self.shuffle_shoe()
        
        self.cumulative_reward = 0  # Track rewards over multiple hands

    def cmp(self, a, b):
        return float(a > b) - float(a < b)

    def shuffle_shoe(self):
        self.shoe = self.single_deck * (4 * self.num_decks)  # 4 suits per deck
        self.count = 0 # Reset count on shuffle
        np.random.shuffle(self.shoe)

    def draw_card(self):
        """Draw a card from the shoe, reshuffling if necessary."""
        if len(self.shoe) < (self.num_decks * 52 * 0.2):  # Reshuffle if <20% cards left
            self.shuffle_shoe()
        return self.shoe.pop()

    def draw_hand(self, dealer):
        card1 = self.draw_card()
        card2 = self.draw_card()
        if card1 == 1 or card1 == 10:
            self.count -= 1
        elif card1 < 7:
            self.count += 1

        # Dont count 2nd card if dealer (Cant be seen)
        if dealer:
            return [card1, card2]

        if card2 == 1 or card2 == 10:
            self.count -= 1
        elif card2 < 7:
            self.count += 1
        return [card1, card2]

    def usable_ace(self, hand):
        return 1 in hand and sum(hand) + 10 <= 21

    def sum_hand(self, hand):
        if self.usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    def is_bust(self, hand):
        return self.sum_hand(hand) > 21

    def score(self, hand):
        return 0 if self.is_bust(hand) else self.sum_hand(hand)

    def is_natural(self, hand):
        return sorted(hand) == [1, 10]


    def _get_obs(self):
        return (self.sum_hand(self.player), self.dealer[0], self.usable_ace(self.player))

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit
            next_card = self.draw_card()
            if next_card == 1 or next_card == 10:
                self.count -= 1
            elif next_card < 7:
                self.count += 1
            self.player.append(next_card)
            if self.is_bust(self.player):
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
        else:  # stick
            terminated = True
            if self.dealer[1] == 1 or self.dealer[1] == 10:
                    self.count -= 1
            elif self.dealer[1] < 7:
                self.count += 1
            while self.sum_hand(self.dealer) < 17:
                next_card = self.draw_card()
                if next_card == 1 or next_card == 10:
                    self.count -= 1
                elif next_card < 7:
                    self.count += 1
                self.dealer.append(next_card)
            reward = self.cmp(self.score(self.player), self.score(self.dealer))
            if self.sab and self.is_natural(self.player) and not self.is_natural(self.dealer):
                reward = 1.0
            elif not self.sab and self.natural and self.is_natural(self.player) and reward == 1.0:
                reward = 1.5
        
        self.cumulative_reward += reward  # Track total reward over multiple hands

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, {'cumulative_reward': self.cumulative_reward, 'count': self.count, 'dealer': self.dealer, 'player': self.player}
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.dealer = self.draw_hand(True)
        self.player = self.draw_hand(False)

        _, dealer_card_value, _ = self._get_obs()

        suits = ["C", "D", "H", "S"]
        self.dealer_top_card_suit = self.np_random.choice(suits)

        if dealer_card_value == 1:
            self.dealer_top_card_value_str = "A"
        elif dealer_card_value == 10:
            self.dealer_top_card_value_str = self.np_random.choice(["J", "Q", "K"])
        else:
            self.dealer_top_card_value_str = str(dealer_card_value)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {'cumulative_reward': self.cumulative_reward, 'count': self.count, 'dealer': self.dealer, 'player': self.player}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        player_sum, dealer_card_value, usable_ace = self._get_obs()
        screen_width, screen_height = 600, 500
        card_img_height = screen_height // 3
        card_img_width = int(card_img_height * 142 / 197)
        spacing = screen_height // 20

        bg_color = (7, 99, 36)
        white = (255, 255, 255)

        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        self.screen.fill(bg_color)

        def get_image(path):
            cwd = os.path.dirname(__file__)
            image = pygame.image.load(os.path.join(cwd, path))
            return image

        def get_font(path, size):
            cwd = os.path.dirname(__file__)
            font = pygame.font.Font(os.path.join(cwd, path), size)
            return font

        small_font = get_font(
            os.path.join("font", "Minecraft.ttf"), screen_height // 15
        )
        dealer_text = small_font.render(
            "Dealer: " + str(dealer_card_value), True, white
        )
        dealer_text_rect = self.screen.blit(dealer_text, (spacing, spacing))

        def scale_card_img(card_img):
            return pygame.transform.scale(card_img, (card_img_width, card_img_height))

        dealer_card_img = scale_card_img(
            get_image(
                os.path.join(
                    "img",
                    f"{self.dealer_top_card_suit}{self.dealer_top_card_value_str}.png",
                )
            )
        )
        dealer_card_rect = self.screen.blit(
            dealer_card_img,
            (
                screen_width // 2 - card_img_width - spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        hidden_card_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
        self.screen.blit(
            hidden_card_img,
            (
                screen_width // 2 + spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        player_text = small_font.render("Player", True, white)
        player_text_rect = self.screen.blit(
            player_text, (spacing, dealer_card_rect.bottom + 1.5 * spacing)
        )

        large_font = get_font(os.path.join("font", "Minecraft.ttf"), screen_height // 6)
        player_sum_text = large_font.render(str(player_sum), True, white)
        player_sum_text_rect = self.screen.blit(
            player_sum_text,
            (
                screen_width // 2 - player_sum_text.get_width() // 2,
                player_text_rect.bottom + spacing,
            ),
        )

        if usable_ace:
            usable_ace_text = small_font.render("usable ace", True, white)
            self.screen.blit(
                usable_ace_text,
                (
                    screen_width // 2 - usable_ace_text.get_width() // 2,
                    player_sum_text_rect.bottom + spacing // 2,
                ),
            )
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        pass
