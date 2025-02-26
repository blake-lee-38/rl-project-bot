#!/usr/bin/env python3
"""
Advanced Blackjack Environment

This environment implements a blackjack game with the following features:
  • Betting phase: The player chooses a bet (1–10 units). The bet is recorded but not deducted upfront.
  • Playing phase: The player may take actions on their hand(s):
      - Hit: Draw another card.
      - Stick: End the current hand.
      - Double Down: Only allowed on a 2‑card hand if available funds (bankroll minus committed bets) are sufficient; doubles the bet, draws one card, and ends that hand.
      - Split: Only allowed on a 2‑card hand if both cards have the same rank and available funds are sufficient; splits the hand into two hands.
  • Round Resolution: After all player hands are finished, the dealer plays. For each hand:
      - Win: Outcome = +bet (or +1.5×bet if the hand sums exactly to 21)
      - Tie: Outcome = 0
      - Loss (or bust): Outcome = –bet
    The bankroll is updated only at round resolution.
    
Cards are represented as tuples (suit, rank) where suit is one of 'C', 'D', 'H', 'S' and rank is one of:
  '2','3','4','5','6','7','8','9','T','J','Q','K','A'

This version includes extra observation keys for the playing phase:
  • can_double: True if a double-down move is allowed.
  • can_split: True if a split move is allowed.
  
Additionally, if the player's hand sums exactly to 21 (and they win the round), the win is rewarded at 1.5× the bet.
"""

import random
from typing import Optional, Dict, Any, List, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Type alias for a card.
Card = Tuple[str, str]

class AdvancedBlackjackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode: Optional[str] = None, natural: bool = False,
                 initial_bankroll: int = 1000, max_rounds: int = 20):
        self.render_mode = render_mode
        self.natural = natural
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.max_rounds = max_rounds
        self.current_round = 0

        # Build a standard deck.
        self.suits = ['C', 'D', 'H', 'S']
        self.ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
        self.single_deck = [(s, r) for s in self.suits for r in self.ranks]
        self.num_decks = 6
        self.shoe: List[Card] = []
        self.shuffle_shoe()

        # Game phases: "bet" or "play"
        self.phase = "bet"

        # Player hands: list of dictionaries. Each dictionary stores:
        #   'cards': List[Card]
        #   'bet': int
        #   'doubled': bool
        #   'finished': bool
        self.player_hands: List[Dict[str, Any]] = []
        self.current_hand_index = 0  # pointer to the current hand

        # Dealer hand (list of Card)
        self.dealer: List[Card] = []

        # Action spaces
        self.max_bet = 10
        self.bet_action_space = spaces.Discrete(self.max_bet)  # actions 0-9 correspond to bet=action+1
        self.play_action_space = spaces.Discrete(4)  # 0: Hit, 1: Stick, 2: Double Down, 3: Split
    
    def shuffle_shoe(self):
        """Construct and shuffle the shoe using num_decks decks."""
        self.shoe = self.single_deck * self.num_decks
        random.shuffle(self.shoe)
    
    def draw_card(self) -> Card:
        """Draw a card from the shoe; reshuffle if needed."""
        if len(self.shoe) < 52 * self.num_decks * 0.2:
            self.shuffle_shoe()
        return self.shoe.pop()
    
    def card_value(self, card: Card) -> int:
        """Return the blackjack value of a card. Face cards count as 10; Ace counts as 1 (can be 11 via sum_hand)."""
        rank = card[1]
        if rank in ['T','J','Q','K']:
            return 10
        elif rank == 'A':
            return 1
        else:
            return int(rank)
    
    def sum_hand(self, cards: List[Card]) -> int:
        """Return the best score for a hand of cards, counting an Ace as 11 if it does not bust."""
        total = sum(self.card_value(card) for card in cards)
        num_aces = sum(1 for card in cards if card[1] == 'A')
        if num_aces and total + 10 <= 21:
            return total + 10
        return total
    
    def usable_ace(self, cards: List[Card]) -> bool:
        """Return True if the hand has an ace that can be counted as 11."""
        total = sum(self.card_value(card) for card in cards)
        num_aces = sum(1 for card in cards if card[1] == 'A')
        return num_aces > 0 and total + 10 <= 21
    
    def is_bust(self, cards: List[Card]) -> bool:
        """Return True if the hand is bust (sum > 21)."""
        return self.sum_hand(cards) > 21
    
    def committed_bet(self) -> int:
        """Return the total bet committed in the current round."""
        return sum(hand["bet"] for hand in self.player_hands)
    
    def reset_round(self) -> Dict[str, Any]:
        """Reset state for a new round while preserving bankroll."""
        self.phase = "bet"
        self.player_hands = []
        self.current_hand_index = 0
        self.dealer = []
        return self._get_obs()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> (Dict[str, Any], Dict):
        """Reset the environment for a new episode."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.bankroll = self.initial_bankroll
        self.current_round = 0
        return self.reset_round(), {"bankroll": self.bankroll, "round": self.current_round}
    
    def _deal_initial_cards(self):
        """Deal initial cards to player and dealer."""
        initial_hand = {
            "cards": [self.draw_card(), self.draw_card()],
            "bet": self.current_bet,  # set during betting phase
            "doubled": False,
            "finished": False
        }
        self.player_hands.append(initial_hand)
        self.dealer = [self.draw_card(), self.draw_card()]
    
    def _get_obs(self) -> Dict[str, Any]:
        """
        Return an observation dictionary.
        'phase': 0 for betting, 1 for playing.
        In playing phase, current_hand is a list of card strings (e.g. "H2") for the current hand.
        Additionally, two flags are added: 'can_double' and 'can_split'.
        """
        if self.phase == "bet":
            return {
                "phase": 0,
                "bankroll": self.bankroll,
                "current_bet": 0,
                "current_hand": [],
                "hand_sum": 0,
                "usable_ace": False,
                "dealer_card": None,
                "hand_index": 0,
                "total_hands": 0
            }
        else:
            current_hand = self.player_hands[self.current_hand_index]["cards"]
            hand_strs = [card[0] + card[1] for card in current_hand]
            obs = {
                "phase": 1,
                "bankroll": self.bankroll,
                "current_bet": self.player_hands[self.current_hand_index]["bet"],
                "current_hand": hand_strs,
                "hand_sum": self.sum_hand(current_hand),
                "usable_ace": self.usable_ace(current_hand),
                "dealer_card": self.dealer[0][0] + self.dealer[0][1] if self.dealer else None,
                "hand_index": self.current_hand_index,
                "total_hands": len(self.player_hands)
            }
            # Add flags for double down and split.
            current_hand_data = self.player_hands[self.current_hand_index]
            cards = current_hand_data["cards"]
            can_double = (len(cards) == 2 and not current_hand_data["doubled"] and
                          (self.bankroll - self.committed_bet() >= current_hand_data["bet"]))
            can_split = (len(cards) == 2 and cards[0][1] == cards[1][1] and
                         (self.bankroll - self.committed_bet() >= current_hand_data["bet"]))
            obs["can_double"] = can_double
            obs["can_split"] = can_split
            return obs
    
    def _all_hands_finished(self) -> bool:
        """Return True if all player hands are finished."""
        return all(hand["finished"] for hand in self.player_hands)
    
    def _move_to_next_hand(self):
        """Advance pointer to the next unfinished hand, if any."""
        for idx, hand in enumerate(self.player_hands):
            if not hand["finished"]:
                self.current_hand_index = idx
                return
    
    def _play_dealer(self):
        """Dealer draws until reaching at least 17."""
        while self.sum_hand(self.dealer) < 17:
            self.dealer.append(self.draw_card())
    
    def _resolve_round(self) -> int:
        """
        After all player hands are finished, let the dealer play and resolve each hand.
        For each hand:
          - If busted: outcome = -bet
          - Else, if dealer busts: outcome = +bet
          - Else, if player's sum > dealer's sum: outcome = +bet
          - Else, if player's sum < dealer's sum: outcome = -bet
          - Tie: outcome = 0
        Additionally, if the player's hand sums exactly to 21 (and they win), the outcome is
        increased to 1.5× the bet.
        Update bankroll with the total outcome and return the round reward.
        """
        self._play_dealer()
        total_reward = 0
        dealer_score = self.sum_hand(self.dealer) if not self.is_bust(self.dealer) else 0
        for hand in self.player_hands:
            player_cards = hand["cards"]
            if self.is_bust(player_cards):
                outcome = -hand["bet"]
            else:
                player_score = self.sum_hand(player_cards)
                if self.is_bust(self.dealer):
                    outcome = hand["bet"]
                else:
                    if player_score > dealer_score:
                        outcome = hand["bet"]
                    elif player_score < dealer_score:
                        outcome = -hand["bet"]
                    else:
                        outcome = 0
                # Apply bonus: if player's hand sums exactly to 21 and outcome is a win, reward 1.5× bet.
                if not self.is_bust(player_cards) and player_score == 21 and outcome > 0:
                    outcome = hand["bet"] * 1.5
            total_reward += outcome
            hand["outcome"] = outcome
        self.bankroll += total_reward
        return total_reward
    
    def step(self, action: int) -> (Dict[str, Any], int, bool, bool, Dict):
        """
        Execute an action.
        
        In Betting Phase (phase "bet"):
          - Action (0...9) sets bet = action+1 (but cannot exceed bankroll).
          - Then initial cards are dealt and phase transitions to "play".
        
        In Playing Phase (phase "play") for the current hand:
          - 0: Hit – add a card.
          - 1: Stick – mark current hand as finished.
          - 2: Double Down – if allowed, double the bet, draw one card, and finish the hand.
          - 3: Split – if allowed, split the hand into two separate hands.
        
        After each action, if all player hands are finished, the dealer plays and the round is resolved.
        The round reward is returned (and bankroll is updated).
        An episode ends if current_round >= max_rounds or bankroll <= 0.
        """
        info = {}
        reward = 0
        done = False

        if self.phase == "bet":
            bet = min(action + 1, self.bankroll)
            self.current_bet = bet
            self._deal_initial_cards()
            self.player_hands[0]["bet"] = bet
            self.phase = "play"
            self.current_hand_index = 0
            obs = self._get_obs()
            return obs, 0, done, False, info

        elif self.phase == "play":
            current_hand = self.player_hands[self.current_hand_index]
            if current_hand["finished"]:
                self._move_to_next_hand()
                obs = self._get_obs()
                return obs, 0, done, False, info

            if action == 0:  # Hit
                current_hand["cards"].append(self.draw_card())
                if self.is_bust(current_hand["cards"]):
                    current_hand["finished"] = True
            elif action == 1:  # Stick
                current_hand["finished"] = True
            elif action == 2:  # Double Down
                if self._get_obs().get("can_double", False):
                    current_hand["doubled"] = True
                    current_hand["bet"] *= 2
                    current_hand["cards"].append(self.draw_card())
                    current_hand["finished"] = True
                else:
                    pass  # Invalid move; do nothing.
            elif action == 3:  # Split
                if self._get_obs().get("can_split", False):
                    card = current_hand["cards"][0]
                    new_hand1 = {
                        "cards": [card, self.draw_card()],
                        "bet": current_hand["bet"],
                        "doubled": False,
                        "finished": False
                    }
                    new_hand2 = {
                        "cards": [current_hand["cards"][1], self.draw_card()],
                        "bet": current_hand["bet"],
                        "doubled": False,
                        "finished": False
                    }
                    self.player_hands[self.current_hand_index] = new_hand1
                    self.player_hands.insert(self.current_hand_index + 1, new_hand2)
                else:
                    pass  # Invalid move; do nothing.

            if self._all_hands_finished():
                round_reward = self._resolve_round()
                reward += round_reward
                self.current_round += 1
                info["round_reward"] = round_reward
                info["dealer_hand"] = [card[0] + card[1] for card in self.dealer]
                self.phase = "bet"
                obs = self.reset_round()
                return obs, reward, (self.current_round >= self.max_rounds or self.bankroll <= 0), False, info
            else:
                self._move_to_next_hand()
                obs = self._get_obs()
                return obs, 0, done, False, info

        obs = self._get_obs()
        return obs, reward, done, False, info

    def render(self, mode: Optional[str] = None):
        if self.render_mode == "human":
            print("-----")
            print(f"Round: {self.current_round+1}/{self.max_rounds} | Bankroll: {self.bankroll}")
            if self.phase == "bet":
                print("Phase: Betting")
            else:
                print(f"Phase: Playing (Hand {self.current_hand_index+1} of {len(self.player_hands)})")
                current_hand = self.player_hands[self.current_hand_index]["cards"]
                hand_str = " ".join([card[0]+card[1] for card in current_hand])
                print(f"Your Hand: {hand_str} | Sum: {self.sum_hand(current_hand)} | Usable Ace: {self.usable_ace(current_hand)}")
                if self.dealer:
                    print(f"Dealer's Visible Card: {self.dealer[0][0]+self.dealer[0][1]}")
        else:
            pass
    
    def close(self):
        pass

# For testing via command line.
if __name__ == "__main__":
    env = AdvancedBlackjackEnv(render_mode="human", natural=True, initial_bankroll=1000, max_rounds=20)
    obs, info = env.reset()
    done = False
    print("Initial Observation:", obs)
    while not done:
        if obs["phase"] == 0:
            bet = int(input("Enter your bet (1-10): "))
            action = bet - 1
        else:
            print("Choose action: 0: Hit, 1: Stick, 2: Double Down, 3: Split")
            action = int(input("Enter action: "))
        obs, reward, done, truncated, info = env.step(action)
        print("Observation:", obs, "Reward:", reward)
        env.render()
        if "round_reward" in info:
            print("Round result:", info["round_reward"])
            print("Dealer's hand:", info.get("dealer_hand", []))
    env.close()