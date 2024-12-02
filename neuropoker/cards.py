"""Functions for managing cards.
"""

import random
import time
from typing import Final, List, Optional

from pypokerengine.engine.card import Card
from pypokerengine.engine.deck import Deck

SHORT_SUITS: Final[List[str]] = ["C", "D", "H", "S"]
SHORT_RANKS: Final[List[str]] = ["6", "7", "8", "9", "T", "J", "Q", "K", "A"]


def get_card_list(
    suits: Optional[List[str]] = None,
    ranks: Optional[List[str]] = None,
) -> List[str]:
    """Get the list of cards in a full deck of a given set of suits and ranks.

    Parameters:
        suits: List[str]
            The list of suits.
        ranks: List[str]
            The list of ranks.

    Returns:
        cards: List[str]
            The list of cards in the deck.

    Suits:
        C: Clubs,
        D: Diamonds,
        H: Hearts,
        S: Spades

    Ranks:
        2-9
        T: 10
        J: Jack
        Q: Queen
        K: King
        A: Ace
    """
    if suits is None:
        suits = SHORT_SUITS
    if ranks is None:
        ranks = SHORT_RANKS

    return [f"{suite}{rank}" for suite in suits for rank in ranks]


def get_card_index(
    card: str, ranks: Optional[List[str]] = None, suits: Optional[List[str]] = None
) -> int:
    """Get the index of a card in a deck of cards.

    Parameters:
        card: str
            The card to get the index of.
        ranks: List[str]
            The list of ranks.
        suits: List[str]
            The list of suits.

    Returns:
        index: int
            The index of the card.
    """
    if ranks is None:
        ranks = SHORT_RANKS
    if suits is None:
        suits = SHORT_SUITS
    if len(card) != 2:
        raise ValueError("Card must be a 2-character string.")

    return ranks.index(card[1]) + suits.index(card[0]) * len(ranks)


def get_deck(cards: List[str] = get_card_list(), seed: Optional[int] = None) -> Deck:
    """Get the deck represented by a list of cards, randomly shuffled.

    Parameters:
        cards: List[str]
            The list of cards to use in the deck.
        seed: int | None
            The random seed to use.

    Returns:
        deck: Deck
            The deck of cards.
    """
    card_ids: Final[List[int]] = [Card.from_str(s).to_id() for s in cards]
    if seed is not None:
        # print("Seed used: ", seed)
        random.seed(seed) # Ignoring the seed as a test
        random.shuffle(card_ids)
    return Deck(cheat=True, cheat_card_ids=card_ids)
