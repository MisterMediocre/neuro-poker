"""Functions for managing cards.
"""

import random
from typing import Final, List, Optional

from pypokerengine.engine.card import Card
from pypokerengine.engine.deck import Deck

# Full stack of cards
FULL_RANKS: Final[List[str]] = [
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "T",
    "J",
    "Q",
    "K",
    "A",
]
FULL_SUITS: Final[List[str]] = ["C", "D", "H", "S"]

# Short stack of cards, with all 4 suits
SHORT_RANKS: Final[List[str]] = ["6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SHORT_SUITS: Final[List[str]] = ["C", "D", "H", "S"]

# Short stack of cards, with only 3 suits
SHORTER_RANKS: Final[List[str]] = ["6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SHORTER_SUITS: Final[List[str]] = ["C", "D", "H"]


def get_card_list(
    suits: List[str],
    ranks: List[str],
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
    return [f"{suit}{rank}" for suit in suits for rank in ranks]


def get_card_index(card: str, suits: List[str], ranks: List[str]) -> int:
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
    if len(card) != 2:
        raise ValueError("Card must be a 2-character string.")

    return ranks.index(card[1]) + suits.index(card[0]) * len(ranks)


def get_deck(cards: List[str] = [], seed: Optional[int] = None) -> Deck:
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
    assert len(cards) > 0

    card_ids: Final[List[int]] = [Card.from_str(s).to_id() for s in cards]
    if seed is not None:
        # print("Seed used: ", seed)
        random.seed(seed)  # Ignoring the seed as a test
        random.shuffle(card_ids)
    return Deck(cheat=True, cheat_card_ids=card_ids)
