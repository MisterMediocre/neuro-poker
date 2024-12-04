"""Utility functions for players."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, Optional, Type

from neuropoker.model_utils import load_model
from neuropoker.models.base_model import BaseModel
from neuropoker.models.neat_model import NEATModel
from neuropoker.players.base_player import BasePlayer
from neuropoker.players.naive_player import CallPlayer, FoldPlayer, RandomPlayer
from neuropoker.players.neat_player import NEATPlayer

PLAYER_TYPES: Final[Dict[str, Type[BasePlayer]]] = {
    "fold": FoldPlayer,
    "call": CallPlayer,
    "random": RandomPlayer,
    "neat": NEATPlayer,
    # Alternative names
    "FoldPlayer": FoldPlayer,
    "CallPlayer": CallPlayer,
    "RandomPlayer": RandomPlayer,
    "NEATPlayer": NEATPlayer,
}


@dataclass
class PlayerDefinition:
    """A class that defines players without instantiating them."""

    player_type: Type[BasePlayer]
    model_type: Optional[Type[BaseModel]] = None
    model_file: Optional[Path] = None

    def load(self, uuid: str) -> BasePlayer:
        """Instantiate an instance of a player based on this definition.

        Parameters:
            definition: PlayerDefinition
                The player definition.

        Returns:
            player: BasePlayer
                The loaded player.
        """
        # Check that player_type inherits from BasePlayer
        if not issubclass(self.player_type, BasePlayer):
            raise ValueError(
                f"Player type {self.player_type} invalid, must inherit from BasePlayer"
            )

        if issubclass(self.player_type, NEATPlayer):
            # Check that, if player_type uses a model, that the model_type
            # and model_file are provided.
            if self.model_type is None:
                raise ValueError("Model type must be provided for NEAT player.")
            if self.model_file is None:
                raise ValueError("Model file must be provided for NEAT player.")
            if not issubclass(self.model_type, NEATModel):
                raise ValueError("Model type must inherit from NEATModel.")

            # Load the model
            player_model: Final[NEATModel] = load_model(
                self.model_type, self.model_file
            )

            # Create the player with a NEATModel
            return self.player_type(uuid, net=player_model.get_best_genome_network())

        # Create the player
        return self.player_type(uuid)


def player_type_from_string(player_type_str: str) -> Type[BasePlayer]:
    """Infer the player type from a string.

    Parameters:
        player_type_str: str
            The string representation of the player type.

    Returns:
        player_type: Type[BasePlayer]
            The player type.
    """
    if player_type_str in PLAYER_TYPES:
        return PLAYER_TYPES[player_type_str]

    raise ValueError(f"Player type {player_type_str} not recognized")
