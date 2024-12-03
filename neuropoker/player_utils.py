"""Utility functions for players."""

from pathlib import Path
from typing import Dict, Final, Optional, Type

from neuropoker.model_utils import load_model
from neuropoker.models.base_model import ModelT
from neuropoker.models.neat_model import NEATModel
from neuropoker.players.base_player import BasePlayer, PlayerT
from neuropoker.players.naive_player import CallPlayer, FoldPlayer, RandomPlayer
from neuropoker.players.neat_player import NEATPlayer

PLAYER_TYPES: Final[Dict[str, Type[BasePlayer]]] = {
    "random": RandomPlayer,
    "fold": FoldPlayer,
    "call": CallPlayer,
    "neat": NEATPlayer,
}


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


def load_player(
    player_type: Type[PlayerT],
    player_name: str,
    model_type: Optional[Type[ModelT]] = None,
    model_file: Optional[Path] = None,
) -> PlayerT:
    """Load a player, based on a definition.

    Parameters:
        player_type: PlayerType
            The type of player to load.
        player_name: str
            The name of the player.
        model_type: Optional[Type[BaseModel]]
            The type of model the player uses, if applicable.
        model_file: Path
            The path to the model file, if applicable.

    Returns:
        player: BasePlayer
            The loaded player.
    """
    # Check that player_type inherits from BasePlayer
    if not issubclass(player_type, BasePlayer):
        raise ValueError(
            f"Player type {player_type} invalid, must inherit from BasePlayer"
        )

    if issubclass(player_type, NEATPlayer):
        # Check that, if player_type uses a model, that the model_type
        # and model_file are provided.
        if model_type is None:
            raise ValueError("Model type must be provided for NEAT player.")
        if model_file is None:
            raise ValueError("Model file must be provided for NEAT player.")
        if not issubclass(model_type, NEATModel):
            raise ValueError("Model type must inherit from NEATModel.")

        # Load the model
        player_model: Final[NEATModel] = load_model(model_type, model_file)

        # Create the player with a NEATModel
        return player_type(player_name, net=player_model.get_best_genome_network())

    # Create the player
    return player_type(player_name)
