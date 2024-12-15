"""Utility functions for players."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, Type

from stable_baselines3.ppo.ppo import PPO
from termcolor import colored

from neuropoker.models.base import BaseModel
from neuropoker.models.neat.neat import NEATModel
from neuropoker.models.utils import load_model
from neuropoker.players.base import BasePlayer
from neuropoker.players.naive import CallPlayer, FoldPlayer, RandomPlayer
from neuropoker.players.neat import NEATPlayer
from neuropoker.players.ppo import PPOPlayer

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
    model_type: Type[BaseModel] | None = None
    model_file: str | Path | None = None

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

        if issubclass(self.player_type, PPOPlayer):
            if self.model_file is None:
                raise ValueError("Model file must be provided for PPO player.")

            return PPOPlayer.from_model_file(self.model_file, uuid)

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


def load_ppo_player(
    model_path: str | Path | None, uuid: str, verbose: bool = False
) -> PPOPlayer | CallPlayer:
    """Attempt to load a PPO player player from a PPO model file.

    Parameters:
        model_path: str | Path | None
            The path to the model file.
        uuid: str
            The UUID of the player.
        verbose: bool
            Whether to print verbose messages.

    Returns:
        player: PPOPlayer | CallPlayer
            The loaded player.

    By default, it tries to load a PPOPlayer from the path proivded
    by <model_path>. If <model_path> is not provided or does not exist,
    it returns a CallPlayer.
    """
    if model_path is not None and Path(model_path).with_suffix(".zip").exists():
        if verbose:
            print(
                colored("[load_model_player]", color="blue")
                + f" Model path {model_path} found, returning PPOPlayer"
            )
        model = PPO.load(model_path)
        return PPOPlayer(model, uuid)

    if verbose:
        print(
            colored("[load_model_player]", color="blue")
            + f" Model path {model_path} not found, returning CallPlayer"
        )
    return CallPlayer(uuid)
