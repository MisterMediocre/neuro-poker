"""Neuroevolution model that uses ES-HyperNEAT.
"""

import json
from pathlib import Path
from typing import Any, Dict, Final, override

from neat import DefaultGenome
from neat.nn import FeedForwardNetwork, RecurrentNetwork
from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.substrate import Substrate
from termcolor import colored

from neuropoker.config import Config as NeuropokerConfig
from neuropoker.models.hyperneat_model import CoordinateList
from neuropoker.models.neat_model import NEATModel
from neuropoker.players.neat_player import NEATNetwork


def get_es_config(es_config_file: Path) -> Dict[str, Any]:
    """Get the ES-HyperNEAT configuration.

    Parameters:
        es_config_file: Dict[str, Any] | Path | None
            The path to the ES-HyperNEAT configuration file.

    Returns:
        es_config: Dict[str, Any]
            The ES-HyperNEAT configuration.
    """
    with open(es_config_file, "rt", encoding="utf-8") as f:
        return json.load(f)


class ESHyperNEATModel(NEATModel):
    """Neuroevolution model that uses ES-HyperNEAT."""

    def __init__(
        self,
        neat_config_file: Path = Path("config/3p_4s_neat.txt"),
        es_config_file: Path = Path("config/3p_4s_es.json"),
    ) -> None:
        """Create a HyperNEAT neuroevolution model.

        Parameters:
            neat_config_file: Path
                The path to the NEAT configuration file.
            es_config_file: Path
                The path to the ES-HyperNEAT configuration file.
        """
        super().__init__(neat_config_file=neat_config_file)

        # TODO: Don't hardcode input and output dimensionality
        self.input_dims: Final[int] = 80  # Number of features
        self.output_dims: Final[int] = 6  # Number of actions

        # Input coordinates
        #
        # TODO: Are these coordinates reasonable? I used them for HW2
        # but we may want something different for our poker features.
        #
        # Evenly spaced between (0, 0) and (0, 1)
        self.input_coordinates: Final[CoordinateList] = [
            (0.0, i / self.input_dims) for i in range(self.input_dims)
        ]

        # Output coordinates
        #
        # Evenly spaced between (0, 0) and (0, 1)
        self.output_coordinates: Final[CoordinateList] = [
            (0.0, i / self.output_dims) for i in range(self.output_dims)
        ]

        self.substrate: Final[Substrate] = Substrate(
            self.input_coordinates,
            self.output_coordinates,
        )

        # Load ES-HyperNEAT config
        self.es_config_file: Final[Path] = es_config_file
        self.es_config: Final[Dict[str, Any]] = get_es_config(es_config_file)

    @classmethod
    @override
    def from_config(cls, config: NeuropokerConfig) -> "ESHyperNEATModel":
        """Create a ES-HyperNEAT model from a neuropoker configuration.

        Parameters:
            config: NeuropokerConfig
                The neuropoker configuration.

        Returns:
            model: ESHyperNEATModel
                The ES-HyperNEAT model.
        """
        config = config["model"]

        if config["type"] != "es-hyperneat":
            raise ValueError(f"Config has wrong model type: {config['type']}")

        return ESHyperNEATModel(
            neat_config_file=config["neat_config_file"],
            es_config_file=config["es_config_file"],
        )

    @override
    def print_config(self) -> None:
        """Print the configuration of this model."""
        print(colored("Model:", color="blue", attrs=["bold"]))
        print(
            colored(f"{'    Type':<20q}", color="blue", attrs=["bold"])
            + colored(": HyperNEAT", color="blue")
        )
        print(
            colored(f"{'    NEAT config':<20}", color="blue", attrs=["bold"])
            + colored(f": {self.config_file}", color="blue")
        )
        print(
            colored(f"{'    ES config':<20}", color="blue", attrs=["bold"])
            + colored(f": {self.es_config_file}", color="blue")
        )
        print()

    @override
    def get_network(self, genome: DefaultGenome) -> NEATNetwork:
        """Get the network from a genome.

        Parameters:
            genome: DefaultGenome
                The genome to get the network from.

        Returns:
            net: NEATNetwork
                The network.
        """
        cppn: Final[FeedForwardNetwork] = FeedForwardNetwork.create(genome, self.config)
        es_net: Final[ESNetwork] = ESNetwork(self.substrate, cppn, self.es_config)
        net: Final[RecurrentNetwork] = es_net.create_phenotype_network()

        return net
