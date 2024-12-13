"""Neuroevolution model that uses ES-HyperNEAT."""

import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, Optional, override

from neat import Config as NEATConfig
from neat import DefaultGenome
from neat.nn import FeedForwardNetwork, RecurrentNetwork
from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.substrate import Substrate
from termcolor import colored

from neuropoker.extra.config import Config as NeuropokerConfig
from neuropoker.models.neat.hyperneat import CoordinateList
from neuropoker.models.neat.neat import NEATModel
from neuropoker.players.neat import NEATNetwork

DEFAULT_INPUT_SIZE: Final[int] = 73
DEFAULT_OUTPUT_SIZE: Final[int] = 5


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
        input_size: int = DEFAULT_INPUT_SIZE,
        output_size: int = DEFAULT_OUTPUT_SIZE,
    ) -> None:
        """Create a HyperNEAT neuroevolution model.

        Parameters:
            neat_config_file: Path
                The path to the NEAT configuration file.
            es_config_file: Path
                The path to the ES-HyperNEAT configuration file.
            input_size: int
                The number of input nodes.
            output_size: int
                The number of output nodes.
        """
        super().__init__(neat_config_file=neat_config_file)

        self.input_size: Final[int] = input_size  # Number of features
        self.output_size: Final[int] = output_size  # Number of actions

        # Input coordinates
        #
        # TODO: Are these coordinates reasonable? I used them for HW2
        # but we may want something different for our poker features.
        #
        # Evenly spaced between (0, 0) and (0, 1)
        self.input_coordinates: Final[CoordinateList] = [
            (0.0, i / self.input_size) for i in range(self.input_size)
        ]

        # Output coordinates
        #
        # Evenly spaced between (0, 0) and (0, 1)
        self.output_coordinates: Final[CoordinateList] = [
            (0.0, i / self.output_size) for i in range(self.output_size)
        ]

        self.substrate: Final[Substrate] = Substrate(
            self.input_coordinates,
            self.output_coordinates,
        )

        # Load ES-HyperNEAT config
        self.es_config_file: Final[Path] = es_config_file
        self.es_config: Final[Dict[str, Any]] = get_es_config(es_config_file)

        # Curried function for obtaining the network from a genome
        self._network_fn = partial(
            __class__._get_network,
            config=self.config,
            substrate=self.substrate,
            es_config=self.es_config,
        )

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
            input_size=config["input_size"],
            output_size=config["output_size"],
        )

    @override
    def print_config(self) -> None:
        """Print the configuration of this model."""
        print(
            colored(
                "--------------- Model ----------------", color="blue", attrs=["bold"]
            )
        )
        print(colored(f"{'Type':<12}", color="blue") + ": ES-HyperNEAT")
        print(colored(f"{'NEAT config':<12}: ", color="blue") + f"{self.config_file}")
        print(colored(f"{'ES config':<12}: ", color="blue") + f"{self.es_config_file}")
        print()

    @staticmethod
    @override
    def _get_network(
        genome: DefaultGenome,
        config: NEATConfig,
        substrate: Optional[Substrate] = None,
        es_config: Optional[Dict[str, Any]] = None,
    ) -> NEATNetwork:
        """Get the network from a genome, statically.

        Parameters:
            genome: DefaultGenome
                The genome to get the network from.
            config: Dict[str, Any]
                The NEAT configuration.
            substrate: Substrate
                The substrate.
            es_config: Dict[str, Any]
                The ES-HyperNEAT configuration.

        Returns:
            net: NEATNetwork
                The network.
        """
        if substrate is None:
            raise ValueError("Substrate not provided")
        if es_config is None:
            raise ValueError("ES-HyperNEAT config not provided")

        cppn: Final[FeedForwardNetwork] = FeedForwardNetwork.create(genome, config)
        es_net: Final[ESNetwork] = ESNetwork(substrate, cppn, es_config)
        net: Final[RecurrentNetwork] = es_net.create_phenotype_network()

        return net
