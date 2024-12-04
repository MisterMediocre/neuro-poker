"""Neuroevolution model that uses HyperNEAT.
"""

from functools import partial
from pathlib import Path
from typing import Final, List, Optional, Tuple, Union, override

from neat import Config as NEATConfig
from neat import DefaultGenome
from neat.nn import FeedForwardNetwork, RecurrentNetwork
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.substrate import Substrate
from termcolor import colored

from neuropoker.config import Config as NeuropokerConfig
from neuropoker.models.neat_model import NEATModel
from neuropoker.players.neat_player import NEATNetwork

DEFAULT_INPUT_SIZE: Final[int] = 73
DEFAULT_OUTPUT_SIZE: Final[int] = 5
DEFAULT_HIDDEN_SIZES: List[int] = [64, 64]
CoordinateList = List[Tuple[float, float]]


class HyperNEATModel(NEATModel):
    """Neuroevolution model that uses HyperNEAT."""

    def __init__(
        self,
        neat_config_file: Path = Path("configs/3p_4s_neat.txt"),
        input_size: int = DEFAULT_INPUT_SIZE,
        output_size: int = DEFAULT_OUTPUT_SIZE,
        hidden_sizes: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """Create a HyperNEAT neuroevolution model.

        Parameters:
            config_file: Path
                The path to the NEAT configuration file.
            input_size: int
                The number of input nodes.
            output_size: int
                The number of output nodes.
            hidden_sizes: int | List[int]
                The number of nodes in each layer.
                hidden1 or [hidden1, hidden2, ..., hiddenN] or []
        """
        super().__init__(neat_config_file=neat_config_file)

        self.hidden_sizes: List[int]
        match hidden_sizes:
            case list():
                self.hidden_sizes = hidden_sizes
            case None:
                self.hidden_sizes = DEFAULT_HIDDEN_SIZES
            case _:
                raise ValueError(f"Ill-defined layer_sizes: {hidden_sizes}")

        # TODO: Don't hardcode input and output dimensionality
        self.input_size: Final[int] = input_size  # Number of features
        self.output_size: Final[int] = output_size

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

        # Hidden coordinates
        #
        # For each layer, evenly spaced between (0, 0) and (0, 1)
        self.hidden_coordinates: Final[List[CoordinateList]] = [
            [(0.0, i / size) for i in range(size)] for size in self.hidden_sizes
        ]
        self.activations: Final[int] = len(self.hidden_sizes) + 2
        self.substrate: Final[Substrate] = Substrate(
            self.input_coordinates,
            self.output_coordinates,
            self.hidden_coordinates,
        )

        # Curried function for obtaining the network from a genome
        self._network_fn = partial(
            __class__._get_network,
            config=self.config,
            substrate=self.substrate,
        )

    @classmethod
    @override
    def from_config(cls, config: NeuropokerConfig) -> "HyperNEATModel":
        """Create a HyperNEAT model from a configuration.

        Parameters:
            config: NeuropokerConfig
                The configuration to use.

        Returns:
            model: HyperNEATModel
                The HyperNEAT model.
        """
        config = config["model"]

        if config["type"] != "hyperneat":
            raise ValueError(f"Config has wrong model type: {config['type']}")

        return HyperNEATModel(
            neat_config_file=config["neat_config_file"],
            input_size=config["input_size"],
            output_size=config["output_size"],
            hidden_sizes=config["hidden_sizes"],
        )

    @override
    def print_config(self) -> None:
        """Print the configuration of this model."""

        print(
            colored(
                "--------------- Model ----------------", color="blue", attrs=["bold"]
            )
        )
        print(colored(f"{'Type':<12}: ", color="blue") + "HyperNEAT")
        print(colored(f"{'NEAT config':<12}: ", color="blue") + f"{self.config_file}")
        print(colored(f"{'Input size':<12}: ", color="blue") + f"{self.input_size}")
        print(colored(f"{'Output size':<12}: ", color="blue") + f"{self.output_size}")
        print(colored(f"{'Hidden sizes':<12}: ", color="blue") + f"{self.hidden_sizes}")
        print()

    @staticmethod
    @override
    def _get_network(
        genome: DefaultGenome,
        config: NEATConfig,
        substrate: Optional[Substrate] = None,
    ) -> NEATNetwork:
        """Get the network from a genome, statically.

        Parameters:
            genome: DefaultGenome
                The genome to get the network from.
            config: NEATConfig
                The NEAT configuration.
            substrate: Substrate
                The substrate.

        Returns:
            net: NEATNetwork
                The network.
        """
        cppn: Final[FeedForwardNetwork] = FeedForwardNetwork.create(genome, config)
        net: Final[RecurrentNetwork] = create_phenotype_network(cppn, substrate)
        return net
