"""Neuroevolution model that uses HyperNEAT.
"""

from pathlib import Path
from typing import Final, List, Optional, Tuple, Union

from neat import DefaultGenome
from neat.nn import FeedForwardNetwork, RecurrentNetwork
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.substrate import Substrate

from neuropoker.models.neat import NEATEvolution, NEATModel, NEATNetwork

DEFAULT_HIDDEN_SIZES: List[int] = [64, 64]
CoordinateList = List[Tuple[float, float]]


class HyperNEATModel(NEATModel):
    """Neuroevolution model that uses HyperNEAT."""

    def __init__(
        self,
        evolution: Optional[NEATEvolution] = None,
        config_file: Path = Path("config-feedforward.txt"),
        hidden_sizes: Union[int, List[int]] = DEFAULT_HIDDEN_SIZES,
    ) -> None:
        """Create a HyperNEAT neuroevolution model.

        Parameters:
            evolution: NEATEvolution | None
                The initial evolution to evolve from, if any.
            config_file: Path
                The path to the NEAT configuration file.
            hidden_sizes: int | List[int]
                The number of hidden nodes in each layer.
        """
        super().__init__(evolution=evolution, config_file=config_file)

        self.hidden_sizes: List[int]
        match hidden_sizes:
            case int():
                self.hidden_sizes = [hidden_sizes]
            case list():
                self.hidden_sizes = hidden_sizes
            case _:
                self.hidden_sizes = DEFAULT_HIDDEN_SIZES

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
        net: Final[RecurrentNetwork] = create_phenotype_network(cppn, self.substrate)
        return net
