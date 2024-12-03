"""Neuroevolution model that uses ES-HyperNEAT.
"""

import json
from pathlib import Path
from typing import Any, Dict, Final, Optional

from neat import DefaultGenome
from neat.nn import FeedForwardNetwork, RecurrentNetwork
from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.substrate import Substrate

from neuropoker.models.hyperneat import CoordinateList
from neuropoker.models.neat import NEATEvolution, NEATModel, NEATNetwork


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
        evolution: Optional[NEATEvolution] = None,
        config_file: Path = Path("config-feedforward.txt"),
        es_config_file: Path = Path("config-es-feedforward.json"),
    ) -> None:
        """Create a HyperNEAT neuroevolution model.

        Parameters:
            evolution: NEATEvolution | None
                The initial evolution to evolve from, if any.
            config_file: Path
                The path to the NEAT configuration file.
            es_config_file: Path
                The path to the ES-HyperNEAT configuration file.
        """
        super().__init__(evolution=evolution, config_file=config_file)

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
        self.es_config: Final[Dict[str, Any]] = get_es_config(es_config_file)

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
