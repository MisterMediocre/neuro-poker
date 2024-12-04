"""Neuroevolution model that uses NEAT.
"""

import random
import time
from functools import partial
from pathlib import Path
from typing import Dict, Final, List, Optional, override

from neat import Config as NEATConfig
from neat import (
    DefaultGenome,
    DefaultReproduction,
    DefaultSpeciesSet,
    DefaultStagnation,
    Population,
    StatisticsReporter,
    StdOutReporter,
)
from neat.nn import FeedForwardNetwork
from neat.parallel import ParallelEvaluator
from termcolor import colored

from neuropoker.config import Config as NeuropokerConfig
from neuropoker.game import Game, PlayerStats
from neuropoker.models.base_model import BaseModel
from neuropoker.players.base_player import BasePlayer
from neuropoker.players.neat_player import NEATNetwork, NEATPlayer


class NEATModel(BaseModel):
    """Neuroevolution model that uses NEAT."""

    def __init__(
        self,
        neat_config_file: Path = Path("configs/3p_4s_neat.txt"),
    ) -> None:
        """Create a NEAT neuroevolution model.

        Parameters:
            neat_config_file: Path
                The path to the NEAT configuration file.
        """
        # Load NEAT config
        self.config_file: Final[Path] = neat_config_file
        self.config: Final[NEATConfig] = NEATConfig(
            DefaultGenome,
            DefaultReproduction,
            DefaultSpeciesSet,
            DefaultStagnation,
            neat_config_file,
        )

        # Create population
        self.population: Final[Population] = Population(self.config)
        self.best_genome: Optional[DefaultGenome] = None

        # Reset fitness values and recompute them.
        #
        # Since our evaluations are stochastic, this will prevent bad
        # fitness values from being carried over.
        for _genome_id, genome in self.population.population.items():
            genome.fitness = None

        # Add reporters
        self.stats: Final[StatisticsReporter] = StatisticsReporter()

        self.population.add_reporter(StdOutReporter(True))
        self.population.add_reporter(self.stats)

        # Curried function for obtaining the network from a genome
        self._network_fn = partial(__class__._get_network, config=self.config)

    @classmethod
    @override
    def from_config(cls, config: NeuropokerConfig) -> "NEATModel":
        """Create a NEAT model from a neuropoker configuration.

        Parameters:
            config: NeuropokerConfig
                The neuropoker configuration.

        Returns:
            model: NEATModel
                The NEAT model.
        """
        config = config["model"]

        if config["type"] != "neat":
            raise ValueError(f"Config has wrong model type: {config['type']}")

        return NEATModel(
            neat_config_file=config["neat_config_file"],
        )

    @override
    def print_config(self) -> None:
        """Print the configuration of this model."""
        print(
            colored(
                "--------------- Model ----------------", color="blue", attrs=["bold"]
            )
        )
        print(colored(f"{'Type':<12}: ", color="blue") + "NEAT")
        print(colored(f"{'NEAT config':<12}: ", color="blue") + f"{self.config_file}")
        print()

    def get_network(self, genome: DefaultGenome) -> NEATNetwork:
        """Get the network from a genome.

        Parameters:
            genome: DefaultGenome
                The genome to get the network from.

        Returns:
            net: NEATNetwork
                The network.
        """
        return self._network_fn(genome)

    def get_best_genome_network(self) -> NEATNetwork:
        """Get the network from the best genome.

        Returns:
            net: NEATNetwork
                The network created from the best genome.
        """
        if self.best_genome is None:
            raise ValueError("No best genome found")

        return self._network_fn(self.best_genome)

    @staticmethod
    def _get_network(genome: DefaultGenome, config: NEATConfig) -> NEATNetwork:
        """Get the network from a genome, statically.

        Parameters:
            genome: DefaultGenome
                The genome to get the network from.
            config: NEATConfig
                The NEAT configuration.

        Returns:
            net: NEATNetwork
                The network.
        """
        return FeedForwardNetwork.create(genome, config)

    def run(
        self,
        opponents,  # List[PlayerDefinition]
        neuropoker_config: NeuropokerConfig,
        num_generations: int = 50,
        num_games: int = 400,
        seed: Optional[int] = None,
        num_cores: int = 1,
    ) -> DefaultGenome:
        """Run evolutionary training to evolve poker players.

        Parameters:
            opponents: List[PlayerDefinition]
                The list of opponents to play against.
            neuropoker_config: NeuropokerConfig
                The game configuration.
            num_generations: int
                The number of generations to run.
            num_games: int
                The number of games to run to evaluate each genome,
                in each position, in each generation.
            seed: int | None
                The seed to use for the evaluation.
            num_cores: int
                The number of cores to use in parallel.

        Returns:
            best_genome: DefaultGenome
                The winning genome.
        """
        # Reset fitness values and recompute them.
        #
        # Since our evaluations are stochastic, this will prevent bad
        # fitness values from being carried over.
        for genome in self.population.population.values():
            genome.fitness = None

        curried_evaluate_genome = partial(
            evaluate_genome,
            network_fn=self._network_fn,
            opponents=opponents,
            neuropoker_config=neuropoker_config,
            num_games=num_games,
            seed=seed,
        )

        evaluator: Final[ParallelEvaluator] = ParallelEvaluator(
            num_cores, curried_evaluate_genome
        )

        best_genome: Final[DefaultGenome] = self.population.run(  # type: ignore
            evaluator.evaluate, n=num_generations
        )

        self.best_genome = best_genome

        return best_genome


def evaluate_genome(
    genome: DefaultGenome,
    _neat_config: NEATConfig,
    # Curried arguments
    network_fn,
    opponents,  # List[PlayerDefinition]
    neuropoker_config: NeuropokerConfig,
    num_games: int = 400,
    seed: Optional[int] = None,
) -> float:
    """Evaluate a single genome.

    Parameters:
        genome: DefaultGenome
            The genome to evaluate.
        _neat_config: NEATConfig
            The NEAT configuration.
        opponents: List[PlayerDefinition]
            The list of opponents to play against.
        cards: List[str]
            The set of cards to use in each game.
        num_games: int
            The number of games to play in each position.
        seed: int | None
            The seed to use for the evaluation.

    Returns:
        fitness: float
            The fitness of this genome.
    """
    if len(opponents) <= 0:
        raise ValueError("At least one opponent must be provided")

    num_players: Final[int] = neuropoker_config["game"]["players"]

    # Randomize seed if not provided
    if seed is None:
        random.seed(time.time())
        seed = random.randint(0, 1000)

    # Initialize our player
    player = NEATPlayer("player", network_fn(genome), training=True)

    # Play each position at the table
    f1: float = 0
    for i in range(num_players):
        # Permute player position
        player_pos: int = i

        # Opponents: Sample with replacement
        players: List[BasePlayer] = [
            opp.load(f"opponent-{j}")  # type: ignore
            for j, opp in enumerate(random.choices(opponents, k=num_players - 1))
        ]

        # Player: Add at <player_pos>
        players.insert(player_pos, player)

        # Convert to sorted list
        game: Game = Game.from_config(players, neuropoker_config)
        game_stats: Dict[str, PlayerStats] = game.play_multiple(
            num_games=num_games, seed=seed
        )

        average_winnings: float = (
            game_stats["player"]["winnings"] / game_stats["player"]["num_games"]
        )

        f1 += average_winnings

    return f1 / num_players
