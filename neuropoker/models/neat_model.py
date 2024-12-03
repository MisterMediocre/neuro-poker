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
from neuropoker.game import PlayerStats, evaluate_performance
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
        print(colored("Model:", color="blue", attrs=["bold"]))
        print(
            colored(f"{'    Type':<20}", color="blue", attrs=["bold"])
            + colored(": HyperNEAT", color="blue")
        )
        print(
            colored(f"{'    NEAT config':<20}", color="blue", attrs=["bold"])
            + colored(f": {self.config_file}", color="blue")
        )
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
        return FeedForwardNetwork.create(genome, self.config)

    def get_best_genome_network(self) -> NEATNetwork:
        """Get the network from the best genome.

        Returns:
            net: NEATNetwork
                The network created from the best genome.
        """
        if self.best_genome is None:
            raise ValueError("No best genome found")
        return self.get_network(self.best_genome)

    def evaluate_genome(
        self,
        genome: DefaultGenome,
        _neat_config: NEATConfig,
        opponents: List[BasePlayer],
        neuropoker_config: NeuropokerConfig,
        num_games: int = 500,
        seed: Optional[int] = None,
    ) -> float:
        """Evaluate a single genome.

        Parameters:
            genome: DefaultGenome
                The genome to evaluate.
            _neat_config: NEATConfig
                The NEAT configuration.
            opponents: List[BasePlayer]
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
        if len(opponents) < 1:
            raise ValueError("At least one opponent must be provided")

        # Randomize seed if not provided
        random.seed(seed if seed is not None else time.time())
        net: Final[NEATNetwork] = self.get_network(genome)

        f1: float = 0
        for i in range(3):
            player_pos: int = i
            opponent_1_pos: int = (player_pos + 1) % 3
            opponent_2_pos: int = (player_pos + 2) % 3

            # Initialize players
            player_dict: Dict[int, BasePlayer] = {}
            player_dict[player_pos] = NEATPlayer("player", net, training=False)

            opponent_1_index, opponent_2_index = random.sample(range(len(opponents)), 2)

            player_dict[opponent_1_pos] = opponents[opponent_1_index]
            player_dict[opponent_2_pos] = opponents[opponent_2_index]

            # player_dict[opponent_1_pos] = copy.deepcopy(opponents[opponent_1_index])
            # player_dict[opponent_2_pos] = copy.deepcopy(opponents[opponent_2_index])
            # player_dict[opponent_1_pos].uuid = "opponent-1"  # type: ignore
            # player_dict[opponent_2_pos].uuid = "opponent-2"  # type: ignore

            # Convert to sorted list
            players: List[BasePlayer] = [player_dict[i] for i in range(3)]

            # Play each position 100 times, but with same seeds and hence same cards drawn
            player_performances: Dict[str, PlayerStats] = evaluate_performance(
                players, neuropoker_config, num_games=num_games, seed=seed
            )

            our_average_winnings: float = (
                player_performances["player"]["winnings"]
                / player_performances["player"]["num_games"]
            )

            f1 += our_average_winnings

        return f1 / 3

    def run(
        self,
        opponents: List[BasePlayer],
        neuropoker_config: NeuropokerConfig,
        num_generations: int = 50,
        num_games: int = 500,
        num_cores: int = 1,
    ) -> DefaultGenome:
        """Run evolutionary training to evolve poker players.

        Parameters:
            opponents: List[BasePlayer]
                The list of opponents to play against.
            neuropoker_config: NeuropokerConfig
                The game configuration.
            num_generations: int
                The number of generations to run.
            num_games: int
                The number of games to run to evaluate each genome,
                in each position, in each generation.
            num_cores: int
                The number of cores to use in parallel.

        Returns:
            best_genome: DefaultGenome
                The winning genome.
        """
        curried_evaluate_genome = partial(
            self.evaluate_genome,
            opponents=opponents,
            neuropoker_config=neuropoker_config,
            num_games=num_games,
        )

        evaluator: Final[ParallelEvaluator] = ParallelEvaluator(
            num_cores, curried_evaluate_genome
        )

        best_genome: Final[DefaultGenome] = self.population.run(  # type: ignore
            evaluator.evaluate, n=num_generations
        )

        self.best_genome = best_genome

        return best_genome
