"""Neuroevolution model that uses NEAT.
"""

import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, List, Optional

import neat
import neat.parallel

from neuropoker.game import evaluate_performance
from neuropoker.models.base import BaseModel
from neuropoker.player_utils import load_player
from neuropoker.players.base import BasePlayer
from neuropoker.players.neat import NEATPlayer


@dataclass
class NEATEvolution:
    """The result of a NEAT neuroevolution training run."""

    population: neat.Population
    stats: neat.StatisticsReporter
    best_genome: neat.DefaultGenome


def get_network(evolution: NEATEvolution) -> neat.nn.FeedForwardNetwork:
    """Get the network from the best genome.

    Parameters:
        evolution: NEATEvolution
            The evolution to get the network from.

    Returns:
        net: neat.nn.FeedForwardNetwork
            The network.
    """
    return neat.nn.FeedForwardNetwork.create(
        evolution.best_genome, evolution.population.config
    )


class NEATModel(BaseModel):
    """Neuroevolution model that uses NEAT."""

    def __init__(
        self,
        evolution: Optional[NEATEvolution] = None,
        config_file: Path = Path("config-feedforward.txt"),
    ) -> None:
        """Create a NEAT neuroevolution model.

        Parameters:
            evolution: NEATEvolution | None
                The initial evolution to evolve from, if any.
            config_file: Path
                The path to the NEAT configuration file.
            num_generations: int
                The number of generations to run.
            num_cores: int
                The number of cores to use in parallel.
        """
        # Load NEAT config
        self.config: Final[neat.Config] = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file,
        )

        # Get or create population
        self.population: Final[neat.Population] = (
            evolution.population
            if evolution is not None
            else neat.Population(self.config)
        )

        # Reset fitness values and recompute them.
        #
        # Since our evaluations are stochastic, this will prevent bad
        # fitness values from being carried over.
        for _genome_id, genome in self.population.population.items():
            genome.fitness = None

        # Add reporters
        self.stats: Final[neat.StatisticsReporter] = neat.StatisticsReporter()

        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(self.stats)

    def get_network(self, genome: neat.DefaultGenome) -> neat.nn.FeedForwardNetwork:
        """Get the network from a genome.

        Parameters:
            genome: neat.DefaultGenome
                The genome to get the network from.

        Returns:
            net: neat.nn.FeedForwardNetwork
                The network.
        """
        return neat.nn.FeedForwardNetwork.create(genome, self.config)

    def evaluate_genome(
        self,
        genome: neat.DefaultGenome,
        config: neat.Config,
        opponents: List[str],
        seed: Optional[int] = None,
    ) -> float:
        """Evaluate a single genome.

        Parameters:
            genome: neat.DefaultGenome
                The genome to evaluate.
            config: neat.Config
                The NEAT configuration.
            opponents: List[str] | None
                The list of opponents to play against.
            seed: int | None
                The seed to use for the evaluation.
        """

        assert len(opponents) > 0, "At least one opponent must be provided"

        net = self.get_network(genome)

        random.seed(time.time())
        seed = random.randint(0, 1000)

        f1 = 0
        for i in range(3):
            player_pos = i
            opponent_1_pos = (player_pos + 1) % 3
            opponent_2_pos = (player_pos + 2) % 3

            player_names = [
                f"player-{i}" if i == player_pos else f"opponent-{i}" for i in range(3)
            ]
            player_name = player_names[player_pos]

            players = [BasePlayer()] * 3  # Initialize a list of size 3 with None
            players[player_pos] = NEATPlayer(net, player_names[player_pos])
            players[opponent_1_pos] = load_player(
                random.choice(opponents), player_names[opponent_1_pos]
            )
            players[opponent_2_pos] = load_player(
                random.choice(opponents), player_names[opponent_2_pos]
            )

            # Play each position 100 times, but with same seeds and hence same cards drawn
            player_performance = evaluate_performance(
                player_names, players, seed=seed, num_games=500
            )[player_name]
            average_winnings = (
                player_performance["winnings"] / player_performance["num_games"]
            )
            f1 += average_winnings

        return f1 / 3

    def run(
        self,
        opponents: List[str],
        num_generations: int = 50,
        num_cores: int = 1,
    ) -> neat.DefaultGenome:
        """Run evolutionary training to evolve poker players.

        Parameters:
            opponents: List[str]
                The list of opponents to play against.
            num_generations: int
                The number of generations to run.
            num_cores: int
                The number of cores to use in parallel.

        Returns:
            winner: neat.DefaultGenome
                The winning genome.
        """
        curried_evaluate_genome = partial(self.evaluate_genome, opponents=opponents)

        evaluator: Final[neat.ParallelEvaluator] = neat.parallel.ParallelEvaluator(
            num_cores, curried_evaluate_genome
        )

        winner: Final[neat.DefaultGenome] = self.population.run(  # type: ignore
            evaluator.evaluate, n=num_generations
        )

        return winner
