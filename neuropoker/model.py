"""Classes and functions for NEAT poker models.
"""

import pickle
import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, List, Optional, Tuple

import neat
import neat.parallel

from neuropoker.game import evaluate_performance
from neuropoker.player import (
    BasePlayer,
    CallPlayer,
    FoldPlayer,
    NEATPlayer,
    RandomPlayer,
)


@dataclass
class NEATEvolution:
    """The result of a NEAT neuroevolution training run."""

    population: neat.Population
    stats: neat.StatisticsReporter
    best_genome: neat.DefaultGenome


def evaluate_genome(
    genome: neat.DefaultGenome,
    config: neat.Config,
    seed: Optional[int] = None,
    opponents: List[str] = ["RandomPlayer"],
) -> float:
    """Evaluate a single genome.

    Parameters:
        genome: neat.DefaultGenome
            The genome to evaluate.
        config: neat.Config
            The NEAT configuration.
        seed: int | None
            The seed to use for the evaluation.
    """

    if seed is None:
        random.seed(time.time())
        seed = random.randint(0, 1000)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    f1 = 0
    for i in range(10):
        # Play against some selection of available opponents

        player_pos = random.randint(0, 2)
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

        player_performance = evaluate_performance(player_names, players, seed=seed)[player_name]
        f1 += player_performance["winnings"]/player_performance["num_games"]

    return f1 / 10


def eval_genomes(
    genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config
) -> None:
    """Evaluate the fitness for a list of genomes.

    Parameters:
        genomes: List[Tuple[int, neat.DefaultGenome]]
            The genomes to evaluate.
        config: neat.Config
            The NEAT configuration.
    """
    seed = random.randint(0, 1000)
    for genome_id, genome in genomes:
        # TODO: Fix type warning
        genome.fitness = evaluate_genome(genome, config, seed)  # type: ignore


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


def load_player(definition: str, name: str) -> BasePlayer:
    if definition == "RandomPlayer":
        return RandomPlayer()
    if definition == "CallPlayer":
        return CallPlayer()
    if definition == "FoldPlayer":
        return FoldPlayer()

    # Else, the definition is a path to a NEAT file
    with open(definition, "rb") as f:
        evolution = pickle.load(f)
        net = get_network(evolution)
        return NEATPlayer(net, name)


def run_neat(
    evolution: Optional[NEATEvolution] = None,
    config_file: Path = Path("config-feedforward.txt"),
    num_generations: int = 50,
    num_cores: int = 1,
    opponents: List[str] = ["RandomPlayer"],
) -> NEATEvolution:
    """Run NEAT neuroevolution to evolve poker players.

    Parameters:
        evolution: NEATEvolution | None
            The initial evolution to evolve from, if any.
        config_file: Path
            The path to the NEAT configuration file.
        num_generations: int
            The number of generations to run.
        num_cores: int
            The number of cores to use in parallel.

    Returns:
        evolution: NEATEvolution
            The evolved population.
    """
    # Load configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    # Get or create population
    population: neat.Population = (
        evolution.population if evolution is not None else neat.Population(config)
    )

    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT
    curried_evaluate_genome = partial(evaluate_genome, opponents=opponents)

    evaluator = neat.parallel.ParallelEvaluator(num_cores, curried_evaluate_genome)

    # winner = population.run(eval_genomes, n=50)
    # TODO: Fix type warning
    winner: Final[neat.DefaultGenome] = population.run(  # type: ignore
        evaluator.evaluate, n=num_generations
    )

    # print("Best genome:", winner)
    return NEATEvolution(population, stats, winner)
