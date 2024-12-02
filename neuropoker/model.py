"""Classes and functions for NEAT poker models.
"""

import pickle
import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, List, Optional, Tuple, Union

import neat
import neat.parallel
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.substrate import Substrate

from neuropoker.game import evaluate_performance
from neuropoker.player.base import BasePlayer
from neuropoker.player.naive import CallPlayer, FoldPlayer, RandomPlayer
from neuropoker.player.neat import NEATPlayer


@dataclass
class NEATEvolution:
    """The result of a NEAT neuroevolution training run."""

    population: neat.Population
    stats: neat.StatisticsReporter
    best_genome: neat.DefaultGenome


def evaluate_genome(
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

    net = neat.nn.FeedForwardNetwork.create(genome, config)

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
    """Load a player, based on a string definition.

    Parameters:
        definition: str
            The player to load, or a path to an evolution file.
        name: str
            The name to assign to the player.

    Returns:
        player: BasePlayer
            The loaded player.
    """
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
    opponents: List[str],
    evolution: Optional[NEATEvolution] = None,
    config_file: Path = Path("config-feedforward.txt"),
    num_generations: int = 50,
    num_cores: int = 1,
) -> NEATEvolution:
    """Run NEAT neuroevolution to evolve poker players.

    Parameters:
        opponents: List[str]
            The list of opponents to play against.
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

    for _genome_id, genome in population.population.items():
        # Reset fitness values and recompute them.
        #
        # Since our evaluations are stochastic, this will prevent bad
        # fitness values from being carried over.
        genome.fitness = None

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


DEFAULT_HIDDEN_SIZES: List[int] = [64, 64]
CoordinateList = List[Tuple[float, float]]


def run_hyperneat(
    opponents: List[str],
    evolution: Optional[NEATEvolution] = None,
    hidden_sizes: Optional[Union[int, List[int]]] = None,
    config_file: Path = Path("config-feedforward.txt"),
    num_generations: int = 50,
    num_cores: int = 1,
) -> NEATEvolution:
    """Run HyperNEAT neuroevolution to evolve poker players.

    Parameters:
        opponents: List[str]
            The list of opponents to play against.
        evolution: NEATEvolution | None
            The initial evolution to evolve from, if any.
        hidden_sizes: int | List[int] | None
            The sizes of each hidden layer.
        config_file: Path
            The path to the NEAT configuration file.
        num_generations: int
            The number of generations to run.
        num_cores: int
            The number of cores to use in parallel.
    """
    # Get list of hidden sizes
    hidden_sizes_: List[int]
    match hidden_sizes:
        case int():
            hidden_sizes_ = [hidden_sizes]
        case list():
            hidden_sizes_ = hidden_sizes
        case _:
            hidden_sizes_ = DEFAULT_HIDDEN_SIZES

    # TODO: Don't hardcode input and output dimensionality
    input_dims: Final[int] = 80  # Number of features
    output_dims: Final[int] = 6  # Number of actions

    # Input coordinates
    #
    # TODO: Are these coordinates reasonable? I used them for HW2
    # but we may want something different for our poker features.
    #
    # Evenly spaced between (0, 0) and (0, 1)
    input_coordinates: Final[CoordinateList] = [
        (0.0, i / input_dims) for i in range(input_dims)
    ]

    # Output coordinates
    #
    # Evenly spaced between (0, 0) and (0, 1)
    output_coordinates: Final[CoordinateList] = [
        (0.0, i / output_dims) for i in range(output_dims)
    ]

    # Hidden coordinates
    #
    # For each layer, evenly spaced between (0, 0) and (0, 1)
    hidden_coordinates: Final[List[CoordinateList]] = [
        [(0.0, i / size) for i in range(size)] for size in hidden_sizes_
    ]

    activations: Final[int] = len(hidden_sizes_) + 2
    substrate: Final[Substrate] = Substrate(
        input_coordinates,
        output_coordinates,
        hidden_coordinates,
    )
