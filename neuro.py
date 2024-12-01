#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import pickle
import random
import time
from pathlib import Path
from typing import Final, Optional, Tuple

import neat
import neat.parallel
import numpy as np

from simulation import extract_features
from simulation import RandomPlayer
from simulation import evaluate_fitness
from simulation import NUM_PLAYERS, SMALL_BLIND_AMOUNT, BIG_BLIND_AMOUNT, STACK

DEFAULT_EVOLUTION_FILE: Final[Path] = Path("models/neat_poker.pkl")


@dataclass
class NEATEvolution:
    population: neat.Population
    stats: neat.StatisticsReporter
    best_genome: neat.DefaultGenome


class NEATPlayer(RandomPlayer):
    def __init__(self, net, uuid) -> None:
        self.net = net
        self.uuid = uuid

    def declare_action(self, valid_actions, hole_card, round_state) -> Tuple[str, int]:
        features = extract_features(hole_card, round_state, self.uuid)
        output = self.net.activate(features)  # Neural network output
        chosen_action_idx = np.argmax(output)

        # Following are the allowed actions
        # 0: fold
        # 1: call
        # 2: raise min
        # 3: raise 2x min
        # 4: raise 3x min
        # 5: raise max

        if chosen_action_idx == 0:  # Fold
            return "fold", 0

        if chosen_action_idx == 1:  # Call
            return "call", valid_actions[1]["amount"]

        raise_action = valid_actions[2]
        min_raise = raise_action["amount"]["min"]
        max_raise = raise_action["amount"]["max"]
        
        if chosen_action_idx == 2:  # Raise min
            return "raise", min_raise
        
        if chosen_action_idx == 3:  # Raise 2x min
            return "raise", min_raise * 2

        if chosen_action_idx == 4:  # Raise 3x min
            return "raise", min_raise * 3

        if chosen_action_idx == 5:  # Raise max
            return "raise", max_raise # All-in
        
        raise ValueError(f"Invalid action index: {chosen_action_idx}")


def evaluate_genome(genome, config, seed=None):
    if seed is None:
        random.seed(time.time())
        seed = random.randint(0, 1000)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    player = NEATPlayer(net, "uuid-1")
    f1 = evaluate_fitness(player, [RandomPlayer(), RandomPlayer()], seed=seed)
    return f1


def eval_genomes(genomes, config):
    seed = random.randint(0, 1000)
    for genome_id, genome in genomes:
        genome.fitness = evaluate_genome(genome, config, seed)


def run_neat(
    evolution: Optional[NEATEvolution] = None, 
    n: int = 50
) -> NEATEvolution:
    """Run NEAT neuroevolution to evolve poker players.
    
    Parameters:
        evolution: NEATEvolution | None
            The initial evolution to evolve from, if any.
        n: int
            The number of generations to run.

    Returns:
        evolution: NEATEvolution
            The evolved population.
    """
    # Load configuration
    config_path = "config-feedforward.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Get or create population
    population: neat.Population = evolution.population if evolution is not None else neat.Population(config)


    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT
    evaluator = neat.parallel.ParallelEvaluator(4, evaluate_genome)

    # winner = population.run(eval_genomes, n=50)
    # TODO: Fix type warning
    winner: Final[neat.DefaultGenome] = population.run(evaluator.evaluate, n=n)  # type: ignore


    # print("Best genome:", winner)
    return NEATEvolution(population, stats, winner)


def get_args():
    """Get arguments for the script.
    
    Returns:
        args: argparse.Namespace
            Arguments.
    """
    parser = argparse.ArgumentParser(description="Train a poker player using NEAT.")

    parser.add_argument(
        "-f", "--evolution-file", 
        type=Path, 
        default=DEFAULT_EVOLUTION_FILE, 
        help="File to read/save the evolution to/from."
    )

    return parser.parse_args()


def main():
    """Run the script."""

    # Read args
    args: Final[argparse.Namespace] = get_args()
    evolution_file: Final[Path] = args.evolution_file

    # Initialize variables
    evolution: Optional[NEATEvolution] = None  # Evolved population
    n: int = 1  # Generation to run until

    # Load previous evolution, if it exists
    if evolution_file.exists():
        # Load evolution
        with evolution_file.open("rb") as ef:
            evolution = pickle.load(ef)

            if not isinstance(evolution, NEATEvolution):
                raise ValueError(f"Invalid evolution file: {evolution_file}")

            print(f"Loaded evolution at {evolution_file}")
            print(f"Running until generation {n + evolution.population.generation}...")
    else:
        # Start from scratch
        print("Starting new evolution")
        print(f"Running until generation {n}...")

    # Run NEAT
    evolution = run_neat(evolution, n)

    # Save model
    with evolution_file.open("wb") as ef:
        pickle.dump(evolution, ef)
        print(f"Saved evolution at {evolution_file}")


if __name__ == "__main__":
    main()
