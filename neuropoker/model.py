"""Classes and functions for NEAT poker models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import random
import time

import neat
import neat.parallel

from neuropoker.player import NEATPlayer, RandomPlayer
from neuropoker.game import evaluate_fitness


@dataclass
class NEATEvolution:
    """The result of a NEAT neuroevolution training run.
    """
    population: neat.Population
    stats: neat.StatisticsReporter
    best_genome: neat.DefaultGenome


def evaluate_genome(
    genome: neat.DefaultGenome, 
    config: neat.Config, 
    seed: Optional[int] = None
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
    player = NEATPlayer(net, "uuid-1")
    f1 = evaluate_fitness(player, [RandomPlayer(), RandomPlayer()], seed=seed)
    return f1


def eval_genomes(
    genomes: List[Tuple[int, neat.DefaultGenome]], 
    config: neat.Config
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


def run_neat(
    evolution: Optional[NEATEvolution] = None, 
    config_file: Path = Path("config-feedforward.txt"),
    num_generations: int = 50,
    num_cores: int = 1,
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
        config_file
    )

    # Get or create population
    population: neat.Population = (
        evolution.population 
        if evolution is not None 
        else neat.Population(config)
    )

    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT
    evaluator = neat.parallel.ParallelEvaluator(num_cores, evaluate_genome)

    # winner = population.run(eval_genomes, n=50)
    # TODO: Fix type warning
    winner: Final[neat.DefaultGenome] = population.run(  # type: ignore
        evaluator.evaluate, 
        n=num_generations
    )

    # print("Best genome:", winner)
    return NEATEvolution(population, stats, winner)
