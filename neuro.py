import neat
import neat.parallel

import numpy as np
import random
import pickle
import time

from simulation import extract_features
from simulation import RandomPlayer
from simulation import evaluate_fitness
from simulation import NUM_PLAYERS, SMALL_BLIND_AMOUNT, BIG_BLIND_AMOUNT, STACK

class NEATPlayer(RandomPlayer):
    def __init__(self, net, uuid):
        self.net = net
        self.uuid = uuid

    def declare_action(self, valid_actions, hole_card, round_state):
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


def run_neat(population = None, n = 50):
    # Load configuration
    config_path = "config-feedforward.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create population
    if population is None:
        population = neat.Population(config)


    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT
    evaluator = neat.parallel.ParallelEvaluator(4, evaluate_genome)

    # winner = population.run(eval_genomes, n=50)
    winner = population.run(evaluator.evaluate, n=50)


    # print("Best genome:", winner)
    return population, stats, winner


FILE = "models/neat_poker.pkl"
if __name__ == "__main__":

    population = None
    n = 50
    # Check if the file exists
    try:
        with open(FILE, "rb") as f:
            population, stats, best_genome = pickle.load(f)
            n = n + stats.generation
            print("Loaded population")
    except Exception as e:
        print(e)


    population, stats, best_genome = run_neat(population, n)


    with open(FILE, "wb") as f:
        pickle.dump((population, stats, best_genome), f)


