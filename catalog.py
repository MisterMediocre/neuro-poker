## Catalog of players to use

from neuropoker.game import evaluate_fitness
from neuropoker.player import RandomPlayer
from neuropoker.player import FoldPlayer
from neuropoker.player import CallPlayer
from neuro import DEFAULT_CONFIG_FILE
from neuropoker.model import load_player




catalog = {
    "random": load_player("RandomPlayer", "random"),
    "random2": load_player("RandomPlayer", "random2"),
    "random3": load_player("RandomPlayer", "random3"),

    "fold": load_player("FoldPlayer", "fold"),
    "fold2": load_player("FoldPlayer", "fold2"),
    "fold3": load_player("FoldPlayer", "fold3"),

    "call": load_player("CallPlayer", "call"),
    "call2": load_player("CallPlayer", "call2"),
    "call3": load_player("CallPlayer", "call3"),

    "model_0": load_player("models/neat_poker.pkl", "model_0"),
    "model_0_2": load_player("models/neat_poker.pkl", "model_0_2"),
}


def compete(player_1: str, player_2: str, player_3: str, num_games: int = 100): 
    print("\n\n")

    player_names = [player_1, player_2, player_3]
    print("In the competition, the players are: ", player_names)
    print("They will play", num_games, "games.")
    players = [catalog[player_name] for player_name in player_names]

    winnings = evaluate_fitness(player_names, players, num_games=num_games, seed=1)

    # Print the results

    for i, player_name in enumerate(player_names):
        print(f"{player_name}: {winnings[i]}")

    print("\n\n")


compete("fold", "fold2", "fold3", 3)

compete("call", "fold", "fold2", 3)
# Expect call player to win every single round, on average 25

# compete("call", "call2", "call3", 10)
# compete("random", "fold", "call", 10)

