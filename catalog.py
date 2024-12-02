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


    # model_0 has been trained against the fold player
    "model_0": load_player("models/model_0.pkl", "model_0"),

    # model_1 has been trained against the call player
    "model_1": load_player("models/model_1.pkl", "model_1"),
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
# Expect each to win 0 on average

compete("call", "fold", "fold2", 3)
# Expect call player to win every single round, on average 50
# When small blind, will win 50
# When big blind, will win 25
# When neither, will win 75


compete("random", "fold", "call", 30)
# Random is an absolutely horrible player, since he will often raise first and fold later.


compete("model_0", "fold", "fold2", 30)
# Expect to match call player's performance, ie average 50

compete("model_0", "call", "fold", 30)
# Expect to win the rounds with good cards, and be conservative with bad cards


# compete("call", "call2", "call3", 10)
# compete("random", "fold", "call", 10)

