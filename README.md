# Neuroevolution for Short-Hand Poker


This is an experiment to evolve agents to play multiplayer short-hand poker. 
In particular, we're interested in modelling collusion between agents.

## Installation
Tested on Python 3.13.1
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python train.py # To train the agents
python catalog.py # To play the agents against each other
```

## Models 
1. RandomPlayer: Plays randomly
2. FoldPlayer: Always folds
3. CallPlayer: Always calls

4. "model\_0": Evolved to beat 2x FoldPlayers
5. "model\_1": Evolved to beat 2x CallPlayers


```bash
python train.py -g 50 -o FoldPlayer -f "models/model_0.pkl" -c 10
python train.py -g 50 -o CallPlayer -f "models/model_1.pkl" -c 10
```

TODO: Add more models
- [ ] "model\_2": Evolved to beat (CallPlayer, FoldPlayer)
- [ ] "model\_3": Evolved to beat (CallPlayer, FoldPlayer, model\_1, model\_2)

