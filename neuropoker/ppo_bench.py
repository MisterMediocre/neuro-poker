"""Bench for training and evaluating PPO models."""

from pathlib import Path
from typing import Any, Dict, Final, List

from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.ppo.ppo import PPO
from termcolor import colored

from neuropoker.extra.torch import get_device
from neuropoker.game.cards import SHORT_RANKS, SHORTER_SUITS, get_card_list  # 3p_3s
from neuropoker.game.features import (
    CNNFeaturesCollector,
    FeaturesCollector,
    LinearFeaturesCollector,
)
from neuropoker.game.game import (
    Game,
    PlayerStats,
    default_player_stats,
    merge,
)
from neuropoker.game.gym import make_env
from neuropoker.players.base import BasePlayer
from neuropoker.players.naive import CallPlayer
from neuropoker.players.ppo import PPOPlayer


class PPOBench:
    def __init__(
        self,
        output_model_dir: Path,
        starting_model_path: Path | None = None,
        features_collector: FeaturesCollector | None = None,
        policy_kwargs: Dict[str, Any] = {},
        model_kwargs: Dict[str, Any] = {},
        num_environments: int = 8,
        device: str = "auto",
    ) -> None:
        """Initialize the PPOBench.

        Parameters:
            output_model_dir: Path
                The path to save the trained models and stats to
                at the end of each epoch.
            starting_model_path: Path | None
                The path to the model to bootstrap training from.
                (default: new model)
            features_collector: FeaturesCollector | None
                The features collector to use.
                (default: LinearFeaturesCollector)
            policy_kwargs: Dict[str, Any]
                The policy kwargs to pass to the PPO model.
            num_environments: int
                The number of environments to run in parallel.
            num_timesteps: int
                The number of timesteps to train for.
            num_epochs: int
                The maximum number of epochs to train for.
            device: str
                The device to run the model on.
        """

        #
        # Set configuration
        #
        self.starting_model_path: Final[Path | None] = starting_model_path
        self.output_model_dir: Final[Path] = output_model_dir

        self.num_environments: Final[int] = num_environments
        self.device: Final[str] = get_device() if device == "auto" else device
        self.features_collector: Final[FeaturesCollector] = (
            LinearFeaturesCollector()
            if features_collector is None
            else features_collector
        )
        self.policy_kwargs: Final[Dict[str, Any]] = policy_kwargs
        self.model_kwargs: Final[Dict[str, Any]] = model_kwargs

        #
        # Print configuration
        #
        print(
            colored("------------ PPOBench -------------", color="blue", attrs=["bold"])
        )

        print(colored("Models", color="blue", attrs=["bold"]))
        print(
            "    "
            + colored(f'{"Starting model":<16}:', color="blue")
            + f" {self.starting_model_path}"
        )
        print(
            "    "
            + colored(f'{"Output models":<16}:', color="blue")
            + f" {self.output_model_dir}"
        )
        print(
            "    "
            + colored(f'{"Policy kwargs":<16}:', color="blue")
            + f" {self.policy_kwargs}"
        )
        print(
            "    "
            + colored(f'{"Model kwargs":<16}:', color="blue")
            + f" {self.model_kwargs}"
        )

        print(colored("Training", color="blue", attrs=["bold"]))
        print(
            "    "
            + colored(f'{"Environments":<16}: ', color="blue")
            + f"{num_environments}"
        )
        print("    " + colored(f'{"Device":<16}: ', color="blue") + f"{self.device}")
        print(
            "    "
            + colored(f'{"Features":<16}: ', color="blue")
            + f"{self.features_collector}"
        )
        print()

        #
        # Set up the environments
        #
        print(
            colored(
                f"Setting up {num_environments} environments...",
                color="blue",
                attrs=["bold"],
            )
        )
        self.env: Final[VecEnv] = SubprocVecEnv(
            [
                make_env(
                    starting_model_path=self.starting_model_path,
                    opponent_model_path=self.starting_model_path,
                    feature_collector=self.features_collector,
                    reset_threshold=30000,
                    suits=SHORTER_SUITS,
                    ranks=SHORT_RANKS,
                )
                for _ in range(num_environments)
            ]
        )
        print()

    @staticmethod
    def _model_path_exists(model_path: str | Path | None) -> bool:
        """Helper function to check if a model path exists.

        Parameters:
            model_path: str | Path | None
                The model path to check.

        Returns:
            exists: bool
                Whether the model path exists.
        """
        return (
            model_path is not None
            and (
                Path(model_path).exists()
                or Path(model_path).with_suffix(".zip").exists()
            )
            and not Path(model_path).is_dir()
        )

    def _create_trainee(self, policy_type: str = "MlpPolicy", **kwargs) -> PPOPlayer:
        """Create the trainee.

        Parameters:
            policy_type: str
                The type of policy to use.
            **kwargs
                Additional arguments to pass to the constructor.

        Returns:
            player: PPOPlayer
                The created player, either from scratch or from the starting
                model path.
        """

        #
        # Infer policy type
        #
        if isinstance(self.features_collector, LinearFeaturesCollector):
            policy_type = "MlpPolicy"
        elif isinstance(self.features_collector, CNNFeaturesCollector):
            policy_type = "CnnPolicy"
        else:
            raise ValueError(
                f"Unknown policy type for features collector {self.features_collector}"
            )

        #
        # Instantiate model
        #
        model: Final[PPO] = PPO(
            policy_type,
            self.env,
            policy_kwargs=self.policy_kwargs,
            device=self.device,
            **self.model_kwargs,
        )

        #
        # Load starting model's policy
        #
        if self._model_path_exists(self.starting_model_path):
            print(
                colored("[PPOBench._create_trainee]", color="blue", attrs=["bold"])
                + f" Creating trainee from starting model at {self.starting_model_path}"
            )
            old_model: Final[PPO] = PPO.load(
                self.starting_model_path,  # type: ignore
                env=self.env,
            )
            model.policy.load_state_dict(old_model.policy.state_dict(), strict=True)
        else:
            print(
                colored("[PPOBench._create_trainee]", color="blue", attrs=["bold"])
                + " Creating trainee from scratch"
            )

        #
        # Return player with model
        #
        return PPOPlayer(model, "me", feature_collector=self.features_collector)

    def _create_opponent(
        self, model_path: Path | None, uuid: str, default: BasePlayer | None = None
    ) -> BasePlayer:
        """Load an opponent, based on the starting model.

        Parameters:
            uuid: str
                The player's UUID.
            default: BasePlayer
                The player to use if the model is not found.

        Returns:
            player: BasePlayer
                The loaded player, from either the starting model path
                or the default player.
        """

        #
        # Check if the starting model exists.
        #
        if not self._model_path_exists(self.starting_model_path):
            if default is None:
                raise FileNotFoundError(f"Opponent model not found: {model_path}")
            else:
                print(
                    colored("[PPOBench._create_opponent]", color="blue", attrs=["bold"])
                    + f" Opponent model not found: {model_path}. Using default player {default}..."
                )
                return default

        #
        # Load the starting model
        #
        print(
            colored("[PPOBench._create_opponent]", color="blue", attrs=["bold"])
            + f" Loading opponent {uuid} from {model_path}..."
        )
        return PPOPlayer.from_model_file(
            self.starting_model_path,  # type: ignore
            uuid,
            feature_collector=self.features_collector,
        )

    def _save_trainee(self, trainee_player: PPOPlayer, epoch: int) -> None:
        """Save a trainee model after it has been trained for an epoch.

        Parameters:
            trainee_player: PPOPlayer
                The trainee player to save.
            epoch: int
                The current epoch number.
        """
        self.output_model_dir.mkdir(exist_ok=True, parents=True)
        output_epoch_model_path: Path = self.output_model_dir / f"epoch_{epoch}"

        print(
            colored("[PPOBench.save_trainee]", color="blue", attrs=["bold"])
            + f" Saving trainee model to {output_epoch_model_path}..."
        )
        trainee_player.model.save(output_epoch_model_path)

    def _train_epoch(
        self,
        epoch: int,
        trainee_player: PPOPlayer,
        opponent_players: List[BasePlayer],
        num_timesteps: int = 100000,
    ):
        """Train the PPO model for a single epoch.

        Parameters:
            epoch: int
                The current epoch number.
            trainee_player: PPOPlayer
                The player to train.
            opponent_players: List[BasePlayer]
                The list of opponents to train against.
            num_timesteps: int
                The number of timesteps to train for.
        """
        #
        # Train the model
        #
        print(
            colored("[PPOBench._train_epoch]", color="blue", attrs=["bold"])
            + colored(f" (epoch {epoch})", color="blue")
            + " Training model..."
        )
        trainee_player.model.learn(
            total_timesteps=num_timesteps, reset_num_timesteps=False
        )

        #
        # Save the model
        #
        self._save_trainee(trainee_player, epoch)

        #
        # Evaluate the model
        #
        print(
            colored("[PPOBench._train_epoch]", color="blue", attrs=["bold"])
            + colored(f" (epoch {epoch})", color="blue")
            + " Evaluating model..."
        )

        players: List[BasePlayer] = [trainee_player, *opponent_players]
        overall_performance: PlayerStats = default_player_stats()
        overall_performance["uuid"] = "me"

        # Try each position
        for i in range(0, 3):
            players_: List[BasePlayer] = players[i:] + players[:i]

            game = Game(players_, get_card_list(SHORTER_SUITS, SHORT_RANKS))
            performances: Dict[str, PlayerStats] = game.play_multiple(
                num_games=2000, seed=-1
            )
            overall_performance = merge(overall_performance, performances["me"])

        #
        # Print evaluation results
        #
        average_winnings: float = (
            overall_performance["winnings"] / overall_performance["num_games"]
        )
        print(
            colored("[PPOBench._train_epoch]", color="blue", attrs=["bold"])
            + colored(f" (epoch {epoch})", color="blue")
            + f" Average winnings: {average_winnings}"
        )
        print(
            colored("[PPOBench._train_epoch]", color="blue", attrs=["bold"])
            + colored(f" (epoch {epoch})", color="blue")
            + f" Overall performance: {overall_performance}"
        )

    def train(self, num_epochs: int = 100, num_timesteps: int = 100000) -> None:
        """Train the PPO model against the baseline opponents.

        Parameters:
            num_epochs: int
                The number of epochs to train for.
            num_timesteps: int
                The number of timesteps to train for in each epoch.
        """
        #
        # Load the trainee player
        #
        trainee_player: Final[PPOPlayer] = self._create_trainee(
            verbose=1,
            ent_coef=0.01,
            vf_coef=0.7,
            n_steps=256,
            # learning_rate=0.003,
        )

        opponent_players: Final[List[BasePlayer]] = [
            self._create_opponent(
                self.starting_model_path,
                f"opponent_{i}",
                default=CallPlayer(f"opponent_{i}"),
            )
            for i in range(1, 3)
        ]

        #
        # Print info
        #
        print()
        print(colored("-------- Training PPO --------", color="blue", attrs=["bold"]))
        print(colored(f'{"Timesteps":<16}: ', color="blue") + f"{num_timesteps}")
        print(colored(f'{"Epochs":<16}: ', color="blue") + f"{num_epochs}")

        #
        # Train the trainee player
        #
        for epoch in range(num_epochs):
            print()
            print(
                colored(
                    f"--------- epoch {epoch} / {num_epochs} ----------",
                    color="blue",
                    attrs=["bold"],
                )
            )
            self._train_epoch(epoch, trainee_player, opponent_players, num_timesteps)
