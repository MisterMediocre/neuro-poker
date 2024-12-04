"""Utility functions for models."""

import pickle
from pathlib import Path
from typing import Dict, Final, Optional, Type

from termcolor import colored

from neuropoker.config import Config as NeuropokerConfig
from neuropoker.models.base_model import BaseModel, ModelT
from neuropoker.models.es_hyperneat_model import ESHyperNEATModel
from neuropoker.models.hyperneat_model import HyperNEATModel
from neuropoker.models.neat_model import NEATModel

MODEL_TYPES: Final[Dict[str, Type[BaseModel]]] = {
    "neat": NEATModel,
    "hyperneat": HyperNEATModel,
    "es-hyperneat": ESHyperNEATModel,
}


def model_type_from_string(model_type_str: str) -> Type[BaseModel]:
    """Infer the model type from a string.

    Parameters:
        model_type_str: str
            The string representation of the model type.

    Returns:
        model_type: Type[BaseModel]
            The model type.
    """
    model_type_str = model_type_str.lower()

    if model_type_str.lower() in MODEL_TYPES:
        return MODEL_TYPES[model_type_str]

    raise ValueError(f"Model type {model_type_str} not recognized")


def get_model_from_config(config: NeuropokerConfig) -> BaseModel:
    """Load a model from a configuration dictionary.

    Parameters:
        config: NeuropokerConfig
            The configuration dictionary.

    Returns:
        model: BaseModel
            The loaded model.
    """
    print("Creating model from config...")
    print()

    match config["model"]["type"]:
        case "neat":
            return NEATModel.from_config(config)
        case "hyperneat":
            return HyperNEATModel.from_config(config)
        case "es-hyperneat":
            return ESHyperNEATModel.from_config(config)
        case _:
            raise ValueError(f"Model type {config['type']} not recognized")


def get_model_from_pickle(model_file: Path) -> BaseModel:
    """Load a model from a model file.

    Parameters:
        model_file: Path
            The path to the model file.

    Returns:
        model: BaseModel
            The loaded model.
    """
    print(f"Loading model from {model_file}...")
    print()

    # Check if model file exists
    if not model_file.exists():
        raise FileNotFoundError(f"Model file {model_file} does not exist")

    # Load model file
    with model_file.open("rb") as f:
        model = pickle.load(f)

    if not issubclass(type(model), BaseModel):
        raise ValueError(
            f"Invalid model file {model_file}: Expected an object of "
            f"type BaseModel (or subclass), found {type(model)}"
        )

    return model


def save_model_to_pickle(model: BaseModel, model_file: Path) -> None:
    """Save a model to a file.

    Parameters:
        model: BaseModel
            The model to save.
        model_file: Path
            The path to the model file.
    """
    print(f"Saving model to {model_file}...")
    print()

    # Check if the model file's parent directory exists
    if not model_file.parent.exists():
        print(colored(f"Creating directory {model_file.parent}", color="yellow"))
        print()
        model_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if the model file exists
    if model_file.exists():
        print(colored(f"Overwriting {model_file}", color="yellow"))
        print()

    # Save the model
    with model_file.open("wb") as f:
        pickle.dump(model, f)


def load_model(
    model_type: Type[ModelT],
    model_file: Optional[Path] = None,
    **kwargs,
) -> ModelT:
    """Load a model from a file.

    Parameters:
        model_type: Type[ModelT]
            The type of model to load.
        model_file: Optional[Path]
            The path to the model file, if it exists.
        kwargs:
            Keyword arguments to pass to the model constructor.

    Returns:
        model: ModelT
            The loaded model.
    """
    # Check that model_type inherits from BaseModel
    if not issubclass(model_type, BaseModel):
        raise ValueError(
            f"Model type {model_type} invalid, must inherit from BaseModel"
        )

    if model_file is not None:
        return get_model_from_pickle(model_file)  # type: ignore

    return model_type(**kwargs)
