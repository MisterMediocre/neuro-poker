"""Utility functions for models."""

from pathlib import Path
from typing import Dict, Final, Optional, Type

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
        return model_type.from_pickle(model_file)

    return model_type(**kwargs)
