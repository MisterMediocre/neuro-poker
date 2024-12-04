"""Base class for neuroevolution models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type, TypeVar

from neuropoker.config import Config as NeuropokerConfig

ModelT = TypeVar("ModelT", bound="BaseModel")


class BaseModel(ABC):
    """Base class for neuroevolution models."""

    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def from_config(cls: Type[ModelT], config: NeuropokerConfig) -> ModelT:
        """Create a model from a configuration object.

        Parameters:
            config: NeuropokerConfig
                The configuration object.

        Returns:
            model: ModelT
                The model created from the configuration.
        """

    @abstractmethod
    def print_config(self) -> None:
        """Print the configuration of the model."""
