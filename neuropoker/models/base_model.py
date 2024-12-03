"""Base class for neuroevolution models.
"""

import pickle
from pathlib import Path
from typing import Type, TypeVar

ModelT = TypeVar("ModelT", bound="BaseModel")


class BaseModel:
    """Base class for neuroevolution models."""

    def __init__(self):
        pass

    @classmethod
    def from_pickle(cls: Type[ModelT], pickle_file: Path) -> ModelT:
        """Load a model from a pickle file.

        Parameters:
            pickle_file: Path
                The path to the pickle file.

        Returns:
            model: ModelT
                The model.
        """
        if not pickle_file.exists():
            raise FileNotFoundError(f"File {pickle_file} does not exist")

        # Load pickle file.
        with pickle_file.open("rb") as f:
            model = pickle.load(f)

        # Check that the model is this *exact* type, i.e. not a child
        # type or something else.
        if not isinstance(model, cls):
            raise ValueError(
                f"Invalid pickle file {pickle_file}: "
                f"Expected {__class__}, found {type(model)}"
            )

        return model

    def to_pickle(self, pickle_file: Path) -> None:
        """Save the model to a pickle file.

        Parameters:
            pickle_file: Path
                The path to the pickle file.
        """
        if not pickle_file.parent.exists():
            print(f"Creating directory {pickle_file.parent}")
            pickle_file.parent.mkdir(parents=True, exist_ok=True)
        if pickle_file.exists():
            print(f"Overwriting {pickle_file}")

        print(f"Saving to {pickle_file}")
        with pickle_file.open("wb") as f:
            pickle.dump(self, f)
