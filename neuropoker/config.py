"""Read config files for models/games.
"""

from pathlib import Path
from typing import Any, Dict, Final

import toml


class Config:
    def __init__(self, config_file: Path | str) -> None:
        """Initialize the config.

        Parameters:
            config_file: Path | str
                The path to the configuration file.
        """
        self.config_file: Final[Path] = Path(config_file)
        self.config: Final[Dict[str, Any]] = toml.load(config_file)

    # Access it as if it were just another dict
    def __getitem__(self, key: str) -> Any:
        return self.config[key]
