"""Configuration management for Document Intelligence application."""

import os
import logging
from pathlib import Path
from typing import Any
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


class DotDict(dict):
    """Dictionary with dot notation access."""

    def __getattr__(self, name):
        try:
            value = self[name]
            if isinstance(value, dict):
                return DotDict(value)
            return value
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """Get nested value using dot notation (e.g., 'section.subsection.key')."""
        keys = key_path.split(".")
        value = self
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


class DocConfig(DotDict):
    """Simplified configuration class that uses DotDict"""

    def __init__(self, config_path: str):
        """Initialize configuration from YAML file."""
        # Determine the actual config path
        if not os.path.isabs(config_path):
            project_root = Path(__file__).parent.parent.parent
            config_path = str(project_root / config_path)

        self.config_path = config_path
        config_data = load_config(self.config_path)
        super().__init__(config_data)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        return self.get_nested(key_path, default)

    def reload_config(self):
        """Reload configuration from file."""
        config_data = load_config(self.config_path)
        self.clear()
        self.update(config_data)
        logger.info("Configuration reloaded successfully")


def check_config(config: DocConfig) -> bool:
    """
    Check if the config is valid.
    """
    assert isinstance(config, DocConfig)
    try:
        config.get("application.name")
        return True
    except Exception as e:
        return False
