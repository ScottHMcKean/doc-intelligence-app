"""Configuration management for Document Intelligence application."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

logger = logging.getLogger(__name__)

# Detect Databricks environment
try:
    import IPython

    get_ipython = IPython.get_ipython
    if get_ipython() is not None and hasattr(get_ipython(), "user_ns"):
        DATABRICKS_ENVIRONMENT = True
    else:
        DATABRICKS_ENVIRONMENT = False
except ImportError:
    DATABRICKS_ENVIRONMENT = False

# Import dbutils if available
try:
    if DATABRICKS_ENVIRONMENT:
        from pyspark.sql import SparkSession
        from pyspark.dbutils import DBUtils

        spark = SparkSession.getActiveSession()
        if spark:
            dbutils = DBUtils(spark)
        else:
            dbutils = None
    else:
        dbutils = None
except:
    dbutils = None


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
    """Main configuration class with dot notation access and secret management."""

    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file and secrets.

        Args:
            config_path: Path to YAML config file.
        """
        # Determine the actual config path
        if not os.path.isabs(config_path):
            # Relative path - make it relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = str(project_root / config_path)

        config_data = load_config(config_path)
        super().__init__(config_data)

        # Service availability checks (cached)
        self._databricks_available = None
        self._database_available = None

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value from config file."""
        return self.get_nested(key_path, default)

    def _get_secret(self, secret_key: str) -> Optional[str]:
        """Get a secret value from Databricks secrets."""
        if dbutils and DATABRICKS_ENVIRONMENT:
            secrets_scope = self.get("application.secrets_scope", "doc-intelligence")
            try:
                value = dbutils.secrets.get(scope=secrets_scope, key=secret_key)
                if value:
                    logger.debug(
                        f"Retrieved secret '{secret_key}' from Databricks secrets"
                    )
                    return value
            except Exception as e:
                logger.debug(
                    f"Failed to get secret '{secret_key}' from Databricks: {e}"
                )
        return None

    @property
    def databricks_host(self) -> Optional[str]:
        """Get Databricks host from config file."""
        return self.get("databricks.host")

    @property
    def databricks_token(self) -> Optional[str]:
        """Get Databricks token from config file."""
        return self.get("databricks.token")

    @property
    def database_user(self) -> Optional[str]:
        """Get database user from config file."""
        return self.get("database.user")

    @property
    def database_password(self) -> Optional[str]:
        """Get database password dynamically using Databricks SDK."""
        # This will be generated dynamically when needed
        return None

    @property
    def database_connection_string(self) -> Optional[str]:
        """Get database connection string if credentials are available."""
        # We don't build the connection string here since we need dynamic credentials
        # The database service will handle this
        return None

    @property
    def databricks_available(self) -> bool:
        """Check if Databricks credentials are available."""
        if self._databricks_available is None:
            self._databricks_available = bool(
                self.databricks_host and self.databricks_token
            )
            if not self._databricks_available:
                logger.warning(
                    "Databricks credentials not found. Some features will be limited."
                )
        return self._databricks_available

    @property
    def database_available(self) -> bool:
        """Check if database credentials are available."""
        if self._database_available is None:
            self._database_available = bool(self.database_connection_string)
            if not self._database_available:
                logger.warning(
                    "Database credentials not found. Database features will be limited."
                )
        return self._database_available

    @property
    def effective_checkpointer_type(self) -> str:
        """Get the effective checkpointer type based on configuration and availability."""
        checkpointer_type = self.get("agent.checkpointer.type", "auto").lower()

        if checkpointer_type == "postgres":
            if self.database_available:
                return "postgres"
            else:
                logger.warning(
                    "Postgres checkpointer requested but not available, falling back to memory"
                )
                return "memory"
        elif checkpointer_type == "memory":
            return "memory"
        elif checkpointer_type == "auto":
            return "postgres" if self.database_available else "memory"
        else:
            logger.warning(
                f"Unknown checkpointer type '{checkpointer_type}', defaulting to memory"
            )
            return "memory"

    def get_status(self) -> Dict[str, Any]:
        """Get configuration status."""
        secrets_scope = self.get("auth.secrets_scope", "doc-intelligence")

        return {
            "application": {
                "name": self.get("application.name", "Document Intelligence"),
                "debug_mode": self.get("application.debug_mode", False),
                "log_level": self.get("application.log_level", "INFO"),
                "environment": "databricks" if DATABRICKS_ENVIRONMENT else "local",
            },
            "services": {
                "databricks_available": self.databricks_available,
                "database_available": self.database_available,
                "checkpointer_type": self.effective_checkpointer_type,
            },
            "secrets": {
                "using_dbutils": dbutils is not None,
                "secrets_scope": secrets_scope,
            },
            "config": {
                "file_loaded": len(self) > 0,
                "sections": (list(self.keys()) if isinstance(self, dict) else []),
            },
        }

    def reload_config(self, config_file_path: str):
        """Reload configuration from file."""
        logger.info("Reloading configuration...")

        # Determine the actual config path
        if not os.path.isabs(config_file_path):
            # Relative path - make it relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_file_path = str(project_root / config_file_path)

        config_data = load_config(config_file_path)
        self.clear()
        self.update(config_data)

        # Reset cached availability checks
        self._databricks_available = None
        self._database_available = None

        logger.info("Configuration reloaded successfully")
