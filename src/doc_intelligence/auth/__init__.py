"""Authentication module for Databricks integration."""

from .databricks_auth import (
    get_databricks_client,
    get_current_user,
    validate_databricks_connection,
)

__all__ = [
    "get_databricks_client",
    "get_current_user",
    "validate_databricks_connection",
]
