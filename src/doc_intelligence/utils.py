"""Utility functions for the Document Intelligence application."""

import logging
from typing import Optional, Tuple
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

logger = logging.getLogger(__name__)


def create_workspace_client(
    host: Optional[str] = None, token: Optional[str] = None
) -> Optional[WorkspaceClient]:
    """
    Create a Databricks WorkspaceClient with graceful degradation.

    Args:
        host: Databricks workspace host
        token: Databricks access token

    Returns:
        WorkspaceClient or None if creation fails
    """
    try:
        # Try to get configuration from environment or Databricks CLI
        sdk_config = Config()

        # If no config found, try to use provided parameters
        if not sdk_config.host and host:
            sdk_config.host = host
            sdk_config.token = token

        if not sdk_config.host:
            logger.warning("Databricks host not configured")
            return None

        client = WorkspaceClient(config=sdk_config)
        # Test the connection
        client.current_user.me()
        logger.info("Successfully connected to Databricks")
        return client

    except Exception as e:
        logger.error(f"Failed to create Databricks client: {str(e)}")
        return None


def get_current_user(client: Optional[WorkspaceClient] = None) -> str:
    """
    Get the current authenticated user's username with fallback.

    Args:
        client: Optional WorkspaceClient to use

    Returns:
        Username string
    """
    try:
        if client:
            current_user = client.current_user.me()
            username = current_user.user_name or "unknown_user"
            logger.info(f"Retrieved current user: {username}")
            return username
        else:
            logger.warning("No Databricks client available, using fallback user")
            return "demo_user@example.com"
    except Exception as e:
        logger.error(f"Failed to get current user: {str(e)}")
        return "demo_user@example.com"


def validate_databricks_connection(
    client: Optional[WorkspaceClient],
) -> Tuple[bool, str]:
    """
    Validate that Databricks connection is working.

    Args:
        client: WorkspaceClient to validate

    Returns:
        Tuple of (is_valid, message)
    """
    if not client:
        return False, "No Databricks client available"

    try:
        client.current_user.me()
        return True, "Connected to Databricks successfully"
    except Exception as e:
        return False, f"Databricks connection failed: {str(e)}"
