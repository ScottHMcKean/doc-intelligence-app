"""Utility functions for the Document Intelligence application."""

import logging
from typing import Optional, Tuple
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

logger = logging.getLogger(__name__)


def get_workspace_client(
    host: Optional[str] = None, token: Optional[str] = None
) -> Optional[WorkspaceClient]:
    """
    Get authenticated Databricks workspace client with graceful error handling.

    This function combines the functionality of both create_workspace_client and
    get_databricks_client into a single, unified interface.

    Args:
        host: Optional Databricks workspace host. If not provided, will try to
              get from environment or Databricks CLI configuration.
        token: Optional Databricks access token. If not provided, will try to
               get from environment or Databricks CLI configuration.

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


def check_workspace_client(client: WorkspaceClient) -> bool:
    """
    Check if the workspace client is valid.
    """
    assert isinstance(client, WorkspaceClient)
    try:
        client.current_user.me()
        return True
    except Exception as e:
        return False


def get_current_user(client: Optional[WorkspaceClient] = None) -> Tuple[str, str]:
    """
    Get the current authenticated user's username and user id.

    Args:
        client: WorkspaceClient to use

    Returns:
        Tuple of (username, user_id)

    Raises:
        ValueError: If client is not provided or user cannot be retrieved.
    """
    try:
        current_user = client.current_user.me()
        username = current_user.user_name
        user_id = current_user.id
        if not username or not user_id:
            logger.error("User information is incomplete: username or user_id missing.")
            raise ValueError(
                "Could not retrieve username or user_id from Databricks client."
            )
        logger.info(f"Retrieved current user: {username} (id: {user_id})")
        return username, user_id
    except Exception as e:
        logger.error(f"Failed to get current user: {str(e)}")
        raise


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
    try:
        client.current_user.me()
        return True, "Connected to Databricks successfully"
    except Exception as e:
        return False, f"Databricks connection failed: {str(e)}"
