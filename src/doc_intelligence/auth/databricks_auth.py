"""Databricks authentication utilities."""

import logging
from typing import Optional

import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

from ..config import config

logger = logging.getLogger(__name__)


@st.cache_resource
def get_databricks_client() -> Optional[WorkspaceClient]:
    """Get authenticated Databricks workspace client with graceful error handling."""
    if not config.databricks_available:
        logger.warning("Databricks credentials not available")
        return None

    try:
        # Try to get configuration from environment or Databricks CLI
        sdk_config = Config()

        # If no config found, try to use environment variables
        if not sdk_config.host:
            sdk_config.host = config.databricks_host
            sdk_config.token = config.databricks_token

        if not sdk_config.host:
            logger.error("Databricks host not configured")
            return None

        client = WorkspaceClient(config=sdk_config)
        # Test the connection
        client.current_user.me()
        logger.info("Successfully connected to Databricks")
        return client

    except Exception as e:
        logger.error(f"Failed to create Databricks client: {str(e)}")
        return None


def get_current_user() -> str:
    """Get the current authenticated user's username with fallback."""
    try:
        client = get_databricks_client()
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


def validate_databricks_connection() -> bool:
    """Validate that Databricks connection is working."""
    if not config.databricks_available:
        st.warning(
            "⚠️ Databricks credentials not configured. Some features will be limited."
        )
        return False

    try:
        client = get_databricks_client()
        if client:
            st.success("✅ Connected to Databricks successfully")
            return True
        else:
            st.error("❌ Failed to connect to Databricks")
            return False
    except Exception as e:
        st.error(f"❌ Databricks connection failed: {str(e)}")
        logger.error(f"Databricks validation failed: {str(e)}")
        return False
