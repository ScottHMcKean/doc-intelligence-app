"""Authentication service for Databricks."""

import logging
from typing import Optional
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

from ..config import config

logger = logging.getLogger(__name__)


class AuthService:
    """Service for Databricks authentication with graceful degradation."""
    
    def __init__(self):
        self._client: Optional[WorkspaceClient] = None
        self._current_user: Optional[str] = None
        
    @property
    def is_available(self) -> bool:
        """Check if Databricks authentication is available."""
        return config.databricks_available
    
    def get_client(self) -> Optional[WorkspaceClient]:
        """Get authenticated Databricks workspace client."""
        if not self.is_available:
            logger.warning("Databricks credentials not available")
            return None
            
        if self._client is not None:
            return self._client
            
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

            self._client = WorkspaceClient(config=sdk_config)
            # Test the connection
            self._client.current_user.me()
            logger.info("Successfully connected to Databricks")
            return self._client
            
        except Exception as e:
            logger.error(f"Failed to create Databricks client: {str(e)}")
            self._client = None
            return None
    
    def get_current_user(self) -> str:
        """Get the current authenticated user's username with fallback."""
        if self._current_user is not None:
            return self._current_user
            
        try:
            client = self.get_client()
            if client:
                current_user = client.current_user.me()
                self._current_user = current_user.user_name or "unknown_user"
                logger.info(f"Retrieved current user: {self._current_user}")
                return self._current_user
            else:
                logger.warning("No Databricks client available, using fallback user")
                self._current_user = "demo_user@example.com"
                return self._current_user
        except Exception as e:
            logger.error(f"Failed to get current user: {str(e)}")
            self._current_user = "demo_user@example.com"
            return self._current_user
    
    def validate_connection(self) -> tuple[bool, str]:
        """
        Validate that Databricks connection is working.
        
        Returns:
            Tuple of (is_valid, message)
        """
        if not self.is_available:
            return False, "Databricks credentials not configured"

        try:
            client = self.get_client()
            if client:
                return True, "Connected to Databricks successfully"
            else:
                return False, "Failed to connect to Databricks"
        except Exception as e:
            return False, f"Databricks connection failed: {str(e)}"
    
    def clear_cache(self):
        """Clear cached client and user data."""
        self._client = None
        self._current_user = None
