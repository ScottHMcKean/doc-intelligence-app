"""Configuration settings for the application."""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ServiceConfig:
    """Configuration for external services with graceful degradation."""
    
    def __init__(self):
        # Databricks Configuration
        self.databricks_host = os.getenv("DATABRICKS_HOST")
        self.databricks_token = os.getenv("DATABRICKS_TOKEN")
        self.databricks_volume_path = os.getenv("DATABRICKS_VOLUME_PATH", "/Volumes/main/default/documents")
        self.databricks_job_id = os.getenv("DATABRICKS_JOB_ID")
        self.databricks_llm_endpoint = os.getenv("DATABRICKS_LLM_ENDPOINT")
        self.databricks_embedding_endpoint = os.getenv("DATABRICKS_EMBEDDING_ENDPOINT")
        
        # PostgreSQL Configuration
        self.postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        self.postgres_port = os.getenv("POSTGRES_PORT", "5432")
        self.postgres_db = os.getenv("POSTGRES_DB", "doc_intelligence")
        self.postgres_user = os.getenv("POSTGRES_USER")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD")
        
        # Application Configuration
        self.app_name = os.getenv("APP_NAME", "Document Intelligence")
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Service availability checks
        self._databricks_available = None
        self._postgres_available = None
        
    @property
    def postgres_connection_string(self) -> Optional[str]:
        """Get PostgreSQL connection string if credentials are available."""
        if not all([self.postgres_user, self.postgres_password]):
            return None
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def databricks_available(self) -> bool:
        """Check if Databricks credentials are available."""
        if self._databricks_available is None:
            self._databricks_available = bool(self.databricks_host and self.databricks_token)
            if not self._databricks_available:
                logger.warning("Databricks credentials not found. Some features will be limited.")
        return self._databricks_available
    
    @property
    def postgres_available(self) -> bool:
        """Check if PostgreSQL credentials are available."""
        if self._postgres_available is None:
            self._postgres_available = bool(self.postgres_connection_string)
            if not self._postgres_available:
                logger.warning("PostgreSQL credentials not found. Some features will be limited.")
        return self._postgres_available
    
    def get_status(self) -> Dict[str, Any]:
        """Get configuration status."""
        return {
            "databricks_available": self.databricks_available,
            "postgres_available": self.postgres_available,
            "debug_mode": self.debug_mode,
            "app_name": self.app_name
        }

# Global configuration instance
config = ServiceConfig()
