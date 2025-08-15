"""PostgreSQL client for Lakebase managed database."""

import os
from typing import Optional
from contextlib import contextmanager

import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ..config import MOCK_MODE


class PostgresClient:
    """Client for managing PostgreSQL database connections."""

    def __init__(self):
        self.engine = None
        self.session_factory = None
        self.mock_data = {}  # In-memory storage for mock mode
        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Initialize database connection."""
        if MOCK_MODE:
            print("🧪 PostgreSQL running in mock mode - using in-memory storage")
            return

        try:
            # Get database configuration from environment
            db_host = os.getenv("POSTGRES_HOST")
            db_port = os.getenv("POSTGRES_PORT", "5432")
            db_name = os.getenv("POSTGRES_DB", "doc_intelligence")
            db_user = os.getenv("POSTGRES_USER")
            db_password = os.getenv("POSTGRES_PASSWORD")

            if not all([db_host, db_user, db_password]):
                raise ValueError(
                    "Missing required environment variables: POSTGRES_HOST, "
                    "POSTGRES_USER, POSTGRES_PASSWORD"
                )

            # Create database URL
            db_url = (
                f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            )

            # Create engine with connection pooling
            self.engine = create_engine(
                db_url, poolclass=StaticPool, pool_pre_ping=True, echo=False
            )

            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            print("PostgreSQL connection established successfully")

        except Exception as e:
            st.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise

    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup."""
        if MOCK_MODE:
            # Yield a mock session object for mock mode
            yield None
            return

        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def execute_query(self, query: str, params: Optional[dict] = None):
        """Execute a raw SQL query."""
        if MOCK_MODE:
            # Return empty results for mock mode
            return []

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                return result.fetchall()
        except Exception as e:
            st.error(f"Query execution failed: {str(e)}")
            raise

    def execute_update(self, query: str, params: Optional[dict] = None) -> int:
        """Execute an update/insert/delete query."""
        if MOCK_MODE:
            # Return success for mock mode
            return 1

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                conn.commit()
                return result.rowcount
        except Exception as e:
            st.error(f"Update execution failed: {str(e)}")
            raise

    def test_connection(self) -> bool:
        """Test database connection."""
        if MOCK_MODE:
            return True

        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            st.error(f"Database connection test failed: {str(e)}")
            return False


# Global client instance
_postgres_client: Optional[PostgresClient] = None


@st.cache_resource
def get_postgres_client() -> PostgresClient:
    """Get a cached PostgreSQL client instance."""
    global _postgres_client
    if _postgres_client is None:
        _postgres_client = PostgresClient()
    return _postgres_client
