"""
LangGraph checkpointing utilities for state persistence.

This module provides utilities for:
- Postgres-based checkpointing for LangGraph
- In-memory checkpointing for development/testing
- Conversation state persistence
- Thread management across sessions
"""

import logging
from typing import Optional, Union

# Config will be passed as parameter to functions that need it

logger = logging.getLogger(__name__)

# Import checkpointer classes
try:
    from langgraph.checkpoint.postgres import PostgresSaver, AsyncPostgresSaver
    from langgraph.checkpoint.memory import MemorySaver
except ImportError as e:
    logger.error(f"Failed to import LangGraph checkpointer dependencies: {e}")
    PostgresSaver = None
    AsyncPostgresSaver = None
    MemorySaver = None


def create_checkpointer(
    connection_string: Optional[str] = None,
    async_mode: bool = False,
    checkpointer_type: Optional[str] = None,
    config=None,
) -> Optional[Union[PostgresSaver, MemorySaver]]:
    """
    Create a checkpointer based on configuration.

    Args:
        connection_string: PostgreSQL connection string (required for postgres checkpointer)
        async_mode: Whether to use async postgres checkpointer
        checkpointer_type: Override config checkpointer type ("postgres", "memory", or None for auto)

    Returns:
        Checkpointer instance or None if creation fails
    """
    if config is None:
        # Backwards compatibility - import global config if none provided
        from ..config import DocConfig

        config = DocConfig("./config.yaml")

    effective_type = checkpointer_type or config.get(
        "agent.conversation.checkpointer_type", "auto"
    )

    if effective_type == "memory":
        return _create_memory_checkpointer()
    elif effective_type == "postgres":
        if not connection_string:
            # Connection string will be generated dynamically by database service
            logger.warning(
                "No connection string provided, postgres checkpointer not available"
            )
            return None
        if not connection_string:
            logger.error(
                "Postgres checkpointer requested but no connection string available"
            )
            logger.info("Falling back to memory checkpointer")
            return _create_memory_checkpointer()
        return _create_postgres_checkpointer(connection_string, async_mode)
    else:
        logger.error(f"Unknown checkpointer type: {effective_type}")
        return None


def _create_postgres_checkpointer(
    connection_string: str, async_mode: bool = False
) -> Optional[PostgresSaver]:
    """Create a Postgres checkpointer for LangGraph state persistence."""
    if not PostgresSaver or not AsyncPostgresSaver:
        logger.error("Postgres checkpointer dependencies not available")
        return None

    try:
        if async_mode:
            checkpointer = AsyncPostgresSaver.from_conn_string(connection_string)
        else:
            checkpointer = PostgresSaver.from_conn_string(connection_string)

        # Setup database tables
        checkpointer.setup()

        logger.info(
            f"Created {'async' if async_mode else 'sync'} Postgres checkpointer"
        )
        return checkpointer

    except Exception as e:
        logger.error(f"Error creating Postgres checkpointer: {e}")
        raise


def _create_memory_checkpointer() -> Optional[MemorySaver]:
    """Create an in-memory checkpointer for development/testing."""
    if not MemorySaver:
        logger.error("Memory checkpointer dependencies not available")
        return None

    try:
        checkpointer = MemorySaver()
        logger.info("Created in-memory checkpointer")
        return checkpointer

    except Exception as e:
        logger.error(f"Error creating memory checkpointer: {e}")
        raise


# Backwards compatibility alias
def create_postgres_checkpointer(
    connection_string: str, async_mode: bool = False
) -> Optional[PostgresSaver]:
    """Create a Postgres checkpointer for LangGraph state persistence."""
    return _create_postgres_checkpointer(connection_string, async_mode)
