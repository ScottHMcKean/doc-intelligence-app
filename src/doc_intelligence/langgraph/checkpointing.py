"""
LangGraph checkpointing utilities for Postgres persistence.

This module provides utilities for:
- Postgres-based checkpointing for LangGraph
- Conversation state persistence
- Thread management across sessions
"""

import logging
from typing import Optional
from ..config import MOCK_MODE

logger = logging.getLogger(__name__)

# Conditional imports for production mode
if not MOCK_MODE:
    try:
        from langgraph.checkpoint.postgres import PostgresSaver, AsyncPostgresSaver
    except ImportError as e:
        logger.warning(f"Production dependencies not available: {e}")
        PostgresSaver = None
        AsyncPostgresSaver = None
else:
    PostgresSaver = None
    AsyncPostgresSaver = None


def create_postgres_checkpointer(
    connection_string: str,
    async_mode: bool = False
) -> Optional[PostgresSaver]:
    """Create a Postgres checkpointer for LangGraph state persistence."""
    
    if MOCK_MODE:
        logger.info("Mock mode enabled - using in-memory checkpointing")
        return None
    
    try:
        if async_mode:
            checkpointer = AsyncPostgresSaver.from_conn_string(connection_string)
        else:
            checkpointer = PostgresSaver.from_conn_string(connection_string)
        
        # Setup database tables
        checkpointer.setup()
        
        logger.info(f"Created {'async' if async_mode else 'sync'} Postgres checkpointer")
        return checkpointer
        
    except Exception as e:
        logger.error(f"Error creating Postgres checkpointer: {e}")
        if MOCK_MODE:
            return None
        raise


class MockCheckpointer:
    """Mock checkpointer for development mode."""
    
    def __init__(self):
        self.memory = {}
    
    def setup(self):
        """Setup mock checkpointer - no-op."""
        pass
    
    def get(self, config):
        """Get checkpoint from memory."""
        thread_id = config.get("configurable", {}).get("thread_id")
        return self.memory.get(thread_id)
    
    def put(self, config, checkpoint):
        """Store checkpoint in memory."""
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id:
            self.memory[thread_id] = checkpoint
    
    def list(self, config):
        """List checkpoints - returns empty for mock."""
        return []
