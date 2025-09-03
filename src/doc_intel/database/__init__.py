"""Database module for conversation history and document management."""

from .schema import (
    User,
    Conversation,
    Document,
    DocumentChunk,
    Message,
    create_tables,
    Base,
    DatabaseManager,
)
from .service import DatabaseService

__all__ = [
    "User",
    "Conversation",
    "Document",
    "DocumentChunk",
    "Message",
    "create_tables",
    "Base",
    "DatabaseManager",
    "DatabaseService",
]
