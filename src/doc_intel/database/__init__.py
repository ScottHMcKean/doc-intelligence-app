"""Database module for conversation history and document management."""

# Note: Old database classes are deprecated in favor of the new DatabaseService
# Only schema definitions are exposed for the new architecture

from .schema import (
    User, Conversation, Document, DocumentChunk, Message,
    create_tables, Base, DatabaseManager
)

__all__ = [
    "User", "Conversation", "Document", "DocumentChunk", "Message",
    "create_tables", "Base", "DatabaseManager"
]
