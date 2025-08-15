"""Conversation history storage and management."""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st

from .postgres_client import get_postgres_client
from ..config import MOCK_MODE, MOCK_CONVERSATIONS, MOCK_MESSAGES


class ConversationStore:
    """Manages conversation history in PostgreSQL."""

    def __init__(self):
        self.client = get_postgres_client()
        self._ensure_tables_exist()

    def _ensure_tables_exist(self) -> None:
        """Create necessary tables if they don't exist."""
        if MOCK_MODE:
            # In mock mode, just initialize empty data structures
            return

        create_conversations_table = """
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            conversation_id VARCHAR(255) UNIQUE NOT NULL,
            username VARCHAR(255) NOT NULL,
            title VARCHAR(500),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            document_hash VARCHAR(255),
            metadata JSONB DEFAULT '{}'
        )
        """

        create_messages_table = """
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            conversation_id VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB DEFAULT '{}',
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        )
        """

        create_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_conversations_username ON conversations(username)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_doc_hash ON conversations(document_hash)",
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)",
        ]

        try:
            self.client.execute_update(create_conversations_table)
            self.client.execute_update(create_messages_table)

            for index_query in create_indexes:
                self.client.execute_update(index_query)

        except Exception as e:
            st.error(f"Failed to create database tables: {str(e)}")
            raise

    def create_conversation(
        self,
        conversation_id: str,
        username: str,
        title: Optional[str] = None,
        document_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a new conversation."""
        if MOCK_MODE:
            # In mock mode, just return success
            return True

        query = """
        INSERT INTO conversations (conversation_id, username, title, document_hash, metadata)
        VALUES (:conversation_id, :username, :title, :document_hash, :metadata)
        ON CONFLICT (conversation_id) DO NOTHING
        """

        try:
            self.client.execute_update(
                query,
                {
                    "conversation_id": conversation_id,
                    "username": username,
                    "title": title or "New Conversation",
                    "document_hash": document_hash,
                    "metadata": json.dumps(metadata or {}),
                },
            )
            return True
        except Exception as e:
            st.error(f"Failed to create conversation: {str(e)}")
            return False

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a message to a conversation."""
        if MOCK_MODE:
            # In mock mode, just return success
            return True

        # Update conversation timestamp
        update_conversation = """
        UPDATE conversations 
        SET updated_at = CURRENT_TIMESTAMP 
        WHERE conversation_id = :conversation_id
        """

        # Insert message
        insert_message = """
        INSERT INTO messages (conversation_id, role, content, metadata)
        VALUES (:conversation_id, :role, :content, :metadata)
        """

        try:
            self.client.execute_update(
                update_conversation, {"conversation_id": conversation_id}
            )

            self.client.execute_update(
                insert_message,
                {
                    "conversation_id": conversation_id,
                    "role": role,
                    "content": content,
                    "metadata": json.dumps(metadata or {}),
                },
            )
            return True
        except Exception as e:
            st.error(f"Failed to add message: {str(e)}")
            return False

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages in a conversation."""
        if MOCK_MODE:
            return MOCK_MESSAGES.get(conversation_id, []).copy()

        query = """
        SELECT role, content, timestamp, metadata
        FROM messages
        WHERE conversation_id = :conversation_id
        ORDER BY timestamp ASC
        """

        try:
            results = self.client.execute_query(
                query, {"conversation_id": conversation_id}
            )

            return [
                {
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2],
                    "metadata": json.loads(row[3]) if row[3] else {},
                }
                for row in results
            ]
        except Exception as e:
            st.error(f"Failed to get conversation history: {str(e)}")
            return []

    def get_user_conversations(self, username: str) -> List[Dict[str, Any]]:
        """Get all conversations for a user."""
        if MOCK_MODE:
            return MOCK_CONVERSATIONS.copy()

        query = """
        SELECT conversation_id, title, created_at, updated_at, document_hash, metadata
        FROM conversations
        WHERE username = :username
        ORDER BY updated_at DESC
        """

        try:
            results = self.client.execute_query(query, {"username": username})

            return [
                {
                    "conversation_id": row[0],
                    "title": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                    "document_hash": row[4],
                    "metadata": json.loads(row[5]) if row[5] else {},
                }
                for row in results
            ]
        except Exception as e:
            st.error(f"Failed to get user conversations: {str(e)}")
            return []

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title."""
        query = """
        UPDATE conversations 
        SET title = :title, updated_at = CURRENT_TIMESTAMP
        WHERE conversation_id = :conversation_id
        """

        try:
            self.client.execute_update(
                query, {"conversation_id": conversation_id, "title": title}
            )
            return True
        except Exception as e:
            st.error(f"Failed to update conversation title: {str(e)}")
            return False

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        delete_messages = """
        DELETE FROM messages WHERE conversation_id = :conversation_id
        """

        delete_conversation = """
        DELETE FROM conversations WHERE conversation_id = :conversation_id
        """

        try:
            self.client.execute_update(
                delete_messages, {"conversation_id": conversation_id}
            )

            self.client.execute_update(
                delete_conversation, {"conversation_id": conversation_id}
            )
            return True
        except Exception as e:
            st.error(f"Failed to delete conversation: {str(e)}")
            return False
