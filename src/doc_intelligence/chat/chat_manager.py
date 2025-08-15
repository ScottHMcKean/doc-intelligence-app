"""Chat manager for orchestrating conversations and document interactions."""

import uuid
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd

from .databricks_llm import ChatDatabricks
from ..database import ConversationStore, DocumentStore


class ChatManager:
    """Manages chat conversations with document context."""

    def __init__(self):
        self.llm = ChatDatabricks()
        self.conversation_store = ConversationStore()
        self.document_store = DocumentStore()

    def start_new_conversation(
        self, username: str, document_hash: Optional[str] = None
    ) -> str:
        """Start a new conversation."""
        conversation_id = str(uuid.uuid4())

        # Create conversation in database
        self.conversation_store.create_conversation(
            conversation_id=conversation_id,
            username=username,
            title="New Conversation",
            document_hash=document_hash,
        )

        return conversation_id

    def send_message(
        self,
        conversation_id: str,
        user_message: str,
        username: str,
        chat_mode: str = "document",  # "document", "vector_search", "general"
        document_hash: Optional[str] = None,
        vector_search_config: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Send a message and get a response.

        Args:
            conversation_id: Unique conversation identifier
            user_message: User's message
            username: Username
            chat_mode: Type of chat ("document", "vector_search", "general")
            document_hash: Hash of document to chat with
            vector_search_config: Vector search configuration

        Returns:
            AI assistant's response
        """
        try:
            # Store user message
            self.conversation_store.add_message(
                conversation_id=conversation_id, role="user", content=user_message
            )

            # Get conversation history
            history = self.conversation_store.get_conversation_history(conversation_id)

            # Generate response based on chat mode
            if chat_mode == "document" and document_hash:
                response = self._chat_with_document(
                    user_message=user_message,
                    document_hash=document_hash,
                    conversation_history=history[
                        :-1
                    ],  # Exclude the message we just added
                )
            elif chat_mode == "vector_search" and vector_search_config:
                response = self._chat_with_vector_search(
                    user_message=user_message,
                    vector_search_config=vector_search_config,
                    conversation_history=history[:-1],
                )
            else:
                response = self._general_chat(
                    user_message=user_message, conversation_history=history[:-1]
                )

            # Store assistant response
            self.conversation_store.add_message(
                conversation_id=conversation_id, role="assistant", content=response
            )

            # Update conversation title if this is the first exchange
            if len(history) == 1:  # Only user message exists
                title = self.llm.generate_conversation_title(user_message)
                self.conversation_store.update_conversation_title(
                    conversation_id, title
                )

            return response

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            return "I apologize, but I encountered an error. Please try again."

    def _chat_with_document(
        self,
        user_message: str,
        document_hash: str,
        conversation_history: List[Dict[str, Any]],
    ) -> str:
        """Chat with a specific document."""
        # Get document chunks
        chunks_df = self.document_store.get_document_chunks(document_hash)

        if chunks_df.empty:
            return "I don't have access to the document chunks yet. The document may still be processing."

        # Search for relevant chunks
        relevant_chunks_df = self.document_store.search_chunks(
            doc_hash=document_hash, query=user_message
        )

        # If no relevant chunks found, use first few chunks as context
        if relevant_chunks_df.empty:
            context_chunks = chunks_df.head(5)["content"].tolist()
        else:
            context_chunks = relevant_chunks_df.head(5)["content"].tolist()

        # Convert conversation history to the format expected by LLM
        llm_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversation_history
            if msg["role"] in ["user", "assistant"]
        ]

        return self.llm.chat_with_context(
            user_message=user_message,
            context_chunks=context_chunks,
            conversation_history=llm_history,
        )

    def _chat_with_vector_search(
        self,
        user_message: str,
        vector_search_config: Dict[str, str],
        conversation_history: List[Dict[str, Any]],
    ) -> str:
        """Chat using vector search for context."""
        # Convert conversation history
        llm_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversation_history
            if msg["role"] in ["user", "assistant"]
        ]

        return self.llm.chat_with_vector_search(
            user_message=user_message,
            vector_search_endpoint=vector_search_config.get("endpoint", ""),
            vector_search_index=vector_search_config.get("index", ""),
            conversation_history=llm_history,
        )

    def _general_chat(
        self, user_message: str, conversation_history: List[Dict[str, Any]]
    ) -> str:
        """General chat without document context."""
        # Convert conversation history
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

        # Add conversation history
        for msg in conversation_history[-10:]:  # Last 10 messages
            if msg["role"] in ["user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current message
        messages.append({"role": "user", "content": user_message})

        return self.llm.chat_completion(messages)

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get formatted conversation history for display."""
        return self.conversation_store.get_conversation_history(conversation_id)

    def get_user_conversations(self, username: str) -> List[Dict[str, Any]]:
        """Get all conversations for a user."""
        return self.conversation_store.get_user_conversations(username)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        return self.conversation_store.delete_conversation(conversation_id)

    def get_user_documents(self, username: str) -> List[Dict[str, Any]]:
        """Get all documents for a user."""
        return self.document_store.get_user_documents(username)

    def get_document_info(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document."""
        try:
            query = """
            SELECT filename, status, created_at, processed_at, metadata
            FROM documents
            WHERE doc_hash = :doc_hash
            """

            results = self.document_store.client.execute_query(
                query, {"doc_hash": doc_hash}
            )

            if results:
                row = results[0]
                return {
                    "filename": row[0],
                    "status": row[1],
                    "created_at": row[2],
                    "processed_at": row[3],
                    "metadata": row[4],
                }
            return None

        except Exception as e:
            st.error(f"Failed to get document info: {str(e)}")
            return None
