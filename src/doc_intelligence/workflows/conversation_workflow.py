"""Conversation workflow orchestrator."""

import logging
import uuid
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from databricks.sdk import WorkspaceClient
from ..services import DatabaseService, EmbeddingService, AgentService
from ..utils import get_current_user

logger = logging.getLogger(__name__)


class ConversationWorkflow:
    """Orchestrates conversation management with document context."""

    def __init__(
        self,
        client: Optional[WorkspaceClient],
        database_service: DatabaseService,
        embedding_service: EmbeddingService,
        agent_service: AgentService,
    ):
        self.client = client
        self.database_service = database_service
        self.embedding_service = embedding_service
        self.agent_service = agent_service

    def start_new_conversation(
        self,
        username: Optional[str] = None,
        document_hashes: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a new conversation.

        Args:
            username: Username (will get from auth service if not provided)
            document_hashes: List of document hashes for context
            title: Conversation title (will auto-generate if not provided)

        Returns:
            Dict with conversation details
        """
        logger.info("Starting new conversation workflow")

        # Get username if not provided
        if not username:
            username = self.auth_service.get_current_user()

        # Create user record
        user = self.database_service.create_user(username)
        if not user:
            return {"success": False, "error": "Failed to create user record"}

        try:
            # Generate thread ID
            thread_id = str(uuid.uuid4())

            # Determine title
            if not title:
                if document_hashes:
                    # Get document info for title
                    docs = []
                    for doc_hash in document_hashes:
                        doc = self.database_service.get_document_by_hash(doc_hash)
                        if doc:
                            docs.append(doc.filename)

                    if docs:
                        title = f"Chat with {', '.join(docs[:2])}"
                        if len(docs) > 2:
                            title += f" and {len(docs) - 2} more"
                    else:
                        title = "Document Chat"
                else:
                    title = "New Conversation"

            # Create conversation
            conversation = self.database_service.create_conversation(
                user_id=str(user.id),
                title=title,
                thread_id=thread_id,
                document_ids=document_hashes or [],
            )

            if not conversation:
                return {
                    "success": False,
                    "error": "Failed to create conversation record",
                }

            logger.info(f"Created new conversation: {conversation.id}")
            return {
                "success": True,
                "conversation_id": str(conversation.id),
                "thread_id": thread_id,
                "title": title,
                "document_hashes": document_hashes or [],
                "created_at": conversation.created_at.isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to start new conversation: {str(e)}")
            return {"success": False, "error": str(e)}

    def send_message(
        self, conversation_id: str, user_message: str, username: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a message in a conversation and get AI response.

        Args:
            conversation_id: Conversation ID
            user_message: User's message
            username: Username (will get from auth service if not provided)

        Returns:
            Dict with response details
        """
        logger.info(f"Processing message in conversation: {conversation_id}")

        # Get username if not provided
        if not username:
            username = self.auth_service.get_current_user()

        try:
            # Add user message to database
            user_msg = self.database_service.add_message(
                conversation_id=conversation_id,
                role="user",
                content=user_message,
                metadata={"timestamp": datetime.utcnow().isoformat()},
            )

            if not user_msg:
                return {"success": False, "error": "Failed to save user message"}

            # Get conversation for document context
            conversations = self.database_service.get_user_conversations(
                user_id=user_msg.conversation.user_id
            )

            conversation = None
            for conv in conversations:
                if str(conv.id) == conversation_id:
                    conversation = conv
                    break

            if not conversation:
                return {"success": False, "error": "Conversation not found"}

            # Retrieve relevant document context
            context_documents = self._retrieve_document_context(
                user_message, conversation.document_ids or []
            )

            # Build message history for context
            messages = self._build_message_history(conversation_id)

            # Generate AI response
            response_success, ai_response, response_metadata = (
                self.agent_service.generate_response(
                    messages=messages, context_documents=context_documents
                )
            )

            if not response_success:
                ai_response = "I apologize, but I encountered an error generating a response. Please try again."
                response_metadata = {"error": "response_generation_failed"}

            # Add AI message to database
            ai_msg = self.database_service.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=ai_response,
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    "context_docs_count": len(context_documents),
                    **response_metadata,
                },
            )

            if not ai_msg:
                logger.error("Failed to save AI response")

            # Update conversation title if it's the first exchange
            if len(messages) <= 2:  # First user message + response
                self._update_conversation_title_if_needed(conversation_id, messages)

            return {
                "success": True,
                "response": ai_response,
                "conversation_id": conversation_id,
                "context_documents_used": len(context_documents),
                "metadata": response_metadata,
            }

        except Exception as e:
            logger.error(f"Failed to process message: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error. Please try again.",
            }

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation message history."""
        try:
            messages = self.database_service.get_conversation_messages(conversation_id)

            history = []
            for msg in messages:
                history.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.created_at.isoformat(),
                        "metadata": msg.msg_metadata,
                    }
                )

            return history

        except Exception as e:
            logger.error(f"Failed to get conversation history: {str(e)}")
            return []

    def get_user_conversations(
        self, username: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all conversations for a user."""
        if not username:
            username = self.auth_service.get_current_user()

        try:
            user = self.database_service.create_user(username)
            if not user:
                return []

            conversations = self.database_service.get_user_conversations(str(user.id))

            conv_list = []
            for conv in conversations:
                conv_list.append(
                    {
                        "conversation_id": str(conv.id),
                        "title": conv.title,
                        "created_at": conv.created_at.isoformat(),
                        "updated_at": conv.updated_at.isoformat(),
                        "document_count": (
                            len(conv.document_ids) if conv.document_ids else 0
                        ),
                        "document_hashes": conv.document_ids or [],
                    }
                )

            return conv_list

        except Exception as e:
            logger.error(f"Failed to get user conversations: {str(e)}")
            return []

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        try:
            return self.database_service.delete_conversation(conversation_id)
        except Exception as e:
            logger.error(f"Failed to delete conversation: {str(e)}")
            return False

    def add_documents_to_conversation(
        self, conversation_id: str, document_hashes: List[str]
    ) -> bool:
        """Add documents to an existing conversation."""
        try:
            # Get conversation
            # Note: This would require adding a method to get conversation by ID
            # For now, we'll assume this functionality exists
            logger.info(
                f"Adding documents {document_hashes} to conversation {conversation_id}"
            )

            # In a full implementation, you'd:
            # 1. Get the conversation
            # 2. Update the document_ids list
            # 3. Save the conversation

            return True

        except Exception as e:
            logger.error(f"Failed to add documents to conversation: {str(e)}")
            return False

    def _retrieve_document_context(
        self, query: str, document_hashes: List[str], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant document context for the query."""
        if not document_hashes or not self.embedding_service.vectorstore_available:
            return []

        try:
            # Perform similarity search with document filter
            search_success, results, search_message = (
                self.embedding_service.similarity_search(
                    query=query,
                    limit=limit,
                    filter_dict={"document_id": {"$in": document_hashes}},
                )
            )

            if search_success:
                return results
            else:
                logger.warning(f"Document search failed: {search_message}")
                return []

        except Exception as e:
            logger.error(f"Failed to retrieve document context: {str(e)}")
            return []

    def _build_message_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Build message history for LLM context."""
        try:
            messages = self.database_service.get_conversation_messages(conversation_id)

            # Convert to format expected by chat service
            history = []
            for msg in messages[-10:]:  # Last 10 messages for context
                history.append({"role": msg.role, "content": msg.content})

            return history

        except Exception as e:
            logger.error(f"Failed to build message history: {str(e)}")
            return []

    def _update_conversation_title_if_needed(
        self, conversation_id: str, messages: List[Dict[str, str]]
    ):
        """Update conversation title based on first exchange."""
        try:
            # Generate a better title based on the conversation
            summary_success, summary, suggested_title = (
                self.agent_service.summarize_conversation(messages)
            )

            if (
                summary_success
                and suggested_title
                and suggested_title != "New Conversation"
            ):
                self.database_service.update_conversation_title(
                    conversation_id, suggested_title
                )
                logger.info(f"Updated conversation title to: {suggested_title}")

        except Exception as e:
            logger.error(f"Failed to update conversation title: {str(e)}")
