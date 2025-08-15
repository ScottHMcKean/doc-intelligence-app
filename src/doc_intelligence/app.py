"""Main application interface for Document Intelligence."""

import logging
from typing import Optional, Dict, Any, List

from .services import (
    AuthService,
    StorageService,
    ProcessingService,
    DatabaseService,
    EmbeddingService,
    ChatService,
)
from .workflows import DocumentWorkflow, ConversationWorkflow
from .config import config

logger = logging.getLogger(__name__)


class DocumentIntelligenceApp:
    """
    Main application interface that orchestrates all document intelligence operations.

    This is the single entry point that can be used by:
    - Streamlit UI
    - API endpoints
    - Background jobs
    - Command line tools
    """

    def __init__(self):
        """Initialize the application with all services and workflows."""
        logger.info("Initializing Document Intelligence Application")

        # Initialize core services
        self.auth_service = AuthService()
        self.database_service = DatabaseService()
        self.embedding_service = EmbeddingService()
        self.chat_service = ChatService()

        # Initialize services that depend on auth
        databricks_client = self.auth_service.get_client()
        self.storage_service = StorageService(client=databricks_client)
        self.processing_service = ProcessingService(client=databricks_client)

        # Initialize vector store if available
        if self.database_service.is_available:
            self.embedding_service.init_vectorstore(
                self.database_service.connection_string
            )

        # Initialize workflows
        self.document_workflow = DocumentWorkflow(
            auth_service=self.auth_service,
            storage_service=self.storage_service,
            processing_service=self.processing_service,
            database_service=self.database_service,
            embedding_service=self.embedding_service,
        )

        self.conversation_workflow = ConversationWorkflow(
            auth_service=self.auth_service,
            database_service=self.database_service,
            embedding_service=self.embedding_service,
            chat_service=self.chat_service,
        )

        logger.info("Document Intelligence Application initialized successfully")

    # System Status and Health

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and service availability."""
        auth_valid, auth_msg = self.auth_service.validate_connection()
        db_valid, db_msg = self.database_service.test_connection()

        return {
            "services": {
                "databricks_auth": {"available": auth_valid, "message": auth_msg},
                "database": {"available": db_valid, "message": db_msg},
                "embeddings": {
                    "available": self.embedding_service.is_available,
                    "vectorstore_available": self.embedding_service.vectorstore_available,
                    "message": (
                        "Databricks embedding endpoint configured"
                        if self.embedding_service.is_available
                        else "Databricks embedding endpoint not configured"
                    ),
                },
                "chat": {
                    "available": self.chat_service.is_available,
                    "message": (
                        "Databricks LLM endpoint configured"
                        if self.chat_service.is_available
                        else "Databricks LLM endpoint not configured"
                    ),
                },
                "storage": {
                    "available": self.auth_service.is_available,  # Storage depends on auth
                    "message": (
                        "Databricks connection available"
                        if self.auth_service.is_available
                        else "Databricks connection not available"
                    ),
                },
                "processing": {
                    "available": self.auth_service.is_available,  # Processing depends on auth
                    "message": (
                        "Databricks connection available"
                        if self.auth_service.is_available
                        else "Databricks connection not available"
                    ),
                },
            },
            "configuration": config.get_status(),
            "overall_health": all(
                [
                    auth_valid
                    or not config.databricks_available,  # OK if not configured
                    db_valid or not config.postgres_available,  # OK if not configured
                    True,  # Other services degrade gracefully
                ]
            ),
        }

    def get_current_user(self) -> str:
        """Get current authenticated user."""
        return self.auth_service.get_current_user()

    # Document Operations

    def upload_and_process_document(
        self, file_content: bytes, filename: str, username: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload and process a document end-to-end.

        This is the main entry point for document processing.
        """
        return self.document_workflow.process_document(
            file_content=file_content, filename=filename, username=username
        )

    def get_document_status(self, doc_hash: str) -> Dict[str, Any]:
        """Get document processing status."""
        return self.document_workflow.get_document_status(doc_hash)

    def poll_document_processing(
        self, doc_hash: str, run_id: Optional[int] = None, timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """Poll for document processing completion."""
        return self.document_workflow.poll_and_finalize_processing(
            doc_hash=doc_hash, run_id=run_id, timeout_seconds=timeout_seconds
        )

    def get_user_documents(
        self, username: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all documents for a user."""
        if not username:
            username = self.get_current_user()

        user = self.database_service.create_user(username)
        if not user:
            return []

        documents = self.database_service.get_user_documents(str(user.id))

        doc_list = []
        for doc in documents:
            doc_list.append(
                {
                    "doc_hash": doc.doc_hash,
                    "filename": doc.filename,
                    "status": doc.status,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat(),
                    "file_size": doc.file_size,
                    "content_type": doc.content_type,
                    "metadata": doc.doc_metadata,
                }
            )

        return doc_list

    def get_document_info(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document."""
        document = self.database_service.get_document_by_hash(doc_hash)
        if not document:
            return None

        return {
            "doc_hash": doc_hash,
            "filename": document.filename,
            "status": document.status,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
            "file_size": document.file_size,
            "content_type": document.content_type,
            "metadata": document.doc_metadata,
        }

    # Conversation Operations

    def start_new_conversation(
        self,
        username: Optional[str] = None,
        document_hashes: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a new conversation."""
        return self.conversation_workflow.start_new_conversation(
            username=username, document_hashes=document_hashes, title=title
        )

    def send_chat_message(
        self, conversation_id: str, user_message: str, username: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a message in a conversation."""
        return self.conversation_workflow.send_message(
            conversation_id=conversation_id,
            user_message=user_message,
            username=username,
        )

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation message history."""
        return self.conversation_workflow.get_conversation_history(conversation_id)

    def get_user_conversations(
        self, username: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all conversations for a user."""
        return self.conversation_workflow.get_user_conversations(username)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        return self.conversation_workflow.delete_conversation(conversation_id)

    def add_documents_to_conversation(
        self, conversation_id: str, document_hashes: List[str]
    ) -> bool:
        """Add documents to an existing conversation."""
        return self.conversation_workflow.add_documents_to_conversation(
            conversation_id=conversation_id, document_hashes=document_hashes
        )

    # Search and Discovery

    def search_documents(
        self, query: str, document_hashes: Optional[List[str]] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search across documents using vector similarity."""
        filter_dict = None
        if document_hashes:
            filter_dict = {"document_id": {"$in": document_hashes}}

        search_success, results, message = self.embedding_service.similarity_search(
            query=query, limit=limit, filter_dict=filter_dict
        )

        if search_success:
            return results
        else:
            logger.warning(f"Document search failed: {message}")
            return []

    # Utility Methods

    def validate_system(self) -> Dict[str, Any]:
        """Run system validation checks."""
        status = self.get_system_status()

        issues = []
        recommendations = []

        # Check critical services
        if not status["services"]["database"]["available"]:
            if config.postgres_available:
                issues.append(
                    "PostgreSQL connection failed despite credentials being configured"
                )
            else:
                recommendations.append(
                    "Configure PostgreSQL for persistent storage and vector search"
                )

        if not status["services"]["databricks_auth"]["available"]:
            if config.databricks_available:
                issues.append(
                    "Databricks connection failed despite credentials being configured"
                )
            else:
                recommendations.append("Configure Databricks for full AI capabilities")

        if not status["services"]["embeddings"]["available"]:
            recommendations.append(
                "Configure Databricks embedding endpoint for vector search"
            )

        if not status["services"]["chat"]["available"]:
            recommendations.append(
                "Configure Databricks LLM endpoint for AI chat responses"
            )

        return {
            "overall_health": status["overall_health"],
            "issues": issues,
            "recommendations": recommendations,
            "services": status["services"],
        }

    def cleanup_resources(self):
        """Cleanup resources (for graceful shutdown)."""
        logger.info("Cleaning up Document Intelligence Application resources")
        # Add any cleanup logic here if needed
