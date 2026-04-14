"""Service integration module for connecting Document and Database services."""

import logging
from typing import Optional, Dict, Any, List, Tuple
from databricks.sdk import WorkspaceClient

from doc_intel.database.service import DatabaseService
from doc_intel.document.service import DocumentService
from doc_intel.agent.service import AgentService
from doc_intel.config import load_config

logger = logging.getLogger(__name__)


class ServiceIntegration:
    """Integration layer that connects all services together."""

    def __init__(
        self, client: Optional[WorkspaceClient] = None, config: Optional[dict] = None
    ):
        """Initialize service integration with optional client and config."""
        self.client = client or WorkspaceClient()
        self.config = config or load_config()

        # Initialize services
        self.db_service = DatabaseService(self.client, self.config)
        self.doc_service = DocumentService(self.client, self.config)
        self.agent_service = AgentService(self.client, self.config)

        logger.info("Service integration initialized")

    def process_document_with_database(
        self,
        input_path: str,
        output_path: str,
        doc_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str], str]:
        """
        Process a document and store chunks in the database.

        Args:
            input_path: Path to input document
            output_path: Path to processed document
            doc_hash: Unique document hash
            metadata: Additional metadata for the document

        Returns:
            Tuple of (success, document_id, message)
        """
        try:
            # Ensure user exists
            if not self.db_service.user_exists():
                logger.info("Creating user...")
                self.db_service.create_user()

            # Create document record
            logger.info(f"Creating document record for {doc_hash}")
            document = self.db_service.create_document(
                raw_path=input_path, processed_path=output_path, metadata=metadata or {}
            )

            if not document:
                return False, None, "Failed to create document record"

            # Process document (integrate with your document processing logic)
            logger.info(f"Processing document {document.id}")
            chunks = self._process_document_content(input_path, doc_hash)

            if not chunks:
                return False, document.id, "Failed to process document content"

            # Store chunks in database
            logger.info(f"Storing {len(chunks)} chunks for document {document.id}")
            if not self.db_service.store_document_chunks(document.id, chunks):
                return False, document.id, "Failed to store document chunks"

            logger.info(f"Document processing completed successfully: {document.id}")
            return (
                True,
                document.id,
                f"Document processed and stored with {len(chunks)} chunks",
            )

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return False, None, f"Document processing failed: {e}"

    def _process_document_content(
        self, input_path: str, doc_hash: str
    ) -> List[Dict[str, Any]]:
        """
        Process document content and return chunks.
        This is a placeholder - integrate with your actual document processing logic.
        """
        # Placeholder implementation - replace with your document processing logic
        chunks = [
            {
                "content": f"Processed content from {input_path} (hash: {doc_hash})",
                "page_ids": [1],
                "embedding": [0.1] * 384,  # Placeholder embedding vector
                "metadata": {
                    "source_file": input_path,
                    "doc_hash": doc_hash,
                    "chunk_index": 0,
                    "processed_at": "2024-01-01T00:00:00Z",
                },
            }
        ]

        # Add more chunks as needed based on your processing logic
        for i in range(1, 3):  # Example: create 3 chunks
            chunks.append(
                {
                    "content": f"Additional chunk {i} from {input_path}",
                    "page_ids": [i + 1],
                    "embedding": [0.1] * 384,
                    "metadata": {
                        "source_file": input_path,
                        "doc_hash": doc_hash,
                        "chunk_index": i,
                        "processed_at": "2024-01-01T00:00:00Z",
                    },
                }
            )

        return chunks

    def run_agent_workflow_with_database(
        self,
        conversation_id: str,
        user_message: str,
        doc_ids: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[str], str]:
        """
        Run agent workflow with database integration.

        Args:
            conversation_id: ID of the conversation
            user_message: User's message
            doc_ids: Optional list of document IDs to reference

        Returns:
            Tuple of (success, response_message, message)
        """
        try:
            # Ensure user exists
            if not self.db_service.user_exists():
                logger.info("Creating user...")
                self.db_service.create_user()

            # Get or create conversation
            conversation = self.db_service.get_conversation_by_id(conversation_id)
            if not conversation:
                logger.info(f"Creating new conversation: {conversation_id}")
                conversation = self.db_service.create_conversation(
                    conversation_id=conversation_id, doc_ids=doc_ids
                )
                if not conversation:
                    return False, None, "Failed to create conversation"

            # Add user message
            logger.info(f"Adding user message to conversation {conversation_id}")
            user_msg = self.db_service.add_message(
                conv_id=conversation_id, role="user", content={"text": user_message}
            )

            if not user_msg:
                return False, None, "Failed to add user message"

            # Run agent workflow (integrate with your agent logic)
            logger.info(f"Running agent workflow for conversation {conversation_id}")
            response = self._run_agent_logic(conversation_id, user_message, doc_ids)

            # Add assistant response
            logger.info(f"Adding assistant response to conversation {conversation_id}")
            assistant_msg = self.db_service.add_message(
                conv_id=conversation_id, role="assistant", content={"text": response}
            )

            if not assistant_msg:
                return False, None, "Failed to add assistant response"

            logger.info(f"Agent workflow completed for conversation {conversation_id}")
            return True, response, "Agent workflow completed successfully"

        except Exception as e:
            logger.error(f"Agent workflow failed: {e}")
            return False, None, f"Agent workflow failed: {e}"

    def _run_agent_logic(
        self,
        conversation_id: str,
        user_message: str,
        doc_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Run the actual agent logic.
        This is a placeholder - integrate with your actual agent workflow.
        """
        # Placeholder implementation - replace with your agent logic
        if doc_ids:
            return f"I understand you're asking about documents {doc_ids}. Based on the context, here's my response to: '{user_message}'"
        else:
            return f"Thank you for your message: '{user_message}'. How can I help you today?"

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        return self.db_service.get_document_chunks(document_id)

    def get_user_documents(self) -> List[Dict[str, Any]]:
        """Get all documents for the current user."""
        return self.db_service.get_user_documents()

    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation."""
        return self.db_service.get_conversation_messages(conversation_id)

    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents using vector similarity.
        This is a placeholder - implement with your vector search logic.
        """
        # Placeholder implementation - replace with your vector search logic
        logger.info(f"Searching documents with query: {query}")
        return []











