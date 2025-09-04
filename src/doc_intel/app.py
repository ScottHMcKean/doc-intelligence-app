"""Main application interface for Document Intelligence."""

import logging
import time
import uuid
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime

from .database.service import DatabaseService
from .storage.service import StorageService
from .document.service import DocumentService
from .agent.service import AgentService
from .utils import (
    get_workspace_client,
    get_current_user,
    validate_databricks_connection,
)
from .config import DocConfig

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

    def __init__(self, config_path: str = "./config.yaml"):
        """Initialize the application with all services."""
        logger.info("Initializing Document Intelligence Application")

        # Initialize configuration
        self.config = DocConfig(config_path)

        # Create Databricks workspace client
        self.databricks_client = get_workspace_client(
            host=self.config.get("application.databricks_host"), token=None
        )

        # Initialize core services with consistent API
        self.database_service = DatabaseService(
            client=self.databricks_client, config=self.config
        )

        self.storage_service = StorageService(
            client=self.databricks_client, config=self.config
        )

        self.document_service = DocumentService(
            client=self.databricks_client, config=self.config
        )

        self.agent_service = AgentService(
            client=self.databricks_client,
            config=self.config,
        )

        logger.info("Document Intelligence Application initialized successfully")

    # System Status and Health

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and service availability."""
        # Validate Databricks connection
        databricks_valid, databricks_msg = validate_databricks_connection(
            self.databricks_client
        )

        # Validate database connection
        db_valid, db_msg = self.database_service.connection_live

        return {
            "services": {
                "database": {"available": self.database_service.connection_live},
                "agent": {
                    "available": self.agent_service.is_available,
                    "rag_available": self.agent_service.rag_available,
                    "vector_search_available": self.agent_service.rag_available,
                    "message": (
                        "AI agent and vector search capabilities available"
                        if self.agent_service.is_available
                        else "AI agent capabilities not available"
                    ),
                },
                "storage": {
                    "available": databricks_valid,  # Storage depends on Databricks client
                    "message": (
                        "Databricks connection available"
                        if databricks_valid
                        else "Databricks connection not available"
                    ),
                },
                "document": {
                    "available": databricks_valid,  # Document processing depends on Databricks client
                    "message": (
                        "Databricks connection available"
                        if databricks_valid
                        else "Databricks connection not available"
                    ),
                },
            },
            "configuration": {
                "name": self.config.get("application.name", "Document Intelligence"),
                "environment": (
                    "databricks"
                    if self.config.get("application.databricks_host")
                    else "local"
                ),
            },
            "overall_health": all(
                [
                    db_valid,  # Database must be available
                    databricks_valid,  # Databricks must be available
                    self.agent_service.is_available,  # Agent service must be available
                ]
            ),
        }

    def get_current_user(self) -> Tuple[str, str]:
        """Get current authenticated user."""
        return get_current_user(self.databricks_client)

    # Document Operations

    def upload_and_process_document(
        self, file_content: bytes, filename: str, username: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload and process a document end-to-end.

        This is the main entry point for document processing.
        """
        logger.info(f"Starting document processing for: {filename}")

        # Get username and user ID if not provided
        if not username:
            username, user_id = get_current_user(self.databricks_client)
        else:
            try:
                username, user_id = get_current_user(self.databricks_client)
            except:
                return {
                    "success": False,
                    "error": "Could not get current user information",
                    "stage": "user_creation",
                }

        # Create user record
        user = self.database_service.create_user(username, user_id)
        if not user:
            return {
                "success": False,
                "error": "Failed to create user record",
                "stage": "user_creation",
            }

        try:
            # Stage 1: Upload document
            upload_success, upload_path, upload_message = (
                self.storage_service.upload_file(
                    file_content=file_content,
                    file_name=filename,
                    volume_name="documents",
                )
            )

            if not upload_success:
                logger.warning(f"Upload failed but continuing: {upload_message}")

            # Stage 2: Create document record
            document = self.database_service.create_document(
                raw_path=upload_path,
                metadata={"filename": filename, "status": "uploaded"},
            )

            if not document:
                return {
                    "success": False,
                    "error": "Failed to create document record",
                    "stage": "document_creation",
                }

            # Stage 3: Queue processing job (if upload successful)
            run_id = None
            if upload_success:
                output_path = f"{Path(upload_path).parent}/processed"
                job_success, run_id, job_message = (
                    self.document_service.queue_document_processing(
                        input_path=upload_path,
                        output_path=str(output_path),
                        doc_hash=str(document.id),  # Use document ID instead of hash
                    )
                )

                if job_success:
                    # Update document status
                    self.database_service.update_document_status(
                        str(document.id), "processing"
                    )
                    logger.info(f"Job queued successfully: {run_id}")
                else:
                    logger.warning(f"Job queue failed: {job_message}")

            # Stage 4: Process content directly for immediate chunking
            # This allows the chat to work even if Databricks processing fails
            content_str = self._extract_text_content(file_content, filename)
            if content_str:
                self._process_content_immediately(str(document.id), content_str)

            return {
                "success": True,
                "document_id": str(document.id),
                "filename": filename,
                "upload_success": upload_success,
                "upload_message": upload_message,
                "run_id": run_id,
                "stage": "completed",
            }

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return {"success": False, "error": str(e), "stage": "workflow_error"}

    def upload_and_process_document_sync(
        self, file_content: bytes, filename: str, username: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload and process a document synchronously (for testing).

        This method waits for document processing to complete.
        """
        # First upload the document
        upload_result = self.upload_and_process_document(
            file_content=file_content, filename=filename, username=username
        )

        if not upload_result["success"]:
            return upload_result

        # Wait for processing to complete
        document_id = upload_result["document_id"]
        run_id = upload_result.get("run_id")

        # Poll for completion
        final_result = self.poll_document_processing(
            document_id=document_id, run_id=run_id, timeout_seconds=300
        )

        return final_result

    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """Get document processing status."""
        document = self.database_service.get_document_by_id(document_id)
        if not document:
            return {"success": False, "error": "Document not found"}

        return {
            "success": True,
            "document_id": document_id,
            "filename": document.get("metadata", {}).get("filename", "Unknown"),
            "status": document.get("metadata", {}).get("status", "Unknown"),
            "created_at": (
                document["created_at"].isoformat()
                if document.get("created_at")
                else "Unknown"
            ),
            "metadata": document.get("metadata", {}),
        }

    def poll_document_processing(
        self, document_id: str, run_id: Optional[int] = None, timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Poll for AI parsing completion by checking the database for processed results.
        The AI parsing job will append rows to the database with user_id and file_path.

        Returns:
            Dict with polling results
        """
        logger.info(f"Polling for AI parsing completion: {document_id}")

        # Get document
        document = self.database_service.get_document_by_id(document_id)
        if not document:
            return {
                "success": False,
                "error": "Document not found",
                "document_id": document_id,
            }

        # Extract file path and user ID for polling
        file_path = document.get("raw_path") or document.get("metadata", {}).get(
            "filename"
        )
        user_id = document.get("user_id")

        if not file_path or not user_id:
            return {
                "success": False,
                "error": "Document missing file path or user ID",
                "document_id": document_id,
            }

        logger.info(f"Polling database for file: {file_path}, user: {user_id}")

        # Poll the database every 2 seconds for AI parsing completion
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            # Check if AI parsing has completed by looking for processed chunks
            completion_result = self.database_service.check_ai_parsing_completion(
                file_path, user_id
            )

            if completion_result and completion_result.get("completed"):
                logger.info(f"AI parsing completed for document {document_id}")

                # Update document status to processed
                self.database_service.update_document_status(
                    str(document["id"]), "processed"
                )

                return {
                    "success": True,
                    "document_id": document_id,
                    "message": "AI parsing completed successfully",
                    "results": completion_result,
                    "total_chunks": completion_result.get("total_chunks", 0),
                }

            # Wait 2 seconds before next poll
            time.sleep(2)

            # Log progress every 10 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0:
                logger.info(f"Still polling... {int(elapsed)}s elapsed")

        # Timeout reached
        logger.warning(f"AI parsing polling timeout for document {document_id}")
        self.database_service.update_document_status(
            str(document["id"]), "failed", "AI parsing timeout"
        )

        return {
            "success": False,
            "error": "AI parsing timeout - no results found in database",
            "document_id": document_id,
        }

    def get_user_documents(self) -> List[Dict[str, Any]]:
        """Get all documents for a user."""

        username, user_id = self.get_current_user()

        user = self.database_service.create_user()
        if not user:
            return []

        documents = self.database_service.get_user_documents()

        doc_list = []
        for doc in documents:
            doc_list.append(
                {
                    "document_id": doc.id,
                    "filename": (
                        doc.metadata.get("filename", "Unknown")
                        if doc.metadata
                        else "Unknown"
                    ),
                    "status": (
                        doc.metadata.get("status", "Unknown")
                        if doc.metadata
                        else "Unknown"
                    ),
                    "created_at": doc.created_at.isoformat(),
                    "raw_path": doc.raw_path,
                    "processed_path": doc.processed_path,
                    "metadata": doc.metadata,
                }
            )

        return doc_list

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document."""
        document = self.database_service.get_document_by_id(document_id)
        if not document:
            return None

        return {
            "document_id": document_id,
            "filename": (
                document.get("metadata", {}).get("filename", "Unknown")
                if document.get("metadata")
                else "Unknown"
            ),
            "status": (
                document.get("metadata", {}).get("status", "Unknown")
                if document.get("metadata")
                else "Unknown"
            ),
            "created_at": (
                document["created_at"].isoformat()
                if document.get("created_at")
                else "Unknown"
            ),
            "raw_path": document.get("raw_path"),
            "processed_path": document.get("processed_path"),
            "metadata": document.get("metadata", {}),
        }

    # Conversation Operations

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
        logger.info("Starting new conversation")

        # Get username and user ID if not provided
        if not username:
            username, user_id = get_current_user(self.databricks_client)
        else:
            # If username provided, we need to get the user ID from Databricks
            # This is a simplified approach - in practice you might want to look up the user ID
            try:
                username, user_id = get_current_user(self.databricks_client)
            except:
                return {
                    "success": False,
                    "error": "Could not get current user information",
                }

        # Create user record
        user = self.database_service.create_user(username, user_id)
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
                    for document_id in document_hashes:
                        doc = self.database_service.get_document_by_id(document_id)
                        if doc:
                            filename = (
                                doc.get("metadata", {}).get("filename", "Unknown")
                                if doc.get("metadata")
                                else "Unknown"
                            )
                            docs.append(filename)

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
                conversation_id=thread_id,
                doc_ids=document_hashes or [],
                metadata={"title": title} if title else None,
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
                "title": title,
                "document_hashes": document_hashes or [],
                "created_at": conversation.created_at.isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to start new conversation: {str(e)}")
            return {"success": False, "error": str(e)}

    def send_chat_message(
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

        # Get username and user ID if not provided
        if not username:
            username, user_id = get_current_user(self.databricks_client)
        else:
            try:
                username, user_id = get_current_user(self.databricks_client)
            except:
                return {
                    "success": False,
                    "error": "Could not get current user information",
                }

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

            # Retrieve relevant document context using agent service
            # If conversation has specific documents, use those; otherwise search all user documents
            document_ids = conversation.document_ids or []
            context_documents = self._retrieve_document_context(
                user_message, document_ids
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
            username, user_id = get_current_user(self.databricks_client)
        else:
            try:
                username, user_id = get_current_user(self.databricks_client)
            except:
                return []

        try:
            user = self.database_service.create_user(username, user_id)
            if not user:
                return []

            conversations = self.database_service.get_user_conversations(user_id)

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

    def get_conversation_info(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific conversation."""
        return self.database_service.get_conversation_by_id_with_documents(
            conversation_id
        )

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
            logger.info(
                f"Adding documents {document_hashes} to conversation {conversation_id}"
            )

            # Use the database service to add documents
            return self.database_service.add_documents_to_conversation(
                conversation_id, document_hashes
            )

        except Exception as e:
            logger.error(f"Failed to add documents to conversation: {str(e)}")
            return False

    # Search and Discovery

    def search_documents(
        self, query: str, document_hashes: Optional[List[str]] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search across documents using vector similarity."""
        # If no specific documents provided, search across all user documents
        if not document_hashes:
            username, user_id = self.get_current_user()
            user_docs = self.database_service.get_user_documents_by_username(username)
            # Extract document IDs from user documents
            document_hashes = [
                doc["document_id"]
                for doc in user_docs
                if doc.get("status") == "processed"
            ]

        search_success, results, message = self.agent_service.search_documents(
            query=query, limit=limit, document_ids=document_hashes
        )

        if search_success:
            return results
        else:
            logger.warning(f"Document search failed: {message}")
            return []

    # Private Helper Methods

    def _extract_text_content(
        self, file_content: bytes, filename: str
    ) -> Optional[str]:
        """Extract text content from file for immediate processing."""
        try:
            # Simple text extraction - in production this would be more sophisticated
            file_extension = Path(filename).suffix.lower()

            if file_extension in [".txt", ".md"]:
                return file_content.decode("utf-8", errors="ignore")
            elif file_extension == ".pdf":
                # Placeholder - would use PyPDF2 or similar
                return f"[PDF Content] Filename: {filename}\nThis is placeholder text content extracted from the PDF file. In production, this would use proper PDF parsing."
            elif file_extension in [".doc", ".docx"]:
                # Placeholder - would use python-docx or similar
                return f"[Document Content] Filename: {filename}\nThis is placeholder text content extracted from the document file. In production, this would use proper document parsing."
            else:
                return f"[File Content] Filename: {filename}\nUnsupported file type for text extraction: {file_extension}"

        except Exception as e:
            logger.error(f"Failed to extract text content: {e}")
            return f"[Error] Failed to extract content from {filename}: {str(e)}"

    def _process_content_immediately(self, document_id: str, content: str):
        """Process content immediately for basic chunking and storage."""
        try:
            # Simple chunking - split by paragraphs or sentences
            chunks = self._simple_chunk_text(content)

            # Prepare chunk data for database storage
            chunk_data = []
            for i, chunk in enumerate(chunks):
                chunk_data.append(
                    {
                        "content": chunk,
                        "embedding": None,  # Embeddings are handled separately
                        "metadata": {
                            "chunk_index": i,
                            "processing_method": "immediate",
                            "chunk_type": "text",
                        },
                        "token_count": len(chunk.split()),  # Simple token count
                    }
                )

            # Store chunks in database
            self.database_service.store_document_chunks(document_id, chunk_data)
            logger.info(f"Stored {len(chunk_data)} chunks for immediate processing")

        except Exception as e:
            logger.error(f"Failed to process content immediately: {e}")

    def _simple_chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Simple text chunking by paragraphs and sentences."""
        # Split by double newlines first (paragraphs)
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed chunk size, save current chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())

        # If we have no chunks or chunks are too large, split by sentences
        if not chunks or any(len(chunk) > chunk_size * 2 for chunk in chunks):
            sentences = text.replace("\n", " ").split(". ")
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip() + ".")
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip() + ".")

        return chunks

    def _retrieve_document_context(
        self, query: str, document_hashes: List[str], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant document context for the query using agent service."""
        if not self.agent_service.rag_available:
            return []

        try:
            # If no specific documents provided, search across all user documents
            if not document_hashes:
                username, user_id = get_current_user(self.databricks_client)
                # Get all processed documents for the user
                user_docs = self.database_service.get_user_documents_by_username(
                    username
                )
                processed_docs = [
                    doc["document_id"]
                    for doc in user_docs
                    if doc.get("status") == "processed"
                ]
                document_hashes = processed_docs

            if not document_hashes:
                logger.info("No documents available for context retrieval")
                return []

            # Use agent service for similarity search
            search_success, results, search_message = (
                self.agent_service.similarity_search(
                    query=query,
                    limit=limit,
                    document_ids=document_hashes,
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
                self.database_service.update_conversation_title_by_id(
                    conversation_id, suggested_title
                )
                logger.info(f"Updated conversation title to: {suggested_title}")

        except Exception as e:
            logger.error(f"Failed to update conversation title: {str(e)}")

    # Utility Methods

    def validate_system(self) -> Dict[str, Any]:
        """Run system validation checks."""
        status = self.get_system_status()

        issues = []
        recommendations = []

        # Check critical services
        if not status["services"]["database"]["available"]:
            recommendations.append(
                "Configure PostgreSQL for persistent storage and vector search"
            )

        if not status["services"]["agent"]["rag_available"]:
            recommendations.append(
                "Configure Databricks embedding endpoint for vector search"
            )

        if not status["services"]["agent"]["available"]:
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
