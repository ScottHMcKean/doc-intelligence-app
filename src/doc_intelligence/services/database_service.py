"""Database service for PostgreSQL operations."""

import logging
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from databricks.sdk import WorkspaceClient

from ..database.schema import (
    User,
    Conversation,
    Document,
    DocumentChunk,
    Message,
    create_tables,
    Base,
)

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for all database operations with graceful degradation."""

    def __init__(self, client: Optional[WorkspaceClient], config: dict):
        self.client = client
        self.config = config

        # Build connection string from config
        self.connection_string = self._build_connection_string()
        self.engine = None
        self.SessionLocal = None
        self._initialize()

    def _build_connection_string(self) -> Optional[str]:
        """Build database connection string from config and secrets."""
        # For now, we'll need to get these from the global config's secrets
        # This is a temporary approach until we refactor secret handling
        from ..config import config as global_config

        user = global_config.database_user
        password = global_config.database_password

        if not all([user, password]):
            return None

        host = self.config.get("host", "localhost")
        port = self.config.get("port", 5432)
        database = self.config.get("database", "doc_intelligence")

        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    def _initialize(self):
        """Initialize database connection."""
        if not self.connection_string:
            logger.warning("PostgreSQL connection string not available")
            return

        try:
            self.engine = create_engine(self.connection_string)
            self.SessionLocal = sessionmaker(bind=self.engine)
            # Create tables if they don't exist
            create_tables(self.engine)
            logger.info("Successfully initialized database connection")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            self.engine = None
            self.SessionLocal = None

    @property
    def is_available(self) -> bool:
        """Check if database is available."""
        return self.engine is not None

    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup."""
        if not self.is_available:
            yield None
            return

        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test database connection.

        Returns:
            Tuple of (success, message)
        """
        if not self.is_available:
            return False, "Database not configured"

        try:
            with self.get_session() as session:
                if session:
                    session.execute(text("SELECT 1"))
                    return True, "Database connection successful"
                else:
                    return False, "Database session not available"
        except Exception as e:
            return False, f"Database connection failed: {str(e)}"

    # User operations
    def create_user(self, username: str) -> Optional[User]:
        """Create or get existing user."""
        if not self.is_available:
            return None

        try:
            with self.get_session() as session:
                if not session:
                    return None

                # Check if user exists
                user = session.query(User).filter(User.username == username).first()
                if user:
                    return user

                # Create new user
                user = User(username=username)
                session.add(user)
                session.commit()
                session.refresh(user)
                logger.info(f"Created new user: {username}")
                return user
        except Exception as e:
            logger.error(f"Failed to create user {username}: {str(e)}")
            return None

    # Conversation operations
    def create_conversation(
        self,
        user_id: str,
        title: str,
        thread_id: str,
        document_ids: Optional[List[str]] = None,
    ) -> Optional[Conversation]:
        """Create a new conversation."""
        if not self.is_available:
            return None

        try:
            with self.get_session() as session:
                if not session:
                    return None

                conversation = Conversation(
                    user_id=user_id,
                    title=title,
                    thread_id=thread_id,
                    document_ids=document_ids or [],
                )
                session.add(conversation)
                session.commit()
                session.refresh(conversation)
                logger.info(f"Created conversation: {conversation.id}")
                return conversation
        except Exception as e:
            logger.error(f"Failed to create conversation: {str(e)}")
            return None

    def get_user_conversations(self, user_id: str) -> List[Conversation]:
        """Get all conversations for a user."""
        if not self.is_available:
            return []

        try:
            with self.get_session() as session:
                if not session:
                    return []

                conversations = (
                    session.query(Conversation)
                    .filter(
                        Conversation.user_id == user_id, Conversation.status == "active"
                    )
                    .order_by(Conversation.updated_at.desc())
                    .all()
                )

                return conversations
        except Exception as e:
            logger.error(f"Failed to get conversations for user {user_id}: {str(e)}")
            return []

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title."""
        if not self.is_available:
            return False

        try:
            with self.get_session() as session:
                if not session:
                    return False

                conversation = (
                    session.query(Conversation)
                    .filter(Conversation.id == conversation_id)
                    .first()
                )

                if conversation:
                    conversation.title = title
                    session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to update conversation title: {str(e)}")
            return False

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if not self.is_available:
            return False

        try:
            with self.get_session() as session:
                if not session:
                    return False

                conversation = (
                    session.query(Conversation)
                    .filter(Conversation.id == conversation_id)
                    .first()
                )

                if conversation:
                    conversation.status = "deleted"
                    session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete conversation: {str(e)}")
            return False

    # Message operations
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> Optional[Message]:
        """Add a message to a conversation."""
        if not self.is_available:
            return None

        try:
            with self.get_session() as session:
                if not session:
                    return None

                message = Message(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    msg_metadata=metadata or {},
                )
                session.add(message)
                session.commit()
                session.refresh(message)
                return message
        except Exception as e:
            logger.error(f"Failed to add message: {str(e)}")
            return None

    def get_conversation_messages(self, conversation_id: str) -> List[Message]:
        """Get all messages for a conversation."""
        if not self.is_available:
            return []

        try:
            with self.get_session() as session:
                if not session:
                    return []

                messages = (
                    session.query(Message)
                    .filter(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at)
                    .all()
                )

                return messages
        except Exception as e:
            logger.error(
                f"Failed to get messages for conversation {conversation_id}: {str(e)}"
            )
            return []

    # Document operations
    def create_document(
        self, user_id: str, doc_hash: str, filename: str, status: str = "uploaded"
    ) -> Optional[Document]:
        """Create a new document record."""
        if not self.is_available:
            return None

        try:
            with self.get_session() as session:
                if not session:
                    return None

                document = Document(
                    user_id=user_id, doc_hash=doc_hash, filename=filename, status=status
                )
                session.add(document)
                session.commit()
                session.refresh(document)
                logger.info(f"Created document: {document.id}")
                return document
        except Exception as e:
            logger.error(f"Failed to create document: {str(e)}")
            return None

    def update_document_status(
        self, document_id: str, status: str, error: Optional[str] = None
    ) -> bool:
        """Update document status."""
        if not self.is_available:
            return False

        try:
            with self.get_session() as session:
                if not session:
                    return False

                document = (
                    session.query(Document).filter(Document.id == document_id).first()
                )

                if document:
                    document.status = status
                    if error:
                        document.doc_metadata = document.doc_metadata or {}
                        document.doc_metadata["error"] = error
                    session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to update document status: {str(e)}")
            return False

    def get_document_by_hash(self, doc_hash: str) -> Optional[Document]:
        """Get document by hash."""
        if not self.is_available:
            return None

        try:
            with self.get_session() as session:
                if not session:
                    return None

                document = (
                    session.query(Document)
                    .filter(Document.doc_hash == doc_hash)
                    .first()
                )

                return document
        except Exception as e:
            logger.error(f"Failed to get document by hash {doc_hash}: {str(e)}")
            return None

    def get_user_documents(self, user_id: str) -> List[Document]:
        """Get all documents for a user."""
        if not self.is_available:
            return []

        try:
            with self.get_session() as session:
                if not session:
                    return []

                documents = (
                    session.query(Document)
                    .filter(Document.user_id == user_id)
                    .order_by(Document.created_at.desc())
                    .all()
                )

                return documents
        except Exception as e:
            logger.error(f"Failed to get documents for user {user_id}: {str(e)}")
            return []

    # Document chunk operations
    def store_document_chunks(
        self, document_id: str, chunks: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """Store document chunks with embeddings."""
        if not self.is_available:
            return []

        try:
            with self.get_session() as session:
                if not session:
                    return []

                chunk_objects = []
                for i, chunk_data in enumerate(chunks):
                    chunk = DocumentChunk(
                        document_id=document_id,
                        chunk_index=i,
                        content=chunk_data["content"],
                        embedding=chunk_data.get("embedding"),
                        chunk_metadata=chunk_data.get("metadata", {}),
                        token_count=chunk_data.get("token_count"),
                    )
                    chunk_objects.append(chunk)
                    session.add(chunk)

                session.commit()
                for chunk in chunk_objects:
                    session.refresh(chunk)

                logger.info(
                    f"Stored {len(chunk_objects)} chunks for document {document_id}"
                )
                return chunk_objects
        except Exception as e:
            logger.error(f"Failed to store document chunks: {str(e)}")
            return []

    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        if not self.is_available:
            return []

        try:
            with self.get_session() as session:
                if not session:
                    return []

                chunks = (
                    session.query(DocumentChunk)
                    .filter(DocumentChunk.document_id == document_id)
                    .order_by(DocumentChunk.chunk_index)
                    .all()
                )

                return chunks
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {str(e)}")
            return []
