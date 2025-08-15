"""
Enhanced database schema with pgvector support for document intelligence.

This module defines the database schema that supports:
- Vector embeddings using pgvector extension
- Document chunks with semantic search
- Conversation history with LangGraph integration
- User management and document metadata
"""

from typing import Optional, List, Dict, Any
import logging
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    Integer,
    DateTime,
    Boolean,
    JSON,
    ForeignKey,
    Index,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

Base = declarative_base()


class User(Base):
    """User table for authentication and session management."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    documents = relationship("Document", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")


class Document(Base):
    """Document metadata table."""
    
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_hash = Column(String(64), unique=True, nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    original_path = Column(String(1024), nullable=True)
    processed_path = Column(String(1024), nullable=True)
    status = Column(String(50), default="uploaded")  # uploaded, processing, processed, failed
    file_size = Column(Integer, nullable=True)
    content_type = Column(String(100), nullable=True)
    doc_metadata = Column(JSON, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")
    
    # Indexes
    __table_args__ = (
        Index("idx_documents_user_status", "user_id", "status"),
        Index("idx_documents_created", "created_at"),
    )


class DocumentChunk(Base):
    """Document chunks with vector embeddings for semantic search."""
    
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=True)  # Default to 768 dimensions
    chunk_metadata = Column(JSON, nullable=True)  # page_number, section, etc.
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    # Indexes for vector search
    __table_args__ = (
        Index("idx_chunks_document", "document_id"),
        Index("idx_chunks_embedding_hnsw", "embedding", postgresql_using="hnsw", postgresql_with={"m": 16, "ef_construction": 64}),
        Index("idx_chunks_embedding_ivfflat", "embedding", postgresql_using="ivfflat", postgresql_with={"lists": 100}),
    )


class Conversation(Base):
    """Conversation metadata for LangGraph integration."""
    
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String(512), nullable=False)
    thread_id = Column(String(255), unique=True, nullable=False, index=True)
    document_ids = Column(JSON, nullable=True)  # List of document IDs for context
    status = Column(String(50), default="active")  # active, archived, deleted
    conv_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")
    
    # Indexes
    __table_args__ = (
        Index("idx_conversations_user_status", "user_id", "status"),
        Index("idx_conversations_updated", "updated_at"),
    )


class Message(Base):
    """Individual messages within conversations."""
    
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    msg_metadata = Column(JSON, nullable=True)  # tokens, model_used, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    # Indexes
    __table_args__ = (
        Index("idx_messages_conversation", "conversation_id"),
        Index("idx_messages_created", "created_at"),
    )


class VectorSearchCache(Base):
    """Cache for vector search results to improve performance."""
    
    __tablename__ = "vector_search_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_hash = Column(String(64), unique=True, nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    query_embedding = Column(Vector(768), nullable=False)
    results = Column(JSON, nullable=False)  # Cached search results
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("idx_search_cache_expires", "expires_at"),
        Index("idx_search_cache_user", "user_id"),
    )


def create_database_engine(connection_string: str, echo: bool = False):
    """Create SQLAlchemy engine with proper configuration."""
    engine = create_engine(
        connection_string,
        echo=echo,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    return engine


def create_tables(engine):
    """Create all tables and extensions."""
    
    # Create pgvector extension
    with engine.connect() as conn:
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            logger.info("pgvector extension created/verified")
        except Exception as e:
            logger.warning(f"Could not create pgvector extension: {e}")
    
    # Create all tables
    Base.metadata.create_all(engine)
    logger.info("Database tables created successfully")


def get_session_factory(connection_string: str):
    """Get a session factory for database operations."""
    engine = create_database_engine(connection_string)
    create_tables(engine)
    return sessionmaker(bind=engine)


class DatabaseManager:
    """Enhanced database manager with vector search capabilities."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = create_database_engine(connection_string)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize database
        create_tables(self.engine)
    
    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()
    
    def create_user(self, username: str, email: Optional[str] = None) -> User:
        """Create or get a user."""
        with self.get_session() as session:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                user = User(username=username, email=email)
                session.add(user)
                session.commit()
                session.refresh(user)
            return user
    
    def create_document(
        self, 
        user_id: str, 
        doc_hash: str, 
        filename: str, 
        **kwargs
    ) -> Document:
        """Create a new document record."""
        with self.get_session() as session:
            document = Document(
                user_id=user_id,
                doc_hash=doc_hash,
                filename=filename,
                **kwargs
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            return document
    
    def create_conversation(
        self, 
        user_id: str, 
        title: str, 
        thread_id: str,
        document_ids: Optional[List[str]] = None
    ) -> Conversation:
        """Create a new conversation."""
        with self.get_session() as session:
            conversation = Conversation(
                user_id=user_id,
                title=title,
                thread_id=thread_id,
                document_ids=document_ids or []
            )
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            return conversation
    
    def store_document_chunks(
        self, 
        document_id: str, 
        chunks: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """Store document chunks with embeddings."""
        with self.get_session() as session:
            chunk_objects = []
            for i, chunk_data in enumerate(chunks):
                chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    content=chunk_data["content"],
                    embedding=chunk_data.get("embedding"),
                    chunk_metadata=chunk_data.get("metadata", {}),
                    token_count=chunk_data.get("token_count")
                )
                chunk_objects.append(chunk)
                session.add(chunk)
            
            session.commit()
            for chunk in chunk_objects:
                session.refresh(chunk)
            return chunk_objects
    
    def vector_search(
        self, 
        query_embedding: List[float], 
        limit: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[DocumentChunk]:
        """Perform vector similarity search."""
        with self.get_session() as session:
            query = session.query(DocumentChunk)
            
            if document_ids:
                query = query.filter(DocumentChunk.document_id.in_(document_ids))
            
            results = query.order_by(
                DocumentChunk.embedding.cosine_distance(query_embedding)
            ).limit(limit).all()
            
            return results
    
    def get_conversation_messages(self, conversation_id: str) -> List[Message]:
        """Get all messages for a conversation."""
        with self.get_session() as session:
            messages = session.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at).all()
            return messages
    
    def add_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str,
        metadata: Optional[Dict] = None
    ) -> Message:
        """Add a message to a conversation."""
        with self.get_session() as session:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                msg_metadata=metadata or {}
            )
            session.add(message)
            session.commit()
            session.refresh(message)
            return message
