"""Database service for PostgreSQL operations."""

import logging
import uuid
import psycopg2
import psycopg2.extras
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
from databricks.sdk import WorkspaceClient
from datetime import datetime, timezone

# SQLAlchemy imports for schema-based operations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .schema import Base, User, Document, DocumentChunk, Conversation, Message

logger = logging.getLogger(__name__)


class DatabaseService:
    """Simplified database service using direct psycopg2 connections."""

    def __init__(self, client: Optional[WorkspaceClient], config: dict):
        self.client = client
        self.config = config

    @contextmanager
    def get_connection(self):
        """Get a database connection using the simple psycopg2 pattern."""
        try:
            # Get database instance name from config
            instance_name = self.config.get("database.instance_name")
            user = self.config.get("database.user", "databricks")
            database = self.config.get("database.database", "databricks_postgres")

            if not instance_name:
                logger.error("No database instance_name configured")
                yield None
                return

            # Get database instance from Databricks (following the user's example)
            instance = self.client.database.get_database_instance(name=instance_name)
            cred = self.client.database.generate_database_credential(
                request_id=str(uuid.uuid4()), instance_names=[instance_name]
            )

            # Create psycopg2 connection (following the user's example)
            conn = psycopg2.connect(
                host=instance.read_write_dns,
                dbname=database,
                user=user,
                password=cred.token,
                sslmode="require",
            )

            try:
                yield conn
            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Failed to create database connection: {str(e)}")
            yield None

    def test_connection(self) -> Tuple[bool, str]:
        """Test database connection."""
        try:
            with self.get_connection() as conn:
                if conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        result = cur.fetchone()
                    return True, "Database connection successful"
                else:
                    return False, "Database connection not available"
        except Exception as e:
            return False, f"Database connection failed: {str(e)}"

    def create_tables(self) -> bool:
        """Create basic database tables if they don't exist."""
        try:
            with self.get_connection() as conn:
                if not conn:
                    return False

                with conn.cursor() as cur:
                    # Create pgvector extension first
                    try:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        logger.info("pgvector extension created/verified")
                    except Exception as e:
                        logger.warning(f"Could not create pgvector extension: {e}")

                    # Create users table - use Databricks user ID as primary key
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY, -- Databricks user ID
                            username VARCHAR(255) UNIQUE NOT NULL,
                            email VARCHAR(255),
                            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                            updated_at TIMESTAMP WITH TIME ZONE NOT NULL
                        )
                    """
                    )

                    # Create documents table
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS documents (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            doc_hash VARCHAR(64) UNIQUE NOT NULL,
                            filename VARCHAR(512) NOT NULL,
                            original_path VARCHAR(1024),
                            processed_path VARCHAR(1024),
                            status VARCHAR(50) DEFAULT 'uploaded',
                            file_size INTEGER,
                            content_type VARCHAR(100),
                            doc_metadata JSONB,
                            user_id INTEGER NOT NULL REFERENCES users(id),
                            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                            updated_at TIMESTAMP WITH TIME ZONE NOT NULL
                        )
                    """
                    )

                    # Create document_chunks table with pgvector support
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS document_chunks (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            document_id UUID NOT NULL REFERENCES documents(id),
                            chunk_index INTEGER NOT NULL,
                            content TEXT NOT NULL,
                            embedding VECTOR(768),
                            chunk_metadata JSONB,
                            token_count INTEGER,
                            created_at TIMESTAMP WITH TIME ZONE NOT NULL
                        )
                    """
                    )

                    # Create conversations table
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS conversations (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            user_id INTEGER NOT NULL REFERENCES users(id),
                            title VARCHAR(512) NOT NULL,
                            thread_id VARCHAR(255) UNIQUE NOT NULL,
                            document_ids JSONB,
                            status VARCHAR(50) DEFAULT 'active',
                            conv_metadata JSONB,
                            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                            updated_at TIMESTAMP WITH TIME ZONE NOT NULL
                        )
                    """
                    )

                    # Create messages table
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS messages (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            conversation_id UUID NOT NULL REFERENCES conversations(id),
                            role VARCHAR(20) NOT NULL,
                            content TEXT NOT NULL,
                            msg_metadata JSONB,
                            created_at TIMESTAMP WITH TIME ZONE NOT NULL
                        )
                    """
                    )

                    # Create vector search cache table
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS vector_search_cache (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            query_hash VARCHAR(64) UNIQUE NOT NULL,
                            query_text TEXT NOT NULL,
                            query_embedding VECTOR(768) NOT NULL,
                            results JSONB NOT NULL,
                            user_id INTEGER,
                            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                            expires_at TIMESTAMP
                        )
                    """
                    )

                    # Create indexes for performance
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_documents_user_status ON documents(user_id, status)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_conversations_user_status ON conversations(user_id, status)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at)"
                    )

                    # Create vector indexes for similarity search
                    try:
                        cur.execute(
                            """
                            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
                            ON document_chunks 
                            USING hnsw (embedding vector_cosine_ops)
                            WITH (m = 16, ef_construction = 64)
                            """
                        )
                        cur.execute(
                            """
                            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat 
                            ON document_chunks 
                            USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100)
                            """
                        )
                        logger.info("Vector indexes created successfully")
                    except Exception as e:
                        logger.warning(f"Could not create vector indexes: {e}")

                    conn.commit()
                    logger.info("Database tables and indexes created successfully")
                    return True

        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            return False

    def check_ai_parsing_completion(
        self, file_path: str, user_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Check if AI parsing has completed by looking for processed results in the database.

        Args:
            file_path: Path to the file that was processed
            user_id: User ID who owns the document

        Returns:
            Processed results if found, None otherwise
        """

        try:
            with self.get_connection() as conn:
                if not conn:
                    return None

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    # Check if we have processed results for this file and user
                    cur.execute(
                        """
                        SELECT * FROM parsed_pages 
                        WHERE document_id IN (
                            SELECT id FROM documents 
                            WHERE original_path = %s AND user_id = %s
                        )
                        ORDER BY chunk_index
                    """,
                        (file_path, user_id),
                    )

                    chunks = cur.fetchall()

                    if chunks:
                        # Get document info
                        cur.execute(
                            """
                            SELECT * FROM documents 
                            WHERE original_path = %s AND user_id = %s
                        """,
                            (file_path, user_id),
                        )

                        document = cur.fetchone()

                        if document:
                            return {
                                "document": dict(document),
                                "chunks": [dict(chunk) for chunk in chunks],
                                "total_chunks": len(chunks),
                                "completed": True,
                            }

                    return None

        except Exception as e:
            logger.error(f"Failed to check AI parsing completion: {str(e)}")
            return None

    def get_document_by_file_path(
        self, file_path: str, user_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get document by file path and user ID."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return None

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM documents WHERE original_path = %s AND user_id = %s",
                        (file_path, user_id),
                    )
                    document = cur.fetchone()
                    return dict(document) if document else None
        except Exception as e:
            logger.error(f"Failed to get document by file path: {str(e)}")
            return None

    # User operations
    def create_user(
        self,
        username: str,
        databricks_user_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Create or get existing user by Databricks user ID and username.

        Args:
            username (str): The user's username (email).
            databricks_user_id (int): Databricks user ID.

        Returns:
            Optional[Dict[str, Any]]: The user record as a dict, or None if creation failed.
        """

        try:
            with self.get_connection() as conn:
                if not conn:
                    return None

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    # Check if user exists by Databricks user ID
                    cur.execute(
                        "SELECT * FROM users WHERE id = %s",
                        (databricks_user_id,),
                    )
                    user = cur.fetchone()
                    if user:
                        return dict(user)

                    # Check if user exists by username
                    cur.execute(
                        "SELECT * FROM users WHERE username = %s",
                        (username,),
                    )
                    user = cur.fetchone()
                    if user:
                        return dict(user)

                    # If not found, create new user with Databricks user ID
                    now = datetime.now(timezone.utc)

                    cur.execute(
                        """INSERT INTO users (id, username, created_at, updated_at) 
                           VALUES (%s, %s, %s, %s) RETURNING *""",
                        (databricks_user_id, username, now, now),
                    )
                    conn.commit()
                    new_user = cur.fetchone()
                    logger.info(
                        f"Created new user: {username} with ID: {databricks_user_id}"
                    )
                    return dict(new_user)
        except Exception as e:
            logger.error(
                f"Failed to create user {username} (databricks_id={databricks_user_id}): {str(e)}"
            )
            return None

    def user_exists(self, username: str) -> bool:
        """Check if a user exists in the database."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return False

                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM users WHERE username = %s", (username,)
                    )
                    count = cur.fetchone()[0]
                    return count > 0

        except Exception as e:
            logger.error(f"Failed to check if user {username} exists: {str(e)}")
            return False

    def verify_user_authentication(
        self, username: str, databricks_user_id: int
    ) -> bool:
        """Verify that a user is authenticated and exists in the database."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return False

                with conn.cursor() as cur:
                    # Check if user exists with matching Databricks ID
                    cur.execute(
                        "SELECT COUNT(*) FROM users WHERE username = %s AND id = %s",
                        (username, databricks_user_id),
                    )
                    count = cur.fetchone()[0]
                    return count > 0

        except Exception as e:
            logger.error(f"Failed to verify user authentication: {str(e)}")
            return False

    # Conversation operations
    def create_conversation(
        self,
        user_id: int,  # User ID is now Databricks user ID (integer)
        title: str,
        thread_id: str,
        document_ids: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a new conversation."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return None

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    conversation_id = str(uuid.uuid4())
                    now = datetime.now(timezone.utc)
                    cur.execute(
                        """INSERT INTO conversations 
                           (id, user_id, title, thread_id, document_ids, status, created_at, updated_at) 
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *""",
                        (
                            conversation_id,
                            user_id,
                            title,
                            thread_id,
                            psycopg2.extras.Json(document_ids or []),
                            "active",
                            now,
                            now,
                        ),
                    )
                    conn.commit()
                    conversation = cur.fetchone()
                    logger.info(f"Created conversation: {conversation_id}")
                    return dict(conversation)
        except Exception as e:
            logger.error(f"Failed to create conversation: {str(e)}")
            return None

    def get_user_conversations(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all conversations for a user."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return []

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(
                        """SELECT * FROM conversations 
                           WHERE user_id = %s AND status = %s 
                           ORDER BY updated_at DESC""",
                        (user_id, "active"),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get conversations for user {user_id}: {str(e)}")
            return []

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return False

                with conn.cursor() as cur:
                    cur.execute(
                        """UPDATE conversations SET title = %s, updated_at = %s 
                           WHERE id = %s""",
                        (title, datetime.now(timezone.utc), conversation_id),
                    )
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update conversation title: {str(e)}")
            return False

    def update_conversation_title_by_id(self, conversation_id: str, title: str) -> bool:
        """Update conversation title by conversation ID."""
        return self.update_conversation_title(conversation_id, title)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return False

                with conn.cursor() as cur:
                    cur.execute(
                        """UPDATE conversations SET status = %s, updated_at = %s 
                           WHERE id = %s""",
                        (
                            "deleted",
                            datetime.now(timezone.utc),
                            conversation_id,
                        ),
                    )
                    conn.commit()
                    return cur.rowcount > 0
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
    ) -> Optional[Dict[str, Any]]:
        """Add a message to a conversation."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return None

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    message_id = str(uuid.uuid4())
                    now = datetime.now(timezone.utc)
                    cur.execute(
                        """INSERT INTO messages 
                           (id, conversation_id, role, content, msg_metadata, created_at) 
                           VALUES (%s, %s, %s, %s, %s, %s) RETURNING *""",
                        (
                            message_id,
                            conversation_id,
                            role,
                            content,
                            psycopg2.extras.Json(metadata or {}),
                            now,
                        ),
                    )
                    conn.commit()
                    message = cur.fetchone()
                    return dict(message)
        except Exception as e:
            logger.error(f"Failed to add message: {str(e)}")
            return None

    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return []

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(
                        """SELECT * FROM messages 
                           WHERE conversation_id = %s 
                           ORDER BY created_at""",
                        (conversation_id,),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(
                f"Failed to get messages for conversation {conversation_id}: {str(e)}"
            )
            return []

    # Document operations
    def create_document(
        self,
        user_id: int,  # User ID is now Databricks user ID (integer)
        doc_hash: str,
        filename: str,
        status: str = "uploaded",
    ) -> Optional[Dict[str, Any]]:
        """Create a new document record."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return None

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    document_id = str(uuid.uuid4())
                    now = datetime.now(timezone.utc)
                    cur.execute(
                        """INSERT INTO documents 
                           (id, user_id, doc_hash, filename, status, created_at, updated_at) 
                           VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *""",
                        (document_id, user_id, doc_hash, filename, status, now, now),
                    )
                    conn.commit()
                    document = cur.fetchone()
                    logger.info(f"Created document: {document_id}")
                    return dict(document)
        except Exception as e:
            logger.error(f"Failed to create document: {str(e)}")
            return None

    def update_document_status(
        self, document_id: str, status: str, error: Optional[str] = None
    ) -> bool:
        """Update document status."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return False

                with conn.cursor() as cur:
                    if error:
                        # Update with error in metadata
                        cur.execute(
                            """UPDATE documents 
                               SET status = %s, updated_at = %s, 
                                   doc_metadata = COALESCE(doc_metadata, '{}') || %s 
                               WHERE id = %s""",
                            (
                                status,
                                datetime.now(timezone.utc),
                                psycopg2.extras.Json({"error": error}),
                                document_id,
                            ),
                        )
                    else:
                        # Update status only
                        cur.execute(
                            """UPDATE documents SET status = %s, updated_at = %s 
                               WHERE id = %s""",
                            (status, datetime.now(timezone.utc), document_id),
                        )
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update document status: {str(e)}")
            return False

    def get_document_by_hash(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        """Get document by hash."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return None

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM documents WHERE doc_hash = %s", (doc_hash,)
                    )
                    document = cur.fetchone()
                    return dict(document) if document else None
        except Exception as e:
            logger.error(f"Failed to get document by hash {doc_hash}: {str(e)}")
            return None

    def get_user_documents(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all documents for a user."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return []

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(
                        """SELECT * FROM documents 
                           WHERE user_id = %s 
                           ORDER BY created_at DESC""",
                        (user_id,),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get documents for user {user_id}: {str(e)}")
            return []

    def get_user_documents_by_username(self, username: str) -> List[Dict[str, Any]]:
        """Get all documents for a user by username."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return []

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    # First get the user ID
                    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
                    user_result = cur.fetchone()

                    if not user_result:
                        logger.warning(f"User {username} not found")
                        return []

                    user_id = user_result[0]

                    # Then get the documents
                    cur.execute(
                        """SELECT * FROM documents 
                           WHERE user_id = %s 
                           ORDER BY created_at DESC""",
                        (user_id,),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get documents for user {username}: {str(e)}")
            return []

    # Document chunk operations
    def store_document_chunks(
        self, document_id: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Store document chunks with embeddings."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return []

                chunk_objects = []
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    for i, chunk_data in enumerate(chunks):
                        chunk_id = str(uuid.uuid4())
                        now = datetime.now(timezone.utc)
                        cur.execute(
                            """INSERT INTO document_chunks 
                               (id, document_id, chunk_index, content, embedding, 
                                chunk_metadata, token_count, created_at) 
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *""",
                            (
                                chunk_id,
                                document_id,
                                i,
                                chunk_data["content"],
                                chunk_data.get("embedding"),
                                psycopg2.extras.Json(chunk_data.get("metadata", {})),
                                chunk_data.get("token_count"),
                                now,
                            ),
                        )
                        chunk = cur.fetchone()
                        chunk_objects.append(dict(chunk))

                    conn.commit()
                    logger.info(
                        f"Stored {len(chunk_objects)} chunks for document {document_id}"
                    )
                    return chunk_objects
        except Exception as e:
            logger.error(f"Failed to store document chunks: {str(e)}")
            return []

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return []

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(
                        """SELECT * FROM document_chunks 
                           WHERE document_id = %s 
                           ORDER BY chunk_index""",
                        (document_id,),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {str(e)}")
            return []

    def vector_search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return []

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    # Build the query
                    query = """
                        SELECT dc.id, dc.content, dc.chunk_metadata, dc.token_count,
                               d.filename, d.doc_metadata,
                               dc.embedding <-> %s as distance
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                    """
                    params = [query_embedding]

                    if document_ids:
                        query += " WHERE d.id = ANY(%s)"
                        params.append(document_ids)

                    query += " ORDER BY distance LIMIT %s"
                    params.append(limit)

                    cur.execute(query, params)
                    results = cur.fetchall()

                    return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    def add_documents_to_conversation(
        self, conversation_id: str, document_hashes: List[str]
    ) -> bool:
        """Add documents to an existing conversation."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return False

                with conn.cursor() as cur:
                    # Get current document_ids for the conversation
                    cur.execute(
                        "SELECT document_ids FROM conversations WHERE id = %s",
                        (conversation_id,),
                    )
                    result = cur.fetchone()

                    if not result:
                        logger.error(f"Conversation {conversation_id} not found")
                        return False

                    current_doc_ids = result[0] or []

                    # Add new document hashes (avoid duplicates)
                    for doc_hash in document_hashes:
                        if doc_hash not in current_doc_ids:
                            current_doc_ids.append(doc_hash)

                    # Update conversation with new document_ids
                    cur.execute(
                        """UPDATE conversations 
                           SET document_ids = %s, updated_at = %s 
                           WHERE id = %s""",
                        (
                            psycopg2.extras.Json(current_doc_ids),
                            datetime.now(timezone.utc),
                            conversation_id,
                        ),
                    )

                    conn.commit()
                    logger.info(
                        f"Added {len(document_hashes)} documents to conversation {conversation_id}"
                    )
                    return True

        except Exception as e:
            logger.error(f"Failed to add documents to conversation: {str(e)}")
            return False

    def get_conversation_by_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by ID."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return None

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM conversations WHERE id = %s", (conversation_id,)
                    )
                    conversation = cur.fetchone()
                    return dict(conversation) if conversation else None

        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {str(e)}")
            return None

    def get_conversation_by_id_with_documents(
        self, conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get conversation by ID with associated document information."""

        try:
            with self.get_connection() as conn:
                if not conn:
                    return None

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    # Get conversation with document details
                    cur.execute(
                        """SELECT c.*, 
                                  array_agg(d.filename) as document_filenames,
                                  array_agg(d.status) as document_statuses
                           FROM conversations c
                           LEFT JOIN documents d ON d.doc_hash = ANY(c.document_ids)
                           WHERE c.id = %s
                           GROUP BY c.id""",
                        (conversation_id,),
                    )
                    conversation = cur.fetchone()
                    return dict(conversation) if conversation else None

        except Exception as e:
            logger.error(
                f"Failed to get conversation {conversation_id} with documents: {str(e)}"
            )
            return None


# ===== SQLAlchemy-based Database Utilities =====


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
    """Enhanced database manager with vector search capabilities using SQLAlchemy."""

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
        self, user_id: str, doc_hash: str, filename: str, **kwargs
    ) -> Document:
        """Create a new document record."""
        with self.get_session() as session:
            document = Document(
                user_id=user_id, doc_hash=doc_hash, filename=filename, **kwargs
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
        document_ids: Optional[List[str]] = None,
    ) -> Conversation:
        """Create a new conversation."""
        with self.get_session() as session:
            conversation = Conversation(
                user_id=user_id,
                title=title,
                thread_id=thread_id,
                document_ids=document_ids or [],
            )
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            return conversation

    def store_document_chunks(
        self, document_id: str, chunks: List[Dict[str, Any]]
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
                    token_count=chunk_data.get("token_count"),
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
        document_ids: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        """Perform vector similarity search."""
        with self.get_session() as session:
            query = session.query(DocumentChunk)

            if document_ids:
                query = query.filter(DocumentChunk.document_id.in_(document_ids))

            results = (
                query.order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
                .limit(limit)
                .all()
            )

            return results

    def get_conversation_messages(self, conversation_id: str) -> List[Message]:
        """Get all messages for a conversation."""
        with self.get_session() as session:
            messages = (
                session.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
                .all()
            )
            return messages

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> Message:
        """Add a message to a conversation."""
        with self.get_session() as session:
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
