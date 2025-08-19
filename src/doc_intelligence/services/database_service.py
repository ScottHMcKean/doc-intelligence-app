"""Database service for PostgreSQL operations."""

import logging
import uuid
import psycopg2
import psycopg2.extras
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
from databricks.sdk import WorkspaceClient
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DatabaseService:
    """Simplified database service using direct psycopg2 connections."""

    def __init__(self, client: Optional[WorkspaceClient], config: dict):
        self.client = client
        self.config = config

    @contextmanager
    def get_connection(self):
        """Get a database connection using the simple psycopg2 pattern."""
        if not self.client:
            logger.warning("No Databricks client available for database connection")
            yield None
            return

        try:
            # Get database instance name from config
            if hasattr(self.config, "database"):
                instance_name = self.config.database.instance_name
                user = self.config.database.user
                database = self.config.database.database
            else:
                instance_name = self.config.get("instance_name")
                user = self.config.get("user", "databricks")
                database = self.config.get("database", "databricks_postgres")

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
        if not self.client:
            return False, "Database not configured"

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
        if not self.client:
            return False

        try:
            with self.get_connection() as conn:
                if not conn:
                    return False

                with conn.cursor() as cur:
                    # Create users table - use UUID for flexibility
                    cur.execute(
                        """
                        CREATE OR REPLACE TABLE IF NOT EXISTS users (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            username VARCHAR(255) UNIQUE NOT NULL,
                            email VARCHAR(255),
                            databricks_user_id BIGINT, -- Store Databricks user ID separately if available
                            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                            updated_at TIMESTAMP WITH TIME ZONE NOT NULL
                        )
                    """
                    )

                    # Create conversations table
                    cur.execute(
                        """
                        CREATE OR REPLACE TABLE conversations (
                            id UUID PRIMARY KEY,
                            user_id UUID NOT NULL REFERENCES users(id),
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
                        CREATE OR REPLACE TABLE messages (
                            id UUID PRIMARY KEY,
                            conversation_id UUID NOT NULL REFERENCES conversations(id),
                            role VARCHAR(20) NOT NULL,
                            content TEXT NOT NULL,
                            msg_metadata JSONB,
                            created_at TIMESTAMP WITH TIME ZONE NOT NULL
                        )
                    """
                    )

                    # Create documents table
                    cur.execute(
                        """
                        CREATE OR REPLACE TABLE documents (
                            id UUID PRIMARY KEY,
                            doc_hash VARCHAR(64) UNIQUE NOT NULL,
                            filename VARCHAR(512) NOT NULL,
                            original_path VARCHAR(1024),
                            processed_path VARCHAR(1024),
                            status VARCHAR(50) DEFAULT 'uploaded',
                            file_size INTEGER,
                            content_type VARCHAR(100),
                            doc_metadata JSONB,
                            user_id UUID NOT NULL REFERENCES users(id),
                            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                            updated_at TIMESTAMP WITH TIME ZONE NOT NULL
                        )
                    """
                    )

                    # Create document_chunks table (try pgvector first, fallback to JSONB)
                    try:
                        # Try to create with pgvector extension
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        cur.execute(
                            """
                            CREATE OR REPLACE TABLE document_chunks (
                                id UUID PRIMARY KEY,
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
                        logger.info(
                            "Created document_chunks table with pgvector support"
                        )
                    except Exception as e:
                        logger.warning(
                            f"pgvector not available, using JSONB fallback: {e}"
                        )
                        # Fallback without VECTOR type
                        cur.execute(
                            """
                            CREATE OR REPLACE TABLE document_chunks (
                                id UUID PRIMARY KEY,
                                document_id UUID NOT NULL REFERENCES documents(id),
                                chunk_index INTEGER NOT NULL,
                                content TEXT NOT NULL,
                                embedding JSONB, -- Store as JSON array
                                chunk_metadata JSONB,
                                token_count INTEGER,
                                created_at TIMESTAMP WITH TIME ZONE NOT NULL
                            )
                        """
                        )
                        logger.info("Created document_chunks table with JSONB fallback")

                    conn.commit()
                    logger.info("Database tables created successfully")
                    return True

        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            return False

    def check_ai_parsing_completion(
        self, file_path: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if AI parsing has completed by looking for processed results in the database.

        Args:
            file_path: Path to the file that was processed
            user_id: User ID who owns the document

        Returns:
            Processed results if found, None otherwise
        """
        if not self.client:
            return None

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
        self, file_path: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get document by file path and user ID."""
        if not self.client:
            return None

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
    def create_user(self, username_or_id: str | int) -> Optional[Dict[str, Any]]:
        """Create or get existing user by username or ID."""
        if not self.client:
            return None

        try:
            with self.get_connection() as conn:
                if not conn:
                    return None

                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    # Determine if input is username or Databricks user ID
                    if isinstance(username_or_id, int):
                        # This is a Databricks user ID
                        databricks_user_id = username_or_id
                        username = str(databricks_user_id)

                        # Check if user exists by Databricks user ID
                        cur.execute(
                            "SELECT * FROM users WHERE databricks_user_id = %s",
                            (databricks_user_id,),
                        )
                        user = cur.fetchone()
                        if user:
                            return dict(user)
                    else:
                        # This is a username string
                        username = username_or_id
                        databricks_user_id = None

                        # Check if user exists by username
                        cur.execute(
                            "SELECT * FROM users WHERE username = %s", (username,)
                        )
                        user = cur.fetchone()
                        if user:
                            return dict(user)

                    # Create new user with UUID
                    user_id = str(uuid.uuid4())
                    now = datetime.now(timezone.utc)

                    cur.execute(
                        """INSERT INTO users (id, username, databricks_user_id, created_at, updated_at) 
                           VALUES (%s, %s, %s, %s, %s) RETURNING *""",
                        (user_id, username, databricks_user_id, now, now),
                    )
                    conn.commit()
                    new_user = cur.fetchone()
                    logger.info(f"Created new user: {username} with ID: {user_id}")
                    return dict(new_user)
        except Exception as e:
            logger.error(f"Failed to create user {username_or_id}: {str(e)}")
            return None

    # Conversation operations
    def create_conversation(
        self,
        user_id: str,  # User ID is now a UUID string
        title: str,
        thread_id: str,
        document_ids: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a new conversation."""
        if not self.client:
            return None

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

    def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all conversations for a user."""
        if not self.client:
            return []

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
        if not self.client:
            return False

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

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if not self.client:
            return False

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
        if not self.client:
            return None

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
        if not self.client:
            return []

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
        user_id: str,  # User ID is now a UUID string
        doc_hash: str,
        filename: str,
        status: str = "uploaded",
    ) -> Optional[Dict[str, Any]]:
        """Create a new document record."""
        if not self.client:
            return None

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
        if not self.client:
            return False

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
        if not self.client:
            return None

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

    def get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a user."""
        if not self.client:
            return []

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

    # Document chunk operations
    def store_document_chunks(
        self, document_id: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Store document chunks with embeddings."""
        if not self.client:
            return []

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
        if not self.client:
            return []

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
        if not self.client:
            return []

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
