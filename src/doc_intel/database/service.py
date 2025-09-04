"""Simplified database service using only psycopg2 with Pydantic models."""

import logging
import uuid
import psycopg2
import psycopg2.extras
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import DatabaseInstance
from datetime import datetime, timezone

from .models import (
    User,
    Document,
    DocumentChunk,
    Conversation,
    Message,
    SQLTranslator,
    create_user_sql,
    create_document_sql,
    create_document_chunk_sql,
    create_conversation_sql,
    create_message_sql,
    create_all_tables_sql,
)

logger = logging.getLogger(__name__)


class DatabaseService:
    """Simplified database service using only psycopg2 with convenience methods."""

    def __init__(self, client: WorkspaceClient, config: dict):
        self.client = client
        self.config = config

        self._user_id = None
        self._user_email = None
        self._connection_params = None
        self._credential = None

        self._initialize_user_and_connection()

    def setup_database_instance(self) -> bool:
        """Create database instance if it doesn't exist and create PostgreSQL extensions."""
        try:
            instance_name = self.config.get("database", {}).get("instance_name")
            capacity = self.config.get("database", {}).get("capacity")

            if not instance_name:
                raise ValueError("No database instance_name configured")

            # Try to create the database instance
            try:
                self.client.database.create_database_instance(
                    DatabaseInstance(name=instance_name, capacity=capacity)
                )
                logger.info(f"Created database instance: {instance_name}")
            except Exception as e:
                logger.info(f"Database instance already exists: {e}")

            # Create PostgreSQL extensions
            self._create_postgres_extensions()

            return True

        except Exception as e:
            logger.error(f"Failed to setup database instance: {str(e)}")
            return False

    def _create_postgres_extensions(self):
        """Create required PostgreSQL extensions."""
        try:
            conn = self.connect_to_pg()
            with conn.cursor() as cur:
                # Create vector extension for embeddings
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                logger.info("Created PostgreSQL vector extension")
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL extensions: {str(e)}")
            raise
        finally:
            if "conn" in locals():
                conn.close()

    def _initialize_user_and_connection(self):
        """Initialize user info and connection parameters."""
        try:
            # Get current user info
            user = self.client.current_user.me()
            self._user_id = user.id
            self._user_email = user.emails[0].value

            # Get database instance and credentials
            instance_name = self.config.get("database", {}).get("instance_name")
            if not instance_name:
                raise ValueError("No database instance_name configured")

            instance = self.client.database.get_database_instance(name=instance_name)
            self._credential = self.client.database.generate_database_credential(
                request_id=str(uuid.uuid4()), instance_names=[instance_name]
            )

            # Cache connection parameters
            self._connection_params = {
                "host": instance.read_write_dns,
                "dbname": "databricks_postgres",
                "user": self._user_email,
                "password": self._credential.token,
                "sslmode": "require",
            }

            logger.info(f"Initialized connection for user: {self._user_email}")

        except Exception as e:
            logger.error(f"Failed to initialize user and connection: {str(e)}")
            raise

    @property
    def user(self) -> str:
        """Get the current user info."""
        return self._user_email

    def connect_to_pg(self):
        """Get a direct psycopg2 connection using cached parameters."""
        try:
            conn = psycopg2.connect(**self._connection_params)
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise

    def run_pg_query(
        self,
        query: str,
        data_tuples=None,
        return_dataframe=False,
        use_dict_cursor=False,
        fetch_one=False,
        params=None,
    ):
        """Execute a PostgreSQL query with optional data and return results.

        Args:
            query: SQL query to execute
            data_tuples: Optional data tuples for parameterized queries (for executemany)
            params: Optional single parameter tuple for parameterized queries
            return_dataframe: If True, return results as pandas DataFrame (default: False)
            use_dict_cursor: If True, use DictCursor for results (default: False)
            fetch_one: If True, return only first result (default: False)

        Returns:
            Query results as list of tuples/dicts, pandas DataFrame, single result, or True for DML operations
        """
        conn = self.connect_to_pg()
        try:
            cursor_factory = psycopg2.extras.DictCursor if use_dict_cursor else None
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                if data_tuples:
                    cur.executemany(query, data_tuples)
                elif params:
                    cur.execute(query, params)
                else:
                    cur.execute(query)

                # Check if this is a SELECT query that should return data
                query_upper = query.strip().upper()
                if query_upper.startswith(
                    ("SELECT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN")
                ):
                    # Return query results
                    if fetch_one:
                        results = cur.fetchone()
                    else:
                        results = cur.fetchall()

                    conn.commit()

                    # Convert to DataFrame if requested
                    if return_dataframe and not fetch_one:
                        import pandas as pd

                        # Get column names from cursor description
                        columns = (
                            [desc[0] for desc in cur.description]
                            if cur.description
                            else []
                        )
                        return pd.DataFrame(results, columns=columns)

                    # Convert DictCursor results to regular dicts for consistency
                    if use_dict_cursor and results:
                        if fetch_one:
                            return dict(results) if results else None
                        else:
                            return [dict(row) for row in results]

                    return results
                else:
                    # For INSERT, UPDATE, DELETE, etc., just commit and return True
                    conn.commit()
                    return True

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name;
        """
        results = self.run_pg_query(query)
        return [row[0] for row in results] if results else []

    @property
    def connection_live(self) -> bool:
        """Test database connection."""
        try:
            conn = self.connect_to_pg()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return False
        finally:
            if "conn" in locals():
                conn.close()

    def create_tables(self) -> bool:
        """Create all database tables using generated SQL."""
        try:
            sql = create_all_tables_sql()
            self.run_pg_query(sql)
            logger.info("Created all database tables")
            return True
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            return False

    def create_user(self) -> Optional[User]:
        """Create or get existing user using cached user info."""
        try:
            conn = self.connect_to_pg()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Check if user exists by Databricks user ID
                cur.execute("SELECT * FROM users WHERE id = %s", (str(self._user_id),))
                user_data = cur.fetchone()
                if user_data:
                    return SQLTranslator.dict_to_model(User, dict(user_data))

                # Check if user exists by username
                cur.execute(
                    "SELECT * FROM users WHERE username = %s", (self._user_email,)
                )
                user_data = cur.fetchone()
                if user_data:
                    return SQLTranslator.dict_to_model(User, dict(user_data))

                # If not found, create new user with Databricks user ID
                now = datetime.now(timezone.utc)
                user_model = User(
                    id=str(self._user_id),  # Convert to string for large Databricks IDs
                    username=self._user_email,
                    created_at=now,
                )

                # Generate SQL and values using translator
                sql, fields = create_user_sql()
                values = SQLTranslator.model_to_values(user_model, fields)

                cur.execute(sql, values)
                conn.commit()
                new_user_data = cur.fetchone()

                logger.info(
                    f"Created new user: {self._user_email} with ID: {self._user_id}"
                )
                return SQLTranslator.dict_to_model(User, dict(new_user_data))
        except Exception as e:
            logger.error(f"Failed to create user {self._user_email}: {str(e)}")
            return None
        finally:
            if "conn" in locals():
                conn.close()

    def user_exists(self) -> bool:
        """Check if the current user exists in the database."""
        try:
            result = self.run_pg_query(
                "SELECT COUNT(*) FROM users WHERE id = %s",
                params=(self._user_id,),
                fetch_one=True,
            )
            return result[0] > 0 if result else False
        except Exception as e:
            logger.error(f"Failed to check if user {self._user_email} exists: {str(e)}")
            return False

    def create_document(
        self,
        raw_path: Optional[str] = None,
        processed_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Document]:
        """Create a new document record."""
        try:
            conn = self.connect_to_pg()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                now = datetime.now(timezone.utc)
                user_id = self._user_id  # Use cached user ID

                # Create document model with required fields
                document_model = Document(
                    id=str(uuid.uuid4()),  # Generate UUID for document ID
                    user_id=str(user_id),  # Convert to string for large Databricks IDs
                    raw_path=raw_path,
                    processed_path=processed_path,
                    metadata=metadata,
                    created_at=now,
                )

                # Generate SQL and values using translator
                sql, fields = create_document_sql()
                values = SQLTranslator.model_to_values(document_model, fields)

                cur.execute(sql, values)
                conn.commit()
                document_data = cur.fetchone()

                logger.info(f"Created document: {document_model.id}")
                return SQLTranslator.dict_to_model(Document, dict(document_data))
        except Exception as e:
            logger.error(f"Failed to create document: {str(e)}")
            return None
        finally:
            if "conn" in locals():
                conn.close()

    def get_document_by_id(self, document_id: str, return_dataframe=False):
        """Get document by ID for current user."""
        try:
            if return_dataframe:
                # For dataframe, we need to return a list of one item, not fetch_one
                results = self.run_pg_query(
                    "SELECT * FROM documents WHERE id = %s AND user_id = %s",
                    params=(document_id, str(self._user_id)),
                    use_dict_cursor=True,
                    return_dataframe=True,
                )
                return results
            else:
                return self.run_pg_query(
                    "SELECT * FROM documents WHERE id = %s AND user_id = %s",
                    params=(document_id, str(self._user_id)),
                    use_dict_cursor=True,
                    fetch_one=True,
                )
        except Exception as e:
            logger.error(f"Failed to get document by ID {document_id}: {str(e)}")
            return None

    def get_user_documents(self, return_dataframe=False):
        """Get all documents for the current user."""
        try:
            results = self.run_pg_query(
                """SELECT * FROM documents 
                   WHERE user_id = %s 
                   ORDER BY created_at DESC""",
                params=(str(self._user_id),),
                use_dict_cursor=True,
                return_dataframe=return_dataframe,
            )

            if return_dataframe:
                return results  # run_pg_query already returns DataFrame when requested
            else:
                return [SQLTranslator.dict_to_model(Document, row) for row in results]
        except Exception as e:
            logger.error(
                f"Failed to get documents for user {self._user_email}: {str(e)}"
            )
            return [] if not return_dataframe else None

    # Conversation operations
    def create_conversation(
        self,
        conversation_id: str,
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Conversation]:
        """Create a new conversation."""
        try:
            conn = self.connect_to_pg()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                now = datetime.now(timezone.utc)
                user_id = self._user_id  # Use cached user ID

                # Create conversation model
                conversation_model = Conversation(
                    id=conversation_id,
                    user_id=str(user_id),  # Convert to string for large Databricks IDs
                    doc_ids=doc_ids,
                    metadata=metadata,
                    created_at=now,
                    updated_at=now,
                )

                # Generate SQL and values using translator
                sql, fields = create_conversation_sql()
                values = SQLTranslator.model_to_values(conversation_model, fields)

                cur.execute(sql, values)
                conn.commit()
                conversation_data = cur.fetchone()

                logger.info(f"Created conversation: {conversation_id}")
                return SQLTranslator.dict_to_model(
                    Conversation, dict(conversation_data)
                )
        except Exception as e:
            logger.error(f"Failed to create conversation: {str(e)}")
            return None
        finally:
            if "conn" in locals():
                conn.close()

    def get_user_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversations for the current user."""
        try:
            return self.run_pg_query(
                """SELECT * FROM conversations 
                   WHERE user_id = %s 
                   ORDER BY updated_at DESC""",
                params=(str(self._user_id),),
                use_dict_cursor=True,
            )
        except Exception as e:
            logger.error(
                f"Failed to get conversations for user {self._user_email}: {str(e)}"
            )
            return []

    def get_conversation_by_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific conversation by ID."""
        try:
            return self.run_pg_query(
                "SELECT * FROM conversations WHERE id = %s AND user_id = %s",
                params=(conversation_id, str(self._user_id)),
                use_dict_cursor=True,
                fetch_one=True,
            )
        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {str(e)}")
            return None

    # Message operations
    def add_message(
        self,
        conv_id: str,
        role: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Message]:
        """Add a message to a conversation."""
        try:
            conn = self.connect_to_pg()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                message_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc)

                # Create message model
                message_model = Message(
                    id=message_id,
                    conv_id=conv_id,
                    role=role,
                    content=content,
                    metadata=metadata,
                    created_at=now,
                )

                # Generate SQL and values using translator
                sql, fields = create_message_sql()
                values = SQLTranslator.model_to_values(message_model, fields)

                cur.execute(sql, values)
                conn.commit()
                message_data = cur.fetchone()

                logger.info(f"Added message: {message_id}")
                return SQLTranslator.dict_to_model(Message, dict(message_data))
        except Exception as e:
            logger.error(f"Failed to add message: {str(e)}")
            return None
        finally:
            if "conn" in locals():
                conn.close()

    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation."""
        try:
            return self.run_pg_query(
                """SELECT * FROM messages 
                   WHERE conv_id = %s 
                   ORDER BY created_at""",
                params=(conversation_id,),
                use_dict_cursor=True,
            )
        except Exception as e:
            logger.error(
                f"Failed to get messages for conversation {conversation_id}: {str(e)}"
            )
            return []

    def store_document_chunks(
        self, document_id: str, chunks: List[Dict[str, Any]]
    ) -> bool:
        """Store document chunks in the database."""
        try:
            conn = self.connect_to_pg()
            with conn.cursor() as cur:
                for i, chunk_data in enumerate(chunks):
                    chunk_id = str(uuid.uuid4())
                    now = datetime.now(timezone.utc)

                    # Insert chunk with embedding
                    import json

                    page_ids = chunk_data.get("page_ids")
                    if isinstance(page_ids, list):
                        page_ids = json.dumps(page_ids)

                    metadata = chunk_data.get("metadata", {})
                    if isinstance(metadata, dict):
                        metadata = json.dumps(metadata)

                    cur.execute(
                        """INSERT INTO chunks 
                           (id, doc_id, content, page_ids, embedding, 
                            metadata, created_at) 
                           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        (
                            chunk_id,
                            document_id,
                            chunk_data.get("content", ""),
                            page_ids,
                            chunk_data.get("embedding"),
                            metadata,
                            now,
                        ),
                    )

                conn.commit()
                logger.info(f"Stored {len(chunks)} chunks for document {document_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to store document chunks: {str(e)}")
            return False
        finally:
            if "conn" in locals():
                conn.close()

    def get_document_chunks(
        self, document_id: str, return_dataframe=False
    ) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        try:
            return self.run_pg_query(
                """SELECT * FROM chunks 
                   WHERE doc_id = %s 
                   ORDER BY created_at""",
                params=(document_id,),
                use_dict_cursor=True,
                return_dataframe=return_dataframe,
            )
        except Exception as e:
            logger.error(f"Failed to get document chunks: {str(e)}")
            return []
