"""
Pydantic models for database operations with automated SQL translation.
"""

from typing import Optional, List, Dict, Any, Type, get_origin, get_args, Union
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class KeyType(Enum):
    """Types of database keys."""

    PRIMARY = "PRIMARY KEY"
    FOREIGN = "FOREIGN KEY"
    UNIQUE = "UNIQUE"
    INDEX = "INDEX"


def PrimaryKey(description: str = "Primary key field"):
    """Create a primary key field annotation."""
    return Field(
        description=description, json_schema_extra={"key_type": KeyType.PRIMARY}
    )


def ForeignKey(references: str, description: str = "Foreign key field"):
    """Create a foreign key field annotation.

    Args:
        references: The table and column being referenced (e.g., "users(id)")
        description: Field description
    """
    return Field(
        description=description,
        json_schema_extra={"key_type": KeyType.FOREIGN, "references": references},
    )


def UniqueKey(description: str = "Unique key field"):
    """Create a unique key field annotation."""
    return Field(
        description=description, json_schema_extra={"key_type": KeyType.UNIQUE}
    )


def IndexKey(index_type: str = "btree", description: str = "Indexed field"):
    """Create an indexed field annotation.

    Args:
        index_type: Type of index (btree, hash, gin, gist, etc.)
        description: Field description
    """
    return Field(
        description=description,
        json_schema_extra={"key_type": KeyType.INDEX, "index_type": index_type},
    )


class BaseDatabaseModel(BaseModel):
    """Base model for all database entities with common configuration."""

    model_config = ConfigDict(from_attributes=True)


class User(BaseDatabaseModel):
    """User model for database operations."""

    id: str = PrimaryKey("Databricks user ID")
    username: str = UniqueKey("Unique username")
    created_at: datetime


class Document(BaseDatabaseModel):
    """Document model for database operations."""

    id: str = PrimaryKey("Document UUID")
    user_id: str = ForeignKey("users(id)", "Reference to user who owns this document")
    raw_path: str = UniqueKey("There can only be one raw path for a document")
    processed_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = IndexKey("btree", "Index for querying by creation time")


class DocumentChunk(BaseDatabaseModel):
    """Document chunk model for database operations."""

    id: str = PrimaryKey("Document chunk UUID")
    doc_id: str = ForeignKey("documents(id)", "Reference to parent document")
    content: str
    page_ids: Optional[List[str]] = None
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime


class Conversation(BaseDatabaseModel):
    """Conversation model for database operations."""

    id: str = PrimaryKey("Conversation ID (can be any string)")
    user_id: str = ForeignKey(
        "users(id)", "Reference to user who owns this conversation"
    )
    doc_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime = IndexKey("btree", "Index for querying by update time")


class Message(BaseDatabaseModel):
    """Message model for database operations."""

    id: str = PrimaryKey("Message UUID")
    conv_id: str = ForeignKey("conversations(id)", "Reference to parent conversation")
    role: str = Field(max_length=20)
    content: Dict[str, Any] = Field(
        description="Message content supporting multimodal formats"
    )
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = IndexKey("btree", "Index for querying by creation time")


class SQLTranslator:
    """Automated SQL generation from Pydantic models."""

    TYPE_MAPPINGS = {
        int: "INTEGER",
        str: "TEXT",
        datetime: "TIMESTAMP WITH TIME ZONE",
        bool: "BOOLEAN",
        List[float]: "VECTOR(768)",
        Dict[str, Any]: "JSONB",
        Dict: "JSONB",
        List[str]: "JSONB",
        Optional[int]: "INTEGER",
        Optional[str]: "TEXT",
        Optional[datetime]: "TIMESTAMP WITH TIME ZONE",
        Optional[Dict[str, Any]]: "JSONB",
        Optional[Dict]: "JSONB",
        List[str]: "JSONB",
    }

    @classmethod
    def get_postgres_type(
        cls, field_type: Type, field_name: str, table_name: str
    ) -> str:
        """Get PostgreSQL type for a Python type."""
        # Special case for user ID - now uses TEXT for large Databricks IDs
        if field_name == "id" and "user" in table_name.lower():
            return "TEXT"

        # Special case for user_id foreign keys - use TEXT for large Databricks IDs
        if field_name == "user_id":
            return "TEXT"

        # Handle Optional types
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is Union and type(None) in args:
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                field_type = non_none_args[0]

        return cls.TYPE_MAPPINGS.get(field_type, "TEXT")

    @classmethod
    def get_constraints(cls, field_name: str, field_info, table_name: str) -> str:
        """Get SQL constraints from Pydantic field annotations."""
        constraints = []

        # Check for key type annotations
        if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
            key_type = field_info.json_schema_extra.get("key_type")

            if key_type == KeyType.PRIMARY:
                if field_name == "id" and "user" not in table_name.lower():
                    constraints.append("PRIMARY KEY DEFAULT gen_random_uuid()")
                else:
                    constraints.append("PRIMARY KEY")

            elif key_type == KeyType.FOREIGN:
                references = field_info.json_schema_extra.get("references", "")
                if references:
                    constraints.append(f"REFERENCES {references}")

            elif key_type == KeyType.UNIQUE:
                constraints.append("UNIQUE")

        # Not null for required fields
        if field_info.is_required():
            constraints.append("NOT NULL")

        return " ".join(constraints)

    @classmethod
    def get_indexes(
        cls, model_class: Type[BaseDatabaseModel], table_name: str
    ) -> List[str]:
        """Generate indexes from model field annotations."""
        indexes = []

        for field_name, field_info in model_class.model_fields.items():
            if (
                hasattr(field_info, "json_schema_extra")
                and field_info.json_schema_extra
            ):
                key_type = field_info.json_schema_extra.get("key_type")

                if key_type == KeyType.INDEX:
                    indexes.append(
                        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{field_name} ON {table_name}({field_name});"
                    )

        return indexes

    @classmethod
    def model_to_create_table_sql(
        cls, model_class: Type[BaseDatabaseModel], table_name: str
    ) -> str:
        """Generate CREATE TABLE SQL from Pydantic model."""
        columns = []

        for field_name, field_info in model_class.model_fields.items():
            field_type = field_info.annotation
            postgres_type = cls.get_postgres_type(field_type, field_name, table_name)
            constraints = cls.get_constraints(field_name, field_info, table_name)

            column_def = f"    {field_name} {postgres_type}"
            if constraints:
                column_def += f" {constraints}"

            columns.append(column_def)

        table_sql = (
            f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
            + ",\n".join(columns)
            + "\n);"
        )

        # Add indexes
        indexes = cls.get_indexes(model_class, table_name)
        if indexes:
            table_sql += "\n\n" + "\n".join(indexes)

        return table_sql

    @classmethod
    def model_to_insert_sql(
        cls, model_class: Type[BaseDatabaseModel], table_name: str
    ) -> tuple[str, List[str]]:
        """Generate INSERT SQL from model."""
        fields = list(model_class.model_fields.keys())
        placeholders = ", ".join(["%s"] * len(fields))
        field_names = ", ".join(fields)

        sql = f"INSERT INTO {table_name} ({field_names}) VALUES ({placeholders}) RETURNING *"
        return sql, fields

    @classmethod
    def model_to_values(cls, model: BaseDatabaseModel, fields: List[str]) -> tuple:
        """Extract values from model for SQL parameters."""
        import json

        values = []
        for field in fields:
            value = getattr(model, field)
            # Convert dict/list to JSON string for JSONB columns
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            values.append(value)
        return tuple(values)

    @classmethod
    def dict_to_model(
        cls, model_class: Type[BaseDatabaseModel], data: Dict[str, Any]
    ) -> BaseDatabaseModel:
        """Convert dictionary to Pydantic model."""
        return model_class(**data)

    @classmethod
    def model_to_dict(cls, model: BaseDatabaseModel) -> Dict[str, Any]:
        """Convert Pydantic model to dictionary."""
        return model.model_dump()


def create_all_tables_sql() -> str:
    """Generate CREATE TABLE SQL for all models."""
    tables = [
        SQLTranslator.model_to_create_table_sql(User, "users"),
        SQLTranslator.model_to_create_table_sql(Document, "documents"),
        SQLTranslator.model_to_create_table_sql(DocumentChunk, "chunks"),
        SQLTranslator.model_to_create_table_sql(Conversation, "conversations"),
        SQLTranslator.model_to_create_table_sql(Message, "messages"),
    ]

    extension_sql = "CREATE EXTENSION IF NOT EXISTS vector;"
    return extension_sql + "\n\n" + "\n\n".join(tables)


def create_user_sql() -> tuple[str, List[str]]:
    """Generate INSERT SQL for users."""
    return SQLTranslator.model_to_insert_sql(User, "users")


def create_document_sql() -> tuple[str, List[str]]:
    """Generate INSERT SQL for documents."""
    return SQLTranslator.model_to_insert_sql(Document, "documents")


def create_document_chunk_sql() -> tuple[str, List[str]]:
    """Generate INSERT SQL for chunks."""
    return SQLTranslator.model_to_insert_sql(DocumentChunk, "chunks")


def create_conversation_sql() -> tuple[str, List[str]]:
    """Generate INSERT SQL for conversations."""
    return SQLTranslator.model_to_insert_sql(Conversation, "conversations")


def create_message_sql() -> tuple[str, List[str]]:
    """Generate INSERT SQL for messages."""
    return SQLTranslator.model_to_insert_sql(Message, "messages")
