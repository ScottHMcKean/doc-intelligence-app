"""Document chunk storage and in-memory database management."""

import json
from typing import List, Dict, Any, Optional
import pandas as pd

import streamlit as st

from .postgres_client import get_postgres_client
from ..config import MOCK_MODE, MOCK_DOCUMENTS, MOCK_DOCUMENT_CHUNKS


class DocumentStore:
    """Manages document chunks and in-memory database."""

    def __init__(self):
        self.client = get_postgres_client()
        self._in_memory_chunks: Dict[str, pd.DataFrame] = {}
        self._ensure_tables_exist()

    def _ensure_tables_exist(self) -> None:
        """Create necessary tables for document storage."""
        if MOCK_MODE:
            # In mock mode, populate in-memory chunks
            for doc_hash, chunks in MOCK_DOCUMENT_CHUNKS.items():
                self._in_memory_chunks[doc_hash] = pd.DataFrame(chunks)
            return

        create_documents_table = """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            doc_hash VARCHAR(255) UNIQUE NOT NULL,
            filename VARCHAR(500) NOT NULL,
            username VARCHAR(255) NOT NULL,
            upload_path VARCHAR(1000),
            processed_path VARCHAR(1000),
            status VARCHAR(50) DEFAULT 'uploaded',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP,
            metadata JSONB DEFAULT '{}'
        )
        """

        create_chunks_table = """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            doc_hash VARCHAR(255) NOT NULL,
            chunk_id VARCHAR(255) NOT NULL,
            content TEXT NOT NULL,
            chunk_type VARCHAR(100),
            page_number INTEGER,
            position_info JSONB,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_hash) REFERENCES documents(doc_hash)
        )
        """

        create_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(doc_hash)",
            "CREATE INDEX IF NOT EXISTS idx_documents_username ON documents(username)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_hash ON document_chunks(doc_hash)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_type ON document_chunks(chunk_type)",
        ]

        try:
            self.client.execute_update(create_documents_table)
            self.client.execute_update(create_chunks_table)

            for index_query in create_indexes:
                self.client.execute_update(index_query)

        except Exception as e:
            st.error(f"Failed to create document tables: {str(e)}")
            raise

    def store_document_metadata(
        self,
        doc_hash: str,
        filename: str,
        username: str,
        upload_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store document metadata."""
        if MOCK_MODE:
            # In mock mode, just return success
            return True

        query = """
        INSERT INTO documents (doc_hash, filename, username, upload_path, metadata)
        VALUES (:doc_hash, :filename, :username, :upload_path, :metadata)
        ON CONFLICT (doc_hash) DO UPDATE SET
            filename = EXCLUDED.filename,
            upload_path = EXCLUDED.upload_path,
            metadata = EXCLUDED.metadata
        """

        try:
            self.client.execute_update(
                query,
                {
                    "doc_hash": doc_hash,
                    "filename": filename,
                    "username": username,
                    "upload_path": upload_path,
                    "metadata": json.dumps(metadata or {}),
                },
            )
            return True
        except Exception as e:
            st.error(f"Failed to store document metadata: {str(e)}")
            return False

    def update_document_status(
        self, doc_hash: str, status: str, processed_path: Optional[str] = None
    ) -> bool:
        """Update document processing status."""
        if MOCK_MODE:
            # In mock mode, just return success
            return True

        if processed_path:
            query = """
            UPDATE documents 
            SET status = :status, processed_path = :processed_path, processed_at = CURRENT_TIMESTAMP
            WHERE doc_hash = :doc_hash
            """
            params = {
                "doc_hash": doc_hash,
                "status": status,
                "processed_path": processed_path,
            }
        else:
            query = """
            UPDATE documents 
            SET status = :status
            WHERE doc_hash = :doc_hash
            """
            params = {"doc_hash": doc_hash, "status": status}

        try:
            self.client.execute_update(query, params)
            return True
        except Exception as e:
            st.error(f"Failed to update document status: {str(e)}")
            return False

    def store_document_chunks(
        self, doc_hash: str, parsed_content: Dict[str, Any]
    ) -> bool:
        """Store document chunks from parsed content."""
        if MOCK_MODE:
            # In mock mode, just load chunks into memory
            chunks = self._extract_chunks_from_parsed_content(parsed_content)
            self._in_memory_chunks[doc_hash] = pd.DataFrame(chunks)
            return True

        try:
            # Clear existing chunks for this document
            self.client.execute_update(
                "DELETE FROM document_chunks WHERE doc_hash = :doc_hash",
                {"doc_hash": doc_hash},
            )

            # Extract chunks from parsed content
            chunks = self._extract_chunks_from_parsed_content(parsed_content)

            # Store chunks in database
            insert_query = """
            INSERT INTO document_chunks 
            (doc_hash, chunk_id, content, chunk_type, page_number, position_info, metadata)
            VALUES (:doc_hash, :chunk_id, :content, :chunk_type, :page_number, :position_info, :metadata)
            """

            for chunk in chunks:
                self.client.execute_update(
                    insert_query,
                    {
                        "doc_hash": doc_hash,
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        "chunk_type": chunk.get("chunk_type"),
                        "page_number": chunk.get("page_number"),
                        "position_info": json.dumps(chunk.get("position_info", {})),
                        "metadata": json.dumps(chunk.get("metadata", {})),
                    },
                )

            # Load chunks into memory for faster access
            self._load_chunks_to_memory(doc_hash)

            return True
        except Exception as e:
            st.error(f"Failed to store document chunks: {str(e)}")
            return False

    def _extract_chunks_from_parsed_content(
        self, parsed_content: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract chunks from ai_parse_document output."""
        chunks = []

        # This is a simplified extraction - adjust based on actual ai_parse_document output format
        if "pages" in parsed_content:
            for page_num, page_data in enumerate(parsed_content["pages"]):
                if "text_blocks" in page_data:
                    for block_idx, block in enumerate(page_data["text_blocks"]):
                        chunks.append(
                            {
                                "chunk_id": f"page_{page_num}_block_{block_idx}",
                                "content": block.get("text", ""),
                                "chunk_type": "text_block",
                                "page_number": page_num + 1,
                                "position_info": block.get("position", {}),
                                "metadata": {
                                    "confidence": block.get("confidence"),
                                    "block_type": block.get("type"),
                                },
                            }
                        )

                if "tables" in page_data:
                    for table_idx, table in enumerate(page_data["tables"]):
                        chunks.append(
                            {
                                "chunk_id": f"page_{page_num}_table_{table_idx}",
                                "content": json.dumps(table.get("data", [])),
                                "chunk_type": "table",
                                "page_number": page_num + 1,
                                "position_info": table.get("position", {}),
                                "metadata": {
                                    "rows": table.get("rows"),
                                    "columns": table.get("columns"),
                                },
                            }
                        )

        return chunks

    def _load_chunks_to_memory(self, doc_hash: str) -> None:
        """Load document chunks into memory for faster access."""
        query = """
        SELECT chunk_id, content, chunk_type, page_number, position_info, metadata
        FROM document_chunks
        WHERE doc_hash = :doc_hash
        ORDER BY page_number, chunk_id
        """

        try:
            results = self.client.execute_query(query, {"doc_hash": doc_hash})

            chunks_data = []
            for row in results:
                chunks_data.append(
                    {
                        "chunk_id": row[0],
                        "content": row[1],
                        "chunk_type": row[2],
                        "page_number": row[3],
                        "position_info": json.loads(row[4]) if row[4] else {},
                        "metadata": json.loads(row[5]) if row[5] else {},
                    }
                )

            self._in_memory_chunks[doc_hash] = pd.DataFrame(chunks_data)

        except Exception as e:
            st.error(f"Failed to load chunks to memory: {str(e)}")

    def get_document_chunks(self, doc_hash: str) -> pd.DataFrame:
        """Get document chunks as a DataFrame."""
        if doc_hash not in self._in_memory_chunks:
            self._load_chunks_to_memory(doc_hash)

        return self._in_memory_chunks.get(doc_hash, pd.DataFrame())

    def search_chunks(
        self, doc_hash: str, query: str, chunk_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Search chunks using simple text matching."""
        chunks_df = self.get_document_chunks(doc_hash)

        if chunks_df.empty:
            return chunks_df

        # Filter by chunk type if specified
        if chunk_type:
            chunks_df = chunks_df[chunks_df["chunk_type"] == chunk_type]

        # Simple text search - can be enhanced with vector search
        mask = chunks_df["content"].str.contains(query, case=False, na=False)
        return chunks_df[mask]

    def get_user_documents(self, username: str) -> List[Dict[str, Any]]:
        """Get all documents for a user."""
        if MOCK_MODE:
            return MOCK_DOCUMENTS.copy()

        query = """
        SELECT doc_hash, filename, status, created_at, processed_at, metadata
        FROM documents
        WHERE username = :username
        ORDER BY created_at DESC
        """

        try:
            results = self.client.execute_query(query, {"username": username})

            return [
                {
                    "doc_hash": row[0],
                    "filename": row[1],
                    "status": row[2],
                    "created_at": row[3],
                    "processed_at": row[4],
                    "metadata": json.loads(row[5]) if row[5] else {},
                }
                for row in results
            ]
        except Exception as e:
            st.error(f"Failed to get user documents: {str(e)}")
            return []
