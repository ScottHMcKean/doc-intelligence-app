"""
RAG (Retrieval Augmented Generation) workflow for document processing.

This module handles:
- Document chunking and embedding
- Vector storage with pgvector
- Similarity search and retrieval
- Integration with Databricks embeddings
"""

import logging
from typing import Dict, List, Any, Optional, TypedDict
import hashlib
from pathlib import Path

from ..config import config
from ..database.schema import DatabaseManager

logger = logging.getLogger(__name__)

# Required imports - now always available
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores.pgvector import PGVector
    from databricks_langchain import DatabricksEmbeddings
except ImportError as e:
    logger.error(f"Failed to import required dependencies: {e}")
    raise ImportError(
        f"Missing required dependencies for RAG workflow: {e}. "
        "Please ensure all dependencies are installed."
    )


class DocumentRAGState(TypedDict):
    """State for document RAG processing."""
    document_id: str
    filename: str
    content: str
    chunks: List[Dict[str, Any]]
    embeddings: List[List[float]]
    status: str
    error: Optional[str]


class RAGWorkflow:
    """RAG workflow manager for document processing and retrieval."""
    
    def __init__(
        self,
        postgres_connection_string: str,
        databricks_host: Optional[str] = None,
        databricks_token: Optional[str] = None,
        embedding_endpoint: str = "databricks-bge-large-en",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.postgres_connection = postgres_connection_string
        self.databricks_host = databricks_host
        self.databricks_token = databricks_token
        self.embedding_endpoint = embedding_endpoint
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.db_manager = DatabaseManager(postgres_connection_string)
        self._init_embeddings()
        self._init_vectorstore()
        self._init_text_splitter()
    
    def _init_embeddings(self):
        """Initialize Databricks embeddings."""
        if MOCK_MODE:
            logger.info("Using mock embeddings in development mode")
            self.embeddings = None
        else:
            self.embeddings = DatabricksEmbeddings(
                endpoint=self.embedding_endpoint,
                databricks_host=self.databricks_host,
                databricks_token=self.databricks_token,
            )
    
    def _init_vectorstore(self):
        """Initialize PGVector store."""
        if MOCK_MODE:
            logger.info("Using mock vector store in development mode")
            self.vectorstore = None
        else:
            self.vectorstore = PGVector(
                connection_string=self.postgres_connection,
                embedding_function=self.embeddings,
                collection_name="document_chunks",
                pre_delete_collection=False,
            )
    
    def _init_text_splitter(self):
        """Initialize text splitter for chunking."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def process_document(
        self,
        document_content: str,
        document_id: str,
        filename: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Process a document through the RAG pipeline."""
        
        logger.info(f"Processing document {filename} for user {user_id}")
        
        try:
            # Step 1: Chunk the document
            chunks = self._chunk_document(document_content)
            logger.info(f"Created {len(chunks)} chunks from document")
            
            # Step 2: Generate embeddings
            if MOCK_MODE:
                embeddings = self._generate_mock_embeddings(chunks)
            else:
                embeddings = self._generate_embeddings(chunks)
            
            logger.info(f"Generated embeddings for {len(embeddings)} chunks")
            
            # Step 3: Store in vector database
            self._store_chunks_with_embeddings(
                document_id, chunks, embeddings
            )
            
            # Step 4: Update document status
            self._update_document_status(document_id, "processed")
            
            return {
                "success": True,
                "document_id": document_id,
                "chunks_created": len(chunks),
                "status": "processed"
            }
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            self._update_document_status(document_id, "failed", str(e))
            
            return {
                "success": False,
                "document_id": document_id,
                "error": str(e),
                "status": "failed"
            }
    
    def _chunk_document(self, content: str) -> List[str]:
        """Split document into chunks."""
        try:
            chunks = self.text_splitter.split_text(content)
            return chunks
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise
    
    def _generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for document chunks."""
        if not self.embeddings:
            raise ValueError("Embeddings not initialized")
        
        try:
            # Batch embedding generation for efficiency
            embeddings = []
            batch_size = 10  # Process in batches to avoid timeout
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch)
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _generate_mock_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate mock embeddings for development."""
        import random
        
        embeddings = []
        for chunk in chunks:
            # Generate deterministic but varied embeddings based on content
            seed = hash(chunk) % 2**31
            random.seed(seed)
            embedding = [random.uniform(-1, 1) for _ in range(768)]
            embeddings.append(embedding)
        
        return embeddings
    
    def _store_chunks_with_embeddings(
        self,
        document_id: str,
        chunks: List[str],
        embeddings: List[List[float]]
    ):
        """Store chunks with embeddings in the database."""
        
        try:
            # Prepare chunk data
            chunk_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_data.append({
                    "content": chunk,
                    "embedding": embedding,
                    "metadata": {
                        "chunk_index": i,
                        "document_id": document_id,
                        "chunk_length": len(chunk)
                    },
                    "token_count": len(chunk.split())  # Rough token estimate
                })
            
            # Store in database
            self.db_manager.store_document_chunks(document_id, chunk_data)
            
            # Also store in vector store if not in mock mode
            if not MOCK_MODE and self.vectorstore:
                texts = [chunk for chunk in chunks]
                metadatas = [{"document_id": document_id, "chunk_index": i} 
                           for i in range(len(chunks))]
                
                self.vectorstore.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            
            logger.info(f"Stored {len(chunk_data)} chunks for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            raise
    
    def _update_document_status(
        self, 
        document_id: str, 
        status: str, 
        error: Optional[str] = None
    ):
        """Update document processing status."""
        try:
            with self.db_manager.get_session() as session:
                from ..database.schema import Document
                
                document = session.query(Document).filter(
                    Document.id == document_id
                ).first()
                
                if document:
                    document.status = status
                    if error:
                        document.doc_metadata = document.doc_metadata or {}
                        document.doc_metadata["error"] = error
                    
                    session.commit()
                    logger.info(f"Updated document {document_id} status to {status}")
                else:
                    logger.warning(f"Document {document_id} not found for status update")
                    
        except Exception as e:
            logger.error(f"Error updating document status: {e}")
    
    def search_documents(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        
        logger.info(f"Searching documents for query: {query}")
        
        try:
            if MOCK_MODE:
                return self._mock_document_search(query, document_ids, limit)
            
            if not self.vectorstore:
                raise ValueError("Vector store not initialized")
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Perform vector search
            if document_ids:
                # Filter by document IDs
                filter_dict = {"document_id": {"$in": document_ids}}
                results = self.vectorstore.similarity_search_with_score(
                    query, k=limit, filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search_with_score(
                    query, k=limit
                )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                if score >= similarity_threshold:
                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": score,
                        "document_id": doc.metadata.get("document_id"),
                        "chunk_index": doc.metadata.get("chunk_index")
                    })
            
            logger.info(f"Found {len(formatted_results)} relevant chunks")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def _mock_document_search(
        self, 
        query: str, 
        document_ids: Optional[List[str]], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Mock document search for development."""
        
        # Return mock search results
        mock_results = [
            {
                "content": f"This is a relevant chunk about '{query}' from the uploaded document. It contains information that matches your query and provides context for answering questions.",
                "metadata": {
                    "document_id": document_ids[0] if document_ids else "mock_doc_1",
                    "chunk_index": 0,
                    "page": 1
                },
                "similarity_score": 0.9,
                "document_id": document_ids[0] if document_ids else "mock_doc_1",
                "chunk_index": 0
            },
            {
                "content": f"Additional context related to '{query}' can be found in this section. This chunk provides supplementary information that helps provide comprehensive answers.",
                "metadata": {
                    "document_id": document_ids[0] if document_ids else "mock_doc_1", 
                    "chunk_index": 1,
                    "page": 2
                },
                "similarity_score": 0.85,
                "document_id": document_ids[0] if document_ids else "mock_doc_1",
                "chunk_index": 1
            }
        ]
        
        return mock_results[:limit]
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        try:
            chunks = self.db_manager.vector_search(
                query_embedding=[0] * 768,  # Dummy embedding
                document_ids=[document_id],
                limit=1000  # Get all chunks
            )
            
            return [
                {
                    "content": chunk.content,
                    "metadata": chunk.chunk_metadata,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count
                }
                for chunk in chunks
            ]
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            return []
    
    def delete_document_chunks(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            with self.db_manager.get_session() as session:
                from ..database.schema import DocumentChunk
                
                # Delete from database
                deleted_count = session.query(DocumentChunk).filter(
                    DocumentChunk.document_id == document_id
                ).delete()
                
                session.commit()
                
                # Also delete from vector store if not in mock mode
                if not MOCK_MODE and self.vectorstore:
                    self.vectorstore.delete(filter={"document_id": document_id})
                
                logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting document chunks: {e}")
            return False
