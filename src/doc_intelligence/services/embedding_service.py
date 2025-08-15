"""Embedding service for vector operations."""

import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from ..config import config

logger = logging.getLogger(__name__)

try:
    from databricks_langchain import DatabricksEmbeddings
    from langchain_community.vectorstores.pgvector import PGVector
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as e:
    logger.error(f"Failed to import required dependencies: {e}")
    raise ImportError(
        f"Missing required dependencies for embedding service: {e}. "
        "Please ensure all dependencies are installed."
    )


class EmbeddingService:
    """Service for embeddings and vector operations."""
    
    def __init__(self, databricks_host: Optional[str] = None, databricks_token: Optional[str] = None):
        self.databricks_host = databricks_host or config.databricks_host
        self.databricks_token = databricks_token or config.databricks_token
        self.embedding_endpoint = config.databricks_embedding_endpoint
        
        self.embeddings = None
        self.text_splitter = None
        self.vectorstore = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize embedding components."""
        self._init_text_splitter()
        self._init_embeddings()
    
    def _init_text_splitter(self):
        """Initialize text splitter for chunking."""
        try:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            logger.info("Successfully initialized text splitter")
        except Exception as e:
            logger.error(f"Failed to initialize text splitter: {e}")
            self.text_splitter = None
    
    def _init_embeddings(self):
        """Initialize Databricks embeddings."""
        if not config.databricks_available or not self.embedding_endpoint:
            logger.warning("Databricks embeddings not available")
            self.embeddings = None
            return
            
        try:
            self.embeddings = DatabricksEmbeddings(
                endpoint=self.embedding_endpoint,
                databricks_host=self.databricks_host,
                databricks_token=self.databricks_token,
            )
            logger.info("Successfully initialized Databricks embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize Databricks embeddings: {e}")
            self.embeddings = None
    
    def init_vectorstore(self, connection_string: str) -> bool:
        """Initialize PGVector store for document chunks."""
        if not config.postgres_available or not self.embeddings:
            logger.warning("Vector store not available")
            return False
            
        try:
            self.vectorstore = PGVector(
                connection_string=connection_string,
                embedding_function=self.embeddings,
                collection_name="document_chunks",
                pre_delete_collection=False,
            )
            logger.info("Successfully initialized PGVector store")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.vectorstore = None
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if embedding service is available."""
        return self.embeddings is not None
    
    @property
    def vectorstore_available(self) -> bool:
        """Check if vector store is available."""
        return self.vectorstore is not None
    
    def chunk_text(self, text: str) -> Tuple[bool, List[str], str]:
        """
        Chunk text into smaller pieces.
        
        Returns:
            Tuple of (success, chunks, message)
        """
        if not self.text_splitter:
            # Fallback chunking
            chunks = []
            chunk_size = 512
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return True, chunks, f"Text chunked into {len(chunks)} pieces (fallback method)"
        
        try:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Successfully chunked text into {len(chunks)} pieces")
            return True, chunks, f"Text chunked into {len(chunks)} pieces"
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            return False, [], f"Failed to chunk text: {str(e)}"
    
    def generate_embeddings(self, texts: List[str]) -> Tuple[bool, Optional[List[List[float]]], str]:
        """
        Generate embeddings for a list of texts.
        
        Returns:
            Tuple of (success, embeddings, message)
        """
        if not self.embeddings:
            logger.warning("Embeddings not available, generating mock embeddings")
            # Generate mock embeddings for testing
            mock_embeddings = []
            for _ in texts:
                # Generate random 768-dimensional vector
                embedding = np.random.normal(0, 1, 768).tolist()
                mock_embeddings.append(embedding)
            return True, mock_embeddings, f"Generated {len(mock_embeddings)} mock embeddings"
        
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return True, embeddings, f"Generated {len(embeddings)} embeddings"
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return False, None, f"Failed to generate embeddings: {str(e)}"
    
    def generate_single_embedding(self, text: str) -> Tuple[bool, Optional[List[float]], str]:
        """
        Generate embedding for a single text.
        
        Returns:
            Tuple of (success, embedding, message)
        """
        if not self.embeddings:
            # Generate mock embedding
            embedding = np.random.normal(0, 1, 768).tolist()
            return True, embedding, "Generated mock embedding"
        
        try:
            embedding = self.embeddings.embed_query(text)
            logger.info("Successfully generated single embedding")
            return True, embedding, "Generated embedding"
        except Exception as e:
            logger.error(f"Failed to generate single embedding: {e}")
            return False, None, f"Failed to generate embedding: {str(e)}"
    
    def similarity_search(
        self, 
        query: str, 
        limit: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Perform similarity search in vector store.
        
        Returns:
            Tuple of (success, results, message)
        """
        if not self.vectorstore:
            logger.warning("Vector store not available")
            # Return mock results
            mock_results = [
                {
                    "content": f"Mock search result {i+1} for query: {query}",
                    "metadata": {"score": 0.9 - (i * 0.1), "document_id": f"mock_doc_{i+1}"},
                    "score": 0.9 - (i * 0.1)
                }
                for i in range(min(limit, 3))
            ]
            return True, mock_results, f"Generated {len(mock_results)} mock search results"
        
        try:
            # Perform similarity search
            docs = self.vectorstore.similarity_search_with_score(
                query=query,
                k=limit,
                filter=filter_dict
            )
            
            results = []
            for doc, score in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
            
            logger.info(f"Found {len(results)} similar documents")
            return True, results, f"Found {len(results)} similar documents"
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return False, [], f"Search failed: {str(e)}"
    
    def add_documents_to_vectorstore(
        self, 
        texts: List[str], 
        metadatas: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Add documents to vector store.
        
        Returns:
            Tuple of (success, message)
        """
        if not self.vectorstore:
            logger.warning("Vector store not available for document addition")
            return False, "Vector store not available"
        
        try:
            self.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            logger.info(f"Successfully added {len(texts)} documents to vector store")
            return True, f"Added {len(texts)} documents to vector store"
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            return False, f"Failed to add documents: {str(e)}"
    
    def delete_documents_from_vectorstore(self, filter_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Delete documents from vector store.
        
        Returns:
            Tuple of (success, message)
        """
        if not self.vectorstore:
            return False, "Vector store not available"
        
        try:
            self.vectorstore.delete(filter=filter_dict)
            logger.info(f"Successfully deleted documents matching filter: {filter_dict}")
            return True, f"Deleted documents matching filter"
        except Exception as e:
            logger.error(f"Failed to delete documents from vector store: {e}")
            return False, f"Failed to delete documents: {str(e)}"
