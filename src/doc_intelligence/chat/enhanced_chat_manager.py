"""Enhanced chat manager with LangGraph and vector search integration."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import uuid
import os

import streamlit as st

from ..database.schema import DatabaseManager
from ..config import MOCK_MODE, get_mock_config

logger = logging.getLogger(__name__)

# Conditional imports for LangGraph components
if not MOCK_MODE:
    try:
        from ..langgraph import ConversationManager, RAGWorkflow
    except ImportError as e:
        logger.warning(f"LangGraph dependencies not available: {e}")
        ConversationManager = None
        RAGWorkflow = None
else:
    ConversationManager = None
    RAGWorkflow = None


class EnhancedChatManager:
    """Enhanced chat manager with LangGraph, Postgres, and Databricks integration."""
    
    def __init__(self):
        # Get configuration
        self.postgres_connection = self._get_postgres_connection()
        self.databricks_host = os.getenv("DATABRICKS_HOST")
        self.databricks_token = os.getenv("DATABRICKS_TOKEN")
        
        # Initialize components
        if MOCK_MODE:
            self.db_manager = None  # Use mock methods instead
        else:
            self.db_manager = DatabaseManager(self.postgres_connection)
        
        # Initialize LangGraph components if available
        if ConversationManager and RAGWorkflow:
            self.conversation_manager = ConversationManager(
                postgres_connection_string=self.postgres_connection,
                databricks_host=self.databricks_host,
                databricks_token=self.databricks_token,
            )
            self.rag_workflow = RAGWorkflow(
                postgres_connection_string=self.postgres_connection,
                databricks_host=self.databricks_host,
                databricks_token=self.databricks_token,
            )
        else:
            self.conversation_manager = None
            self.rag_workflow = None
            logger.info("Running in mock mode - LangGraph components not initialized")
        
        # Create event loop for async operations
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
    
    def _get_postgres_connection(self) -> str:
        """Get PostgreSQL connection string."""
        if MOCK_MODE:
            # Return a placeholder connection string for mock mode
            return "postgresql://mock:mock@localhost:5432/mock"
        
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "doc_intelligence")
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        
        if not user or not password:
            logger.warning("PostgreSQL credentials not found, using mock mode")
            return "postgresql://mock:mock@localhost:5432/mock"
        
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    
    def start_new_conversation(
        self, 
        username: str, 
        document_hash: Optional[str] = None
    ) -> str:
        """Start a new conversation."""
        try:
            if MOCK_MODE:
                # Return mock conversation ID
                return f"mock_conversation_{len(MOCK_CONVERSATIONS) + 1}"
            
            # Ensure user exists
            user = self.db_manager.create_user(username)
            
            # Determine title and document IDs
            if document_hash:
                doc_info = self.get_document_info(document_hash)
                title = f"Chat with {doc_info.get('filename', 'Document')}" if doc_info else "Document Chat"
                document_ids = [document_hash]
            else:
                title = "New Conversation"
                document_ids = []
            
            if self.conversation_manager:
                # Create conversation using LangGraph manager
                conversation_data = self.conversation_manager.create_conversation(
                    user_id=str(user.id),
                    title=title,
                    document_ids=document_ids
                )
                return conversation_data["conversation_id"]
            else:
                # Fallback to direct database creation in mock mode
                conversation = self.db_manager.create_conversation(
                    user_id=str(user.id),
                    title=title,
                    thread_id=f"thread_{uuid.uuid4()}",
                    document_ids=document_ids
                )
                return str(conversation.id)
            
        except Exception as e:
            logger.error(f"Error starting new conversation: {e}")
            if MOCK_MODE:
                return str(uuid.uuid4())
            raise
    
    def send_message(
        self,
        conversation_id: str,
        user_message: str,
        username: str,
        document_hash: Optional[str] = None,
    ) -> str:
        """Send a message and get a response using LangGraph."""
        try:
            if MOCK_MODE:
                return self._generate_mock_response(user_message, document_hash)
            
            # Ensure user exists
            user = self.db_manager.create_user(username)
            
            # Determine document context
            document_ids = [document_hash] if document_hash else []
            
            if self.conversation_manager:
                # Send message through LangGraph conversation manager
                if self.loop.is_running():
                    # If loop is already running, create a new thread
                    import threading
                    import concurrent.futures
                    
                    def run_async():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                self.conversation_manager.send_message(
                                    user_message=user_message,
                                    user_id=str(user.id),
                                    conversation_id=conversation_id,
                                    document_ids=document_ids
                                )
                            )
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_async)
                        result = future.result(timeout=30)  # 30 second timeout
                else:
                    # Run directly in the current loop
                    result = self.loop.run_until_complete(
                        self.conversation_manager.send_message(
                            user_message=user_message,
                            user_id=str(user.id),
                            conversation_id=conversation_id,
                            document_ids=document_ids
                        )
                    )
                
                if result.get("success"):
                    return result["response"]
                else:
                    logger.error(f"Error in LangGraph response: {result.get('error')}")
                    return "I apologize, but I encountered an error processing your message. Please try again."
            else:
                # Fallback to mock response in mock mode
                return self._generate_mock_response(user_message, document_hash)
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            if MOCK_MODE:
                return self._generate_mock_response(user_message, document_hash)
            return "I apologize, but I encountered an error. Please try again."
    
    def _generate_mock_response(self, user_message: str, document_hash: Optional[str]) -> str:
        """Generate mock response for development mode."""
        user_lower = user_message.lower()
        
        if document_hash:
            return f"Based on the uploaded document, I can help you with '{user_message}'. The document contains relevant information about this topic and I can provide detailed insights."
        elif "hello" in user_lower or "hi" in user_lower:
            return "Hello! I'm your enhanced document intelligence assistant powered by LangGraph and Databricks. You can upload documents and ask questions about them, or have a general conversation with me."
        elif "document" in user_lower:
            return "I can help you analyze and understand documents using advanced RAG (Retrieval Augmented Generation) technology. Please upload a document to get started with document-specific questions."
        elif "help" in user_lower:
            return "I can assist you with:\n• Advanced document analysis using vector search\n• Intelligent question answering about document content\n• Persistent conversation history\n• Multi-document conversations\n• General AI assistance"
        else:
            return f"I understand you're asking about '{user_message}'. This enhanced version uses LangGraph for conversation management and Postgres for persistent state. In production, this would be powered by Databricks LLM services with full vector search capabilities."
    
    def process_document(
        self,
        document_content: str,
        document_id: str,
        filename: str,
        username: str,
    ) -> Dict[str, Any]:
        """Process a document through the RAG pipeline."""
        try:
            if MOCK_MODE:
                # Mock processing in development mode
                logger.info(f"Mock processing document {filename} in development mode")
                return {
                    "success": True,
                    "document_id": document_id,
                    "filename": filename,
                    "chunks_created": 5,
                    "status": "processed"
                }
            
            # Ensure user exists
            user = self.db_manager.create_user(username)
            
            # Create document record
            doc_hash = document_id  # Assuming document_id is the hash
            document = self.db_manager.create_document(
                user_id=str(user.id),
                doc_hash=doc_hash,
                filename=filename,
                status="processing"
            )
            
            if self.rag_workflow:
                # Process through RAG workflow
                result = self.rag_workflow.process_document(
                    document_content=document_content,
                    document_id=str(document.id),
                    filename=filename,
                    user_id=str(user.id)
                )
                return result
            else:
                # Mock processing in development mode
                logger.info(f"Mock processing document {filename} in development mode")
                return {
                    "success": True,
                    "document_id": str(document.id),
                    "chunks_created": 5,  # Mock chunk count
                    "status": "processed"
                }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                "success": False,
                "error": str(e),
                "status": "failed"
            }
    
    def search_documents(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        try:
            if self.rag_workflow:
                return self.rag_workflow.search_documents(
                    query=query,
                    document_ids=document_ids,
                    limit=limit
                )
            else:
                # Mock search results
                return [
                    {
                        "content": f"Mock search result for '{query}' - relevant document content would appear here",
                        "metadata": {"document_id": document_ids[0] if document_ids else "mock_doc"},
                        "similarity_score": 0.9
                    }
                ]
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history."""
        try:
            if MOCK_MODE:
                from ..config import MOCK_MESSAGES
                return [msg for msg in MOCK_MESSAGES if msg.get("conversation_id") == conversation_id]
            
            if self.conversation_manager:
                return self.conversation_manager.get_conversation_history(conversation_id)
            else:
                # Fallback to direct database query
                return self.db_manager.get_conversation_messages(conversation_id)
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            if MOCK_MODE:
                from ..config import MOCK_MESSAGES
                return [msg for msg in MOCK_MESSAGES if msg.get("conversation_id") == conversation_id]
            return []
    
    def get_user_conversations(self, username: str) -> List[Dict[str, Any]]:
        """Get all conversations for a user."""
        try:
            if MOCK_MODE:
                from ..config import MOCK_CONVERSATIONS
                return [conv for conv in MOCK_CONVERSATIONS if conv.get("username") == username]
            
            # Get user
            user = self.db_manager.create_user(username)
            
            # Get conversations from database
            with self.db_manager.get_session() as session:
                from ..database.schema import Conversation
                
                conversations = session.query(Conversation).filter(
                    Conversation.user_id == user.id,
                    Conversation.status == "active"
                ).order_by(Conversation.updated_at.desc()).all()
                
                return [
                    {
                        "conversation_id": str(conv.id),
                        "title": conv.title,
                        "created_at": conv.created_at.isoformat(),
                        "updated_at": conv.updated_at.isoformat(),
                        "document_hash": conv.document_ids[0] if conv.document_ids else None,
                        "thread_id": conv.thread_id
                    }
                    for conv in conversations
                ]
                
        except Exception as e:
            logger.error(f"Error getting user conversations: {e}")
            if MOCK_MODE:
                from ..config import MOCK_CONVERSATIONS
                return MOCK_CONVERSATIONS
            return []
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        try:
            if MOCK_MODE:
                logger.info(f"Mock deleting conversation {conversation_id}")
                return True
            
            with self.db_manager.get_session() as session:
                from ..database.schema import Conversation
                
                conversation = session.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()
                
                if conversation:
                    conversation.status = "deleted"
                    session.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            return MOCK_MODE  # Return True in mock mode
    
    def get_user_documents(self, username: str) -> List[Dict[str, Any]]:
        """Get all documents for a user."""
        try:
            if MOCK_MODE:
                from ..config import MOCK_DOCUMENTS
                return [doc for doc in MOCK_DOCUMENTS if doc.get("username") == username]
            
            # Get user
            user = self.db_manager.create_user(username)
            
            # Get documents from database
            with self.db_manager.get_session() as session:
                from ..database.schema import Document
                
                documents = session.query(Document).filter(
                    Document.user_id == user.id
                ).order_by(Document.created_at.desc()).all()
                
                return [
                    {
                        "doc_hash": doc.doc_hash,
                        "filename": doc.filename,
                        "status": doc.status,
                        "created_at": doc.created_at.isoformat(),
                        "file_size": doc.file_size,
                        "content_type": doc.content_type
                    }
                    for doc in documents
                ]
                
        except Exception as e:
            logger.error(f"Error getting user documents: {e}")
            if MOCK_MODE:
                from ..config import MOCK_DOCUMENTS
                return MOCK_DOCUMENTS
            return []
    
    def get_document_info(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document."""
        try:
            if MOCK_MODE:
                return {
                    "filename": "sample_document.pdf",
                    "status": "processed",
                    "created_at": "2025-01-15T12:00:00",
                    "file_size": 1024000,
                    "content_type": "application/pdf",
                    "metadata": {}
                }
            
            with self.db_manager.get_session() as session:
                from ..database.schema import Document
                
                document = session.query(Document).filter(
                    Document.doc_hash == doc_hash
                ).first()
                
                if document:
                    return {
                        "filename": document.filename,
                        "status": document.status,
                        "created_at": document.created_at.isoformat(),
                        "file_size": document.file_size,
                        "content_type": document.content_type,
                        "metadata": document.doc_metadata
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            if MOCK_MODE:
                return {
                    "filename": "sample_document.pdf",
                    "status": "processed",
                    "created_at": "2024-01-01T00:00:00",
                    "file_size": 1024000,
                    "content_type": "application/pdf",
                    "metadata": {}
                }
            return None
    
    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title."""
        try:
            if MOCK_MODE:
                logger.info(f"Mock updating conversation {conversation_id} title to {title}")
                return True
            
            with self.db_manager.get_session() as session:
                from ..database.schema import Conversation
                
                conversation = session.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()
                
                if conversation:
                    conversation.title = title
                    session.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error updating conversation title: {e}")
            return MOCK_MODE  # Return True in mock mode


# Create a compatibility layer for existing code
class ChatManager(EnhancedChatManager):
    """Compatibility layer for existing ChatManager interface."""
    
    def send_message(
        self,
        conversation_id: str,
        user_message: str,
        username: str,
        chat_mode: str = "document",
        document_hash: Optional[str] = None,
        vector_search_config: Optional[Dict[str, str]] = None,
    ) -> str:
        """Send message with legacy interface compatibility."""
        # Map legacy chat_mode to new system
        if chat_mode == "document" and document_hash:
            return super().send_message(
                conversation_id=conversation_id,
                user_message=user_message,
                username=username,
                document_hash=document_hash
            )
        else:
            # General chat or vector search
            return super().send_message(
                conversation_id=conversation_id,
                user_message=user_message,
                username=username,
                document_hash=None
            )
