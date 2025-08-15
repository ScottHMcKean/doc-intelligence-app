"""
LangGraph-based conversation manager for document intelligence.

This module implements a StateGraph for managing conversations with:
- Multi-turn conversation state
- Document context management
- Persistent history via Postgres
- Integration with Databricks LLM endpoints
"""

import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from ..config import config
from ..database.schema import DatabaseManager

logger = logging.getLogger(__name__)

# Required imports - now always available
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from databricks_langchain import ChatDatabricks, DatabricksEmbeddings
    from langchain_community.vectorstores.pgvector import PGVector
except ImportError as e:
    logger.error(f"Failed to import required dependencies: {e}")
    raise ImportError(
        f"Missing required dependencies for LangGraph conversation management: {e}. "
        "Please ensure all dependencies are installed."
    )


class ChatState(TypedDict):
    """State for conversation management."""
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    conversation_id: str
    thread_id: str
    document_ids: List[str]
    context_documents: List[Dict[str, Any]]
    last_retrieval: Optional[str]
    metadata: Dict[str, Any]


class ConversationManager:
    """LangGraph-based conversation manager with Postgres persistence."""
    
    def __init__(
        self,
        postgres_connection_string: str,
        databricks_host: Optional[str] = None,
        databricks_token: Optional[str] = None,
        embedding_endpoint: str = "databricks-bge-large-en",
        chat_endpoint: str = "databricks-dbrx-instruct",
    ):
        self.postgres_connection = postgres_connection_string
        self.databricks_host = databricks_host
        self.databricks_token = databricks_token
        self.embedding_endpoint = embedding_endpoint
        self.chat_endpoint = chat_endpoint
        
        # Initialize database manager
        self.db_manager = DatabaseManager(postgres_connection_string)
        
        # Initialize components
        self._init_embeddings()
        self._init_vectorstore()
        self._init_llm()
        self._init_graph()
    
    def _init_embeddings(self):
        """Initialize Databricks embeddings with graceful fallback."""
        if not config.databricks_available or not self.embedding_endpoint:
            logger.warning("Databricks embeddings not available - some features will be limited")
            self.embeddings = None
        else:
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
    
    def _init_vectorstore(self):
        """Initialize PGVector store for document chunks with graceful fallback."""
        if not config.postgres_available or not self.embeddings:
            logger.warning("Vector store not available - document search will be limited")
            self.vectorstore = None
        else:
            try:
                self.vectorstore = PGVector(
                    connection_string=self.postgres_connection,
                    embedding_function=self.embeddings,
                    collection_name="document_chunks",
                    pre_delete_collection=False,
                )
                logger.info("Successfully initialized PGVector store")
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
                self.vectorstore = None
    
    def _init_llm(self):
        """Initialize Databricks LLM with graceful fallback."""
        if not config.databricks_available or not self.chat_endpoint:
            logger.warning("Databricks LLM not available - chat responses will be limited")
            self.llm = None
        else:
            try:
                self.llm = ChatDatabricks(
                    endpoint=self.chat_endpoint,
                    databricks_host=self.databricks_host,
                    databricks_token=self.databricks_token,
                    max_tokens=512,
                    temperature=0.1,
                )
                logger.info("Successfully initialized Databricks LLM")
            except Exception as e:
                logger.error(f"Failed to initialize Databricks LLM: {e}")
                self.llm = None
    
    def _init_graph(self):
        """Initialize the LangGraph StateGraph."""
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("save_message", self._save_message_node)
        
        # Define the flow
        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "save_message")
        workflow.set_finish_point("save_message")
        
        # Compile with checkpointing
        checkpointer = self._create_checkpointer()
        self.graph = workflow.compile(checkpointer=checkpointer)
    
    def _create_checkpointer(self):
        """Create Postgres checkpointer for state persistence with graceful fallback."""
        if not config.postgres_available:
            logger.warning("Postgres not available - conversation state will not persist")
            return None
        
        try:
            return PostgresSaver.from_conn_string(self.postgres_connection)
        except Exception as e:
            logger.error(f"Failed to create Postgres checkpointer: {e}")
            return None
    
    async def _retrieve_context_node(self, state: ChatState) -> ChatState:
        """Retrieve relevant document context for the conversation."""
        logger.info(f"Retrieving context for conversation {state['conversation_id']}")
        
        if not state.get("document_ids") or not self.vectorstore:
            # No documents or vector store not available - return empty context
            state["context_documents"] = []
            state["last_retrieval"] = datetime.utcnow().isoformat()
            return state
        
        try:
            # Get the last user message
            last_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_message = msg.content
                    break
            
            if not last_message:
                state["context_documents"] = []
                return state
            
            # Perform vector search
            relevant_chunks = self.vectorstore.similarity_search(
                last_message,
                k=5,
                filter={"document_id": {"$in": state["document_ids"]}}
            )
            
            # Format context documents
            context_docs = []
            for chunk in relevant_chunks:
                context_docs.append({
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "relevance_score": getattr(chunk, "relevance_score", 1.0)
                })
            
            state["context_documents"] = context_docs
            state["last_retrieval"] = datetime.utcnow().isoformat()
            
            logger.info(f"Retrieved {len(context_docs)} relevant chunks")
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            state["context_documents"] = []
        
        return state
    
    async def _generate_response_node(self, state: ChatState) -> ChatState:
        """Generate AI response using retrieved context."""
        logger.info(f"Generating response for conversation {state['conversation_id']}")
        
        try:
            # Get the last user message
            last_user_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg.content
                    break
            
            if not last_user_message:
                return state
            
            if not self.llm:
                # Fallback response generation when LLM not available
                response = self._generate_fallback_response(last_user_message, state)
            else:
                try:
                    # Build prompt with context
                    prompt = self._build_prompt(last_user_message, state)
                    
                    # Generate response
                    response = await self.llm.ainvoke(prompt)
                    response = response.content if hasattr(response, 'content') else str(response)
                except Exception as e:
                    logger.error(f"Failed to generate LLM response: {e}")
                    response = self._generate_fallback_response(last_user_message, state)
            
            # Add AI message to state
            ai_message = AIMessage(content=response)
            state["messages"].append(ai_message)
            
            logger.info("Response generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_message = AIMessage(
                content="I apologize, but I encountered an error while processing your request. Please try again."
            )
            state["messages"].append(error_message)
        
        return state
    
    def _generate_fallback_response(self, user_message: str, state: ChatState) -> str:
        """Generate fallback responses when LLM is not available."""
        user_lower = user_message.lower()
        
        if state.get("context_documents"):
            return f"Based on the uploaded documents, I can help you with '{user_message}'. (Note: LLM service is currently unavailable, so this is a simplified response. Full AI analysis requires Databricks LLM connection.)"
        elif "hello" in user_lower or "hi" in user_lower:
            return "Hello! I'm your document intelligence assistant. You can upload documents and ask questions about them. (Note: Full AI capabilities require Databricks LLM connection.)"
        elif "document" in user_lower:
            return "I can help you analyze and understand documents. Please upload a document to get started. (Note: Advanced AI analysis requires Databricks LLM connection.)"
        elif "help" in user_lower:
            return "I can assist you with basic tasks:\n• Document upload and storage\n• Basic document information\n• General assistance\n\n(Note: Advanced AI features require Databricks LLM connection.)"
        else:
            return f"I understand you're asking about '{user_message}'. (Note: Full AI response capabilities require Databricks LLM connection. Please configure your Databricks credentials for complete functionality.)"
    
    def _build_prompt(self, user_message: str, state: ChatState) -> str:
        """Build prompt with context for LLM."""
        prompt_parts = []
        
        # System prompt
        prompt_parts.append(
            "You are a helpful AI assistant that specializes in document analysis and intelligent conversation. "
            "Use the provided context from documents to answer questions accurately and comprehensively."
        )
        
        # Add document context if available
        if state.get("context_documents"):
            prompt_parts.append("\nRelevant Document Context:")
            for i, doc in enumerate(state["context_documents"][:3]):  # Limit to top 3
                prompt_parts.append(f"\nContext {i+1}:")
                prompt_parts.append(doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"])
        
        # Add conversation history (last few messages)
        if len(state["messages"]) > 1:
            prompt_parts.append("\nRecent Conversation:")
            for msg in state["messages"][-6:]:  # Last 3 exchanges
                role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
                prompt_parts.append(f"{role}: {msg.content}")
        
        # Add current question
        prompt_parts.append(f"\nHuman: {user_message}")
        prompt_parts.append("\nAssistant:")
        
        return "\n".join(prompt_parts)
    
    async def _save_message_node(self, state: ChatState) -> ChatState:
        """Save the conversation to database."""
        logger.info(f"Saving messages for conversation {state['conversation_id']}")
        
        try:
            # Save the last message (AI response) to database
            if state["messages"]:
                last_message = state["messages"][-1]
                if isinstance(last_message, AIMessage):
                    self.db_manager.add_message(
                        conversation_id=state["conversation_id"],
                        role="assistant",
                        content=last_message.content,
                        metadata=state.get("metadata", {})
                    )
            
            logger.info("Messages saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving messages: {e}")
        
        return state
    
    async def send_message(
        self,
        user_message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a message and get a response."""
        
        # Generate IDs if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        if not thread_id:
            thread_id = f"thread_{conversation_id}"
        
        # Create initial state
        initial_state = ChatState(
            messages=[HumanMessage(content=user_message)],
            user_id=user_id,
            conversation_id=conversation_id,
            thread_id=thread_id,
            document_ids=document_ids or [],
            context_documents=[],
            last_retrieval=None,
            metadata={}
        )
        
        # Save user message to database
        try:
            self.db_manager.add_message(
                conversation_id=conversation_id,
                role="user",
                content=user_message,
                metadata={}
            )
        except Exception as e:
            logger.error(f"Error saving user message: {e}")
        
        # Process through graph
        try:
            config = {"configurable": {"thread_id": thread_id}}
            result = await self.graph.ainvoke(initial_state, config=config)
            
            # Extract AI response
            ai_response = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    ai_response = msg.content
                    break
            
            return {
                "response": ai_response,
                "conversation_id": conversation_id,
                "thread_id": thread_id,
                "context_used": len(result.get("context_documents", [])),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your message. Please try again.",
                "conversation_id": conversation_id,
                "thread_id": thread_id,
                "context_used": 0,
                "success": False,
                "error": str(e)
            }
    
    def create_conversation(
        self,
        user_id: str,
        title: str,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a new conversation."""
        try:
            conversation_id = str(uuid.uuid4())
            thread_id = f"thread_{conversation_id}"
            
            conversation = self.db_manager.create_conversation(
                user_id=user_id,
                title=title,
                thread_id=thread_id,
                document_ids=document_ids
            )
            
            return {
                "conversation_id": str(conversation.id),
                "thread_id": conversation.thread_id,
                "title": conversation.title,
                "document_ids": conversation.document_ids,
                "created_at": conversation.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation message history."""
        try:
            messages = self.db_manager.get_conversation_messages(conversation_id)
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.created_at.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
