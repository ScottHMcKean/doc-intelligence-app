"""Agent service combining chat and document intelligence capabilities."""

import logging
import uuid
import asyncio
import os
from typing import Optional, List, Dict, Any, Tuple, TypedDict, Annotated
from datetime import datetime
from databricks.sdk import WorkspaceClient

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from ..config import config
from ..database.schema import DatabaseManager

logger = logging.getLogger(__name__)

# Required imports with graceful fallback
try:
    from databricks_langchain import ChatDatabricks, DatabricksEmbeddings
    from langgraph.checkpoint.postgres import PostgresSaver
    from langchain_community.vectorstores.pgvector import PGVector
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as e:
    logger.error(f"Failed to import required dependencies: {e}")
    raise ImportError(
        f"Missing required dependencies for agent service: {e}. "
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


class DocumentRAGState(TypedDict):
    """State for document RAG processing."""

    document_id: str
    filename: str
    content: str
    chunks: List[Dict[str, Any]]
    embeddings: List[List[float]]
    status: str
    error: Optional[str]


class AgentService:
    """
    Unified agent service combining chat, LangGraph, and RAG capabilities.

    This service provides:
    - LLM interactions with Databricks
    - State-based conversation management with LangGraph
    - Document processing and vector search (RAG)
    - Persistent conversation history with Postgres checkpointing
    - Graceful fallbacks for offline/mock mode
    """

    def __init__(self, client: Optional[WorkspaceClient], config: dict):
        """Initialize the unified agent service."""
        self.client = client
        self.config = config

        # Get configuration values
        self.chat_endpoint = config.get("llm_endpoint")
        self.embedding_endpoint = config.get("embedding_endpoint")
        self.job_id = config.get("job_id")
        self.default_cluster_key = config.get("default_cluster_key", "default_cluster")
        self.processing_notebook_path = config.get(
            "processing_notebook_path", "/Workspace/notebooks/document_processing"
        )

        # LLM config
        self.llm_max_tokens = config.get("llm", {}).get("max_tokens", 512)
        self.llm_temperature = config.get("llm", {}).get("temperature", 0.1)

        # RAG config
        self.rag_chunk_size = config.get("rag", {}).get("chunk_size", 1000)
        self.rag_chunk_overlap = config.get("rag", {}).get("chunk_overlap", 200)
        self.rag_similarity_threshold = config.get("rag", {}).get(
            "similarity_threshold", 0.7
        )
        self.rag_max_results = config.get("rag", {}).get("max_results", 5)

        # Conversation config
        self.conversation_max_history_messages = config.get("conversation", {}).get(
            "max_history_messages", 10
        )
        self.conversation_title_generation_enabled = config.get("conversation", {}).get(
            "title_generation_enabled", True
        )
        self.conversation_auto_title = config.get("conversation", {}).get(
            "auto_title", True
        )
        self.conversation_persistent_state = config.get("conversation", {}).get(
            "persistent_state", True
        )

        # Checkpointer config
        self.checkpointer_type = config.get("checkpointer", {}).get("type", "auto")

        # Get Databricks credentials from client
        self.databricks_host = None
        self.databricks_token = None
        if self.client:
            try:
                self.databricks_host = getattr(self.client.config, "host", None)
                self.databricks_token = getattr(self.client.config, "token", None)
            except:
                pass

        # Database configuration - will need to get from global config temporarily
        from ..config import config as global_config

        self.postgres_connection = global_config.database_connection_string
        self.db_manager = (
            DatabaseManager(self.postgres_connection)
            if self.postgres_connection
            else None
        )

        # Initialize core components
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.conversation_graph = None
        self.checkpointer = None
        self.text_splitter = None

        # Create event loop for async operations
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self._initialize()

    def _initialize(self):
        """Initialize all components."""
        self._initialize_llm()
        self._initialize_embeddings()
        self._initialize_text_splitter()
        self._initialize_vectorstore()
        self._initialize_checkpointer()
        self._initialize_conversation_graph()

    def _initialize_llm(self):
        """Initialize Databricks LLM."""
        if not config.databricks_available or not self.chat_endpoint:
            logger.warning("Databricks LLM not available")
            return

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

    def _initialize_embeddings(self):
        """Initialize Databricks embeddings."""
        if not config.databricks_available or not self.embedding_endpoint:
            logger.warning("Databricks embeddings not available")
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

    def _initialize_text_splitter(self):
        """Initialize text splitter for document chunking."""
        try:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            logger.info("Successfully initialized text splitter")
        except Exception as e:
            logger.error(f"Failed to initialize text splitter: {e}")
            self.text_splitter = None

    def _initialize_vectorstore(self):
        """Initialize vector store."""
        if not self.embeddings or not self.postgres_connection:
            logger.warning(
                "Vector store not available - missing embeddings or database"
            )
            return

        try:
            self.vectorstore = PGVector(
                connection_string=self.postgres_connection,
                embedding_function=self.embeddings,
                collection_name="document_embeddings",
            )
            logger.info("Successfully initialized vector store")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.vectorstore = None

    def _initialize_checkpointer(self):
        """Initialize checkpointer for conversation state."""
        try:
            # Import here to avoid circular imports
            from ..langgraph.checkpointing import create_checkpointer

            self.checkpointer = create_checkpointer(
                connection_string=self.postgres_connection
            )

            if self.checkpointer:
                checkpointer_type = config.effective_checkpointer_type
                logger.info(
                    f"Successfully initialized {checkpointer_type} checkpointer"
                )
            else:
                logger.warning("Failed to initialize any checkpointer")

        except Exception as e:
            logger.error(f"Failed to initialize checkpointer: {e}")
            self.checkpointer = None

    def _initialize_conversation_graph(self):
        """Initialize LangGraph conversation workflow."""
        if not self.llm:
            logger.warning("Conversation graph not available - missing LLM")
            return

        try:
            # Create the conversation graph
            workflow = StateGraph(ChatState)

            # Add nodes
            workflow.add_node("retrieve_documents", self._retrieve_documents_node)
            workflow.add_node("generate_response", self._generate_response_node)

            # Add edges
            workflow.set_entry_point("retrieve_documents")
            workflow.add_edge("retrieve_documents", "generate_response")
            workflow.set_finish_point("generate_response")

            # Compile with checkpointer if available
            if self.checkpointer:
                self.conversation_graph = workflow.compile(
                    checkpointer=self.checkpointer
                )
            else:
                self.conversation_graph = workflow.compile()

            logger.info("Successfully initialized conversation graph")
        except Exception as e:
            logger.error(f"Failed to initialize conversation graph: {e}")
            self.conversation_graph = None

    @property
    def is_available(self) -> bool:
        """Check if agent service is available."""
        return self.llm is not None

    @property
    def rag_available(self) -> bool:
        """Check if RAG capabilities are available."""
        return self.vectorstore is not None and self.embeddings is not None

    @property
    def conversation_state_available(self) -> bool:
        """Check if conversation state management is available."""
        return self.conversation_graph is not None and self.checkpointer is not None

    # ===== Chat Interface =====

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        context_documents: Optional[List[Dict[str, Any]]] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Generate a chat response using available capabilities.

        Args:
            messages: List of message dicts with 'role' and 'content'
            context_documents: Optional list of relevant documents for context
            conversation_id: Optional conversation ID for state tracking
            user_id: Optional user ID for state tracking

        Returns:
            Tuple of (success, response, metadata)
        """

        # Use LangGraph conversation if available and we have conversation context
        if (
            self.conversation_graph
            and conversation_id
            and user_id
            and self.conversation_state_available
        ):
            return self._generate_stateful_response(
                messages, context_documents, conversation_id, user_id
            )

        # Fall back to direct LLM interaction
        return self._generate_direct_response(messages, context_documents)

    def _generate_stateful_response(
        self,
        messages: List[Dict[str, str]],
        context_documents: Optional[List[Dict[str, Any]]],
        conversation_id: str,
        user_id: str,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate response using LangGraph stateful conversation."""
        try:
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))

            # Create initial state
            initial_state = ChatState(
                messages=langchain_messages,
                user_id=user_id,
                conversation_id=conversation_id,
                thread_id=conversation_id,  # Use conversation_id as thread_id
                document_ids=[],
                context_documents=context_documents or [],
                last_retrieval=None,
                metadata={},
            )

            # Run the graph
            config_dict = {"configurable": {"thread_id": conversation_id}}
            result = self.conversation_graph.invoke(initial_state, config=config_dict)

            # Extract response
            if result["messages"]:
                last_message = result["messages"][-1]
                response_text = last_message.content
            else:
                response_text = "I apologize, but I couldn't generate a response."

            metadata = {
                "model_used": "databricks_llm_stateful",
                "context_docs_count": (
                    len(context_documents) if context_documents else 0
                ),
                "conversation_id": conversation_id,
                "retrieval_performed": result.get("last_retrieval") is not None,
            }

            return True, response_text, metadata

        except Exception as e:
            logger.error(f"Failed to generate stateful response: {e}")
            return self._generate_direct_response(messages, context_documents)

    def _generate_direct_response(
        self,
        messages: List[Dict[str, str]],
        context_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate response using direct LLM interaction."""
        if not self.llm:
            return self._generate_fallback_response(messages, context_documents)

        try:
            # Build prompt with context
            prompt = self._build_prompt(messages, context_documents)

            # Generate response
            response = self.llm.invoke(prompt)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            metadata = {
                "model_used": "databricks_llm_direct",
                "context_docs_count": (
                    len(context_documents) if context_documents else 0
                ),
                "prompt_length": len(prompt),
            }

            logger.info("Successfully generated direct LLM response")
            return True, response_text, metadata

        except Exception as e:
            logger.error(f"Failed to generate direct LLM response: {e}")
            return self._generate_fallback_response(messages, context_documents)

    def _generate_mock_response(
        self,
        messages: List[Dict[str, str]],
        context_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate mock response for testing."""
        if not messages:
            return False, "No messages provided", {}

        last_message = messages[-1]["content"].lower()

        if context_documents:
            response = f"[MOCK] Based on {len(context_documents)} document(s), I can help you with: '{messages[-1]['content']}'"
        elif "hello" in last_message or "hi" in last_message:
            response = "[MOCK] Hello! I'm your document intelligence assistant."
        elif "document" in last_message:
            response = "[MOCK] I can help you analyze and understand documents."
        else:
            response = (
                f"[MOCK] I understand you're asking about: '{messages[-1]['content']}'"
            )

        metadata = {
            "model_used": "mock",
            "context_docs_count": len(context_documents) if context_documents else 0,
            "mock_mode": True,
        }

        return True, response, metadata

    def _generate_fallback_response(
        self,
        messages: List[Dict[str, str]],
        context_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate fallback response when LLM is not available."""
        if not messages:
            return False, "No messages provided", {}

        last_message = messages[-1]["content"].lower()

        if context_documents:
            response = f"Based on the uploaded documents, I can help you with your question about '{messages[-1]['content']}'. (Note: Full AI analysis requires Databricks LLM connection.)"
        elif "hello" in last_message or "hi" in last_message:
            response = "Hello! I'm your document intelligence assistant. (Note: Full AI capabilities require Databricks LLM connection.)"
        elif "document" in last_message:
            response = "I can help you analyze and understand documents. (Note: Advanced AI analysis requires Databricks LLM connection.)"
        elif "help" in last_message:
            response = "I can assist you with basic tasks:\n• Document upload and storage\n• Basic document information\n• General assistance\n\n(Note: Advanced AI features require Databricks LLM connection.)"
        else:
            response = f"I understand you're asking about '{messages[-1]['content']}'. (Note: Full AI response capabilities require Databricks LLM connection.)"

        metadata = {
            "model_used": "fallback",
            "context_docs_count": len(context_documents) if context_documents else 0,
            "fallback_reason": "llm_not_available",
        }

        return True, response, metadata

    def _build_prompt(
        self,
        messages: List[Dict[str, str]],
        context_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build prompt with context for LLM."""
        prompt_parts = []

        # System prompt
        prompt_parts.append(
            "You are a helpful document intelligence assistant. "
            "You can answer questions about uploaded documents and provide general assistance."
        )

        # Add document context if available
        if context_documents:
            prompt_parts.append("\n\nRelevant document context:")
            for i, doc in enumerate(context_documents[:3]):  # Limit to top 3 docs
                content = doc.get("content", "")[:500]  # Limit content length
                prompt_parts.append(f"\nDocument {i+1}: {content}")

        # Add conversation history
        prompt_parts.append("\n\nConversation:")
        for message in messages[-5:]:  # Include last 5 messages
            role = message["role"]
            content = message["content"]
            prompt_parts.append(f"\n{role.title()}: {content}")

        prompt_parts.append("\n\nAssistant:")
        return "".join(prompt_parts)

    # ===== LangGraph Node Functions =====

    def _retrieve_documents_node(self, state: ChatState) -> ChatState:
        """LangGraph node for document retrieval."""
        try:
            if not self.vectorstore or not state["messages"]:
                return state

            # Get the last user message
            last_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_message = msg.content
                    break

            if not last_message:
                return state

            # Perform similarity search
            docs = self.vectorstore.similarity_search(last_message, k=3)

            # Convert to context documents
            context_docs = []
            for doc in docs:
                context_docs.append(
                    {"content": doc.page_content, "metadata": doc.metadata}
                )

            # Update state
            state["context_documents"] = context_docs
            state["last_retrieval"] = datetime.now().isoformat()

            logger.info(f"Retrieved {len(context_docs)} documents for context")

        except Exception as e:
            logger.error(f"Error in document retrieval node: {e}")

        return state

    def _generate_response_node(self, state: ChatState) -> ChatState:
        """LangGraph node for response generation."""
        try:
            if not self.llm or not state["messages"]:
                return state

            # Convert messages to prompt format
            messages_dict = []
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    messages_dict.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages_dict.append({"role": "assistant", "content": msg.content})

            # Build prompt with context
            prompt = self._build_prompt(messages_dict, state["context_documents"])

            # Generate response
            response = self.llm.invoke(prompt)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Add response to messages
            state["messages"].append(AIMessage(content=response_text))

            # Update metadata
            state["metadata"]["response_generated"] = datetime.now().isoformat()
            state["metadata"]["context_docs_used"] = len(state["context_documents"])

            logger.info("Generated response in LangGraph node")

        except Exception as e:
            logger.error(f"Error in response generation node: {e}")
            # Add error message
            state["messages"].append(
                AIMessage(
                    content="I apologize, but I encountered an error generating a response."
                )
            )

        return state

    # ===== Document Processing (RAG) =====

    def process_document(
        self, content: str, filename: str, document_id: Optional[str] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Process a document for RAG capabilities.

        Args:
            content: Document content
            filename: Document filename
            document_id: Optional document ID

        Returns:
            Tuple of (success, message, metadata)
        """

        if not self.text_splitter or not self.embeddings or not self.vectorstore:
            return (
                False,
                "Document processing not available - missing required components",
                {},
            )

        try:
            doc_id = document_id or str(uuid.uuid4())

            # Split document into chunks
            chunks = self.text_splitter.split_text(content)

            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "document_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                }
                documents.append({"content": chunk, "metadata": doc_metadata})

            # Add to vector store
            texts = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]

            self.vectorstore.add_texts(texts, metadatas)

            metadata = {
                "document_id": doc_id,
                "filename": filename,
                "chunks_created": len(chunks),
                "processed_at": datetime.now().isoformat(),
            }

            logger.info(
                f"Successfully processed document {filename} into {len(chunks)} chunks"
            )
            return True, f"Successfully processed {filename}", metadata

        except Exception as e:
            logger.error(f"Failed to process document {filename}: {e}")
            return False, f"Failed to process document: {str(e)}", {}

    def search_documents(
        self, query: str, limit: int = 5
    ) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Search for relevant documents using vector similarity.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            Tuple of (success, results, metadata)
        """

        if not self.vectorstore:
            return False, [], {"error": "Vector search not available"}

        try:
            docs = self.vectorstore.similarity_search(query, k=limit)

            results = []
            for doc in docs:
                results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": getattr(doc, "score", None),
                    }
                )

            metadata = {
                "query": query,
                "results_count": len(results),
                "searched_at": datetime.now().isoformat(),
            }

            logger.info(f"Found {len(results)} documents for query: {query}")
            return True, results, metadata

        except Exception as e:
            logger.error(f"Failed to search documents for query '{query}': {e}")
            return False, [], {"error": str(e)}

    # ===== Conversation Management =====

    def summarize_conversation(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[bool, str, str]:
        """
        Generate a conversation summary for title generation.

        Returns:
            Tuple of (success, summary, suggested_title)
        """
        if not messages:
            return False, "", "New Conversation"

        if not self.llm:
            # Fallback summary generation
            first_user_msg = None
            for msg in messages:
                if msg["role"] == "user":
                    first_user_msg = msg["content"]
                    break

            if first_user_msg:
                # Create simple title from first message
                title = first_user_msg[:50].strip()
                if len(first_user_msg) > 50:
                    title += "..."
                return True, f"Conversation about: {title}", title
            else:
                return True, "General conversation", "General Chat"

        try:
            # Build summarization prompt
            prompt = "Summarize this conversation in 1-2 sentences and suggest a short title (max 50 chars):\n\n"
            for msg in messages[-10:]:  # Last 10 messages
                prompt += f"{msg['role'].title()}: {msg['content']}\n"
            prompt += "\nProvide: Summary: [summary]\nTitle: [title]"

            response = self.llm.invoke(prompt)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Parse response
            lines = response_text.split("\n")
            summary = ""
            title = "Conversation"

            for line in lines:
                if line.startswith("Summary:"):
                    summary = line.replace("Summary:", "").strip()
                elif line.startswith("Title:"):
                    title = line.replace("Title:", "").strip()[:50]

            return True, summary or "Conversation summary", title or "Conversation"

        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            return False, "", "Conversation"

    def validate_response(self, response: str) -> Tuple[bool, str]:
        """
        Validate LLM response for quality and safety.

        Returns:
            Tuple of (is_valid, reason)
        """
        if not response or not response.strip():
            return False, "Empty response"

        if len(response) < 10:
            return False, "Response too short"

        if len(response) > 2000:
            return False, "Response too long"

        # Add more validation rules as needed
        return True, "Response is valid"

    # ===== Service Status =====

    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            "agent_service": {
                "available": self.is_available,
                "components": {
                    "llm": self.llm is not None,
                    "embeddings": self.embeddings is not None,
                    "vectorstore": self.vectorstore is not None,
                    "conversation_graph": self.conversation_graph is not None,
                    "checkpointer": self.checkpointer is not None,
                    "text_splitter": self.text_splitter is not None,
                },
                "capabilities": {
                    "chat": self.is_available,
                    "rag": self.rag_available,
                    "conversation_state": self.conversation_state_available,
                },
            }
        }
