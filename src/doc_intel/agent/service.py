"""Simplified agent service for chat interface with LangGraph and PGVector."""

import logging
from typing import Optional, List, Dict, Any, Tuple, TypedDict, Annotated
from datetime import datetime
from databricks.sdk import WorkspaceClient
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, add_messages

logger = logging.getLogger(__name__)

from databricks_langchain import ChatDatabricks
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_postgres import PGVector


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


class AgentService:
    """
    Simplified agent service for chat interface with LangGraph and PGVector.

    Features:
    - LLM interactions with Databricks using automated authentication
    - State-based conversation management with LangGraph
    - Vector search against existing PGVector database vectors
    - Persistent conversation history with Postgres checkpointing

    Note: This service expects document chunks and vectors to already exist in the
    managed Databricks PostgreSQL instance with PGVector extension.
    """

    def __init__(self, client: Optional[WorkspaceClient], config: dict):
        """Initialize the agent service."""
        self.client = client
        self.config = config

        # Initialize core components
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.conversation_graph = None
        self.checkpointer = None

        # Initialize components
        self._initialize()

    def _get_database_connection_string(self) -> Optional[str]:
        """Get database connection string using the same pattern as database service."""
        try:
            # Get database instance name from config
            instance_name = self.config.get("database.instance_name")
            user = self.config.get("database.user", "databricks")
            database = self.config.get("database.database", "databricks_postgres")

            if not instance_name:
                return None

            # Get database instance from Databricks (same pattern as database service)
            instance = self.client.database.get_database_instance(name=instance_name)
            cred = self.client.database.generate_database_credential(
                request_id=str(uuid.uuid4()), instance_names=[instance_name]
            )

            # Return connection string
            return f"postgresql://{user}:{cred.token}@{instance.read_write_dns}:5432/{database}?sslmode=require"

        except Exception as e:
            logger.error(f"Failed to generate connection string: {str(e)}")
            return None

    def _initialize(self):
        """Initialize all components."""
        self._initialize_llm()
        self._initialize_embeddings()
        self._initialize_vectorstore()
        self._initialize_checkpointer()
        self._initialize_conversation_graph()

    def _initialize_llm(self):
        """Initialize Databricks LLM with automated authentication."""
        try:
            # Try to get endpoint from agent config first, then fall back to llm config
            endpoint = self.config.get("agent.llm.endpoint") or self.config.get(
                "agent.llm_endpoint"
            )
            if not endpoint:
                logger.warning("No LLM endpoint configured")
                return

            self.llm = ChatDatabricks(
                endpoint=endpoint,
                max_tokens=self.config.get("agent.llm.max_tokens", 512),
                temperature=self.config.get("agent.llm.temperature", 0.1),
            )
            logger.info("Successfully initialized Databricks LLM")
        except Exception as e:
            logger.error(f"Failed to initialize Databricks LLM: {e}")
            self.llm = None

    def _initialize_embeddings(self):
        """Initialize Databricks embeddings for user queries."""
        try:
            from databricks_langchain import DatabricksEmbeddings

            # Try to get endpoint from agent config first, then fall back to embedding config
            endpoint = self.config.get(
                "agent.retrieval.embedding_endpoint"
            ) or self.config.get("embedding.endpoint")
            if not endpoint:
                logger.warning("No embedding endpoint configured")
                return

            self.embeddings = DatabricksEmbeddings(endpoint=endpoint)
            logger.info("Successfully initialized Databricks embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize Databricks embeddings: {e}")
            self.embeddings = None

    def _initialize_vectorstore(self):
        """Initialize PGVector store - already exists in managed Databricks PostgreSQL."""
        try:
            connection_string = self._get_database_connection_string()
            if not connection_string:
                logger.warning("Connection string not available for vector store")
                return

            # PGVector already exists in the managed instance, just initialize connection
            # Use the initialized embeddings instead of None
            if not self.embeddings:
                logger.warning("No embeddings available for vector store")
                return

            # Create a collection first to ensure proper table structure
            # We'll use a simple collection name that PGVector can manage
            collection_name = "doc_intel_chunks"

            self.vectorstore = PGVector(
                embeddings=self.embeddings,  # First positional parameter
                connection=connection_string,
                collection_name=collection_name,
                pre_delete_collection=False,
            )

            # Test the connection by trying to get collection info
            try:
                # This will create the collection if it doesn't exist
                collection = self.vectorstore.get_collection()
                logger.info(
                    f"Successfully connected to PGVector collection: {collection_name}"
                )
            except Exception as collection_error:
                logger.warning(f"Collection initialization issue: {collection_error}")
                # Try to create the collection manually
                try:
                    self.vectorstore.create_collection()
                    logger.info(f"Created new PGVector collection: {collection_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create collection: {create_error}")
                    self.vectorstore = None
                    return

        except Exception as e:
            logger.error(f"Failed to connect to PGVector store: {e}")
            self.vectorstore = None

    def _initialize_checkpointer(self):
        """Initialize checkpointer for conversation state."""
        try:
            connection_string = self._get_database_connection_string()
            if not connection_string:
                logger.warning("Connection string not available for checkpointer")
                return

            self.checkpointer = PostgresSaver.from_conn_string(connection_string)
            logger.info("Successfully initialized Postgres checkpointer")
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
        # We now use custom vector search, so we only need embeddings
        return self.embeddings is not None

    @property
    def conversation_state_available(self) -> bool:
        """Check if conversation state management is available."""
        return self.conversation_graph is not None and self.checkpointer is not None

    # ===== Vector Search Methods =====

    def generate_query_embedding(
        self, query: str
    ) -> Tuple[bool, Optional[List[float]], str]:
        """Generate embedding for a user query."""
        if not self.embeddings:
            logger.warning("Embeddings not available for query")
            return False, None, "Embeddings not available"

        try:
            embedding = self.embeddings.embed_query(query)
            logger.info("Successfully generated query embedding")
            return True, embedding, "Generated query embedding"
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return False, None, f"Failed to generate embedding: {str(e)}"

    def similarity_search(
        self, query: str, limit: int = 5, document_ids: Optional[List[str]] = None
    ) -> Tuple[bool, List[Dict[str, Any]], str]:
        """Perform similarity search against existing database vectors."""
        try:
            # Generate embedding for the user query
            embed_success, query_embedding, embed_message = (
                self.generate_query_embedding(query)
            )
            if not embed_success:
                return False, [], f"Failed to generate query embedding: {embed_message}"

            # Search against existing vectors in the database using our custom method
            try:
                # Use our database service for vector search instead of PGVector
                # This gives us more control over the query and schema
                search_results = self._perform_custom_vector_search(
                    query_embedding, limit, document_ids
                )

                if search_results:
                    return True, search_results, "Vector search completed successfully"
                else:
                    return False, [], "No results found"

            except Exception as e:
                logger.error(f"Vector search failed: {str(e)}")
                return False, [], f"Vector search failed: {str(e)}"

        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return False, [], f"Search failed: {str(e)}"

    def _perform_custom_vector_search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform vector search using our custom database schema."""
        try:
            # Get database connection string
            connection_string = self._get_database_connection_string()
            if not connection_string:
                logger.error("No database connection string available")
                return []

            # Import psycopg2 for direct database access
            import psycopg2
            import psycopg2.extras

            # Connect to database and perform vector search
            with psycopg2.connect(connection_string) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    # Build the query based on whether we're filtering by document IDs
                    if document_ids:
                        query = """
                            SELECT dc.id, dc.content, dc.chunk_metadata, dc.token_count,
                                   d.filename, d.doc_metadata, d.doc_hash,
                                   dc.embedding <-> %s as distance
                            FROM document_chunks dc
                            JOIN documents d ON dc.document_id = d.id
                            WHERE d.doc_hash = ANY(%s)
                            ORDER BY distance
                            LIMIT %s
                        """
                        cur.execute(query, (query_embedding, document_ids, limit))
                    else:
                        query = """
                            SELECT dc.id, dc.content, dc.chunk_metadata, dc.token_count,
                                   d.filename, d.doc_metadata, d.doc_hash,
                                   dc.embedding <-> %s as distance
                            FROM document_chunks dc
                            JOIN documents d ON dc.document_id = d.id
                            ORDER BY distance
                            LIMIT %s
                        """
                        cur.execute(query, (query_embedding, limit))

                    results = cur.fetchall()

                    # Format results to match expected structure
                    formatted_results = []
                    for row in results:
                        formatted_results.append(
                            {
                                "content": row["content"],
                                "metadata": {
                                    "document_id": row["doc_hash"],
                                    "filename": row["filename"],
                                    "chunk_index": row.get("chunk_metadata", {}).get(
                                        "chunk_index"
                                    ),
                                    "token_count": row["token_count"],
                                    "distance": (
                                        float(row["distance"])
                                        if row["distance"] is not None
                                        else None
                                    ),
                                },
                                "score": (
                                    1.0 - float(row["distance"])
                                    if row["distance"] is not None
                                    else None
                                ),
                                "document_id": row["doc_hash"],
                                "chunk_index": row.get("chunk_metadata", {}).get(
                                    "chunk_index"
                                ),
                            }
                        )

                    return formatted_results

        except Exception as e:
            logger.error(f"Custom vector search failed: {str(e)}")
            return []

    def add_document_chunks_to_vectorstore(
        self, document_id: str, chunks: List[Dict[str, Any]]
    ) -> bool:
        """Add document chunks with embeddings to the vector store."""
        try:
            # Get database connection string
            connection_string = self._get_database_connection_string()
            if not connection_string:
                logger.error("No database connection string available")
                return False

            # Import psycopg2 for direct database access
            import psycopg2
            import psycopg2.extras
            import uuid
            from datetime import datetime, timezone

            # Connect to database and add chunks
            with psycopg2.connect(connection_string) as conn:
                with conn.cursor() as cur:
                    for i, chunk_data in enumerate(chunks):
                        chunk_id = str(uuid.uuid4())
                        now = datetime.now(timezone.utc)

                        # Insert chunk with embedding
                        cur.execute(
                            """INSERT INTO document_chunks 
                               (id, document_id, chunk_index, content, embedding, 
                                chunk_metadata, token_count, created_at) 
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
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

                    conn.commit()
                    logger.info(
                        f"Added {len(chunks)} chunks to vector store for document {document_id}"
                    )
                    return True

        except Exception as e:
            logger.error(f"Failed to add document chunks to vector store: {str(e)}")
            return False

    # ===== Chat Interface =====

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        context_documents: Optional[List[Dict[str, Any]]] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate a chat response."""
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
                thread_id=conversation_id,
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
            return False, "LLM not available", {}

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
            return False, f"Failed to generate response: {str(e)}", {}

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
            for i, doc in enumerate(context_documents[:3]):
                content = doc.get("content", "")[:500]
                prompt_parts.append(f"\nDocument {i+1}: {content}")

        # Add conversation history
        prompt_parts.append("\n\nConversation:")
        for message in messages[-5:]:
            role = message["role"]
            content = message["content"]
            prompt_parts.append(f"\n{role.title()}: {content}")

        prompt_parts.append("\n\nAssistant:")
        return "".join(prompt_parts)

    # ===== LangGraph Node Functions =====

    def _retrieve_documents_node(self, state: ChatState) -> ChatState:
        """LangGraph node for document retrieval."""
        try:
            if not state["messages"]:
                return state

            # Get the last user message
            last_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_message = msg.content
                    break

            if not last_message:
                return state

            # Perform similarity search using our method
            search_success, results, message = self.similarity_search(
                query=last_message, limit=3, document_ids=state.get("document_ids", [])
            )

            if search_success:
                # Convert to context documents format
                context_docs = []
                for result in results:
                    context_docs.append(
                        {"content": result["content"], "metadata": result["metadata"]}
                    )

                # Update state
                state["context_documents"] = context_docs
                state["last_retrieval"] = datetime.now().isoformat()

                logger.info(f"Retrieved {len(context_docs)} documents for context")
            else:
                logger.warning(f"Document retrieval failed: {message}")

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

    # ===== Document Search =====

    def search_documents(
        self, query: str, limit: int = 5, document_ids: Optional[List[str]] = None
    ) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
        """Search for relevant documents using vector similarity."""
        search_success, results, message = self.similarity_search(
            query=query, limit=limit, document_ids=document_ids
        )

        if not search_success:
            return False, [], {"error": message}

        metadata = {
            "query": query,
            "results_count": len(results),
            "searched_at": datetime.now().isoformat(),
        }

        logger.info(f"Found {len(results)} documents for query: {query}")
        return True, results, metadata

    # ===== Conversation Management =====

    def summarize_conversation(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[bool, str, Optional[str]]:
        """Summarize a conversation and suggest a title."""
        if not self.llm:
            return False, "LLM not available", None

        try:
            # Build a prompt for conversation summarization
            prompt = "Please analyze this conversation and provide:\n"
            prompt += "1. A brief summary (2-3 sentences)\n"
            prompt += "2. A short, descriptive title (max 50 characters)\n\n"
            prompt += "Conversation:\n"

            for message in messages[-5:]:  # Last 5 messages for context
                role = message["role"]
                content = message["content"][:200]  # Limit content length
                prompt += f"{role.title()}: {content}\n"

            prompt += (
                "\nPlease format your response as:\nSummary: [summary]\nTitle: [title]"
            )

            # Generate response
            response = self.llm.invoke(prompt)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Parse response to extract summary and title
            summary = ""
            title = None

            lines = response_text.split("\n")
            for line in lines:
                if line.startswith("Summary:"):
                    summary = line.replace("Summary:", "").strip()
                elif line.startswith("Title:"):
                    title = line.replace("Title:", "").strip()

            if not title:
                title = "New Conversation"

            return True, summary, title

        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            return False, f"Failed to summarize: {str(e)}", None

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
                    "client": self.client is not None,
                },
                "capabilities": {
                    "chat": self.is_available,
                    "rag": self.rag_available,
                    "conversation_state": self.conversation_state_available,
                    "vector_search": self.rag_available,
                },
            }
        }
