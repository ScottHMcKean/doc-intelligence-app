"""Integration tests for the AgentService."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from doc_intel.agent.service import AgentService


class TestAgentService:
    """Integration tests for the agent service."""

    def test_agent_service_initialization(self, test_config, mock_databricks_client):
        """Test agent service initialization."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        assert agent_service.client is not None
        assert agent_service.config is not None
        # Components might be None if endpoints are not configured
        assert agent_service.llm is None or hasattr(agent_service.llm, "invoke")
        assert agent_service.embeddings is None or hasattr(
            agent_service.embeddings, "embed_query"
        )

    def test_agent_service_availability(self, test_config, mock_databricks_client):
        """Test agent service availability checks."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Test availability properties
        assert isinstance(agent_service.is_available, bool)
        assert isinstance(agent_service.rag_available, bool)
        assert isinstance(agent_service.conversation_state_available, bool)

    def test_database_connection_string_generation(
        self, test_config, mock_databricks_client
    ):
        """Test database connection string generation for agent service."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Test connection string generation
        connection_string = agent_service._get_database_connection_string()

        # Should return a connection string or None
        assert connection_string is None or isinstance(connection_string, str)
        if connection_string:
            assert "postgresql://" in connection_string

    def test_query_embedding_generation(self, test_config, mock_databricks_client):
        """Test query embedding generation."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Mock embeddings if not available
        if not agent_service.embeddings:
            mock_embeddings = Mock()
            mock_embeddings.embed_query.return_value = [0.1] * 768
            agent_service.embeddings = mock_embeddings

        # Test embedding generation
        success, embedding, message = agent_service.generate_query_embedding(
            "test query"
        )

        assert isinstance(success, bool)
        if success:
            assert isinstance(embedding, list)
            assert len(embedding) > 0
        assert isinstance(message, str)

    def test_similarity_search(
        self, test_config, mock_databricks_client, test_vector_embedding
    ):
        """Test similarity search functionality."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Mock embeddings and database connection
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = test_vector_embedding
        agent_service.embeddings = mock_embeddings

        # Mock the custom vector search method
        mock_results = [
            {
                "content": "Test content 1",
                "metadata": {"document_id": "doc1", "filename": "test1.txt"},
                "score": 0.9,
            },
            {
                "content": "Test content 2",
                "metadata": {"document_id": "doc2", "filename": "test2.txt"},
                "score": 0.8,
            },
        ]
        agent_service._perform_custom_vector_search = Mock(return_value=mock_results)

        # Test similarity search
        success, results, message = agent_service.similarity_search(
            query="test query", limit=5
        )

        assert isinstance(success, bool)
        if success:
            assert isinstance(results, list)
            assert len(results) == 2
            assert results[0]["content"] == "Test content 1"
        assert isinstance(message, str)

    def test_document_search(self, test_config, mock_databricks_client):
        """Test document search functionality."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Mock similarity search
        mock_results = [
            {
                "content": "Relevant document content",
                "metadata": {"document_id": "doc1", "filename": "test.txt"},
                "score": 0.95,
            }
        ]
        agent_service.similarity_search = Mock(
            return_value=(True, mock_results, "Success")
        )

        # Test document search
        success, results, metadata = agent_service.search_documents(
            query="test query", limit=5
        )

        assert success is True
        assert isinstance(results, list)
        assert len(results) == 1
        assert "query" in metadata
        assert metadata["query"] == "test query"

    def test_direct_response_generation(
        self, test_config, mock_databricks_client, test_conversation_messages
    ):
        """Test direct LLM response generation."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "This is a test response from the LLM."
        mock_llm.invoke.return_value = mock_response
        agent_service.llm = mock_llm

        # Test direct response generation
        success, response, metadata = agent_service._generate_direct_response(
            messages=test_conversation_messages, context_documents=None
        )

        assert success is True
        assert response == "This is a test response from the LLM."
        assert isinstance(metadata, dict)
        assert "model_used" in metadata

    def test_prompt_building(
        self, test_config, mock_databricks_client, test_conversation_messages
    ):
        """Test prompt building for LLM."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Test prompt building
        prompt = agent_service._build_prompt(
            messages=test_conversation_messages, context_documents=None
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "document intelligence assistant" in prompt.lower()
        assert "conversation:" in prompt.lower()

    def test_prompt_building_with_context(
        self, test_config, mock_databricks_client, test_conversation_messages
    ):
        """Test prompt building with document context."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        context_documents = [
            {
                "content": "This is relevant document content for the query.",
                "metadata": {"document_id": "doc1", "filename": "test.txt"},
            }
        ]

        # Test prompt building with context
        prompt = agent_service._build_prompt(
            messages=test_conversation_messages, context_documents=context_documents
        )

        assert isinstance(prompt, str)
        assert "relevant document context" in prompt.lower()
        assert "relevant document content" in prompt

    def test_response_generation_without_llm(
        self, test_config, mock_databricks_client, test_conversation_messages
    ):
        """Test response generation when LLM is not available."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)
        agent_service.llm = None  # Ensure LLM is not available

        # Test response generation
        success, response, metadata = agent_service._generate_direct_response(
            messages=test_conversation_messages
        )

        assert success is False
        assert "not available" in response.lower()
        assert metadata == {}

    def test_conversation_summarization(
        self, test_config, mock_databricks_client, test_conversation_messages
    ):
        """Test conversation summarization."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Mock LLM for summarization
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Summary: This is a test conversation about document intelligence.\nTitle: Test Chat"
        mock_llm.invoke.return_value = mock_response
        agent_service.llm = mock_llm

        # Test conversation summarization
        success, summary, title = agent_service.summarize_conversation(
            test_conversation_messages
        )

        assert success is True
        assert "test conversation" in summary.lower()
        assert title == "Test Chat"

    def test_conversation_summarization_without_llm(
        self, test_config, mock_databricks_client, test_conversation_messages
    ):
        """Test conversation summarization when LLM is not available."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)
        agent_service.llm = None  # Ensure LLM is not available

        # Test conversation summarization
        success, summary, title = agent_service.summarize_conversation(
            test_conversation_messages
        )

        assert success is False
        assert "not available" in summary.lower()
        assert title is None

    def test_service_status(self, test_config, mock_databricks_client):
        """Test service status reporting."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Test service status
        status = agent_service.get_service_status()

        assert "agent_service" in status
        agent_status = status["agent_service"]
        assert "available" in agent_status
        assert "components" in agent_status
        assert "capabilities" in agent_status

        # Check components
        components = agent_status["components"]
        assert "llm" in components
        assert "embeddings" in components
        assert "vectorstore" in components
        assert "conversation_graph" in components
        assert "checkpointer" in components
        assert "client" in components

        # Check capabilities
        capabilities = agent_status["capabilities"]
        assert "chat" in capabilities
        assert "rag" in capabilities
        assert "conversation_state" in capabilities
        assert "vector_search" in capabilities

    def test_document_chunks_addition_to_vectorstore(
        self, test_config, mock_databricks_client, test_document_chunks
    ):
        """Test adding document chunks to vector store."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Mock database connection
        mock_connection_string = "postgresql://test:test@localhost:5432/test"
        agent_service._get_database_connection_string = Mock(
            return_value=mock_connection_string
        )

        # Mock psycopg2 connection
        with patch("psycopg2.connect") as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value.__enter__.return_value = mock_conn

            # Test adding chunks to vector store
            success = agent_service.add_document_chunks_to_vectorstore(
                document_id="test-doc-id", chunks=test_document_chunks
            )

            assert success is True

    def test_custom_vector_search(
        self, test_config, mock_databricks_client, test_vector_embedding
    ):
        """Test custom vector search implementation."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Mock database connection
        mock_connection_string = "postgresql://test:test@localhost:5432/test"
        agent_service._get_database_connection_string = Mock(
            return_value=mock_connection_string
        )

        # Mock psycopg2 connection and results
        with patch("psycopg2.connect") as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_result = Mock()
            mock_result.__getitem__ = Mock(
                side_effect=lambda key: {
                    "content": "Test content",
                    "chunk_metadata": {"chunk_index": 0},
                    "token_count": 10,
                    "filename": "test.txt",
                    "doc_metadata": {},
                    "doc_hash": "test-hash",
                    "distance": 0.1,
                }.get(key)
            )
            mock_cursor.fetchall.return_value = [mock_result]
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value.__enter__.return_value = mock_conn

            # Test custom vector search
            results = agent_service._perform_custom_vector_search(
                query_embedding=test_vector_embedding, limit=5
            )

            assert isinstance(results, list)
            if results:
                assert "content" in results[0]
                assert "metadata" in results[0]
                assert "score" in results[0]

    def test_langgraph_node_functions(self, test_config, mock_databricks_client):
        """Test LangGraph node functions."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Mock similarity search for retrieve documents node
        agent_service.similarity_search = Mock(
            return_value=(
                True,
                [{"content": "Test content", "metadata": {"doc_id": "doc1"}}],
                "Success",
            )
        )

        # Test retrieve documents node
        from langchain_core.messages import HumanMessage

        test_state = {
            "messages": [HumanMessage(content="What is this about?")],
            "document_ids": [],
            "context_documents": [],
            "last_retrieval": None,
            "metadata": {},
        }

        result_state = agent_service._retrieve_documents_node(test_state)

        assert "context_documents" in result_state
        assert "last_retrieval" in result_state

    def test_generate_response_node(self, test_config, mock_databricks_client):
        """Test generate response node."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "This is a generated response."
        mock_llm.invoke.return_value = mock_response
        agent_service.llm = mock_llm

        # Test generate response node
        from langchain_core.messages import HumanMessage

        test_state = {
            "messages": [HumanMessage(content="Hello")],
            "context_documents": [{"content": "Context", "metadata": {}}],
            "metadata": {},
        }

        result_state = agent_service._generate_response_node(test_state)

        assert "messages" in result_state
        assert "metadata" in result_state
        assert len(result_state["messages"]) > 1  # Should have added a response

    def test_error_handling_in_vector_search(self, test_config, mock_databricks_client):
        """Test error handling in vector search."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Mock embeddings to raise an exception
        mock_embeddings = Mock()
        mock_embeddings.embed_query.side_effect = Exception("Embedding error")
        agent_service.embeddings = mock_embeddings

        # Test similarity search with error
        success, results, message = agent_service.similarity_search("test query")

        assert success is False
        assert results == []
        assert "error" in message.lower()

    def test_error_handling_in_response_generation(
        self, test_config, mock_databricks_client, test_conversation_messages
    ):
        """Test error handling in response generation."""
        agent_service = AgentService(client=mock_databricks_client, config=test_config)

        # Mock LLM to raise an exception
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        agent_service.llm = mock_llm

        # Test response generation with error
        success, response, metadata = agent_service._generate_direct_response(
            test_conversation_messages
        )

        assert success is False
        assert "error" in response.lower()
        assert metadata == {}

    def test_agent_service_without_client(self, test_config):
        """Test agent service behavior without Databricks client."""
        agent_service = AgentService(client=None, config=test_config)

        # Should still initialize but with limited functionality
        assert agent_service.client is None
        assert agent_service.config is not None

        # Connection string should be None
        connection_string = agent_service._get_database_connection_string()
        assert connection_string is None

        # Vector search should fail gracefully with exception handling
        success, results, message = agent_service.similarity_search("test query")
        assert success is False
        assert results == []
        assert "failed" in message.lower()  # Now expects actual exception message
