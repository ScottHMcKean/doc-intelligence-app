"""Databricks LLM integration for chat functionality."""

import os
from typing import List, Dict, Any, Optional, Generator

import streamlit as st
from databricks.sdk import WorkspaceClient

from ..auth import get_databricks_client
from ..config import MOCK_MODE


class ChatDatabricks:
    """Databricks LLM client for chat conversations."""

    def __init__(
        self,
        model_name: str = "databricks-meta-llama-3-1-70b-instruct",
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ):
        self.client = get_databricks_client()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat_completion(
        self, messages: List[Dict[str, str]], stream: bool = False
    ) -> str:
        """
        Generate chat completion using Databricks serving endpoint.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            stream: Whether to stream the response

        Returns:
            Generated response text
        """
        if MOCK_MODE:
            # Return a mock response based on the last user message
            last_user_message = ""
            for message in reversed(messages):
                if message["role"] == "user":
                    last_user_message = message["content"].lower()
                    break

            if "document" in last_user_message or "pdf" in last_user_message:
                return "Based on the document you've uploaded, I can see that it contains information about document intelligence and AI processing capabilities. The document discusses features like automated parsing, content extraction, and multi-format support."
            elif "hello" in last_user_message or "hi" in last_user_message:
                return "Hello! I'm an AI assistant that can help you with questions about your documents or general conversation. What would you like to know?"
            elif "features" in last_user_message:
                return "The key features include:\n• Automated document parsing\n• Intelligent content extraction\n• Multi-format support (PDF, DOCX, TXT)\n• Real-time processing\n• AI-powered analysis"
            else:
                return f"I understand you're asking about '{last_user_message}'. I can help you analyze documents, answer questions about their content, and provide insights based on the information they contain."

        try:
            # Format messages for Databricks API
            formatted_messages = []
            for message in messages:
                formatted_messages.append(
                    {"role": message["role"], "content": message["content"]}
                )

            # Call the serving endpoint
            response = self.client.serving_endpoints.query(
                name=self.model_name,
                dataframe_records=[
                    {
                        "messages": formatted_messages,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "stream": stream,
                    }
                ],
            )

            if stream:
                return self._handle_streaming_response(response)
            else:
                return self._handle_response(response)

        except Exception as e:
            st.error(f"Chat completion failed: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    def _handle_response(self, response) -> str:
        """Handle non-streaming response."""
        try:
            if hasattr(response, "predictions") and response.predictions:
                prediction = response.predictions[0]
                if isinstance(prediction, dict) and "choices" in prediction:
                    return prediction["choices"][0]["message"]["content"]
                elif isinstance(prediction, dict) and "content" in prediction:
                    return prediction["content"]
                else:
                    return str(prediction)
            return "No response generated."
        except Exception as e:
            st.error(f"Error parsing response: {str(e)}")
            return "Error parsing response."

    def _handle_streaming_response(self, response) -> Generator[str, None, None]:
        """Handle streaming response."""
        try:
            for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield delta.content
        except Exception as e:
            st.error(f"Error in streaming response: {str(e)}")
            yield "Error in streaming response."

    def chat_with_context(
        self,
        user_message: str,
        context_chunks: List[str],
        conversation_history: List[Dict[str, str]] = None,
    ) -> str:
        """
        Chat with document context.

        Args:
            user_message: User's question
            context_chunks: Relevant document chunks
            conversation_history: Previous conversation messages

        Returns:
            Generated response
        """
        # Build context from chunks
        context = "\n\n".join(
            [
                f"Document Context {i+1}:\n{chunk}"
                for i, chunk in enumerate(context_chunks[:5])  # Limit to 5 chunks
            ]
        )

        # Create system prompt
        system_prompt = f"""You are an AI assistant helping users understand documents. 
Use the following document context to answer questions accurately and helpfully.

{context}

Instructions:
- Answer based primarily on the provided document context
- If the context doesn't contain enough information, say so clearly
- Be concise but comprehensive
- Cite specific parts of the document when relevant
- If asked about something not in the context, explain that limitation"""

        # Build message history
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history[-10:])  # Limit to last 10 messages

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return self.chat_completion(messages)

    def chat_with_vector_search(
        self,
        user_message: str,
        vector_search_endpoint: str,
        vector_search_index: str,
        conversation_history: List[Dict[str, str]] = None,
    ) -> str:
        """
        Chat using vector search for context retrieval.

        Args:
            user_message: User's question
            vector_search_endpoint: Databricks vector search endpoint
            vector_search_index: Vector search index name
            conversation_history: Previous conversation messages

        Returns:
            Generated response
        """
        try:
            # Query vector search for relevant chunks
            # Note: This is a placeholder - implement actual vector search query
            search_results = self._query_vector_search(
                user_message, vector_search_endpoint, vector_search_index
            )

            # Extract text content from search results
            context_chunks = [
                result.get("content", "") for result in search_results[:5]
            ]

            return self.chat_with_context(
                user_message, context_chunks, conversation_history
            )

        except Exception as e:
            st.error(f"Vector search chat failed: {str(e)}")
            return (
                "I'm having trouble accessing the document database. Please try again."
            )

    def _query_vector_search(
        self, query: str, endpoint: str, index: str, num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query Databricks vector search.

        This is a placeholder implementation - replace with actual vector search API calls.
        """
        try:
            # Placeholder for vector search query
            # In a real implementation, you would:
            # 1. Encode the query using the same embedding model
            # 2. Query the vector search index
            # 3. Return the most similar chunks

            # For now, return empty results
            return []

        except Exception as e:
            st.error(f"Vector search query failed: {str(e)}")
            return []

    def generate_conversation_title(self, first_message: str) -> str:
        """Generate a title for the conversation based on the first message."""
        prompt = f"""Generate a concise, descriptive title (max 50 characters) for a conversation that starts with this message:

"{first_message}"

Return only the title, nothing else."""

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates conversation titles.",
            },
            {"role": "user", "content": prompt},
        ]

        title = self.chat_completion(messages)
        return title.strip().strip('"')[:50]  # Clean and limit length
