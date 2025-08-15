"""Chat service for LLM interactions."""

import logging
from typing import Optional, List, Dict, Any, Tuple

from ..config import config

logger = logging.getLogger(__name__)

try:
    from databricks_langchain import ChatDatabricks
except ImportError as e:
    logger.error(f"Failed to import required dependencies: {e}")
    raise ImportError(
        f"Missing required dependencies for chat service: {e}. "
        "Please ensure all dependencies are installed."
    )


class ChatService:
    """Service for LLM interactions with graceful fallback."""
    
    def __init__(
        self, 
        databricks_host: Optional[str] = None, 
        databricks_token: Optional[str] = None,
        chat_endpoint: Optional[str] = None
    ):
        self.databricks_host = databricks_host or config.databricks_host
        self.databricks_token = databricks_token or config.databricks_token
        self.chat_endpoint = chat_endpoint or config.databricks_llm_endpoint
        
        self.llm = None
        self._initialize()
    
    def _initialize(self):
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
    
    @property
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.llm is not None
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        context_documents: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Generate a chat response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            context_documents: Optional list of relevant documents for context
            
        Returns:
            Tuple of (success, response, metadata)
        """
        if not self.llm:
            return self._generate_fallback_response(messages, context_documents)
        
        try:
            # Build prompt with context
            prompt = self._build_prompt(messages, context_documents)
            
            # Generate response
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            metadata = {
                "model_used": "databricks_llm",
                "context_docs_count": len(context_documents) if context_documents else 0,
                "prompt_length": len(prompt)
            }
            
            logger.info("Successfully generated LLM response")
            return True, response_text, metadata
            
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            return self._generate_fallback_response(messages, context_documents)
    
    def _generate_fallback_response(
        self, 
        messages: List[Dict[str, str]], 
        context_documents: Optional[List[Dict[str, Any]]] = None
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
            "fallback_reason": "llm_not_available"
        }
        
        return True, response, metadata
    
    def _build_prompt(
        self, 
        messages: List[Dict[str, str]], 
        context_documents: Optional[List[Dict[str, Any]]] = None
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
    
    def summarize_conversation(self, messages: List[Dict[str, str]]) -> Tuple[bool, str, str]:
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
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            lines = response_text.split('\n')
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
