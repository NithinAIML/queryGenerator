"""
OpenAI API client wrapper that supports both old (0.28.0) and new OpenAI SDK versions
"""

import os
import importlib.util
import pkg_resources
import json
import time
from typing import Dict, List, Any, Optional

class OpenAIClient:
    """
    A wrapper class for OpenAI API that supports both legacy and new client versions
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_endpoint: Optional[str] = None, 
                api_base: Optional[str] = None):
        """
        Initialize the OpenAI client with appropriate version
        
        Parameters:
        -----------
        api_key : str, optional
            OpenAI API key (will use OPENAI_API_KEY env var if not provided)
        api_endpoint : str, optional
            API endpoint URL (will use OPENAI_API_ENDPOINT env var if not provided)
        api_base : str, optional
            Legacy parameter, same as api_endpoint (will use OPENAI_API_BASE env var if not provided)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Use api_endpoint, falling back to api_base for compatibility
        self.api_base = api_endpoint or api_base or os.environ.get("OPENAI_API_ENDPOINT") or os.environ.get("OPENAI_API_BASE")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Provide it via the api_key parameter or set OPENAI_API_KEY environment variable.")
        
        # Detect which OpenAI package version is installed
        self.openai_version = self._get_openai_version()
        self.client = self._initialize_client()
    
    def _get_openai_version(self) -> str:
        """Detects installed OpenAI package version"""
        try:
            version = pkg_resources.get_distribution("openai").version
            major_version = int(version.split('.')[0])
            return "new" if major_version >= 1 else "legacy"
        except pkg_resources.DistributionNotFound:
            raise ImportError("OpenAI package is not installed. Install it with 'pip install openai'")
    
    def _initialize_client(self):
        """Initialize the appropriate OpenAI client based on version"""
        import openai
        
        if self.openai_version == "legacy":
            # Legacy client (0.x.x versions)
            openai.api_key = self.api_key
            if self.api_base:
                openai.api_base = self.api_base
            return openai
        else:
            # New client (1.x.x versions)
            client_params = {"api_key": self.api_key}
            if self.api_base:
                client_params["base_url"] = self.api_base
            return openai.OpenAI(**client_params)
    
    def chat_completion(self, 
                        messages: List[Dict[str, str]], 
                        model: str = "gpt-3.5-turbo",
                        temperature: float = 0.7,
                        max_tokens: int = 1000,
                        **kwargs) -> Dict[str, Any]:
        """
        Send a chat completion request to OpenAI
        
        Parameters:
        -----------
        messages : List[Dict[str, str]]
            List of messages in the conversation
        model : str
            Model to use for completion
        temperature : float
            Sampling temperature between 0 and 2
        max_tokens : int
            Maximum number of tokens to generate
        **kwargs : dict
            Additional parameters to pass to the API
            
        Returns:
        --------
        Dict[str, Any]
            API response with completion
        """
        if self.openai_version == "legacy":
            # Legacy API (0.x.x)
            response = self.client.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            # Convert to dict to ensure consistent return format
            if not isinstance(response, dict):
                response = response.to_dict()
            return response
        else:
            # New API (1.x.x)
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            # Convert model object to dict for consistent handling
            return self._response_to_dict(response)
    
    def _response_to_dict(self, response) -> Dict[str, Any]:
        """Convert OpenAI response object to dictionary"""
        if hasattr(response, "model_dump"):
            # New OpenAI models have model_dump method
            return response.model_dump()
        
        # Fallback for other object types
        result = {}
        try:
            # Try to convert to dict using built-in methods if available
            if hasattr(response, "to_dict"):
                return response.to_dict()
            elif hasattr(response, "__dict__"):
                for key, value in response.__dict__.items():
                    if not key.startswith("_"):
                        if hasattr(value, "__dict__") or hasattr(value, "model_dump"):
                            result[key] = self._response_to_dict(value)
                        else:
                            result[key] = value
        except (AttributeError, TypeError):
            pass
        
        return result

    def get_chat_completion_content(self, response: Dict[str, Any]) -> str:
        """
        Extract the content from a chat completion response
        
        Parameters:
        -----------
        response : Dict[str, Any]
            Response from chat_completion
            
        Returns:
        --------
        str
            Text content of the completion
        """
        try:
            if self.openai_version == "legacy":
                # Legacy response format
                return response.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                # New response format
                choices = response.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    if isinstance(message, dict):
                        return message.get("content", "")
                    # Handle when message is an object with attributes
                    if hasattr(message, "content"):
                        return message.content or ""
            
            # If extraction methods above fail, try a more flexible approach
            # This handles unexpected response structures
            if "choices" in response:
                choices = response["choices"]
                if len(choices) > 0:
                    choice = choices[0]
                    
                    # Try to get content from various possible formats
                    if isinstance(choice, dict):
                        if "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
                            return choice["message"]["content"] or ""
                    
                    # If choice is an object with a message attribute
                    if hasattr(choice, "message"):
                        message = choice.message
                        if hasattr(message, "content"):
                            return message.content or ""
            
            # Return empty string if no content found
            return ""
        except Exception as e:
            print(f"Error extracting content from response: {e}")
            return ""