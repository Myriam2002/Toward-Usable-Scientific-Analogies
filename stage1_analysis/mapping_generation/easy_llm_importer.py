"""
Easy LLM Importer - Unified interface for multiple LLM providers
Supports OpenAI, OpenRouter, DeepInfra with automatic model routing
"""

import os
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum
import json


class Provider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    DEEPINFRA = "deepinfra"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    provider: Provider
    api_name: str  # The actual API identifier
    
    
class ModelRegistry:
    """Registry mapping model names to their providers and configurations"""
    
    # Model mappings - easily extensible
    MODELS = {
        # GPT models
        "gpt-oss-20b": ModelConfig("gpt-oss-20b", Provider.OPENROUTER, "openai/gpt-oss-20b"),
        "gpt-oss-120b": ModelConfig("gpt-oss-120b", Provider.OPENROUTER, "openai/gpt-oss-120b"),

        "gpt-4.1-mini": ModelConfig("gpt-4.1-mini", Provider.OPENAI, "gpt-4.1-mini"),
        "gpt-4.1-nano": ModelConfig("gpt-4.1-nano", Provider.OPENAI, "gpt-4.1-nano"),
        #"gpt-5.1": ModelConfig("gpt-5.1", Provider.OPENAI, "gpt-5.1"),
        
        # Grok models
        "grok-4-fast": ModelConfig("grok-4-fast", Provider.OPENROUTER, "x-ai/grok-4-fast"),
        # Gemini models
        "gemini-2.5-flash-lite": ModelConfig("gemini-2.5-flash-lite", Provider.OPENROUTER, "google/gemini-2.5-flash-lite"),
    
        # Llama models
        "llama-3.1-405b-instruct": ModelConfig("llama-3.1-405b-instruct", Provider.OPENROUTER, "meta-llama/llama-3.1-405b-instruct"),
        "meta-llama-3-1-70b-instruct": ModelConfig("meta-llama-3-1-70b-instruct", Provider.DEEPINFRA, "meta-llama/Meta-Llama-3.1-70B-Instruct"),
        "meta-llama-3-1-8b-instruct": ModelConfig("meta-llama-3-1-8b-instruct", Provider.DEEPINFRA, "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        
        # Deepseek models
        "deepseek-r1": ModelConfig("deepseek-r1", Provider.DEEPINFRA, "deepseek-ai/DeepSeek-R1"),
        
        # Qwen models
        "qwen3-14b": ModelConfig("qwen3-14b", Provider.DEEPINFRA, "Qwen/Qwen3-14B"),
        "qwen3-32b": ModelConfig("qwen3-32b", Provider.DEEPINFRA, "Qwen/Qwen3-32B"),
    }
    
    @classmethod
    def get_model(cls, model_name: str) -> ModelConfig:
        """Get model configuration by name"""
        # Normalize model name (case-insensitive, handle spaces)
        normalized_name = model_name.lower().strip().replace(" ", "-")
        
        if normalized_name not in cls.MODELS:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(cls.MODELS.keys())}"
            )
        return cls.MODELS[normalized_name]
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all available model names"""
        return list(cls.MODELS.keys())


class LLMClient:
    """Unified LLM client that routes to appropriate provider"""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        deepinfra_api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ):
        """
        Initialize LLM client with API keys
        
        Args:
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            openrouter_api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            deepinfra_api_key: DeepInfra API key (or set DEEPINFRA_API_KEY env var)
            site_url: Your site URL for OpenRouter rankings (optional)
            site_name: Your site name for OpenRouter rankings (optional)
        """
        # Load API keys from environment or parameters
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.deepinfra_api_key = deepinfra_api_key or os.getenv("DEEPINFRA_API_KEY")
        
        self.site_url = site_url or os.getenv("SITE_URL", "")
        self.site_name = site_name or os.getenv("SITE_NAME", "")
        
        # Initialize provider clients lazily
        self._openai_client = None
        self._openrouter_client = None
        self._deepinfra_client = None
    
    def _get_openai_client(self):
        """Lazy initialization of OpenAI client"""
        if self._openai_client is None:
            from openai import OpenAI
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY or pass openai_api_key parameter")
            self._openai_client = OpenAI(api_key=self.openai_api_key)
        return self._openai_client
    
    def _get_openrouter_client(self):
        """Lazy initialization of OpenRouter client (via OpenAI SDK)"""
        if self._openrouter_client is None:
            from openai import OpenAI
            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API key not provided. Set OPENROUTER_API_KEY or pass openrouter_api_key parameter")
            self._openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
            )
        return self._openrouter_client
    
    def _get_deepinfra_client(self):
        """Lazy initialization of DeepInfra client (via OpenAI SDK)"""
        if self._deepinfra_client is None:
            from openai import OpenAI
            if not self.deepinfra_api_key:
                raise ValueError("DeepInfra API key not provided. Set DEEPINFRA_API_KEY or pass deepinfra_api_key parameter")
            self._deepinfra_client = OpenAI(
                base_url="https://api.deepinfra.com/v1/openai",
                api_key=self.deepinfra_api_key,
            )
        return self._deepinfra_client
    
    def chat(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send a chat completion request to the appropriate provider
        
        Args:
            model_name: Name of the model (e.g., "gpt-5-mini", "llama-3.3-70b")
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The generated text response
        """
        # Get model configuration
        model_config = ModelRegistry.get_model(model_name)
        
        # Route to appropriate provider
        if model_config.provider == Provider.OPENAI:
            return self._chat_openai(model_config, messages, temperature, max_tokens, **kwargs)
        elif model_config.provider == Provider.OPENROUTER:
            return self._chat_openrouter(model_config, messages, temperature, max_tokens, **kwargs)
        elif model_config.provider == Provider.DEEPINFRA:
            return self._chat_deepinfra(model_config, messages, temperature, max_tokens, **kwargs)
    
    def _chat_openai(self, model_config: ModelConfig, messages: List[Dict], temperature: float, max_tokens: Optional[int], **kwargs) -> str:
        """Handle OpenAI API calls"""
        client = self._get_openai_client()
        
        params = {
            "model": model_config.api_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
        params.update(kwargs)
        
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    def _chat_openrouter(self, model_config: ModelConfig, messages: List[Dict], temperature: float, max_tokens: Optional[int], **kwargs) -> str:
        """Handle OpenRouter API calls"""
        client = self._get_openrouter_client()
        
        # Build extra headers for OpenRouter
        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name
        
        params = {
            "model": model_config.api_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
        if extra_headers:
            params["extra_headers"] = extra_headers
        params.update(kwargs)
        
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    def _chat_deepinfra(self, model_config: ModelConfig, messages: List[Dict], temperature: float, max_tokens: Optional[int], **kwargs) -> str:
        """Handle DeepInfra API calls"""
        client = self._get_deepinfra_client()
        
        params = {
            "model": model_config.api_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
        params.update(kwargs)
        
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    def stream_chat(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Stream a chat completion request
        
        Args:
            model_name: Name of the model
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Yields:
            Text chunks as they arrive
        """
        model_config = ModelRegistry.get_model(model_name)
        
        # Get appropriate client
        if model_config.provider == Provider.OPENAI:
            client = self._get_openai_client()
        elif model_config.provider == Provider.OPENROUTER:
            client = self._get_openrouter_client()
        elif model_config.provider == Provider.DEEPINFRA:
            client = self._get_deepinfra_client()
        
        # Build parameters
        params = {
            "model": model_config.api_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        # Add OpenRouter headers if needed
        if model_config.provider == Provider.OPENROUTER:
            extra_headers = {}
            if self.site_url:
                extra_headers["HTTP-Referer"] = self.site_url
            if self.site_name:
                extra_headers["X-Title"] = self.site_name
            if extra_headers:
                params["extra_headers"] = extra_headers
        
        params.update(kwargs)
        
        # Stream response
        stream = client.chat.completions.create(**params)
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class DSPyAdapter:
    """
    Adapter for DSPy integration
    Allows using DSPy signatures with the unified LLM client
    """
    
    def __init__(self, llm_client: LLMClient, model_name: str):
        """
        Initialize DSPy adapter
        
        Args:
            llm_client: The unified LLM client instance
            model_name: Model to use for DSPy
        """
        self.llm_client = llm_client
        self.model_name = model_name
    
    def get_dspy_lm(self, max_tokens: int = 8000, temperature: float = 0.1):
        """
        Get a DSPy-compatible LM object
        
        Args:
            max_tokens: Maximum tokens for generation (default 8000 to avoid truncation)
            temperature: Sampling temperature (default 0.1 to prevent repetition loops while staying focused)
        
        Returns:
            A DSPy LM that uses the unified client
        """
        try:
            import dspy
        except ImportError:
            raise ImportError("DSPy not installed. Install with: pip install dspy-ai")
        
        # Get model config to determine provider
        model_config = ModelRegistry.get_model(self.model_name)
        
        # Check if this is a reasoning model (GPT-5 or GPT-OSS)
        is_reasoning_model = "gpt-5" in model_config.api_name.lower() or "gpt-oss" in model_config.api_name.lower()
        
        # Set LM parameters - use higher limits for reasoning models
        if is_reasoning_model:
            lm_params = {
                "temperature": 1.0,
                "max_tokens": 16000
            }
        else:
            # Default params for all other models to prevent truncation and repetition
            lm_params = {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        
        # Create appropriate DSPy LM based on provider
        # Modern DSPy uses dspy.LM() with openai/ prefix for OpenAI-compatible endpoints
        if model_config.provider == Provider.OPENAI:
            return dspy.LM(
                model=f"openai/{model_config.api_name}",
                api_key=self.llm_client.openai_api_key,
                **lm_params
            )
        elif model_config.provider == Provider.OPENROUTER:
            return dspy.LM(
                model=f"openai/{model_config.api_name}",
                api_key=self.llm_client.openrouter_api_key,
                api_base="https://openrouter.ai/api/v1",
                **lm_params
            )
        elif model_config.provider == Provider.DEEPINFRA:
            return dspy.LM(
                model=f"openai/{model_config.api_name}",
                api_key=self.llm_client.deepinfra_api_key,
                api_base="https://api.deepinfra.com/v1/openai",
                **lm_params
            )


# Convenience functions for quick usage
def create_client(**kwargs) -> LLMClient:
    """
    Create a new LLM client
    
    Example:
        client = create_client(openrouter_api_key="your-key")
    """
    return LLMClient(**kwargs)


def list_available_models() -> List[str]:
    """List all available models"""
    return ModelRegistry.list_models()


def quick_chat(model_name: str, prompt: str, **kwargs) -> str:
    """
    Quick one-off chat completion
    
    Args:
        model_name: Name of the model to use
        prompt: The prompt/question
        **kwargs: Additional parameters
        
    Returns:
        Generated text response
        
    Example:
        response = quick_chat("llama-3.3-70b", "Explain quantum computing")
    """
    client = LLMClient()
    messages = [{"role": "user", "content": prompt}]
    return client.chat(model_name, messages, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Easy LLM Importer")
    print("=" * 50)
    print("\nAvailable models:")
    for model in list_available_models():
        config = ModelRegistry.get_model(model)
        print(f"  • {model:30s} → {config.provider.value:12s} ({config.api_name})")
    
    print("\n" + "=" * 50)
    print("\nExample usage:")
    print("""
# Basic usage
from easy_llm_importer import create_client

client = create_client(
    openrouter_api_key="your-key",
    deepinfra_api_key="your-key"
)

# Simple chat
response = client.chat(
    model_name="llama-3.3-70b",
    messages=[
        {"role": "user", "content": "Explain scientific analogies"}
    ]
)

# With streaming
for chunk in client.stream_chat(
    model_name="deepseek-r1",
    messages=[{"role": "user", "content": "Hello!"}]
):
    print(chunk, end="", flush=True)

# Quick one-off
from easy_llm_importer import quick_chat
response = quick_chat("claude-3.5-haiku", "What is AI?")

# DSPy integration
from easy_llm_importer import DSPyAdapter
import dspy

adapter = DSPyAdapter(client, "llama-3.3-70b")
lm = adapter.get_dspy_lm()
dspy.configure(lm=lm)

# Use DSPy signatures
class AnalogySig(dspy.Signature):
    '''Map source concept to target analogy'''
    source = dspy.InputField(desc="Source concept")
    target = dspy.OutputField(desc="Target analogy")

analogy_gen = dspy.Predict(AnalogySig)
result = analogy_gen(source="quantum entanglement")
    """)

