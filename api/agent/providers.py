import tiktoken
from openai import OpenAI
import json
from api.agent.config import get_config
from api.agent.search_context import SearchContext
from api.agent.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
import os
import sys
from typing import Dict, Any, Optional
import openai


# Initialize clients based on configuration
def _get_openai_client():
    """Get OpenAI client if configured"""
    config = get_config()
    if config["openai"]["api_key"]:
        return OpenAI(
            api_key=config["openai"]["api_key"],
            base_url=config["openai"]["base_url"]
        )
    return None


# Get clients
openai_client = _get_openai_client()

# Constants
MIN_CHUNK_SIZE = 140
# encoder = tiktoken.get_encoding("cl100k_base")  # Using cl100k as a replacement for o200k_base

logger.info(f"OpenAI client: {openai_client}, model: {get_config()['openai']['model']}")


def get_model() -> Dict[str, Any]:
    """
    Get model configuration including client and model name.
    
    Returns:
        Dict containing model configuration
    """
    config = get_config()
    
    # Get OpenAI configuration from config
    openai_config = config.get("openai", {})
    api_key = openai_config.get("api_key", "")
    model = openai_config.get("model", "gpt-4o")
    base_url = openai_config.get("base_url", None)
    
    # Initialize OpenAI client
    client_args = {"api_key": api_key}
    if base_url:
        client_args["base_url"] = base_url
    
    client = openai.OpenAI(**client_args)
    async_client = openai.AsyncOpenAI(**client_args)
    
    return {
        "client": client,
        "async_client": async_client,
        "model": model
    }


def trim_prompt(prompt, context_size=None):
    """Trim prompt to maximum context size"""
    if context_size is None:
        context_size = get_config()["research"]["context_size"]

    if not prompt:
        return ""

    length = len(prompt.slit())
    # length = len(encoder.encode(prompt))
    if length <= context_size:
        return prompt

    overflow_tokens = length - context_size
    # Estimate characters to remove (3 chars per token on average)
    chunk_size = len(prompt) - overflow_tokens * 3

    if chunk_size < MIN_CHUNK_SIZE:
        return prompt[:MIN_CHUNK_SIZE]

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    trimmed_prompt = splitter.split_text(prompt)[0] or ""

    # Handle case where trimmed prompt is same length as original
    if len(trimmed_prompt) == len(prompt):
        return trim_prompt(prompt[:chunk_size], context_size)

    # Recursively trim until within context size
    return trim_prompt(trimmed_prompt, context_size)


def generate_object(model_config, system, prompt, schema):
    """Generate structured output from the model"""
    client = model_config["client"]
    model_name = model_config["model"]
    if schema:
        prompt = f"{prompt}\n\njson schema: \n{schema}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.7
    )

    # Extract and return the JSON content
    content = response.choices[0].message.content
    logger.debug(f"Generated content: {content}")
    # In a real implementation, we would validate against the schema here
    # For simplicity, we're just returning the parsed JSON
    ans = json.loads(content)
    return ans


def get_search_provider(search_source=None, context=None):
    """
    Get the appropriate search provider based on configuration.

    Returns:
        An instance of the search provider class
    """
    config = get_config()
    if search_source is None:
        search_source = config.get("research", {}).get("search_source", "auto")

    if search_source == "auto":
        # Keep intentionally configured incumbent credentials in use. Explicit
        # selections never fall back, even when their credentials are missing.
        if os.getenv("SERPER_API_KEY") or config.get("serper", {}).get("api_key"):
            search_source = "serper"
        elif config.get("tavily", {}).get("api_key"):
            search_source = "tavily"
        else:
            search_source = "parallel"

    if search_source == "none":
        return DisabledSearchClient()
    if search_source == "parallel":
        from .parallel_client import ParallelSearchClient
        # Direct resolver callers start a standalone research run. The main
        # research entrypoints supply their own context for all follow-up calls.
        return ParallelSearchClient(context=context or SearchContext())

    if search_source == "mp_search":
        from .mp_search_client import MPSearchClient
        return MPSearchClient()
    elif search_source == "tavily":
        from .tavily_client import TavilyClient
        return TavilyClient()
    elif search_source == "qdrant":
        from .qd_client import QdrantSearchClient
        return QdrantSearchClient()
    else:  # Default to serper
        from .serper_client import SerperClient
        return SerperClient()


class DisabledSearchClient:
    """Allow research callers to explicitly disable external search."""

    async def search(self, query):
        """Return no sources without contacting a search service."""
        return []
