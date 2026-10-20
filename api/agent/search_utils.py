"""
Utility functions for handling search operations, including concurrent searches.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from loguru import logger

from api.agent.config import get_config
from api.agent.providers import get_search_provider


async def search_and_summarize(
        query: str,
        config: Dict[str, Any] = None,
        search_provider=None
) -> Dict[str, Any]:
    """
    Search and summarize results for a query.
    
    Args:
        query: The query to search for
        config: Configuration dictionary
        search_provider: Search provider instance
        
    Returns:
        Dict with summary and URLs
    """
    if config is None:
        config = get_config()

    if search_provider is None:
        search_provider = get_search_provider()

    try:
        # Perform the search
        results = await search_provider.search(query)

        # For web search, ensure URLs are captured
        if hasattr(search_provider, 'get_organic_urls'):
            urls = search_provider.get_organic_urls()
        else:
            # Derive URLs from this response instead of mutable provider state,
            # so concurrent searches retain their respective sources.
            items = results.get("data", []) if isinstance(results, dict) else results
            urls = [item["url"] for item in items
                    if isinstance(item, dict) and item.get("url")] if isinstance(items, list) else []

        # Format the search results
        formatted_results = []
        # Default to 5 results if not specified in config
        max_results = config.get("research", {}).get("max_results_per_query", 5)

        # Ensure results is a list before slicing
        if isinstance(results, list):
            results_to_process = results[:max_results]
        elif isinstance(results, dict) and "data" in results and isinstance(results["data"], list):
            # Some search providers return a dict with a "data" field containing the results
            results_to_process = results["data"][:max_results]
        else:
            # If results is not a list or a dict with a "data" field, use an empty list
            results_to_process = []
            logger.warning(f"Unexpected search results format: {type(results)}")

        for idx, result in enumerate(results_to_process):
            formatted_results.append(f"[{idx + 1}] {result}")

        search_results_text = "\n\n".join(formatted_results)

        # Check if we have results to summarize
        if not formatted_results:
            return {
                "summary": f"No search results found for query: {query}",
                "urls": []
            }

        # Return the formatted results and URLs
        return {
            "summary": search_results_text,
            "urls": urls,
            "raw_results": results_to_process
        }

    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return {
            "summary": f"Error searching for '{query}': {str(e)}",
            "urls": [],
            "raw_results": []
        }


async def concurrent_search(
        queries: List[str],
        config: Dict[str, Any] = None,
        search_provider=None
) -> List[Dict[str, Any]]:
    """
    Perform concurrent searches for multiple queries.
    
    Args:
        queries: List of queries to search for
        config: Configuration dictionary
        search_provider: Search provider instance
        
    Returns:
        List of search results
    """
    if config is None:
        config = get_config()

    if search_provider is None:
        search_provider = get_search_provider()

    # Get the concurrency limit from config
    concurrency_limit = config.get("research", {}).get("concurrency_limit", 1)

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def search_with_semaphore(query: str) -> Dict[str, Any]:
        """Perform a search with semaphore-based concurrency control."""
        async with semaphore:
            return await search_and_summarize(query, config, search_provider)

    # Create tasks for all queries
    tasks = [search_with_semaphore(query) for query in queries]

    # Run all tasks concurrently and gather results
    results = await asyncio.gather(*tasks)

    return results
