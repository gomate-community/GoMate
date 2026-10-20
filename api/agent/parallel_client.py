"""Anonymous Parallel Search MCP provider for DeepResearch."""

import asyncio
import json
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Implementation

from .search_context import SearchContext

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "TrustRAG/0.15.0 (https://github.com/gomate-community/TrustRAG)"


class ParallelSearchClient:
    """Search without credentials using the supported MCP HTTP transport."""

    def __init__(self, context: SearchContext, timeout: float = 30.0):
        """Use the caller's run identity and a bounded request deadline."""
        self.context = context
        self.timeout = timeout

    async def search(self, query: str):
        """Return excerpts and source URLs in the research result format."""
        # The caller owns context; connections can change without changing the
        # research identity. Each call closes its transport, including on cancel.
        async with asyncio.timeout(self.timeout):
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT}, timeout=self.timeout
            ) as http:
                async with streamable_http_client(MCP_URL, http_client=http) as streams:
                    async with ClientSession(
                        streams[0], streams[1],
                        client_info=Implementation(name="TrustRAG", version="0.15.0")
                    ) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        if not any(tool.name == "web_search" for tool in tools.tools):
                            raise RuntimeError("Parallel MCP does not expose web_search")
                        result = await session.call_tool("web_search", {
                            "objective": query,
                            "search_queries": [query],
                            "session_id": self.context.session_id,
                        })

        text = "\n".join(
            block.text for block in result.content if block.type == "text"
        )
        if result.isError:
            raise RuntimeError(f"Parallel search failed: {text}")
        payload = result.structuredContent
        if payload is None:
            payload = json.loads(text)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("Unexpected Parallel search response")

        results = []
        for item in payload["results"]:
            url = item.get("url", "")
            if urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).netloc:
                continue
            results.append({
                "title": item.get("title", ""),
                "url": url,
                "content": "\n\n".join(item.get("excerpts", [])),
                "source": "parallel",
            })
        return results
