"""Exercise the resolver and real MCP transport with scripted HTTP/model peers."""

import asyncio
from copy import deepcopy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from api.agent import config, deep_research, providers
from api.agent.parallel_client import MCP_URL, USER_AGENT, ParallelSearchClient
from api.agent.search_context import SearchContext
from api.agent.search_utils import concurrent_search, search_and_summarize

HTTP_CLIENT = httpx.AsyncClient


class MCPPeer:
    """Serve JSON-RPC replies to the actual SDK HTTP transport."""

    def __init__(self):
        """Start an independent scripted peer."""
        self.requests = []
        self.calls = []
        self.error = False
        self.delay = False
        self.malformed = False
        self.missing_tool = False
        self.status = 200

    def client(self, **kwargs):
        """Create an actual HTTP client using this deterministic peer."""
        return HTTP_CLIENT(transport=httpx.MockTransport(self.handle), **kwargs)

    async def handle(self, request):
        """Handle SDK requests without replacing any MCP session code."""
        self.requests.append(request)
        if request.method != "POST":
            return httpx.Response(405)
        message = json.loads(request.content)
        method = message["method"]
        if "id" not in message:
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "test-peer", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [] if self.missing_tool else [{
                "name": "web_search", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            args = message["params"]["arguments"]
            self.calls.append(args)
            if self.status != 200:
                return httpx.Response(self.status, json={"error": "permission denied"})
            if self.delay:
                await asyncio.sleep(30)
            query = args["objective"]
            data = {"results": [
                {"title": query, "url": "https://example.org/" + query,
                 "excerpts": ["Evidence for " + query]},
                {"title": "unsafe", "url": "javascript:alert(1)", "excerpts": []}]}
            text = ("quota exceeded" if self.error else
                    "invalid JSON" if self.malformed else json.dumps(data))
            result = {"content": [{"type": "text", "text": text}],
                      "isError": self.error}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})


class ConfigTests(unittest.TestCase):
    """Check fresh defaults and existing user selections in isolated config."""

    def setUp(self):
        """Isolate configuration and credentials."""
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.temp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.temp.name)
        config._config = None

    def tearDown(self):
        """Restore process state."""
        os.chdir(self.cwd)
        self.temp.cleanup()
        self.env.stop()
        config._config = None

    def test_fresh_and_saved_choices(self):
        """Preserve saved settings without mutating subsequent defaults."""
        self.assertIsInstance(providers.get_search_provider(), ParallelSearchClient)
        for selection, class_name in [("serper", "SerperClient"), ("tavily", "TavilyClient"),
                                      ("parallel", "ParallelSearchClient"), ("none", "DisabledSearchClient")]:
            Path("config.yaml").write_text("research:\n  search_source: " + selection)
            config.load_config()
            self.assertEqual(type(providers.get_search_provider()).__name__, class_name)
        self.assertIsInstance(providers.get_search_provider("parallel"), ParallelSearchClient)
        Path("config.yaml").unlink()
        self.assertIsInstance(providers.get_search_provider(), providers.DisabledSearchClient)
        config.load_config()
        self.assertIsInstance(providers.get_search_provider(), ParallelSearchClient)

    def test_existing_credentials(self):
        """Keep intentionally configured search credentials in use."""
        for key, class_name in [("SERPER_KEY", "SerperClient"), ("SERPER_API_KEY", "SerperClient"),
                                ("TAVILY_KEY", "TavilyClient")]:
            with patch.dict(os.environ, {key: "configured"}):
                config.load_config()
                self.assertEqual(type(providers.get_search_provider()).__name__, class_name)
                self.assertIsInstance(providers.get_search_provider("parallel"), ParallelSearchClient)
            config.load_config()
            self.assertIsInstance(providers.get_search_provider(), ParallelSearchClient)
        Path("config.yaml").write_text("serper:\n  api_key: saved-key\n")
        config.load_config()
        self.assertEqual(type(providers.get_search_provider()).__name__, "SerperClient")


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    """Keep resolver, SDK and result consumption in the executed test path."""

    def setUp(self):
        """Isolate configuration and credentials."""
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        config._config = None
        self.peer = MCPPeer()
        self.http = patch("api.agent.parallel_client.httpx.AsyncClient", self.peer.client)
        self.http.start()
        self.config_patch = patch.object(config, "_config", deepcopy(config.DEFAULT_CONFIG))
        self.config_patch.start()

    def tearDown(self):
        """Restore process state."""
        self.config_patch.stop()
        config._config = None
        self.http.stop()
        self.env.stop()

    async def test_resolver_transport_and_concurrent_result_consumption(self):
        """Consume sources through the resolver and real MCP client."""
        context = SearchContext()
        provider = providers.get_search_provider(context=context)
        results = await concurrent_search(["first", "second"], search_provider=provider)
        for query, result in zip(["first", "second"], results):
            self.assertEqual(result["urls"], ["https://example.org/" + query])
            self.assertIn("Evidence for " + query, result["summary"])
        await search_and_summarize("followup", search_provider=provider)
        self.assertEqual({c["session_id"] for c in self.peer.calls}, {context.session_id})
        for request in self.peer.requests:
            self.assertEqual(str(request.url), MCP_URL)
            self.assertEqual(request.headers["User-Agent"], USER_AGENT)
            self.assertNotIn("Authorization", request.headers)
        methods = [json.loads(r.content)["method"] for r in self.peer.requests if r.method == "POST"]
        self.assertIn("tools/list", methods)
        self.assertIn("tools/call", methods)

    async def test_errors_retry_timeout_cancellation_and_disabled(self):
        """Retain identity and avoid fallback during failures and disabling."""
        provider = providers.get_search_provider()
        self.peer.error = True
        result = await search_and_summarize("error", search_provider=provider)
        self.assertIn("quota exceeded", result["summary"])
        self.assertEqual(result["urls"], [])
        self.peer.error = False
        await search_and_summarize("retry", search_provider=provider)
        self.assertEqual(self.peer.calls[0]["session_id"], self.peer.calls[1]["session_id"])
        self.peer.malformed = True
        self.assertIn("Error searching", (await search_and_summarize("bad", search_provider=provider))["summary"])
        self.peer.malformed = False
        self.peer.missing_tool = True
        self.assertIn("Error searching", (await search_and_summarize("missing", search_provider=provider))["summary"])
        self.peer.missing_tool = False
        self.peer.delay = True
        provider.timeout = 0.03
        with self.assertRaises(TimeoutError):
            await provider.search("slow")
        provider.timeout = 10
        task = asyncio.create_task(search_and_summarize("cancel", search_provider=provider))
        while not any(c["objective"] == "cancel" for c in self.peer.calls):
            await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        count = len(self.peer.requests)
        self.assertEqual((await search_and_summarize("disabled", search_provider=providers.get_search_provider("none")))["urls"], [])
        self.assertEqual(len(self.peer.requests), count)

    async def test_permission_and_transport_errors_do_not_change_provider(self):
        """Surface denied or unavailable search without choosing another service."""
        provider = providers.get_search_provider()
        for status in [403, 503]:
            self.peer.status = status
            result = await search_and_summarize("unavailable", search_provider=provider)
            self.assertIn("Error searching", result["summary"])
            self.assertEqual(result["urls"], [])
        self.peer.status = 200
        result = await search_and_summarize("recovered", search_provider=provider)
        self.assertIn("Evidence for recovered", result["summary"])
        self.assertEqual(len({call["session_id"] for call in self.peer.calls}), 1)
        count = len(self.peer.requests)
        # An explicit failing incumbent must not cause a Parallel request.
        with patch("api.agent.serper_client.SerperClient.search", new=AsyncMock(side_effect=RuntimeError("selected failed"))):
            result = await search_and_summarize("incumbent", search_provider=providers.get_search_provider("serper"))
        self.assertIn("selected failed", result["summary"])
        self.assertEqual(len(self.peer.requests), count)

    async def test_research_entrypoints_own_interleaved_run_identity(self):
        """Scope identity to each real research entrypoint invocation."""
        prompts = []

        async def plan(query, history_context):
            return {"steps": [{"search_query": query}]}

        async def completion(prompt, **kwargs):
            prompts.append(prompt)
            return "scripted summary"

        async def json_completion(prompt, **kwargs):
            return {"nextQueries": ["followup"], "learnings": ["learned evidence"], "findings": []}

        async def run(query):
            return await deep_research.deep_research(query, depth=2, breadth=1)

        with patch.object(deep_research, "generate_research_plan", plan), \
                patch.object(deep_research, "generate_completion", completion), \
                patch.object(deep_research, "generate_json_completion", json_completion), \
                patch.object(deep_research, "should_clarify_query", new=AsyncMock(return_value=False)):
            first, second = await asyncio.gather(run("alpha"), run("beta"))
            third = await run("alpha")  # New research/reset, even with the same text.
            streamed = [event async for event in deep_research.deep_research_stream("stream", depth=2, breadth=1)]
        for result in [first, second, third, streamed[-1]]:
            self.assertIn("learned evidence", result["learnings"])
            self.assertTrue(result["visitedUrls"])
        self.assertEqual(streamed[-1]["stage"], "completed")
        self.assertTrue(any("Evidence for alpha" in prompt for prompt in prompts))
        calls = self.peer.calls
        ids = {c["session_id"] for c in calls}
        self.assertEqual(len(ids), 4)
        self.assertTrue(all(sum(c["session_id"] == identity for c in calls) == 2 for identity in ids))
        alpha_ids = [c["session_id"] for c in calls if c["objective"] == "alpha"]
        self.assertEqual(len(set(alpha_ids)), 2)


if __name__ == "__main__":
    unittest.main()
