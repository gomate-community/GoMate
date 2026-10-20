# DeepResearch web search

The Python research entrypoints in this directory support free web search through
[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp).
Anonymous search uses Fast mode and needs no search account or API key.

From the repository root, install the research dependencies and configure your LLM:

```bash
python -m pip install -r api/agent/requirements.txt
export OPENAI_KEY="your-model-api-key"
```

Run a research task from the repository root:

```python
import asyncio
from api.agent.deep_research import deep_research

result = asyncio.run(deep_research(
    "What are the latest developments in retrieval augmented generation?",
    depth=1,
    breadth=1,
))
print(result["learnings"])
print(result["visitedUrls"])
```

`research.search_source` defaults to `auto`. It uses configured Serper credentials
(`SERPER_API_KEY`, `SERPER_KEY`, or `serper.api_key`), then Tavily credentials
(`TAVILY_KEY` or `tavily.api_key`), and otherwise Parallel. Existing `config.yaml`
provider selections and each call's `search_source` override remain in effect.
For example, `deep_research(..., search_source="parallel")` explicitly selects
Parallel even if another provider has credentials. Selected providers never fall
back to another provider after an error.

Copy `config.example.yaml` to `config.yaml` in your working directory to configure
these choices. Available providers are `parallel`, `serper`, `tavily`, `mp_search`
and `qdrant`; `none` disables external search. The Gradio provider dropdown also
exposes these choices. Local/vector providers keep their existing setup needs.
These settings apply to `api.agent`, independently of the CLI under
`trustrag/modules/deepsearch`.

Each invocation of `deep_research` or `deep_research_stream` creates an opaque
research identity, reused for its follow-up searches and isolated from concurrent
runs. Parallel requests have a 30-second timeout and propagate cancellation.
Search failures appear in the existing search error result with no source URLs.
Anonymous service limits apply; a quota error does not trigger a provider switch.
Only web search is integrated here, with excerpts and URLs passed into research
summaries. This provider does not add a page-reading action.
