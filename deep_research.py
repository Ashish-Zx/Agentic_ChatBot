"""
deep_research.py
================
Deep Research module for the Agentic ChatBot.

3-agent pipeline:
  1. Planner Agent – turns a query into HOW_MANY_SEARCHES targeted search terms
  2. Search Agent  – wraps Tavily as a @function_tool and summarises each result
  3. Writer Agent  – synthesises the summaries into a full markdown report

Results are returned directly to the caller (displayed in chat).

Secrets read from .env: HOW_MANY_SEARCHES, TAVILY_API_KEY, INCEPTION_API_KEY.

Public async entrypoint:
    report: ReportData = await run_deep_research(query: str)
"""

from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel
from agents.model_settings import ModelSettings
from openai import AsyncOpenAI
from tavily import TavilyClient

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HOW_MANY_SEARCHES: int = int(os.getenv("HOW_MANY_SEARCHES", "5"))

# ---------------------------------------------------------------------------
# Model setup (Inception Labs mercury-3, free)
# ---------------------------------------------------------------------------
_inception_client = AsyncOpenAI(
    base_url="https://api.inceptionlabs.ai/v1",
    api_key=os.getenv("INCEPTION_API_KEY", ""),
)
_inception_model = OpenAIChatCompletionsModel(
    model="mercury-3",
    openai_client=_inception_client,
)

# ---------------------------------------------------------------------------
# Tavily search client (free tier)
# ---------------------------------------------------------------------------
_tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))


# ---------------------------------------------------------------------------
# Structured output types
# ---------------------------------------------------------------------------
class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(
        description="A list of web searches to perform to best answer the query."
    )


class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final report in markdown format.")
    follow_up_questions: list[str] = Field(description="Suggested topics to research further.")


# ---------------------------------------------------------------------------
# Agent 1: Search Agent (Tavily as function_tool, free replacement for WebSearchTool)
# ---------------------------------------------------------------------------
@function_tool
def web_search(query: str) -> str:
    """
    Search the web for the given query and return a summary of the top results.

    Args:
        query: The search term to look up on the web.
    """
    response = _tavily_client.search(
        query=query,
        max_results=5,
        search_depth="basic",
    )
    results_text = []
    for r in response.get("results", []):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        results_text.append(f"**{title}**\nURL: {url}\n{content}")
    return "\n\n---\n\n".join(results_text) if results_text else "No results found."


SEARCH_INSTRUCTIONS = """\
You are a research assistant. Given a search term, you search the web for that
term using your web_search tool and produce a concise summary of the results.
The summary must be 2-3 paragraphs and less than 300 words.
Capture the main points and be succinct. Reply only with the summary.
"""

search_agent = Agent(
    name="Search Agent",
    instructions=SEARCH_INSTRUCTIONS,
    tools=[web_search],
    model=_inception_model,
    model_settings=ModelSettings(tool_choice="required"),
)

# ---------------------------------------------------------------------------
# Agent 2: Planner Agent
# ---------------------------------------------------------------------------
PLANNER_INSTRUCTIONS = f"""\
You are a research assistant. Given a user query, come up with a set of web
searches to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms
to query for.
"""

planner_agent = Agent(
    name="Planner Agent",
    instructions=PLANNER_INSTRUCTIONS,
    model=_inception_model,
    output_type=WebSearchPlan,
)

# ---------------------------------------------------------------------------
# Agent 3: Writer Agent
# ---------------------------------------------------------------------------
WRITER_INSTRUCTIONS = """\
You are a senior researcher tasked with writing a cohesive report for a research
query. You will be provided with the original query and some research.
Generate a comprehensive report based on the research and the query.
The final output should be in markdown format, and it should be lengthy and
detailed. Aim for 5-10 pages of content, at least 1000 words.
"""

writer_agent = Agent(
    name="Writer Agent",
    instructions=WRITER_INSTRUCTIONS,
    model=_inception_model,
    output_type=ReportData,
)

# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------
async def _search(item: WebSearchItem) -> str:
    """Run the search agent for one search item."""
    input_message = f"Search term: {item.query}\nReason for searching: {item.reason}"
    result = await Runner.run(search_agent, input_message)
    return result.final_output


async def _run_searches(query: str) -> list[str]:
    """Plan and execute all searches concurrently."""
    print("[DeepResearch] Planning searches…")
    result = await Runner.run(planner_agent, f"Query: {query}")
    searches = result.final_output.searches
    print(f"[DeepResearch] Will perform {len(searches)} searches…")
    tasks = [_search(item) for item in searches]
    results = await asyncio.gather(*tasks)
    print("[DeepResearch] Finished searching.")
    return list(results)


async def _write_report(query: str, search_results: list[str]) -> ReportData:
    """Synthesise search summaries into a full markdown report."""
    print("[DeepResearch] Writing report…")
    input_message = (
        f"Original query: {query}\nSummarized search results: {search_results}"
    )
    result = await Runner.run(writer_agent, input_message)
    print("[DeepResearch] Report written.")
    return result.final_output



# ---------------------------------------------------------------------------
# Public async entrypoint
# ---------------------------------------------------------------------------
async def run_deep_research(query: str) -> ReportData:
    """
    Run the 3-agent deep-research pipeline for *query* and return a ReportData.

    Steps:
        1. Planner generates HOW_MANY_SEARCHES search terms.
        2. Search agent runs all searches concurrently via Tavily.
        3. Writer agent synthesises a structured markdown report.

    The report is returned to the caller and displayed directly in chat.
    """
    search_results = await _run_searches(query)
    report = await _write_report(query, search_results)
    return report

