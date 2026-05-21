import json
import re
import logging
from typing import Dict
from .llm import OllamaClient

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a research planning assistant. "
    "You output only valid JSON with no explanation, no markdown fences."
)


def plan_research(query: str, llm: OllamaClient, context_summary: str = "") -> Dict:
    """
    Before we search, ask the LLM to think about what to search for.
    
    This makes a big difference for multi-hop questions. For example
    "what caused the 2023 SVB collapse" benefits from queries like
    "Silicon Valley Bank failure causes", "SVB bond portfolio interest rates",
    "bank run 2023 Twitter" rather than dumping the whole question in.
    """
    prior = f"\nConversation so far: {context_summary}" if context_summary else ""

    prompt = f"""Question: {query}{prior}

Respond ONLY with this JSON (no markdown, no extra text):
{{
  "plan": "your 1-2 sentence research approach",
  "search_queries": ["query1", "query2", "query3"],
  "key_aspects": ["aspect1", "aspect2"]
}}

For search_queries:
- 2 to 4 queries max
- Different angles / phrasings - don't repeat words across queries
- Specific and targeted, not generic
- Think like a journalist, not a search bar"""

    try:
        raw = llm.chat([{"role": "user", "content": prompt}], system=SYSTEM)

        # strip markdown fences if the model ignored our instructions
        raw = raw.strip()
        raw = re.sub(r"^```(json)?", "", raw).rstrip("`").strip()

        # sometimes models add trailing commas - attempt a lenient parse
        return json.loads(raw)

    except json.JSONDecodeError as e:
        logger.warning("Planner returned bad JSON (%s), using raw query fallback", e)
    except Exception as e:
        logger.warning("Planner LLM call failed: %s", e)

    return {
        "plan": "Direct search on the query.",
        "search_queries": [query],
        "key_aspects": [],
    }
