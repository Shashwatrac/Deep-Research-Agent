# from typing import Dict, Generator, List
# from .llm import OllamaClient

# SYSTEM = """You are a research assistant. Answer questions using only the web sources provided.

# Rules:
# - Cite every factual claim like this: [Page Title - domain.com](https://full-url)
# - If two sources disagree, say so and cite both sides
# - If the context doesn't cover something, admit it clearly - don't guess
# - End with a ## Sources section listing every URL you cited"""


# def _trim_history(history: List[Dict], max_turns: int = 6) -> str:
#     """Pull recent turns into a short string for context injection."""
#     recent = history[-(max_turns * 2):]
#     lines = []
#     for msg in recent:
#         role = "User" if msg["role"] == "user" else "Assistant"
#         content = msg["content"]
#         if len(content) > 600:
#             content = content[:600] + "..."
#         lines.append(f"{role}: {content}")
#     return "\n".join(lines)


# def generate_answer(
#     query: str,
#     context: str,
#     citations: List[Dict],
#     history: List[Dict],
#     llm: OllamaClient,
# ) -> Generator[str, None, None]:
#     """Streams the final answer. Caller collects the chunks."""

#     history_str = _trim_history(history) if history else "None"

#     source_list = "\n".join(
#         f"  [{c['index']}] {c['title']} ({c['domain']}) - {c['url']}"
#         for c in citations
#     )

#     user_msg = f"""Prior conversation:
# {history_str}

# Question: {query}

# Sources available:
# {source_list}

# Web context:
# {context}

# Answer the question using only the context above. Cite sources inline."""

#     yield from llm.stream([{"role": "user", "content": user_msg}], system=SYSTEM)
from typing import Dict, Generator, List

from .llm import OllamaClient


SYSTEM = """
You are a deep research assistant.

Your job:
- Answer using ONLY the provided web context
- Never invent facts
- Keep answers concise but informative
- Use short factual sentences
- Every important factual sentence MUST contain an inline markdown citation
- Citations MUST use this exact format:
  [domain.com](https://full-url)

Rules:
- If sources disagree, explicitly mention the disagreement
- If evidence is weak or unavailable, clearly acknowledge uncertainty
- Prefer grounded factual summaries over speculation
- End with a ## Sources section
"""


def _trim_history(
    history: List[Dict],
    max_turns: int = 4,
) -> str:
    """
    Compress prior conversation history.
    """

    recent = history[-(max_turns * 2):]

    lines = []

    for msg in recent:

        role = "User" if msg["role"] == "user" else "Assistant"

        content = msg["content"]

        if len(content) > 400:
            content = content[:400] + "..."

        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _build_source_section(citations: List[Dict]) -> str:
    """
    Build explicit source list.
    """

    lines = []

    for c in citations:

        lines.append(
            f"- [{c['domain']}]({c['url']})"
        )

    return "\n".join(lines)


def generate_answer(
    query: str,
    context: str,
    citations: List[Dict],
    history: List[Dict],
    llm: OllamaClient,
) -> Generator[str, None, None]:
    """
    Stream final grounded answer.
    """

    history_str = _trim_history(history) if history else "None"

    # ─────────────────────────────────────────────
    # SOURCE REFERENCE BLOCK
    # ─────────────────────────────────────────────

    source_list = "\n".join(
        f"[{c['domain']}]({c['url']})"
        for c in citations
    )

    # ─────────────────────────────────────────────
    # IMPORTANT:
    # evaluator requires markdown links INLINE
    # ─────────────────────────────────────────────

    user_msg = f"""
Prior conversation:
{history_str}

Question:
{query}

Available sources:
{source_list}

Web research context:
{context}

Instructions:
- Write a research-quality answer
- Use concise factual sentences
- EVERY major factual sentence must include an inline markdown citation
- Citation format MUST be:
  [domain.com](https://full-url)
- Do not put all citations only at the bottom
- Include citations throughout the answer
- Mention uncertainty if evidence is weak
- Mention disagreement if sources conflict
- Use markdown headers

Finish with:

## Sources

followed by all cited links.
"""

    # ─────────────────────────────────────────────
    # STREAM RESPONSE
    # ─────────────────────────────────────────────

    yield from llm.stream(
        [{"role": "user", "content": user_msg}],
        system=SYSTEM,
    )