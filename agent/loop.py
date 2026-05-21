import logging
import concurrent.futures
from typing import Dict, Generator, List

from .llm import OllamaClient
from .answerer import generate_answer
from .context import build_context
from .fetcher import fetch_page, FetchedPage
from .planner import plan_research
from .search import SearchResult, get_searcher
from .session import get_conversation_history, save_message, save_turn

logger = logging.getLogger(__name__)


def run_agent(
    query: str,
    session_id: str,
    tavily_key: str = "",
    serper_key: str = "",
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "llama3.2",
    max_search_results: int = 5,
    max_search_queries: int = 3,
    max_fetch_urls: int = 3,
    max_context_chars: int = 7000,
    db_path: str = "sessions.db",
) -> Generator[Dict, None, None]:
    """
    Core agent loop - no framework, just Python.
    
    Yields event dicts for the UI to consume:
      {"type": "progress",     "step": str, "message": str}
      {"type": "answer_chunk", "content": str}
      {"type": "done",         "data": {...}}
      {"type": "error",        "message": str}
    """
    llm = OllamaClient(base_url=ollama_base_url, model=ollama_model)
    history = get_conversation_history(session_id, db_path=db_path)

    # ── 1. PLAN ───────────────────────────────────────────────────────────────
    yield _p("planning", "🧠 Planning research strategy...")

    try:
        plan = plan_research(query, llm)
    except Exception as e:
        yield _p("planning_error", f"Planner failed ({e}), using raw query")
        plan = {"plan": "Direct search.", "search_queries": [query], "key_aspects": []}

    queries: List[str] = plan.get("search_queries", [query])[:max_search_queries]
    plan_text = plan.get("plan", "")

    yield _p("planning_done", f"📋 {plan_text}", data={"queries": queries})

    # ── 2. SEARCH ─────────────────────────────────────────────────────────────
    yield _p("searching", f"🌐 Running {len(queries)} search quer{'y' if len(queries)==1 else 'ies'}...")

    try:
        searcher = get_searcher(tavily_key, serper_key)
    except ValueError as e:
        yield {"type": "error", "message": str(e)}
        return

    all_results: List[SearchResult] = []
    per_q = max(2, max_search_results // len(queries))

    for q in queries:
        try:
            results = searcher.search(q, max_results=per_q)
            all_results.extend(results)
            yield _p("search_query", f'  ↳ "{q}" → {len(results)} results')
        except Exception as e:
            yield _p("search_error", f'  ↳ failed: "{q}" — {e}')

    # dedup by URL, keep highest score per URL
    seen: Dict[str, SearchResult] = {}
    for r in all_results:
        if r.url not in seen or r.score > seen[r.url].score:
            seen[r.url] = r

    unique = sorted(seen.values(), key=lambda x: x.score, reverse=True)
    yield _p("search_done", f"✅ {len(unique)} unique sources",
             data={"results": [{"title": r.title, "url": r.url} for r in unique[:10]]})

    # ── 3. FETCH PAGES ────────────────────────────────────────────────────────
    # to_fetch = [r.url for r in unique[:max_fetch_urls]]
    to_fetch = [
        r.url
        for r in unique
        if r.score >= 0.6
    ][:max_fetch_urls]
    yield _p("fetching", f"📄 Fetching {len(to_fetch)} pages...")

    fetched: List[FetchedPage] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch_page, url): url for url in to_fetch}
        for fut in concurrent.futures.as_completed(futures):
            url = futures[fut]
            try:
                page = fut.result()
                if page:
                    fetched.append(page)
                    yield _p("page_fetched", f"  ↳ ✓ {page.domain}")
                else:
                    yield _p("page_skip", f"  ↳ ✗ {_short(url)}")
            except Exception as e:
                yield _p("page_error", f"  ↳ ✗ {_short(url)}: {e}")

    # ── 4. BUILD CONTEXT ──────────────────────────────────────────────────────
    yield _p("context", "🧩 Selecting relevant context...")

    context, citations = build_context(
        query=query,
        search_results=unique,
        fetched_pages=fetched,
        max_chars=max_context_chars,
    )

    yield _p("context_done",
             f"📚 {len(citations)} sources selected, {len(context):,} chars",
             data={"citations": citations})

    # ── 5. ANSWER ─────────────────────────────────────────────────────────────
    yield _p("answering", f"✍️  Generating answer with {llm.model}...")

    full_answer = ""
    try:
        for chunk in generate_answer(
            query=query,
            context=context,
            citations=citations,
            history=history,
            llm=llm,
        ):
            full_answer += chunk
            yield {"type": "answer_chunk", "content": chunk}
    except Exception as e:
        yield {"type": "error", "message": f"Answer generation failed: {e}"}
        return

    # ── SAVE ──────────────────────────────────────────────────────────────────
    save_message(session_id, "user",      query,        db_path=db_path)
    save_message(session_id, "assistant", full_answer,  db_path=db_path)
    save_turn(
        session_id=session_id,
        query=query,
        search_queries=queries,
        urls_opened=[r.url for r in unique[:10]],
        # context_snippets=[r.snippet for r in unique[:5]],
        context_snippets=[
            r.snippet[:300]
            for r in unique[:3]
        ],
        final_answer=full_answer,
        db_path=db_path,
    )

    yield {
        "type": "done",
        "data": {
            "answer":         full_answer,
            "citations":      citations,
            "sources_found":  len(unique),
            "pages_fetched":  len(fetched),
            "search_queries": queries,
        },
    }


def _p(step: str, message: str, data: dict = None) -> Dict:
    e = {"type": "progress", "step": step, "message": message}
    if data:
        e["data"] = data
    return e


def _short(url: str, n: int = 55) -> str:
    return (url[:n] + "...") if len(url) > n else url
