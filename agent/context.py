import re
from typing import Dict, List, Tuple
from .search import SearchResult
from .fetcher import FetchedPage


def _kw_overlap(text: str, query: str) -> float:
    """Quick keyword overlap score - not fancy but good enough for ranking"""
    qtoks = set(re.findall(r"\w+", query.lower()))
    ttoks = set(re.findall(r"\w+", text.lower()))
    if not qtoks:
        return 0.0
    return len(qtoks & ttoks) / len(qtoks)


def build_context(
    query: str,
    search_results: List[SearchResult],
    fetched_pages: List[FetchedPage],
    max_chars: int = 80000,
    snippet_chars: int = 3000,
) -> Tuple[str, List[Dict]]:
    """
    Puts together the context block we send to the LLM.
    
    Strategy:
    1. Score each search result by combining the API score with keyword overlap
    2. Add results greedily until we hit the char budget
    3. Domain-diversity cap so one site can't eat the whole context
    4. For fetched full pages, extract the most query-relevant paragraphs
       (often the top of the page is nav/boilerplate, actual info is buried)
    
    Returns: (context_string, list of citation dicts)
    """
    parts = []
    citations = []
    domain_counts: Dict[str, int] = {}
    budget = max_chars
    idx = 1

    # --- search snippets first ---
    scored = []
    for r in search_results:
        kw = _kw_overlap(r.title + " " + r.snippet, query)
        # blend the api relevance score with keyword match
        combined = r.score * 0.5 + kw * 0.5
        scored.append((combined, r))
    scored.sort(reverse=True, key=lambda x: x[0])

    for score, result in scored:
        if budget <= 200:
            break
        # don't let one domain dominate - max 3 snippets per domain
        if domain_counts.get(result.domain, 0) >= 3:
            continue

        block = (
            f"[{idx}] {result.title}\n"
            f"URL: {result.url}\n"
            f"{result.snippet}\n"
        )
        if len(block) > budget:
            block = block[:budget]

        parts.append(block)
        citations.append({
            "index": idx,
            "title": result.title,
            "url": result.url,
            "domain": result.domain,
        })
        budget -= len(block)
        domain_counts[result.domain] = domain_counts.get(result.domain, 0) + 1
        idx += 1

    # --- full page content ---
    # For each fetched page, pull the paragraphs most relevant to the query
    # rather than just slicing from the top (which is often nav / cookie banners)
    for page in fetched_pages:
        if budget < 500:
            break
        if domain_counts.get(page.domain, 0) >= 2:
            continue

        paras = [p.strip() for p in page.text.split("\n\n") if len(p.strip()) > 80]
        if paras:
            # sort paragraphs by query relevance, take top 4
            ranked = sorted(paras, key=lambda p: _kw_overlap(p, query), reverse=True)
            best = "\n\n".join(ranked[:4])
        else:
            best = page.text

        alloc = min(snippet_chars, budget - 200)
        best = best[:alloc]

        block = (
            f"\n[Full page - {page.domain}]\n"
            f"Title: {page.title}\n"
            f"URL: {page.url}\n"
            f"Fetched: {page.retrieved_at}\n"
            f"{best}\n"
        )
        parts.append(block)
        budget -= len(block)
        domain_counts[page.domain] = domain_counts.get(page.domain, 0) + 1

    context = "\n---\n".join(parts)
    return context, citations
