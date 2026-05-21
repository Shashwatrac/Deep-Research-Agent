import logging
import requests
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0
    domain: str = ""

    def __post_init__(self):
        # strip www. so domains look cleaner in citations
        if self.url and not self.domain:
            self.domain = urlparse(self.url).netloc.replace("www.", "", 1)


class TavilySearcher:
    """Uses Tavily's search API - supports advanced depth which gives better results"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"

    def search(self, query: str, max_results: int = 8) -> List[SearchResult]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": False,
        }

        try:
            resp = requests.post(self.base_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning("Tavily search failed for query '%s': %s", query, e)
            return []
        except Exception as e:
            logger.error("Unexpected error in Tavily search: %s", e)
            return []

        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                score=float(item.get("score", 0.0)),
            ))
        return results


class SerperSearcher:
    """Serper wraps Google search - good fallback if Tavily quota runs out"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"

    def search(self, query: str, max_results: int = 8) -> List[SearchResult]:
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                self.base_url,
                headers=headers,
                json={"q": query, "num": max_results},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning("Serper search error: %s", e)
            return []

        results = []
        organic = data.get("organic", [])[:max_results]
        for i, item in enumerate(organic):
            # no relevance score from serper so approximate from position
            approx_score = round(1.0 - i * 0.08, 2)
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                score=max(0.1, approx_score),
            ))
        return results


def get_searcher(tavily_key: str = "", serper_key: str = ""):
    """Returns whichever searcher has a valid key. Tavily preferred."""
    if tavily_key:
        return TavilySearcher(tavily_key)
    if serper_key:
        return SerperSearcher(serper_key)
    raise ValueError("Provide at least one of: TAVILY_API_KEY or SERPER_API_KEY")
