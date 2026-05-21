import re
import logging
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse
from typing import Optional

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# tags that are almost never useful for reading
JUNK_TAGS = ["script", "style", "noscript", "nav", "footer", "header",
             "aside", "form", "button", "svg", "iframe"]


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str
    domain: str
    retrieved_at: str   # ISO timestamp so we know how fresh the data is


def fetch_page(url: str, timeout: int = 15, max_chars: int = 50000) -> Optional[FetchedPage]:
    """
    Downloads a URL and returns clean readable text.
    Returns None if anything goes wrong (network error, non-HTML, etc.)
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()

        # skip PDFs and other non-HTML content
        ctype = resp.headers.get("content-type", "")
        if "text/html" not in ctype and "xhtml" not in ctype:
            logger.debug("Skipping non-HTML content at %s (%s)", url, ctype)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(JUNK_TAGS):
            tag.decompose()

        # grab title - try <title> first, fall back to first h1
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)

        raw_text = soup.get_text(separator="\n", strip=True)

        # clean up the wall of whitespace that bs4 tends to produce
        text = re.sub(r"\n{3,}", "\n\n", raw_text)
        text = re.sub(r"[ \t]{3,}", "  ", text)
        text = text.strip()

        domain = urlparse(url).netloc.replace("www.", "", 1)

        return FetchedPage(
            url=url,
            title=title,
            text=text[:max_chars],
            domain=domain,
            retrieved_at=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    except requests.Timeout:
        logger.debug("Timeout fetching %s", url)
        return None
    except requests.RequestException as e:
        logger.debug("Request error for %s: %s", url, e)
        return None
    except Exception as e:
        # catch-all so one bad page doesn't kill the whole research run
        logger.debug("Unexpected error fetching %s: %s", url, e)
        return None
