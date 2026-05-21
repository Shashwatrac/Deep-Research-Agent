"""
Evaluation metrics module.

Defines and computes all metrics used to assess the Deep Research Agent:

  1. citation_rate            – fraction of answer sentences that contain ≥1 citation
  2. grounding_score          – fraction of cited URLs that appeared in actual search results
  3. uncertainty_acknowledgment – 1 if expected-uncertain answer contains hedge language
  4. aspect_coverage          – fraction of expected answer aspects found in the response
  5. source_diversity         – number of unique domains cited
  6. conflict_flagging        – 1 if a conflicting-source question mentions disagreement
  7. answer_length            – raw word count (proxy for completeness)

All metrics return a value in [0, 1] or a non-negative integer.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional


# ── Regex helpers ─────────────────────────────────────────────────────────────

_MARKDOWN_LINK_RE   = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
_SENTENCE_SPLIT_RE  = re.compile(r"[.!?]+\s+")
_HEDGE_PHRASES = [
    "not found", "cannot confirm", "insufficient evidence",
    "unclear", "uncertain", "unknown", "could not find",
    "no information", "unable to verify", "may", "might",
    "it is unclear", "limited information", "suggests",
    "estimated", "projected", "uncertain", "no data",
    "beyond my", "not available",
]
_CONFLICT_PHRASES = [
    "conflict", "contradict", "disagree", "on the other hand",
    "however", "while some", "others argue", "debate", "disputed",
    "mixed evidence", "on one hand", "varies", "whereas",
    "in contrast", "differ",
]


# ── Individual metrics ────────────────────────────────────────────────────────

def citation_rate(answer: str) -> float:
    """Fraction of sentences that contain at least one markdown citation."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if _MARKDOWN_LINK_RE.search(s))
    return cited / len(sentences)


def grounding_score(answer: str, retrieved_urls: List[str]) -> float:
    """
    Fraction of URLs cited in the answer that actually came from the
    retrieved search results.  A score of 1.0 means every cited URL
    was in the source pool (no hallucinated URLs).
    """
    cited_urls = _MARKDOWN_LINK_RE.findall(answer)   # returns (text, url) tuples
    cited_url_set = {url for _, url in cited_urls}

    if not cited_url_set:
        return 0.0   # no citations at all – penalise

    retrieved_set = set(retrieved_urls)
    matched = cited_url_set & retrieved_set
    return len(matched) / len(cited_url_set)


def uncertainty_acknowledgment(answer: str, expects_uncertainty: bool) -> float:
    """
    Returns 1.0 if the question was expected to have uncertain evidence
    AND the answer contains hedge / uncertainty language.
    Returns 1.0 for non-uncertain questions (not applicable).
    """
    if not expects_uncertainty:
        return 1.0   # metric not applicable; full credit
    lower = answer.lower()
    for phrase in _HEDGE_PHRASES:
        if phrase in lower:
            return 1.0
    return 0.0


def aspect_coverage(answer: str, expected_aspects: List[str]) -> float:
    """
    Fraction of expected_aspects that appear (loosely) in the answer.
    Uses case-insensitive substring match for simplicity.
    """
    if not expected_aspects:
        return 1.0
    answer_lower = answer.lower()
    found = sum(
        1 for aspect in expected_aspects
        if any(word.lower() in answer_lower for word in aspect.split())
    )
    return found / len(expected_aspects)


def source_diversity(answer: str) -> int:
    """Number of unique domains cited in the answer."""
    cited_urls = {url for _, url in _MARKDOWN_LINK_RE.findall(answer)}
    domains = set()
    for url in cited_urls:
        m = re.match(r"https?://([^/]+)", url)
        if m:
            domains.add(m.group(1).removeprefix("www."))
    return len(domains)


def conflict_flagging(answer: str, question_type: str) -> float:
    """
    For conflicting-source questions: 1.0 if the answer flags disagreement.
    For all other types: returns 1.0 (not applicable).
    """
    if question_type != "conflicting_sources":
        return 1.0
    lower = answer.lower()
    for phrase in _CONFLICT_PHRASES:
        if phrase in lower:
            return 1.0
    return 0.0


def answer_length(answer: str) -> int:
    """Word count of the answer (proxy for completeness)."""
    return len(answer.split())


# ── Aggregate scorer ──────────────────────────────────────────────────────────

def score_answer(
    answer:            str,
    question:          Dict,
    retrieved_urls:    List[str],
) -> Dict:
    """
    Compute all applicable metrics for one answer and return a dict.

    *question* must have keys: id, type, expected_aspects, requires_recency
    """
    q_type      = question.get("type", "factual")
    aspects     = question.get("expected_aspects", [])
    uncertain   = q_type == "insufficient_evidence"

    scores = {
        "question_id":               question["id"],
        "question_type":             q_type,
        "citation_rate":             round(citation_rate(answer), 3),
        "grounding_score":           round(grounding_score(answer, retrieved_urls), 3),
        "uncertainty_acknowledgment":round(uncertainty_acknowledgment(answer, uncertain), 3),
        "aspect_coverage":           round(aspect_coverage(answer, aspects), 3),
        "source_diversity":          source_diversity(answer),
        "conflict_flagging":         round(conflict_flagging(answer, q_type), 3),
        "answer_length_words":       answer_length(answer),
    }

    # ── Composite score (equal-weighted average of [0,1] metrics) ────────────
    numeric = [
        scores["citation_rate"],
        scores["grounding_score"],
        scores["uncertainty_acknowledgment"],
        scores["aspect_coverage"],
        scores["conflict_flagging"],
    ]
    scores["composite_score"] = round(sum(numeric) / len(numeric), 3)

    return scores
