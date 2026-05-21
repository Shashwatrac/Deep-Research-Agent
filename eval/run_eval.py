#!/usr/bin/env python3
"""
Evaluation runner for Deep Research Agent.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --tavily-key tvly-xxx
    python eval/run_eval.py --questions factual_1 conflict_1 insufficient_1
    python eval/run_eval.py --model mistral --output eval/results.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agent.llm import OllamaClient
from agent.loop import run_agent
from agent.session import create_session, init_db
from eval.metrics import score_answer

logging.basicConfig(level=logging.WARNING)

DATASET_PATH = Path(__file__).parent / "dataset.json"
EVAL_DB      = str(ROOT / "eval_sessions.db")


def load_dataset(ids: List[str] = None) -> List[Dict]:
    data = json.loads(DATASET_PATH.read_text())
    if ids:
        data = [q for q in data if q["id"] in ids]
    return data


def run_one(question: Dict, tavily_key: str, serper_key: str, ollama_url: str, model: str) -> Dict:
    """Run the agent on a single question and return a result dict."""
    init_db(EVAL_DB)
    session_id = create_session(EVAL_DB)

    answer = ""
    retrieved_urls = []
    search_queries = []
    error = None
    t0 = time.time()

    try:
        for event in run_agent(
            query=question["question"],
            session_id=session_id,
            tavily_key=tavily_key,
            serper_key=serper_key,
            ollama_base_url=ollama_url,
            ollama_model=model,
            db_path=EVAL_DB,
        ):
            etype = event.get("type")
            if etype == "answer_chunk":
                answer += event["content"]
            elif etype == "done":
                data = event.get("data", {})
                retrieved_urls = [c["url"] for c in data.get("citations", [])]
                search_queries = data.get("search_queries", [])
            elif etype == "error":
                error = event.get("message")
                break
    except Exception as e:
        error = str(e)

    elapsed = round(time.time() - t0, 2)
    scores  = score_answer(answer, question, retrieved_urls)

    return {
        "question_id":     question["id"],
        "question_type":   question["type"],
        "question_text":   question["question"],
        "session_id":      session_id,
        "search_queries":  search_queries,
        "retrieved_urls":  retrieved_urls,
        "answer":          answer,
        "error":           error,
        "elapsed_seconds": elapsed,
        "scores":          scores,
        "timestamp":       datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def print_summary(results: List[Dict]):
    print("\n" + "=" * 78)
    print("EVALUATION RESULTS")
    print("=" * 78)
    print(f"{'ID':<20} {'Type':<22} {'Cit':>4} {'Gnd':>4} {'Unc':>4} {'Asp':>4} {'Div':>3} {'Comp':>5} {'T(s)':>5}")
    print("-" * 78)

    composites = []
    for r in results:
        if r.get("error"):
            print(f"  ERROR  {r['question_id']} — {r['error'][:50]}")
            continue
        s = r["scores"]
        print(
            f"{s['question_id']:<20} {s['question_type']:<22} "
            f"{s['citation_rate']:>4.2f} {s['grounding_score']:>4.2f} "
            f"{s['uncertainty_acknowledgment']:>4.1f} {s['aspect_coverage']:>4.2f} "
            f"{s['source_diversity']:>3d} {s['composite_score']:>5.2f} "
            f"{r['elapsed_seconds']:>5.1f}"
        )
        composites.append(s["composite_score"])

    if composites:
        print("-" * 78)
        print(f"{'AVERAGE':>60}  {sum(composites)/len(composites):.3f}")

    print("=" * 78)
    print("""
Metrics:
  Cit  — citation_rate: fraction of sentences with ≥1 inline citation
  Gnd  — grounding_score: fraction of cited URLs that came from actual search results
  Unc  — uncertainty_acknowledgment: 1.0 if agent hedged on uncertain questions
  Asp  — aspect_coverage: fraction of expected answer aspects present
  Div  — source_diversity: unique domains cited
  Comp — composite score (avg of the above [0,1] metrics)
""")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Deep Research Agent")
    parser.add_argument("--tavily-key",  default="", help="Tavily API key")
    parser.add_argument("--serper-key",  default="", help="Serper API key")
    parser.add_argument("--ollama-url",  default="http://localhost:11434")
    parser.add_argument("--model",       default="llama3.2", help="Ollama model name")
    parser.add_argument("--questions",   nargs="*", help="Question IDs to run (default: all)")
    parser.add_argument("--output",      default="eval/eval_results.json")
    parser.add_argument("--delay",       type=float, default=1.5,
                        help="Seconds between questions (avoid rate limits)")
    args = parser.parse_args()

    tavily_key = args.tavily_key or os.getenv("TAVILY_API_KEY", "")
    serper_key = args.serper_key or os.getenv("SERPER_API_KEY", "")
    ollama_url = args.ollama_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model      = args.model      or os.getenv("OLLAMA_MODEL", "llama3.2")

    if not (tavily_key or serper_key):
        print("ERROR: Set TAVILY_API_KEY or pass --tavily-key")
        sys.exit(1)

    # quick Ollama check
    llm = OllamaClient(ollama_url, model)
    if not llm.is_available():
        print(f"ERROR: Ollama not reachable at {ollama_url}. Run: ollama serve")
        sys.exit(1)

    avail = llm.list_models()
    if avail and not any(model in m for m in avail):
        print(f"WARNING: model '{model}' not found locally. Pull it with: ollama pull {model}")
        print(f"Available: {avail}")

    dataset = load_dataset(args.questions)
    print(f"\nRunning eval on {len(dataset)} question(s) with model '{model}'...\n")

    results = []
    for i, q in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] {q['id']}  —  {q['question'][:65]}...")
        result = run_one(q, tavily_key, serper_key, ollama_url, model)
        results.append(result)

        comp = result["scores"].get("composite_score", 0)
        status = "✓" if not result["error"] else f"✗ {result['error'][:40]}"
        print(f"         {status}  composite={comp:.2f}  ({result['elapsed_seconds']}s)")

        if i < len(dataset):
            time.sleep(args.delay)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nFull results saved → {out}")

    print_summary(results)


if __name__ == "__main__":
    main()
