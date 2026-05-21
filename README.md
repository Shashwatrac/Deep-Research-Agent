# Deep Research Agent

A web research agent built from scratch in Python. You give it a question, it figures out what to search for, pulls real web sources, and writes back a cited answer — all running locally with no paid LLM API.

**Stack:** Python + Streamlit + Tavily + Ollama (local LLM)  
**No frameworks:** No LangChain, LangGraph, LlamaIndex, CrewAI, or anything like that. Just Python.

---

## Demo

> [Video link here]

---

## Setup

**Prerequisites:**
- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- A free [Tavily API key](https://tavily.com) (1000 searches/month free)

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Pull a local model (llama3.2 is a good default, ~2GB)
ollama pull llama3.2

# 3. Set your keys
cp .env.example .env
# edit .env and add your TAVILY_API_KEY

# 4. Run
python -m streamlit run app.py
```

The app will open at http://localhost:8501. Keys can also be pasted directly into the sidebar instead of using .env.

**Other models that work well:**
```bash
ollama pull mistral        # good for reasoning, ~4GB
ollama pull qwen2.5        # strong on structured output, ~4GB
ollama pull gemma2         # Google's model, fast, ~5GB
ollama pull llama3.2:1b    # tiny, fast, less accurate
```

### Running the eval harness
```bash
python eval/run_eval.py --tavily-key tvly-dev-xxxx

# specific questions only:
python eval/run_eval.py --questions factual_1 conflict_1

# different model:
python eval/run_eval.py --model mistral
```

---

## How it works

Each user query goes through 5 steps:

```
question
   │
   ▼
[Planner]  — LLM reads the question and writes 2-4 targeted search queries
   │           (better than dumping the raw question into a search box)
   ▼
[Searcher] — hits Tavily API with each query, collects results
   │           deduplicates URLs, sorts by relevance score
   ▼
[Fetcher]  — downloads top 5 pages concurrently
   │           strips nav/ads/scripts, keeps readable text
   ▼
[Context]  — ranks content by keyword overlap with the question
   │           enforces domain diversity (no single site dominates)
   │           fits everything into the model's context window
   ▼
[Answerer] — streams the answer from the local LLM
               every claim cited as [Title - domain](URL)
               flags conflicting sources explicitly
```

Nothing fancy — just a generator in `agent/loop.py` that calls each step in order. The UI reads events from the generator and renders them as they arrive.

---

## Project Structure

```
deep_research_agent/
├── app.py                   # Streamlit UI
├── config.py                # env vars and tuning constants
├── agent/
│   ├── llm.py               # Ollama client (streaming + blocking)
│   ├── loop.py              # main agent loop (the orchestrator)
│   ├── planner.py           # query planning via LLM
│   ├── search.py            # Tavily / Serper search wrappers
│   ├── fetcher.py           # HTTP + HTML → clean text
│   ├── context.py           # context selection and ranking
│   ├── answerer.py          # streaming answer generation
│   └── session.py           # SQLite session + history management
├── eval/
│   ├── dataset.json         # 12 test questions
│   ├── metrics.py           # metric definitions
│   └── run_eval.py          # eval CLI
└── requirements.txt
```

---

## Part 1 — Design Note

### Who is this for?

People who need answers they can actually verify. Generic chatbots either make things up or give you stale training-data answers. This agent only answers from URLs it actually fetched during your query, so every claim is traceable.

Main use cases: researchers, journalists doing background checks, students fact-checking before citing something, developers trying to understand a new technology landscape.

### What counts as "deep research" here?

Three things:

1. **Multi-source synthesis** — the agent pulls from at least 3 different domains before forming an answer. Single-source answers are flagged as low-confidence.

2. **Explicit uncertainty** — the agent distinguishes between "I found clear evidence", "sources disagree on this", and "I couldn't find enough info". All three states are communicated differently in the answer.

3. **Traceable claims** — every factual sentence links back to a specific URL. If you want to verify anything, you can.

### Success metrics (and why these five)

| Metric | What it measures | Why it matters |
|---|---|---|
| **Citation Rate** | % of answer sentences with ≥1 citation | Primary trust signal — grounded answers cite their sources |
| **Grounding Score** | % of cited URLs that were in the actual search results | Catches hallucinated URLs — a model might cite a plausible-looking link that doesn't exist |
| **Aspect Coverage** | % of expected answer dimensions found in the response | Completeness check for structured questions (comparisons, multi-hop) |
| **Uncertainty Acknowledgment** | Does the agent hedge when it should? | The failure mode we care most about: confident-sounding wrong answers |
| **Conflict Flagging** | Does the agent mention when sources disagree? | Research honesty — silently picking one side is misleading |

I picked these over simpler metrics like BLEU/ROUGE because this is a retrieval+generation task. Measuring n-gram overlap against a reference answer is useless when the "correct" answer depends on what was retrieved today.

### Data flow

See the diagram in the "How it works" section above.

The one non-obvious part is the context builder. Search results come with a relevance score from Tavily, but that score is computed by Tavily's ranking — it doesn't know specifically what parts of a page answer the question. So the context builder does a second pass: it splits fetched pages into paragraphs, scores each paragraph by keyword overlap with the query, and takes the top 4. This consistently pulls better content than just slicing from the top of the page (which is usually navigation/boilerplate anyway).

### Risks and limitations

| Risk | What I did about it |
|---|---|
| Rate limits on Tavily free tier (1000/month) | Configurable query count, Serper fallback |
| JS-heavy pages not rendering | Skip gracefully, use search snippet as fallback |
| Local models worse at structured JSON (planner) | JSON repair + fallback if parse fails |
| Context length limits on small local models | MAX_CONTEXT_CHARS defaults to 12k (conservative) |
| Conflicting sources silently ignored | LLM system prompt explicitly requires flagging conflicts |
| Hallucinated citations | Grounding Score metric detects URLs not in retrieval pool |

**Two things I'd add next:**

1. **Iterative search loop** — after generating an answer, check which aspects had weak sourcing and run one more targeted search for just those. Right now it's a single pass. A second pass on under-sourced claims would meaningfully improve coverage on complex questions.

2. **Embedding-based context selection** — the current keyword overlap scoring works but misses semantic similarity. Replacing it with a local embedding model (e.g. `nomic-embed-text` via Ollama) would surface relevant paragraphs even when they don't share keywords with the query.

---

## Part 3 — Evaluation

### Dataset

12 questions across 6 types:

| Type | Questions | What it tests |
|---|---|---|
| `factual` | 2 | Basic accuracy, recency |
| `comparison` | 2 | Multi-dimensional analysis |
| `multi_hop` | 2 | Connecting facts across multiple sources |
| `insufficient_evidence` | 2 | Hallucination resistance — does the agent admit it doesn't know? |
| `conflicting_sources` | 2 | Does the agent surface disagreement rather than picking a side? |
| `deep_research` | 2 | Broad synthesis requiring many sources |

### Metric rationale

The five metrics above are chosen to cover different failure modes:

- **Citation Rate + Grounding Score** together = trust. High citation rate with low grounding = the model is inventing plausible-looking URLs (bad). Both high = genuinely sourced answers.
- **Aspect Coverage** = completeness. A 2-sentence answer might cite a source but miss half the question.
- **Uncertainty Acknowledgment** = honesty. The hardest failure to catch with automated metrics.
- **Conflict Flagging** = balance. Specifically targets the conflicting-sources question type.

### Running it

```bash
python eval/run_eval.py --output eval/eval_results.json
```

Output is a JSON file with every question's full answer, retrieved URLs, and per-metric scores. The terminal also prints a summary table.

### Example output

```
================================================================================
EVALUATION RESULTS
================================================================================
ID                   Type                   Cit  Gnd  Unc  Asp  Div  Comp   T(s)
--------------------------------------------------------------------------------
factual_1            factual               0.48 0.92 1.0 0.75   3  0.81   18.2
comparison_1         comparison            0.55 0.88 1.0 0.70   4  0.78   24.5
conflict_1           conflicting_sources   0.42 0.85 1.0 0.65   3  0.79   21.3
insufficient_1       insufficient_evidence 0.20 0.60 1.0 1.00   1  0.76   15.8
...
--------------------------------------------------------------------------------
                                              AVERAGE                      0.77
================================================================================
```

Results vary by model. `llama3.2` and `mistral` both perform well. Smaller models like `llama3.2:1b` struggle with the planner's JSON output but still produce reasonable answers.

---

## Example conversations

**Factual question:**
```
Q: What's the latest stable Python version?

→ Planning: "Search for latest Python release on official site and news"
→ Searching: 2 queries...
→ Fetching: python.org, docs.python.org...
→ Answer:

Python 3.13.1 is the current stable release...
[Python Downloads — python.org](https://www.python.org/downloads/)
```

**Conflicting evidence:**
```
Q: Does coffee help or hurt your health?

→ Answer:

Research on this is genuinely mixed. Several studies link moderate 
coffee consumption to reduced risk of type 2 diabetes and liver disease
[Coffee Health Benefits — health.harvard.edu](https://...).

On the other hand, high caffeine intake has been associated with 
increased anxiety and disrupted sleep in some populations
[Caffeine and Sleep — sleepfoundation.org](https://...).

The disagreement largely comes down to dosage and individual sensitivity...
```

**Insufficient evidence:**
```
Q: What are the financials of NexaCloud AI, a startup founded last month?

→ Answer:

I couldn't find any information about a company called NexaCloud AI.
It may be too new to have any indexed coverage, or the name might differ
in public records. For a recently founded startup, I'd suggest checking
Crunchbase, LinkedIn, or any press releases directly.
```

---

## Limitations

- Pages that require JavaScript to render won't work (SPAs, dynamic dashboards)
- Tavily free tier caps at 1000 searches/month
- Very large contexts may exceed smaller local models' effective window
- The eval metrics are automated proxies — not the same as human judgment
- No image/PDF support, text-only

---

## Notes on model choice

`llama3.2` (default) is a good balance of speed and quality on a typical laptop. If you have a GPU or more RAM:
- `mistral` handles longer context better
- `qwen2.5:7b` is noticeably better at following the JSON planner format
- `llama3.1:8b` is strong overall but slower

The planner is the most sensitive to model quality since it needs to output structured JSON. The answerer is more forgiving.
