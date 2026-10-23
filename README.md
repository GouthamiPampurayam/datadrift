# 🛡️ DataDrift — AI Data Pipeline Guardian

Hacktoberfest Hack Day 2026 · Track 1: Best Open-Source AI Project

Companies lose time and trust to duplicate records, missing values, schema changes and silently wrong numbers.
DataDrift detects these issues in an order/event pipeline, measures their business impact, and uses an
**open-weight LLM running locally** to explain likely causes and suggest safe fixes — grounded strictly in evidence.

## How it works
`dirty data → pipeline.py detects incidents (JSON) → analysis.py scores & measures impact → ai_explainer.py explains → app.py dashboard`

| Module | Owner | What it does |
|---|---|---|
| `pipeline.py` | Yasir | Sample data, issue injection, detection → incident JSON, safe cleaning |
| `analysis.py` | Goutami | Quality score, error rates, revenue impact, Isolation Forest anomalies, charts |
| `ai_explainer.py` | Amaan | Ollama prompt → JSON (explanation, evidence used, fix); says "insufficient evidence" when unsure |
| `app.py` | Sostika | Streamlit dashboard + integration |

**AI guardrails:** temperature 0, JSON-only output, evidence keys must exist in the incident or the answer is downgraded to
"insufficient evidence", fixes must be reversible. If Ollama is offline a rule-based fallback keeps the demo working (clearly labelled).

## Setup
```bash
pip install -r requirements.txt
# AI (open-weight model, runs locally)
ollama pull gemma3:4b        # or set DATADRIFT_MODEL to any model you pulled
streamlit run app.py
```

## Model & license
- Model: **Gemma 3 (4B, a small language model ≤10B)** by Google, open-weight — see terms: https://ai.google.dev/gemma/terms
  *(update this line if you use a different model)*
- Served locally with **Ollama** (MIT).
- This repository: **MIT License** (see `LICENSE`).

## Demo flow (2 min)
1. Pick **Demo: corrupted orders** → quality score drops, incidents appear.
2. **Impact** tab → revenue overstated by duplicates and ×100 totals.
3. **Incidents** tab → *Explain all* → AI cites evidence and suggests fixes.
4. **Fixes** tab → cleaned CSV and incident report download.

## Screenshots
_Add after the final run._
