"""DataDrift - AI module (Member 2). Open-weight model via Ollama, grounded in incident evidence."""
import json
import os
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DATADRIFT_MODEL", "gemma3:4b")
INSUFFICIENT = "insufficient evidence"

SYSTEM = f"""You are DataDrift, a careful data-engineering assistant.
You receive ONE data-quality incident as JSON, with an "evidence" object.
Rules:
1. Use ONLY facts present in the evidence. Never invent systems, vendors, people, dates or numbers.
2. If the evidence does not support a specific cause, set likely_cause and confidence to "{INSUFFICIENT}".
3. List in evidence_used the exact evidence keys you relied on.
4. Suggested fixes must be safe and reversible (no deleting data without a backup, no guessing values).
Reply with JSON only, with keys:
explanation (string, plain language, max 3 sentences), likely_cause (string),
confidence ("high"|"medium"|"low"|"{INSUFFICIENT}"), evidence_used (list of strings),
suggested_fix (string), safe_to_autofix (boolean)."""


def ollama_available():
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).ok
    except requests.RequestException:
        return False


def _validate(out, incident):
    keys = set(incident["evidence"])
    used = [k for k in out.get("evidence_used", []) if k in keys]
    res = {"explanation": str(out.get("explanation", "")), "likely_cause": str(out.get("likely_cause", INSUFFICIENT)),
           "confidence": str(out.get("confidence", INSUFFICIENT)).lower(), "evidence_used": used,
           "suggested_fix": str(out.get("suggested_fix", "")), "safe_to_autofix": bool(out.get("safe_to_autofix", False))}
    if not used:  # guardrail: no cited evidence -> no claimed cause
        res.update(likely_cause=INSUFFICIENT, confidence=INSUFFICIENT)
    return res


def _rule_based(inc):
    e, t = inc["evidence"], inc["type"]
    if t == "duplicate_records":
        near = e["pct_directly_after_original"] >= 70 and e["all_exact_copies"]
        return {"explanation": f"{e['duplicate_rows']} rows are exact copies of earlier orders.",
                "likely_cause": "Records were re-inserted right after the originals, consistent with a retry or replay in ingestion."
                if near else INSUFFICIENT,
                "confidence": "medium" if near else INSUFFICIENT,
                "evidence_used": [k for k in ("duplicate_rows", "all_exact_copies", "pct_directly_after_original") if k in e],
                "suggested_fix": "Drop exact duplicates keeping the first occurrence; add an idempotent upsert on order_id.",
                "safe_to_autofix": True}
    if t == "missing_values":
        late = e["pct_in_last_quarter_of_file"] >= 70
        return {"explanation": f"{e['null_count']} values ({e['null_pct']}%) are missing in this column.",
                "likely_cause": "Nulls are concentrated in the latest part of the file, suggesting a late batch or upstream change."
                if late else INSUFFICIENT,
                "confidence": "medium" if late else INSUFFICIENT,
                "evidence_used": ["null_count", "pct_in_last_quarter_of_file"],
                "suggested_fix": "Check the upstream source for the late batch; do not guess values, flag rows for review.",
                "safe_to_autofix": False}
    if t == "schema_drift" and e.get("likely_rename"):
        return {"explanation": f"Expected column(s) {e['missing_columns']} are missing and {e['new_columns']} appeared.",
                "likely_cause": f"Column rename: {e['likely_rename']}.", "confidence": "medium",
                "evidence_used": ["missing_columns", "new_columns", "likely_rename"],
                "suggested_fix": "Map the new name back to the contract and confirm the change with the data producer.",
                "safe_to_autofix": True}
    if t == "value_inconsistency":
        r = e["median_ratio_total_to_expected"]
        return {"explanation": f"{e['mismatch_rows']} order totals differ from quantity x unit_price (median ratio {r}).",
                "likely_cause": f"Totals are about {r}x the expected value, consistent with a scaling or unit error."
                if r >= 5 else INSUFFICIENT,
                "confidence": "medium" if r >= 5 else INSUFFICIENT,
                "evidence_used": ["mismatch_rows", "median_ratio_total_to_expected"],
                "suggested_fix": "Recalculate order_total from quantity x unit_price and quarantine originals for audit.",
                "safe_to_autofix": True}
    return {"explanation": inc["title"], "likely_cause": INSUFFICIENT, "confidence": INSUFFICIENT,
            "evidence_used": [], "suggested_fix": "Investigate manually.", "safe_to_autofix": False}


def explain_incident(incident, model=DEFAULT_MODEL, use_llm=True):
    if use_llm:
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=180, json={
                "model": model, "stream": False, "format": "json", "options": {"temperature": 0},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": json.dumps(incident, default=str)}]})
            r.raise_for_status()
            out = _validate(json.loads(r.json()["message"]["content"]), incident)
            out["source"] = f"LLM: {model} via Ollama"
            return out
        except (requests.RequestException, ValueError, KeyError):
            pass
    out = _rule_based(incident)
    out["source"] = "Rule-based fallback (Ollama not reachable)" if use_llm else "Rule-based (AI off)"
    return out
