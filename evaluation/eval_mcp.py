"""
evaluation/eval_mcp.py
----------------------
Strict, deterministic MCP (Intent + Entity) evaluation — real mode only.

Pipeline per sample:
  1. POST question to /api/chat/query → read answer from response
  2. Slot coverage:    fraction of required_slot VALUES found in the answer
  3. Keyword retrieval: fraction of expected_keywords found in the answer
  4. Success rate:     fraction of backend calls that returned without error

Score = 0.45 * avg_slot_coverage + 0.45 * avg_keyword_retrieval + 0.10 * success_rate

Per-category breakdown is also included in the output dict (availability,
pricing, discovery, booking) so the dashboard can show which intent category
has the weakest factual grounding.

Dataset schema (mcp_queries.json):
  {
    "id":               int,
    "question":         str,
    "expected_intent":  str,
    "expected_mode":    str,
    "required_slots":   {"doctor"?: str, "specialty"?: str, "clinic"?: str},
    "optional_slots":   {"date"?: str | null, ...},
    "category":         str,   # availability | pricing | discovery | booking
    "expected_keywords": [str, ...]
  }
"""

import json
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import BACKEND_URL, MCP_QUERIES, THRESHOLDS
from .http_client import post_with_retry
from .metrics import arabic_normalize, latency_percentiles

logger = logging.getLogger("eval.mcp")


# ── Slot-level F1 ──────────────────────────────────────────────────────────────

def _norm(val: Any) -> str:
    """Normalise a slot value for fair comparison."""
    if val is None:
        return ""
    return arabic_normalize(str(val)).strip().lower()


def _slot_f1(
    expected: Dict[str, Any],
    extracted: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute strict slot-level Precision, Recall, F1 with explicit TP/FP/FN.

    Only non-null expected slot values are evaluated. Optional slots with null
    values are skipped (they're hints for the evaluator, not hard requirements).

    Returns:
        {"tp": int, "fp": int, "fn": int,
         "precision": float, "recall": float, "f1": float,
         "missed": [slot_name, ...], "hallucinated": [slot_name, ...]}
    """
    # Filter expected: only non-null values are "required"
    required = {k: v for k, v in expected.items() if v is not None}

    tp, fp, fn = 0, 0, 0
    missed: List[str] = []
    hallucinated: List[str] = []

    # FN: expected slots missing or wrong in extracted
    for slot, exp_val in required.items():
        ext_val = extracted.get(slot)
        if ext_val is not None and _norm(ext_val) == _norm(exp_val):
            tp += 1
        else:
            fn += 1
            missed.append(slot)

    # FP: extracted slots not in expected (hallucinated) OR value wrong (already fn'd)
    for slot, ext_val in extracted.items():
        if slot not in required:
            fp += 1
            hallucinated.append(slot)
        elif _norm(ext_val) != _norm(required.get(slot, "")):
            fp += 1   # wrong value counts as FP as well as FN

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not extracted else 0.0)
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp":           tp,
        "fp":           fp,
        "fn":           fn,
        "precision":    precision,
        "recall":       recall,
        "f1":           f1,
        "missed":       missed,
        "hallucinated": hallucinated,
    }


# ── Backend call ───────────────────────────────────────────────────────────────

def _query_backend(
    question: str,
    provider_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    if provider_fn is not None:
        import time
        start = time.monotonic()
        result = provider_fn(question)
        lat = (time.monotonic() - start) * 1000
        return {
            "answer":     result.get("answer", ""),
            "sources":    result.get("sources", []),
            "latency_ms": lat,
            "success":    True,
        }

    resp = post_with_retry("/api/chat/query", json={"query": question})
    if not resp["ok"]:
        logger.error("MCP backend error: %s", resp["error"])
        return {"answer": "", "sources": [],
                "latency_ms": resp["latency_ms"], "success": False, "error": resp["error"]}

    data = resp["json"] or {}
    return {
        "answer":     data.get("answer", ""),
        "sources":    data.get("sources", []),
        "latency_ms": resp["latency_ms"],
        "success":    True,
    }


# ── Main eval ──────────────────────────────────────────────────────────────────

def run_eval(
    provider_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run MCP knowledge retrieval evaluation.

    The backend /api/chat/query returns plain-text answers. We evaluate:
      1. Slot coverage:   fraction of required_slot VALUES found in the answer
      2. Keyword retrieval: fraction of expected_keywords found in the answer
         (measures whether the system retrieved and surfaced the right knowledge)
      3. Success rate:    fraction of samples that got a valid (non-error) response

    Score = 0.45 * avg_slot_coverage + 0.45 * avg_keyword_retrieval + 0.10 * success_rate

    Per-category breakdown (availability, pricing, discovery, booking) is included
    in the output for dashboard display.
    """
    samples: List[dict] = []
    try:
        with open(MCP_QUERIES) as f:
            samples = json.load(f)
    except Exception as e:
        return {"component": "mcp", "error": str(e), "score": 0.0}

    slot_coverage_vals:    List[float] = []
    keyword_retrieval_vals: List[float] = []
    success_flags:         List[bool]  = []
    latencies:             List[float] = []
    details = []

    # Per-category accumulators: {category: {"slot": [], "kw": []}}
    cat_slot: Dict[str, List[float]]  = defaultdict(list)
    cat_kw:   Dict[str, List[float]]  = defaultdict(list)

    for sample in samples:
        if not sample.get("question"):
            continue

        question        = sample["question"]
        required_slots  = sample.get("required_slots", {})
        expected_kws    = sample.get("expected_keywords", [])
        category        = sample.get("category", "unknown")

        result = _query_backend(question, provider_fn=provider_fn)
        latencies.append(result.get("latency_ms", 0))
        success_flags.append(result["success"])

        if not result["success"]:
            details.append({
                "question":          question,
                "category":          category,
                "required_slots":    required_slots,
                "slot_coverage":     0.0,
                "keyword_retrieval": 0.0,
                "error":             result.get("error"),
                "latency_ms":        round(result.get("latency_ms", 0), 1),
            })
            slot_coverage_vals.append(0.0)
            keyword_retrieval_vals.append(0.0)
            cat_slot[category].append(0.0)
            cat_kw[category].append(0.0)
            continue

        answer   = result.get("answer", "")
        norm_ans = arabic_normalize(answer).lower()

        # 1. Slot coverage: required slot values in answer
        required_vals = {k: v for k, v in required_slots.items() if v is not None}
        slot_hits, slot_misses = [], []
        for slot, val in required_vals.items():
            norm_val = arabic_normalize(str(val)).lower()
            if norm_val and norm_val in norm_ans:
                slot_hits.append(slot)
            else:
                slot_misses.append(slot)
        slot_cov = len(slot_hits) / len(required_vals) if required_vals else 1.0

        # 2. Knowledge retrieval: expected Arabic keywords in answer
        kw_hits, kw_misses = [], []
        for kw in expected_kws:
            norm_kw = arabic_normalize(kw).lower()
            if norm_kw and norm_kw in norm_ans:
                kw_hits.append(kw)
            else:
                kw_misses.append(kw)
        kw_rate = len(kw_hits) / len(expected_kws) if expected_kws else 0.0

        slot_coverage_vals.append(slot_cov)
        keyword_retrieval_vals.append(kw_rate)
        cat_slot[category].append(slot_cov)
        cat_kw[category].append(kw_rate)

        details.append({
            "question":          question,
            "category":          category,
            "required_slots":    required_slots,
            "slot_hits":         slot_hits,
            "slot_misses":       slot_misses,
            "slot_coverage":     round(slot_cov, 3),
            "expected_keywords": expected_kws,
            "keyword_hits":      kw_hits,
            "keyword_misses":    kw_misses,
            "keyword_retrieval": round(kw_rate, 3),
            "answer_preview":    answer[:150],
            "latency_ms":        round(result.get("latency_ms", 0), 1),
        })

    total = len(details)
    if total == 0:
        return {"component": "mcp", "score": 0.0, "sample_count": 0, "details": []}

    avg_slot_cov   = sum(slot_coverage_vals)    / total
    avg_kw_ret     = sum(keyword_retrieval_vals) / total
    success_rate   = sum(1 for f in success_flags if f) / len(success_flags) if success_flags else 0.0
    # Weighted composite: slot coverage + keyword retrieval + endpoint health
    score          = 0.45 * avg_slot_cov + 0.45 * avg_kw_ret + 0.10 * success_rate
    perf           = latency_percentiles(latencies)

    # Per-category breakdown
    category_scores: Dict[str, Dict[str, float]] = {}
    all_cats = set(cat_slot.keys()) | set(cat_kw.keys())
    for cat in all_cats:
        s_vals = cat_slot.get(cat, [])
        k_vals = cat_kw.get(cat, [])
        cat_s = sum(s_vals) / len(s_vals) if s_vals else 0.0
        cat_k = sum(k_vals) / len(k_vals) if k_vals else 0.0
        category_scores[cat] = {
            "slot_coverage":     round(cat_s, 3),
            "keyword_retrieval": round(cat_k, 3),
            "score":             round(0.5 * cat_s + 0.5 * cat_k, 3),
            "sample_count":      len(s_vals),
        }

    return {
        "component":               "mcp",
        "score":                   round(score,         3),
        "avg_slot_coverage":       round(avg_slot_cov,  3),
        "avg_keyword_retrieval":   round(avg_kw_ret,    3),
        "success_rate":            round(success_rate,  3),
        "total":                   total,
        "latency":                 perf,
        "sample_count":            total,
        "category_scores":         category_scores,
        "details":                 details,
    }
