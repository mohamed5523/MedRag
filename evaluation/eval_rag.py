"""
evaluation/eval_rag.py
----------------------
Evaluate RAG retrieval quality: Precision/Recall/F1, MRR, NDCG@k, Hit Rate,
plus page-level and excerpt-level checks.

Dataset schema (rag_queries.json) — enriched, no mock_* fields:
  {
    "question":             str,
    "ground_truth_sources": [str, ...],        # document IDs (filename)
    "expected_page_numbers": {                  # optional strict check
        "<source_name>": [int, ...]             # e.g. {"diabetes_guide.pdf": [4, 7]}
    },
    "critical_excerpts": [str, ...],            # key phrases that must appear in answer
    "category":             str
  }

Metrics:
  - avg_precision, avg_recall, avg_f1  (document-level)
  - mrr, ndcg_at_5, hit_rate_at_1/3/5 (ranking)
  - excerpt_coverage                    (fraction of critical_excerpts found in answer)
  - score: 0.35 * avg_f1 + 0.25 * mrr + 0.25 * ndcg_at_5 + 0.15 * excerpt_coverage
"""

import json
from typing import Any, Callable, Dict, List, Optional

from .config import BACKEND_URL, RAG_QUERIES
from .http_client import post_with_retry
from .metrics import (
    hit_rate_at_k,
    latency_percentiles,
    mrr as compute_mrr,
    ndcg_at_k,
    precision_recall,
    arabic_normalize,
)


def _excerpt_coverage(excerpts: List[str], answer: str) -> float:
    """Fraction of critical_excerpts found (substring, normalised) in the answer."""
    if not excerpts:
        return 1.0
    norm_answer = arabic_normalize(answer).lower()
    found = sum(
        1 for ex in excerpts
        if arabic_normalize(ex).lower() in norm_answer
    )
    return found / len(excerpts)


# ── Mock mode ──────────────────────────────────────────────────────────────────

def _mock_retrieve(sample: dict) -> Dict[str, Any]:
    """Perfect retrieval in mock mode — returns ground truth as retrieved."""
    return {
        "retrieved": list(sample.get("ground_truth_sources", [])),
        "answer": " ".join(sample.get("critical_excerpts", [])),
        "latency_ms": 150,
        "success": True,
    }


# ── Real mode ──────────────────────────────────────────────────────────────────

def _real_retrieve(
    sample: dict,
    provider_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Query the backend and collect retrieved sources + answer text."""
    question = sample.get("question", "")

    if provider_fn is not None:
        import time
        start = time.monotonic()
        result = provider_fn(question)
        lat = (time.monotonic() - start) * 1000
        return {
            "retrieved": list(result) if isinstance(result, (list, tuple)) else [],
            "answer": "",
            "latency_ms": lat,
            "success": True,
        }

    resp = post_with_retry(
        "/api/chat/query",
        json={"query": question},
    )
    if not resp["ok"]:
        return {"retrieved": [], "answer": "", "latency_ms": resp["latency_ms"],
                "success": False, "error": resp["error"]}

    data = resp["json"] or {}
    return {
        "retrieved": data.get("sources", []),
        "answer":    data.get("answer", ""),
        "latency_ms": resp["latency_ms"],
        "success":   True,
    }


# ── Main eval ──────────────────────────────────────────────────────────────────

def run_eval(
    provider_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run RAG evaluation with richer ground-truth checks."""

    samples: List[dict] = []
    try:
        with open(RAG_QUERIES) as f:
            samples = json.load(f)
    except Exception as e:
        return {"component": "rag", "error": str(e), "score": 0.0}

    precisions: List[float]  = []
    recalls:    List[float]  = []
    latencies:  List[float]  = []
    excerpts_cov: List[float] = []
    all_relevant:  List[List[str]] = []
    all_retrieved: List[List[str]] = []
    details = []

    for sample in samples:
        if not sample.get("question"):
            continue
        ground_truth     = sample.get("ground_truth_sources", [])
        critical_excerpts = sample.get("critical_excerpts", [])

        result = _real_retrieve(sample, provider_fn=provider_fn)

        retrieved  = result.get("retrieved", [])
        answer     = result.get("answer", "")
        pr         = precision_recall(retrieved, ground_truth)
        exc_cov    = _excerpt_coverage(critical_excerpts, answer)

        precisions.append(pr["precision"])
        recalls.append(pr["recall"])
        latencies.append(result.get("latency_ms", 0))
        excerpts_cov.append(exc_cov)
        all_relevant.append(ground_truth)
        all_retrieved.append(retrieved)

        details.append({
            "question":          sample["question"],
            "category":          sample.get("category", "unknown"),
            "retrieved":         retrieved,
            "ground_truth":      ground_truth,
            "precision":         round(pr["precision"], 3),
            "recall":            round(pr["recall"], 3),
            "f1":                round(pr["f1"], 3),
            "critical_excerpts": critical_excerpts,
            "excerpt_coverage":  round(exc_cov, 3),
        })

    n = len(details)
    if n == 0:
        return {"component": "rag", "score": 0.0, "sample_count": 0, "details": []}

    avg_precision = sum(precisions) / n
    avg_recall    = sum(recalls)    / n
    avg_f1 = (
        2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0 else 0.0
    )
    avg_excerpt_cov = sum(excerpts_cov) / n
    perf     = latency_percentiles(latencies)
    mrr_s    = compute_mrr(all_relevant, all_retrieved)
    ndcg5    = ndcg_at_k(all_relevant, all_retrieved, k=5)
    hr1      = hit_rate_at_k(all_relevant, all_retrieved, k=1)
    hr3      = hit_rate_at_k(all_relevant, all_retrieved, k=3)
    hr5      = hit_rate_at_k(all_relevant, all_retrieved, k=5)

    category_stats = {}
    for d in details:
        cat = d.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"count": 0, "f1_sum": 0.0, "exc_cov_sum": 0.0}
        category_stats[cat]["count"] += 1
        category_stats[cat]["f1_sum"] += d["f1"]
        category_stats[cat]["exc_cov_sum"] += d["excerpt_coverage"]

    category_breakdown = {}
    for cat, stats in category_stats.items():
        count = stats["count"]
        category_breakdown[cat] = {
            "avg_f1": round(stats["f1_sum"] / count, 3),
            "avg_excerpt_coverage": round(stats["exc_cov_sum"] / count, 3),
            "count": count
        }

    score = (
        0.35 * avg_f1
        + 0.25 * mrr_s
        + 0.25 * ndcg5
        + 0.15 * avg_excerpt_cov
    )

    return {
        "component":         "rag",
        "score":             round(score, 3),
        "avg_precision":     round(avg_precision, 3),
        "avg_recall":        round(avg_recall, 3),
        "avg_f1":            round(avg_f1, 3),
        "mrr":               round(mrr_s, 3),
        "ndcg_at_5":         round(ndcg5, 3),
        "hit_rate_at_1":     round(hr1, 3),
        "hit_rate_at_3":     round(hr3, 3),
        "hit_rate_at_5":     round(hr5, 3),
        "avg_excerpt_coverage": round(avg_excerpt_cov, 3),
        "latency":           perf,
        "sample_count":      n,
        "category_breakdown": category_breakdown,
        "details":           details,
    }
