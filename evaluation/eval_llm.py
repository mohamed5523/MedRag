"""
evaluation/eval_llm.py
----------------------
LLM-as-a-Judge evaluation for open-ended Arabic healthcare responses.

Pipeline per sample:
  1. Send question to backend (/api/chat/) → get actual_answer
  2. Send (question, expected_answer, actual_answer) to a configurable judge
     → receive scores (1-5) for Factual Correctness, Relevance, Completeness
  3. Compute deterministic keyword_match_rate from expected_keywords
  4. final_score = 0.70 * judge_score + 0.30 * keyword_match_rate

Judge is configurable:
  - Default: calls JUDGE_URL (OpenAI-compatible /v1/chat/completions)
  - Custom:  pass judge_fn=callable(question, actual, expected) → {"factual": x, "relevance": x, "completeness": x}
  - Both return floats in [1, 5]; the evaluator normalises to [0, 1]

Score weights: factual 0.45 | relevance 0.30 | completeness 0.25
"""

import json
import logging
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from .config import (
    BACKEND_URL, JUDGE_API_KEY, JUDGE_MODEL, JUDGE_URL,
    LLM_QUERIES, THRESHOLDS,
)
from .http_client import openai_chat, post_with_retry
from .metrics import arabic_normalize, latency_percentiles, _punctuation_score, egyptian_arabic_score, rouge_l, is_medical_referral

logger = logging.getLogger("eval.llm")

# ── Rubric weights ─────────────────────────────────────────────────────────────
_FACTUAL_W      = 0.45
_RELEVANCE_W    = 0.30
_COMPLETENESS_W = 0.25
_JUDGE_W        = 0.45    # blend: judge 45%, keywords 20%, ammya 10%, syntax 20%, punct 5%
_KEYWORD_W      = 0.20
_SYNTAX_W       = 0.20
_AMMYA_W        = 0.10
_PUNCT_W        = 0.05

# ── Judge prompt ───────────────────────────────────────────────────────────────
_JUDGE_SYSTEM = """You are a medical AI evaluation expert. Your task is to evaluate an AI assistant's answer to a medical question in Egyptian Arabic.

Score the answer on THREE criteria, each from 1 (very poor) to 5 (excellent):

1. **Factual Correctness** – Is the answer medically accurate? No hallucinations? 
   **Note:** If the question is sensitive or complex, directing the user to consult a doctor is NOT a failure; it is considered factually responsible and professional.
2. **Relevance** – Does the answer directly address the user's question?
3. **Completeness & Simplicity** – Does the answer cover the key points from the reference answer in a simple, direct, conversational Egyptian tone?
   **Note:** If the AI Answer suggests seeing a doctor instead of providing specific advice from the reference answer, score it highly (4-5) if the referral is appropriate for the situation.

Respond ONLY in this exact JSON format (no extra text):
{"factual": <1-5>, "relevance": <1-5>, "completeness": <1-5>, "reasoning": "<one sentence>"}"""

_JUDGE_USER = """**Question:** {question}

**Reference Answer (ground truth):** {expected}

**AI Answer to evaluate:** {actual}

Score the AI Answer."""

_JSON_RE = re.compile(r'\{[^{}]+\}', re.DOTALL)


# ── Judge implementations ──────────────────────────────────────────────────────

def _call_openai_judge(
    question: str,
    actual: str,
    expected: str,
) -> Dict[str, Any]:
    """Call the configured OpenAI-compatible judge endpoint."""
    if not JUDGE_API_KEY:
        logger.warning("JUDGE_API_KEY not set — skipping judge, returning neutral 3/5")
        return {"factual": 3.0, "relevance": 3.0, "completeness": 3.0,
                "reasoning": "Judge API key not configured", "error": "no_api_key"}

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user",   "content": _JUDGE_USER.format(
            question=question, expected=expected, actual=actual)},
    ]
    resp = openai_chat(JUDGE_URL, JUDGE_API_KEY, JUDGE_MODEL, messages, temperature=0.0)
    if not resp["ok"]:
        logger.error("Judge call failed: %s", resp["error"])
        return {"factual": 0.0, "relevance": 0.0, "completeness": 0.0,
                "reasoning": resp["error"], "error": resp["error"]}

    raw_content = resp["json"]["choices"][0]["message"]["content"]
    m = _JSON_RE.search(raw_content)
    if not m:
        logger.error("Judge returned unparseable content: %.200s", raw_content)
        return {"factual": 0.0, "relevance": 0.0, "completeness": 0.0,
                "reasoning": "parse_error", "error": "parse_error"}

    try:
        parsed = json.loads(m.group())
        return {
            "factual":       float(parsed.get("factual",      0)),
            "relevance":     float(parsed.get("relevance",    0)),
            "completeness":  float(parsed.get("completeness", 0)),
            "reasoning":     parsed.get("reasoning", ""),
        }
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Judge JSON parse error: %s | raw=%.100s", exc, raw_content)
        return {"factual": 0.0, "relevance": 0.0, "completeness": 0.0,
                "reasoning": str(exc), "error": str(exc)}


def _judge_score_to_float(scores: Dict[str, Any]) -> float:
    """Weighted combination of 3 judge scores, normalised to [0, 1]."""
    f = scores.get("factual",      0) / 5.0
    r = scores.get("relevance",    0) / 5.0
    c = scores.get("completeness", 0) / 5.0
    return _FACTUAL_W * f + _RELEVANCE_W * r + _COMPLETENESS_W * c


# ── Keyword match ──────────────────────────────────────────────────────────────

def _keyword_match_rate(keywords: List[str], answer: str) -> float:
    """Deterministic: fraction of expected_keywords found in the answer (after Arabic normalise)."""
    if not keywords:
        return 1.0
    norm = arabic_normalize(answer).lower()
    found = sum(1 for kw in keywords if arabic_normalize(kw).lower() in norm)
    return found / len(keywords)


# ── Backend call ───────────────────────────────────────────────────────────────

def _get_actual_answer(question: str) -> Dict[str, Any]:
    resp = post_with_retry(
        "/api/chat/query",
        json={"query": question},
    )
    if not resp["ok"]:
        return {"answer": "", "latency_ms": resp["latency_ms"],
                "success": False, "error": resp["error"]}
    text = (resp["json"] or {}).get("answer", "")
    return {"answer": text, "latency_ms": resp["latency_ms"], "success": True}


# We now use _punctuation_score from metrics.py


# ── Main eval ──────────────────────────────────────────────────────────────────

def run_eval(
    judge_fn: Optional[Callable[[str, str, str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run LLM-as-a-Judge evaluation.

    Args:
        judge_fn: Optional callable(question, actual_answer, expected_answer) → dict
                  with keys: factual (1-5), relevance (1-5), completeness (1-5).
                  If None, uses the configured JUDGE_URL / JUDGE_API_KEY.
    """
    samples: List[dict] = []
    try:
        with open(LLM_QUERIES) as f:
            samples = json.load(f)
    except Exception as e:
        return {"component": "llm", "error": str(e), "score": 0.0}

    _judge_impl = judge_fn or _call_openai_judge

    final_scores:   List[float] = []
    judge_scores:   List[float] = []
    keyword_rates:  List[float] = []
    latencies:      List[float] = []
    cat_scores:     Dict[str, List[float]] = defaultdict(list)
    details = []

    for sample in samples:
        question = sample.get("question", "")
        expected = sample.get("expected_answer", "")
        keywords = sample.get("expected_keywords", [])
        category = sample.get("category", "general")

        if not question or not expected:
            continue

        # 1. Get actual answer from backend
        backend_result = _get_actual_answer(question)
        actual = backend_result.get("answer", "")
        latencies.append(backend_result.get("latency_ms", 0))

        if not backend_result["success"] or not actual:
            # Backend failed — score this sample as 0
            details.append({
                "question":       question[:100],
                "category":       category,
                "actual_preview": "",
                "judge_scores":   {},
                "keyword_rate":   0.0,
                "final_score":    0.0,
                "error":          backend_result.get("error", "empty answer"),
            })
            final_scores.append(0.0)
            judge_scores.append(0.0)
            keyword_rates.append(0.0)
            cat_scores[category].append(0.0)
            continue

        # 2. Judge call
        j_raw    = _judge_impl(question, actual, expected)
        j_norm   = _judge_score_to_float(j_raw)

        # 3. Keyword match
        kw_rate = _keyword_match_rate(keywords, actual)

        # 4. Punctuation quality (Egyptian Arabic)
        punct   = _punctuation_score(actual)
        
        # 4b. Egyptian Ammya Match
        ammya   = egyptian_arabic_score(actual)

        # 4c. Syntactic text overlap (ROUGE-L F1)
        syntax  = rouge_l(expected, actual)["f1"]

        # 4d. Medical Referral Check - If it's a referral, we adjust scores
        is_referral = is_medical_referral(actual)
        if is_referral:
            # If the model wisely refers to a doctor, we don't penalize for missing specific advice keywords
            # or having low syntax overlap with the (possibly detailed) ground truth.
            kw_rate = max(kw_rate, 0.8) # Base success for being safe
            syntax  = max(syntax, 0.5)  # Don't penalize short referrals

        # 5. Combined score
        combined = _JUDGE_W * j_norm + _KEYWORD_W * kw_rate + _SYNTAX_W * syntax + _AMMYA_W * ammya + _PUNCT_W * punct

        final_scores.append(combined)
        judge_scores.append(j_norm)
        keyword_rates.append(kw_rate)
        cat_scores[category].append(combined)

        details.append({
            "question":         question[:100],
            "category":         category,
            "difficulty":       sample.get("difficulty", ""),
            "actual_preview":   actual[:120],
            "judge_scores": {
                "factual":       round(j_raw.get("factual",      0), 2),
                "relevance":     round(j_raw.get("relevance",    0), 2),
                "completeness":  round(j_raw.get("completeness", 0), 2),
                "reasoning":     j_raw.get("reasoning", ""),
            },
            "keyword_rate":     round(kw_rate, 3),
            "syntax_score":     round(syntax,  3),
            "ammya_score":      round(ammya,   3),
            "punctuation_score": round(punct,  3),
            "judge_norm":       round(j_norm,  3),
            "final_score":      round(combined, 3),
            "latency_ms":       round(backend_result["latency_ms"], 1),
        })

    n = len(final_scores)
    if n == 0:
        return {"component": "llm", "score": 0.0, "sample_count": 0, "details": []}

    avg_score    = sum(final_scores)  / n
    avg_judge    = sum(judge_scores)  / n
    avg_keyword  = sum(keyword_rates) / n
    perf         = latency_percentiles(latencies)
    per_category = {cat: round(sum(v)/len(v), 3) for cat, v in cat_scores.items()}

    return {
        "component":          "llm",
        "score":              round(avg_score,   3),
        "avg_judge_score":    round(avg_judge,   3),
        "avg_keyword_rate":   round(avg_keyword, 3),
        "avg_syntax_score":   round(sum(d.get("syntax_score", 0) for d in details) / n if n else 0, 3),
        "avg_ammya_score":    round(sum(d.get("ammya_score", 0) for d in details) / n if n else 0, 3),
        "avg_punct_score":    round(sum(d.get("punctuation_score", 0) for d in details) / n if n else 0, 3),
        "per_category":       per_category,
        "latency":            perf,
        "sample_count":       n,
        "details":            details,
    }
