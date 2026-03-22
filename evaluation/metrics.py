"""
evaluation/metrics.py
---------------------
Shared metric functions used by all eval modules.

Pure Python — zero external dependencies.

Includes:
  Legacy: word_error_rate, latency_percentiles, precision_recall, bleu1, keyword_overlap
  New:    char_error_rate, arabic_normalize, rouge_n, meteor_approx,
          mrr, ndcg_at_k, hit_rate_at_k, entity_slot_f1
"""

import math
import re
import unicodedata
from typing import Dict, List, Optional, Tuple


# ── Arabic normalisation ──────────────────────────────────────────────────────

_HAMZA_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا",
    "ة": "ه",
    "ى": "ي",
})

_DIACRITIC_RE = re.compile(r"[\u064B-\u065F\u0670]")  # harakat + tatweel


def arabic_normalize(text: str) -> str:
    """Strip Arabic diacritics and normalise common hamza/alef variants.

    Ensures that لُّغة and لغة score identically, and that خطأ vs خطا
    are treated the same way for fairer Arabic ASR/LLM comparison.
    """
    text = _DIACRITIC_RE.sub("", text)
    text = text.translate(_HAMZA_MAP)
    # Remove control characters like zero-width non-joiner, LRM, RLM which often plague PDF dumps
    text = re.sub(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]', '', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Whitespace + punctuation tokenizer supporting Arabic and Latin text."""
    text = re.sub(r"[^\w\u0600-\u06FF\s]", " ", text)
    return [t for t in text.split() if t]


# ── Word Error Rate ───────────────────────────────────────────────────────────

def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute WER using dynamic programming.

    WER = (S + D + I) / N  where:
      S = substitutions, D = deletions, I = insertions, N = reference length
    Returns 0.0 for identical strings, ≥1.0 for very poor hypotheses.
    """
    ref_tokens = _tokenize(arabic_normalize(reference))
    hyp_tokens = _tokenize(arabic_normalize(hypothesis))

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    n, m = len(ref_tokens), len(hyp_tokens)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[m] / n


# ── Character Error Rate ──────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[m]


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER).

    CER = Levenshtein(reference, hypothesis) / len(reference)
    Applied after Arabic normalisation.
    """
    ref = arabic_normalize(reference)
    hyp = arabic_normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


# ── ROUGE ─────────────────────────────────────────────────────────────────────

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Longest Common Subsequence length."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def _ngrams(tokens: List[str], n: int) -> Dict[Tuple, int]:
    grams: Dict[Tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i:i + n])
        grams[g] = grams.get(g, 0) + 1
    return grams


def rouge_n(reference: str, hypothesis: str, n: int = 1) -> Dict[str, float]:
    """Compute ROUGE-N precision, recall, and F1.

    Normalises Arabic text before tokenisation.
    n=1 → ROUGE-1, n=2 → ROUGE-2.
    """
    ref_tokens = _tokenize(arabic_normalize(reference))
    hyp_tokens = _tokenize(arabic_normalize(hypothesis))

    ref_grams = _ngrams(ref_tokens, n)
    hyp_grams = _ngrams(hyp_tokens, n)

    if not ref_grams or not hyp_grams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = sum(min(ref_grams.get(g, 0), hyp_grams.get(g, 0)) for g in hyp_grams)
    precision = overlap / sum(hyp_grams.values())
    recall = overlap / sum(ref_grams.values())
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute ROUGE-L (LCS-based) precision, recall, and F1."""
    ref_tokens = _tokenize(arabic_normalize(reference))
    hyp_tokens = _tokenize(arabic_normalize(hypothesis))

    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ── METEOR approximation ──────────────────────────────────────────────────────

def meteor_approx(reference: str, hypothesis: str) -> float:
    """Approximate METEOR score.

    Computes harmonic mean of unigram precision and recall (like F-mean weighted
    towards recall) with a brevity penalty, useful for Arabic medical text.
    No stemming — normalisation handles the most common Arabic morphological
    surface forms (hamza, diacritics).
    """
    ref_tokens = _tokenize(arabic_normalize(reference))
    hyp_tokens = _tokenize(arabic_normalize(hypothesis))

    if not ref_tokens or not hyp_tokens:
        return 0.0

    ref_counts: Dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    matched = 0
    for t in hyp_tokens:
        if ref_counts.get(t, 0) > 0:
            matched += 1
            ref_counts[t] -= 1

    precision = matched / len(hyp_tokens)
    recall = matched / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    # METEOR uses α=0.9 (weight heavily towards recall)
    alpha = 0.9
    f_mean = precision * recall / (alpha * precision + (1 - alpha) * recall)

    # Simple brevity approximation (penalise very short hypotheses)
    brevity = min(1.0, len(hyp_tokens) / len(ref_tokens))
    return f_mean * brevity


_COMMA       = "\u060c"          # Arabic comma used mid-sentence
_EN_COMMA    = ","               # English comma, sometimes used by LLMs
_TERMINATORS = set(".!?\u061f")

def _punctuation_score(text: str) -> float:
    """Score Egyptian-Arabic punctuation quality on [0, 1].
    
    Relaxed for conversational text (Ammya):
      - We don't enforce strict commas or full stops for short messages.
      - We lightly penalize extremely long run-on sentences (>200 chars).
      - We reward any recognizable termination (. ! ؟ ?).
    """
    if not text or len(text) < 30:
        return 1.0

    score = 0.5 # Base score for conversational text

    # Reward termination gracefully
    stripped = text.strip()
    if stripped and stripped[-1] in _TERMINATORS:
        score += 0.3
        
    # Reward presence of any internal pausing (comma, Dash, etc.) if it's longer
    if len(text) > 80:
        if _COMMA in text or _EN_COMMA in text or "،" in text or "-" in text:
            score += 0.2
    else:
        score += 0.2 # Free points if it's short enough to not need commas

    # Run-on penalty for exceedingly long sentences
    segs = re.split(r'[\u060c,.!?\u061f]', text)
    long_seg = sum(1 for s in segs if len(s.strip()) > 200)
    score -= (0.2 * long_seg)

    return min(1.0, max(0.0, score))


# ── Egyptian Arabic (Ammya) Metric ─────────────────────────────────────────────

_AMMYA_LEXICON = {
    "عشان", "علشان", "علاشان", 
    "كده", "كدا", 
    "إيه", "ايه",
    "فين",
    "مش", 
    "دلوقتي", "دلوقت",
    "بتاع", "بتاعة", "بتوع",
    "ازاي", "إزاي",
    "بكرة", "بكره",
    "طب",
    "اللي", "الي",
    "زي",
    "بقى", "بقا",
    "ده", "دي", "دول",
    "حاجة",
    "حد",
    "لسه",
    "عايز", "عاوز", "عايزة", "عاوزة",
    "ياريت", "يا ريت",
    "خالص",
    "برضه", "برده"
}

def egyptian_arabic_score(text: str) -> float:
    """Score the presence of Egyptian Arabic (Ammya) vs strict MSA.
    
    Checks for common Egyptian conversational dialect markers.
    Returns [0, 1] mapped up from density threshold.
    """
    if not text:
        return 0.0
        
    norm_text = arabic_normalize(text).lower()
    tokens = _tokenize(norm_text)
    
    if not tokens:
        return 0.0
        
    ammya_matches = sum(1 for t in tokens if t in _AMMYA_LEXICON)
    
    # We don't expect *every* word to be slang, 
    # roughly 1 slang word per 10-15 words is a solid conversational tone
    expected_matches = max(1, len(tokens) / 12)
    score = min(1.0, ammya_matches / expected_matches)
    
    # Boost if we see 'b-prefix' verbs (often conversational present continuous like بتشوف)
    if any(t.startswith("بت") or t.startswith("بش") or t.startswith("بي") for t in tokens if len(t) > 4):
        score = min(1.0, score + 0.15)
        
    return round(score, 3)


# ── Medical Referral Detection ───────────────────────────────────────────────

_REFERRAL_PHRASES = [
    "تواصل مع دكتور",
    "تواصل مع طبيب",
    "استشير دكتور",
    "استشمارة طبيب",
    "روح لدكتور",
    "اسأل دكتور",
    "اروح لدكتور",
    "اكلم دكتور",
    "راجع طبيبك",
    "راجع الدكتور",
    "كشف عند دكتور",
    "كشف عند طبيب",
    "زيارة الطبيب",
    "طوارئ اقرب مستشفى",
    "كلم الاسعاف",
]

def is_medical_referral(text: str) -> bool:
    """Check if the text primarily serves as a medical referral.
    
    Used to handle cases where the LLM safely refuses to give specific medical 
    advice and instead directs the user to professional help.
    """
    if not text:
        return False
        
    norm_text = arabic_normalize(text)
    # Check for direct phrases
    for phrase in _REFERRAL_PHRASES:
        if arabic_normalize(phrase) in norm_text:
            return True
            
    # Heuristic: if it refers to 'doctor' and 'consult' or 'contact' in various forms
    tokens = _tokenize(norm_text)
    has_doctor = any(t in ["دكتور", "دكتورة", "دكاترة", "طبيب", "طبيبة", "اطباء", "مستشفى", "اسعاف"] for t in tokens)
    has_action = any(t in ["كلم", "تواصل", "استشير", "اسال", "روح", "راجع", "كشف", "زيارة"] for t in tokens)
    
    return has_doctor and has_action

# ── RAG ranking metrics ───────────────────────────────────────────────────────

def mrr(relevant_lists: List[List[str]], retrieved_lists: List[List[str]]) -> float:
    """Mean Reciprocal Rank.

    Args:
        relevant_lists:  List of per-query lists of ground-truth relevant IDs.
        retrieved_lists: List of per-query ranked retrieved IDs (in rank order).
    """
    rr_sum = 0.0
    count = 0
    for relevant, retrieved in zip(relevant_lists, retrieved_lists):
        rel_set = set(relevant)
        for rank, doc in enumerate(retrieved, start=1):
            if doc in rel_set:
                rr_sum += 1.0 / rank
                break
        count += 1
    return rr_sum / count if count else 0.0


def ndcg_at_k(
    relevant_lists: List[List[str]],
    retrieved_lists: List[List[str]],
    k: int = 5,
) -> float:
    """Normalized Discounted Cumulative Gain @ k.

    Binary relevance (1 if in relevant set, 0 otherwise).
    """
    def _dcg(retrieved: List[str], rel_set: set, k: int) -> float:
        return sum(
            1.0 / math.log2(rank + 1)
            for rank, doc in enumerate(retrieved[:k], start=1)
            if doc in rel_set
        )

    ndcg_sum = 0.0
    count = 0
    for relevant, retrieved in zip(relevant_lists, retrieved_lists):
        rel_set = set(relevant)
        dcg = _dcg(retrieved, rel_set, k)
        ideal = _dcg(list(rel_set), rel_set, k)  # ideal ranking puts all relevant first
        ndcg_sum += (dcg / ideal) if ideal > 0 else 0.0
        count += 1
    return ndcg_sum / count if count else 0.0


def hit_rate_at_k(
    relevant_lists: List[List[str]],
    retrieved_lists: List[List[str]],
    k: int = 1,
) -> float:
    """Hit Rate @ k: fraction of queries where at least one relevant doc is in top-k."""
    hits = 0
    count = 0
    for relevant, retrieved in zip(relevant_lists, retrieved_lists):
        rel_set = set(relevant)
        if any(doc in rel_set for doc in retrieved[:k]):
            hits += 1
        count += 1
    return hits / count if count else 0.0


# ── Entity Slot F1 ────────────────────────────────────────────────────────────

def entity_slot_f1(
    expected: Dict[str, str],
    extracted: Dict[str, str],
) -> Dict[str, float]:
    """Compute slot-level F1 between expected and extracted entity dictionaries.

    Performs exact match after lowercasing and Arabic normalisation.
    Useful for MCP intent + entity extraction accuracy.

    Args:
        expected:  {"doctor": "أحمد", "clinic": "القلب"}
        extracted: {"doctor": "احمد"}  (may be incomplete or wrong)
    """
    if not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def _norm(val: str) -> str:
        return arabic_normalize(val).strip().lower()

    true_positive = sum(
        1 for k, v in expected.items()
        if extracted.get(k) is not None and _norm(extracted[k]) == _norm(v)
    )
    precision = true_positive / len(extracted) if extracted else 0.0
    recall = true_positive / len(expected)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ── Latency percentiles ───────────────────────────────────────────────────────

def latency_percentiles(latencies_ms: List[float]) -> dict:
    """Return p50, p95, p99 and mean latency in ms."""
    if not latencies_ms:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "count": 0}

    sorted_l = sorted(latencies_ms)
    n = len(sorted_l)

    def _percentile(p: float) -> float:
        idx = max(0, int(n * p / 100) - 1)
        return sorted_l[idx]

    return {
        "p50": _percentile(50),
        "p95": _percentile(95),
        "p99": _percentile(99),
        "mean": sum(sorted_l) / n,
        "count": n,
    }


# ── Precision / Recall ────────────────────────────────────────────────────────

def precision_recall(
    retrieved: List[str],
    relevant: List[str],
) -> dict:
    """Compute precision / recall / F1 for document retrieval."""
    if not retrieved:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    tp = len(retrieved_set & relevant_set)
    precision = tp / len(retrieved_set)
    recall = tp / len(relevant_set) if relevant_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ── BLEU-1 approximation (kept for backward compat) ──────────────────────────

def bleu1(reference: str, hypothesis: str) -> float:
    """Unigram BLEU score. Deprecated in favour of rouge_n(n=1)."""
    ref_tokens = set(_tokenize(reference.lower()))
    hyp_tokens = _tokenize(hypothesis.lower())
    if not hyp_tokens:
        return 0.0
    matched = sum(1 for t in hyp_tokens if t in ref_tokens)
    return matched / len(hyp_tokens)


# ── Keyword overlap (kept for backward compat) ────────────────────────────────

def keyword_overlap(expected_keywords: List[str], actual_text: str) -> float:
    """Fraction of expected keywords found in actual_text."""
    if not expected_keywords:
        return 1.0
    actual_lower = actual_text.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in actual_lower)
    return found / len(expected_keywords)
