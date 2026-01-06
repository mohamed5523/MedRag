"""Hybrid doctor name matching engine with Arabic/English support.

This module provides normalization, tokenization, and scoring functions
for matching doctor names with high accuracy using positional weights
and fuzzy matching.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from rapidfuzz import fuzz

# ------------------------------------------------------------------------------
# Normalization & Tokenization
# ------------------------------------------------------------------------------

ARABIC_TASHKEEL_PATTERN = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
ARABIC_TATWEEL_PATTERN = re.compile(r"[\u0640]")

TITLE_STOPWORDS = {
    "دكتور", "دكتورة", "د.", "د", "د/‏", "د/‏.",
    "دكتور.", "دكتوره",
    "dr", "dr.", "doctor",
}


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text by removing diacritics and standardizing characters."""
    if not text:
        return ""
    text = ARABIC_TASHKEEL_PATTERN.sub("", text)
    text = ARABIC_TATWEEL_PATTERN.sub("", text)
    text = re.sub("[إأآا]", "ا", text)  # alef variants
    # Normalize common hamza variants for fuzzy matching (helps noisy user input)
    # Note: keep this conservative; it improves recall for both doctor/clinic names.
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = text.replace("ء", "")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_english(text: str) -> str:
    """Normalize English text to lowercase with only alphanumeric characters."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_mixed_text(text: str) -> str:
    """Normalize mixed Arabic/English text for comparison."""
    text = text or ""
    text = text.strip()
    text = normalize_arabic(text)
    text = text.lower()
    # Remove common Arabic punctuation that sits inside the Arabic unicode block
    # (so it would otherwise survive the regex below).
    text = text.replace("؟", " ").replace("،", " ").replace("؛", " ").replace("ـ", " ")
    # keep Arabic range and word characters
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_name(text: str) -> List[str]:
    """Tokenize a name into meaningful tokens, removing stopwords."""
    norm = normalize_mixed_text(text)
    tokens = norm.split()
    return [
        t for t in tokens
        if len(t) > 1 and t not in TITLE_STOPWORDS
    ]


CLINIC_STOPWORDS = {
    # Arabic
    "عيادة",
    "عياده",
    "عيادات",
    "قسم",
    "مركز",
    "مجمع",
    # Very common glue words
    "و",
    "في",
    "ب",
    "بال",
    "ال",
    # Common date/time words that often appear in clinic queries but are NOT part of clinic names
    "النهارده",
    "نهارده",
    "اليوم",
    "بكره",
    "بكرة",
    "غدا",
    "غداً",
    # Common query words (if the upstream extraction accidentally passes whole question)
    "مين",
    "موجود",
    "موجودين",
    "متاح",
    "مواعيد",
    "سعر",
    "كشف",
    # English
    "clinic",
    "department",
    "center",
}


def tokenize_clinic(text: str) -> List[str]:
    """
    Tokenize a clinic phrase robustly (Arabic/English), removing generic words and
    normalizing common variants.

    Key behaviors:
    - Removes "عيادة/قسم/مركز/clinic" and similar generic tokens
    - Strips leading "ال"
    - Handles the Arabic conjunction "و" both standalone ("و توليد") and attached ("وتوليد")
    - Leverages normalize_mixed_text() which calls normalize_arabic() (incl. hamza folding)
    """
    norm = normalize_mixed_text(text)
    if not norm:
        return []

    raw_tokens = norm.split()
    tokens: List[str] = []
    for tok in raw_tokens:
        if tok in CLINIC_STOPWORDS:
            continue
        # Strip Arabic definite article
        if tok.startswith("ال") and len(tok) > 3:
            tok = tok[2:]
        # Drop standalone "و" and strip attached "و" when it's likely a conjunction
        if tok == "و":
            continue
        if tok.startswith("و") and len(tok) > 3:
            tok = tok[1:]
        if len(tok) > 1 and tok not in CLINIC_STOPWORDS:
            tokens.append(tok)

    # Preserve order but de-dupe
    return list(dict.fromkeys(tokens))


# ------------------------------------------------------------------------------
# Positional & Scoring Helpers
# ------------------------------------------------------------------------------

def compute_positional_token_weight(
    query_token: str,
    doctor_tokens: List[str],
    similarity_threshold: float = 0.80,
) -> Tuple[float, bool]:
    """
    Compute weighted score based on token position and similarity.
    
    Returns (weighted_score, matched_as_first_name).

    weighted_score ∈ [0, 1], combining:
      - fuzzy similarity with a doctor token
      - positional weight (first > father > middle > last)

    matched_as_first_name = True if best match was at index 0.
    """
    if not doctor_tokens:
        return 0.0, False

    # Positional weights (can tweak)
    # 0 = first name (highest), 1 = father, middle = default, last = lowest
    FIRST_NAME_WEIGHT = 1.0
    FATHER_NAME_WEIGHT = 0.7
    MIDDLE_NAME_WEIGHT = 0.5
    LAST_NAME_WEIGHT = 0.3

    best_weighted = 0.0
    matched_first = False
    last_idx = len(doctor_tokens) - 1

    for idx, dt in enumerate(doctor_tokens):
        sim = fuzz.ratio(query_token, dt) / 100.0
        if sim < similarity_threshold:
            continue

        # Determine positional weight based on index
        if idx == 0:
            pos_weight = FIRST_NAME_WEIGHT
        elif idx == 1:
            pos_weight = FATHER_NAME_WEIGHT
        elif idx == last_idx and last_idx > 1:
            pos_weight = LAST_NAME_WEIGHT
        else:
            pos_weight = MIDDLE_NAME_WEIGHT

        weighted = sim * pos_weight

        if weighted > best_weighted:
            best_weighted = weighted
            matched_first = (idx == 0)

    return best_weighted, matched_first


def compute_order_score(query_tokens: List[str], doctor_tokens: List[str]) -> float:
    """
    Measure how well the sequence of query tokens matches the doctor's tokens.
    
    Returns:
        - 1.0 = exact prefix (e.g. query = ['mina','shawky'], doctor = ['mina','shawky','...'])
        - 0.85 = ordered subsequence (same order, but not prefix)
        - 0.6 = reversed sequence appears in order
        - 0.0 = no ordered match
    """
    if len(query_tokens) < 2 or len(doctor_tokens) == 0:
        return 0.0

    # Exact prefix
    if doctor_tokens[:len(query_tokens)] == query_tokens:
        return 1.0

    # Ordered subsequence
    try:
        positions = [doctor_tokens.index(t) for t in query_tokens]
        if positions == sorted(positions):
            return 0.85
    except ValueError:
        pass

    # Reversed order (user wrote last/first)
    reversed_q = list(reversed(query_tokens))
    try:
        positions = [doctor_tokens.index(t) for t in reversed_q]
        if positions == sorted(positions):
            return 0.6
    except ValueError:
        pass

    return 0.0

