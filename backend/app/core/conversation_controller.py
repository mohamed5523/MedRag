from __future__ import annotations

import re
from typing import Any, Optional


_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _normalize_arabic_text(text: str) -> str:
    """Lightweight Arabic normalization for matching user replies to candidates."""
    if not text:
        return ""
    t = text.translate(_ARABIC_DIGITS)
    # remove tatweel and common diacritics
    t = re.sub(r"[\u0640\u064B-\u0652\u0670]", "", t)
    # normalize alef variants and yaa/taa marbuta variants
    t = (
        t.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )
    # strip punctuation/symbols (keep letters/numbers/spaces)
    t = re.sub(r"[^\w\u0600-\u06FF\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_candidate_selection(user_text: str, candidates: list[str]) -> Optional[int]:
    """
    Parse user selection for a numbered disambiguation list.

    Returns 0-based index into candidates, or None if not resolvable.
    Supports:
    - Arabic/Latin numerals ("2" or "٢")
    - Fuzzy/partial name match against candidate strings.
    """
    if not user_text or not candidates:
        return None

    normalized = _normalize_arabic_text(user_text)

    # 1) Digit selection (first number found)
    m = re.search(r"\b(\d{1,2})\b", normalized)
    if m:
        try:
            choice = int(m.group(1))
        except ValueError:
            choice = None
        if choice is not None and 1 <= choice <= len(candidates):
            return choice - 1

    # 2) Fuzzy partial-name match (token overlap)
    user_tokens = [tok for tok in normalized.split() if tok]
    if not user_tokens:
        return None

    best_idx: Optional[int] = None
    best_ratio = 0.0

    for idx, cand in enumerate(candidates):
        cand_norm = _normalize_arabic_text(cand)
        cand_tokens = set(cand_norm.split())
        if not cand_tokens:
            continue

        overlap = sum(1 for tok in user_tokens if tok in cand_tokens)
        ratio = overlap / max(1, len(user_tokens))

        # strong match: all user tokens contained in candidate tokens
        if overlap == len(user_tokens):
            return idx

        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx

    # accept if user tokens mostly match a single candidate
    if best_idx is not None and best_ratio >= 0.6:
        return best_idx

    return None


def is_symptom_triage_request(query: str) -> bool:
    """
    Detect 'I have symptoms, who should I see?' requests.
    This is used to route symptom triage to MCP instead of RAG.
    """
    t = _normalize_arabic_text(query)
    if not t:
        return False

    go_who_patterns = [
        "اروح لمين",
        "اروح لدكتور مين",
        "اكشف عند مين",
        "اكشف لمين",
        "اروح لايه دكتور",
        "اروح لاي دكتور",
    ]
    symptom_hints = [
        "وجع",
        "الم",
        "الم",
        "تعبان",
        "تعبانه",
        "سخونيه",
        "سخونه",
        "دوخه",
        "قيء",
        "اسهال",
        "مغص",
        "بطن",
        "بطني",
        "صداع",
    ]

    has_go_who = any(p in t for p in go_who_patterns)
    has_symptom = any(h in t for h in symptom_hints)
    return has_go_who and has_symptom


def infer_specialty_from_symptoms(query: str) -> Optional[str]:
    """
    Rules-first symptom → specialty mapping.
    This should stay conservative; if uncertain, return None and ask follow-up.
    """
    t = _normalize_arabic_text(query)
    if not t:
        return None

    # Abdominal/GI
    if any(k in t for k in ["بطني", "بطن", "مغص", "معده", "معدة", "قولون"]):
        return "باطنة"

    # Pregnancy / OB-GYN
    if any(k in t for k in ["حمل", "حامل", "دوره", "دورة", "نزيف", "افرازات"]):
        return "نسا وتوليد"

    # Chest / cardiac
    if any(k in t for k in ["صدر", "ضيق نفس", "خفقان"]):
        return "قلب"

    return None


def resolve_pending_action(user_text: str, pending_action: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Resolve a stored pending_action using the user's current reply.

    Returns a dict like:
      {"intent": <original_intent>, "selected": <candidate_dict>}
    or None if not resolved.
    """
    if not pending_action:
        return None

    action_type = pending_action.get("type")

    # Symptom triage follow-up: user replies with symptom details after we asked a question
    if action_type == "symptom_triage":
        original = str(pending_action.get("original_question") or "")
        combined = f"{original} {user_text}".strip()
        specialty = infer_specialty_from_symptoms(combined)
        if not specialty:
            return None
        return {
            "intent": pending_action.get("intent") or "list_doctors",
            "specialty": specialty,
            "combined_query": combined,
        }

    # Provider/clinic disambiguation selection
    candidates = pending_action.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return None

    names: list[str] = []
    for cand in candidates:
        if isinstance(cand, dict):
            names.append(str(cand.get("name_ar") or cand.get("name_en") or ""))
        else:
            names.append(str(cand))

    idx = parse_candidate_selection(user_text, names)
    if idx is None:
        return None

    return {
        "intent": pending_action.get("intent"),
        "selected": candidates[idx],
    }


