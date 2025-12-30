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


def format_provider_disambiguation_prompt(candidates: list[dict[str, Any]]) -> str:
    """
    Build a user-facing prompt for provider disambiguation.

    UX rule:
    - 0 candidates: ask user to provide full name.
    - 1 candidate: ask for confirmation (don't claim ambiguity).
    - 2+ candidates: show numbered list.
    """
    names: list[str] = []
    for c in candidates[:5]:
        if not isinstance(c, dict):
            continue
        name = (str(c.get("name_ar") or c.get("name_en") or "")).strip()
        if name:
            names.append(name)

    if not names:
        return "ممكن تكتب الاسم كامل عشان أقدر أحدد الدكتور صح؟"

    if len(names) == 1:
        n = names[0]
        return (
            f"هل تقصد دكتور {n}؟ للتأكيد ابعت رقم 1 أو ١، "
            "أو اكتب الاسم بالكامل بشكل صحيح."
        )

    parts = ["فيه أكتر من دكتور بنفس الاسم. اختار رقم من دول"]
    for i, n in enumerate(names, start=1):
        parts.append(f"{i} {n}")
    parts.append("اكتب رقم الاختيار أو اكتب الاسم كامل")
    return " ".join(parts).strip()


def materialize_intent_query(
    intent: Optional[str],
    *,
    doctor_name: Optional[str] = None,
    clinic_name: Optional[str] = None,
) -> str:
    """Build a stable Arabic query string from intent + entities."""
    doctor = (doctor_name or "").strip()
    clinic = (clinic_name or "").strip()
    i = (intent or "").strip()

    if i == "ask_price":
        if doctor and clinic:
            return f"سعر الكشف عند الدكتور {doctor} في {clinic}"
        if doctor:
            return f"سعر الكشف عند الدكتور {doctor}"
        if clinic:
            return f"سعر الكشف في {clinic}"
        return "سعر الكشف"

    if i in {"check_availability", "book_appointment"}:
        if doctor and clinic:
            return f"مواعيد الدكتور {doctor} في {clinic}"
        if doctor:
            return f"مواعيد الدكتور {doctor}"
        if clinic:
            return f"مواعيد {clinic}"
        return "المواعيد"

    if i == "list_doctors":
        if clinic:
            return f"اسماء الأطباء في {clinic}"
        return "اسماء الأطباء"

    if doctor:
        return f"{i} عند الدكتور {doctor}"
    return i or "استفسار"


def apply_pending_action_resolution(
    pending_action_type: Optional[str],
    resolution: dict[str, Any],
    request_query: str,
) -> tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    str,
]:
    """
    Convert a resolved pending_action into forced intent/entity overrides and a stable state_input_query.
    """
    forced_intent = str(resolution.get("intent") or "").strip() or None
    forced_doctor: Optional[str] = None
    forced_clinic: Optional[str] = None
    forced_specialty: Optional[str] = None
    state_input_query = request_query

    if pending_action_type == "provider_disambiguation":
        selected = resolution.get("selected") or {}
        if isinstance(selected, dict):
            forced_doctor = (selected.get("name_ar") or selected.get("name_en") or "").strip() or None
            forced_clinic = (selected.get("clinic_name") or "").strip() or None
        if forced_intent and forced_doctor:
            state_input_query = materialize_intent_query(
                forced_intent, doctor_name=forced_doctor, clinic_name=forced_clinic
            )

    elif pending_action_type == "symptom_triage":
        forced_intent = "list_doctors"
        forced_specialty = str(resolution.get("specialty") or "").strip() or None
        state_input_query = str(resolution.get("combined_query") or request_query)

    return forced_intent, forced_doctor, forced_clinic, forced_specialty, state_input_query


