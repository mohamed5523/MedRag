from __future__ import annotations

import re
from typing import Any, Optional

# Support both Arabic-Indic digits (٠١٢...) and Eastern Arabic/Persian digits (۰۱۲...).
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


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
            if action_type == "clinic_disambiguation":
                names.append(
                    str(
                        cand.get("clinic_name")
                        or cand.get("name_ar")
                        or cand.get("name_en")
                        or ""
                    )
                )
            else:
                names.append(str(cand.get("name_ar") or cand.get("name_en") or ""))
        else:
            names.append(str(cand))

    idx = parse_candidate_selection(user_text, names)
    if idx is None:
        return None

    return {
        "intent": pending_action.get("intent"),
        "selected": candidates[idx],
        # Keep the original question so that after disambiguation we can continue
        # answering the *full* user request (e.g., "مواعيد + سعر") instead of only
        # materializing a single-intent query like "مواعيد ...".
        "original_question": pending_action.get("original_question"),
    }


def should_abandon_pending_action(user_text: str, pending_action: dict[str, Any]) -> bool:
    """
    Return True when the user reply looks like a NEW request (new doctor/clinic/intent),
    not a selection for the current pending_action.

    Goal: avoid "stuck turn" loops where user changes intent mid-disambiguation.
    """
    if not pending_action:
        return False

    action_type = str(pending_action.get("type") or "").strip()
    if action_type not in {
        "provider_disambiguation",
        "provider_clinic_mismatch",
        "clinic_disambiguation",
    }:
        return False

    t = _normalize_arabic_text(user_text)
    if not t:
        return False

    # If the user typed a number, treat it as an explicit selection attempt.
    if re.search(r"\b(\d{1,2})\b", t):
        return False

    # If they mention a new doctor/clinic/intent phrase with enough content, abandon.
    triggers = [
        # entities
        "دكتور",
        "د",
        "عياده",
        "عيادة",
        "عيادات",
        # intents
        "مواعيد",
        "سعر",
        "كشف",
        "احجز",
        "حجز",
        "متاح",
        # conversational overrides
        "اقصد",
        "قصد",
        "مش",
        "لا",
        "لأ",
        "تمام",
    ]
    if any(k in t for k in triggers) and len(t.split()) >= 3:
        return True

    return False


def _explicit_doctor_mention(text: str) -> bool:
    """
    Detect explicit doctor mention in Arabic/English.
    We require a token boundary to avoid matching inside other words.
    """
    t = _normalize_arabic_text(text)
    if not t:
        return False
    return bool(re.search(r"(?:^|\s)(?:دكتور|د\.?|dr|doctor)(?:\s|$)", t, flags=re.IGNORECASE))


def _explicit_clinic_mention(text: str) -> bool:
    """
    Detect explicit clinic mention, but treat phrases like 'عيادة دكتور X' as a doctor mention
    (not a clinic mention), since users often mean 'the doctor's clinic'.
    """
    t = _normalize_arabic_text(text)
    if not t:
        return False
    # If "عيادة" is immediately followed by a doctor marker, don't treat it as a clinic mention.
    if re.search(r"(?:^|\s)عياد[هة]\s+(?:دكتور|د\.?|dr|doctor)(?:\s|$)", t, flags=re.IGNORECASE):
        return False
    return bool(re.search(r"(?:^|\s)عياد[هة](?:\s|$)", t))


def _extract_clinic_phrase(text: str) -> Optional[str]:
    """
    Extract a clinic phrase from user text, e.g.:
      - "مين موجود في عيادة الاسنان النهارده" -> "عيادة الاسنان"
      - "مواعيد عيادة المسا و التوليد" -> "عيادة المسا و التوليد"
    """
    t = _normalize_arabic_text(text)
    if not t:
        return None

    # Avoid treating "عيادة دكتور X" as a clinic phrase.
    if re.search(r"(?:^|\s)عياد[هة]\s+(?:دكتور|د\.?|dr|doctor)(?:\s|$)", t, flags=re.IGNORECASE):
        return None

    m = re.search(r"(?:^|\s)عياد[هة]\s+(.+)", t, flags=re.IGNORECASE)
    if not m:
        return None

    candidate = m.group(1).strip()
    # Stop at punctuation
    candidate = re.split(r"[?.,!،]", candidate)[0].strip()
    # Stop at common date/time trailing words (they are not part of clinic names)
    candidate = re.split(r"\b(النهارده|نهارده|اليوم|بكره|بكرة|غدا|غداً)\b", candidate)[0].strip()

    words = [w for w in candidate.split() if w]
    if not words:
        return None

    # Limit phrase length to avoid swallowing the whole question.
    words = words[:6]
    phrase = "عيادة " + " ".join(words)
    return phrase.strip() or None


def apply_context_switch_rules(query_text: str, entities: Any) -> None:
    """
    Deterministic post-processing to avoid 'stuck' context when users change intent mid-chat.

    Rules:
    - If user explicitly mentions a doctor but does NOT explicitly mention a clinic,
      clear clinic context (clinic + clinic_id).
    - If user explicitly mentions a clinic but does NOT explicitly mention a doctor,
      clear doctor context (doctor + provider_id).
    - If user asks a generic question with NO entity mentions at all (e.g., "مين الدكاترة؟"),
      clear stale doctor/provider_id so the query isn't constrained to an old context.
    """
    has_doc = _explicit_doctor_mention(query_text)
    has_clinic = _explicit_clinic_mention(query_text)

    if has_doc and not has_clinic:
        if hasattr(entities, "clinic"):
            entities.clinic = None
        if hasattr(entities, "clinic_id"):
            entities.clinic_id = None

    elif has_clinic and not has_doc:
        # If user explicitly mentioned a clinic this turn, overwrite any stale clinic from previous state.
        extracted = _extract_clinic_phrase(query_text)
        if extracted and hasattr(entities, "clinic"):
            entities.clinic = extracted
        # New clinic mention should invalidate a previously resolved clinic_id.
        if hasattr(entities, "clinic_id"):
            entities.clinic_id = None
        if hasattr(entities, "doctor"):
            entities.doctor = None
        if hasattr(entities, "provider_id"):
            entities.provider_id = None

    elif not has_doc and not has_clinic:
        # Generic query with no entity mentions at all.
        # Clear stale doctor/provider_id to avoid constraining new queries.
        # Keep clinic context if it was set (user might still be in same clinic scope).
        _generic_intent_keywords = {
            "مين", "الدكاترة", "دكاترة", "اسماء", "موجود", "موجودين",
            "النهارده", "بكره", "المواعيد", "عايز", "عايزه", "محتاج",
        }
        query_lower = query_text.strip()
        if any(kw in query_lower for kw in _generic_intent_keywords):
            if hasattr(entities, "doctor"):
                entities.doctor = None
            if hasattr(entities, "provider_id"):
                entities.provider_id = None


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


def format_clinic_disambiguation_prompt(candidates: list[dict[str, Any]]) -> str:
    """
    Build a user-facing prompt for clinic disambiguation.

    Mirrors provider disambiguation UX but uses clinic names.
    """
    names: list[str] = []
    for c in candidates[:5]:
        if not isinstance(c, dict):
            continue
        name = (str(c.get("clinic_name") or c.get("name_ar") or c.get("name_en") or "")).strip()
        if name:
            names.append(name)

    if not names:
        return "ممكن تكتب اسم العيادة كامل عشان أقدر أحددها صح؟"

    if len(names) == 1:
        n = names[0]
        return (
            f"هل تقصد عيادة {n}؟ للتأكيد ابعت رقم 1 أو ١، "
            "أو اكتب الاسم بالكامل بشكل صحيح."
        )

    parts = ["فيه أكتر من عيادة بنفس الاسم. اختار رقم من دول"]
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
    original_question = str(resolution.get("original_question") or "").strip() or None

    if pending_action_type in {"provider_disambiguation", "provider_clinic_mismatch"}:
        selected = resolution.get("selected") or {}
        if isinstance(selected, dict):
            forced_doctor = (selected.get("name_ar") or selected.get("name_en") or "").strip() or None
            forced_clinic = (selected.get("clinic_name") or "").strip() or None
        if original_question:
            state_input_query = original_question
        elif forced_intent and forced_doctor:
            state_input_query = materialize_intent_query(
                forced_intent, doctor_name=forced_doctor, clinic_name=forced_clinic
            )

    elif pending_action_type == "clinic_disambiguation":
        selected = resolution.get("selected") or {}
        if isinstance(selected, dict):
            forced_clinic = (
                (selected.get("clinic_name") or selected.get("name_ar") or selected.get("name_en") or "").strip()
                or None
            )
        if original_question:
            state_input_query = original_question
        elif forced_intent and forced_clinic:
            state_input_query = materialize_intent_query(forced_intent, clinic_name=forced_clinic)

    elif pending_action_type == "symptom_triage":
        forced_intent = "list_doctors"
        forced_specialty = str(resolution.get("specialty") or "").strip() or None
        state_input_query = str(resolution.get("combined_query") or request_query)

    return forced_intent, forced_doctor, forced_clinic, forced_specialty, state_input_query


