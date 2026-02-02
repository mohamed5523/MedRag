"""
Integration tests for the MCP matching endpoints using real doctor/clinic names
from the frozen snapshot: heal-query-hub/DB_doctor_clinic_names.json

Goal: validate matching behavior for:
- Exact matches
- Common misspellings (1-char edits like "عبده" -> "عبدو")
- Ambiguity (multiple candidates)

These tests call the running MCP server over HTTP. If the MCP server is not reachable,
the tests are skipped (so unit-only CI environments don't fail).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pytest


pytestmark = pytest.mark.integration


DB_PATH = Path(__file__).resolve().parents[2] / "DB_doctor_clinic_names.json"


def _load_db_snapshot() -> list[dict[str, Any]]:
    payload = json.loads(DB_PATH.read_text(encoding="utf-8"))
    assert payload.get("success") is True
    data = payload.get("data")
    assert isinstance(data, list) and data, "DB snapshot 'data' must be a non-empty list"
    return data


def _candidate_base_urls() -> list[str]:
    # Prefer explicit overrides.
    env_first: list[str] = []
    for k in ("MCP_MATCH_BASE_URL", "MCP_BASE_URL"):
        v = (os.getenv(k) or "").strip()
        if v:
            env_first.append(v.rstrip("/"))

    # Common defaults:
    # - docker-compose publishes MCP server on host:8020
    # - inside compose network, mcp-server:8000
    defaults = [
        "http://localhost:8020",
        "http://127.0.0.1:8020",
        "http://mcp-server:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    # Preserve order but de-dup.
    seen: set[str] = set()
    out: list[str] = []
    for u in (env_first + defaults):
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


@pytest.fixture(scope="session")
def mcp_base_url() -> str:
    """
    Return a reachable MCP base URL. Skip if none of the common URLs respond.
    """
    urls = _candidate_base_urls()
    last_err: Optional[Exception] = None

    for base in urls:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{base}/providers")
                if r.status_code == 200:
                    return base
        except Exception as exc:  # pragma: no cover - depends on environment
            last_err = exc
            continue

    msg = (
        "MCP server is not reachable; skipping integration matching tests. "
        f"Tried: {urls}. "
        "Set MCP_MATCH_BASE_URL or MCP_BASE_URL to your running MCP endpoint."
    )
    if last_err:
        msg += f" Last error: {type(last_err).__name__}: {last_err}"
    pytest.skip(msg)


@pytest.fixture(scope="session")
def db_snapshot(mcp_base_url: str) -> list[dict[str, Any]]:
    """
    Prefer the frozen snapshot file when available.

    If tests are executed from a copied location (e.g., /tmp/tests inside a container),
    the JSON file might not be present. In that case, fall back to the live provider list
    fetched from the MCP server (/providers).
    """
    if DB_PATH.exists():
        return _load_db_snapshot()

    # Fallback: live provider list from MCP (same shape as the snapshot in most deployments)
    with httpx.Client(timeout=20.0) as client:
        r = client.get(f"{mcp_base_url}/providers")
        r.raise_for_status()
        payload = r.json()
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    pytest.skip(f"Could not load DB snapshot and MCP /providers response shape was unexpected: {type(payload).__name__}")


def _find_doctor(db: list[dict[str, Any]], *, clinic_name: str, provider_id: str) -> dict[str, str]:
    for clinic in db:
        if str(clinic.get("clinicName", "")).strip() != clinic_name:
            continue
        cid = str(clinic.get("clinicId", "")).strip()
        for d in (clinic.get("doctors") or []):
            if str(d.get("providerId", "")).strip() != provider_id:
                continue
            return {
                "clinic_id": cid,
                "clinic_name": clinic_name,
                "provider_id": provider_id,
                "name_ar": str(d.get("DoctorNameA") or "").strip(),
                "name_en": str(d.get("DoctorNameL") or "").strip(),
            }
    raise AssertionError(f"Doctor not found in DB snapshot: clinic_name={clinic_name!r}, provider_id={provider_id!r}")


def _find_doctor_by_prefix(db: list[dict[str, Any]], *, clinic_name: str, name_prefix: str) -> dict[str, str]:
    """
    More robust lookup than providerId: find the first doctor in a given clinic whose Arabic name
    starts with a given prefix (e.g., "ميلاد عبده").
    """
    target_prefix = name_prefix.strip()
    for clinic in db:
        if str(clinic.get("clinicName", "")).strip() != clinic_name:
            continue
        cid = str(clinic.get("clinicId", "")).strip()
        for d in (clinic.get("doctors") or []):
            provider_id = str(d.get("providerId", "")).strip()
            name_ar = str(d.get("DoctorNameA") or "").strip()
            if provider_id and name_ar.startswith(target_prefix):
                return {
                    "clinic_id": cid,
                    "clinic_name": clinic_name,
                    "provider_id": provider_id,
                    "name_ar": name_ar,
                    "name_en": str(d.get("DoctorNameL") or "").strip(),
                }
    raise AssertionError(f"Doctor not found by prefix in snapshot: clinic_name={clinic_name!r}, prefix={name_prefix!r}")


def _find_clinic(db: list[dict[str, Any]], *, clinic_name: str) -> dict[str, str]:
    for clinic in db:
        if str(clinic.get("clinicName", "")).strip() == clinic_name:
            return {
                "clinic_id": str(clinic.get("clinicId", "")).strip(),
                "clinic_name": clinic_name,
            }
    raise AssertionError(f"Clinic not found in DB snapshot: clinic_name={clinic_name!r}")


def _match_doctor(
    base_url: str,
    *,
    query: str,
    clinic_id: str | None = None,
    top_k: int = 5,
    min_score_multi: float = 0.6,
    min_score_single: float = 0.55,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "min_score_multi": min_score_multi,
        "min_score_single": min_score_single,
    }
    if clinic_id is not None:
        params["clinic_id"] = clinic_id

    with httpx.Client(timeout=20.0) as client:
        r = client.get(f"{base_url}/providers/match", params=params)
        r.raise_for_status()
        return r.json()


def _match_clinic(
    base_url: str,
    *,
    query: str,
    top_k: int = 5,
    min_score: float = 0.65,
) -> dict[str, Any]:
    params: dict[str, Any] = {"query": query, "top_k": top_k, "min_score": min_score}
    with httpx.Client(timeout=20.0) as client:
        r = client.get(f"{base_url}/clinics/match", params=params)
        r.raise_for_status()
        return r.json()


def _ids_from_doctor_match(payload: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    best = payload.get("best_match") or {}
    if isinstance(best, dict) and best.get("provider_id") is not None:
        ids.add(str(best.get("provider_id")))
    for c in (payload.get("candidates") or []):
        if isinstance(c, dict) and c.get("provider_id") is not None:
            ids.add(str(c.get("provider_id")))
    return ids


def _ids_from_clinic_match(payload: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    best = payload.get("best_match") or {}
    if isinstance(best, dict) and best.get("clinic_id") is not None:
        ids.add(str(best.get("clinic_id")))
    for c in (payload.get("candidates") or []):
        if isinstance(c, dict) and c.get("clinic_id") is not None:
            ids.add(str(c.get("clinic_id")))
    return ids


def _pick_doctor_record(
    db: list[dict[str, Any]],
    *,
    predicate,
    why: str,
) -> dict[str, str]:
    """
    Pick a single real doctor record from the snapshot/provider list that matches a predicate.
    Skips the test if such a record is not found (to keep tests robust across datasets).
    """
    for clinic in db:
        clinic_id = str(clinic.get("clinicId", "")).strip()
        clinic_name = str(clinic.get("clinicName", "")).strip()
        for d in (clinic.get("doctors") or []):
            provider_id = str(d.get("providerId", "")).strip()
            name_ar = str(d.get("DoctorNameA") or "").strip()
            name_en = str(d.get("DoctorNameL") or "").strip()
            if not clinic_id or not provider_id or not name_ar:
                continue
            # Skip obviously non-person placeholders found in some snapshots.
            if "_" in name_ar:
                continue
            if predicate(name_ar):
                return {
                    "clinic_id": clinic_id,
                    "clinic_name": clinic_name,
                    "provider_id": provider_id,
                    "name_ar": name_ar,
                    "name_en": name_en,
                }
    pytest.skip(f"Could not find a suitable doctor record for: {why}")


def _prefix_ending_at_token(tokens: list[str], idx: int) -> str:
    """
    Build a short-but-specific prefix query: include the target token + one more token if available.
    This reduces ambiguity while keeping the query realistic.
    """
    end = min(len(tokens), idx + 2)
    return " ".join(tokens[:end]).strip()


def _split_after_abd(compound: str) -> tuple[str, str]:
    """
    Split "عبدالرحيم" -> ("عبد", "الرحيم"), "عبدالله" -> ("عبد", "الله").
    """
    assert compound.startswith("عبد") and len(compound) > 3
    return "عبد", compound[3:]


def test_doctor_misspelling_milad_abdou_matches(mcp_base_url: str, db_snapshot):
    """
    Real-world regression: user types "ميلاد عبدو" but DB has "ميلاد عبده ...".
    Must not return NO_MATCH, and must surface providerId=10158 from clinic "جراحه".
    """
    rec = _find_doctor_by_prefix(db_snapshot, clinic_name="جراحه", name_prefix="ميلاد عبده")
    assert rec["name_ar"].startswith("ميلاد "), "Snapshot expected 'ميلاد ...' doctor under جراحه"

    payload = _match_doctor(
        mcp_base_url,
        query="دكتور ميلاد عبدو",
        clinic_id=rec["clinic_id"],
        top_k=10,
    )

    assert payload.get("status") != "NO_MATCH", payload
    assert rec["provider_id"] in _ids_from_doctor_match(payload), payload


def test_doctor_exact_full_name_matches(mcp_base_url: str, db_snapshot):
    """
    Exact match should resolve cleanly (sanity check).
    Uses a real doctor from the snapshot.
    """
    rec = _find_doctor_by_prefix(db_snapshot, clinic_name="جراحه", name_prefix="ميلاد عبده")
    payload = _match_doctor(
        mcp_base_url,
        query=rec["name_ar"],
        clinic_id=rec["clinic_id"],
        top_k=5,
    )
    assert payload.get("status") != "NO_MATCH", payload
    best = payload.get("best_match") or {}
    assert isinstance(best, dict) and str(best.get("provider_id")) == rec["provider_id"], payload


def test_clinic_misspelling_nesa_tawleed_matches(mcp_base_url: str, db_snapshot):
    """
    Real clinic misspelling: "النساؤ" vs "نسا/النساء".
    Must not return NO_MATCH and should include clinicId for "نسا وتوليد".
    """
    rec = _find_clinic(db_snapshot, clinic_name="نسا وتوليد")
    payload = _match_clinic(mcp_base_url, query="عيادة النساؤ وتوليد", top_k=10)

    assert payload.get("status") != "NO_MATCH", payload
    assert rec["clinic_id"] in _ids_from_clinic_match(payload), payload


def test_clinic_exact_name_best_match_is_correct(mcp_base_url: str, db_snapshot):
    """
    Exact clinic name should return a best_match that points to the correct clinicId.
    """
    rec = _find_clinic(db_snapshot, clinic_name="جراحه")
    payload = _match_clinic(mcp_base_url, query=rec["clinic_name"], top_k=5)
    assert payload.get("status") != "NO_MATCH", payload
    best = payload.get("best_match") or {}
    assert isinstance(best, dict) and str(best.get("clinic_id")) == rec["clinic_id"], payload


def test_clinic_one_word_surgery_resolves_without_disambiguation(mcp_base_url: str, db_snapshot):
    """
    Special-case UX rule:
    If the user says the generic one-word surgery clinic ("الجراحة/جراحة"),
    we should resolve directly to the base clinic "جراحه" (no disambiguation list).
    """
    rec = _find_clinic(db_snapshot, clinic_name="جراحه")

    # User-like inputs (ta marbuta + "عيادة" prefix)
    payload = _match_clinic(mcp_base_url, query="عيادة الجراحة", top_k=10)
    assert payload.get("status") == "UNAMBIGUOUS_MATCH", payload
    best = payload.get("best_match") or {}
    assert isinstance(best, dict)
    assert str(best.get("clinic_id")) == rec["clinic_id"], payload
    assert str(best.get("clinic_name")) == rec["clinic_name"], payload

    # If time context leaks into clinic text, still treat as base "جراحه"
    payload2 = _match_clinic(mcp_base_url, query="عيادة الجراحة الاسبوع ده", top_k=10)
    assert payload2.get("status") == "UNAMBIGUOUS_MATCH", payload2
    best2 = payload2.get("best_match") or {}
    assert isinstance(best2, dict)
    assert str(best2.get("clinic_id")) == rec["clinic_id"], payload2


def test_clinic_multi_word_surgery_name_still_matches_specific_clinic(mcp_base_url: str, db_snapshot):
    """
    Guardrail:
    If the user provides a 2+ token clinic name that includes "جراحه" (e.g., "جراحة تجميل"),
    do NOT force it to base "جراحه". Keep normal matching behavior.
    """
    rec = _find_clinic(db_snapshot, clinic_name="جراحه تجميل")
    payload = _match_clinic(mcp_base_url, query="جراحة تجميل", top_k=10)
    assert payload.get("status") != "NO_MATCH", payload
    best = payload.get("best_match") or {}
    assert isinstance(best, dict)
    assert str(best.get("clinic_id")) == rec["clinic_id"], payload
    assert str(best.get("clinic_name")) == rec["clinic_name"], payload


def test_doctor_ambiguous_single_token_returns_multiple_candidates(mcp_base_url: str, db_snapshot):
    """
    Doctor ambiguity: a common first-name token that exists for multiple doctors.
    We pick 'ابانوب' from the real snapshot (appears multiple times).
    """
    # Verify snapshot actually contains multiple doctors starting with this token.
    count = 0
    for clinic in db_snapshot:
        for d in (clinic.get("doctors") or []):
            name = str(d.get("DoctorNameA") or "").strip()
            if name.split()[:1] == ["ابانوب"]:
                count += 1
    assert count >= 2, "Snapshot should contain >=2 doctors with first token 'ابانوب'"

    payload = _match_doctor(mcp_base_url, query="ابانوب", clinic_id=None, top_k=10)

    # The key requirement: the matcher must surface multiple candidates (so UX can ask user to pick).
    assert payload.get("status") in {"AMBIGUOUS_NEED_MORE_INFO", "LOW_CONFIDENCE"}, payload
    assert isinstance(payload.get("candidates"), list) and len(payload.get("candidates")) >= 2, payload


def test_clinic_ambiguous_query_exists_in_snapshot_and_returns_multiple_candidates(mcp_base_url: str, db_snapshot):
    """
    Clinic ambiguity: rather than hard-coding a single query that might become unambiguous
    with scoring tweaks, we automatically find a short token that appears in multiple clinic names,
    then assert the MCP clinic matcher returns multiple candidates for at least one such query.
    """
    clinic_names = [str(c.get("clinicName", "")).strip() for c in db_snapshot if str(c.get("clinicName", "")).strip()]
    assert clinic_names

    token_counts = Counter()
    clinic_name_set = set(clinic_names)
    for name in clinic_names:
        for tok in name.split():
            tok = tok.strip()
            if not tok or tok in {"و", "في", "ال"}:
                continue
            if len(tok) < 3:
                continue
            token_counts[tok] += 1

    # Candidate tokens: appear in 2+ clinics, and are not themselves an exact clinic name.
    candidates = [t for (t, n) in token_counts.most_common() if n >= 2 and t not in clinic_name_set]
    assert candidates, "Expected at least one repeated clinic token in snapshot"

    found: Optional[dict[str, Any]] = None
    found_query: Optional[str] = None

    for tok in candidates[:30]:
        payload = _match_clinic(mcp_base_url, query=tok, top_k=10)
        if payload.get("status") == "AMBIGUOUS_NEED_MORE_INFO" and isinstance(payload.get("candidates"), list) and len(payload.get("candidates")) >= 2:
            found = payload
            found_query = tok
            break

    assert found is not None, (
        "Could not find an ambiguous clinic query in the first 30 repeated tokens. "
        "Either the dataset changed or clinic matching is always unambiguous for these tokens."
    )
    assert found_query is not None
    assert found.get("status") == "AMBIGUOUS_NEED_MORE_INFO", {"query": found_query, "payload": found}
    assert len(found.get("candidates") or []) >= 2, {"query": found_query, "payload": found}


def test_snapshot_sample_doctors_partial_name_is_matchable(mcp_base_url: str, db_snapshot):
    """
    Broader coverage: sample multiple real doctors from the snapshot and ensure that
    a realistic user short-form (first 2 tokens) is still matchable within that clinic.
    """
    samples: list[dict[str, str]] = []
    for clinic in db_snapshot:
        clinic_id = str(clinic.get("clinicId", "")).strip()
        clinic_name = str(clinic.get("clinicName", "")).strip()
        if not clinic_id or not clinic_name:
            continue
        for d in (clinic.get("doctors") or []):
            provider_id = str(d.get("providerId", "")).strip()
            name_ar = str(d.get("DoctorNameA") or "").strip()
            if not provider_id or not name_ar:
                continue
            toks = [t for t in name_ar.split() if t]
            if len(toks) < 2:
                continue
            # Skip obviously non-person placeholders found in some snapshots.
            if toks[0] in {"دكتور", "د", "د."}:
                continue
            if "_" in name_ar:
                continue

            samples.append(
                {
                    "clinic_id": clinic_id,
                    "clinic_name": clinic_name,
                    "provider_id": provider_id,
                    "name_ar": name_ar,
                    "query": " ".join(toks[:2]),
                }
            )
            break
        if len(samples) >= 10:
            break

    assert samples, "No suitable doctor samples were found in the snapshot."

    for s in samples:
        payload = _match_doctor(
            mcp_base_url,
            query=s["query"],
            clinic_id=s["clinic_id"],
            top_k=10,
        )
        assert payload.get("status") != "NO_MATCH", {"sample": s, "payload": payload}
        assert s["provider_id"] in _ids_from_doctor_match(payload), {"sample": s, "payload": payload}


def test_snapshot_sample_clinics_exact_name_best_match_is_correct(mcp_base_url: str, db_snapshot):
    """
    Broader coverage: sample multiple real clinics from the snapshot and ensure that
    exact clinic-name matching returns the correct clinicId as best_match.
    """
    samples: list[dict[str, str]] = []
    for clinic in db_snapshot:
        clinic_id = str(clinic.get("clinicId", "")).strip()
        clinic_name = str(clinic.get("clinicName", "")).strip()
        if not clinic_id or not clinic_name:
            continue
        samples.append({"clinic_id": clinic_id, "clinic_name": clinic_name})
        if len(samples) >= 10:
            break

    assert samples, "No clinic samples were found in the snapshot."

    for s in samples:
        payload = _match_clinic(mcp_base_url, query=s["clinic_name"], top_k=10)
        assert payload.get("status") != "NO_MATCH", {"sample": s, "payload": payload}
        best = payload.get("best_match") or {}
        assert isinstance(best, dict) and str(best.get("clinic_id")) == s["clinic_id"], {"sample": s, "payload": payload}


# ------------------------------------------------------------------------------
# Desired behavior tests (expected to FAIL until matching-engine changes land)
# ------------------------------------------------------------------------------

def test_doctor_abdallah_join_split_and_typos_match_same_provider(mcp_base_url: str, db_snapshot):
    """
    'عبدالله' is often written either as:
    - joined: عبدالله
    - split: عبد الله

    Users may also typo 'عبد' or 'الله' (e.g., عبذ / اللاه).

    This test defines the desired behavior: all variants should still surface the
    same provider_id (within the same clinic).
    """
    rec = _pick_doctor_record(
        db_snapshot,
        predicate=lambda name: "عبدالله" in name.split(),
        why="a doctor name containing the standalone token 'عبدالله'",
    )

    toks = [t for t in rec["name_ar"].split() if t]
    idx = toks.index("عبدالله")
    base = _prefix_ending_at_token(toks, idx)

    variants = [
        base,  # stored joined
        base.replace("عبدالله", "عبد الله", 1),   # split
        base.replace("عبدالله", "عبذ الله", 1),   # typo in عبد
        base.replace("عبدالله", "عبد اللاه", 1),  # typo in الله
        base.replace("عبدالله", "عبذ اللاه", 1),  # typos in both
    ]

    for q in variants:
        payload = _match_doctor(
            mcp_base_url,
            query=f"دكتور {q}",
            clinic_id=rec["clinic_id"],
            top_k=20,
        )
        assert payload.get("status") != "NO_MATCH", {"rec": rec, "query": q, "payload": payload}
        assert rec["provider_id"] in _ids_from_doctor_match(payload), {"rec": rec, "query": q, "payload": payload}


def test_doctor_abdal_compound_join_split_and_typo_in_abd_match_same_provider(mcp_base_url: str, db_snapshot):
    """
    'عبدال...' compounds (e.g., عبدالرحيم / عبدالملاك / عبدالوهاب) are often written as:
    - joined: عبدالرحيم
    - split: عبد الرحيم
    Users can also typo عبد (عبذ).

    Desired behavior: these variants should still surface the same provider_id.
    """
    rec = _pick_doctor_record(
        db_snapshot,
        predicate=lambda name: any(tok.startswith("عبدال") and tok != "عبدالله" for tok in name.split()),
        why="a doctor name containing a compound token starting with 'عبدال' (not عبدالله)",
    )

    toks = [t for t in rec["name_ar"].split() if t]
    compound = next(tok for tok in toks if tok.startswith("عبدال") and tok != "عبدالله")
    idx = toks.index(compound)
    base = _prefix_ending_at_token(toks, idx)

    abd, rest = _split_after_abd(compound)
    assert rest, f"Unexpected compound token: {compound!r}"

    # Also include a common real-world variant where users drop the leading "ال" after splitting.
    rest_no_al = rest[2:] if rest.startswith("ال") and len(rest) > 3 else rest

    variants = [
        base,  # stored joined
        base.replace(compound, f"{abd} {rest}", 1),        # split
        base.replace(compound, f"عبذ {rest}", 1),          # typo in عبد
        base.replace(compound, f"{abd} {rest_no_al}", 1),  # split without leading "ال"
    ]

    for q in variants:
        payload = _match_doctor(
            mcp_base_url,
            query=f"دكتور {q}",
            clinic_id=rec["clinic_id"],
            top_k=20,
        )
        assert payload.get("status") != "NO_MATCH", {"rec": rec, "query": q, "payload": payload}
        assert rec["provider_id"] in _ids_from_doctor_match(payload), {"rec": rec, "query": q, "payload": payload}


def test_doctor_allah_standalone_space_vs_join_and_typos_match_same_provider(mcp_base_url: str, db_snapshot):
    """
    Names like 'عوض الله' / 'رزق الله' are often written:
    - spaced: عوض الله
    - joined: عوضالله
    Users can also typo الله (اللاه).

    Desired behavior: all variants should surface the same provider_id.
    """
    rec = _pick_doctor_record(
        db_snapshot,
        predicate=lambda name: "الله" in name.split() and name.split().index("الله") > 0,
        why="a doctor name with a standalone 'الله' token (not first token)",
    )

    toks = [t for t in rec["name_ar"].split() if t]
    allah_idx = toks.index("الله")
    base = _prefix_ending_at_token(toks, allah_idx)
    base_tokens = base.split()

    # Ensure our short prefix still includes 'الله' and a preceding token.
    if "الله" not in base_tokens:
        pytest.skip("Chosen record has الله but the short prefix did not include it.")
    i = base_tokens.index("الله")
    if i == 0:
        pytest.skip("Short prefix begins with الله; need a preceding token.")

    prev = base_tokens[i - 1]
    joined = " ".join(base_tokens[: i - 1] + [prev + "الله"] + base_tokens[i + 1 :]).strip()
    joined_typo = " ".join(base_tokens[: i - 1] + [prev + "اللاه"] + base_tokens[i + 1 :]).strip()

    variants = [
        base,         # spaced
        joined,       # joined
        joined_typo,  # joined + typo in الله
    ]

    for q in variants:
        payload = _match_doctor(
            mcp_base_url,
            query=f"دكتور {q}",
            clinic_id=rec["clinic_id"],
            top_k=20,
        )
        assert payload.get("status") != "NO_MATCH", {"rec": rec, "query": q, "payload": payload}
        assert rec["provider_id"] in _ids_from_doctor_match(payload), {"rec": rec, "query": q, "payload": payload}


def test_doctor_allah_suffix_join_split_and_typos_match_same_provider(mcp_base_url: str, db_snapshot):
    """
    Names can contain joined suffix '*الله' tokens like: 'جادالله', 'نصرالله', 'عطاالله'.
    Users may write them:
    - joined: جادالله
    - split: جاد الله
    and typo الله: جاد اللاه

    Desired behavior: variants should still surface the same provider_id.
    """
    rec = _pick_doctor_record(
        db_snapshot,
        predicate=lambda name: any(
            tok.endswith("الله")
            and len(tok) > 4
            and not tok.startswith("عبد")
            and tok != "عبدالله"
            for tok in name.split()
        ),
        why="a doctor name containing a joined '*الله' suffix token (not عبد... / عبدالله)",
    )

    toks = [t for t in rec["name_ar"].split() if t]
    joined_tok = next(
        tok
        for tok in toks
        if tok.endswith("الله") and len(tok) > 4 and not tok.startswith("عبد") and tok != "عبدالله"
    )
    idx = toks.index(joined_tok)
    base = _prefix_ending_at_token(toks, idx)

    stem = joined_tok[:-4]
    assert stem, f"Unexpected '*الله' token: {joined_tok!r}"

    variants = [
        base,  # joined as stored
        base.replace(joined_tok, f"{stem} الله", 1),
        base.replace(joined_tok, f"{stem} اللاه", 1),
    ]

    for q in variants:
        payload = _match_doctor(
            mcp_base_url,
            query=f"دكتور {q}",
            clinic_id=rec["clinic_id"],
            top_k=20,
        )
        assert payload.get("status") != "NO_MATCH", {"rec": rec, "query": q, "payload": payload}
        assert rec["provider_id"] in _ids_from_doctor_match(payload), {"rec": rec, "query": q, "payload": payload}

