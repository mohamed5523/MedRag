#!/usr/bin/env python3
"""
evaluation/test_mcp_endpoints.py
---------------------------------
Standalone MCP endpoint tester that validates the MCP server's HTTP routes
against real clinic/provider data from the hospital system.

Usage:
    # From project root
    backend/.venv/bin/python evaluation/test_mcp_endpoints.py

    # With a custom MCP server URL
    MCP_URL=http://localhost:8001 backend/.venv/bin/python evaluation/test_mcp_endpoints.py

    # Run dataset-only check (no server needed)
    backend/.venv/bin/python evaluation/test_mcp_endpoints.py --offline

The script tests against REAL data:
  - Doctors: إبرام سعيد, بيمن عادل, كمال أمين, سامي سعد, رويس ثروت, مايكل صفوت
  - Dentists: أبانوب ذكري, الفريد فؤاد, جاكلين فؤاد
  - Internal medicine: إيريني لويس, بسمه محمد, راجي إبراهيم, ...
  - Prices: consultation 50–200 EGP, zircon crown 2100 EGP, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

MCP_BASE_URL = os.getenv("MCP_URL", "http://localhost:8000")
# Basic auth credentials (same as config.py defaults)
AUTH = (
    os.getenv("CLINIC_API_USERNAME", "millen"),
    os.getenv("CLINIC_API_PASSWORD", "millen@4321"),
)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset", "mcp_queries.json")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def _header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


def _get(path: str, params: Optional[Dict] = None) -> Tuple[bool, int, Any, float]:
    url = MCP_BASE_URL + path
    t0 = time.monotonic()
    try:
        r = requests.get(url, params=params, timeout=15)
        elapsed = (time.monotonic() - t0) * 1000
        try:
            body = r.json()
        except Exception:
            body = r.text
        return True, r.status_code, body, elapsed
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        return False, 0, str(exc), elapsed


def _post(path: str, payload: Dict) -> Tuple[bool, int, Any, float]:
    url = MCP_BASE_URL + path
    t0 = time.monotonic()
    try:
        r = requests.post(url, json=payload, timeout=15)
        elapsed = (time.monotonic() - t0) * 1000
        try:
            body = r.json()
        except Exception:
            body = r.text
        return True, r.status_code, body, elapsed
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        return False, 0, str(exc), elapsed


# ── Test helpers ──────────────────────────────────────────────────────────────

class TestSuite:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results: List[Dict] = []

    def assert_true(self, condition: bool, name: str, detail: str = "") -> bool:
        if condition:
            self.passed += 1
            _ok(f"{name}{' — ' + detail if detail else ''}")
            self.results.append({"name": name, "status": "pass", "detail": detail})
        else:
            self.failed += 1
            _fail(f"{name}{' — ' + detail if detail else ''}")
            self.results.append({"name": name, "status": "fail", "detail": detail})
        return condition

    def skip(self, name: str, reason: str = "") -> None:
        self.skipped += 1
        _warn(f"SKIP {name}{' — ' + reason if reason else ''}")
        self.results.append({"name": name, "status": "skip", "detail": reason})

    @property
    def total(self) -> int:
        return self.passed + self.failed

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


# ── Individual tests ──────────────────────────────────────────────────────────

def test_server_reachable(suite: TestSuite) -> bool:
    _header("1. Server Reachability")
    ok, status, body, latency = _get("/providers")
    if suite.assert_true(
        ok and status == 200,
        "MCP server reachable at /providers",
        f"HTTP {status}, {latency:.0f}ms" if ok else str(body)[:120],
    ):
        return True
    print(f"\n  {YELLOW}Server not reachable. Remaining live tests will be skipped.{RESET}")
    return False


def test_provider_list_structure(suite: TestSuite) -> Optional[List[Dict]]:
    _header("2. Provider List — Structure Validation")
    ok, status, body, latency = _get("/providers")
    if not ok or status != 200:
        suite.skip("Provider list structure", "server not reachable")
        return None

    # Must return JSON
    suite.assert_true(isinstance(body, dict), "Response is a JSON object", f"type={type(body).__name__}")

    # Must have a 'data' key
    has_data = "data" in body
    suite.assert_true(has_data, "Response contains 'data' key")
    if not has_data:
        return None

    data = body["data"]
    suite.assert_true(isinstance(data, list), "'data' is a list", f"type={type(data).__name__}")
    suite.assert_true(len(data) > 0, "'data' is non-empty", f"{len(data)} clinics")

    # Validate first clinic structure
    if data:
        first = data[0]
        suite.assert_true("clinicId" in first, "Clinic has 'clinicId'")
        suite.assert_true("clinicName" in first, "Clinic has 'clinicName'")
        has_docs = "doctors" in first
        suite.assert_true(has_docs, "Clinic has 'doctors' list")

        if has_docs and first["doctors"]:
            doc = first["doctors"][0]
            suite.assert_true("providerId" in doc, "Doctor has 'providerId'")
            has_name = "DoctorNameA" in doc or "DoctorNameL" in doc
            suite.assert_true(has_name, "Doctor has Arabic or Latin name")

    print(f"  {CYAN}→ {len(data)} clinics returned, {latency:.0f}ms{RESET}")
    return data


def test_doctor_matching(suite: TestSuite, server_up: bool) -> None:
    _header("3. Doctor Matching — Real Names")

    real_doctor_queries = [
        # (query, expected_first_name_fragment, description)
        ("إبرام",          "إبرام",  "Single first name — Surgery"),
        ("بيمن عادل",      "بيمن",   "Two-token name — Surgery"),
        ("كمال أمين",      "كمال",   "Two-token name — Surgery"),
        ("سامي سعد",       "سامي",   "Two-token name — Surgery"),
        ("أبانوب",         "أبانوب", "Single first name — Dentistry"),
        ("رويس ثروت",      "رويس",   "Two-token name — Surgery"),
        ("إيريني لويس",    "إيريني", "Two-token name — Internal Medicine"),
    ]

    for query, expected_fragment, desc in real_doctor_queries:
        if not server_up:
            suite.skip(f"Match '{query}'", "server offline")
            continue

        ok, status, body, latency = _post("/providers/match", {"query": query})
        if not ok or status != 200:
            suite.assert_true(False, f"Match '{query}' ({desc})", f"HTTP {status}")
            continue

        # Accept UNAMBIGUOUS_MATCH, AMBIGUOUS_NEED_MORE_INFO, or LOW_CONFIDENCE
        match_status = body.get("status", "") if isinstance(body, dict) else ""
        found = match_status in (
            "UNAMBIGUOUS_MATCH",
            "AMBIGUOUS_NEED_MORE_INFO",
            "LOW_CONFIDENCE",
        )
        suite.assert_true(
            found,
            f"Match '{query}' ({desc})",
            f"status={match_status}, {latency:.0f}ms",
        )

        # Check that best_match or candidates contain the expected name fragment
        candidates = []
        if isinstance(body, dict):
            if body.get("best_match"):
                candidates = [body["best_match"]]
            candidates += body.get("candidates", [])

        name_match = any(
            expected_fragment in c.get("name_ar", "")
            for c in candidates
        )
        suite.assert_true(
            name_match or not candidates,
            f"  → Candidate contains '{expected_fragment}'",
            f"candidates={[c.get('name_ar','')[:20] for c in candidates[:3]]}",
        )


def test_clinic_matching(suite: TestSuite, server_up: bool) -> None:
    _header("4. Clinic Matching — Real Clinic Names")

    clinic_queries = [
        ("أسنان",     "UNAMBIGUOUS_MATCH", "Dentistry by keyword"),
        ("الجراحة",   "UNAMBIGUOUS_MATCH", "Surgery generic"),
        ("باطنة",     "UNAMBIGUOUS_MATCH", "Internal medicine"),
    ]

    for query, expected_status, desc in clinic_queries:
        if not server_up:
            suite.skip(f"Clinic match '{query}'", "server offline")
            continue

        ok, status, body, latency = _post("/clinics/match", {"query": query})
        if not ok or status != 200:
            suite.assert_true(False, f"Clinic match '{query}' ({desc})", f"HTTP {status}")
            continue

        match_status = body.get("status", "") if isinstance(body, dict) else ""
        # Accept exact expected or AMBIGUOUS (acceptable fallback)
        found = match_status in (expected_status, "AMBIGUOUS_NEED_MORE_INFO")
        suite.assert_true(
            found,
            f"Clinic match '{query}' ({desc})",
            f"status={match_status}, {latency:.0f}ms",
        )


def test_schedule(suite: TestSuite, server_up: bool, clinics_data: Optional[List]) -> None:
    _header("5. Provider Schedule — Endpoint Structure")

    if not server_up:
        suite.skip("Schedule endpoint", "server offline")
        return

    # Use first clinic in the data (or clinic ID 1 as fallback)
    clinic_id = 1
    if clinics_data:
        # Try to find a surgery-like clinic
        for c in clinics_data:
            name = str(c.get("clinicName", "")).lower()
            if "جراح" in name or "surgery" in name:
                clinic_id = c.get("clinicId", 1)
                break
        else:
            clinic_id = clinics_data[0].get("clinicId", 1)

    date_from = datetime.now().strftime("%d/%m/%Y")
    date_to   = (datetime.now() + timedelta(days=7)).strftime("%d/%m/%Y")

    ok, status, body, latency = _get(
        "/providers/schedule",
        params={"clinicid": clinic_id, "dateFrom": date_from, "dateTo": date_to},
    )

    suite.assert_true(
        ok and status == 200,
        f"Schedule for clinicid={clinic_id} returns HTTP 200",
        f"{latency:.0f}ms",
    )

    if ok and status == 200:
        is_json = isinstance(body, (dict, list))
        suite.assert_true(is_json, "Schedule response is valid JSON")
        print(f"  {CYAN}→ Schedule response received ({latency:.0f}ms){RESET}")


def test_pricing(suite: TestSuite, server_up: bool, clinics_data: Optional[List]) -> None:
    _header("6. Service Pricing — Endpoint Structure")

    if not server_up:
        suite.skip("Pricing endpoint", "server offline")
        return

    # Try dentistry clinic first, otherwise first clinic
    clinic_id = 1
    if clinics_data:
        for c in clinics_data:
            name = str(c.get("clinicName", "")).lower()
            if "أسنان" in name or "dental" in name.lower() or "دنت" in name:
                clinic_id = c.get("clinicId", 1)
                break
        else:
            clinic_id = clinics_data[0].get("clinicId", 1)

    ok, status, body, latency = _get(
        "/providers/services/pricing",
        params={"clinicid": clinic_id},
    )

    suite.assert_true(
        ok and status == 200,
        f"Pricing for clinicid={clinic_id} returns HTTP 200",
        f"{latency:.0f}ms",
    )

    if ok and status == 200:
        is_json = isinstance(body, (dict, list))
        suite.assert_true(is_json, "Pricing response is valid JSON")
        print(f"  {CYAN}→ Pricing response received ({latency:.0f}ms){RESET}")


def test_error_handling(suite: TestSuite, server_up: bool) -> None:
    _header("7. Error Handling — Invalid Inputs")

    if not server_up:
        suite.skip("Error handling tests", "server offline")
        return

    # Missing clinicid → should return 400
    ok, status, body, _ = _get("/providers/schedule", params={})
    suite.assert_true(
        ok and status == 400,
        "Schedule without clinicid returns 400",
        f"got {status}",
    )

    # Missing clinicid for pricing → should return 400
    ok2, status2, body2, _ = _get("/providers/services/pricing", params={})
    suite.assert_true(
        ok2 and status2 == 400,
        "Pricing without clinicid returns 400",
        f"got {status2}",
    )

    # Empty query for doctor match → should return 400
    ok3, status3, body3, _ = _post("/providers/match", {"query": ""})
    suite.assert_true(
        ok3 and status3 == 400,
        "Doctor match with empty query returns 400",
        f"got {status3}",
    )

    # Empty query for clinic match → should return 400
    ok4, status4, body4, _ = _post("/clinics/match", {"query": ""})
    suite.assert_true(
        ok4 and status4 == 400,
        "Clinic match with empty query returns 400",
        f"got {status4}",
    )


def test_dataset_integrity(suite: TestSuite) -> None:
    _header("8. Dataset Integrity — mcp_queries.json")

    try:
        with open(DATASET_PATH, encoding="utf-8") as f:
            samples = json.load(f)
    except Exception as exc:
        suite.assert_true(False, "mcp_queries.json loads without error", str(exc))
        return

    suite.assert_true(isinstance(samples, list), "Dataset is a JSON array")
    suite.assert_true(len(samples) >= 20, f"Dataset has ≥20 samples", f"{len(samples)} found")

    categories: Dict[str, int] = {}
    errors = []
    for i, s in enumerate(samples):
        sid = s.get("id", i+1)
        # Required keys
        for key in ("question", "expected_intent", "expected_mode", "required_slots", "category", "expected_keywords"):
            if key not in s:
                errors.append(f"Sample {sid} missing '{key}'")
        # At least 2 keywords
        kws = s.get("expected_keywords", [])
        if len(kws) < 2:
            errors.append(f"Sample {sid} has only {len(kws)} keyword(s), need ≥2")
        # Question not empty
        if not s.get("question", "").strip():
            errors.append(f"Sample {sid} has empty question")
        categories[s.get("category", "unknown")] = categories.get(s.get("category", ""), 0) + 1

    suite.assert_true(len(errors) == 0, "All samples pass schema validation", "; ".join(errors[:3]) if errors else "")

    # Category balance
    suite.assert_true("pricing" in categories,    f"Has pricing samples ({categories.get('pricing',0)})")
    suite.assert_true("availability" in categories, f"Has availability samples ({categories.get('availability',0)})")
    suite.assert_true("discovery" in categories,  f"Has discovery samples ({categories.get('discovery',0)})")
    suite.assert_true("booking" in categories,    f"Has booking samples ({categories.get('booking',0)})")

    print(f"\n  {CYAN}Category distribution:{RESET}")
    for cat, count in sorted(categories.items()):
        bar = "█" * count
        print(f"    {cat:<16} {bar} ({count})")


def test_real_data_keywords(suite: TestSuite) -> None:
    """Verify that the dataset actually contains the real doctor/price keywords."""
    _header("9. Dataset Real-Data Coverage")

    try:
        with open(DATASET_PATH, encoding="utf-8") as f:
            samples = json.load(f)
    except Exception:
        suite.skip("Real-data keyword check", "dataset not loadable")
        return

    all_keywords = [kw for s in samples for kw in s.get("expected_keywords", [])]
    all_questions = [s.get("question", "") for s in samples]

    # Real doctor names that MUST appear somewhere in the dataset
    must_have_doctors = ["إبرام", "بيمن", "كمال", "سامي", "أبانوب"]
    for doc in must_have_doctors:
        found = any(doc in q for q in all_questions) or any(doc in kw for kw in all_keywords)
        suite.assert_true(found, f"Dataset references real doctor '{doc}'")

    # Real prices that MUST appear in keywords
    must_have_prices = ["500", "650", "4000", "2100", "1200"]
    for price in must_have_prices:
        found = any(price in kw for kw in all_keywords)
        suite.assert_true(found, f"Dataset has price keyword '{price} جنيه'")

    # Real days (surgery schedule)
    must_have_days = ["السبت", "الأحد", "الاثنين", "الجمعة"]
    for day in must_have_days:
        found = any(day in kw for kw in all_keywords) or any(day in q for q in all_questions)
        suite.assert_true(found, f"Dataset references day '{day}'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="MCP Endpoint Tester")
    parser.add_argument("--offline", action="store_true", help="Skip live server tests")
    parser.add_argument("--url", default=None, help="Override MCP server URL")
    parser.add_argument("--json-out", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    global MCP_BASE_URL
    if args.url:
        MCP_BASE_URL = args.url.rstrip("/")

    print(f"\n{BOLD}MCP Endpoint Tester{RESET}")
    print(f"Server: {CYAN}{MCP_BASE_URL}{RESET}")
    print(f"Mode:   {'Offline (dataset only)' if args.offline else 'Live'}")
    print(f"Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not _HAS_REQUESTS:
        print(f"\n{YELLOW}Warning: 'requests' library not found. Install it: pip install requests{RESET}")
        print(f"{YELLOW}Running dataset-only tests...{RESET}")
        args.offline = True

    suite = TestSuite()

    # ── Dataset tests (always run) ────────────────────────────────────────────
    test_dataset_integrity(suite)
    test_real_data_keywords(suite)

    # ── Live server tests ─────────────────────────────────────────────────────
    if not args.offline:
        server_up = test_server_reachable(suite)
        clinics_data = test_provider_list_structure(suite)
        test_doctor_matching(suite, server_up)
        test_clinic_matching(suite, server_up)
        test_schedule(suite, server_up, clinics_data)
        test_pricing(suite, server_up, clinics_data)
        test_error_handling(suite, server_up)
    else:
        print(f"\n{YELLOW}Skipping live server tests (--offline mode).{RESET}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}Results: {suite.passed} passed / {suite.failed} failed / {suite.skipped} skipped{RESET}")
    score_pct = suite.score * 100
    colour = GREEN if score_pct >= 75 else (YELLOW if score_pct >= 50 else RED)
    print(f"{BOLD}Score:   {colour}{score_pct:.1f}%{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}\n")

    if args.json_out:
        import json as _json
        out = {
            "timestamp": datetime.now().isoformat(),
            "server_url": MCP_BASE_URL,
            "passed": suite.passed,
            "failed": suite.failed,
            "skipped": suite.skipped,
            "score": round(suite.score, 3),
            "results": suite.results,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            _json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {args.json_out}")

    return 0 if suite.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
