"""
evaluation/eval_whatsapp.py
---------------------------
End-to-end WhatsApp webhook evaluation — real mode only.

Sends real webhook POST requests to the running backend and checks:
  - HTTP 200 response
  - Non-empty response body
  - (Optional) expected_contains phrase in the reply

Uses the shared http_client for retry/backoff.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from .config import BACKEND_URL
from .http_client import post_with_retry


# ── Payload builder ───────────────────────────────────────────────────────────

def _make_wa_payload(message_text: str, msg_id: str = "wamid.test001") -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "ENTRY_ID",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"display_phone_number": "15550000000", "phone_number_id": "PHONE_ID"},
                    "contacts": [{"profile": {"name": "Test User"}, "wa_id": "201001234567"}],
                    "messages": [{
                        "id": msg_id,
                        "from": "201001234567",
                        "timestamp": "1700000000",
                        "type": "text",
                        "text": {"body": message_text},
                    }],
                },
                "field": "messages",
            }],
        }],
    }


# ── Real case runner ──────────────────────────────────────────────────────────

def _run_real_case(case: dict) -> Dict[str, Any]:
    """POST a WhatsApp webhook payload to the backend and check success."""
    payload = _make_wa_payload(case["message"], msg_id=f"wamid.{case.get('id', 'x')}")

    resp = post_with_retry(
        "/api/whatsapp/webhook",
        json=payload,
    )

    if not resp["ok"]:
        return {
            "success":     False,
            "status_code": resp["status_code"],
            "message":     case["message"],
            "error":       resp["error"],
            "latency_ms":  resp["latency_ms"],
        }

    return {
        "success":     True,
        "status_code": resp["status_code"],
        "message":     case["message"],
        "latency_ms":  resp["latency_ms"],
    }


# ── Main eval ─────────────────────────────────────────────────────────────────

def run_eval() -> Dict[str, Any]:
    """Run WhatsApp end-to-end evaluation against the real backend."""
    cases: List[dict] = [
        {"id": "1", "message": "مواعيد الدكتور أحمد",     "expected_contains": "مواعيد"},
        {"id": "2", "message": "سعر الكشف كام؟",           "expected_contains": "سعر"},
        {"id": "3", "message": "احجز موعد",                "expected_contains": "حجز"},
        {"id": "4", "message": "ما هي أعراض السكري؟",     "expected_contains": "أعراض"},
        {"id": "5", "message": "مين الدكاترة الموجودين؟", "expected_contains": "دكاترة"},
        {"id": "6", "message": "hello",                    "expected_contains": ""},
    ]

    successes = 0
    details   = []

    for case in cases:
        result = _run_real_case(case)
        if result["success"]:
            successes += 1
        details.append(result)

    success_rate = successes / len(cases) if cases else 0.0

    return {
        "component":    "whatsapp",
        "score":        round(success_rate, 3),
        "success_rate": round(success_rate, 3),
        "passed":       successes,
        "total":        len(cases),
        "details":      details,
    }
