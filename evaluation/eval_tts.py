"""
evaluation/eval_tts.py
----------------------
Evaluate TTS latency, RTF (Real-Time Factor), and synthesis success.

Real mode: calls /api/tts/synthesize via shared robust HTTP client.
Mock mode: simulates latency proportional to text length.
"""

import json
from typing import Any, Callable, Dict, List, Optional

from .config import BACKEND_URL, TTS_SAMPLES
from .http_client import post_with_retry
from .metrics import latency_percentiles

_CHARS_PER_SECOND = 4.0  # rough estimate: Egyptian Arabic TTS


def _estimate_audio_s(text: str) -> float:
    return len(text) / _CHARS_PER_SECOND


# ── Mock mode ──────────────────────────────────────────────────────────────────

def _mock_synthesize(text: str) -> Dict[str, Any]:
    simulated_ms = min(len(text) * 10, 800)
    return {"latency_ms": simulated_ms, "audio_size_bytes": len(text) * 80,
            "provider": "mock", "success": True}


# ── Real mode ──────────────────────────────────────────────────────────────────

def _real_synthesize(
    text: str,
    provider_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    if provider_fn is not None:
        import time
        start = time.monotonic()
        result = provider_fn(text)
        lat = (time.monotonic() - start) * 1000
        audio = result.get("audio_bytes", b"")
        return {"latency_ms": lat, "audio_size_bytes": len(audio),
                "provider": "custom", "success": True}

    resp = post_with_retry("/api/tts/synthesize", json={"text": text})
    if not resp["ok"]:
        return {"latency_ms": resp["latency_ms"], "audio_size_bytes": 0,
                "provider": "real", "success": False, "error": resp["error"]}

    audio_bytes = resp["raw"] or b""
    return {"latency_ms": resp["latency_ms"], "audio_size_bytes": len(audio_bytes),
            "provider": "real", "success": True}


# ── Main eval ──────────────────────────────────────────────────────────────────

def run_eval(
    provider_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:

    samples: List[dict] = []
    try:
        with open(TTS_SAMPLES) as f:
            samples = json.load(f)
    except Exception as e:
        return {"component": "tts", "error": str(e), "score": 0.0}

    latencies: List[float] = []
    rtf_values: List[float] = []
    bpc_values: List[float] = []
    successes = 0
    results_detail = []

    for sample in samples:
        text = sample.get("text", "")
        if not text:
            continue

        result = _real_synthesize(text, provider_fn)

        lat_ms  = result["latency_ms"]
        lat_s   = lat_ms / 1000.0
        ab      = result.get("audio_size_bytes", 0)
        est_s   = _estimate_audio_s(text)
        rtf     = est_s / lat_s if lat_s > 0 else float("inf")
        bpc     = ab / len(text) if len(text) > 0 else 0.0

        latencies.append(lat_ms)
        rtf_values.append(rtf)
        bpc_values.append(bpc)
        if result["success"]:
            successes += 1

        results_detail.append({
            "text_preview":  text[:60],
            "latency_ms":    round(lat_ms, 1),
            "rtf":           round(rtf, 3),
            "bytes_per_char": round(bpc, 1),
            "success":       result["success"],
            "error":         result.get("error"),
        })

    n = len(results_detail)
    if n == 0:
        return {"component": "tts", "score": 0.0, "sample_count": 0, "details": []}

    perf         = latency_percentiles(latencies)
    success_rate = successes / n
    avg_rtf      = sum(rtf_values) / n
    avg_bpc      = sum(bpc_values) / n
    # Use trimmed mean (drop highest outlier when n>3) to prevent cold-start zeros
    sorted_lat   = sorted(latencies)
    trim_lats    = sorted_lat[:-1] if n > 3 else sorted_lat
    trim_mean    = sum(trim_lats) / len(trim_lats)
    lat_factor   = max(0.0, 1.0 - trim_mean / 8000.0)
    rtf_penalty  = 0.1 if 0 < avg_rtf < 1.0 else 0.0
    score        = success_rate * lat_factor * (1.0 - rtf_penalty)

    return {
        "component":       "tts",
        "score":           round(score, 3),
        "success_rate":    round(success_rate, 3),
        "avg_rtf":         round(avg_rtf, 3),
        "avg_bytes_per_char": round(avg_bpc, 1),
        "latency":         perf,
        "sample_count":    n,
        "details":         results_detail,
    }
