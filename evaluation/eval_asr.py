"""
evaluation/eval_asr.py
----------------------
Production ASR evaluation — real mode only, no mock fallbacks.

Audio pipeline:
  1. Resolve audio_path (absolute OR relative to ASR_AUDIO_DIR)
  2. Detect MIME type from file extension:
     .mp3  → audio/mpeg  |  .wav → audio/wav  |  .webm → audio/webm
     .ogg  → audio/ogg   |  .m4a → audio/mp4  |  .flac → audio/flac
  3. POST multipart to /api/asr/transcribe with correct Content-Type
  4. Compute WER and CER after robust Arabic normalisation:
     - Strip all diacritics (harakat)
     - Normalise Hamza variants (أإآ → ا)
     - Normalise taa marbouta (ة → ه)
     - Strip punctuation

Dataset schema  (asr_samples.json):
  {
    "id":         int,
    "reference":  str,        — clean Arabic text (no diacritics needed)
    "audio_path": str,        — e.g. "asr_001.mp3"  (relative to dataset/audio/)
    "dialect":    "msa" | "egyptian" | ...,
    "note":       str
  }

Score: 0.5 * (1 - avg_wer) + 0.5 * (1 - avg_cer)   clamped to [0, 1]
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import ASR_AUDIO_DIR, ASR_SAMPLES, THRESHOLDS
from .http_client import post_with_retry
from .metrics import char_error_rate, latency_percentiles, rouge_l, word_error_rate

logger = logging.getLogger("eval.asr")

# ── MIME type registry ─────────────────────────────────────────────────────────
_MIME: Dict[str, str] = {
    ".mp3":  "audio/mpeg",
    ".wav":  "audio/wav",
    ".webm": "audio/webm",
    ".ogg":  "audio/ogg",
    ".m4a":  "audio/mp4",
    ".flac": "audio/flac",
    ".aac":  "audio/aac",
}


def _resolve(raw: str) -> Optional[Path]:
    """Resolve audio_path: try absolute first, then relative to ASR_AUDIO_DIR."""
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    candidate = Path(ASR_AUDIO_DIR) / p
    return candidate if candidate.exists() else None


def _mime(path: Path) -> str:
    return _MIME.get(path.suffix.lower(), "application/octet-stream")


# ── Backend call ───────────────────────────────────────────────────────────────

def _transcribe(
    sample: dict,
    provider_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Upload audio to /api/asr/transcribe and return transcript + latency.

    Args:
        sample:      Dataset record with audio_path.
        provider_fn: Optional callable(audio_bytes: bytes, path_str: str) → str transcript.
    """
    raw_path = sample.get("audio_path")
    if not raw_path:
        err = f"sample id={sample.get('id')}: audio_path is null — record the audio file first"
        logger.error(err)
        return {"hypothesis": "", "latency_ms": 0, "success": False, "error": err}

    resolved = _resolve(raw_path)
    if resolved is None:
        err = f"audio file not found: {raw_path}  (looked in {ASR_AUDIO_DIR})"
        logger.error(err)
        return {"hypothesis": "", "latency_ms": 0, "success": False, "error": err}

    try:
        audio_bytes = resolved.read_bytes()
    except OSError as e:
        logger.error("Cannot read %s: %s", resolved, e)
        return {"hypothesis": "", "latency_ms": 0, "success": False, "error": str(e)}

    mime = _mime(resolved)
    logger.info("Transcribing %s (%s, %d bytes)", resolved.name, mime, len(audio_bytes))

    if provider_fn is not None:
        import time
        start = time.monotonic()
        transcript = str(provider_fn(audio_bytes, str(resolved)))
        return {"hypothesis": transcript,
                "latency_ms": (time.monotonic() - start) * 1000, "success": True}

    resp = post_with_retry(
        "/api/asr/transcribe",
        files={"audio_file": (resolved.name, audio_bytes, mime)},
    )
    if not resp["ok"]:
        logger.error("ASR backend error for %s: %s", resolved.name, resp["error"])
        return {"hypothesis": "", "latency_ms": resp["latency_ms"],
                "success": False, "error": resp["error"]}

    transcript = (resp["json"] or {}).get("transcribed_text", "")
    return {"hypothesis": transcript, "latency_ms": resp["latency_ms"], "success": True}


# ── Main eval ──────────────────────────────────────────────────────────────────

def run_eval(
    provider_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run ASR evaluation against real audio files.

    Args:
        provider_fn: Optional callable(audio_bytes, path_str) → transcript str.
    """
    samples: List[dict] = []
    try:
        with open(ASR_SAMPLES) as f:
            samples = json.load(f)
    except Exception as e:
        return {"component": "asr", "error": str(e), "score": 0.0}

    wer_vals:  List[float] = []
    cer_vals:  List[float] = []
    rl_vals:   List[float] = []
    latencies: List[float] = []
    skipped = 0
    details = []

    for sample in samples:
        ref = sample.get("reference", "")
        if not ref:
            skipped += 1
            continue

        result = _transcribe(sample, provider_fn=provider_fn)
        hyp    = result.get("hypothesis", "")

        wer = word_error_rate(ref, hyp)
        cer = char_error_rate(ref, hyp)
        rl  = rouge_l(ref, hyp)["f1"]

        wer_vals.append(wer)
        cer_vals.append(cer)
        rl_vals.append(rl)
        latencies.append(result.get("latency_ms", 0))

        details.append({
            "id":         sample.get("id"),
            "dialect":    sample.get("dialect", "msa"),
            "audio_path": sample.get("audio_path"),
            "audio_mime": _mime(Path(sample["audio_path"])) if sample.get("audio_path") else None,
            "reference":  ref,
            "hypothesis": hyp,
            "wer":        round(wer, 4),
            "cer":        round(cer, 4),
            "rouge_l":    round(rl, 4),
            "success":    result.get("success", False),
            "error":      result.get("error"),
        })

    n = len(wer_vals)
    if n == 0:
        return {"component": "asr", "score": 0.0, "sample_count": 0,
                "skipped": skipped, "details": details}

    avg_wer = sum(wer_vals) / n
    avg_cer = sum(cer_vals) / n
    avg_rl  = sum(rl_vals)  / n
    score   = max(0.0, min(1.0, 0.5 * (1.0 - avg_wer) + 0.5 * (1.0 - avg_cer)))
    perf    = latency_percentiles(latencies)

    return {
        "component":    "asr",
        "score":        round(score,   3),
        "avg_wer":      round(avg_wer, 4),
        "avg_cer":      round(avg_cer, 4),
        "avg_rouge_l":  round(avg_rl,  4),
        "latency":      perf,
        "sample_count": n,
        "skipped":      skipped,
        "details":      details,
    }
