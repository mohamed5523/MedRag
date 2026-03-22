"""
evaluation/eval_tts_asr.py
--------------------------
End-to-end evaluation of the TTS and ASR components combined.

Pipeline:
  1. Load ASR datasets (asr_samples.json) containing reference text.
  2. Synthesize audio from the reference text using the TTS API.
  3. Transcribe the resulting synthesized audio using the ASR API.
  4. Compare the transcribed text to the original reference text using WER, CER, and ROUGE-L.
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .config import ASR_SAMPLES
from .http_client import post_with_retry
from .metrics import char_error_rate, latency_percentiles, rouge_l, word_error_rate

logger = logging.getLogger("eval.tts_asr")


# ── Backend call ───────────────────────────────────────────────────────────────

import base64

def _run_pipeline_for_sample(text: str) -> Dict[str, Any]:
    """Run text through TTS -> Audio -> ASR -> Transcribed Text."""
    
    # 1. Synthesize Audio
    start_tts = time.monotonic()
    tts_resp = post_with_retry("/api/tts/synthesize", json={"text": text})
    lat_tts = (time.monotonic() - start_tts) * 1000

    if not tts_resp["ok"]:
        return {
            "hypothesis": "", 
            "latency_tts_ms": lat_tts,
            "latency_asr_ms": 0,
            "success": False, 
            "error": f"TTS Error: {tts_resp.get('error', 'Unknown HTTP Error')}"
        }

    try:
        audio_b64 = tts_resp["json"]["audio_data"]
        audio_bytes = base64.b64decode(audio_b64)
    except (KeyError, TypeError) as e:
        return {
            "hypothesis": "", 
            "latency_tts_ms": lat_tts,
            "latency_asr_ms": 0,
            "success": False, 
            "error": "TTS Error: Malformed JSON response, missing audio_data"
        }
    
    # 2. Transcribe Audio
    # We pass the bytes as 'audio_file' resembling an mp3
    start_asr = time.monotonic()
    asr_resp = post_with_retry(
        "/api/asr/transcribe",
        files={"audio_file": ("tts_output.mp3", audio_bytes, "audio/mpeg")},
    )
    lat_asr = (time.monotonic() - start_asr) * 1000

    if not asr_resp["ok"]:
        return {
            "hypothesis": "", 
            "latency_tts_ms": lat_tts,
            "latency_asr_ms": lat_asr,
            "success": False, 
            "error": f"ASR Error: {asr_resp.get('error', 'Unknown')}"
        }

    transcript = (asr_resp["json"] or {}).get("transcribed_text", "")
    return {
        "hypothesis": transcript,
        "latency_tts_ms": lat_tts,
        "latency_asr_ms": lat_asr,
        "success": True
    }


# ── Main eval ──────────────────────────────────────────────────────────────────

def run_eval() -> Dict[str, Any]:
    """Run end-to-end TTS+ASR evaluation."""
    samples: List[dict] = []
    try:
        with open(ASR_SAMPLES) as f:
            samples = json.load(f)
    except Exception as e:
        return {"component": "tts_asr", "error": str(e), "score": 0.0}

    wer_vals:  List[float] = []
    cer_vals:  List[float] = []
    rl_vals:   List[float] = []
    tts_latencies: List[float] = []
    asr_latencies: List[float] = []
    skipped = 0
    successes = 0
    details = []

    for sample in samples:
        ref = sample.get("reference", "")
        if not ref:
            skipped += 1
            continue

        result = _run_pipeline_for_sample(ref)
        hyp = result.get("hypothesis", "")

        if result.get("success", False):
            successes += 1
            
            wer = word_error_rate(ref, hyp)
            cer = char_error_rate(ref, hyp)
            rl  = rouge_l(ref, hyp)["f1"]

            wer_vals.append(wer)
            cer_vals.append(cer)
            rl_vals.append(rl)
            
            tts_latencies.append(result.get("latency_tts_ms", 0))
            asr_latencies.append(result.get("latency_asr_ms", 0))

            details.append({
                "id":         sample.get("id"),
                "text_preview": ref[:60],
                "reference":  ref,
                "hypothesis": hyp,
                "wer":        round(wer, 4),
                "cer":        round(cer, 4),
                "rouge_l":    round(rl, 4),
                "latency_tts_ms": round(result.get("latency_tts_ms", 0), 1),
                "latency_asr_ms": round(result.get("latency_asr_ms", 0), 1),
                "success":    True,
            })
        else:
            details.append({
                "id":         sample.get("id"),
                "text_preview": ref[:60],
                "reference":  ref,
                "hypothesis": "",
                "success":    False,
                "error":      result.get("error"),
            })

    n = len(wer_vals)
    total_samples = len(details)

    if n == 0:
        return {
            "component": "tts_asr", 
            "score": 0.0, 
            "sample_count": total_samples,
            "success_rate": 0.0,
            "skipped": skipped, 
            "details": details
        }

    avg_wer = sum(wer_vals) / n
    avg_cer = sum(cer_vals) / n
    avg_rl  = sum(rl_vals)  / n
    success_rate = successes / total_samples
    
    # Score favors getting it right + doing it consistently
    # 0.5 * WER accuracy + 0.5 * CER accuracy, modulated by success rate
    score_accuracy = max(0.0, min(1.0, 0.5 * (1.0 - avg_wer) + 0.5 * (1.0 - avg_cer)))
    score = score_accuracy * success_rate

    perf_tts = latency_percentiles(tts_latencies)
    perf_asr = latency_percentiles(asr_latencies)

    return {
        "component":    "tts_asr",
        "score":        round(score,   3),
        "success_rate": round(success_rate, 3),
        "avg_wer":      round(avg_wer, 4),
        "avg_cer":      round(avg_cer, 4),
        "avg_rouge_l":  round(avg_rl,  4),
        "latency_tts":  perf_tts,
        "latency_asr":  perf_asr,
        "sample_count": total_samples,
        "skipped":      skipped,
        "details":      details,
    }
