"""
evaluation/config.py
--------------------
Configuration for the MedRAG evaluation framework.
Production-grade: real mode only, no mock fallbacks.

All settings are overridable via environment variables.
Auto-loads backend/.env so you never need to duplicate API keys.
"""

import os
from pathlib import Path

# ── Auto-load backend/.env ─────────────────────────────────────────────────────
# Populate os.environ from backend/.env so OPENAI_API_KEY etc. are available
# without requiring callers to export them manually.
_BACKEND_ENV = Path(__file__).parent.parent / "backend" / ".env"
if _BACKEND_ENV.exists():
    for _line in _BACKEND_ENV.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        _k = _k.strip()
        _v = _v.strip().strip('"').strip("'")
        # Only set if not already overridden by a real shell export
        os.environ.setdefault(_k, _v)

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_ROOT   = Path(__file__).parent
DATASET_DIR = EVAL_ROOT / "dataset"
RESULTS_FILE = EVAL_ROOT / "results.json"

TTS_SAMPLES  = DATASET_DIR / "tts_samples.json"
ASR_SAMPLES  = DATASET_DIR / "asr_samples.json"
LLM_QUERIES  = DATASET_DIR / "llm_queries.json"
MCP_QUERIES  = DATASET_DIR / "mcp_queries.json"
RAG_QUERIES  = DATASET_DIR / "rag_queries.json"

# Audio files for ASR (relative paths in asr_samples.json are resolved here)
ASR_AUDIO_DIR = Path(os.getenv("ASR_AUDIO_DIR", str(DATASET_DIR / "audio")))

# ── Backend ────────────────────────────────────────────────────────────────────
BACKEND_URL  = os.getenv("BACKEND_URL",  "http://localhost:8000")

# ── LLM-as-a-Judge ─────────────────────────────────────────────────────────────
# Reads JUDGE_* vars, falling back to the backend's OPENAI_API_KEY / model.
# Priority: JUDGE_API_KEY → OPENAI_API_KEY (loaded from backend/.env above)
JUDGE_URL    = os.getenv("JUDGE_URL",    "https://api.openai.com/v1/chat/completions")
JUDGE_MODEL  = os.getenv("JUDGE_MODEL",  os.getenv("OPENAI_LLM_MODEL", "gpt-4o"))
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# ── Score thresholds ──────────────────────────────────────────────────────────
THRESHOLDS = {
    "tts_latency_p95_ms":       3000,   # 95th percentile TTS latency (ms)
    "asr_wer_max":              0.30,   # Max acceptable WER (30%)
    "llm_judge_score_min":      0.70,   # Min LLM-as-a-Judge combined score
    "mcp_intent_accuracy_min":  0.80,   # Min intent exact-match accuracy
    "mcp_slot_f1_min":          0.75,   # Min required-slot F1
    "rag_precision_min":        0.60,   # Min retrieval precision
    "whatsapp_success_rate_min": 0.90,  # Min end-to-end success rate
}

# ── Composite pass threshold ───────────────────────────────────────────────────
# If the weighted composite score across all components falls below this,
# run_all_evaluations.py exits with code 1 (CI/CD fail gate).
COMPOSITE_PASS_THRESHOLD = float(os.getenv("COMPOSITE_PASS_THRESHOLD", "0.85"))

# Component weights for composite score
COMPOSITE_WEIGHTS = {
    "llm":      0.30,
    "asr":      0.20,
    "mcp":      0.20,
    "rag":      0.20,
    "tts":      0.05,
    "whatsapp": 0.05,
}

# ── Provider / model labels ───────────────────────────────────────────────────
LLM_MODEL    = os.getenv("LLM_MODEL",    "gpt-4o")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "openai")
ASR_PROVIDER = os.getenv("ASR_PROVIDER", "groq")
