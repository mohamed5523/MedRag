"""
evaluation/run_all_evaluations.py
-----------------------------------
Production orchestrator for the MedRAG evaluation suite.

Real mode ONLY — no mock flags, no synthetic fallbacks.
Exits with code 1 if ANY of the following fails:
  a) A per-component threshold (defined in config.THRESHOLDS) is missed
  b) The weighted composite score < config.COMPOSITE_PASS_THRESHOLD (default 0.85)

Usage:
    # All components against default BACKEND_URL (http://localhost:8000)
    python3 evaluation/run_all_evaluations.py

    # Override backend URL
    python3 evaluation/run_all_evaluations.py --provider http://staging.internal:8000

    # Single component (verbose)
    python3 evaluation/run_all_evaluations.py --component llm --verbose

    # Custom results file
    python3 evaluation/run_all_evaluations.py --output /tmp/ci_results.json

Environment variables:
    BACKEND_URL              Backend to test against
    JUDGE_URL                OpenAI-compatible judge endpoint (for LLM eval)
    JUDGE_API_KEY            API key for the judge
    JUDGE_MODEL              Judge model name (default: gpt-4o)
    COMPOSITE_PASS_THRESHOLD Override pass threshold (default 0.85)

Exit codes:
    0 — all thresholds met, composite score ≥ COMPOSITE_PASS_THRESHOLD
    1 — one or more components missed threshold OR composite below gate
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
EVAL_DIR  = Path(__file__).parent
REPO_ROOT = EVAL_DIR.parent
BACKEND_DIR = REPO_ROOT / "backend"

for _p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.config import (  # noqa: E402
    COMPOSITE_PASS_THRESHOLD,
    COMPOSITE_WEIGHTS,
    RESULTS_FILE,
    THRESHOLDS,
)

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt  = "%H:%M:%S",
    stream   = sys.stderr,
)
logger = logging.getLogger("eval.runner")

# ── Component registry ─────────────────────────────────────────────────────────
COMPONENTS = ["tts", "asr", "llm", "mcp", "rag"]


def _run_component(name: str, provider_url: str | None = None) -> dict:
    """Import and run a single eval module. Never raises — errors captured in result."""
    if provider_url:
        os.environ["BACKEND_URL"] = provider_url

    try:
        if name == "tts":
            from evaluation.eval_tts import run_eval
        elif name == "asr":
            from evaluation.eval_asr import run_eval
        elif name == "llm":
            from evaluation.eval_llm import run_eval
        elif name == "mcp":
            from evaluation.eval_mcp import run_eval
        elif name == "rag":
            from evaluation.eval_rag import run_eval
        else:
            return {"component": name, "error": f"Unknown: {name}", "score": 0.0}

        start  = time.monotonic()
        result = run_eval()
        result["duration_s"] = round(time.monotonic() - start, 2)
        return result

    except Exception as e:
        import traceback
        logger.exception("Component [%s] raised an unhandled exception", name)
        return {
            "component":  name,
            "error":      str(e),
            "traceback":  traceback.format_exc(),
            "score":      0.0,
            "duration_s": 0.0,
        }


# ── Threshold evaluation ────────────────────────────────────────────────────────

def _threshold_status(results: list) -> dict:
    status = {}
    for r in results:
        name  = r.get("component", "?")
        score = r.get("score", 0.0)
        checks = {
            "llm":       ("llm_judge_score_min",      score >= THRESHOLDS.get("llm_judge_score_min", 0.70)),
            "asr":       ("asr_wer_max",               r.get("avg_wer", 1.0) <= THRESHOLDS.get("asr_wer_max", 0.30)),
            "mcp":       ("mcp_intent_accuracy_min",   score >= THRESHOLDS.get("mcp_intent_accuracy_min", 0.80)),
            "rag":       ("rag_precision_min",          r.get("avg_precision", 0.0) >= THRESHOLDS.get("rag_precision_min", 0.60)),
            "tts":       ("tts_latency_p95_ms",         r.get("latency", {}).get("p95", 9999) <= THRESHOLDS.get("tts_latency_p95_ms", 3000)),
            "whatsapp":  ("whatsapp_success_rate_min",  score >= THRESHOLDS.get("whatsapp_success_rate_min", 0.90)),
        }
        if name in checks:
            thr_key, passed = checks[name]
            status[name] = {
                "passed":          passed,
                "threshold_key":   thr_key,
                "threshold_value": THRESHOLDS.get(thr_key, "N/A"),
                "score":           score,
            }
        else:
            status[name] = {"passed": True, "score": score}
    return status


# ── Composite score ─────────────────────────────────────────────────────────────

def _composite(results: list) -> float:
    w_sum = s_sum = 0.0
    for r in results:
        w = COMPOSITE_WEIGHTS.get(r.get("component", ""), 0.0)
        if w > 0:
            s_sum += w * r.get("score", 0.0)
            w_sum += w
    return round(s_sum / w_sum, 3) if w_sum > 0 else 0.0


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="MedRAG Production Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--component", choices=COMPONENTS + ["all"], default="all",
        help="Which component to evaluate (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Results JSON file path (default: evaluation/results.json)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print top-3 sample details per component",
    )
    parser.add_argument(
        "--provider", type=str, default=None, metavar="URL",
        help="Override BACKEND_URL (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--pass-threshold", type=float, default=None,
        help=f"Override composite pass threshold (default: {COMPOSITE_PASS_THRESHOLD})",
    )
    args = parser.parse_args()

    pass_threshold = args.pass_threshold or COMPOSITE_PASS_THRESHOLD
    components_to_run = COMPONENTS if args.component == "all" else [args.component]

    print(f"\n{'═'*72}")
    print(f"  MedRAG Evaluation Suite  —  REAL MODE")
    if args.provider:
        print(f"  Provider:   {args.provider}")
    print(f"  Components: {', '.join(components_to_run)}")
    print(f"  Pass gate:  composite ≥ {pass_threshold:.2f}")
    print(f"{'═'*72}\n")

    all_results = []
    for name in components_to_run:
        print(f"▶ [{name}] …", end=" ", flush=True)
        result   = _run_component(name, provider_url=args.provider)
        score    = result.get("score",    0.0)
        duration = result.get("duration_s", 0.0)
        error    = result.get("error",    "")

        if error:
            print(f"ERROR — {error}")
        else:
            print(f"score={score:.3f}  ({duration:.2f}s)")

        if args.verbose and not error:
            for d in result.get("details", [])[:3]:
                print(f"    ↳ {d}")

        all_results.append(result)

    # ── Results ─────────────────────────────────────────────────────────────────
    threshold_status = _threshold_status(all_results)
    composite        = _composite(all_results)
    components_pass  = all(v["passed"] for v in threshold_status.values())
    composite_pass   = composite >= pass_threshold
    overall_pass     = components_pass and composite_pass

    summary = {
        "run_at":           datetime.now(timezone.utc).isoformat(),
        "mode":             "real",
        "provider":         args.provider or os.getenv("BACKEND_URL", ""),
        "composite_score":  composite,
        "composite_passed": composite_pass,
        "pass_threshold":   pass_threshold,
        "all_passed":       overall_pass,
        "thresholds":       threshold_status,
        "results":          all_results,
    }

    output_path = Path(args.output) if args.output else RESULTS_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── Summary table ────────────────────────────────────────────────────────────
    print(f"\n{'─'*76}")
    print(f"  {'Component':<14} {'Score':>7}  {'Threshold Key':<28}  {'Status'}")
    print(f"{'─'*76}")
    for r in all_results:
        name = r.get("component", "?")
        ts   = threshold_status.get(name, {})
        print(f"  {name:<14} {ts.get('score',0):.3f}   {ts.get('threshold_key',''):<28}  "
              f"{'✅ PASS' if ts.get('passed') else '❌ FAIL'}")
    print(f"{'─'*76}")
    print(f"\n  Composite weighted score:  {composite:.3f}  "
          f"{'✅ ≥' if composite_pass else '❌ <'} {pass_threshold:.2f}")
    print(f"  Overall verdict:  {'✅  GO  — all thresholds met' if overall_pass else '❌  NO-GO — thresholds failed'}")
    print(f"  Results written:  {output_path}\n")

    logger.info("Evaluation complete — composite=%.3f  pass=%s", composite, overall_pass)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
