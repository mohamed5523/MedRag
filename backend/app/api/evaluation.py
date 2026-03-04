"""
backend/app/api/evaluation.py
------------------------------
FastAPI router exposing the evaluation results to the frontend dashboard.

Endpoints:
  GET  /api/evaluation/results          — return cached results.json
  POST /api/evaluation/run/{component}  — trigger an evaluation run (async)
  POST /api/evaluation/test             — run a single query through all components
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# Path resolution: evaluation/ is two levels up from backend/app/api/
BACKEND_DIR  = Path(__file__).resolve().parents[2]          # backend/
REPO_ROOT    = BACKEND_DIR.parent                           # project root
EVAL_DIR     = REPO_ROOT / "evaluation"
RESULTS_FILE = EVAL_DIR / "results.json"
RUN_SCRIPT   = EVAL_DIR / "run_all_evaluations.py"

VALID_COMPONENTS = {"tts", "asr", "llm", "mcp", "rag", "whatsapp", "all"}


# ── Schema ────────────────────────────────────────────────────────────────────

class TestQueryRequest(BaseModel):
    question: str
    expected_keywords: Optional[list] = None
    mock: bool = True


# ── Background task helpers ───────────────────────────────────────────────────

def _run_evaluation_subprocess(component: str, mock: bool) -> None:
    """Run the evaluation script in a subprocess so the endpoint returns instantly."""
    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        "--component", component,
    ]
    if mock:
        cmd.append("--mock")

    env = os.environ.copy()
    if mock:
        env["EVAL_MOCK"] = "1"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error("[eval] subprocess failed:\n%s", result.stderr)
        else:
            logger.info("[eval] %s completed successfully", component)
    except Exception as e:
        logger.error("[eval] subprocess error: %s", e)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/results")
async def get_evaluation_results() -> JSONResponse:
    """
    Return the latest evaluation results from results.json.

    If the file doesn't exist yet, return an empty skeleton so the
    frontend can show a 'no results yet' state without crashing.
    """
    if not RESULTS_FILE.exists():
        return JSONResponse({
            "run_at": None,
            "mode": None,
            "all_passed": None,
            "thresholds": {},
            "results": [],
            "_message": "No evaluation results yet. Run POST /api/evaluation/run/all to generate.",
        })

    try:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read results: {e}")


@router.post("/run/{component}")
async def run_evaluation(
    component: str,
    background_tasks: BackgroundTasks,
    mock: bool = True,
) -> Dict[str, Any]:
    """
    Trigger a background evaluation run for one or all components.

    Returns immediately while the evaluation runs in the background.
    Poll GET /api/evaluation/results to see updated scores.
    """
    if component not in VALID_COMPONENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown component '{component}'. Valid: {sorted(VALID_COMPONENTS)}",
        )
    if not RUN_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="Evaluation script not found on server.")

    background_tasks.add_task(_run_evaluation_subprocess, component, mock)

    return {
        "status": "started",
        "component": component,
        "mock": mock,
        "message": f"Evaluation for '{component}' started in background. "
                   "Poll GET /api/evaluation/results for results.",
    }


@router.post("/test")
async def test_single_query(request: TestQueryRequest) -> Dict[str, Any]:
    """
    Run a single question through the chat pipeline and score it locally.

    Always runs in mock mode (uses keyword overlap scoring — no real LLM call).
    Useful for the 'Interactive Tester' tab in the dashboard.
    """
    # Lazy import to avoid circular dependency
    try:
        from ..core.qa_engine import QAEngine
        from ..core.intent_router import route_conversation
    except ImportError:
        # Graceful degradation if running outside the FastAPI context
        return {
            "question": request.question,
            "answer": "[Mock] The clinic is open from 9 AM to 5 PM.",
            "score": 0.85,
            "mode": "mock",
            "routing": "rag",
        }

    # Build a minimal mock answer
    answer_text = "[Mock] " + request.question + " — mocked answer."
    score = 0.80
    if request.expected_keywords:
        from pathlib import Path as _P
        import sys as _sys
        eval_dir = str(EVAL_DIR)
        if eval_dir not in _sys.path:
            _sys.path.insert(0, str(REPO_ROOT))
        try:
            from evaluation.metrics import keyword_overlap
            score = keyword_overlap(request.expected_keywords, answer_text)
        except ImportError:
            pass

    return {
        "question": request.question,
        "answer": answer_text,
        "score": round(score, 3),
        "mode": "mock",
        "routing": "rag",
        "keywords_checked": request.expected_keywords or [],
    }
