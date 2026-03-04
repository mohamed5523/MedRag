# Evaluation Pipeline Documentation

This document explains the comprehensive evaluation pipeline for `heal-query-hub`, detailing its architecture, components, metrics, and how the underlying dataset `.json` files are used. 

---

## 1. Pipeline Overview

The evaluation suite (located in `evaluation/`) tests the system's ability to handle audio, generate responses, retrieve medical documents, and correctly identify intents (such as booking appointments). 

The evaluation can run in two modes:
1. **Mock Mode (`--mock`)**: Uses local, pre-defined mock outputs from the JSON dataset files. This requires no network connection or running backend.
2. **Real Mode**: Sends actual HTTP requests to the backend (by default, `BACKEND_URL` in `evaluation/config.py`, or override with `--provider`).

The orchestrator script is `evaluation/run_all_evaluations.py`.

### Example Usages:
```bash
# Run everything using mock data (synthetic local tests)
python3 evaluation/run_all_evaluations.py --mock

# Run against a specific running backend URL
python3 evaluation/run_all_evaluations.py --provider http://localhost:8000

# Run a specific component only, with verbose output
python3 evaluation/run_all_evaluations.py --component llm --verbose
```

---

## 2. Architecture & The Base Evaluator

All component evaluators inherit from **`BaseEvaluator`** (`evaluation/base_evaluator.py`). 

### Provider Injection
The architecture supports **Provider Injection**. Instead of hardcoding the evaluation to always hit the REST API, every evaluator accepts an optional `provider_fn` callable.
- **If `provider_fn` is provided**: The evaluator calls this function instead of making HTTP requests. This is useful for passing an alternative model locally or wrapping a different API.
- **If `provider_fn` is missing**: It defaults to hitting the backend API using `httpx`.

---

## 3. Evaluator Components & Metrics

The suite consists of several evaluators testing different stages of the system. Metrics are implemented from scratch in pure Python inside `evaluation/metrics.py`.

### 1. ASR (Speech-to-Text) - `eval_asr.py`
- **Purpose**: Evaluates how accurately spoken audio is transcribed into text.
- **Metrics**: 
  - **WER (Word Error Rate)**: Fraction of words substituted, inserted, or deleted. (Lower is better).
  - **CER (Character Error Rate)**: Character-level Levenshtein distance. (Lower is better).
  - **ROUGE-L**: Measures longest-common-subsequence overlap between transcribed text and reference.
- **Dataset**: `dataset/asr_samples.json`

### 2. TTS (Text-to-Speech) - `eval_tts.py`
- **Purpose**: Tests latency and generation performance of synthesizing Arabic speech.
- **Metrics**:
  - **RTF (Real-Time Factor)**: Ratio of processing latency to estimated audio duration. If RTF > 1.0, generation took longer than the speech itself (which means it's slower than real-time).
  - **Latency Percentiles**: p50, p95, p99.
  - **Bytes per char**: Measures audio naturalness/density.
- **Dataset**: `dataset/tts_samples.json`

### 3. LLM (Text Generation) - `eval_llm.py`
- **Purpose**: Tests the quality of generated answers given a textual question.
- **Metrics**:
  - **ROUGE-1, ROUGE-2, ROUGE-L**: N-gram overlaps and sub-sequence matching scoring.
  - **METEOR Approx**: Harmonic mean of unigram precision and recall with a brevity penalty.
- **Dataset**: `dataset/llm_queries.json`

### 4. MCP (Intent & Entity Routing) - `eval_mcp.py`
- **Purpose**: Tests if the system successfully routes queries to the "Clinic Operations" domain (e.g., booking an appointment) and extracts the right entities.
- **Metrics**:
  - **Intent Accuracy**: Fraction of correctly identified intents. 
  - **Entity Slot F1**: F1-score matching expected extracted entities (e.g., {"doctor": "احمد", "specialty": "قلب"}). Applies Arabic normalization (like mapping "أحمد" to "احمد").
- **Dataset**: `dataset/mcp_queries.json`

### 5. RAG (Document Retrieval) - `eval_rag.py`
- **Purpose**: Tests the semantic search component's ability to fetch the right medical documents from the knowledge base.
- **Metrics**: 
  - **Precision, Recall, F1**: Base set overlap of retrieved documents.
  - **MRR (Mean Reciprocal Rank)**: Emphasizes getting the first relevant document as early as possible.
  - **NDCG@5**: Normalized Discounted Cumulative Gain - considers the ranking quality of the top 5 documents.
  - **Hit Rate @ 1/3/5**: Percentage of queries with at least 1 correct doc in the top K.
- **Dataset**: `dataset/rag_queries.json`

---

## 4. Understanding Dataset Files (JSON)

The evaluators rely on static `.json` files under `evaluation/dataset/`. Here is how they are structured and processed.

### Global Fields
- `id`: Used purely for your own indexing/tracking.
- `note`: Optional comment (usually for the human reviewer).

### `asr_samples.json`
```json
{
  "reference": "الدكتور أحمد محمد متاح يوم الأحد",
  "mock_hypothesis": "الدكتور أحمد محمد متاح يوم الأحد",
  "mock_latency_ms": 450,
  "audio_path": null
}
```
- `reference`: The golden/perfect Arabic transcript.
- `audio_path`: The local `.webm` or audio file sent to the backend realistically.
- `mock_hypothesis` & `mock_latency_ms`: Used to bypass the network in `--mock` mode to simulate ASR text output.

### `llm_queries.json`
```json
{
  "question": "ما هي أعراض السكري؟",
  "expected_answer": "أعراض السكري تشمل العطش الشديد والتبول المتكرر والتعب والجوع المستمر",
  "expected_keywords": ["عطش", "تبول", "تعب", "سكري"],
  "category": "medical_info"
}
```
- `question`: The input fed to the LLM.
- `expected_answer`: The golden response used to calculate METEOR and ROUGE against the actual LLM output.
- `expected_keywords`: Ensures specific critical terms appear in the generation.
- `category`: Used to bucket evaluation performance (e.g., `schedule` vs `medical_info`).

### `mcp_queries.json`
```json
{
  "question": "احجز موعد مع دكتور عظام",
  "expected_intent": "book_appointment",
  "expected_mode": "mcp",
  "expected_entity": {
    "specialty": "عظام"
  }
}
```
- `expected_intent`: The intent the system *should* detect (e.g., `check_availability`, `ask_price`, `book_appointment`).
- `expected_mode`: Typically `"mcp"` (Model Context Protocol). Ensures standard RAG wasn't incorrectly invoked.
- `expected_entity`: A dictionary of variables that should be correctly extracted from the text by the underlying logic.

### `rag_queries.json`
```json
{
  "question": "ما هو علاج ارتفاع ضغط الدم؟",
  "ground_truth_sources": ["hypertension_treatment.pdf"],
  "mock_retrieved": ["hypertension_treatment.pdf", "cardiology_overview.pdf"]
}
```
- `ground_truth_sources`: An array of correct document IDs/names.
- `mock_retrieved`: What the simulator pretends the vector DB returned. In real mode, the actual retrieved list is compared against the `ground_truth_sources` using MRR/NDCG algorithms.

---

## Final Note
The evaluation uses a **Composite Score** calculation inside `run_all_evaluations.py`, calculating an unbalanced geometric mean prioritizing critical components heavily (e.g., LLM and ASR receive heavier weighting than the standalone TTS test score).

> **A note on your real-run failure (`--provider http://localhost:8000`)**: If the backend is not running at that port or is unresponsive, you will get a score of `0.000` consistently across all tools, since `httpx` encounters connection timeouts.
