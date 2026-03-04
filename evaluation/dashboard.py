"""
evaluation/dashboard.py
-----------------------
Streamlit evaluation dashboard for MedRAG.

Run from project root:
    streamlit run evaluation/dashboard.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st

# ── Project root on path ───────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedRAG Eval",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0d1117 0%,#161b22 60%,#0d1117 100%);}

/* Header */
.hdr{background:linear-gradient(90deg,#1f6feb,#388bfd);border-radius:16px;
     padding:24px 32px;margin-bottom:20px;color:#fff;}
.hdr h1{margin:0;font-size:2rem;font-weight:700;}
.hdr p{margin:4px 0 0;opacity:.85;font-size:.95rem;}

/* Cards */
.ecard{background:linear-gradient(135deg,#161b22,#1c2128);border:1px solid #30363d;
       border-radius:14px;padding:20px;text-align:center;transition:.2s;}
.ecard:hover{border-color:#388bfd;transform:translateY(-2px);}
.eicon{font-size:2.2rem;margin-bottom:6px;}
.ename{font-weight:600;color:#c9d1d9;margin-bottom:3px;}
.edesc{font-size:.75rem;color:#8b949e;}

/* Score colors */
.spass{color:#3fb950;font-weight:700;font-size:1.5rem;}
.sfail{color:#f85149;font-weight:700;font-size:1.5rem;}
.swarn{color:#d29922;font-weight:700;font-size:1.5rem;}

/* Report card */
.rcard{background:#161b22;border:1px solid #30363d;border-radius:12px;
       padding:20px;margin-bottom:14px;}

/* Buttons */
.stButton>button{background:linear-gradient(90deg,#1f6feb,#388bfd);
    color:#fff;border:none;border-radius:10px;font-weight:600;
    width:100%;padding:10px;transition:.2s;}
.stButton>button:hover{opacity:.88;transform:translateY(-1px);}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #21262d;}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for _k, _v in {"results": {}, "running": set()}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Load backend .env ──────────────────────────────────────────────────────────
_BENV = _ROOT / "backend" / ".env"

def _env(key: str, default: str = "") -> str:
    val = os.environ.get(key, "")
    if val:
        return val
    if _BENV.exists():
        for line in _BENV.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    return default

# ── Report renderer ────────────────────────────────────────────────────────────
def render_report(comp: str, r: Dict[str, Any]):
    """Render a detailed, interpreted report for a component result."""

    if not r:
        st.warning("No results yet — run the evaluation first.")
        return
    if r.get("error") and r.get("score", 0) == 0:
        st.error(f"❌ Evaluation failed: {r['error']}")
        return

    score = r.get("score", 0.0)

    # ── TTS ───────────────────────────────────────────────────────────────────
    if comp == "tts":
        st.subheader("🔊 TTS — Text-to-Speech Quality")
        c1, c2, c3, c4 = st.columns(4)
        lat = r.get("latency", {})
        c1.metric("Overall score",    f"{score:.3f}", help="0–1, higher is better")
        c2.metric("Success rate",     f"{r.get('success_rate',0):.0%}", help="Requests that returned audio")
        c3.metric("Median latency",   f"{lat.get('p50',0)/1000:.2f}s")
        c4.metric("p95 latency",      f"{lat.get('p95',0)/1000:.2f}s")

        st.markdown("---")
        st.markdown("""
**📖 What these numbers mean:**

| Metric | Target | Meaning |
|--------|--------|---------|
| Score | > 0.70 | `success_rate × latency_factor × rtf_factor` — composite quality |
| Success rate | > 0.95 | Did the TTS API return audio for every request? |
| Median latency | < 1.5 s | Half of synthesis requests finished in this time |
| p95 latency | < 3 s | 95% of requests are faster than this (the worst-case user experience) |

A score of **0 usually means one extremely slow request (Docker cold-start)**
skewed the trimmed mean above the ceiling. Re-run once the backend is warm.
""")
        st.markdown("**💡 How to improve:**")
        tips = []
        if lat.get("p95", 0) > 3000:
            tips.append("🔴 **p95 > 3 s** — switch to a streaming TTS endpoint or a faster provider (OpenAI TTS is usually fastest).")
        if r.get("success_rate", 1) < 0.95:
            tips.append("🔴 **Success < 95%** — check TTS API rate limits and key expiry.")
        if r.get("avg_rtf", 1) < 1:
            tips.append("🟡 **RTF < 1× real-time** — the model generates audio slower than playback speed; try `gpt-4o-mini-tts`.")
        if not tips:
            tips = ["✅ TTS is performing well!"]
        for t in tips:
            st.markdown(t)

        with st.expander("📋 Per-sample details"):
            for d in r.get("details", []):
                ok = d.get("success", False)
                st.markdown(
                    f"{'✅' if ok else '❌'} `{d.get('latency_ms',0)/1000:.2f}s` "
                    f"— {d.get('text_preview','')[:80]}"
                )

    # ── ASR ───────────────────────────────────────────────────────────────────
    elif comp == "asr":
        st.subheader("🎙️ ASR — Speech Recognition Quality")
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall score", f"{score:.3f}")
        c2.metric("Mean WER",  f"{r.get('avg_wer',0):.1%}",  help="Word Error Rate — lower is better")
        c3.metric("Mean CER",  f"{r.get('avg_cer',0):.1%}",  help="Character Error Rate — stricter")

        st.markdown("---")
        st.markdown("""
**📖 What these numbers mean:**

| Metric | Target | Meaning |
|--------|--------|---------|
| Score | > 0.80 | `1 − WER`, clamped to [0, 1] |
| WER | < 15% | % of words wrong (substitutions + deletions + insertions) |
| CER | < 8% | % of *characters* wrong — more sensitive for Arabic |

**WER = 0** → perfect transcription. **WER = 0.30** → 3 in every 10 words are wrong.
Arabic has rich morphology so even a 20% WER can feel acceptable in conversation.
""")
        st.markdown("**💡 How to improve:**")
        wer = r.get("avg_wer", 0)
        if wer > 0.25:
            st.markdown("🔴 **WER > 25%** — use an Egyptian-Arabic tuned model (ElevenLabs Scribe `scribe_v1` or Whisper large-v3 with Egyptian prompt).")
        elif wer > 0.15:
            st.markdown("🟡 **WER 15–25%** — add a few-shot context in Arabic to the transcription prompt to reduce dialect errors.")
        else:
            st.markdown("✅ ASR is performing well!")

        with st.expander("📋 Per-sample details"):
            for d in r.get("details", []):
                st.markdown(f"**`{d.get('audio_file','')}`** — WER `{d.get('wer',0):.1%}` | CER `{d.get('cer',0):.1%}`")
                cols2 = st.columns(2)
                cols2[0].caption(f"📝 Reference: {d.get('reference','')}")
                cols2[1].caption(f"🤖 Hypothesis: {d.get('hypothesis','')}")
                st.divider()

    # ── LLM ───────────────────────────────────────────────────────────────────
    elif comp == "llm":
        st.subheader("🤖 LLM — Answer Quality & Arabic Fluency")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final score",         f"{score:.3f}")
        c2.metric("Avg judge score",      f"{r.get('avg_judge_score', 0):.3f}", help="GPT-4o judge score, normalised to [0,1]")
        c3.metric("Keyword rate",         f"{r.get('avg_keyword_rate', 0):.1%}", help="Expected Arabic terms found in answer")
        c4.metric("Punctuation score",    f"{r.get('avg_punctuation_score', 0):.3f}", help="Egyptian Arabic comma density + termination")

        st.markdown("---")
        st.markdown("""
**📖 What these numbers mean:**

| Metric | Weight | Target | Meaning |
|--------|--------|--------|---------|
| Judge score | **63%** | > 0.70 | GPT-4o evaluates: Factual (45%) · Relevance (30%) · Completeness (25%) on a 1–5 scale |
| Keyword rate | **27%** | > 0.80 | Fraction of expected Arabic medical terms that appear in the answer |
| Punctuation | **10%** | > 0.65 | Did the model use Arabic/English commas correctly and end sentences properly? |

`Final = 0.63 × judge + 0.27 × keywords + 0.10 × punctuation`

**🔍 Behind the Numbers:**
- **Judge Score (63%):** A weighted combination. **Factual Correctness (45%)** strictly checks for medical accuracy and penalises hallucinations. **Relevance (30%)** ensures the user's specific context (e.g., pricing, reservations, illnesses) is directly addressed. **Completeness (25%)** verifies if all core expected information is presented. A Judge score of 0.60 means the raw average is ~3/5; the model answers but lacks depth or misses facts.
- **Keyword Rate (27%):** A deterministic check. A rate of 0.50 means the model answers using generic terms and drops 50% of the specific medical/Egyptian Arabic vocabulary expected (e.g., "عطش", "مايه", "زغللة").
- **Punctuation Score (10%):** Evaluates if the response is formatted well for Text-to-Speech (TTS). It checks for sentence termination (ends with `. ! ?`), comma density (at least 1 comma per 120 chars), and heavily penalises run-on sentences over 120 chars without pauses.
""")
        st.markdown("**💡 How to improve:**")
        tips = []
        if r.get("avg_keyword_rate", 1) < 0.70:
            tips.append("🔴 **Keyword rate < 70%** — your Weaviate index is missing key medical PDFs. Upload more domain-specific Arabic documents.")
        if r.get("avg_punctuation_score", 1) < 0.60:
            tips.append("🟡 **Punctuation < 60%** — add to your system prompt: *\"استخدم الفاصلة العربية، بين الجمل وأنهِ كل إجابة بنقطة.\"*")
        js = r.get("avg_judge_score", 1)
        if js < 0.65:
            tips.append("🔴 **Judge score < 65%** — review the 3 worst samples to see what the judge penalised (hallucinations? off-topic?)")
        if not tips:
            tips = ["✅ LLM quality is solid!"]
        for t in tips:
            st.markdown(t)

        with st.expander("📋 Worst-scoring samples"):
            details = sorted(r.get("details", []), key=lambda x: x.get("final_score", 1))
            for d in details[:6]:
                s_val = d.get("final_score", 0)
                cls = "spass" if s_val >= 0.70 else ("swarn" if s_val >= 0.50 else "sfail")
                st.markdown(
                    f"<span class='{cls}'>{s_val:.3f}</span> **Q:** {d.get('question','')[:90]}",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"KW: {d.get('keyword_rate',0):.0%} | "
                    f"Punct: {d.get('punctuation_score',0):.2f} | "
                    f"Preview: {d.get('actual_preview','')[:150]}"
                )
                st.divider()

    # ── MCP ───────────────────────────────────────────────────────────────────
    elif comp == "mcp":
        st.subheader("📋 MCP — Clinic Operations & Knowledge Retrieval")
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall score",     f"{score:.3f}")
        c2.metric("Slot coverage",     f"{r.get('avg_slot_coverage',0):.1%}",
                  help="Do required entities (doctor/clinic/specialty) appear in the answer?")
        c3.metric("Keyword retrieval", f"{r.get('avg_keyword_retrieval',0):.1%}",
                  help="Do expected Arabic clinic terms appear in the answer?")

        st.markdown("---")
        st.markdown("""
**📖 What these numbers mean:**

| Metric | Weight | Target | Meaning |
|--------|--------|--------|---------|
| Slot coverage | **50%** | > 0.90 | Did the bot echo back the doctor/clinic name it was asked about? |
| Keyword retrieval | **50%** | > 0.80 | Did the bot use the right vocabulary? (مواعيد · حجز · سعر · عيادة) |

`Score = 0.50 × slot_coverage + 0.50 × keyword_retrieval`

**🔍 Behind the Numbers:**
- **Slot Coverage:** Extracts named entities from the AI's response (e.g., Doctor "أحمد", Clinic "القلب") and does an exact match against the parameters it was supposed to fetch from the MCP API. If it's low, the bot is answering without confirming the specific entity context.
- **Keyword Retrieval:** Checks if functional Egyptian Arabic clinic terms (like "استقبال", "كشف", "ميعاد") were correctly included in the response, simulating a natural, helpful clinic workflow.

**High slots + low keywords:** the bot says the doctor's name but answers generically.
**Low slots + any keywords:** the bot didn't understand which entity the user asked about.
**Both low:** the MCP server routing is failing or returning empty results.
""")
        st.markdown("**💡 How to improve:**")
        tips = []
        sc  = r.get("avg_slot_coverage", 1)
        kwr = r.get("avg_keyword_retrieval", 1)
        if sc < 0.85:
            tips.append("🔴 **Slot coverage < 85%** — the MCP server isn't matching entity names. Check that doctor/clinic records in the MCP database have matching Arabic names.")
        if kwr < 0.75:
            tips.append("🟡 **Keyword retrieval < 75%** — improve the MCP answer prompt to include more clinic vocabulary when responding.")
        if score >= 0.80:
            tips = ["✅ MCP is performing well!"]
        if not tips:
            tips = ["✅ MCP looks OK!"]
        for t in tips:
            st.markdown(t)

        with st.expander("📋 Per-sample details"):
            for d in r.get("details", []):
                sc_v  = d.get("slot_coverage", 0)
                kw_v  = d.get("keyword_retrieval", 0)
                icon  = "✅" if (sc_v + kw_v) / 2 >= 0.70 else "❌"
                st.markdown(
                    f"{icon} **{d.get('category','').upper()}** — `{d.get('question','')}`"
                )
                cols3 = st.columns(2)
                cols3[0].markdown(
                    f"**Slots** ✅ `{d.get('slot_hits',[])}` ❌ `{d.get('slot_misses',[])}` → `{sc_v:.0%}`"
                )
                cols3[1].markdown(
                    f"**Keywords** ✅ `{d.get('keyword_hits',[])}` ❌ `{d.get('keyword_misses',[])}` → `{kw_v:.0%}`"
                )
                st.caption(f"Answer: {d.get('answer_preview','')[:200]}")
                st.divider()

    # ── RAG ───────────────────────────────────────────────────────────────────
    elif comp == "rag":
        st.subheader("📚 RAG — Knowledge Retrieval & Excerpt Coverage")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall score",      f"{score:.3f}")
        c2.metric("Excerpt coverage",   f"{r.get('avg_excerpt_coverage',0):.1%}",
                  help="Critical Arabic medical snippets found in answers")
        c3.metric("Avg precision",      f"{r.get('avg_precision',0):.1%}",
                  help="Fraction of retrieved docs that are relevant")
        c4.metric("MRR",                f"{r.get('mrr',0):.3f}",
                  help="Mean Reciprocal Rank of first correct document")

        st.markdown("---")
        st.markdown("""
**📖 What these numbers mean:**

| Metric | Target | Meaning |
|--------|--------|---------|
| Excerpt coverage | > 0.60 | Primary: does the answer contain the critical medical terms expected? |
| Precision | > 0.60 | % of retrieved docs that are relevant (0 until sources API is wired) |
| Recall | > 0.60 | % of relevant docs actually retrieved |
| MRR | > 0.50 | How early was the first correct answer ranked? (1.0 = always first) |

**🔍 Behind the Numbers:**
- **Excerpt Coverage:** Since medical questions can be answered differently, this checks if the exact, critical medical excerpts (or close variations) from the document are present in the final answer.
- **MRR (Mean Reciprocal Rank):** Evaluates the ranking quality of the vector database search. If the correct document is the very first one retrieved (rank 1), MRR is 1.0. If it's second, it's 0.5. A low MRR means the relevant medical knowledge is buried too deep in the search results for the LLM to use effectively.
- **Precision vs Recall:** Precision measures how much of what was retrieved is actually relevant (reduces context window bloat). Recall measures how many of the total relevant documents were successfully retrieved.

> ℹ️ **Why is precision 0?** The backend API response doesn't yet include `sources` (Weaviate document names).
> Excerpt coverage is therefore the **only** meaningful signal until the backend wires source names.
> Upload your medical PDFs and index them to improve coverage.
""")
        st.markdown("**💡 How to improve:**")
        ec = r.get("avg_excerpt_coverage", 0)
        if ec < 0.30:
            st.markdown("🔴 **Coverage < 30%** — your Weaviate index is nearly empty. Upload domain PDFs via `/api/documents/upload`.")
        elif ec < 0.60:
            st.markdown("🟡 **Coverage 30–60%** — partial. Add more documents covering the failing question categories (see below).")
        else:
            st.markdown("✅ RAG excerpt coverage is good!")
        if r.get("avg_precision", 0) == 0:
            st.markdown("ℹ️ **To enable precision metrics**, wire Weaviate chunk source names into `ChatResponse.sources`.")

        with st.expander("📋 Per-query details"):
            for d in r.get("details", []):
                ec_v = d.get("excerpt_coverage", 0)
                icon = "✅" if ec_v >= 0.6 else ("🟡" if ec_v >= 0.3 else "❌")
                st.markdown(
                    f"{icon} {ec_v:.0%} coverage — **{d.get('question','')}**"
                )
                st.caption(
                    f"Expected: {d.get('critical_excerpts',[])} | "
                    f"Sources returned: {d.get('retrieved',[]) or 'none (API not wired)'}"
                )
                st.divider()

    else:
        st.json(r)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    with st.expander("🔗 Backend", expanded=True):
        cfg_backend = st.text_input("Backend URL",
            value=_env("BACKEND_URL", "http://localhost:8000"))

    with st.expander("🤖 LLM Judge"):
        cfg_judge_url   = st.text_input("Judge endpoint",
            value=_env("JUDGE_URL", "https://api.openai.com/v1/chat/completions"))
        _model_opts = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "custom"]
        _cur_model  = _env("JUDGE_MODEL", "gpt-4o")
        _idx        = _model_opts.index(_cur_model) if _cur_model in _model_opts else 0
        cfg_judge_model = st.selectbox("Model", _model_opts, index=_idx)
        if cfg_judge_model == "custom":
            cfg_judge_model = st.text_input("Custom model name")
        cfg_judge_key = st.text_input("OpenAI API key",
            value=_env("JUDGE_API_KEY", _env("OPENAI_API_KEY")),
            type="password")

    with st.expander("🔊 TTS"):
        _tts_opts = ["openai", "elevenlabs", "azure"]
        _tts_cur  = _env("TTS_PROVIDER", "openai")
        cfg_tts_provider = st.selectbox("Provider", _tts_opts,
            index=_tts_opts.index(_tts_cur) if _tts_cur in _tts_opts else 0)
        cfg_elevenlabs_key = st.text_input("ElevenLabs API key",
            value=_env("ELEVENLABS_API_KEY"), type="password")
        cfg_openai_tts_model = st.text_input("OpenAI TTS model",
            value=_env("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"))

    with st.expander("🎙️ ASR"):
        cfg_asr_dir = st.text_input("Audio directory",
            value=_env("ASR_AUDIO_DIR",
                str(_ROOT / "evaluation" / "dataset" / "audio")))
        _asr_opts = ["groq", "openai", "elevenlabs"]
        _asr_cur  = _env("ASR_PROVIDER", "groq")
        cfg_asr_provider = st.selectbox("Provider", _asr_opts,
            index=_asr_opts.index(_asr_cur) if _asr_cur in _asr_opts else 0)
        cfg_groq_key = st.text_input("Groq API key",
            value=_env("GROQ_API_KEY"), type="password")

    with st.expander("📊 Thresholds"):
        th_tts  = st.number_input("TTS p95 (ms) max",  value=3000, step=100)
        th_asr  = st.number_input("ASR WER max",        value=0.30, step=0.01, format="%.2f")
        th_llm  = st.number_input("LLM score min",      value=0.70, step=0.01, format="%.2f")
        th_mcp  = st.number_input("MCP score min",      value=0.80, step=0.01, format="%.2f")
        th_rag  = st.number_input("RAG precision min",  value=0.60, step=0.01, format="%.2f")
        th_comp = st.number_input("Composite pass",     value=0.85, step=0.01, format="%.2f")

    st.divider()

    if st.button("💾 Apply config"):
        os.environ.update({
            "BACKEND_URL":    cfg_backend,
            "JUDGE_URL":      cfg_judge_url,
            "JUDGE_MODEL":    cfg_judge_model,
            "JUDGE_API_KEY":  cfg_judge_key,
            "ASR_AUDIO_DIR":  cfg_asr_dir,
            "TTS_PROVIDER":   cfg_tts_provider,
            "ASR_PROVIDER":   cfg_asr_provider,
        })
        if cfg_elevenlabs_key:
            os.environ["ELEVENLABS_API_KEY"] = cfg_elevenlabs_key
        if cfg_groq_key:
            os.environ["GROQ_API_KEY"] = cfg_groq_key
        st.success("✅ Applied!")

    if st.button("🗑️ Clear results"):
        st.session_state.results = {}
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="hdr">
    <h1>🏥 MedRAG Evaluation Dashboard</h1>
    <p>AI Healthcare System — Production Quality Gate</p>
</div>
""", unsafe_allow_html=True)


# ── Eval runner helper ─────────────────────────────────────────────────────────
def _apply_env():
    try:
        os.environ["BACKEND_URL"]   = cfg_backend
        os.environ["JUDGE_URL"]     = cfg_judge_url
        os.environ["JUDGE_MODEL"]   = cfg_judge_model
        os.environ["JUDGE_API_KEY"] = cfg_judge_key
        os.environ["ASR_AUDIO_DIR"] = cfg_asr_dir
        os.environ["TTS_PROVIDER"]  = cfg_tts_provider
        os.environ["ASR_PROVIDER"]  = cfg_asr_provider
    except NameError:
        pass  # sidebar vars not in scope during first render


def _run(name: str) -> Dict[str, Any]:
    _apply_env()
    import importlib
    import evaluation.config as _ecfg
    importlib.reload(_ecfg)
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
        return {"score": 0.0, "error": f"Unknown: {name}"}
    return run_eval()


# ── Component cards ────────────────────────────────────────────────────────────
COMPS = {
    "tts":  ("🔊", "TTS",  "Synthesis quality & latency"),
    "asr":  ("🎙️", "ASR",  "Word error rate & accuracy"),
    "llm":  ("🤖", "LLM",  "Answer quality & Arabic fluency"),
    "mcp":  ("📋", "MCP",  "Clinic ops & slot coverage"),
    "rag":  ("📚", "RAG",  "Knowledge retrieval coverage"),
}

st.markdown("### 🧪 Evaluation Components")
cols = st.columns(5, gap="medium")

for idx, (key, (icon, name, desc)) in enumerate(COMPS.items()):
    with cols[idx]:
        res   = st.session_state.results.get(key)
        score = res.get("score", 0.0) if res else None

        if score is not None:
            cls = "spass" if score >= 0.70 else "sfail"
            score_html = f'<div class="{cls}">{score:.3f}</div>'
        else:
            score_html = '<div style="color:#8b949e;font-size:.85rem">Not run</div>'

        st.markdown(f"""
<div class="ecard">
    <div class="eicon">{icon}</div>
    <div class="ename">{name}</div>
    <div class="edesc">{desc}</div>
    <div style="margin-top:10px;">{score_html}</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("")

        if st.button(f"▶ Run {name}", key=f"run_{key}"):
            with st.spinner(f"Running {name}…"):
                try:
                    result = _run(key)
                except Exception as e:
                    result = {"score": 0.0, "error": str(e), "component": key}
            st.session_state.results[key] = result
            st.rerun()

# ── Metrics Guide ──────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 MedRAG Metrics Guide: How to Read the Scores (Normal User Guide)", expanded=False):
    guide_path = _ROOT / "evaluation" / "METRICS_GUIDE.md"
    if guide_path.exists():
        st.markdown(guide_path.read_text(encoding="utf-8"))
    else:
        st.info("Metrics guide is not available.")

# ── Run All ────────────────────────────────────────────────────────────────────
st.markdown("---")
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    if st.button("🚀  Run All Evaluations", key="run_all",
                 help="Runs TTS → ASR → LLM → MCP → RAG sequentially"):
        with st.status("Running full evaluation suite…", expanded=True) as status:
            for key, (icon, name, _) in COMPS.items():
                st.write(f"{icon} Evaluating **{name}**…")
                try:
                    result = _run(key)
                except Exception as e:
                    result = {"score": 0.0, "error": str(e), "component": key}
                st.session_state.results[key] = result
                s = result.get("score", 0)
                st.write(f"   → `{s:.3f}` {'✅' if s >= 0.70 else '❌'}")
            status.update(label="✅ Done!", state="complete")
        st.rerun()

# ── Results & reports ─────────────────────────────────────────────────────────
if st.session_state.results:
    st.markdown("---")
    st.markdown("## 📊 Full Evaluation Report")

    # Composite score
    _W = {"llm": 0.30, "asr": 0.20, "mcp": 0.20, "rag": 0.20, "tts": 0.05}
    _ws = _ss = 0.0
    for _k, _v in st.session_state.results.items():
        _w = _W.get(_k, 0.0); _ss += _w * _v.get("score", 0); _ws += _w
    composite = _ss / _ws if _ws else 0.0
    _cls     = "spass" if composite >= th_comp else "sfail"
    _verdict = "✅ GO — All thresholds met" if composite >= th_comp else "❌ NO-GO — Improve flagged components"

    _, centre_col, _ = st.columns([1, 2, 1])
    with centre_col:
        st.markdown(f"""
<div class="rcard" style="text-align:center;padding:28px;">
    <div style="font-size:.85rem;color:#8b949e;margin-bottom:8px;">Composite Weighted Score</div>
    <div class="{_cls}" style="font-size:3rem;">{composite:.3f}</div>
    <div style="margin-top:10px;color:#c9d1d9;font-size:1rem;">{_verdict}</div>
    <div style="margin-top:6px;font-size:.8rem;color:#8b949e;">
        Weights: LLM 30% · ASR 20% · MCP 20% · RAG 20% · TTS 5%
    </div>
</div>
""", unsafe_allow_html=True)

    # Per-component tabs
    available = list(st.session_state.results.keys())
    if available:
        tabs = st.tabs(
            [f"{COMPS[k][0]} {COMPS[k][1]}" for k in available]
        )
        for tab, key in zip(tabs, available):
            with tab:
                render_report(key, st.session_state.results[key])

else:
    st.markdown(
        "<div style='text-align:center;padding:60px;color:#8b949e;'>"
        "Run an evaluation above to see detailed reports here."
        "</div>",
        unsafe_allow_html=True,
    )
