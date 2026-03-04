/**
 * EvaluationDashboard — Phase 5
 *
 * 4-tab dashboard that shows MedRAG evaluation results.
 * Tabs: Overview · Per-Component · Interactive Tester · Dataset
 *
 * Data source: GET /api/evaluation/results (results.json)
 * Run trigger: POST /api/evaluation/run/{component}
 */

import { useState, useEffect, useCallback } from "react";
import {
    RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer,
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell,
} from "recharts";

/* ── Constants ─────────────────────────────────────────────────────────── */

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const TABS = ["Overview", "Per-Component", "Interactive Tester", "Dataset"] as const;
type Tab = typeof TABS[number];

const COMPONENT_COLORS: Record<string, string> = {
    tts: "#6366f1", asr: "#8b5cf6", llm: "#a78bfa",
    mcp: "#34d399", rag: "#10b981", whatsapp: "#f59e0b",
};

/* ── Types ──────────────────────────────────────────────────────────────── */

interface ComponentResult {
    component: string;
    score: number;
    error?: string;
    latency?: { p50: number; p95: number; p99: number; mean: number };
    avg_wer?: number;
    avg_precision?: number;
    avg_recall?: number;
    avg_f1?: number;
    accuracy?: number;
    success_rate?: number;
    sample_count?: number;
    details?: object[];
}

interface EvalResults {
    run_at: string | null;
    mode: string | null;
    all_passed: boolean | null;
    thresholds: Record<string, { passed: boolean; threshold_value: number; score: number }>;
    results: ComponentResult[];
    _message?: string;
}

/* ── Helpers ────────────────────────────────────────────────────────────── */

function pct(v: number | undefined) {
    if (v === undefined || v === null) return "—";
    return `${(v * 100).toFixed(1)}%`;
}

function ms(v: number | undefined) {
    if (v === undefined || v === null) return "—";
    return `${Math.round(v)} ms`;
}

function ScoreBadge({ value, passed }: { value: number; passed?: boolean }) {
    const bg = passed === false ? "bg-red-500/20 text-red-300 border-red-500/40"
        : passed === true ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
            : "bg-indigo-500/20 text-indigo-300 border-indigo-500/40";
    return (
        <span className={`px-2 py-0.5 rounded-full text-xs font-semibold border ${bg}`}>
            {pct(value)}
        </span>
    );
}

/* ── Overview tab ───────────────────────────────────────────────────────── */

function OverviewTab({ data }: { data: EvalResults }) {
    const radarData = data.results.map(r => ({
        component: r.component.toUpperCase(),
        score: +(r.score * 100).toFixed(1),
    }));

    const barData = data.results.map(r => ({
        name: r.component,
        score: +(r.score * 100).toFixed(1),
        fill: COMPONENT_COLORS[r.component] ?? "#94a3b8",
    }));

    const passed = data.all_passed;
    const runAt = data.run_at ? new Date(data.run_at).toLocaleString() : null;

    return (
        <div className="space-y-8">
            {/* Header status */}
            <div className="flex items-center justify-between flex-wrap gap-4">
                <div>
                    <h2 className="text-2xl font-bold text-white">Evaluation Overview</h2>
                    {runAt && <p className="text-slate-400 text-sm mt-0.5">Last run: {runAt} · mode: {data.mode}</p>}
                </div>
                {passed !== null && (
                    <span className={`px-4 py-1.5 rounded-full font-semibold text-sm border ${passed ? "bg-emerald-500/20 border-emerald-500/40 text-emerald-300"
                            : "bg-red-500/20 border-red-500/40 text-red-300"
                        }`}>
                        {passed ? "✅ All Passing" : "⚠️ Some Below Threshold"}
                    </span>
                )}
            </div>

            {/* Score cards */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                {data.results.map(r => {
                    const ts = data.thresholds[r.component];
                    return (
                        <div key={r.component}
                            className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4 flex flex-col gap-1 hover:border-indigo-500/50 transition-colors">
                            <p className="text-slate-400 text-xs uppercase tracking-wider">{r.component}</p>
                            <p className="text-3xl font-bold" style={{ color: COMPONENT_COLORS[r.component] ?? "#94a3b8" }}>
                                {pct(r.score)}
                            </p>
                            {ts && <ScoreBadge value={ts.score} passed={ts.passed} />}
                        </div>
                    );
                })}
            </div>

            {/* Charts row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5">
                    <p className="text-slate-300 font-medium mb-3">Radar: Score by Component</p>
                    <ResponsiveContainer width="100%" height={280}>
                        <RadarChart data={radarData}>
                            <PolarGrid stroke="#334155" />
                            <PolarAngleAxis dataKey="component" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: "#475569", fontSize: 10 }} />
                            <Radar name="Score" dataKey="score" stroke="#6366f1" fill="#6366f1" fillOpacity={0.35} />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>

                <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5">
                    <p className="text-slate-300 font-medium mb-3">Score by Component (%)</p>
                    <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={barData} barCategoryGap="30%">
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                            <YAxis domain={[0, 100]} tick={{ fill: "#475569", fontSize: 10 }} />
                            <Tooltip
                                contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, color: "#e2e8f0" }}
                                formatter={(v: number) => [`${v}%`, "Score"]}
                            />
                            <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                                {barData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}

/* ── Per-Component tab ──────────────────────────────────────────────────── */

function ComponentTab({ data }: { data: EvalResults }) {
    const [selected, setSelected] = useState<string>(data.results[0]?.component ?? "tts");
    const result = data.results.find(r => r.component === selected);
    const ts = data.thresholds[selected];

    return (
        <div className="space-y-6">
            {/* Selector */}
            <div className="flex flex-wrap gap-2">
                {data.results.map(r => (
                    <button
                        key={r.component}
                        id={`comp-btn-${r.component}`}
                        onClick={() => setSelected(r.component)}
                        className={`px-4 py-1.5 rounded-lg text-sm font-medium border transition-all ${selected === r.component
                                ? "border-indigo-500 bg-indigo-500/20 text-indigo-300"
                                : "border-slate-700 bg-slate-800/50 text-slate-400 hover:border-slate-500"
                            }`}
                    >
                        {r.component}
                    </button>
                ))}
            </div>

            {result && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
                    {/* Metrics */}
                    <div className="col-span-1 bg-slate-800/60 border border-slate-700/50 rounded-xl p-5 space-y-4">
                        <h3 className="text-white font-semibold text-lg capitalize">{result.component}</h3>

                        <Metric label="Score" value={pct(result.score)} pass={ts?.passed} />
                        {result.avg_wer !== undefined && <Metric label="Avg WER" value={(result.avg_wer * 100).toFixed(1) + "%"} />}
                        {result.avg_precision !== undefined && <Metric label="Precision" value={pct(result.avg_precision)} />}
                        {result.avg_recall !== undefined && <Metric label="Recall" value={pct(result.avg_recall)} />}
                        {result.avg_f1 !== undefined && <Metric label="F1" value={pct(result.avg_f1)} />}
                        {result.accuracy !== undefined && <Metric label="Accuracy" value={pct(result.accuracy)} />}
                        {result.success_rate !== undefined && <Metric label="Success Rate" value={pct(result.success_rate)} />}
                        {result.sample_count !== undefined && <Metric label="Samples" value={String(result.sample_count)} />}
                        {result.latency && <>
                            <Metric label="Latency p50" value={ms(result.latency.p50)} />
                            <Metric label="Latency p95" value={ms(result.latency.p95)} />
                            <Metric label="Latency p99" value={ms(result.latency.p99)} />
                        </>}
                        {result.error && (
                            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                                <p className="text-red-400 text-sm font-mono">{result.error}</p>
                            </div>
                        )}
                    </div>

                    {/* Latency chart */}
                    {result.latency && (
                        <div className="col-span-2 bg-slate-800/60 border border-slate-700/50 rounded-xl p-5">
                            <p className="text-slate-300 font-medium mb-3">Latency Percentiles (ms)</p>
                            <ResponsiveContainer width="100%" height={220}>
                                <BarChart data={[
                                    { label: "p50", value: Math.round(result.latency.p50) },
                                    { label: "p95", value: Math.round(result.latency.p95) },
                                    { label: "p99", value: Math.round(result.latency.p99) },
                                    { label: "mean", value: Math.round(result.latency.mean) },
                                ]} barCategoryGap="40%">
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                    <XAxis dataKey="label" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                                    <YAxis tick={{ fill: "#475569", fontSize: 11 }} />
                                    <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, color: "#e2e8f0" }} />
                                    <Bar dataKey="value" fill={COMPONENT_COLORS[result.component] ?? "#6366f1"} radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

function Metric({ label, value, pass }: { label: string; value: string; pass?: boolean }) {
    return (
        <div className="flex items-center justify-between border-b border-slate-700/40 pb-2">
            <span className="text-slate-400 text-sm">{label}</span>
            <span className={`text-sm font-semibold ${pass === false ? "text-red-400" : pass === true ? "text-emerald-400" : "text-white"
                }`}>{value}</span>
        </div>
    );
}

/* ── Interactive Tester tab ─────────────────────────────────────────────── */

function TesterTab() {
    const [question, setQuestion] = useState("");
    const [keywords, setKeywords] = useState("");
    const [result, setResult] = useState<null | { answer: string; score: number; routing: string }>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const run = useCallback(async () => {
        if (!question.trim()) return;
        setLoading(true); setError(""); setResult(null);
        try {
            const resp = await fetch(`${API}/api/evaluation/test`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question,
                    expected_keywords: keywords ? keywords.split(",").map(k => k.trim()).filter(Boolean) : null,
                    mock: true,
                }),
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            setResult(data);
        } catch (e: unknown) {
            setError((e as Error).message);
        } finally {
            setLoading(false);
        }
    }, [question, keywords]);

    return (
        <div className="space-y-6 max-w-2xl">
            <div>
                <h2 className="text-xl font-bold text-white mb-1">Interactive Tester</h2>
                <p className="text-slate-400 text-sm">Run a single query through the pipeline and score it against expected keywords.</p>
            </div>

            <div className="space-y-3">
                <label className="block">
                    <span className="text-slate-300 text-sm font-medium block mb-1">Question</span>
                    <textarea
                        id="eval-question"
                        className="w-full bg-slate-800 border border-slate-700 rounded-xl p-3 text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 resize-none text-sm"
                        rows={3}
                        value={question}
                        onChange={e => setQuestion(e.target.value)}
                        placeholder="e.g. متى تفتح العيادة؟"
                    />
                </label>

                <label className="block">
                    <span className="text-slate-300 text-sm font-medium block mb-1">Expected keywords <span className="text-slate-500">(comma-separated, optional)</span></span>
                    <input
                        id="eval-keywords"
                        className="w-full bg-slate-800 border border-slate-700 rounded-xl p-3 text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 text-sm"
                        value={keywords}
                        onChange={e => setKeywords(e.target.value)}
                        placeholder="تفتح, صباح, عيادة"
                    />
                </label>

                <button
                    id="eval-run-btn"
                    onClick={run}
                    disabled={loading || !question.trim()}
                    className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white rounded-xl font-medium text-sm transition-colors"
                >
                    {loading ? "Running…" : "Run Test"}
                </button>
            </div>

            {error && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
                    <p className="text-red-400 text-sm">{error}</p>
                </div>
            )}

            {result && (
                <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5 space-y-3">
                    <div className="flex items-center justify-between">
                        <span className="text-slate-400 text-sm">Score</span>
                        <ScoreBadge value={result.score} />
                    </div>
                    <div className="flex items-center justify-between">
                        <span className="text-slate-400 text-sm">Routing</span>
                        <span className="text-white text-sm font-medium">{result.routing}</span>
                    </div>
                    <div className="border-t border-slate-700/50 pt-3">
                        <p className="text-slate-400 text-xs mb-1">Answer</p>
                        <p className="text-slate-200 text-sm leading-relaxed">{result.answer}</p>
                    </div>
                </div>
            )}
        </div>
    );
}

/* ── Dataset tab ────────────────────────────────────────────────────────── */

function DatasetTab() {
    const datasets = [
        { name: "TTS Samples", file: "tts_samples.json", desc: "12 Arabic/English text samples for TTS latency evaluation" },
        { name: "ASR Samples", file: "asr_samples.json", desc: "8 Arabic audio transcription pairs for WER scoring" },
        { name: "LLM Queries", file: "llm_queries.json", desc: "10 medical Q&A pairs with expected keywords" },
        { name: "MCP Queries", file: "mcp_queries.json", desc: "10 clinic operation queries with expected routing intent" },
        { name: "RAG Queries", file: "rag_queries.json", desc: "10 retrieval queries with ground-truth source documents" },
    ];

    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-xl font-bold text-white mb-1">Evaluation Datasets</h2>
                <p className="text-slate-400 text-sm">Datasets used by the evaluation framework stored in <code className="text-indigo-300">evaluation/dataset/</code>.</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {datasets.map(ds => (
                    <div key={ds.file} className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5 hover:border-indigo-500/50 transition-colors">
                        <p className="text-white font-semibold mb-1">{ds.name}</p>
                        <p className="text-slate-400 text-sm mb-3">{ds.desc}</p>
                        <code className="text-indigo-300 text-xs bg-slate-900/50 px-2 py-1 rounded">{ds.file}</code>
                    </div>
                ))}
            </div>
        </div>
    );
}

/* ── Run Controls ──────────────────────────────────────────────────────── */

function RunControls({ onRun, running }: { onRun: (component: string) => void; running: boolean }) {
    const [component, setComponent] = useState("all");
    return (
        <div className="flex items-center gap-3 flex-wrap">
            <select
                id="eval-component-select"
                value={component}
                onChange={e => setComponent(e.target.value)}
                className="bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-indigo-500"
            >
                {["all", "tts", "asr", "llm", "mcp", "rag", "whatsapp"].map(c => (
                    <option key={c} value={c}>{c}</option>
                ))}
            </select>
            <button
                id="eval-run-all-btn"
                onClick={() => onRun(component)}
                disabled={running}
                className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white rounded-lg font-medium text-sm transition-colors"
            >
                {running ? "Running…" : "▶ Run Evaluation"}
            </button>
        </div>
    );
}

/* ── Main Dashboard Component ───────────────────────────────────────────── */

export default function EvaluationDashboard() {
    const [activeTab, setActiveTab] = useState<Tab>("Overview");
    const [data, setData] = useState<EvalResults | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [running, setRunning] = useState(false);
    const [toast, setToast] = useState("");

    const fetchResults = useCallback(async () => {
        try {
            const resp = await fetch(`${API}/api/evaluation/results`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            setData(await resp.json());
        } catch (e: unknown) {
            setError((e as Error).message);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { fetchResults(); }, [fetchResults]);

    const handleRun = useCallback(async (component: string) => {
        setRunning(true);
        setToast(`Starting ${component} evaluation (mock)…`);
        try {
            await fetch(`${API}/api/evaluation/run/${component}?mock=true`, { method: "POST" });
            setToast("Evaluation started in background. Refreshing in 5s…");
            setTimeout(() => { fetchResults(); setRunning(false); setToast(""); }, 5000);
        } catch {
            setToast("Failed to start evaluation.");
            setRunning(false);
        }
    }, [fetchResults]);

    return (
        <div className="min-h-screen bg-slate-950 text-slate-100" style={{ fontFamily: "'Inter', 'Segoe UI', sans-serif" }}>
            {/* Top nav */}
            <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-20">
                <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-sm font-bold">M</div>
                        <span className="font-semibold text-white">MedRAG Evaluation Dashboard</span>
                    </div>
                    <div className="flex items-center gap-4">
                        <RunControls onRun={handleRun} running={running} />
                        <button
                            id="eval-refresh-btn"
                            onClick={fetchResults}
                            className="px-3 py-2 border border-slate-700 hover:border-slate-500 rounded-lg text-slate-400 hover:text-white text-sm transition-colors"
                        >
                            ↻ Refresh
                        </button>
                    </div>
                </div>

                {/* Toast */}
                {toast && (
                    <div className="bg-indigo-600/90 text-white text-sm px-6 py-2 text-center border-t border-indigo-500/50">
                        {toast}
                    </div>
                )}

                {/* Tabs */}
                <div className="max-w-7xl mx-auto px-6 flex gap-1 pt-1">
                    {TABS.map(tab => (
                        <button
                            key={tab}
                            id={`tab-${tab.toLowerCase().replace(/\s+/g, "-")}`}
                            onClick={() => setActiveTab(tab)}
                            className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${activeTab === tab
                                    ? "border-indigo-500 text-indigo-400"
                                    : "border-transparent text-slate-500 hover:text-slate-300"
                                }`}
                        >
                            {tab}
                        </button>
                    ))}
                </div>
            </header>

            {/* Body */}
            <main className="max-w-7xl mx-auto px-6 py-8">
                {loading && (
                    <div className="flex items-center justify-center py-24">
                        <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                    </div>
                )}

                {error && !loading && (
                    <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 text-center">
                        <p className="text-red-400 font-medium mb-2">Could not load evaluation results</p>
                        <p className="text-slate-500 text-sm">{error}</p>
                        <p className="text-slate-500 text-sm mt-2">Make sure the backend is running at <code className="text-indigo-300">{API}</code></p>
                    </div>
                )}

                {data && !loading && (
                    <>
                        {data._message && (
                            <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 mb-6">
                                <p className="text-amber-300 text-sm">{data._message}</p>
                            </div>
                        )}

                        {activeTab === "Overview" && <OverviewTab data={data} />}
                        {activeTab === "Per-Component" && <ComponentTab data={data} />}
                        {activeTab === "Interactive Tester" && <TesterTab />}
                        {activeTab === "Dataset" && <DatasetTab />}
                    </>
                )}
            </main>
        </div>
    );
}
