## Goal

Fix inconsistent/illogical assistant behavior in the **MCP path** (especially `/api/chat/query-with-voice`) and eliminate RAG “not found” responses for “اروح لمين؟” symptom triage by routing to MCP.

Concrete issues to solve (from examples/screenshots):

- **Intent drift after disambiguation**: user asks price → assistant asks to disambiguate → user selects doctor → system answers schedule (wrong tool path).
- **Confirmation loop**: user confirms doctor (اه/نعم/صح/…) → assistant repeats the same “هل تقصد …؟” question.
- **“Not found” for symptom queries**: “بطني بتوجعني… اروح لمين” currently hits RAG empty retrieval and replies with “I don’t have any relevant information…”.

Constraints:

- Must be **scalable/robust** (not a hardcoded yes-word list).
- Must be implemented with **TDD** (tests first).
- Must be done on a **separate git branch**, merge only after all tests pass.
- Focus first on `/query-with-voice`; later mirror to `/query`.

## Non-goals (for this iteration)

- Rewriting the entire state extraction system.
- Building a full medical knowledge base for general advice. Instead, implement MCP-first triage for “اروح لمين” flows.

---

## High-level Solution Architecture

### 1) Conversation Controller (state machine + classifier)

We will introduce a “conversation controller” layer in the `/query-with-voice` request pipeline that uses:

- **current query**
- **cached conversation messages** (Redis short-term memory)
- **previous extracted state**
- **pending_action** (new per-session state)

It produces an **effective turn decision** (continue vs new-topic vs clarification) and, when needed, an **intent override** / entity reuse policy.

Key design: **state machine first, classifier second**

- If `pending_action` exists → this turn is **clarification** by definition; attempt to resolve it, never re-run the original ambiguity logic.
- Else run **rules-first classification** (deterministic).
- If rules are low-confidence/ambiguous → optional **LLM fallback** (structured output, `temperature=0`).

### 2) `pending_action` (scalable disambiguation, no yes-word dependency)

We add a per-session `pending_action` object stored in Redis, e.g.:

- `type`: `"provider_disambiguation"`
- `intent`: `"ask_price" | "check_availability" | "book_appointment" | "list_doctors"`
- `candidates`: list of candidates (name_ar/name_en + clinic + ids + score)
- `created_at`, `expires_at`, `turns_remaining`

When MCP matching is ambiguous/low-confidence, we:

- store `pending_action` with **the original intent** (prevents intent drift)
- respond with **numbered options** (“اكتب رقم 1/2/3…”)

On the next user message:

- if reply is `1..N` (Arabic or Latin digits) → select that candidate
- else fuzzy-match reply to candidate names → select if confident
- else re-prompt using the same candidate list (decrement turns_remaining), without re-triggering ambiguity.

This fixes:

- price→schedule drift (sticky intent)
- “اه/نعم/صح” confirmation loops (user can always reply with a number; fuzzy-match supports pasting partial name)

### 3) Deterministic routing & intent enforcement for MCP

When `pending_action` exists or when state intent is clearly MCP, routing must be deterministic:

- Router temperature set to **0**
- If extracted state intent is one of MCP intents, the router must **not override** it with a different MCP intent.

### 4) Symptom triage (Option A) → MCP “go to which specialty/doctor?”

For queries like:

- “بطني بتوجعني… اروح لمين؟”
- “عندي ألم… اكشف عند مين؟”

We route to MCP and return a doctor list:

- classify the symptoms to a **specialty** (rules-first + optional LLM fallback)
- call MCP `get_clinic_provider_list`
- filter providers by specialty and return doctor names (and optionally clinic name)

Important policy:

- For symptom triage intents, **never return** the RAG “not found” template.
- If specialty cannot be inferred, ask 1 clarifying question (“الألم فين بالظبط؟ في سخونية/قيء؟”) and keep context for the next turn.

---

## Implementation Plan (Files / Components)

### A) Redis memory: add `pending_action`

File(s):

- `backend/app/core/conversation_memory.py`

Add methods:

- `save_pending_action(session_id, payload)`
- `get_pending_action(session_id)`
- `clear_pending_action(session_id)`

Data stored as JSON in Redis under a new key: `medrag:s:{session_id}:pending_action`

### B) Clinic workflow: expose candidates on ambiguous match

File(s):

- `backend/app/services/clinic_workflow.py`

Extend `MCPWorkflowError` to carry structured `data` (e.g., candidates list).
When hybrid matching returns ambiguous/low-confidence, raise `MCPWorkflowError` with:

- reason: `provider_ambiguous` / `provider_low_confidence`
- data: `{"candidates": [...], "query": "...", "clinic_name": "..." }`

### C) Chat endpoint: integrate controller in `/query-with-voice`

File(s):

- `backend/app/api/chat.py`
- (new) `backend/app/core/conversation_controller.py` (or similar)

Pipeline change for `/query-with-voice`:

1. Load history + previous_state + pending_action
2. If pending_action:
   - try resolve selection from current user message
   - if resolved: update state entities + force intent = pending_action.intent; clear pending_action
   - if not resolved: return re-prompt with numbered options; keep pending_action
3. Else:
   - run rules-first (and optional LLM fallback) classification using query + history + previous_state
   - decide whether to reuse doctor/clinic context, and whether to set symptom triage intent
4. Route to MCP and execute workflow; for symptom triage ensure MCP list-doctors path.

### D) Intent router: enforce MCP intent stability

File(s):

- `backend/app/core/intent_router.py`

Changes:

- set router `temperature=0`
- if `state.intent` is MCP intent (ask_price, check_availability, book_appointment, list_doctors), enforce MCP route and preserve intent (don’t override via LLM).

### E) Symptom triage inference

File(s):

- `backend/app/core/conversation_controller.py` (new helper)

Implement:

- `is_symptom_triage_request(query, history) -> bool`
- `infer_specialty_from_symptoms(query, available_specialties=None) -> Optional[str]`

Rules-first mapping (initial):

- abdominal pain keywords → “باطنة” / “جهاز هضمي”
- chest pain → “قلب”
- pregnancy-related → “نسا وتوليد”
- etc.

Optional LLM fallback (temperature 0) when:

- rules fail AND OpenAI key is configured AND we have a list of available specialties from MCP.

---

## TDD Test Plan

We’ll add tests before implementing behavior changes.

### 1) Disambiguation: intent stickiness (prevents price→schedule drift)

Test name idea:

- `test_pending_action_keeps_original_intent_on_doctor_selection`

Scenario:

- initial intent `ask_price`
- MCP returns ambiguous candidates
- user selects candidate `2`
Expected:
- system proceeds with `ask_price` workflow (calls `get_service_price`), not schedule.

### 2) Disambiguation: selection parsing

Test name idea:

- `test_parse_candidate_selection_accepts_arabic_and_latin_digits`
- `test_parse_candidate_selection_fuzzy_matches_partial_name`

### 3) Symptom triage routes to MCP

Test name idea:

- `test_symptom_triage_routes_to_mcp_list_doctors`

Scenario:

- user: “بطني بتوجعني… اروح لمين”
Expected:
- decision is MCP and returns doctor list (from fake provider list), not RAG not-found.

### 4) Router stability

Test name idea:

- `test_route_conversation_preserves_mcp_intent_when_state_has_mcp_intent`

---

## Rollout / Safety

- Implement for `/query-with-voice` first.
- Keep changes behind simple flags if needed:
  - `ENABLE_PENDING_ACTION=true`
  - `ENABLE_SYMPTOM_TRIAGE=true`
- Add Phoenix span attributes for:
  - pending_action present/resolved
  - forced intent / forced MCP route
  - inferred specialty

---

## Execution Steps (Agent Mode)

- Create branch: `feature/conversation-controller`
- Write failing tests (pytest)
- Implement in small increments until green
- Run full test suite
- Provide summary + next steps to mirror to `/query`
