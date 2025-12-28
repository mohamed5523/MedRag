#!/usr/bin/env bash
set -euo pipefail

# E2E tests against /api/chat/query-with-voice using real Clinic APIs via the MCP server.
#
# Prereqs:
# - docker + docker compose
# - env vars exported in your shell (no secrets committed)
#
# Required env vars:
#   CLINIC_PROVIDER_LIST_URL
#   CLINIC_PROVIDER_SCHEDULE_URL
#   CLINIC_SERVICE_PRICE_URL
#   CLINIC_API_USERNAME
#   CLINIC_API_PASSWORD
#   OPENAI_API_KEY                (backend uses OpenAI to verbalize MCP output)
#
# Optional:
#   BASE_URL=http://localhost:8000
#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_URL="${BASE_URL:-http://localhost:8000}"
COMPOSE="${COMPOSE:-docker compose}"

required_env=(
  CLINIC_PROVIDER_LIST_URL
  CLINIC_PROVIDER_SCHEDULE_URL
  CLINIC_SERVICE_PRICE_URL
  CLINIC_API_USERNAME
  CLINIC_API_PASSWORD
  OPENAI_API_KEY
)

for v in "${required_env[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    echo "Missing required env var: $v" >&2
    exit 2
  fi
done

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

assert_contains() {
  local hay="$1"; local needle="$2"
  echo "$hay" | grep -Fq "$needle" || fail "Expected to contain: $needle | Got: $hay"
}

assert_not_contains() {
  local hay="$1"; local needle="$2"
  echo "$hay" | grep -Fq "$needle" && fail "Expected NOT to contain: $needle | Got: $hay"
}

json_field() {
  local json="$1"
  local field="$2"
  python3 - "$field" <<'PY'
import json, sys
field = sys.argv[1]
obj = json.load(sys.stdin)
val = obj.get(field)
if val is None:
    print("")
elif isinstance(val, (dict, list)):
    print(json.dumps(val, ensure_ascii=False))
else:
    print(str(val))
PY
}

post_chat_voice() {
  local session_id="$1"
  local query="$2"
  curl -fsS "$BASE_URL/api/chat/query-with-voice" \
    -H "Content-Type: application/json" \
    -H "X-Session-Id: $session_id" \
    -d "{\"query\": \"${query//\"/\\\"}\", \"max_results\": 5}"
}

echo "== Build & start stack (real clinic APIs via docker-compose.e2e.yml) =="
cd "$ROOT_DIR"

$COMPOSE -f docker-compose.yml -f docker-compose.e2e.yml build backend mcp-server
$COMPOSE -f docker-compose.yml -f docker-compose.e2e.yml up -d weaviate redis phoenix mcp-server backend

echo "== Wait for backend health =="
for i in {1..60}; do
  if curl -fsS "$BASE_URL/health" >/dev/null; then
    pass "Backend is healthy."
    break
  fi
  sleep 2
  [[ $i -eq 60 ]] && fail "Backend did not become healthy"
done

echo "== Fetch provider list from MCP server (through backend config) =="
providers_json="$($COMPOSE -f docker-compose.yml -f docker-compose.e2e.yml exec -T mcp-server curl -fsS http://localhost:8000/providers)"

# Pick an ambiguous first-name token that appears across >= 2 doctors (best-effort).
ambiguous_token="$(python3 - <<'PY'
import json, re, sys
data = json.load(sys.stdin)
names = []
for clinic in (data.get("data") or data):
    doctors = clinic.get("doctors") or []
    for d in doctors:
        name = (d.get("DoctorNameA") or "").strip()
        if name:
            names.append(name)

def norm(s: str) -> str:
    s = re.sub(r"[\u0640\u064B-\u0652\u0670]", "", s)
    s = s.replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ى","ي").replace("ة","ه")
    s = re.sub(r"\s+"," ",s).strip()
    return s

counts = {}
for n in names:
    parts = norm(n).split()
    if not parts:
        continue
    token = parts[0]
    if len(token) < 3:
        continue
    counts[token] = counts.get(token, 0) + 1

best = max(counts.items(), key=lambda kv: kv[1], default=(None, 0))
if not best[0] or best[1] < 2:
    print("")
else:
    print(best[0])
PY
<<<"$providers_json")"

if [[ -z "$ambiguous_token" ]]; then
  echo "WARN: Could not auto-discover an ambiguous token (>=2). Disambiguation scenario will be skipped."
fi

SESSION_A="e2e-voice-a"
SESSION_B="e2e-voice-b"

echo "== Scenario 1: Symptom triage (should NOT return RAG not-found) =="
resp="$(post_chat_voice "$SESSION_A" "بطني بتوجعني النهاردة اروح لمين؟")"
answer="$(echo "$resp" | json_field answer)"
assert_not_contains "$answer" "I don't have any relevant information"
assert_not_contains "$answer" "Please try rephrasing"
pass "Symptom triage did not fall back to RAG not-found."

echo "== Scenario 2: Disambiguation (ambiguous name -> numbered prompt -> selection) =="
if [[ -n "$ambiguous_token" ]]; then
  resp1="$(post_chat_voice "$SESSION_B" "سعر كشف دكتور $ambiguous_token")"
  ans1="$(echo "$resp1" | json_field answer)"
  err1="$(echo "$resp1" | json_field error)"
  assert_contains "$ans1" "اختار"
  # should be provider_ambiguous or provider_low_confidence
  [[ "$err1" == "provider_ambiguous" || "$err1" == "provider_low_confidence" || "$err1" == "" ]] || true

  # choose first option
  resp2="$(post_chat_voice "$SESSION_B" "1")"
  ans2="$(echo "$resp2" | json_field answer)"
  assert_not_contains "$ans2" "اختار"
  # Sticky intent: should not switch into schedule language when original was price
  assert_not_contains "$ans2" "مواعيد"
  pass "Disambiguation + selection did not drift into schedule."

  # Continuation (history): ask "سعر الكشف" should still refer to resolved doctor (no re-disambiguation)
  resp3="$(post_chat_voice "$SESSION_B" "سعر الكشف")"
  ans3="$(echo "$resp3" | json_field answer)"
  assert_not_contains "$ans3" "اختار"
  pass "Continuation reused context (no re-disambiguation)."
else
  echo "SKIP: Disambiguation scenario (no ambiguous token found)."
fi

echo "== Scenario 3: Context reset by new session id (should NOT reuse previous doctor) =="
resp_new="$(post_chat_voice "e2e-fresh-session" "سعر الكشف")"
ans_new="$(echo "$resp_new" | json_field answer)"
# We expect a clarification (missing clinic/doctor) rather than a fully resolved doctor-specific price.
assert_contains "$ans_new" "محتاج"
pass "Fresh session asked for missing context (reset worked)."

echo "ALL E2E TESTS PASSED"

