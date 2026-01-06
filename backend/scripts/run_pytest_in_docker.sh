#!/usr/bin/env bash
set -euo pipefail

# Run pytest inside the running backend container (without rebuilding the image).
# This script copies the repo's test files into the container temporarily and executes them.
#
# Usage:
#   cd heal-query-hub
#   ./backend/scripts/run_pytest_in_docker.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

cid="$(docker compose ps -q backend)"
if [[ -z "${cid}" ]]; then
  echo "Backend container is not running. Start it first with: docker compose up -d backend" >&2
  exit 2
fi

echo "== Ensuring pytest is available in backend venv =="
docker compose exec -T backend uv pip install pytest==7.4.3 >/dev/null

echo "== Copying tests into container ($cid) =="
docker exec -i "$cid" rm -rf /tmp/tests >/dev/null 2>&1 || true
docker cp "$ROOT_DIR/backend/tests" "$cid:/tmp/tests"

if [[ -f "$ROOT_DIR/backend/test_phoenix_connection.py" ]]; then
  docker cp "$ROOT_DIR/backend/test_phoenix_connection.py" "$cid:/tmp/test_phoenix_connection.py"
fi

echo "== Running pytest inside backend container =="
docker compose exec -T backend uv run python -m pytest -q /tmp/tests

echo "PASS"


