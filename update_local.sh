#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# update_local.sh  — Local development launcher
# Usage: ./update_local.sh [--stop | --restart | --logs | --build]
#
# What it does:
#   1. Brings up all backend services (Weaviate, Redis, MCP, Backend, Phoenix,
#      Prometheus, Grafana) via Docker Compose.
#   2. Launches the Vite dev server (frontend) with hot-reload on :8080.
#
# This script is COMPLETELY SEPARATE from update_vps.sh and only affects
# your local machine.  It never SSH-es or touches the remote VPS.
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m' ; YELLOW='\033[1;33m' ; RED='\033[0;31m' ; NC='\033[0m'
info()    { echo -e "${GREEN}[LOCAL]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN] ${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.local.yml"

# ── Argument handling ─────────────────────────────────────────────────────────
case "${1:-}" in
  --stop)
    info "Stopping all local services..."
    $COMPOSE down
    info "Done."
    exit 0
    ;;
  --restart)
    info "Restarting backend services..."
    $COMPOSE down
    ;;
  --logs)
    $COMPOSE logs -f
    exit 0
    ;;
  --build)
    info "Forcing rebuild of backend images..."
    $COMPOSE build --no-cache
    ;;
  "")
    # default: start everything
    ;;
  *)
    echo "Usage: $0 [--stop | --restart | --logs | --build]"
    exit 1
    ;;
esac

# ── Pre-flight checks ─────────────────────────────────────────────────────────
command -v docker  &>/dev/null || { error "docker not found. Please install Docker."; exit 1; }
command -v node    &>/dev/null || { error "node not found. Please install Node.js."; exit 1; }
command -v npm     &>/dev/null || { error "npm not found. Please install npm."; exit 1; }

info "Starting LOCAL development environment..."
echo ""

# ── 1) Start backend services ─────────────────────────────────────────────────
info "→ Bringing up backend services via Docker Compose..."
$COMPOSE up -d --build

echo ""
info "Backend services started:"
echo "   • Backend API   → http://localhost:8000  (docs: http://localhost:8000/docs)"
echo "   • MCP Server    → http://localhost:8020"
echo "   • Weaviate      → http://localhost:8081"
echo "   • Redis         → localhost:6379"
echo "   • Phoenix UI    → http://localhost:6006"
echo "   • Prometheus    → http://localhost:9090"
echo "   • Grafana       → http://localhost:3000  (admin / admin)"
echo ""

# ── 2) Wait for the backend to be healthy ─────────────────────────────────────
info "→ Waiting for backend API to become ready..."
MAX_WAIT=120
ELAPSED=0
until curl -fsS http://localhost:8000/health &>/dev/null; do
  if [ $ELAPSED -ge $MAX_WAIT ]; then
    warn "Backend API did not respond after ${MAX_WAIT}s. Check logs with: ./update_local.sh --logs"
    break
  fi
  printf '.'
  sleep 3
  ELAPSED=$((ELAPSED + 3))
done
echo ""

# ── 3) Install frontend dependencies if needed ────────────────────────────────
if [ ! -d "frontend/node_modules" ]; then
  info "→ Installing frontend dependencies (first run)..."
  npm --prefix ./frontend install
fi

# ── 4) Launch Vite dev server (foreground) ────────────────────────────────────
info "→ Starting Vite dev server (hot-reload)..."
echo ""
echo -e "${GREEN}────────────────────────────────────────────────────${NC}"
echo -e "${GREEN}  Frontend (Vite):  http://localhost:8080            ${NC}"
echo -e "${GREEN}  Press Ctrl+C to stop the frontend.                 ${NC}"
echo -e "${GREEN}  To stop ALL services run: ./update_local.sh --stop ${NC}"
echo -e "${GREEN}────────────────────────────────────────────────────${NC}"
echo ""

npm --prefix ./frontend run dev
