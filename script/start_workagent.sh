#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend"
LOG_DIR="$ROOT/logs"
BACKEND_URL="http://127.0.0.1:8001/api/status"
FRONTEND_URL="http://localhost:5173"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

find_python() {
    if [[ -x "$ROOT/.venv/bin/python" ]]; then
        printf '%s\n' "$ROOT/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    elif command -v python >/dev/null 2>&1; then
        command -v python
    else
        fail "Python 3 was not found. Run ./install_workagent.sh first."
    fi
}

http_ready() {
    local url="$1"
    "$PYTHON" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).close()' "$url" \
        >/dev/null 2>&1
}

wait_for_url() {
    local label="$1"
    local url="$2"
    local log_file="$3"
    local attempt
    for ((attempt = 1; attempt <= 30; attempt += 1)); do
        if http_ready "$url"; then
            printf '%s is ready.\n' "$label"
            return 0
        fi
        sleep 1
    done
    printf '%s did not become ready. Check %s\n' "$label" "$log_file" >&2
    return 1
}

open_browser() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        open "$FRONTEND_URL"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$FRONTEND_URL" >/dev/null 2>&1 &
    elif command -v sensible-browser >/dev/null 2>&1; then
        sensible-browser "$FRONTEND_URL" >/dev/null 2>&1 &
    else
        printf 'Open %s in your browser.\n' "$FRONTEND_URL"
    fi
}

[[ -d "$BACKEND_DIR" ]] || fail "Backend directory not found: $BACKEND_DIR"
[[ -d "$FRONTEND_DIR" ]] || fail "Frontend directory not found: $FRONTEND_DIR"
command -v npm >/dev/null 2>&1 || fail "npm was not found. Run ./install_workagent.sh first."

PYTHON="$(find_python)"
mkdir -p "$LOG_DIR"

if http_ready "$BACKEND_URL"; then
    printf 'Backend is already running.\n'
else
    printf 'Starting WorkAgent backend...\n'
    (
        cd "$BACKEND_DIR"
        nohup "$PYTHON" -m uvicorn api_server:app --host 127.0.0.1 --port 8001 \
            >>"$LOG_DIR/backend.log" 2>&1 &
        printf '%s\n' "$!" >"$LOG_DIR/backend.pid"
    )
    wait_for_url "Backend" "$BACKEND_URL" "$LOG_DIR/backend.log" || exit 1
fi

if http_ready "$FRONTEND_URL"; then
    printf 'Frontend is already running.\n'
else
    printf 'Starting WorkAgent frontend...\n'
    (
        cd "$FRONTEND_DIR"
        nohup npm run dev >>"$LOG_DIR/frontend.log" 2>&1 &
        printf '%s\n' "$!" >"$LOG_DIR/frontend.pid"
    )
    wait_for_url "Frontend" "$FRONTEND_URL" "$LOG_DIR/frontend.log" || exit 1
fi

printf 'Opening %s ...\n' "$FRONTEND_URL"
open_browser
printf 'WorkAgent is ready.\n'
