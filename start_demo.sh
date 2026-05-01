#!/usr/bin/env bash
# Interview-morning starter — one command to get a clean, running pipeline.
#
# Usage:
#   ./start_demo.sh
#
# What it does:
#   1. Kills any zombie process on port 8002 (idempotent)
#   2. Loads .env (Claude CLI provider, in-memory Qdrant, local embeddings)
#   3. Starts FastAPI server in background, logs to /tmp/rag-server.log
#   4. Waits up to 60s for /health to return 200
#   5. Ingests sample.md so the server has retrievable content
#   6. Runs one smoke query end-to-end
#   7. Prints the URLs and PID for easy cleanup
#
# If anything fails, the script exits non-zero and points at the log.

set -euo pipefail

PORT="${PORT:-8002}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/rag-server.log"

cd "$ROOT"

echo "==> rag-eval-system demo starter"
echo "    Repo:  $ROOT"
echo "    Port:  $PORT"
echo "    Log:   $LOG_FILE"
echo

# --- Step 1: free the port ---------------------------------------------------
if command -v powershell.exe >/dev/null 2>&1; then
  powershell.exe -NoProfile -Command "
    \$c = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue
    foreach (\$conn in \$c) {
      Write-Output \"    killing PID \$(\$conn.OwningProcess) on port $PORT\"
      Stop-Process -Id \$conn.OwningProcess -Force -ErrorAction SilentlyContinue
    }
  " 2>/dev/null || true
  sleep 1
fi

# --- Step 2: env -------------------------------------------------------------
if [[ ! -f ".env" ]]; then
  echo "ERROR: .env not found. Copy .env.example to .env first." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a
echo "==> env loaded: provider=${LLM_PROVIDER:-?}, model=${LLM_MODEL:-?}, embed=${EMBED_PROVIDER:-?}"
echo

# --- Step 3: start server ----------------------------------------------------
echo "==> starting uvicorn on http://127.0.0.1:$PORT"
nohup python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" \
  > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "    PID: $SERVER_PID"
echo

# --- Step 4: wait for /health ------------------------------------------------
echo "==> waiting for /health"
for i in {1..30}; do
  if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q "200"; then
    echo "    ready after ${i}x2s"
    break
  fi
  if [[ $i -eq 30 ]]; then
    echo "ERROR: server didn't come up in 60s. Last 20 lines of log:" >&2
    tail -20 "$LOG_FILE"
    exit 1
  fi
  sleep 2
done
echo

# --- Step 5: ingest sample.md ------------------------------------------------
if [[ -f "sample.md" ]]; then
  echo "==> ingesting sample.md"
  RESPONSE=$(curl -s -X POST -F "file=@sample.md" "http://127.0.0.1:$PORT/ingest/")
  echo "    $RESPONSE"
  echo
fi

# --- Step 6: smoke query -----------------------------------------------------
echo "==> smoke query: 'What is machine learning?'"
QUERY_RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"question":"What is machine learning?","rewrite_query":false}' \
  "http://127.0.0.1:$PORT/query/")

# Extract just the answer field for a clean preview
ANSWER=$(echo "$QUERY_RESPONSE" | python -c "import sys, json; d=json.loads(sys.stdin.read()); print(d.get('answer','')[:200])" 2>/dev/null || echo "(parse error)")
LATENCY=$(echo "$QUERY_RESPONSE" | python -c "import sys, json; d=json.loads(sys.stdin.read()); print(d.get('latency_ms','?'))" 2>/dev/null || echo "?")
CHUNKS=$(echo "$QUERY_RESPONSE" | python -c "import sys, json; d=json.loads(sys.stdin.read()); print(d.get('retrieved_chunks','?'))" 2>/dev/null || echo "?")
echo "    chunks: $CHUNKS    latency: ${LATENCY}ms"
echo "    answer: ${ANSWER}..."
echo

# --- Step 7: ready -----------------------------------------------------------
cat <<EOF
==> READY

   UI:        http://127.0.0.1:$PORT/
   API docs:  http://127.0.0.1:$PORT/docs
   Config:    http://127.0.0.1:$PORT/query/config
   Health:    http://127.0.0.1:$PORT/health
   Logs:      $LOG_FILE
   PID:       $SERVER_PID

   To stop:   kill $SERVER_PID
              or run: ./stop_demo.sh

EOF
