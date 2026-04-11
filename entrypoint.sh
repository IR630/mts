#!/usr/bin/env bash
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5-coder:7b-instruct-q4_K_M}"

echo "=== Waiting for Ollama at ${OLLAMA_HOST} ==="
until curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; do
    echo "  Ollama not ready, retrying in 3s…"
    sleep 3
done
echo "=== Ollama is up ==="

echo "=== Pulling model: ${OLLAMA_MODEL} ==="
curl -sf -X POST "${OLLAMA_HOST}/api/pull" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"${OLLAMA_MODEL}\",\"stream\":false}" \
    --max-time 900 || echo "Pull request sent (may already exist)"
echo "=== Model ready ==="

echo "=== Starting FastAPI ==="
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
