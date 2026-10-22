#!/bin/zsh
set -eu

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

export ANONYMIZED_TELEMETRY=False
export LLAMA_HOST="${LLAMA_HOST:-http://127.0.0.1:8080}"
export PYTHONIOENCODING=utf-8
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

PY="$PROJECT_DIR/.venv/bin/python"
SERVER="$PROJECT_DIR/llama.cpp-macos/llama-server"

if [[ ! -x "$PY" ]]; then
  echo "Python environment is missing. Run: ./setup.sh"
  exit 1
fi

NEEDS_SERVER=false
case "${1:-app}" in
  app|ingest|reload) NEEDS_SERVER=true ;;
esac

if [[ "$NEEDS_SERVER" == true ]] && ! curl -fsS "$LLAMA_HOST/health" >/dev/null 2>&1; then
  if [[ ! -x "$SERVER" ]]; then
    echo "Local model server is missing: $SERVER"
    exit 1
  fi
  mkdir -p data
  echo "Starting the local model server..."
  "$SERVER" --models-preset "$PROJECT_DIR/models.ini" --host 127.0.0.1 --port 8080 --models-max 4 \
    >"$PROJECT_DIR/data/llama-server.log" 2>&1 &
  for _ in {1..120}; do
    curl -fsS "$LLAMA_HOST/health" >/dev/null 2>&1 && break
    sleep 1
  done
  if ! curl -fsS "$LLAMA_HOST/health" >/dev/null 2>&1; then
    echo "The local model server did not start. See data/llama-server.log"
    exit 1
  fi
fi

case "${1:-app}" in
  app)
    exec "$PY" -m streamlit run app.py
    ;;
  ingest)
    shift
    [[ $# -ge 1 ]] || { echo "Usage: ./run.sh ingest path/to/file.pdf [--no-vision] [--fresh]"; exit 2; }
    exec "$PY" library.py add "$@"
    ;;
  list)
    exec "$PY" library.py list
    ;;
  reload)
    exec "$PY" library.py reload
    ;;
  calculations)
    shift
    exec "$PY" generic_metrics.py "$@"
    ;;
  business-rules)
    shift
    [[ $# -ge 1 ]] || { echo "Usage: ./run.sh business-rules WORKBOOK_OR_ZIP [--apply] [--approve-examples] [--pdf PDF]"; exit 2; }
    exec "$PY" business_rules_pipeline.py "$@"
    ;;
  *)
    echo "Usage: ./run.sh [app|list|reload|calculations ...|business-rules FILE ...|ingest PDF [options]]"
    exit 2
    ;;
esac
