#!/usr/bin/env bash
set -euo pipefail

PORT="8000"
SUBDOMAIN=""
RESTART_DELAY_SECONDS="5"
STOP_REQUESTED="0"
CHILD_PID=""

usage() {
  cat <<'EOF'
Usage: scripts/start-nport.sh [--port PORT] [--subdomain NAME]

Start an NPort tunnel for an already-running RMBG HTTP API.
The tunnel is restarted when NPort exits, including after NPort's 4-hour cleanup window.

Options:
  --port PORT        Local API port to tunnel. Defaults to 8000.
  --subdomain NAME   Optional nport.link subdomain.
  -h, --help         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:?Missing value for --port}"
      shift 2
      ;;
    --subdomain|-s)
      SUBDOMAIN="${2:?Missing value for --subdomain}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if command -v nport >/dev/null 2>&1; then
  COMMAND=(nport "${PORT}")
elif command -v npx >/dev/null 2>&1; then
  COMMAND=(npx --yes nport "${PORT}")
else
  echo "nport is not installed and npx is not available." >&2
  echo "Install Node.js/npm, then run 'npm install -g nport' or use this script with npx available." >&2
  exit 1
fi

if [[ -n "${SUBDOMAIN}" ]]; then
  COMMAND+=("--subdomain" "${SUBDOMAIN}")
fi

stop() {
  STOP_REQUESTED="1"
  if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" >/dev/null 2>&1; then
    kill "${CHILD_PID}" >/dev/null 2>&1 || true
  fi
}

trap stop INT TERM

echo "Starting NPort tunnel supervisor for local port ${PORT}"
echo "Press Ctrl+C to stop."

while [[ "${STOP_REQUESTED}" != "1" ]]; do
  echo "Starting NPort tunnel for local port ${PORT}"
  "${COMMAND[@]}" &
  CHILD_PID="$!"

  set +e
  wait "${CHILD_PID}"
  STATUS="$?"
  set -e
  CHILD_PID=""

  if [[ "${STOP_REQUESTED}" == "1" ]]; then
    break
  fi

  echo "NPort exited with status ${STATUS}. Restarting in ${RESTART_DELAY_SECONDS}s..."
  sleep "${RESTART_DELAY_SECONDS}"
done

echo "NPort tunnel supervisor stopped."
