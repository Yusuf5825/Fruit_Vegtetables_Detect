#!/usr/bin/env bash
set -e
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
. .venv/bin/activate
if [ ! -f .venv/.deps_ready ]; then
  python -m pip install --upgrade pip
  python -m pip install -r web_flask/requirements.txt
  touch .venv/.deps_ready
fi
python web_flask/app.py &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
sleep 3
if command -v xdg-open >/dev/null 2>&1; then xdg-open http://127.0.0.1:5000 >/dev/null 2>&1 || true; fi
wait $SERVER_PID
