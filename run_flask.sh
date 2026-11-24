#!/usr/bin/env bash
set -e
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
. .venv/bin/activate
python -m pip install -r web_flask/requirements.txt
python web_flask/app.py
