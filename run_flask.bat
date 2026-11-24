@echo off
setlocal
if not exist .venv\Scripts\python.exe py -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r web_flask\requirements.txt
python web_flask\app.py
