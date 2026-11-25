@echo off
setlocal
title Meyve Sebze Tespit - Flask
if not exist .venv\Scripts\python.exe py -m venv .venv
call .venv\Scripts\activate.bat
if not exist .venv\.deps_ready (
  python -m pip install --upgrade pip
  python -m pip install -r web_flask\requirements.txt
  type nul > .venv\.deps_ready
)
start "Meyve Sebze Tespit Sunucusu" /b .venv\Scripts\python.exe web_flask\app.py
timeout /t 3 /nobreak > nul
start "" http://127.0.0.1:5000
echo Uygulama acildi: http://127.0.0.1:5000
echo Sunucuyu kapatmak icin bu pencereyi kapatin.
pause
