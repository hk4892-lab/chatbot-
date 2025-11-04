@echo off
call .venv\Scripts\activate
set HOST=127.0.0.1
set PORT=8000
python -m uvicorn bankbot.app.api:app --host %HOST% --port %PORT% --reload
