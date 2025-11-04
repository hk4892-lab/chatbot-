@echo off
REM Serve the UI folder on http://127.0.0.1:5173
python -m http.server 5173 --directory bankbot/app/ui/web
