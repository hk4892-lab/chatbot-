# BankBot

A production-ready multilingual banking assistant powered by FastAPI, multilingual sentence-transformer embeddings, a FAISS vector index (with NumPy fallback), and the Phi-3 mini SLM for optional generation. The assistant supports English, Tamil, and Hindi queries with safe redaction and deterministic mock tools.

## Features
- FastAPI backend exposing `/`, `/healthz`, and `/chat` endpoints with CORS enabled.
- Retrieval-augmented generation using multilingual sentence-transformer embeddings and FAISS (auto-falling back to NumPy cosine search).
- Optional Phi-3 Mini SLM fallback for low-confidence answers.
- Banking tools for EMI estimates, card blocking tickets, and interest rate lookups.
- Multilingual language detection (English, Tamil, Hindi) with PII redaction before logging.
- Lightweight dark-mode HTML client for quick manual testing.

## Requirements
- Python 3.12

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```
Update `.env` as needed, then launch the API:
```powershell
.\run_api.bat
```
Open http://127.0.0.1:8000/docs to explore the OpenAPI docs.

To serve the static web UI:
```powershell
.\run_ui_http.bat
```
Open http://127.0.0.1:5173 to chat with the assistant.

## Notes
- The system uses embeddings + FAISS only—no TF-IDF components.
- If you switch to an `intfloat/multilingual-e5-*` model set, remember to include the `"query:"` / `"passage:"` prefixes (handled automatically by the code).
- Set `USE_SLM=0` on low-RAM CPUs to skip Phi-3 loading and rely solely on RAG.
- If FAISS is unavailable, the app will automatically switch to a NumPy cosine similarity search.
- Ensure cross-origin requests are allowed when hosting the UI remotely; configure proxies or adjust API base URL as needed.
