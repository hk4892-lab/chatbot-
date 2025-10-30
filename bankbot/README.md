# BankBot

BankBot is a production-style multilingual banking support chatbot covering English, Hindi, and Tamil with code-mix handling. It delivers a FastAPI backend, deterministic PII redaction, TF-IDF retrieval augmented generation (RAG) with evidence-or-silence behavior, and a Streamlit dark-mode chat UI. The project includes tool-call mocks for EMI calculation, card blocking, and rate lookups alongside observability, tests, and evaluation utilities.

## Features

- **Multilingual understanding** across English, Hindi, and Tamil with Unicode script detection and code-mix synonym normalization.
- **Deterministic PII redaction** for phone numbers, email, PAN, card, account, and Aadhaar values with reversible tokens kept server-side only.
- **Evidence-or-silence RAG** using TF-IDF retrieval across language-specific knowledge bases with cosine gating and citation rendering.
- **Tool calling** for EMI computation, card blocking ticket generation, and mock rate lookup with safe detokenization.
- **Dialogue policy** that routes between answering, clarifying follow-ups, tool usage, or escalation.
- **Dark mode Streamlit UI** featuring language preference, confidence threshold controls, quick actions, and safety messaging.
- **Observability** via JSONL audit logs and a CLI pretty-printer, plus an evaluation harness for quick quality checks.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api:app --reload --app-dir bankbot
streamlit run bankbot/ui/app_streamlit.py
```

The Streamlit UI expects the FastAPI service to be available at `http://localhost:8000`. Update `~/.streamlit/secrets.toml` with `api_url` if the backend is hosted elsewhere.

## Knowledge base management

Knowledge base entries live in `bankbot/app/data/kb_en.json`, `kb_hi.json`, and `kb_ta.json`. Each entry uses the schema `{id, title, content, tags}`. Add new items, then restart the FastAPI server to reload the in-memory index.

## Sentence-transformer toggle

BankBot defaults to TF-IDF retrieval. Enable semantic search by installing `sentence-transformers` and selecting the checkbox in the Streamlit sidebar. The backend automatically detects availability when `use_sentence_transformers=true` is passed in the request.

## Example queries

- English: "What is the NEFT timing?"
- Hindi: "यूपीआई लिमिट कितनी है"
- Tamil: "ஸ்டேட்மெண்ட் வேண்டுமென்றால் எப்படி"
- Code-mix: "statement send pannunga"

## Evaluation

Run the evaluation harness against the running API:

```bash
python bankbot/scripts/evaluate.py
```

It prints per-prompt JSON summaries, top-1 accuracy against expected substrings, refusal rate, and average confidence. Update `bankbot/scripts/prompts.csv` to customise checks.

## Audit log viewer

Pretty-print recent conversations with:

```bash
python bankbot/scripts/show_logs.py --tail 20
```

## Tests

Execute the automated tests with:

```bash
pytest
```

## Safety and limitations

- This demo does **not** perform actual banking operations; tool calls are mock implementations.
- The reversible PII tokens are only resolved inside tool functions and never exposed in logs or UI.
- Retrieval content is a concise seed KB—expand it to cover production scenarios.

## Disclaimer

BankBot is a demonstration project. It should not be used for real banking operations without comprehensive security, compliance, and risk assessments.
