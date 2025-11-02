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
uvicorn app.api:app --reload --app-dir bankbot  # optional for REST integration
streamlit run bankbot/ui/app_streamlit.py
```

> **Note:** Reinstall dependencies (`pip install --upgrade -r requirements.txt`) after pulling
> updates so FastAPI 0.111 and Pydantic 2.7 are available—these versions fix import errors on
> Python 3.13.

The Streamlit UI runs the chat engine locally, so it does not require the FastAPI service. Start the API if you want REST access from other clients.

## Knowledge base management

Knowledge base entries live in `bankbot/app/data/kb_en.json`, `kb_hi.json`, and `kb_ta.json`. Each entry uses the schema `{id, title, content, tags}`. Add new items, then restart the FastAPI server to reload the in-memory index.

## Sentence-transformer toggle

BankBot defaults to TF-IDF retrieval. Enable semantic search by installing `sentence-transformers` and selecting the checkbox in the Streamlit sidebar. The backend automatically detects availability when `use_sentence_transformers=true` is passed in the request.

## Example queries

- English: "What is the NEFT timing?"
- Hindi: "यूपीआई लिमिट कितनी है"
- Tamil: "ஸ்டேட்மெண்ட் வேண்டுமென்றால் எப்படி"
- Code-mix: "statement send pannunga"

## Sample output preview

Running the shared chat engine locally (without starting the API) produces structured
responses. The snippet below shows an English query, a Hinglish code-mix query, and an
EMI calculation request handled via the tool router:

```
============================================================
User: Hi, what are the NEFT timings?
Route: answer
Confidence: 0.232
Answer: NEFT transfers are available 24x7 with settlement in half-hour batches except during maintenance windows.
Citation: {'id': 'kb_en_5', 'title': 'NEFT Timings'}
Citation: {'id': 'kb_en_4', 'title': 'EMI Formula'}
Citation: {'id': 'kb_en_1', 'title': 'Savings Account Minimum Balance'}
============================================================
User: UPI limit kitna hai?
Route: answer
Confidence: 0.296
Answer: Daily UPI limit is INR 1 lakh with up to 20 transfers; limits may vary by merchant category.
Citation: {'id': 'kb_en_3', 'title': 'UPI Transaction Limits'}
Citation: {'id': 'kb_ta_1', 'title': 'சேமிப்பு கணக்கு குறைந்த இருப்பு'}
Citation: {'id': 'kb_ta_2', 'title': 'டெபிட் கார்டு முடக்கம்'}
============================================================
User: emi 500000 10% 60 months
Route: tool
Confidence: 1.0
Answer: Calculated EMI displayed below.
Tool Result: {'emi': 10623.52}
```

These transcripts are produced by executing:

```bash
PYTHONPATH=. python - <<'PY'
from pathlib import Path
from bankbot.app.core.engine import ChatEngine

engine = ChatEngine(Path('bankbot/app/data'))

for message in [
    "Hi, what are the NEFT timings?",
    "UPI limit kitna hai?",
    "emi 500000 10% 60 months",
]:
    result = engine.run(message)
    print('=' * 60)
    print('User:', message)
    print('Route:', result.decision.route)
    print('Confidence:', round(result.decision.confidence, 3))
    if result.decision.answer:
        print('Answer:', result.decision.answer)
    if result.decision.tool_result:
        print('Tool Result:', result.decision.tool_result)
    for citation in result.decision.citations or []:
        print('Citation:', citation)
PY
```

Feel free to swap in your own prompts or integrate the engine directly into custom pipelines.

## Evaluation

Run the offline evaluation harness against your prompt dataset:

```bash
python bankbot/scripts/evaluate.py --prompts bankbot/scripts/prompts.csv
```

It prints per-prompt JSON summaries, top-1 accuracy against expected substrings, refusal rate, and average confidence. Provide your own CSV with `--prompts` to evaluate custom datasets.

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
