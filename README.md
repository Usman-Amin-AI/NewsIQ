# NewsIQ

NewsIQ is a Streamlit-based news research assistant that ingests article URLs, builds searchable embeddings, and helps users query content with grounded answers. It supports local vector stores or a Chroma-backed service, user authentication, admin metrics, and Docker-based deployment.

## What it does

- Ingests news article URLs and extracts readable content
- Splits content into chunks and creates embeddings
- Stores and queries document vectors using FAISS or Chroma
- Supports a simple sign-in flow with admin-only monitoring views
- Tracks usage metrics and application events for observability

## Project structure

- [app.py](app.py) — main Streamlit application entry point
- [src/](src) — ingestion, embeddings, vector store, auth, evaluation, and metrics modules
- [tests/](tests) — regression tests for ingestion, providers, and evaluation
- [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml) — containerized deployment setup

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the sample environment file and adjust values:
   ```bash
   copy .env.example .env
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Environment variables

The app uses the following environment variables, most of which are already documented in [.env.example](.env.example):

- `EMBEDDING_PROVIDER` — `openai` or `sentence-transformer`
- `EMBEDDING_MODEL` — embedding model name
- `LLM_PROVIDER` — `openai` or `anthropic`
- `LLM_MODEL` — LLM model name
- `VECTORSTORE_BACKEND` — `faiss` or `chroma`
- `VECTORSTORE_PATH` — local path for persisted vector data
- `USE_CHROMA_SERVICE` — enable remote Chroma service mode
- `AUTH_USERS` — `username:password` pairs for login
- `ADMIN_USERS` — admin usernames
- `LOG_PATH` — application log file path
- `METRICS_PATH` — metrics JSON output path

## Docker

To run the app with Docker Compose:

```bash
docker compose up --build
```

Then open http://localhost:8501.

## Testing

Run the test suite with:

```bash
pytest -q
```
