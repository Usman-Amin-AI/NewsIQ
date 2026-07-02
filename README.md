# NewsBot

NewsBot is a modular Streamlit news research assistant with URL ingestion, per-user vector store isolation, service-backed Chroma support, structured logging, and admin metrics.

## Features

- URL-based article ingestion with HTML extraction and content hashing
- Local FAISS or Chroma backend with optional remote Chroma service
- Per-user vector index isolation and document cache
- Basic Streamlit login with user roles and admin dashboard
- Sentence-level citations and groundedness evaluation
- Containerized deployment using Docker and Docker Compose

## Running locally

1. Create a `.env` file or set environment variables.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

## Docker Compose deployment

This repository includes a `Dockerfile` and `docker-compose.yml` for running NewsBot with a Chroma service backend.

1. Copy `.env.example` to `.env` and update credentials:
   ```bash
   cp .env.example .env
   ```
2. Build and start the services:
   ```bash
   docker compose up --build
   ```
3. Visit `http://localhost:8501`

## Environment variables

- `EMBEDDING_PROVIDER` — `openai` or `sentence-transformer`
- `EMBEDDING_MODEL` — embedding model name
- `LLM_PROVIDER` — `openai` or `anthropic`
- `LLM_MODEL` — LLM model name
- `VECTORSTORE_BACKEND` — `faiss` or `chroma`
- `USE_CHROMA_SERVICE` — `true` to connect to a remote Chroma API server
- `CHROMA_SERVER_HOST` — host for the Chroma service in Docker Compose (`chroma`)
- `CHROMA_SERVER_PORT` — port for the Chroma service (`8000`)
- `AUTH_USERS` — comma-delimited `username:password` pairs
- `ADMIN_USERS` — comma-delimited admin usernames
- `LOG_PATH` — path to application event log
- `METRICS_PATH` — path to metrics JSON file

## Notes

- Per-user vector stores are isolated under the configured `VECTORSTORE_PATH`.
- The admin dashboard displays daily query counts, average latency, token spend, and recent event logs.
- In Docker Compose, the `chroma` service is available at `http://chroma:8000`.
