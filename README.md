# Persistent-Memory Personal AI Assistant

> **Nebius x NVIDIA Global AI Hackathon — Personal AI Track**

> [!WARNING]
> **DEMO ONLY — DO NOT SEND REAL PERSONAL DATA.**
> This is a hackathon demonstration system. Do not transmit, import, or store any sensitive personal data, credentials, real addresses, confidential documents, or financial information.

---

## 1. Overview

A personal AI assistant with **persistent, structured memory** accessible via **Telegram (text or voice notes)** and a companion **web app** featuring a Memory Inspector.

Unlike chatbots that forget everything across sessions, this assistant:
1. **Extracts structured memories** from user conversations.
2. **Stores memories in PostgreSQL with vector embeddings** (via `pgvector`).
3. **Decides when to retrieve relevant memories** in subsequent sessions using LLM judgment.
4. **Decides proactive interventions** (weaving context into replies, flagging contradictions, tracking promises).

Powered by **NVIDIA Nemotron models served via Nebius Token Factory**.

---

## 2. Architecture

```text
User on Telegram (text / voice)          User on web (browser)
 ↓                                        ↓
Telegram Bot worker (python-telegram-bot) Next.js Frontend (thin chat + Memory Inspector)
 ↓ voice → faster-whisper (local STT)     ↓
 └────────────────┬───────────────────────┘
                  ↓
         FastAPI Backend API
                  ↓
         Shared Service Layer
          ├── Input Router (Nemotron fast tier)
          ├── Memory Extraction & Deduplication (Nemotron fast tier + Nebius embeddings)
          ├── Relevance Retrieval (pgvector + Nemotron reasoning tier)
          ├── Intervention & Contradiction Detection (Nemotron reasoning tier)
          └── Action Executor & APScheduler (prompts, promises, background triggers)
                  ↓
         PostgreSQL 16 + pgvector
```

---

## 3. Project Structure

```text
.
├── backend/
│   ├── api/          # FastAPI REST endpoints
│   ├── bot/          # Telegram bot worker
│   ├── config/       # Pydantic settings & environment configuration
│   └── services/     # Shared AI/ML services (router, extraction, retrieval, intervention)
├── frontend/         # Next.js 14+ (App Router, TypeScript, Tailwind)
├── docker/           # Docker setup and database initialization scripts
├── docs/             # PRD and build roadmap
├── docker-compose.yml
├── LICENSE           # MIT License
└── README.md
```

---

## 4. Quick Start (Development)

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker & Docker Compose
- Nebius Token Factory API Key
- Telegram Bot Token (from BotFather)

### Environment Configuration
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

### Start Database
```bash
docker compose up -d
```

### Start Backend API
```bash
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```
Verify the API health check at [http://localhost:8000/health](http://localhost:8000/health).

### Start Frontend
```bash
cd frontend
npm install
npm run dev
```
Visit [http://localhost:3000](http://localhost:3000).

---

## 5. Deploy (Railway container + hosted Postgres)

The backend ships as a single container built from the repo-root `Dockerfile`
(base image `python:3.12-slim`). `railway.toml` selects the Dockerfile builder and
health-checks `GET /health`.

### Build notes
- The image installs dependencies from `backend/requirements.lock` (exact pinned versions).
- The build **bakes both Whisper models** into the image at build time, into
  `/app/whisper_models/whisper-small` and `/app/whisper_models/whisper-base` (plain
  folders, no symlinks). This layer comes before the application code, so code changes
  do not re-download the models. At runtime the app reads `WHISPER_MODEL_ROOT`.
- The container runs as a non-root user. No secrets are baked into the image.

### Start command
The container's entrypoint is `backend/deploy/start_api.sh`, which runs migrations and
then serves the API:

```bash
sh backend/deploy/start_api.sh
# -> alembic upgrade head
# -> exec uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

`alembic upgrade head` must succeed first; if it fails the container exits non-zero.
There is no `--reload`.

### Environment variables for the API service (names only; set values in the platform)
Required:
- `APP_ENV` — must be `production` in the cloud.
- `DATABASE_URL` — Postgres async connection string (hosted, with the `vector` extension).
- `NEBIUS_API_KEY`
- `CHAT_MODEL_FAST`
- `CHAT_MODEL_REASONING`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIM`

Optional (sensible defaults exist):
- `NEBIUS_BASE_URL`
- `CHAT_MODEL_TOOLCALL`
- `EMBEDDING_TIMEOUT_SECONDS`, `EMBEDDING_HEDGE_SECONDS`, `EMBEDDING_BATCH_TIMEOUT_SECONDS`
- `RETRIEVAL_TOP_K`, `RETRIEVAL_MIN_SIMILARITY`, `RELEVANCE_MIN_CONFIDENCE`,
  `RETRIEVAL_TIMEOUT_SECONDS`
- `DEDUPE_THRESHOLD`, `PENDING_DEDUPE_TTL_HOURS`
- `ONBOARDING_MAX_CHARS`, `ONBOARDING_MAX_CANDIDATES`
- `WHISPER_MODEL_NAME` (defaults to `small`; `base` is also baked)
- `WHISPER_MODEL_ROOT` (the image sets this to `/app/whisper_models`)
- `REMINDER_LEAD_MINUTES`, `INTERVENTION_TIMEOUT_SECONDS`

`TELEGRAM_BOT_TOKEN` is **not** required by the API service (only the bot worker needs it).

### Security notes
- The development identity seam (`X-Dev-User-Id` header) only works when
  `APP_ENV == "development"`. In production (`APP_ENV=production`) that header is
  **rejected** (the request is treated as unauthenticated). The default `APP_ENV` is
  `dev`, which also fails closed for the seam.
- API docs (`/docs`) are only exposed in development.
- CORS is disabled for the API.

### Verifying a deploy (skeleton smoke test)
`backend/scripts/deploy_smoke.py` runs the same checks locally and inside the container
(runtime, required settings, DB latency, pgvector, migration head, embedding, chat, and
one end-to-end orchestrator turn). It creates and removes a throwaway user and never
prints secret values. `backend/scripts/whisper_probe.py --model small|base` measures
Whisper load time, transcribe time and peak RSS against the baked models.

---

## 6. License

This project is licensed under the terms of the [MIT License](LICENSE).

