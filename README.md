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

## 5. License

This project is licensed under the terms of the [MIT License](LICENSE).

