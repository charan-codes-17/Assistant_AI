# Personal AI: a friend that remembers what matters

A personal AI assistant with **persistent, structured memory**. You talk to it on **Telegram** (text or voice notes) or on a **web app**. It remembers what matters, reminds you of your promises, and stays quiet when it has nothing useful to say.

Built for the **Nebius x NVIDIA Global AI Hackathon, Personal AI track**.

- Demo video: https://youtu.be/Gzjdfl_CE3c
- Web app: https://assistant-ai-gamma.vercel.app/
- Telegram bot: https://t.me/urmemory_assistant_bot
- License: MIT

> **Demo only. Please do not send real personal data.**

---

## What it does

- **Chat on Telegram or web.** Send text, or a voice note on Telegram. Voice is turned into text first, and the bot shows "Heard: ..." so you can see mistakes.
- **Structured memory, not chat logs.** The assistant extracts memories (knowledge, decisions, preferences) with a certainty level and where they came from.
- **You confirm before saving.** Every new memory waits for one tap (Save / Edit / Discard). Unconfirmed memories are never used.
- **Recall across sessions.** In a new session, the assistant finds related memories, judges if they are relevant, and shows a **confidence %** on Telegram replies with a "Show source" button.
- **Promise tracking.** Say "I'll send Arun the PDF tomorrow". It becomes a task with a due date, and you get one reminder before it is due.
- **Skills.** 8 built-in skills (`weekly-review`, `day-plan`, `open-loops`, `topic-brief`, `explain-simply`, `decision-recall`, `goal-breakdown`, `stress-test-idea`) and your own custom skills. Skills only read your memory. Only `goal-breakdown` can suggest tasks, and you confirm those with one tap.
- **You control your data.** Telegram commands: `/memory`, `/forget` (two steps), `/newsession`, `/skills`, `/skill`. The web app has a Memory page, a Tasks page and a chat history sidebar.

## Where Nemotron and Nebius Token Factory are used

All AI judgment runs on **NVIDIA Nemotron 3 Super** (`nvidia/nemotron-3-super-120b-a12b`) served by **Nebius Token Factory** through its OpenAI-compatible API.

| Step | Model | Service |
|---|---|---|
| Input router (question / note / command) | Nemotron 3 Super | Token Factory |
| Memory extraction and dedupe | Nemotron 3 Super | Token Factory |
| Relevance judgment with confidence | Nemotron 3 Super | Token Factory |
| Intervention decision (remind / surface / none) | Nemotron 3 Super | Token Factory |
| Chat replies and tool calling | Nemotron 3 Super | Token Factory |
| Skills | Nemotron 3 Super | Token Factory |
| Memory embeddings | Qwen3-Embedding-8B, 1024 dimensions | Token Factory |

One model is used for everything, to keep the system simple. Speech-to-text is local (`faster-whisper`), not Nemotron.

## How it works

1. A message arrives from Telegram or the web.
2. The router decides what it is.
3. The assistant may look up memory (vector search in Postgres with pgvector, then a relevance check).
4. The reply is written, with the memories it used.
5. New memories are extracted and wait for your confirmation.
6. A background scheduler checks task due dates and sends one reminder.

## Tech stack

- **Backend:** Python, FastAPI, SQLAlchemy 2.0 (async), Alembic, `openai` SDK, `python-telegram-bot` (long polling), APScheduler, `faster-whisper`
- **Database:** PostgreSQL with pgvector (hosted on Neon)
- **Frontend:** Next.js, React, TypeScript, Tailwind CSS
- **Hosting:** Railway (API and bot), Vercel (frontend), Neon (database)

## Setup

You need: Python 3.12 (the version used in the Docker image), Node.js 20+, a Postgres database with the `vector` extension (Neon works), a Nebius Token Factory API key, and a Telegram bot token from BotFather.

Run all backend commands from the **project root folder** (the folder that contains `alembic.ini`).

### 1. Clone

```bash
git clone https://github.com/charan-codes-17/Assistant_AI.git
cd Assistant_AI
```

### 2. Create your Telegram bot

1. In Telegram, open **@BotFather** and send `/newbot`.
2. Follow the steps and copy the bot token.
3. Use a separate bot for local testing. Only one process can poll a bot token at a time.

### 3. Backend

```bash
python -m venv venv
venv\Scripts\activate        # Windows. On Mac/Linux: source venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env         # Windows: copy .env.example .env
```

Open `.env` and fill in your own values:

- `DATABASE_URL`: your Postgres connection string
- `NEBIUS_API_KEY`: your Token Factory key
- `TELEGRAM_BOT_TOKEN`: the token from BotFather
- `JWT_SECRET`: a long random secret
- `INVITE_CODES`: comma-separated codes that people use to sign up

Also set both chat model variables to the model this project uses (the example file still lists an older model for the reasoning tier):

```
CHAT_MODEL_FAST=nvidia/nemotron-3-super-120b-a12b
CHAT_MODEL_REASONING=nvidia/nemotron-3-super-120b-a12b
```

Embeddings use `Qwen/Qwen3-Embedding-8B` with `EMBEDDING_DIM=1024` (already set in `.env.example`).

In your database, enable the vector extension once:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Create the tables, then start the API:

```bash
alembic upgrade head
uvicorn backend.api.main:app --reload
```

Check it works: open http://localhost:8000/health

**Telegram bot (optional for local use):** set `BOT_ENABLED=true` in `.env` before starting the API. The bot and the reminder scheduler then run inside the same process.

**Voice notes (optional):** download a local Whisper model once, then set `WHISPER_MODEL_NAME=base` in `.env`:

```bash
python -c "from huggingface_hub import snapshot_download as d; d(repo_id='Systran/faster-whisper-base', local_dir='backend/whisper_models/whisper-base')"
```

### 4. Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The web app sends `/api/*` requests to the backend. By default it uses `http://localhost:8000`, so no extra setting is needed for local use. Open http://localhost:3000, sign up with one of your invite codes, and start chatting.

## For judges

Login details and a quick tour are in the **judges-only file on the Devpost submission**. No passwords or invite codes are stored in this repository.

You can also make your own account:
- **Web:** open the web app, choose Sign up, and use an invite code from the judges file.
- **Telegram:** open the bot and send `/start <invite code>`.

## Security notes

- No API keys or bot tokens are in the repo or the frontend bundle. Secrets live only in environment variables.
- Sign-up needs an invite code, and the demo is limited to 25 users.
- Passwords are hashed. Login attempts are throttled. Skills are limited to 10 runs per 10 minutes per user.

## Honest limits

- **Logout does not revoke tokens on the server.** Tokens expire on their own.
- **Skill results are not saved.** They disappear when you reload or switch chat.
- **Voice transcripts can have mistakes.** That is why the bot shows "Heard: ..." and asks before saving.
- Only the features described above are built.

## What's next

Morning brief, a richer memory timeline, and an easier way to link Telegram and web accounts.

## License

MIT. See the [LICENSE](LICENSE) file.
