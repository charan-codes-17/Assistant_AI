# Product Requirements Document (PRD) — Persistent-Memory Personal AI Assistant

**Event:** Nebius x NVIDIA Global AI Hackathon — Personal AI Track
**Version:** 3.0 (derived from Build Roadmap revision of Oct 16, 2026)
**Deadline:** Oct 30, 2026, 10:00am PT (Oct 30, 10:30pm IST)
**Team:** Solo build
**Status:** Draft — pending sign-off on the open items in §14

> Note: FR1–FR14 are reconstructed from the roadmap's PRD→implementation mapping (§9 of the roadmap); the original PRD v2 wording may differ slightly. FR15–FR24 are taken verbatim from the roadmap's new requirements table.

---

## 1. Overview

### 1.1 Product summary
A personal AI assistant with **persistent, structured memory** that you talk to on **Telegram (text or voice notes)** — like a friend who takes notes of everything you say and helps you later — with a secondary **web app** that provides a chat client and a Memory Inspector. It extracts structured memories from what the user says, stores them with vector embeddings, decides when to retrieve them in later conversations, and decides what to do with what it retrieves (weave into an answer, flag a contradiction, log a task update, remind of a promise).

All judgment runs on **NVIDIA Nemotron models served via Nebius Token Factory** (OpenAI-compatible API).

### 1.2 Problem statement
Chatbots forget everything between sessions, and users must keep going back to an app or website to use them. Users re-brief the assistant every time, and nothing proactively helps them follow up on what they said.

### 1.3 Vision
An assistant that lives where the user already chats, remembers what matters (not raw logs), uses it across sessions without re-briefing, and speaks up only when it's useful — including saying nothing.

### 1.4 Goals
1. Demonstrate genuine cross-session memory: structured extraction → confirmed storage → relevance-judged retrieval → intervention.
2. Make Telegram (text + voice) the primary, frictionless interface.
3. Show transparent memory: confidence %, provenance, and source record visible to the user.
4. Show background proactivity (promise/deadline reminders).
5. Meet all hackathon submission requirements (Nebius + NVIDIA usage, public repo, OSI license, ≤3 min demo video, working demo URL).

### 1.4.1 Non-goals (this release)
- Memory Graph visualization (cut)
- Learning features (SM-2, prerequisites), calendar integration, screenshot/vision filer, research/content pipelines, relationship tracker (Future)
- Full multi-tenant public signup; fine-tuning; agent frameworks; Neo4j; Redis/Celery

### 1.5 Success criteria
- The "functions → calculus" cross-session loop runs live on the deployed URL **and** the deployed Telegram bot.
- A voice-note promise produces a single Telegram reminder from the scheduler.
- Judged criteria satisfied: real functionality over scripted video, design (Inspector), Nebius/NVIDIA usage.
- Zero real personal data in repo, README or video.

---

## 2. Users and Personas

- **Primary (demo) user:** a single generic persona — provisionally "Riya, a CS student" (needs sign-off; student / professional / general are alternatives).
- **Testers/judges:** access via invite code (Telegram `/start <code>`) or web login; each gets an isolated account; capped at ~10 concurrent users.
- No open multi-tenant signup.

---

## 3. Scope

### 3.1 In scope (MVP)
Telegram text and voice; web chat (thin); input router; LLM memory extraction with merge/dedupe; structured memory with certainty and provenance; reactive relevance retrieval with confidence; intervention layer; promise tracking and reminders; decision log; onboarding (import + questionnaire); Memory Inspector and timeline; one-tap write confirmation; session simulation; auth, Telegram linking, concurrency cap, rate limits; design safeguards.

### 3.2 Tiering of the extended feature list
| Tier | Features |
|---|---|
| **MVP** | Telegram text/voice, import and questionnaire onboarding, semantic memory search, provenance + certainty, epiphany capture (voice → cleaned thesis + tags), Memory Inspector + `/memory` + `/forget`, triggered callback, decision log, promise tracking |
| **Stretch** (only after core verified) | Richer memory timeline, ephemeral memory, morning brief, late-night nudge, deadline-risk warnings |
| **Comes free with core loop** | Glossary, item-location memory, open-loop answers, preferences profile, idea stress-testing, decision matrix, tone calibration/drafting (draft-only) |
| **Future (V1+)** | Learning features, calendar/meeting briefs, relationship tracker, research collector, content pipeline, README/pitch generators, weekly check-ins, screenshot filer, appliance helper, return-window reminders |

Anything not built must **not** be claimed in the README or demo video.

---

## 4. Functional Requirements

### 4.1 Memory and AI core
| ID | Requirement | Priority |
|---|---|---|
| FR1 | LLM-driven memory extraction after each turn, with similarity-based merge/dedupe against existing records; records carry kind, certainty and provenance | Must |
| FR2 | Structured storage in Postgres + pgvector: Knowledge, Decision, Preference, Glossary, Ephemeral records; Goals/Projects/Milestones/Tasks/Progress; free-text profile blob | Must |
| FR3 | Reactive relevance retrieval: the model decides when to look up memory; pgvector search over confirmed, non-expired records; LLM relevance judgment with a confidence score; includes "triggered callback" | Must |
| FR4 | Intervention decision (none / remind / surface / suggest / act), including contradiction flagging against stored Goals/Tasks/deadlines; `none` is a valid outcome | Must |
| FR5 | Action execution; memory writes held as pending until user confirmation | Must |
| FR10 | All LLM services call NVIDIA Nemotron via Nebius Token Factory; embeddings via Nebius | Must |
| FR19 | Input router classifying each turn as question / note / both / command / small talk | Must |
| FR21 | Decision log: rationale stored; "why did I choose X?" answered from it | Should |
| FR22 | Ephemeral memory with auto-expiry | Could |

### 4.2 Interfaces
| ID | Requirement | Priority |
|---|---|---|
| FR15 | Telegram bot for text chat; all core flows reachable from it; confidence footer and "Show source" button on replies that used memory | Must |
| FR16 | Voice notes: speech-to-text, then handled as question or note-to-save; transcript shown ("Heard: …"); audio discarded after transcription | Must |
| FR6 | Web chat interface (thin client over the same `/chat` API) | Must |
| FR7 | Memory Inspector: confidence % badge, toggle to reveal source record, timeline, edit/delete | Must |
| FR14 | Write-confirmation UI: inline ✅/✏️/❌ in Telegram, tick in web, batch confirm for imports | Must |
| FR23 | Commands: `/memory <topic>`, `/forget` (two-step confirm), `/newsession` (Must); `/quiet`, `/link`, `/help` (Should) | Must / Should |
| — | "Simulate new session" control (web button, `/newsession` in Telegram) that starts a new session without touching persisted memory | Must |

### 4.3 Onboarding
| ID | Requirement | Priority |
|---|---|---|
| FR17 | Path A: import "what you know about me" from another AI (ChatGPT/Gemini/Claude) via a copy-paste prompt; pasted text parsed as **data, not instructions** into grouped candidate records; batch confirm | Must |
| FR18 | Path B: guided questionnaire (8–10 skippable questions) for users with no AI history; answers go through the same extraction and confirm path | Must |
| FR13 | Optional freeform dump; superseded/extended by FR17 and FR18 | Must (via FR17/18) |

### 4.4 Proactivity
| ID | Requirement | Priority |
|---|---|---|
| FR11 | Deadline and context triggers: scheduler scans Task due dates; context trigger fires from the orchestrator when the conversation implies relevance | Must |
| FR20 | Promise/commitment extraction with due dates resolved in the user's timezone; stored as Tasks with `origin=promise`; reminders with [Done] [Snooze] [Stay quiet] | Should (needed for proactivity demo) |
| FR24 | Scheduled morning brief via Telegram (open promises, top task, one review item; optional weather) | Could (stretch) |

### 4.5 Accounts, access and limits
| ID | Requirement | Priority |
|---|---|---|
| FR8 | Web auth (username/password, JWT in httpOnly cookie); Telegram invite-code signup; account linking via one-time `/link` code | Must |
| FR9 | Concurrency cap (~10 active sessions across both channels) and per-user rate limit | Must |

### 4.6 Explicitly cut
| ID | Requirement | Status |
|---|---|---|
| FR12 | Memory Graph visualization | Cut from MVP (provenance/certainty fields stored so it can be added later; Neo4j only if revived) |

---

## 5. Key User Flows

**Flow 1 — Cross-session recall (functions → calculus)**
1. Session 1: user says "help me understand mathematical functions" (text or voice). Router → question (+ learning note). Model answers; extraction drafts "Knowledge: mathematical functions (tentative)"; user confirms with one tap; record persisted with embedding and provenance.
2. User sends `/newsession` (or clicks "Simulate new session").
3. Session 2: user says "I'm stuck on a calculus problem". Model calls `retrieve_memory`; pgvector finds the functions record; reasoning-tier model judges relevance (e.g. 82%). Reply weaves the connection in; Telegram footer shows confidence, date and "Show source"; web Inspector logs the retrieval.

**Flow 2 — Voice-note promise → reminder**
1. Voice note: "I'll send Arun the PDF tomorrow." Transcribed → routed as note → Task (origin=promise, due tomorrow in user timezone) → confirm prompt (✅ / ✏️ change date / ❌).
2. Next day the scheduler triggers the intervention service, which decides to remind (or stay silent under quiet mode / already done). Telegram reminder with [Done] [Snooze] [Stay quiet].

**Flow 3 — Onboarding by import**
1. `/start <invite_code>` → "Have you used ChatGPT, Gemini or Claude?"
2. Yes: bot supplies a copy-paste prompt; user pastes the answer; extraction (data-only) → grouped candidates (Background, Goals, Projects, Preferences, People) → [Save all] [Review one by one] [Discard].
3. No: 8–10 question questionnaire (skippable) → same extraction and confirm.

---

## 6. System Architecture (summary)

One **shared service library** (router, extraction, retrieval, intervention, action executor, onboarding) with entrypoints: **FastAPI API** and **Telegram bot worker** (hosts APScheduler). The Next.js frontend talks only to the API.

| Component | Technology |
|---|---|
| Telegram bot worker | `python-telegram-bot` (async), long polling |
| Speech-to-text | `faster-whisper` (small/base, int8, CPU), local |
| Backend API | FastAPI, SQLAlchemy 2.0 async, Alembic |
| Database | Postgres 15+ with pgvector (`pgvector/pgvector:pg16`) |
| LLMs | NVIDIA Nemotron via Nebius Token Factory using the `openai` SDK; fast tier for routing/extraction/onboarding/chat, reasoning tier for relevance and intervention |
| Embeddings | Nebius embedding model (model ID and dimension in config); local `all-MiniLM-L6-v2` fallback |
| Scheduler | APScheduler inside the bot worker (single instance) |
| Frontend | Next.js 14+ (App Router), React 18, TypeScript, Tailwind, zustand, axios |
| Deployment | Railway (API + bot worker + Postgres), Vercel (frontend) — provisional |

---

## 7. Data Requirements

| Entity | Key fields |
|---|---|
| `User` | id, username, password_hash (nullable), telegram_chat_id (nullable), timezone, quiet_mode, onboarding_state |
| `LinkCode` | code, user_id, expires_at |
| `ConversationSession` | id, user_id, channel (telegram/web), started_at |
| `KnowledgeRecord` | id, user_id, kind (knowledge/decision/preference/glossary/ephemeral), content, certainty (idea/tentative/decided), source (text/voice/import/questionnaire), source_ref, rationale, embedding (Vector, dimension from config), status (pending/confirmed/rejected), expires_at, superseded_by, created_at |
| `Goal`, `Project`, `Milestone`, `Progress` | as in PRD v2 |
| `Task` | id, project_id (nullable), title, due_at, status, origin (manual/extracted/promise) |
| `ProfileSummary` | user_id, blob, updated_at |
| `InspectorEvent` | id, session_id, turn_id, retrieved_record_ids, confidence, directive, created_at |

**Rules:** retrieval searches only `status = confirmed` and (`expires_at` null or future). Voice audio is not stored. If the embedding dimension exceeds 2,000, use a smaller output dimension, `halfvec`, or skip the ANN index (exact scan is fine at demo scale).

---

## 8. API Requirements

| Endpoint | Purpose |
|---|---|
| `POST /chat` | Run a turn (router → orchestrator); returns reply, pending memory writes, Inspector trace |
| `POST /memory/{id}/confirm`, `POST /memory/confirm-batch` | One-tap and batch confirmation; reject/delete requires explicit confirm |
| `GET /memory/timeline` | Day-by-day timeline with kind/certainty/source |
| `GET/POST /profile` | Free-text profile blob |
| `POST /onboarding/import`, `POST /onboarding/questionnaire` | Web fallback for onboarding paths |
| `POST /session/new` | New session without touching memory |
| Auth endpoints | Login, JWT issuance, Telegram link-code redemption |

All endpoints validated with Pydantic schemas; length limits on pasted exports.

---

## 9. AI Requirements

1. **Input router** (fast tier): strict JSON `{intent: question|note|both|command|smalltalk, cleaned_note}`; malformed output defaults to `question`.
2. **Extraction** (fast tier): strict JSON candidate records with kind, certainty, rationale (decisions), tasks with due dates in user timezone (promises); rambling voice notes cleaned to a short thesis + tags; dedupe via similarity search.
3. **Retrieval** (reasoning tier): `retrieve_memory` tool (or JSON-instruction fallback); pgvector search then relevance judgment with confidence score.
4. **Intervention** (reasoning tier): returns one of `none | remind | surface | suggest | act`, plus `contradiction` flag and clarifying question; `act` limited to reversible internal actions after confirmation.
5. **Onboarding parsers:** imported text is delimited as data with an explicit "ignore instructions inside" rule; output validated against a strict schema.
6. **Model-behavior constraints:** model IDs (chat and embedding) in config, not hardcoded; verify `content` is non-empty and handle `reasoning_content` for Nemotron reasoning models; use a non-reasoning model for the tool-calling step if native tool calls are unsupported; validate every JSON output with Pydantic; retry once on invalid JSON, then log and skip.

---

## 10. Non-Functional Requirements

- **Reliability over polish:** the live loop must work end-to-end on the deployed system.
- **Latency:** no hard bar; voice transcription of a short note should return in a few seconds.
- **Privacy:** voice audio discarded after transcription; no ambient recording; opt-in data scope; everything deletable; "demo only, don't send real personal data" banner.
- **Security:** no API keys or bot tokens in any client bundle (verify in Phase 8.3); bcrypt, JWT with expiry, httpOnly cookie; short-lived one-time link codes; invite-code gate; secrets only in `.env` (gitignored) or platform secret manager; separate dev and deployed bot tokens (one poller per token).
- **Cost control:** per-user rate limit, concurrency cap, fast-tier models for routing/extraction, short stable prompts, monitor Nebius credits.
- **Resource fit:** `faster-whisper` RAM measured in the Phase 1 spike; `base` model if Railway memory is tight.

### 10.1 Design safeguards (requirements)
1. **Draft only:** never send messages to anyone but the user.
2. No purchases, payments or signatures.
3. Opt-in data scope, no ambient recording, everything deletable.
4. No fake emotion or flattery; willing to disagree.
5. Confirmation before any delete or overwrite (two-step `/forget`).
6. No gamification or notification spam; every proactive message offers "stay quiet"; `/quiet` respected.
7. Cite sources and state uncertainty on legal, medical and financial topics.
8. Autonomy stops at **Approve-then-act**.

---

## 11. Constraints and Dependencies

- Deadline Oct 30, 2026; **~14 days remain** from Oct 16.
- Solo; stated budget 5–10 hrs/week (≈10–20 hrs available vs ≈35–40 hrs planned).
- Must run on Nebius Token Factory / AI Cloud with ≥1 NVIDIA open model.
- Public repo with OSI license (MIT proposed); ≤3 min demo video; working demo URL required (Telegram alone insufficient).
- No real personal data in repo, README or video; use a dedicated demo Telegram account and the demo persona.
- Locked decisions: Postgres + pgvector (not SQLite/Chroma, not Neo4j); Next.js/React frontend; fresh repo with zero SecondSelf code reuse.

---

## 12. Milestones and Scope Control

| Dates | Phase |
|---|---|
| Oct 16–17 | Phase 0–1: credits, repo, DB, Nebius/embedding/Telegram/STT spikes |
| Oct 18–20 | Phase 2: AI layer (protect this time) |
| Oct 21–23 | Phase 3–4: API + Telegram |
| Oct 24 | Phase 5: promise/deadline reminder |
| Oct 25–26 | Phase 6: thin web chat + Inspector |
| Oct 27 | Phase 7: auth, linking, caps |
| Oct 28 | Phase 8: deploy |
| Oct 29 | Demo video, README, text description, submit (Oct 30 = buffer) |

**Cut order if behind (top first):** (1) stretch tier, (2) web onboarding fallback and `/quiet` polish, (3) one onboarding path (keep whichever demos better), (4) web chat client (keep Inspector), (5) polish on commands other than `/memory`, `/forget`, `/newsession`.

**Never cut:** input router → extraction → confirm → retrieval with confidence → intervention, working end-to-end on one channel, deployed, with a demo video.

**Reduced MVP (if hours stay at 5–10/week):** Telegram text + voice → router → extraction → inline confirm → retrieval with confidence footer → one promise reminder; identity by Telegram chat ID + invite code only (no JWT/web login); Inspector as a single read-only web page.

---

## 13. Acceptance Criteria (Definition of Done)

- [ ] Functions → calculus example runs live on the deployed web URL and the deployed Telegram bot.
- [ ] Voice note is transcribed, routed, and (if a note) becomes a pending memory confirmable with one tap.
- [ ] Onboarding works via import and via questionnaire, both ending in batch confirm.
- [ ] Unconfirmed memories are never retrieved.
- [ ] Inspector shows confidence % and reveals source on toggle; Telegram replies carry a confidence footer with "Show source".
- [ ] "Simulate new session" / `/newsession` proves cross-session persistence.
- [ ] A voice-note promise yields exactly one Telegram reminder; quiet mode suppresses it.
- [ ] Contradictions between new input and stored Goal/Task/deadline are flagged and resolved per the user's choice; no false contradictions invented.
- [ ] `/memory`, `/forget` (two-step), `/newsession` work; deletion requires confirmation.
- [ ] Auth works; Telegram signup requires invite code; accounts link; 11th concurrent session rejected; rate limit holds.
- [ ] Safeguards visible in behavior (no flattery, no third-party messages, "stay quiet" offered, uncertainty on legal/medical/financial).
- [ ] No API keys or bot tokens in the frontend bundle or network requests.
- [ ] Prompt-injection tests pass (user message and pasted export containing instructions).
- [ ] Repo public with visible OSI license and no real personal data.
- [ ] README covers setup (incl. creating the Telegram bot), explicit Nebius/NVIDIA usage, and how judges get an invite code.
- [ ] Demo video ≤3 min, narrated, public on YouTube, covers Nebius/NVIDIA usage, voice/Telegram interaction, and background proactivity.
- [ ] Text description and feedback section written; Nebius credits confirmed sufficient.
- [ ] Unbuilt Stretch/Future features are not claimed anywhere.

---

## 14. Open Questions and Decisions Needed

| Item | Status |
|---|---|
| Demo persona ("Riya, a CS student" or other) | Needs confirmation before frontend seed data (Phase 6) |
| Nemotron variant split (fast tier for chat/routing/extraction, Ultra for relevance/intervention) | Provisional: split by task |
| Deployment target | Provisional: Vercel + Railway; confirm before Phase 8 |
| Memory Graph | Provisional: cut |
| Nebius credits | Verify the ~$29.50 balance is the hackathon credit (code `NEBIUS-DEVPOST-GLOBAL26` + Builder Program); do not rely on the $1 trial credit; Phase 0 |
| Builders & Brews / Tavily bonus track | Recommend skipping |
| Judge/tester access to Telegram bot | Provisional: invite code + rate limit + demo-only banner + `/forget` |
| Keep thin web chat? | Provisional: yes (working URL required; Inspector needs a screen) |
| STT hosting | Provisional: local `faster-whisper`; hosted API as fallback; check Railway RAM in Phase 1 |
| Embeddings | Provisional: Nebius; verify model ID and dimension in Phase 1; local MiniLM fallback |
| Raise weekly hours or adopt Reduced MVP | **Needs decision this week** |

---

## 15. Risks and Mitigations

| Risk | Impact | Mitigation / Fallback |
|---|---|---|
| Nemotron on Nebius returns `reasoning_content` with empty `content` and may not support standard tool calls | Breaks retrieval tool-calling | Test in Phase 1.3; parse `reasoning_content` or use a non-reasoning model for the tool-call step (e.g. DeepSeek-V3.2, Llama-3.3-70B-Instruct, Hermes-4-70B); fall back to JSON-instruction retrieval |
| ~14 days left vs ~35–40 hrs planned | Biggest overall risk | Raise hours or adopt the cut line / Reduced MVP; follow phase order |
| Scope creep from the long ideal-feature list | High | §3.2 tiers are the contract; nothing outside MVP until Phase 6 is verified |
| Embedding endpoint/dimension mismatch | Schema rework | Record real dimension in Phase 1.4; config-driven; `halfvec`/no index; local MiniLM fallback |
| Voice quality, latency or RAM on Railway | Bad memories, deploy failure | "Heard: …" line; confirm-before-save; measure in Phase 1.6; use `base` model; hosted STT fallback |
| Telegram issues (409 double-poller, rate/file limits) | Bot stops or double-replies | Separate dev/deployed tokens; single worker; short voice notes; friendly errors |
| Public bot misuse or surprise cost | Credit burn, privacy | Invite codes, rate limit, concurrency cap, banner, `/forget`; revoke codes if needed |
| Prompt injection via pasted exports/transcripts | Behavior steering | Delimited data, ignore-instructions rule, strict schema validation, batch confirm |
| Malformed JSON from LLM | Silent pipeline failure | Strict prompts, Pydantic validation, retry once, log and skip |
| Model deprecation on Nebius | Broken hardcoded IDs | IDs in config; check deprecation notices before submission |
| Postgres + pgvector setup overhead | Lost time | Get running in Phase 0; SQLite + `sqlite-vec` fallback if blocked with <5 days left |
| Live demo failure | Lost score | Rehearse against deployed URL and bot; recorded video as narrative fallback |

---

## 16. Out of Scope / Future (V1+)

Memory Graph (pair with Neo4j if revived), learning features, calendar read and meeting briefs (requires Google OAuth), screenshot filer (requires a Nebius vision model), research collector, content pipeline, README/pitch generators, relationship tracker, weekly check-ins, A/B baseline evaluation harness, full 5-trigger proactivity, Tavily / Builders & Brews tracks.
