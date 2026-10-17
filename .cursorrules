Read docs/PRD.md and docs/build-roadmap.md before any task.
- Follow the phase order in roadmap §10. Only build the phase I name.
- Model IDs, embedding model ID and embedding dimension live in config/.env, never hardcoded.
- All LLM JSON output is validated with Pydantic; retry once, then log and skip.
- Always check that `content` is non-empty and handle `reasoning_content` (roadmap §16 Risk 1).
- Retrieval only searches status=confirmed and non-expired records.
- No secrets in committed files or the frontend bundle.
- Do not build anything in the Stretch/Future tiers (roadmap §9.1).
- Never report "expected output" as if it were a result. Only report what you actually ran.
- After each task, tell me exactly how to run it (PowerShell syntax) and what output proves it works.

