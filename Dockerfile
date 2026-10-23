# Skeleton cloud image for the API service (Railway). Build-only concerns; no secrets.
# Do NOT add ARGs or ENV that carry secret values: .env is never copied.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Dependencies first (installed from the exact lock file).
COPY backend/requirements.lock ./backend/requirements.lock
RUN pip install --no-cache-dir -r backend/requirements.lock

# 2) Bake the Whisper models in their own layer, BEFORE app code, so code changes
#    do not re-download them. snapshot_download(..., local_dir=...) writes plain
#    folders (no symlinks); the guard below fails the build if any symlink appears.
#    Only base + tiny: single-process mode must fit the container's memory budget.
ENV WHISPER_MODEL_NAME=base
ENV WHISPER_MODEL_ROOT=/app/whisper_models
RUN python -c "from huggingface_hub import snapshot_download as d; d(repo_id='Systran/faster-whisper-base', local_dir='/app/whisper_models/whisper-base'); d(repo_id='Systran/faster-whisper-tiny', local_dir='/app/whisper_models/whisper-tiny')"
RUN python -c "import os,sys; bad=[os.path.join(r,f) for r,_,fs in os.walk('/app/whisper_models') for f in fs if os.path.islink(os.path.join(r,f))]; sys.exit('symlinks found: '+repr(bad[:3])) if bad else None"

# 3) Application code + Alembic config only.
COPY alembic.ini ./alembic.ini
COPY backend/ ./backend/

# 4) Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# 5) Start: run migrations, then exec uvicorn (no --reload).
CMD ["sh", "backend/deploy/start_api.sh"]
