#!/bin/sh
# Container entrypoint for the API service. No secrets are read or written here.
set -e

cd /app

# Apply migrations first; if this fails the script exits non-zero (set -e) and the
# container does not start. Alembic reads DATABASE_URL from the environment.
alembic upgrade head

# Exec so uvicorn becomes PID 1's replacement and receives signals.
exec uvicorn backend.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
