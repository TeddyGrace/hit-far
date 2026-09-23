#!/bin/sh
# Container entrypoint. SERVICE_ROLE=worker runs the pipeline worker, SERVICE_ROLE=trainer runs the
# model-training worker; anything else runs the API (after applying migrations, so a fresh database
# is ready before the first request).
set -e
if [ "$SERVICE_ROLE" = "worker" ] || [ "$SERVICE_ROLE" = "trainer" ]; then
  exec python -m app.worker "$SERVICE_ROLE"
fi
alembic upgrade head
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips='*'
