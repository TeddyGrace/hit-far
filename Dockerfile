# One image for both Railway services: the API (default) and the worker (SERVICE_ROLE=worker).

FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# ffmpeg: playback proxies. libegl/libgles/libgl: MediaPipe's native library loads them even on CPU.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libegl1 libgles2 libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Bake the baseline pose model into the image so workers don't download it at runtime.
ARG POSE_MODEL_URL=https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
RUN mkdir -p /app/models && curl -fsSL -o /app/models/pose_landmarker_heavy.task "$POSE_MODEL_URL"
ENV POSE_MODEL_PATH=/app/models/pose_landmarker_heavy.task \
    FFMPEG_BIN=ffmpeg \
    WEB_DIST_DIR=/app/web/dist

COPY backend/ .
COPY --from=web /web/dist /app/web/dist

EXPOSE 8000
CMD ["sh", "/app/backend/start.sh"]
