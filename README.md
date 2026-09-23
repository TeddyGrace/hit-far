# hit-far

A personal golf swing analysis app. You upload a swing video, and it runs pose estimation, finds the swing events and computes deterministic biomechanics metrics. You can then review everything in a simple React viewer and correct it by hand.

This is **v0, a vertical slice**. The whole data model from the project brief exists as tables. The perception, event and metrics pipeline runs end to end on baseline models, and every output is attributed to the model that produced it. Trained models, club tracking, 3D calibration, comparison and diagnosis build on top of this (see [Roadmap](#roadmap)).

## How it works

```
browser ──presigned PUT──▶ bucket (raw video)
   │                          │
   └─ POST /complete ─▶ jobs table ─▶ worker
                                        1. preprocess: sha256 (dedupe) → H.264 CFR proxy (rotation applied,
                                           native fps kept, ≤1280px) → probe
                                        2. swing: one swing per video (v0)
                                        3. pose: MediaPipe Pose Landmarker heavy → 2D keypoints + world
                                           landmarks (.npz artifact; pose_sequences + pose_3d_sequences rows)
                                        4. events: rule-based detector over the pose time series → swing_events
                                        5. metrics: deterministic geometry → metrics (pipeline_version)
viewer ◀── presigned GET proxy + /api/swings/:id/pose (skeleton overlay)
corrections ─▶ labels (task=event) ─▶ metrics recomputed from corrected events
```

- **Frame indices are proxy frames.** Pose, events, labels and the viewer all index the same constant-frame-rate proxy, so they always agree. Slow-motion clips keep every frame (e.g. 240 fps), and playback speed is a viewer control.
- **Model registry.** `mediapipe-pose-landmarker-heavy` and `rule-events` are registered in `models` the first time they run. Trained successors get new rows. Outputs from older models stay queryable, and artifacts are never overwritten.
- **Metrics are never a model.** Metrics are pure functions in `backend/app/pipeline/metrics.py`, versioned by `PIPELINE_VERSION`.
  - 2D image-plane metrics (tilts, sway, tempo) are measurements of the video.
  - Metrics from MediaPipe world landmarks (turns, X-factor, elbow/knee angles) are flagged `is_estimate` and shown with a **3D estimate** badge. Monocular depth is a learned guess with no error bound until triangulated dual-camera sessions exist.
- **Event review.**
  - Events below 0.6 confidence are flagged "to review".
  - You can drag a timeline marker, or press `shift+1…8`, to set an event to the current frame.
  - Each correction is a `labels` row (`task=event`, `target_type=swing`) that records the model's prediction next to your value. That makes it training data for the future event model.
- **Queue.** The job queue is a Postgres table consumed with `FOR UPDATE SKIP LOCKED`, so there's no Redis.
  - Transient failures retry with backoff.
  - Bad input (for example a duplicate file, or no person detected) fails permanently with a readable error.
  - Jobs held by a crashed worker are reclaimed after 30 min.

### Rule-based events (v0), what to expect

The detector is designed for **face-on** video of a single swing per clip:

- **Impact:** found from the peak hand speed.
- **Top:** the highest hand position before impact.
- **Address and finish:** the still periods on either side.
- **Toe-up, mid-backswing, mid-downswing and mid-follow-through:** arm-angle crossings. The club isn't tracked yet, so these are lower-confidence proxies.

A clip with no real swing produces uniformly low confidence. Trim long idle footage before uploading for best results; multi-swing clipping is on the roadmap.

## Local development

Requirements:
- Python 3.11+
- Node 20+
- A local Postgres

```bash
# backend
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # STORAGE_BACKEND=local keeps files in backend/.data/
createdb hitfar && alembic upgrade head
uvicorn app.main:app --reload --port 8000    # API
python -m app.worker                          # worker, in a second terminal (downloads the pose model on first run)

# web (third terminal): Vite on :5173, proxying /api to :8000
cd web && npm install && npm run dev
```

On Linux, MediaPipe needs `libegl1 libgles2 libgl1` installed.

### Tests

```bash
cd backend
createdb hitfar_test
pytest                            # events + metrics on synthetic swings, API/pipeline integration tests
cd ../web && npm run typecheck && npm run build
```

The integration tests run the real upload → worker → ffmpeg proxy → events → metrics → correction flow. They use local storage and a synthetic stick-figure swing in place of MediaPipe, because MediaPipe needs a real person to detect.

## Deploying on Railway

The whole project is **one Docker image, run as two services**: the API, which also serves the built React app, and the worker. Alongside them you need a Postgres database and a storage bucket.

1. **Create a project** and add:
   - a **PostgreSQL** database
   - a **Bucket** (Railway Storage Bucket)
2. **Create two services from this repo** (branch `main`). Both use the root `Dockerfile`, and `backend/start.sh` picks the role:
   - **API**: no `SERVICE_ROLE`. It runs `alembic upgrade head`, then serves the app. Generate a public domain for it, and optionally set the healthcheck path to `/api/health`.
   - **Worker**: `SERVICE_ROLE=worker`. It needs no domain. Give it at least 2 GB of RAM. Pose runs on CPU at about 80 ms per frame, so a 3 s clip at 240 fps takes roughly a minute.
   - `deploy/railway.*.json` are optional config-as-code equivalents.
3. **Set these variables on both services.** Use shared variables or reference variables.

   | Variable | Value |
   |---|---|
   | `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` |
   | `APP_PASSWORD` | your login password |
   | `SECRET_KEY` | a long random string (e.g. `openssl rand -hex 32`) |
   | `COOKIE_SECURE` | `true` |
   | `STORAGE_BACKEND` | `s3` |
   | `S3_ENDPOINT_URL`, `S3_BUCKET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_REGION` | from the bucket's credentials (`BUCKET_*` names are also accepted) |
   | `PUBLIC_ORIGIN` | the API's public URL, e.g. `https://hit-far-production.up.railway.app` |
   | `GOLFER_HANDEDNESS` | `right` or `left` |
   | `ANTHROPIC_API_KEY` | API service only; enables diagnosis |

   If you get signature or host errors, set `S3_ADDRESSING_STYLE=path`.
4. **Browser uploads go straight to the bucket** through presigned URLs, so the bucket needs CORS for your origin. The API sets it at startup whenever `PUBLIC_ORIGIN` is set. Check the API logs for `bucket CORS set for …`.

## Diagnosis (Claude proposes, you confirm)

On a swing's page, describe what's happening ("slicing with the driver") and press **Diagnose**.

1. **Rules first.** The deterministic rule engine (`backend/app/diagnosis/rules.py`) checks the swing's metrics against the fault catalog's seed thresholds (`catalog.py`). These are tunable starting points, not ground truth.
2. **Then Claude.** Claude receives the metrics (3D estimates flagged), the events (with confidence and whether you corrected them), the rule hits, four key frames with the skeleton drawn on, and your description. It returns structured proposals: faults with likelihoods, cited metrics and frames, what it saw in the frames, what *can't* be assessed from this video, and suggested checks.
3. **Citations are checked.** Every cited metric value and frame number is compared with the stored data. Mismatches are marked **unverified** in the UI rather than shown as fact.
4. **You decide.** Confirm, reject or mark unsure for each proposal, and add faults the model missed. Each verdict updates `diagnoses.human_confirmed_faults` and appends a `labels` row (`task=fault`), recording whether the fault was proposed and at what likelihood. Only your verdicts become labels; Claude's output never does.
5. **Instructor check.** Set "Labeling as: instructor" to record an instructor's read separately, for the periodic calibration check against your own labels.

Details:
- **Stale diagnoses:** each diagnosis stores a snapshot of exactly what the model saw. If you later correct an event and the metrics change, it's flagged stale.
- **Limits:** clubface angle and swing path aren't measured, so for ball-flight symptoms such as a slice the diagnosis says so and suggests how to find out (a down-the-line clip, or later club tracking).
- **Setup:** set `ANTHROPIC_API_KEY` on the API service. Without it, diagnosing returns a clear 503 and the rest of the app works normally.
- **Optional settings:** `DIAGNOSIS_MODEL` (default `claude-sonnet-5`), `DIAGNOSIS_EFFORT` (default `medium`) and `DIAGNOSIS_TIMEOUT_S`. The default is chosen for efficiency. For the most thorough read, set `DIAGNOSIS_MODEL=claude-opus-5` and `DIAGNOSIS_EFFORT=high`. Opus 5 also gets server-side refusal fallback, so a rare false-positive refusal is rerouted instead of failing.

## Layout

```
backend/app/
  models.py              full schema (sessions, videos, swings, pose/3D/club/event outputs, metrics,
                         reference profiles, comparisons, faults, diagnoses, models, datasets, labels, jobs)
  pipeline/ingest.py     checksum, proxy transcode, probe
  pipeline/pose.py       MediaPipe baseline
  pipeline/events.py     rule-based event detector
  pipeline/metrics.py    deterministic metrics (PIPELINE_VERSION)
  pipeline/run.py        orchestration + effective events (predictions overridden by labels)
  worker.py, jobs.py     Postgres-backed queue
  routers/               sessions, videos, swings (detail, pose frames, event corrections), models
web/src/
  pages/SwingView.tsx    viewer: video + skeleton, timeline, events, metrics
```

## Roadmap

These tables are already in the schema; the code for them is still to come.

1. **Trained event model.** Train a BiLSTM or small transformer on pose sequences: pretrain on GolfDB, then fine-tune on the `labels` corrections. Register it as `experimental`, evaluate it on a held-out split, then promote it.
2. **Pose fine-tuning.** Fine-tune the keypoint model with a Label Studio round-trip for occluded and blurred frames (the top of the backswing, impact).
3. **Club/shaft detector.** Bootstrap it from hand labels and grow it through the correction loop. Shaft-parallel events then stop using arm proxies.
4. **Dual-camera calibration sessions.** Sync two cameras by clap or flash, then triangulate. Use the triangulated poses to fine-tune the monocular lifter and to fill `pose_3d_sequences.error_estimate`.
5. **Reference profiles and comparison.** Compare against archetypes or your own reference swings, with DTW event alignment and per-metric deltas.
6. **Trained fault classifier.** Once enough confirmed verdicts accumulate, train gradient-boosted trees on metrics → confirmed faults to replace the seed rules, then evaluate them against held-out verdicts (and instructor labels).
