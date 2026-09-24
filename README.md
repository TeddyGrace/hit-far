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
   - **Trainer**: `SERVICE_ROLE=trainer`, with the same variables as the worker. It runs only training jobs, and more vCPUs make pose extraction and training faster.
   - `deploy/railway.*.json` are optional config-as-code equivalents.
3. **Set these variables on both services.** Use shared variables or reference variables.

   | Variable | Value |
   |---|---|
   | `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` |
   | `APP_PASSWORD` | the first user's password (see **Users** below) |
   | `OWNER_USERNAME` | optional; the first user's name (default `owner`) |
   | `SECRET_KEY` | a long random string (e.g. `openssl rand -hex 32`) |
   | `COOKIE_SECURE` | `true` |
   | `STORAGE_BACKEND` | `s3` |
   | `S3_ENDPOINT_URL`, `S3_BUCKET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_REGION` | from the bucket's credentials (`BUCKET_*` names are also accepted) |
   | `PUBLIC_ORIGIN` | the API's public URL, e.g. `https://hit-far-production.up.railway.app` |
   | `GOLFER_HANDEDNESS` | `right` or `left` |
   | `ANTHROPIC_API_KEY` | API service only; enables diagnosis |

   If you get signature or host errors, set `S3_ADDRESSING_STYLE=path`.
4. **Browser uploads go straight to the bucket** through presigned URLs, so the bucket needs CORS for your origin. The API sets it at startup whenever `PUBLIC_ORIGIN` is set. Check the API logs for `bucket CORS set for …`.

## Users

A few golfers can share one deployment. Everyone logs in with a username and password. Passwords are stored as salted scrypt hashes. There is no sign-up page: users are managed from the command line on the API service (for example `railway ssh`; the container starts in `/app/backend`):

```sh
python -m app.users list
python -m app.users add alice      # prompts for the password (8+ characters)
python -m app.users passwd alice
python -m app.users delete alice   # refused while alice still owns sessions
```

- **First start:** when no users exist yet, the API creates `OWNER_USERNAME` (default `owner`) with `APP_PASSWORD` and gives it every existing session and outcome model. After that, `APP_PASSWORD` is not used; change the password with `passwd`.
- **What is private:** each user sees only their own sessions, videos, swings, outcome tags, diagnoses and outcome models. Outcome models are trained per golfer, never on pooled swings, because each golfer's faults are their own.
- **What is shared:** the pose, event and club models learn from everyone's swings, so more golfers means more training data for them. Anyone can see and promote the shared models.
- **Handedness** is still one setting (`GOLFER_HANDEDNESS`) for the whole deployment.

## Outcome models: what in your swing predicts your slice?

This is the part that replaces the instructor. You tag each shot's outcome with one tap, and small models trained only on your swings learn which of your measured positions go with your bad shots. No LLM is involved in the finding.

- **Tagging:** outcome chips sit next to every video on the session page and on the swing page: shape (slice / fade / straight / draw / hook, relative to you, so it means the same thing for left-handers), start line, and contact (fat / solid / thin). Tap again to clear. Each change is stored in `shot_outcomes` and appended to `labels` (`task=outcome`). Launch-monitor numbers (club path, face-to-path, etc.) can be stored too; they are not used as model features, because they would give the answer away.
- **Problems:** slice (slice or fade vs. straight, draw or hook), hook, fat and thin, each a separate binary model.
- **Features:** every metric at every event, one row per swing (`backend/app/outcomes/features.py`). Pipeline v0.2.0 adds the slice- and contact-relevant set: shoulders and hips open (signed, relative to address and oriented by your own backswing) at mid-downswing and impact, hip-before-shoulder sequencing, pelvis position in the stance, hands ahead at impact, and transition time. v0.3.0 adds lead-wrist metrics from MediaPipe's coarse hand points: wrist bow/cup at the top and impact, hinge at the top, forearm roll (face-closing rotation of the lead hand about the forearm, relative to the torso and address) at mid-downswing, impact and mid-follow-through, and roll speed at impact. These are the least reliable landmarks (the hands overlap on the grip and blur near impact), so they are 3D estimates emitted only when the hand points are visible; the outcome models decide whether they carry signal.
- **Training (`outcomes/train.py`):**
  - Candidates are an L2 logistic regression and very shallow gradient-boosted trees. The trees win only if they beat the regression by 0.03 AUC.
  - Scored by repeated stratified cross-validation (5-fold × 10), with a bootstrap 95% CI on AUC.
- **Honesty gates:**
  - No training until a problem has at least 20 tagged swings and 6 of each class. The Analysis page shows how many more you need.
  - A model counts as reliable only when the lower bound of its AUC CI is above 0.60 *and* it beats the 95th percentile of the same procedure run on shuffled labels. Otherwise the page says there is no reliable pattern yet.
  - A factor is reported only when the model relies on it (permutation importance across CV folds, more than 2 standard errors) *and* it separates your good and bad shots on its own (Mann-Whitney test, Bonferroni-corrected for the number of metrics).
- **Explanations:** for each factor, your good-shot and bad-shot medians plus a strip chart of every swing, with your latest session marked. There is also a what-if for your latest swing (move one metric to your good-shot median and see how the predicted probability changes), and your best good swing next to a typical bad one, both paused at impact.
- **Automation:**
  - Every 5 outcome changes (`OUTCOME_RETRAIN_EVERY`), the worker retrains every eligible problem, which takes seconds.
  - A new version goes live only if it scores at least as well as the active model's recipe re-scored on the same swings. The comparison is recorded in `eval_metrics.compared_to`.
  - After a metric formula change, API start-up queues a `recompute_metrics` job, then a retrain.
  - Checkpoints go to `models/outcome-<problem>/<version>/model.joblib`, and a `datasets` row stores the swing IDs and the feature snapshot.
- **Optional words:** with `OUTCOME_LLM_EXPLAIN=true`, an "Explain in words" button sends only the model's numbers to Claude to rephrase them. It is off by default.
- **Limits:** the models see your body, not the club face or path, and the findings are correlations in your own data. The suggested fix is to move toward your own good-shot range, which you can test: tag the next sessions and watch the predictions and outcomes.

## Training the event model

The first training run is queued automatically when the API starts, if no trained event model exists yet (`AUTO_START_JOBS`). **Models → Train event model** starts another run. It trains a learned swing-event detector to replace the rule-based one.

- **Where it runs:** on the `hit-far-trainer` Railway service, which handles only training jobs so a run never blocks swing processing.
- **Data:** GolfDB's 1,400 labeled swings (CC BY-NC 4.0, fine for personal use). The labels come from the GolfDB GitHub repo. The 160×160 clips come from the authors' Google Drive link, with `GOLFDB_VIDEOS_URL` as an override.
  - If Drive refuses the download (it rate-limits popular files), put `videos_160.zip` in the bucket at `datasets/golfdb/videos_160.zip` yourself and retry.
- **Your swings (optional):** swings you've marked **"All 8 events checked"** on the swing page are included and weighted up. Every 5th one is held out for scoring.
- **Pose:** MediaPipe runs on every GolfDB clip in parallel across all CPU cores, with each clip's result cached in the bucket (`datasets/golfdb/pose/`). The first run takes a while on CPU; later runs skip it.
- **Model:** a bidirectional LSTM that classifies each frame (8 events plus background) from normalized pose features: hip-centered, torso-scaled joints, their velocities and visibility.
  - Training variation: random time-scaling (to cover any frame rate), mirroring (left-handers), padding with idle frames, and noise.
  - At inference it tries several time scales and keeps the most confident one, so 240 fps slow-motion and 30 fps both work. The 8 events are decoded in order with dynamic programming.
- **Evaluation:** PCE on GolfDB's held-out split 1, using GolfDB's own tolerance (`max(1, round((impact - address) / 30))` frames). It's reported overall and for face-on, next to the rule-based detector on the same swings, plus your held-out swings.
- **Registry:** the checkpoint goes to `models/event-bilstm/<version>/model.pt`, linked to a `datasets` row with its manifest.
- **Auto-promotion:** a new model goes live automatically when it beats the rules on GolfDB's held-out face-on swings (and on your held-out swings, if you have any) and is at least as good as the trained model currently active. Every swing is then re-detected with it. Otherwise it stays `experimental`.
- **Promote** makes a model the active event detector by hand. The previous one is kept (deprecated) and can be promoted back. **Re-run events on all swings** applies the active model to existing swings; your corrections still override it.
- **Safety:** if a trained model fails to load or run, the pipeline falls back to the rule-based detector rather than failing the swing.

Local development needs PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

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
  outcomes/              outcome models: problems, features, train (CV, gates, factors), service, explain
  automation.py          start-up jobs (metric recompute, first event-model training)
  worker.py, jobs.py     Postgres-backed queue
  routers/               sessions, videos, swings (detail, pose frames, event corrections), models
web/src/
  pages/SwingView.tsx    viewer: video + skeleton, timeline, events, outcome, predictions, metrics
  pages/Analysis.tsx     outcome models: data status, reliability, factors, what-if, reference swings
```

## Roadmap

These tables are already in the schema; the code for them is still to come.

1. **Trained event model:** the first GolfDB run is queued automatically on deploy and promotes itself if it beats the rules.
2. **Pose fine-tuning.** Fine-tune the keypoint model with a Label Studio round-trip for occluded and blurred frames (the top of the backswing, impact).
3. **Club tracking.** The golfer's part: tag shot outcomes, and on a few swings per session check the shaft at the top, mid-downswing and impact.
   - **Stage 1 is built:** a label-free shaft-line tracker. The shaft is found as the strongest thin straight line leaving the hands, measured against a median background so static range clutter drops out, and smoothed over time with dynamic programming. It has a per-frame confidence that also falls when the shaft sweeps too far between frames.
     - Metrics come only from confident frames: shaft lean at address and impact, past-parallel at the top, wrist hinge at the top, lag at mid-downswing, and release speed at impact.
     - To correct a frame, click **Fix shaft on this frame**, then click the clubhead. If the line shown is right, click **Shaft looks right**. Both are saved as `labels` rows (`task=club`); a fix also stores the clubhead point you clicked.
     - Record at **240 fps**. At low frame rates the shaft is a blur near impact, and those frames are flagged rather than guessed.
   - **Stage 2 is built, and switches on by itself once there's enough data:** a learned detector (`club-heatmap-cnn`), a small CNN on a crop around the hands that outputs grip and clubhead heatmaps.
     - **Training data:** the line tracker's confident frames (the direction, plus the clubhead where the shaft's end isn't motion-blurred), and your shaft checks, which override the tracker on their frames.
     - **Decoding:** the detector's per-direction scores go through the same temporal decoding as the tracker. Its confidence is the probability it puts on the chosen direction, and it also reports the clubhead position.
     - **Data gate:** it trains on the `hit-far-trainer` service (`train_club` job) once there are at least 8 swings, and at least 20 checked frames on at least 2 held-out swings. Every 3rd swing you've checked is held out. The job is queued automatically when the gate opens, and again after every `CLUB_RETRAIN_EVERY` (default 25) new checks or a large batch of new swings. The Models page shows progress toward the gate and has a manual **Train club detector** button.
     - **Promotion:** it goes live only if, on the held-out swings, it has a lower median error than the tracker on the frames you fixed, at least as many of your checked frames within 10°, agreement with the tracker where the tracker was confident, and no worse results than the detector already active. Then every swing is re-tracked. If the detector fails at runtime, tracking falls back to the line tracker.
   - **Stage 3:** clubhead and face orientation. A shaft line seen face-on can't show the face rotating, which is the real hook signal.
4. **Dual-camera calibration sessions.** Sync two cameras by clap or flash, then triangulate. Use the triangulated poses to fine-tune the monocular lifter and to fill `pose_3d_sequences.error_estimate`.
5. **Reference profiles and comparison.** Compare against archetypes or your own reference swings, with DTW event alignment and per-metric deltas.
6. **Outcome models, next steps:** use launch-monitor face and path numbers as targets when present, and add club and face features once club tracking exists.
