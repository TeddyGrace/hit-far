# hit-far: notes for Claude

A personal golf swing app for one golfer. Its purpose is to **replace an instructor with models
trained on the golfer's own swings, not with an LLM**. The golfer tags shot outcomes with one tap
per swing. Outcome models learn which measured positions predict bad shots. Training, retraining
and promotion run automatically. The LLM is optional and only rephrases model output.
The golfer's main miss is a **hook / duck-hook**, and they suspect lead-wrist twisting.

## Club tracking plan (agreed with the user)

- **Stage 1: BUILT** (pipeline v0.4.0, tracker `shaft-line-tracker` 0.1.0). A label-free shaft-line tracker in
  `backend/app/pipeline/club.py`:
  - The shaft is found as the strongest thin straight ridge leaving the hands; the pose model
    provides the grip point.
  - Tracking is smoothed over time with dynamic programming and gives a confidence per frame.
  - Output: `club_tracks` rows plus an `.npz` artifact.
  - The user corrects the shaft on the swing page, stored as `labels` rows with `task=club`.
  - Club metrics are 2D and emitted only on confidently tracked frames: shaft lean, past-parallel,
    wrist hinge, lag, release speed.
  - Existing swings are tracked automatically by `track_club` jobs queued at API start-up.
  - On synthetic video (clutter, motion blur, 120–240 fps) it is within about 3–7° on confident
    frames. At 30 fps it reports low confidence at impact instead of guessing.
  - It is not yet validated on real footage. Check it on the user's first range session: the
    shaft line on the swing page, and the confidence it reports.
- **Stage 2: pending. Do next, once the user has uploaded a few range sessions.** Train a learned
  shaft/clubhead detector (keypoints: grip end and clubhead) from the stage-1 tracker's
  high-confidence frames (pseudo-labels) plus the user's `task=club` corrections. Register it in
  `models` and auto-promote it when it beats the line tracker on held-out corrected frames.
- **Stage 3: pending.** Clubhead and face orientation. A shaft *line* seen face-on cannot show the
  face rotating about the shaft axis, and that rotation is the true signal for the golfer's
  hook/duck-hook. It needs clubhead detection, ideally from a down-the-line camera or at a high
  frame rate. Until then, the hook model relies on release speed, shaft lean and hand-point
  forearm roll as proxies.

## Conventions
- The default branch is `main`, and Railway auto-deploys from it (project "beautiful-cooperation":
  API, worker and trainer services, Postgres, and the swing-media bucket). The user wants changes
  on `main`.
- Metrics are deterministic functions in `backend/app/pipeline/metrics.py`. Bump `PIPELINE_VERSION`
  on any formula change. API start-up then queues a recompute and an outcome retrain.
- Flag 3D, monocular or hand-point metrics with `is_estimate=True`. Emit nothing when the landmarks
  aren't visible, rather than guessing.
- Tests: `cd backend && python -m pytest -q`, which needs a Postgres `hitfar_test` database. Web:
  `cd web && npm run typecheck && npm run build`.
- Users: username + scrypt password,
  managed with `python -m app.users`. Sessions belong to a user; look rows up through the
  `get_*_or_404(db, id, user)` helpers so other users get a 404. Outcome models and their datasets
  carry `user_id` and are per golfer; shared models (pose, events, club) have `user_id` null.
- Keep secrets (for example `ANTHROPIC_API_KEY`) out of the repo. Don't put model identifiers in
  commits or repo text.
