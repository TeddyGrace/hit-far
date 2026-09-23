# hit-far: notes for Claude

A personal golf swing app for one golfer. Its purpose is to **replace an instructor with models
trained on the golfer's own swings, not with an LLM**. The golfer tags shot outcomes with one tap
per swing. Outcome models learn which measured positions predict bad shots. Training, retraining
and promotion run automatically. The LLM is optional and only rephrases model output.
The golfer's main miss is a **hook / duck-hook**, and they suspect lead-wrist twisting.

## Pending builds (do these next, in order)

1. **Club/shaft tracking.** The user asked for this to be kept as the immediate next build. It
   replaces the proxy for the clubface closing. Detect the shaft, and later the clubhead, in every
   proxy frame:
   - Bootstrap from a small set of hand-labelled frames, then grow the labels through the existing
     correction loop (`labels` rows).
   - Store the output as `club_tracks` rows plus artifacts. The table already exists in
     `backend/app/models.py`.
   - New metrics: wrist hinge and lag (the angle from lead forearm to shaft), shaft lean at impact,
     and shaft rotation or closing rate through impact. The rotation metric should supersede the
     hand-point `lead_forearm_roll` and `forearm_roll_speed`.
   - Feed the new metrics into the outcome models, especially the hook problem. Compare them with
     the hand-point wrist metrics on the Analysis page.
   - Shaft-parallel events (toe-up, mid-downswing) then stop relying on arm proxies.
2. After that, see the Roadmap in `README.md`.

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
- Keep secrets (for example `ANTHROPIC_API_KEY`) out of the repo. Don't put model identifiers in
  commits or repo text.
