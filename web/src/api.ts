// Typed client for the FastAPI backend. Mirrors backend/app/schemas.py.

export type CameraRole = "face_on" | "down_the_line" | "other";
export type VideoStatus = "pending_upload" | "uploaded" | "preprocessed" | "failed";
export type JobStatus = "queued" | "running" | "done" | "failed";
export const EVENT_TYPES = [
  "address",
  "toe_up",
  "mid_backswing",
  "top",
  "mid_downswing",
  "impact",
  "mid_follow_through",
  "finish",
] as const;
export type EventType = (typeof EVENT_TYPES)[number];

export interface Session {
  id: string;
  recorded_at: string;
  name: string | null;
  location: string | null;
  club_used: string | null;
  notes: string | null;
  created_at: string;
}
export interface SessionSummary extends Session {
  num_videos: number;
  num_swings: number;
}
export interface Job {
  id: string;
  type: string;
  status: JobStatus;
  stage: string | null;
  attempts: number;
  max_attempts: number;
  error: string | null;
  updated_at: string;
}
export interface Video {
  id: string;
  session_id: string;
  uploaded_at: string;
  original_filename: string | null;
  camera_role: CameraRole;
  status: VideoStatus;
  error: string | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  num_frames: number | null;
  duration_s: number | null;
}
export const SHAPES = ["slice", "fade", "straight", "draw", "hook"] as const;
export const START_LINES = ["left", "straight", "right"] as const;
export const CONTACTS = ["fat", "solid", "thin"] as const;
export type Shape = (typeof SHAPES)[number];
export type StartLine = (typeof START_LINES)[number];
export type Contact = (typeof CONTACTS)[number];
export interface ShotOutcome {
  shape: Shape | null;
  start_line: StartLine | null;
  contact: Contact | null;
  source: "self" | "launch_monitor";
  club_path: number | null;
  face_to_path: number | null;
  face_angle: number | null;
  carry: number | null;
  offline: number | null;
  updated_at: string;
}
export type OutcomePatch = Partial<Omit<ShotOutcome, "updated_at">>;

export interface VideoWithStatus extends Video {
  job: Job | null;
  swing_id: string | null;
  outcome: ShotOutcome | null;
}
export interface SessionDetail extends Session {
  videos: VideoWithStatus[];
}
export interface ModelRef {
  id: string;
  name: string;
  version: string;
}
export interface ModelInfo extends ModelRef {
  task: string;
  status: "active" | "experimental" | "deprecated";
  checkpoint_uri: string | null;
  eval_metrics: Record<string, unknown> | null;
  notes: string | null;
  created_at: string;
}
export interface SwingEvent {
  event_type: EventType;
  frame_index: number;
  confidence: number | null;
  predicted_frame_index: number | null;
  corrected: boolean;
  model: ModelRef | null;
}
export interface Metric {
  metric_name: string;
  event_ref: EventType | null;
  value: number;
  unit: string;
  is_estimate: boolean;
}
export interface SwingDetail {
  id: string;
  session_id: string;
  is_reference: boolean;
  club_used: string | null;
  created_at: string;
  video: Video & { playback_url: string | null };
  job: Job | null;
  pose: { model: ModelRef; num_frames: number; mean_confidence: number | null; landmark_schema_version: string } | null;
  events: SwingEvent[];
  events_reviewed: boolean;
  outcome: ShotOutcome | null;
  metrics: Metric[];
  pipeline_version: string;
}
export interface PoseFrames {
  fps: number;
  width: number;
  height: number;
  connections: [number, number][];
  frames: (number[] | null)[];
}

export interface ClubFrames {
  fps: number;
  length_px: number;
  angle_deg: (number | null)[];
  confidence: number[];
  grip: ([number, number] | null)[];
  corrected: number[];
}

export type Verdict = "confirmed" | "rejected" | "unsure";
export type Labeler = "self" | "instructor";

export interface FaultInfo {
  name: string;
  title: string;
  description: string;
  assessable: boolean;
  partial: boolean;
  views: string[];
  needs: string[];
  rules: string[];
}
export interface Evidence {
  metric: string;
  event: string | null;
  value: number;
  note: string;
}
export interface ProposedFault {
  fault: string;
  likelihood: number;
  evidence: Evidence[];
  frames: number[];
  explanation: string;
  visual_observation: string | null;
  unverified: string[];
}
export interface DiagnosisOutput {
  summary: string;
  faults: ProposedFault[];
  cannot_assess: { topic: string; reason: string }[];
  suggested_checks: string[];
  narrative: string;
  narrative_unverified_frames: number[];
}
export interface Diagnosis {
  id: string;
  swing_id: string;
  created_at: string;
  symptom_text: string | null;
  error: string | null;
  output: DiagnosisOutput | null;
  rule_hits: { fault: string; metric: string; event: string | null; value: number; threshold: number }[] | null;
  verdicts: Partial<Record<Labeler, Record<string, Verdict>>>;
  model: ModelRef | null;
  served_model: string | null;
  stale: boolean;
}

export interface TrainingJob extends Job {
  payload: { config?: Record<string, unknown> };
}
export interface EvalResult {
  n: number;
  pce?: number;
  per_event?: Record<string, number>;
}

// --- Outcome models ---

export type ProblemKey = "slice" | "hook" | "fat" | "thin";
export interface Need {
  swings: number;
  positive: number;
  negative: number;
}
export interface OutcomeModel {
  id: string;
  version: string;
  created_at: string;
  kind: string;
  kind_title: string;
  n: number;
  cv_auc: number | null;
  cv_auc_ci: [number | null, number | null];
  null_auc_95: number | null;
  reliable: boolean;
  reliable_rule: string;
  candidates: Record<string, { cv_auc: number | null; cv_auc_ci: [number | null, number | null] }>;
  protocol: string;
  pipeline_version: string;
}
export interface ProblemStatus {
  key: ProblemKey;
  title: string;
  description: string;
  n: number;
  n_pos: number;
  n_neg: number;
  eligible: boolean;
  need: Need;
  model: OutcomeModel | null;
}
export interface OutcomesSummary {
  problems: ProblemStatus[];
  tagged: number;
  with_metrics: number;
  changes_since_training: number;
  retrain_every: number;
  training: { status: JobStatus; stage: string | null; error: string | null; updated_at: string } | null;
  llm_enabled: boolean;
  pipeline_version: string;
}
export interface Factor {
  feature: string;
  metric?: string;
  event?: string | null;
  unit?: string;
  is_estimate?: boolean;
  importance: number;
  importance_se: number;
  univariate_auc: number | null;
  univariate_ci: [number | null, number | null];
  direction: "higher" | "lower";
  good_median: number | null;
  bad_median: number | null;
  n: number;
  supported: boolean;
  points?: { swing_id: string; value: number; bad: boolean }[];
  latest_median?: number | null;
}
export interface WhatIf {
  feature: string;
  value: number;
  target: number;
  probability: number;
  probability_if: number;
  delta: number;
}
export interface ImpactRef {
  swing_id: string;
  filename: string | null;
  playback_url: string | null;
  fps: number | null;
  impact_frame: number | null;
  probability: number | null;
  outcome: ShotOutcome | null;
}
export interface OutcomeAnalysis {
  problem: { key: ProblemKey; title: string; description: string };
  n: number;
  n_pos: number;
  n_neg: number;
  eligible: boolean;
  need: Need;
  model: OutcomeModel | null;
  factors: Factor[];
  references: { good: ImpactRef | null; bad: ImpactRef | null } | null;
  latest: { session_id: string | null; n: number } | null;
  what_if?: { swing_id: string; items: WhatIf[] };
}
export interface Prediction {
  problem: ProblemKey;
  title: string;
  probability: number;
  reliable: boolean;
  model_version: string;
  what_if: WhatIf[];
}

export interface Me {
  ok: boolean;
  username: string;
  is_admin: boolean;
}
export interface UserInfo {
  id: string;
  username: string;
  is_admin: boolean;
  created_at: string;
  num_sessions: number;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

let onUnauthorized: () => void = () => {};
export function setUnauthorizedHandler(fn: () => void) {
  onUnauthorized = fn;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    credentials: "same-origin",
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (res.status === 401 && !path.startsWith("/api/auth/login")) onUnauthorized();
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      /* not json */
    }
    throw new ApiError(res.status, msg);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const api = {
  login: (username: string, password: string) =>
    request<Me>("POST", "/api/auth/login", { username, password }),
  logout: () => request<{ ok: boolean }>("POST", "/api/auth/logout"),
  me: () => request<Me>("GET", "/api/auth/me"),

  listUsers: () => request<UserInfo[]>("GET", "/api/users"),
  createUser: (body: { username: string; password: string; is_admin: boolean }) =>
    request<UserInfo>("POST", "/api/users", body),
  updateUser: (id: string, body: { password?: string; is_admin?: boolean }) =>
    request<UserInfo>("PATCH", `/api/users/${id}`, body),
  deleteUser: (id: string) => request<void>("DELETE", `/api/users/${id}`),

  listSessions: () => request<SessionSummary[]>("GET", "/api/sessions"),
  createSession: (body: Partial<Pick<Session, "recorded_at" | "name" | "location" | "club_used" | "notes">>) =>
    request<Session>("POST", "/api/sessions", body),
  getSession: (id: string) => request<SessionDetail>("GET", `/api/sessions/${id}`),
  updateSession: (id: string, body: Partial<Pick<Session, "name" | "location" | "club_used" | "notes">>) =>
    request<Session>("PATCH", `/api/sessions/${id}`, body),
  deleteSession: (id: string) => request<void>("DELETE", `/api/sessions/${id}`),

  createUpload: (sessionId: string, filename: string, contentType: string, cameraRole: CameraRole) =>
    request<{ video: Video; upload_url: string; upload_headers: Record<string, string> }>(
      "POST",
      `/api/sessions/${sessionId}/videos`,
      { filename, content_type: contentType, camera_role: cameraRole },
    ),
  completeUpload: (videoId: string) => request<Job>("POST", `/api/videos/${videoId}/complete`),
  reprocess: (videoId: string) => request<Job>("POST", `/api/videos/${videoId}/reprocess`, { force: true }),
  deleteVideo: (videoId: string) => request<void>("DELETE", `/api/videos/${videoId}`),

  getSwing: (id: string) => request<SwingDetail>("GET", `/api/swings/${id}`),
  updateSwing: (id: string, body: { is_reference?: boolean; club_used?: string | null }) =>
    request<SwingDetail>("PATCH", `/api/swings/${id}`, body),
  getPose: (id: string) => request<PoseFrames>("GET", `/api/swings/${id}/pose`),
  correctEvent: (id: string, event: EventType, frame_index: number | null) =>
    request<SwingDetail>("PUT", `/api/swings/${id}/events/${event}`, { frame_index }),

  getClub: (id: string) => request<ClubFrames>("GET", `/api/swings/${id}/club`),
  correctClub: (id: string, frame: number, angle_deg: number | null) =>
    request<SwingDetail>("PUT", `/api/swings/${id}/club/${frame}`, { angle_deg }),

  listModels: () => request<ModelInfo[]>("GET", "/api/models"),

  startEventTraining: (epochs = 40) => request<TrainingJob>("POST", "/api/training/events", { epochs }),
  trainingJobs: () => request<TrainingJob[]>("GET", "/api/training/jobs"),
  trainingStatus: () => request<{ reviewed_swings: number }>("GET", "/api/training/status"),
  promoteModel: (id: string) => request<ModelInfo[]>("POST", `/api/models/${id}/promote`),
  redetectAll: () => request<{ queued: number }>("POST", "/api/swings/redetect-events"),
  setReviewed: (swingId: string, reviewed: boolean) =>
    request<SwingDetail>("PUT", `/api/swings/${swingId}/review`, { reviewed }),

  setOutcome: (swingId: string, patch: OutcomePatch) =>
    request<ShotOutcome | null>("PUT", `/api/swings/${swingId}/outcome`, patch),
  swingPredictions: (swingId: string) => request<Prediction[]>("GET", `/api/swings/${swingId}/predictions`),
  outcomesSummary: () => request<OutcomesSummary>("GET", "/api/outcomes"),
  outcomeAnalysis: (problem: ProblemKey) => request<OutcomeAnalysis>("GET", `/api/outcomes/${problem}`),
  trainOutcomes: () => request<{ queued: boolean }>("POST", "/api/outcomes/train"),
  explainOutcome: (problem: ProblemKey) =>
    request<{ explanation: string; served_model: string }>("POST", `/api/outcomes/${problem}/explain`),

  listFaults: () => request<FaultInfo[]>("GET", "/api/faults"),
  listDiagnoses: (swingId: string) => request<Diagnosis[]>("GET", `/api/swings/${swingId}/diagnoses`),
  diagnose: (swingId: string, symptom_text: string) =>
    request<Diagnosis>("POST", `/api/swings/${swingId}/diagnoses`, { symptom_text }),
  setVerdict: (diagnosisId: string, fault: string, verdict: Verdict | null, labeled_by: Labeler = "self") =>
    request<Diagnosis>("PUT", `/api/diagnoses/${diagnosisId}/faults/${fault}`, { verdict, labeled_by }),
};

/** PUT a file to a presigned URL with progress (fetch has no upload progress). */
export function uploadFile(
  url: string,
  file: File,
  headers: Record<string, string>,
  onProgress: (fraction: number) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    for (const [k, v] of Object.entries(headers)) xhr.setRequestHeader(k, v);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () =>
      xhr.status >= 200 && xhr.status < 300 ? resolve() : reject(new Error(`upload failed: HTTP ${xhr.status}`));
    xhr.onerror = () => reject(new Error("upload failed (network or CORS)"));
    xhr.send(file);
  });
}

export const EVENT_LABELS: Record<EventType, string> = {
  address: "Address",
  toe_up: "Toe up",
  mid_backswing: "Mid backswing",
  top: "Top",
  mid_downswing: "Mid downswing",
  impact: "Impact",
  mid_follow_through: "Mid follow-through",
  finish: "Finish",
};
