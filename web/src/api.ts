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
export interface VideoWithStatus extends Video {
  job: Job | null;
  swing_id: string | null;
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
  login: (password: string) => request<{ ok: boolean }>("POST", "/api/auth/login", { password }),
  logout: () => request<{ ok: boolean }>("POST", "/api/auth/logout"),
  me: () => request<{ ok: boolean }>("GET", "/api/auth/me"),

  listSessions: () => request<SessionSummary[]>("GET", "/api/sessions"),
  createSession: (body: Partial<Pick<Session, "recorded_at" | "location" | "club_used" | "notes">>) =>
    request<Session>("POST", "/api/sessions", body),
  getSession: (id: string) => request<SessionDetail>("GET", `/api/sessions/${id}`),
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

  listModels: () => request<ModelInfo[]>("GET", "/api/models"),

  startEventTraining: (epochs = 40) => request<TrainingJob>("POST", "/api/training/events", { epochs }),
  trainingJobs: () => request<TrainingJob[]>("GET", "/api/training/jobs"),
  trainingStatus: () => request<{ reviewed_swings: number }>("GET", "/api/training/status"),
  promoteModel: (id: string) => request<ModelInfo[]>("POST", `/api/models/${id}/promote`),
  redetectAll: () => request<{ queued: number }>("POST", "/api/swings/redetect-events"),
  setReviewed: (swingId: string, reviewed: boolean) =>
    request<SwingDetail>("PUT", `/api/swings/${swingId}/review`, { reviewed }),

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
