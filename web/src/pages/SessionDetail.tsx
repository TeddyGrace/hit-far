import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, CameraRole, SessionDetail as SessionDetailT, uploadFile, VideoWithStatus } from "../api";

interface PendingUpload {
  key: string;
  name: string;
  progress: number;
  error?: string;
}

function StatusChip({ v }: { v: VideoWithStatus }) {
  const job = v.job;
  if (v.status === "failed") return <span className="chip bad">failed</span>;
  if (job && (job.status === "queued" || job.status === "running")) {
    const label = job.status === "running" ? job.stage ?? "running" : "queued";
    const retry = job.attempts > 1 ? ` (attempt ${job.attempts}/${job.max_attempts})` : "";
    return <span className="chip busy">{label + retry}</span>;
  }
  if (job?.status === "failed") return <span className="chip bad">pipeline failed</span>;
  if (job?.status === "done") return <span className="chip ok">ready</span>;
  return <span className="chip">{v.status.replace("_", " ")}</span>;
}

export default function SessionDetail() {
  const { id } = useParams<{ id: string }>();
  const [session, setSession] = useState<SessionDetailT | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [role, setRole] = useState<CameraRole>("face_on");
  const [uploads, setUploads] = useState<PendingUpload[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const load = useCallback(() => {
    if (id) api.getSession(id).then(setSession, (e) => setError(e.message));
  }, [id]);

  useEffect(load, [load]);

  // Poll while anything is in flight.
  const busy = session?.videos.some((v) => v.job && (v.job.status === "queued" || v.job.status === "running"));
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [busy, load]);

  async function handleFiles(files: FileList | null) {
    if (!files || !id) return;
    const list = Array.from(files).map((file) => ({ file, key: crypto.randomUUID() }));
    setUploads((u) => [...u, ...list.map(({ file, key }) => ({ key, name: file.name, progress: 0 }))]);
    for (const { file, key } of list) {
      const update = (patch: Partial<PendingUpload>) =>
        setUploads((u) => u.map((p) => (p.key === key ? { ...p, ...patch } : p)));
      try {
        const contentType = file.type || "video/mp4";
        const up = await api.createUpload(id, file.name, contentType, role);
        await uploadFile(up.upload_url, file, up.upload_headers, (progress) => update({ progress }));
        await api.completeUpload(up.video.id);
        setUploads((u) => u.filter((p) => p.key !== key));
        load();
      } catch (e) {
        update({ error: (e as Error).message });
      }
    }
    if (fileInput.current) fileInput.current.value = "";
  }

  async function act(fn: () => Promise<unknown>) {
    try {
      await fn();
      load();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (!session) return error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>;

  return (
    <>
      <p>
        <Link to="/">← Sessions</Link>
      </p>
      <h1>
        {new Date(session.recorded_at).toLocaleDateString()}
        {session.location && <span className="muted"> · {session.location}</span>}
        {session.club_used && <span className="muted"> · {session.club_used}</span>}
      </h1>
      {session.notes && <p>{session.notes}</p>}
      {error && <p className="error">{error}</p>}

      <div className="card row-form">
        <label>
          Camera
          <select value={role} onChange={(e) => setRole(e.target.value as CameraRole)}>
            <option value="face_on">Face-on</option>
            <option value="down_the_line">Down-the-line</option>
            <option value="other">Other</option>
          </select>
        </label>
        <label className="grow">
          Swing videos (one swing per clip; record slow-mo in the phone's camera app)
          <input
            ref={fileInput}
            type="file"
            accept="video/*,.mov,.mp4,.m4v"
            multiple
            onChange={(e) => handleFiles(e.target.files)}
          />
        </label>
      </div>
      {uploads.map((u) => (
        <div key={u.key} className="upload">
          <span>{u.name}</span>
          {u.error ? <span className="error">{u.error}</span> : <progress value={u.progress} max={1} />}
        </div>
      ))}

      {session.videos.length === 0 ? (
        <p className="muted">No videos yet.</p>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>File</th>
              <th>Camera</th>
              <th>Frames</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {session.videos.map((v) => (
              <tr key={v.id}>
                <td>
                  {v.swing_id && v.job?.status === "done" ? (
                    <Link to={`/swings/${v.swing_id}`}>{v.original_filename}</Link>
                  ) : (
                    v.original_filename
                  )}
                  {(v.error || v.job?.error) && v.job?.status !== "done" && (
                    <details className="error-details">
                      <summary>error</summary>
                      <pre>{v.error || v.job?.error}</pre>
                    </details>
                  )}
                </td>
                <td>{v.camera_role.replace(/_/g, " ")}</td>
                <td>
                  {v.num_frames != null
                    ? `${v.num_frames} @ ${v.fps?.toFixed(0)} fps · ${v.width}×${v.height}`
                    : "—"}
                </td>
                <td>
                  <StatusChip v={v} />
                </td>
                <td className="actions">
                  <button className="link" onClick={() => act(() => api.reprocess(v.id))} disabled={!!busy}>
                    Reprocess
                  </button>
                  <button
                    className="link danger"
                    onClick={() => confirm(`Delete ${v.original_filename}?`) && act(() => api.deleteVideo(v.id))}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <p>
        <button
          className="link danger"
          onClick={async () => {
            if (!confirm("Delete this session and all its videos, swings and outputs?")) return;
            await api.deleteSession(session.id);
            navigate("/");
          }}
        >
          Delete session
        </button>
      </p>
    </>
  );
}
