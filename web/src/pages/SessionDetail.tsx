import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, CameraRole, SessionDetail as SessionDetailT, uploadFile, VideoWithStatus } from "../api";
import OutcomeChips from "../components/OutcomeChips";
import { useUi } from "../components/ui";

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
  const ui = useUi();
  const { id } = useParams<{ id: string }>();
  const [session, setSession] = useState<SessionDetailT | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [role, setRole] = useState<CameraRole>("face_on");
  const [uploads, setUploads] = useState<PendingUpload[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [location, setLocation] = useState("");
  const [club, setClub] = useState("");
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);

  const load = useCallback(() => {
    if (id) api.getSession(id).then(setSession, (e) => setError(e.message));
  }, [id]);

  useEffect(load, [load]);

  // The page polls while jobs run. Only overwrite a field if the user hasn't edited it.
  const prevSession = useRef<SessionDetailT | null>(null);
  useEffect(() => {
    if (!session) return;
    const prev = prevSession.current;
    const fresh = !prev || prev.id !== session.id;
    const sync = (cur: string, was: string | null | undefined, next: string | null | undefined) =>
      fresh || cur === (was ?? "") ? (next ?? "") : cur;
    setName((c) => sync(c, prev?.name, session.name));
    setLocation((c) => sync(c, prev?.location, session.location));
    setClub((c) => sync(c, prev?.club_used, session.club_used));
    setNotes((c) => sync(c, prev?.notes, session.notes));
    prevSession.current = session;
  }, [session]);

  const dirty =
    !!session &&
    (name !== (session.name ?? "") ||
      location !== (session.location ?? "") ||
      club !== (session.club_used ?? "") ||
      notes !== (session.notes ?? ""));

  async function saveDetails() {
    if (!session) return;
    setSaving(true);
    try {
      const updated = await api.updateSession(session.id, {
        name: name || null,
        location: location || null,
        club_used: club || null,
        notes: notes || null,
      });
      setSession({ ...session, ...updated });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

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
        {session.name || new Date(session.recorded_at).toLocaleDateString()}
        {session.name && <span className="muted"> · {new Date(session.recorded_at).toLocaleDateString()}</span>}
      </h1>
      {error && <p className="error">{error}</p>}

      <div className="card row-form">
        <label>
          Name
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Tuesday range" />
        </label>
        <label>
          Location
          <input value={location} onChange={(e) => setLocation(e.target.value)} placeholder="Range, sim, course…" />
        </label>
        <label>
          Club
          <input value={club} onChange={(e) => setClub(e.target.value)} placeholder="7i" />
        </label>
        <label className="grow">
          Notes
          <input value={notes} onChange={(e) => setNotes(e.target.value)} />
        </label>
        <button onClick={saveDetails} disabled={!dirty || saving}>
          {saving ? "Saving…" : "Save"}
        </button>
      </div>

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

      {session.videos.length > 0 && (
        <p className="muted small">
          Tag each shot's outcome (tap again to clear). The outcome models on the Analysis page retrain automatically as
          you tag.
        </p>
      )}
      {session.videos.length === 0 ? (
        <p className="muted">No videos yet.</p>
      ) : (
        <div className="table-wrap"><table className="table">
          <thead>
            <tr>
              <th>File</th>
              <th>Camera</th>
              <th>Frames</th>
              <th>Status</th>
              <th>Shot outcome</th>
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
                <td>
                  {v.swing_id && <OutcomeChips swingId={v.swing_id} outcome={v.outcome} />}
                </td>
                <td className="actions">
                  <button className="link" onClick={() => act(() => api.reprocess(v.id))} disabled={!!busy}>
                    Reprocess
                  </button>
                  <button
                    className="link danger"
                    onClick={async () => {
                      if (await ui.confirm({ title: `Delete ${v.original_filename ?? "this video"}?`, confirmLabel: "Delete", danger: true }))
                        act(() => api.deleteVideo(v.id));
                    }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table></div>
      )}

      <p>
        <button
          className="link danger"
          onClick={async () => {
            const ok = await ui.confirm({
              title: "Delete this session?",
              body: "This removes all its videos, swings and outputs.",
              confirmLabel: "Delete session",
              danger: true,
            });
            if (!ok) return;
            try {
              await api.deleteSession(session.id);
              navigate("/");
            } catch (e) {
              ui.toast((e as Error).message, "error");
            }
          }}
        >
          Delete session
        </button>
      </p>
    </>
  );
}
