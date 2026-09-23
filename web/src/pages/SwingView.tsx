import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, EVENT_LABELS, EVENT_TYPES, EventType, PoseFrames, SwingDetail } from "../api";
import DiagnosisPanel from "../components/DiagnosisPanel";
import EventTimeline, { REVIEW_THRESHOLD } from "../components/EventTimeline";
import MetricsTable from "../components/MetricsTable";
import VideoOverlay from "../components/VideoOverlay";

const RATES = [1, 0.5, 0.25, 0.1];

export default function SwingView() {
  const { id } = useParams<{ id: string }>();
  const [swing, setSwing] = useState<SwingDetail | null>(null);
  const [pose, setPose] = useState<PoseFrames | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);
  const [rate, setRate] = useState(0.25);
  const [playing, setPlaying] = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [saving, setSaving] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  // Presigned URLs change on every response; pin the first one per video so the <video> doesn't reload.
  const [src, setSrc] = useState<{ videoId: string; url: string } | null>(null);

  useEffect(() => {
    if (!id) return;
    api.getSwing(id).then(setSwing, (e) => setError(e.message));
    api.getPose(id).then(setPose, () => setPose(null));
  }, [id]);

  useEffect(() => {
    const url = swing?.video.playback_url;
    if (swing && url && src?.videoId !== swing.video.id) setSrc({ videoId: swing.video.id, url });
  }, [swing, src]);

  const fps = swing?.video.fps ?? 30;
  const numFrames = swing?.video.num_frames ?? 1;

  const seek = useCallback(
    (f: number) => {
      const v = videoRef.current;
      if (!v) return;
      const clamped = Math.max(0, Math.min(numFrames - 1, f));
      v.pause();
      v.currentTime = (clamped + 0.5) / fps; // mid-frame: robust to rounding
      setFrame(clamped);
    },
    [fps, numFrames],
  );

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    v.playbackRate = rate;
    const on = () => setPlaying(true);
    const off = () => setPlaying(false);
    v.addEventListener("play", on);
    v.addEventListener("pause", off);
    return () => {
      v.removeEventListener("play", on);
      v.removeEventListener("pause", off);
    };
  }, [rate, src]);

  const correct = useCallback(
    async (event: EventType, f: number | null) => {
      if (!id) return;
      setSaving(true);
      try {
        setSwing(await api.correctEvent(id, event, f));
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setSaving(false);
      }
    },
    [id],
  );

  const eventFrames = useMemo(() => {
    const m: Partial<Record<EventType, number>> = {};
    swing?.events.forEach((e) => (m[e.event_type] = e.frame_index));
    return m;
  }, [swing]);

  // Keyboard: ←/→ step (shift = 10), space play/pause, 1–8 jump to event, shift+1–8 set event here.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (["INPUT", "SELECT", "TEXTAREA"].includes((e.target as HTMLElement).tagName)) return;
      const v = videoRef.current;
      if (!v) return;
      const step = e.shiftKey ? 10 : 1;
      if (e.key === "ArrowRight" || e.key === ".") {
        e.preventDefault();
        seek(frame + step);
      } else if (e.key === "ArrowLeft" || e.key === ",") {
        e.preventDefault();
        seek(frame - step);
      } else if (e.key === " ") {
        e.preventDefault();
        if (v.paused) v.play();
        else v.pause();
      } else if (/^Digit[1-8]$/.test(e.code)) {
        const et = EVENT_TYPES[Number(e.code.slice(5)) - 1];
        if (e.shiftKey) correct(et, frame);
        else if (eventFrames[et] != null) seek(eventFrames[et]!);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frame, seek, correct, eventFrames]);

  if (!swing) return error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>;

  const address = eventFrames.address;
  const toReview = swing.events.filter((e) => !e.corrected && (e.confidence ?? 0) < REVIEW_THRESHOLD).length;

  return (
    <>
      <p>
        <Link to={`/sessions/${swing.session_id}`}>← Session</Link>
      </p>
      <div className="swing-header">
        <h1>{swing.video.original_filename}</h1>
        <label className="inline">
          <input
            type="checkbox"
            checked={swing.is_reference}
            onChange={async (e) => setSwing(await api.updateSwing(swing.id, { is_reference: e.target.checked }))}
          />
          Reference swing
        </label>
      </div>
      {error && <p className="error">{error}</p>}

      <div className="swing-layout">
        <div className="viewer">
          {src ? (
            <VideoOverlay
              src={src.url}
              fps={fps}
              numFrames={numFrames}
              pose={pose}
              frame={frame}
              showSkeleton={showSkeleton}
              videoRef={videoRef}
              onFrame={setFrame}
            />
          ) : (
            <p className="muted">No playback video.</p>
          )}
          <div className="controls">
            <button onClick={() => seek(frame - 1)} title="Previous frame (←)">
              ◀
            </button>
            <button
              onClick={() => {
                const v = videoRef.current!;
                if (v.paused) v.play();
                else v.pause();
              }}
            >
              {playing ? "Pause" : "Play"}
            </button>
            <button onClick={() => seek(frame + 1)} title="Next frame (→)">
              ▶
            </button>
            <select value={rate} onChange={(e) => setRate(Number(e.target.value))} title="Playback speed">
              {RATES.map((r) => (
                <option key={r} value={r}>
                  {r}×
                </option>
              ))}
            </select>
            <label className="inline">
              <input type="checkbox" checked={showSkeleton} onChange={(e) => setShowSkeleton(e.target.checked)} />
              Skeleton
            </label>
            <span className="frame-info">
              frame {frame} / {numFrames - 1} · {(frame / fps).toFixed(3)} s
              {address != null && ` · ${(((frame - address) / fps) * 1000).toFixed(0)} ms from address`}
            </span>
          </div>
          <EventTimeline
            numFrames={numFrames}
            frame={frame}
            events={swing.events}
            onSeek={seek}
            onCorrect={(et, f) => correct(et, f)}
          />
          <p className="muted small">
            ←/→ step (shift ×10) · space play/pause · 1–8 jump to event · shift+1–8 set event to current frame · drag
            markers to correct
          </p>
        </div>

        <aside className="side">
          <section>
            <h2>
              Events {saving && <span className="muted small">saving…</span>}
              {toReview > 0 && !swing.events_reviewed && <span className="chip busy">{toReview} to review</span>}
            </h2>
            <label className="inline small" title="Reviewed swings become training data for the event model">
              <input
                type="checkbox"
                checked={swing.events_reviewed}
                onChange={async (e) => setSwing(await api.setReviewed(swing.id, e.target.checked))}
              />
              All 8 events checked (use as training data)
            </label>
            <table className="table events">
              <tbody>
                {EVENT_TYPES.map((et, i) => {
                  const ev = swing.events.find((e) => e.event_type === et);
                  const low = ev && !ev.corrected && (ev.confidence ?? 0) < REVIEW_THRESHOLD;
                  return (
                    <tr key={et} className={ev?.frame_index === frame ? "current" : ""}>
                      <td className="muted">{i + 1}</td>
                      <td>
                        <button className="link" disabled={!ev} onClick={() => ev && seek(ev.frame_index)}>
                          {EVENT_LABELS[et]}
                        </button>
                      </td>
                      <td className="num">{ev?.frame_index ?? "—"}</td>
                      <td>
                        {ev?.corrected ? (
                          <span className="chip corrected" title={`model said ${ev.predicted_frame_index}`}>
                            corrected
                          </span>
                        ) : ev?.confidence != null ? (
                          <span className={`chip ${low ? "busy" : ""}`}>{ev.confidence.toFixed(2)}</span>
                        ) : null}
                      </td>
                      <td className="actions">
                        <button className="link small" onClick={() => correct(et, frame)} title="Set to current frame">
                          set here
                        </button>
                        {ev?.corrected && (
                          <button className="link small" onClick={() => correct(et, null)} title="Revert to model">
                            revert
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </section>

          <DiagnosisPanel
            swingId={swing.id}
            eventFrames={eventFrames}
            refreshKey={swing.events.map((e) => e.frame_index).join(",")}
            onSeek={seek}
          />

          <section>
            <h2>Metrics</h2>
            <MetricsTable metrics={swing.metrics} onJump={seek} eventFrames={eventFrames} />
            <p className="muted small">
              pipeline v{swing.pipeline_version}
              {swing.pose &&
                ` · pose: ${swing.pose.model.name} (mean visibility ${swing.pose.mean_confidence?.toFixed(2) ?? "—"})`}
              {swing.events[0]?.model && ` · events: ${swing.events[0].model.name} v${swing.events[0].model.version}`}
            </p>
          </section>
        </aside>
      </div>
    </>
  );
}
