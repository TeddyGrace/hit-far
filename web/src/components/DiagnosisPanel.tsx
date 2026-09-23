import { ReactNode, useEffect, useMemo, useState } from "react";
import { api, Diagnosis, EventType, FaultInfo, Labeler, ProposedFault, Verdict } from "../api";

interface Props {
  swingId: string;
  eventFrames: Partial<Record<EventType, number>>;
  refreshKey: string; // changes when events/metrics change -> refetch (stale flags)
  onSeek: (frame: number) => void;
}

const VERDICTS: { value: Verdict; label: string }[] = [
  { value: "confirmed", label: "Confirm" },
  { value: "rejected", label: "Reject" },
  { value: "unsure", label: "Unsure" },
];

const FRAME_REF = /\[frame (\d+)\]/g;

/** Render text with [frame N] turned into seek links. */
function withFrameLinks(text: string, onSeek: (f: number) => void, bad: number[] = []): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(FRAME_REF)) {
    const f = Number(m[1]);
    out.push(text.slice(last, m.index));
    out.push(
      bad.includes(f) ? (
        <span key={m.index} className="badge bad-badge" title="Frame is outside this clip">
          frame {f}?
        </span>
      ) : (
        <button key={m.index} className="link small" onClick={() => onSeek(f)}>
          frame {f}
        </button>
      ),
    );
    last = (m.index ?? 0) + m[0].length;
  }
  out.push(text.slice(last));
  return out;
}

export default function DiagnosisPanel({ swingId, eventFrames, refreshKey, onSeek }: Props) {
  const [catalog, setCatalog] = useState<FaultInfo[]>([]);
  const [history, setHistory] = useState<Diagnosis[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [symptom, setSymptom] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [labeler, setLabeler] = useState<Labeler>("self");
  const [addFault, setAddFault] = useState("");

  useEffect(() => {
    api.listFaults().then(setCatalog, () => {});
  }, []);

  useEffect(() => {
    api.listDiagnoses(swingId).then(setHistory, (e) => setError(e.message));
  }, [swingId, refreshKey]);

  const titles = useMemo(() => Object.fromEntries(catalog.map((f) => [f.name, f])), [catalog]);
  const dx = history.find((d) => d.id === selected) ?? history[0] ?? null;

  const replace = (d: Diagnosis) => setHistory((h) => h.map((x) => (x.id === d.id ? d : x)));

  async function run() {
    setRunning(true);
    setError(null);
    try {
      const d = await api.diagnose(swingId, symptom);
      setHistory((h) => [d, ...h]);
      setSelected(d.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  }

  async function verdict(fault: string, v: Verdict | null) {
    if (!dx) return;
    try {
      replace(await api.setVerdict(dx.id, fault, v, labeler));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const myVerdicts = dx?.verdicts[labeler] ?? {};
  const proposed = new Set(dx?.output?.faults.map((f) => f.fault) ?? []);
  const extraVerdicts = Object.keys(myVerdicts).filter((f) => !proposed.has(f));

  const verdictButtons = (fault: string) => (
    <div className="verdicts">
      {VERDICTS.map((v) => (
        <button
          key={v.value}
          className={`small ${myVerdicts[fault] === v.value ? `on ${v.value}` : ""}`}
          onClick={() => verdict(fault, myVerdicts[fault] === v.value ? null : v.value)}
        >
          {v.label}
        </button>
      ))}
    </div>
  );

  const faultCard = (f: ProposedFault) => (
    <div key={f.fault} className="fault-card">
      <div className="fault-head">
        <strong title={titles[f.fault]?.description}>{titles[f.fault]?.title ?? f.fault}</strong>
        <div className="likelihood" title={`likelihood ${(f.likelihood * 100).toFixed(0)}%`}>
          <div style={{ width: `${f.likelihood * 100}%` }} />
        </div>
        <span className="muted small">{(f.likelihood * 100).toFixed(0)}%</span>
      </div>
      <p className="small">{withFrameLinks(f.explanation, onSeek)}</p>
      {f.visual_observation && (
        <p className="small muted">
          <em>Seen in frames:</em> {withFrameLinks(f.visual_observation, onSeek)}
        </p>
      )}
      <div className="chips">
        {f.evidence.map((e, i) => {
          const frame = e.event ? eventFrames[e.event as EventType] : undefined;
          const bad = f.unverified.some((u) => u.startsWith(`${e.metric}@${e.event ?? "swing"}`));
          return (
            <button
              key={i}
              className={`chip ${bad ? "bad" : ""}`}
              title={e.note}
              onClick={() => frame != null && onSeek(frame)}
            >
              {e.metric}
              {e.event ? `@${e.event}` : ""} = {e.value}
              {bad && " (unverified)"}
            </button>
          );
        })}
        {f.frames.map((fr) => (
          <button key={`f${fr}`} className="chip" onClick={() => onSeek(fr)}>
            frame {fr}
          </button>
        ))}
      </div>
      {f.unverified.length > 0 && (
        <p className="small error">Unverified citations: {f.unverified.join("; ")}</p>
      )}
      {verdictButtons(f.fault)}
    </div>
  );

  return (
    <section className="diagnosis">
      <h2>Diagnosis</h2>
      <textarea
        rows={2}
        placeholder="What's happening? e.g. slicing with the driver, thin contact, losing balance…"
        value={symptom}
        onChange={(e) => setSymptom(e.target.value)}
        maxLength={2000}
      />
      <div className="controls">
        <button type="submit" onClick={run} disabled={running}>
          {running ? "Analyzing… (up to a minute)" : "Diagnose"}
        </button>
        <label className="inline small">
          Labeling as
          <select value={labeler} onChange={(e) => setLabeler(e.target.value as Labeler)}>
            <option value="self">me</option>
            <option value="instructor">instructor</option>
          </select>
        </label>
        {history.length > 1 && (
          <select value={dx?.id} onChange={(e) => setSelected(e.target.value)} title="Previous diagnoses">
            {history.map((d) => (
              <option key={d.id} value={d.id}>
                {new Date(d.created_at).toLocaleString()} {d.symptom_text ? `– ${d.symptom_text.slice(0, 30)}` : ""}
              </option>
            ))}
          </select>
        )}
      </div>
      {error && <p className="error">{error}</p>}

      {dx && (
        <div className="dx-body">
          {dx.stale && (
            <p className="chip busy">Events or metrics changed since this diagnosis; run it again to update.</p>
          )}
          {dx.error ? (
            <p className="error">Diagnosis failed: {dx.error}</p>
          ) : dx.output ? (
            <>
              <p>{withFrameLinks(dx.output.summary, onSeek)}</p>
              {dx.output.faults.length === 0 && <p className="muted">No faults proposed.</p>}
              {dx.output.faults.map(faultCard)}

              {extraVerdicts.length > 0 && (
                <div className="fault-card">
                  <strong className="small">Your additions</strong>
                  {extraVerdicts.map((f) => (
                    <div key={f} className="fault-head">
                      <span>{titles[f]?.title ?? f}</span>
                      {verdictButtons(f)}
                    </div>
                  ))}
                </div>
              )}
              <div className="controls">
                <select value={addFault} onChange={(e) => setAddFault(e.target.value)}>
                  <option value="">Add a fault the model missed…</option>
                  {catalog
                    .filter((f) => !proposed.has(f.name) && !myVerdicts[f.name])
                    .map((f) => (
                      <option key={f.name} value={f.name}>
                        {f.title}
                        {f.assessable ? "" : " (not measurable yet)"}
                      </option>
                    ))}
                </select>
                <button
                  className="small"
                  disabled={!addFault}
                  onClick={() => {
                    verdict(addFault, "confirmed");
                    setAddFault("");
                  }}
                >
                  Add as confirmed
                </button>
              </div>

              {dx.output.cannot_assess.length > 0 && (
                <>
                  <h3>Can't assess from this video</h3>
                  <ul className="small">
                    {dx.output.cannot_assess.map((c, i) => (
                      <li key={i}>
                        <strong>{c.topic}:</strong> {c.reason}
                      </li>
                    ))}
                  </ul>
                </>
              )}
              {dx.output.suggested_checks.length > 0 && (
                <>
                  <h3>Suggested checks</h3>
                  <ul className="small">
                    {dx.output.suggested_checks.map((c, i) => (
                      <li key={i}>{c}</li>
                    ))}
                  </ul>
                </>
              )}
              <details>
                <summary className="small">Full explanation</summary>
                {dx.output.narrative.split(/\n{2,}/).map((para, i) => (
                  <p key={i} className="small">
                    {withFrameLinks(para, onSeek, dx.output!.narrative_unverified_frames)}
                  </p>
                ))}
              </details>
            </>
          ) : null}
          <p className="muted small">
            {dx.served_model ?? dx.model?.name} · proposals only; your verdicts become training data
          </p>
        </div>
      )}
    </section>
  );
}
