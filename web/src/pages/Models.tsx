import { useCallback, useEffect, useState } from "react";
import { api, EvalResult, ModelInfo, TrainingJob } from "../api";

const pct = (r?: EvalResult) => (r && r.pce != null ? `${(r.pce * 100).toFixed(1)}%` : "—");

function EvalSummary({ m }: { m: ModelInfo }) {
  const em = m.eval_metrics as Record<string, EvalResult & number> | null;
  if (!em) return <span className="muted">—</span>;
  if (m.task.startsWith("outcome_")) {
    const o = m.eval_metrics as { cv_auc?: number; cv_auc_ci?: [number, number]; reliable?: boolean; n?: number;
      kind_title?: string; auto_promoted?: boolean; compared_to?: { version: string; cv_auc_same_data: number } | null };
    return (
      <div className="small">
        <div>
          CV AUC <strong>{o.cv_auc?.toFixed(2)}</strong> (CI {o.cv_auc_ci?.[0]?.toFixed(2)}–{o.cv_auc_ci?.[1]?.toFixed(2)})
          on {o.n} swings · {o.reliable ? "reliable" : "not reliable yet"}
        </div>
        <div className="muted">
          {o.kind_title}
          {o.compared_to &&
            ` · previous recipe scored ${o.compared_to.cv_auc_same_data?.toFixed(2)} on the same swings` +
              (o.auto_promoted ? " → promoted" : " → kept previous")}
        </div>
      </div>
    );
  }
  if (!("golfdb_test" in em)) return <span className="muted small">{JSON.stringify(em)}</span>;
  const test = em.golfdb_test as EvalResult;
  const face = em.golfdb_test_face_on as EvalResult;
  const rules = em.rules_golfdb_test_face_on as EvalResult;
  const self = em.self_holdout as EvalResult;
  const selfRules = em.rules_self_holdout as EvalResult;
  return (
    <div className="small">
      <div>
        GolfDB test PCE <strong>{pct(test)}</strong> (n={test.n}) · face-on <strong>{pct(face)}</strong> vs rules{" "}
        {pct(rules)}
      </div>
      {self && self.n > 0 && (
        <div>
          Your held-out swings <strong>{pct(self)}</strong> vs rules {pct(selfRules)} (n={self.n})
        </div>
      )}
      <div className="muted">
        trained on {String(em.n_train_golfdb)} GolfDB + {String(em.n_train_self)} of your swings · best epoch{" "}
        {String(em.best_epoch)}
      </div>
      {face?.per_event && (
        <details>
          <summary>per event (face-on)</summary>
          {Object.entries(face.per_event).map(([k, v]) => (
            <div key={k}>
              {k}: {(v * 100).toFixed(0)}%{rules?.per_event ? ` (rules ${(rules.per_event[k] * 100).toFixed(0)}%)` : ""}
            </div>
          ))}
        </details>
      )}
    </div>
  );
}

export default function Models() {
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [runs, setRuns] = useState<TrainingJob[]>([]);
  const [reviewed, setReviewed] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const load = useCallback(() => {
    api.listModels().then(setModels, (e) => setError(e.message));
    api.trainingJobs().then(setRuns, () => {});
    api.trainingStatus().then((s) => setReviewed(s.reviewed_swings), () => {});
  }, []);
  useEffect(load, [load]);

  const running = runs.find((r) => r.status === "queued" || r.status === "running");
  useEffect(() => {
    if (!running) return;
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [running, load]);

  async function act(fn: () => Promise<unknown>, msg?: string) {
    setError(null);
    try {
      await fn();
      if (msg) setNote(msg);
      load();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <>
      <h1>Models</h1>
      {error && <p className="error">{error}</p>}
      {note && <p className="chip ok">{note}</p>}

      <section className="card">
        <h2>Event model training</h2>
        <p className="small muted">
          Trains a BiLSTM on GolfDB's 1,400 labelled swings plus any swings you marked "events reviewed" (
          {reviewed ?? "…"} so far; optional). Runs on the <code>hit-far-trainer</code> service; the first run is
          queued automatically on deploy, downloads GolfDB and extracts pose for every clip (cached afterwards), so
          expect it to take a while. A new model goes live automatically when it beats the rules (and the current
          model) on held-out swings, and every swing is re-detected with it.
        </p>
        <div className="controls">
          <button
            type="submit"
            disabled={!!running}
            onClick={() => act(() => api.startEventTraining(), "Training queued")}
          >
            {running ? "Training in progress…" : "Train event model"}
          </button>
          <button onClick={() => act(async () => {
            const r = await api.redetectAll();
            setNote(`Re-running events on ${r.queued} swings with the active model`);
          })}>
            Re-run events on all swings
          </button>
        </div>
        {runs.length > 0 && (
          <table className="table">
            <tbody>
              {runs.slice(0, 5).map((r) => (
                <tr key={r.id}>
                  <td className="small">{new Date(r.updated_at).toLocaleString()}</td>
                  <td>
                    <span className={`chip ${r.status === "done" ? "ok" : r.status === "failed" ? "bad" : "busy"}`}>
                      {r.status}
                    </span>
                  </td>
                  <td className="small">
                    {r.stage}
                    {r.error && (
                      <details className="error-details">
                        <summary className="error">error</summary>
                        <pre>{r.error}</pre>
                      </details>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <p className="muted small">
        Every pose, event and diagnosis output records the model that produced it. Promoting a model makes it active
        for its task; the previous one is kept (deprecated) and can be promoted back.
      </p>
      {models && (
        <table className="table">
          <thead>
            <tr>
              <th>Task</th>
              <th>Name</th>
              <th>Version</th>
              <th>Status</th>
              <th>Evaluation</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.id}>
                <td>{m.task}</td>
                <td>
                  {m.name}
                  {m.notes && <div className="muted small">{m.notes}</div>}
                </td>
                <td className="small">{m.version}</td>
                <td>
                  <span className={`chip ${m.status === "active" ? "ok" : ""}`}>{m.status}</span>
                </td>
                <td>
                  <EvalSummary m={m} />
                </td>
                <td className="actions">
                  {m.status !== "active" && m.task === "event_segmentation" && (
                    <button
                      className="link"
                      onClick={() => {
                        const em = m.eval_metrics as Record<string, EvalResult> | null;
                        const mine = em?.golfdb_test_face_on?.pce;
                        const rules = em?.rules_golfdb_test_face_on?.pce;
                        if (mine != null && rules != null && mine < rules &&
                          !confirm(`This model scores ${(mine * 100).toFixed(1)}% vs the rules' ${(rules * 100).toFixed(1)}% on held-out face-on swings. Promote anyway?`))
                          return;
                        act(() => api.promoteModel(m.id), `${m.name} ${m.version} is now the active event model`);
                      }}
                    >
                      Promote
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}
