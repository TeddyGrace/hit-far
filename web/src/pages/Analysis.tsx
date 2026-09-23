import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  api,
  Factor,
  ImpactRef,
  OutcomeAnalysis,
  OutcomesSummary,
  ProblemKey,
  ProblemStatus,
  WhatIf,
} from "../api";
import { AucInterval, StripChart } from "../components/Charts";
import { ESTIMATE_NOTE, featureLabel, fmtValue } from "../components/MetricsTable";

const GATE = 0.6;
const pct = (p: number) => `${Math.round(p * 100)}%`;
const num = (v: number | null | undefined, d = 2) => (v == null ? "—" : v.toFixed(d));

function Progress({ have, need, label }: { have: number; need: number; label: string }) {
  const total = have + need;
  return (
    <div className="progress-row">
      <span className="small">{label}</span>
      <progress value={have} max={total || 1} />
      <span className="small muted">{need > 0 ? `${have} / ${total}` : `${have} ✓`}</span>
    </div>
  );
}

function DataStatus({ p }: { p: ProblemStatus | OutcomeAnalysis }) {
  const title = "problem" in p ? p.problem.title : p.title;
  return (
    <section className="card">
      <h2>Your data</h2>
      <p className="small">
        {p.n} tagged swings for this question: <strong>{p.n_pos}</strong> {title.toLowerCase()}
        {" · "}
        <strong>{p.n_neg}</strong> good.
      </p>
      {!p.eligible && (
        <>
          <Progress label="Tagged swings" have={p.n} need={p.need.swings} />
          <Progress label={`${title} shots`} have={p.n_pos} need={p.need.positive} />
          <Progress label="Good shots" have={p.n_neg} need={p.need.negative} />
          <p className="small muted">
            The model trains automatically once there are at least 20 tagged swings with 6 of each kind.
          </p>
        </>
      )}
    </section>
  );
}

function Reliability({ a }: { a: OutcomeAnalysis }) {
  const m = a.model;
  if (!m) return null;
  const title = a.problem.title.toLowerCase();
  const others = Object.entries(m.candidates ?? {}).filter(([k]) => k !== m.kind);
  return (
    <section className="card">
      <h2>
        Can it predict your {title}s?{" "}
        <span className={`chip ${m.reliable ? "ok" : "busy"}`}>{m.reliable ? "yes" : "not reliably yet"}</span>
      </h2>
      <AucInterval auc={m.cv_auc} ci={m.cv_auc_ci} null95={m.null_auc_95} gate={GATE} reliable={m.reliable} />
      <p className="small">
        On swings it didn't train on, it ranks a {title} above a good shot{" "}
        <strong>{m.cv_auc != null ? pct(m.cv_auc) : "—"}</strong> of the time (AUC {num(m.cv_auc)}, 95% CI{" "}
        {num(m.cv_auc_ci[0])}–{num(m.cv_auc_ci[1])}). Chance is 50%; with shuffled labels, 95% of runs score below{" "}
        {num(m.null_auc_95)} at this sample size.
      </p>
      <p className="small muted">
        {m.kind_title}, trained on {m.n} swings, version {m.version}.
        {others.map(([k, c]) => ` Also tried ${k}: AUC ${num(c.cv_auc)}.`)} Rule: {m.reliable_rule}.
      </p>
    </section>
  );
}

function direction(f: Factor, title: string): string {
  const more = f.direction === "higher" ? "higher" : "lower";
  return `${title}s come with ${more} values`;
}

function FactorRow({ f, title }: { f: Factor; title: string }) {
  const unit = f.unit ?? "";
  return (
    <div className="factor">
      <div className="factor-head">
        <strong>{featureLabel(f.feature)}</strong>
        {f.is_estimate && <span className="badge" title={ESTIMATE_NOTE}>estimate</span>}
        {!f.supported && <span className="chip">unconfirmed</span>}
      </div>
      <p className="small">
        {direction(f, title)}: your {title.toLowerCase()} median{" "}
        <strong>{f.bad_median != null ? fmtValue(f.bad_median, unit) : "—"}</strong> vs good-shot median{" "}
        <strong>{f.good_median != null ? fmtValue(f.good_median, unit) : "—"}</strong>
        {f.latest_median != null && <> · latest session {fmtValue(f.latest_median, unit)}</>}
      </p>
      {f.points && (
        <StripChart points={f.points} goodMedian={f.good_median} badMedian={f.bad_median} latest={f.latest_median}
          unit={unit} badLabel={title.toLowerCase()} />
      )}
      <p className="muted small">
        Model importance {num(f.importance, 3)} ± {num(f.importance_se, 3)} AUC · alone this metric separates good
        from {title.toLowerCase()} at AUC {num(f.univariate_auc)} (CI {num(f.univariate_ci[0])}–
        {num(f.univariate_ci[1])})
      </p>
    </div>
  );
}

function Factors({ a }: { a: OutcomeAnalysis }) {
  const m = a.model;
  if (!m) return null;
  const title = a.problem.title;
  const supported = a.factors.filter((f) => f.supported);
  const rest = a.factors.filter((f) => !f.supported).slice(0, 5);
  if (!m.reliable) {
    return (
      <section className="card">
        <h2>What goes with your {title.toLowerCase()}s</h2>
        <p className="small">
          No reliable pattern yet, so no factors are reported. Keep tagging; this retrains automatically.
        </p>
        <details>
          <summary className="small muted">Show the tentative ranking anyway (not reliable)</summary>
          {a.factors.slice(0, 5).map((f) => <FactorRow key={f.feature} f={f} title={title} />)}
        </details>
      </section>
    );
  }
  return (
    <section className="card">
      <h2>What goes with your {title.toLowerCase()}s</h2>
      <p className="small muted">
        Listed when the model relies on the metric <em>and</em> the metric alone separates your good and{" "}
        {title.toLowerCase()} shots (a rank test corrected for checking many metrics at once). Green dots are good
        shots, red are {title.toLowerCase()}s, and the blue line is your latest session.
      </p>
      {supported.length === 0 && <p className="small">The model predicts well, but no single metric passed both checks.</p>}
      {supported.map((f) => <FactorRow key={f.feature} f={f} title={title} />)}
      {rest.length > 0 && (
        <details>
          <summary className="small muted">Other metrics the model uses (unconfirmed)</summary>
          {rest.map((f) => <FactorRow key={f.feature} f={f} title={title} />)}
        </details>
      )}
    </section>
  );
}

export function WhatIfList({ items, title }: { items: WhatIf[]; title: string }) {
  if (items.length === 0) return <p className="small muted">No confirmed factor to change on this swing.</p>;
  return (
    <ul className="what-if">
      {items.map((w) => (
        <li key={w.feature} className="small">
          {featureLabel(w.feature, true)}: <strong>{w.value.toFixed(1)}</strong> → your good-shot median{" "}
          <strong>{w.target.toFixed(1)}</strong>. Predicted {title.toLowerCase()} chance {pct(w.probability)} →{" "}
          <strong className={w.delta < 0 ? "good-text" : "error"}>{pct(w.probability_if)}</strong>
        </li>
      ))}
    </ul>
  );
}

function ImpactClip({ r, label }: { r: ImpactRef | null; label: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  if (!r) return null;
  const seek = () => {
    const v = ref.current;
    if (v && r.impact_frame != null && r.fps) v.currentTime = (r.impact_frame + 0.5) / r.fps;
  };
  return (
    <figure className="impact-clip">
      {r.playback_url ? (
        <video ref={ref} src={r.playback_url} muted playsInline preload="auto" onLoadedMetadata={seek} />
      ) : (
        <div className="muted small">no video</div>
      )}
      <figcaption className="small">
        <strong>{label}</strong> · {r.outcome?.shape ?? r.outcome?.contact ?? ""}
        {r.probability != null && <> · predicted {pct(r.probability)}</>} ·{" "}
        <Link to={`/swings/${r.swing_id}`}>open</Link>
      </figcaption>
    </figure>
  );
}

function Detail({ a, llm }: { a: OutcomeAnalysis; llm: boolean }) {
  const [words, setWords] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const title = a.problem.title;
  return (
    <>
      <div className="analysis-grid">
        <DataStatus p={a} />
        <Reliability a={a} />
      </div>
      <Factors a={a} />
      {a.model?.reliable && a.what_if && (
        <section className="card">
          <h2>What if: your most recent swing</h2>
          <WhatIfList items={a.what_if.items} title={title} />
          <p className="muted small">
            Changes one metric at a time to your good-shot median and asks the model again. It's a correlation in
            your data, not a proven cause. Test it: work toward the range, tag the next sessions and watch this move.{" "}
            <Link to={`/swings/${a.what_if.swing_id}`}>Open the swing</Link>
          </p>
        </section>
      )}
      {a.references && (
        <section className="card">
          <h2>At impact: your best good swing vs a typical {title.toLowerCase()}</h2>
          <div className="impact-pair">
            <ImpactClip r={a.references.good} label="Good" />
            <ImpactClip r={a.references.bad} label={title} />
          </div>
        </section>
      )}
      {llm && a.model && (
        <section className="card">
          <h2>In words</h2>
          {words ? <p className="small">{words}</p> : (
            <button disabled={busy} onClick={async () => {
              setBusy(true);
              setError(null);
              try {
                setWords((await api.explainOutcome(a.problem.key)).explanation);
              } catch (e) {
                setError((e as Error).message);
              } finally {
                setBusy(false);
              }
            }}>
              {busy ? "Writing…" : "Explain in words"}
            </button>
          )}
          {error && <p className="error small">{error}</p>}
          <p className="muted small">Sends only the model's numbers above to Claude, which rephrases them.</p>
        </section>
      )}
    </>
  );
}

export default function Analysis() {
  const [summary, setSummary] = useState<OutcomesSummary | null>(null);
  const [problem, setProblem] = useState<ProblemKey>("slice");
  const [analysis, setAnalysis] = useState<OutcomeAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.outcomesSummary().then(setSummary, (e) => setError(e.message));
    api.outcomeAnalysis(problem).then(setAnalysis, (e) => setError(e.message));
  }, [problem]);
  useEffect(load, [load]);

  const training = summary?.training && (summary.training.status === "queued" || summary.training.status === "running");
  useEffect(() => {
    if (!training) return;
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [training, load]);

  return (
    <>
      <h1>Analysis</h1>
      <p className="muted small">
        Models trained only on your tagged swings learn which of your measured positions go with each bad shot.
        Tag outcomes on the session or swing pages (one tap per swing); training and model updates happen
        automatically.
      </p>
      {error && <p className="error">{error}</p>}
      {summary && (
        <div className="status-line small">
          <span>{summary.tagged} tagged of {summary.with_metrics} analysed swings</span>
          <span className="muted">
            retrains every {summary.retrain_every} tags ({summary.changes_since_training} since last)
          </span>
          {summary.training && (
            <span className={`chip ${summary.training.status === "failed" ? "bad" : training ? "busy" : "ok"}`}>
              training {training ? summary.training.stage ?? summary.training.status : summary.training.status}
            </span>
          )}
          <button className="link" disabled={!!training || summary.tagged === 0}
            onClick={() => api.trainOutcomes().then(load, (e) => setError(e.message))}>
            Retrain now
          </button>
        </div>
      )}

      <div className="tabs" role="tablist">
        {summary?.problems.map((p) => (
          <button key={p.key} role="tab" aria-selected={p.key === problem}
            className={`tab ${p.key === problem ? "on" : ""}`} onClick={() => setProblem(p.key)}>
            {p.title}
            <span className="muted small">
              {" "}
              {p.model ? (p.model.reliable ? "✓" : "…") : `${p.n_pos}/${p.n}`}
            </span>
          </button>
        ))}
      </div>
      {summary && <p className="muted small">{summary.problems.find((p) => p.key === problem)?.description}</p>}

      {analysis && analysis.problem.key === problem ? (
        <Detail a={analysis} llm={!!summary?.llm_enabled} />
      ) : (
        <p className="muted">Loading…</p>
      )}

      <p className="muted small limits">
        Limits: the models see your body (pose), not the club face or path, so they find what in your movement goes
        with the shot, not why the ball curved. Metrics marked <em>estimate</em> come from single-camera 3D. Findings
        are correlations in your own data.
      </p>
    </>
  );
}
