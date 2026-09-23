// Small SVG charts for the outcome models. Colors come from the CSS tokens (light/dark aware).

import { fmtValue } from "./MetricsTable";

const W = 320;

function scale(lo: number, hi: number, x0: number, x1: number) {
  const span = hi - lo || 1;
  return (v: number) => x0 + ((v - lo) / span) * (x1 - x0);
}

/** Your good vs bad shots on one metric: dots per swing, medians, and your latest session. */
export function StripChart({ points, goodMedian, badMedian, latest, unit, badLabel }: {
  points: { value: number; bad: boolean }[];
  goodMedian: number | null;
  badMedian: number | null;
  latest: number | null | undefined;
  unit: string;
  badLabel: string;
}) {
  const vals = points.map((p) => p.value).concat(latest != null ? [latest] : []);
  if (vals.length === 0) return null;
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  const pad = (hi - lo) * 0.06 || 1;
  lo -= pad;
  hi += pad;
  const x = scale(lo, hi, 58, W - 8);
  const rows = { good: 16, bad: 40 };
  // Deterministic vertical jitter so overlapping swings stay visible.
  const jitter = (i: number) => (((i * 2654435761) % 1000) / 1000 - 0.5) * 9;
  return (
    <svg className="chart strip" viewBox={`0 0 ${W} 66`} role="img"
      aria-label={`Good vs ${badLabel} shots; medians ${goodMedian?.toFixed(1)} vs ${badMedian?.toFixed(1)} ${unit}`}>
      <text x={0} y={rows.good + 4} className="chart-label">good</text>
      <text x={0} y={rows.bad + 4} className="chart-label bad">{badLabel}</text>
      <line x1={58} x2={W - 8} y1={rows.good} y2={rows.good} className="chart-grid" />
      <line x1={58} x2={W - 8} y1={rows.bad} y2={rows.bad} className="chart-grid" />
      {points.map((p, i) => (
        <circle key={i} cx={x(p.value)} cy={(p.bad ? rows.bad : rows.good) + jitter(i)} r={3}
          className={p.bad ? "dot bad" : "dot good"} />
      ))}
      {goodMedian != null && <line x1={x(goodMedian)} x2={x(goodMedian)} y1={rows.good - 9} y2={rows.good + 9} className="median good" />}
      {badMedian != null && <line x1={x(badMedian)} x2={x(badMedian)} y1={rows.bad - 9} y2={rows.bad + 9} className="median bad" />}
      {latest != null && (
        <g>
          <line x1={x(latest)} x2={x(latest)} y1={4} y2={52} className="latest" />
          <title>Your latest session: {fmtValue(latest, unit)}</title>
        </g>
      )}
      <text x={58} y={64} className="chart-tick">{fmtValue(lo + pad, unit)}</text>
      <text x={W - 8} y={64} textAnchor="end" className="chart-tick">{fmtValue(hi - pad, unit)}</text>
    </svg>
  );
}

/** Cross-validated AUC with its CI against chance, the shuffled-label null and the reliability gate. */
export function AucInterval({ auc, ci, null95, gate, reliable }: {
  auc: number | null;
  ci: [number | null, number | null];
  null95: number | null;
  gate: number;
  reliable: boolean;
}) {
  const x = scale(0.3, 1, 8, W - 8);
  const ref = (v: number, label: string, cls: string, y: number) => (
    <g>
      <line x1={x(v)} x2={x(v)} y1={10} y2={34} className={`ref ${cls}`} />
      <text x={x(v)} y={y} textAnchor="middle" className="chart-tick">{label}</text>
    </g>
  );
  return (
    <svg className="chart auc" viewBox={`0 0 ${W} 58`} role="img"
      aria-label={`AUC ${auc?.toFixed(2)}, 95% CI ${ci[0]?.toFixed(2)} to ${ci[1]?.toFixed(2)}`}>
      <line x1={x(0.3)} x2={x(1)} y1={22} y2={22} className="chart-grid" />
      {ref(0.5, "chance", "chance", 8)}
      {null95 != null && ref(null95, "shuffled 95%", "null", 44)}
      {ref(gate, `gate ${gate.toFixed(2)}`, "gate", 56)}
      {ci[0] != null && ci[1] != null && (
        <line x1={x(ci[0])} x2={x(ci[1])} y1={22} y2={22} className={`ci ${reliable ? "ok" : "weak"}`} />
      )}
      {auc != null && <circle cx={x(auc)} cy={22} r={5} className={`point ${reliable ? "ok" : "weak"}`} />}
      <text x={x(1)} y={8} textAnchor="end" className="chart-tick">1.0</text>
    </svg>
  );
}
