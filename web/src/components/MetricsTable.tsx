import { EVENT_LABELS, EVENT_TYPES, EventType, Metric } from "../api";

const NAMES: Record<string, string> = {
  tempo_ratio: "Tempo (backswing : downswing)",
  backswing_time: "Backswing time",
  downswing_time: "Downswing time",
  shoulder_tilt: "Shoulder tilt (lead side up +)",
  hip_tilt: "Hip tilt (lead side up +)",
  spine_tilt_away_from_target: "Spine tilt away from target",
  head_sway_toward_target: "Head sway toward target",
  head_rise: "Head rise",
  hip_sway_toward_target: "Hip sway toward target",
  shoulder_turn: "Shoulder turn",
  hip_turn: "Hip turn",
  x_factor: "X-factor (shoulder − hip turn)",
  lead_elbow_angle: "Lead elbow angle (180 = straight)",
  lead_knee_flex: "Lead knee flex",
  trail_knee_flex: "Trail knee flex",
};

const ESTIMATE_NOTE =
  "Monocular 3D estimate (MediaPipe world landmarks). Depth is inferred, not measured; " +
  "no error bound until a dual-camera calibration session exists.";

function fmt(m: Metric): string {
  if (m.unit === "ratio") return `${m.value.toFixed(2)} : 1`;
  if (m.unit === "s") return `${m.value.toFixed(2)} s`;
  if (m.unit === "deg") return `${m.value.toFixed(1)}°`;
  if (m.unit.startsWith("%")) return `${m.value.toFixed(0)} ${m.unit}`;
  return `${m.value.toFixed(2)} ${m.unit}`;
}

export default function MetricsTable({ metrics, onJump, eventFrames }: {
  metrics: Metric[];
  onJump: (frame: number) => void;
  eventFrames: Partial<Record<EventType, number>>;
}) {
  if (metrics.length === 0) return <p className="muted">No metrics yet.</p>;
  const groups: [EventType | null, Metric[]][] = [[null, metrics.filter((m) => m.event_ref === null)]];
  for (const et of EVENT_TYPES) {
    const ms = metrics.filter((m) => m.event_ref === et);
    if (ms.length) groups.push([et, ms]);
  }
  return (
    <table className="table metrics">
      <tbody>
        {groups.map(([et, ms]) =>
          ms.length === 0 ? null : (
            <GroupRows key={et ?? "overall"} et={et} ms={ms} onJump={onJump} frame={et ? eventFrames[et] : undefined} />
          ),
        )}
      </tbody>
    </table>
  );
}

function GroupRows({ et, ms, onJump, frame }: { et: EventType | null; ms: Metric[]; onJump: (f: number) => void; frame?: number }) {
  return (
    <>
      <tr className="group">
        <th colSpan={2}>
          {et ? EVENT_LABELS[et] : "Timing"}
          {et && frame != null && (
            <button className="link small" onClick={() => onJump(frame)}>
              frame {frame}
            </button>
          )}
        </th>
      </tr>
      {ms.map((m) => (
        <tr key={`${m.metric_name}-${m.event_ref}`}>
          <td>
            {NAMES[m.metric_name] ?? m.metric_name}
            {m.is_estimate && (
              <span className="badge" title={ESTIMATE_NOTE}>
                3D estimate
              </span>
            )}
          </td>
          <td className="num">{fmt(m)}</td>
        </tr>
      ))}
    </>
  );
}
