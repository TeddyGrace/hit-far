import { EVENT_LABELS, EVENT_TYPES, EventType, Metric } from "../api";

export const NAMES: Record<string, string> = {
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
  transition_time: "Transition time (top → mid-downswing)",
  shoulders_open: "Shoulders open (vs address; − = closed)",
  hips_open: "Hips open (vs address; − = closed)",
  sequencing_hip_lead: "Hips lead shoulders (peak rotation speed)",
  pelvis_toward_lead_foot: "Pelvis over stance (0 trail – 100 lead ankle)",
  hands_ahead: "Hands ahead of address position",
  lead_wrist_bow: "Lead wrist bow (+ bowed, − cupped)",
  lead_wrist_hinge: "Lead wrist hinge (+ cocked)",
  lead_forearm_roll: "Forearm roll (+ = rotated closed vs address)",
  forearm_roll_speed: "Forearm roll speed (+ = closing)",
  shaft_lean: "Shaft lean (+ = hands ahead)",
  shaft_past_parallel: "Shaft past parallel (+ past, − short)",
  wrist_hinge_shaft: "Wrist hinge (forearm–shaft angle)",
  lag_angle: "Lag (forearm–shaft angle)",
  shaft_release_speed: "Shaft release speed through impact",
};

/** "shoulders_open@impact" -> "Shoulders open (…) at Impact". */
export function featureLabel(feature: string, short = false): string {
  const [metric, event] = feature.split("@");
  let name = NAMES[metric] ?? metric.replace(/_/g, " ");
  if (short) name = name.replace(/\s*\(.*\)$/, "");
  return event ? `${name} at ${EVENT_LABELS[event as EventType] ?? event}` : name;
}

export function fmtValue(value: number, unit: string): string {
  if (unit === "ratio") return `${value.toFixed(2)} : 1`;
  if (unit === "s") return `${value.toFixed(2)} s`;
  if (unit === "ms") return `${value.toFixed(0)} ms`;
  if (unit === "deg/s") return `${value.toFixed(0)}°/s`;
  if (unit === "deg") return `${value.toFixed(1)}°`;
  if (unit.startsWith("%")) return `${value.toFixed(0)} ${unit}`;
  return `${value.toFixed(2)} ${unit}`;
}

export const ESTIMATE_NOTE =
  "Monocular 3D estimate (MediaPipe world landmarks). Depth is inferred, not measured; " +
  "no error bound until a dual-camera calibration session exists.";

function fmt(m: Metric): string {
  return fmtValue(m.value, m.unit);
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
