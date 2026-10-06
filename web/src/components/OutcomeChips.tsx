import { useRef, useState } from "react";
import { api, CONTACTS, OutcomePatch, SHAPES, ShotOutcome, START_LINES } from "../api";

const SHAPE_HINT: Record<string, string> = {
  slice: "Curves hard away from you: a miss",
  fade: "Gentle, controlled curve away from you: counts as a good shot",
  straight: "No curve: a good shot",
  draw: "Gentle, controlled curve in toward you: counts as a good shot",
  hook: "Curves hard in toward you: a miss",
};

type Key = "shape" | "start_line" | "contact";

/** Optimistically apply a patch to the current outcome. */
function applyPatch(cur: ShotOutcome | null, patch: OutcomePatch): ShotOutcome {
  const base: ShotOutcome = cur ?? {
    shape: null, start_line: null, contact: null, source: "self",
    club_path: null, face_to_path: null, face_angle: null, carry: null, offline: null,
    updated_at: new Date().toISOString(),
  };
  return { ...base, ...patch } as ShotOutcome;
}

/**
 * One-tap shot outcome. Tapping the selected chip again clears it. Taps apply
 * immediately and roll back if the save fails, so tagging never blocks on the network.
 */
export default function OutcomeChips({ swingId, outcome, onChange, full = false }: {
  swingId: string;
  outcome: ShotOutcome | null;
  onChange?: (o: ShotOutcome | null) => void;
  full?: boolean;
}) {
  const [cur, setCur] = useState(outcome);
  const [pending, setPending] = useState(0);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const latest = useRef(outcome);
  const savedTimer = useRef<number | undefined>(undefined);

  async function set(patch: OutcomePatch) {
    const before = latest.current;
    const optimistic = applyPatch(before, patch);
    latest.current = optimistic;
    setCur(optimistic);
    setPending((n) => n + 1);
    setError(null);
    setSaved(false);
    try {
      const o = await api.setOutcome(swingId, patch);
      // Keep any taps made while this request was in flight.
      latest.current = o;
      setCur(o);
      onChange?.(o);
      setSaved(true);
      window.clearTimeout(savedTimer.current);
      savedTimer.current = window.setTimeout(() => setSaved(false), 1500);
    } catch (e) {
      latest.current = before;
      setCur(before);
      setError((e as Error).message);
    } finally {
      setPending((n) => n - 1);
    }
  }

  const group = (key: Key, values: readonly string[], label: string) => (
    <div className="outcome-group" role="group" aria-label={label}>
      <span className="muted small outcome-label">{label}</span>
      {values.map((v) => {
        const on = cur?.[key] === v;
        return (
          <button
            key={v}
            className={`chip outcome ${on ? "on" : ""} ${v}`}
            aria-pressed={on}
            onClick={() => set({ [key]: on ? null : v } as OutcomePatch)}
          >
            {v}
          </button>
        );
      })}
    </div>
  );

  const hint = cur?.shape ? SHAPE_HINT[cur.shape] : null;

  return (
    <div className={`outcome-chips ${full ? "full" : ""}`}>
      {group("shape", SHAPES, "Shape")}
      {hint && <span className="muted small outcome-hint">{hint}</span>}
      {group("start_line", START_LINES, "Start")}
      {group("contact", CONTACTS, "Contact")}
      <span className="outcome-status small" role="status" aria-live="polite">
        {pending > 0 ? <span className="muted">saving…</span> : saved ? <span className="good-text">✓ saved</span> : null}
      </span>
      {error && <span className="error small" role="alert">{error}</span>}
    </div>
  );
}
