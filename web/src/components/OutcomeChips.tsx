import { useState } from "react";
import { api, CONTACTS, OutcomePatch, SHAPES, ShotOutcome, START_LINES } from "../api";

const SHAPE_HINT: Record<string, string> = {
  slice: "Curves hard away from you",
  fade: "Curves gently away from you",
  straight: "No curve",
  draw: "Curves gently in toward you",
  hook: "Curves hard in toward you",
};

/** One-tap shot outcome. Tapping the selected chip again clears it. */
export default function OutcomeChips({ swingId, outcome, onChange, full = false }: {
  swingId: string;
  outcome: ShotOutcome | null;
  onChange?: (o: ShotOutcome | null) => void;
  full?: boolean;
}) {
  const [cur, setCur] = useState(outcome);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function set(patch: OutcomePatch) {
    setSaving(true);
    setError(null);
    try {
      const o = await api.setOutcome(swingId, patch);
      setCur(o);
      onChange?.(o);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  const group = <K extends "shape" | "start_line" | "contact">(key: K, values: readonly string[], label: string) => (
    <div className="outcome-group" role="group" aria-label={label}>
      {full && <span className="muted small outcome-label">{label}</span>}
      {values.map((v) => {
        const on = cur?.[key] === v;
        return (
          <button
            key={v}
            className={`chip outcome ${on ? "on" : ""} ${v}`}
            aria-pressed={on}
            disabled={saving}
            title={key === "shape" ? SHAPE_HINT[v] : undefined}
            onClick={() => set({ [key]: on ? null : v } as OutcomePatch)}
          >
            {v}
          </button>
        );
      })}
    </div>
  );

  return (
    <div className={`outcome-chips ${full ? "full" : ""}`}>
      {group("shape", SHAPES, "Shape")}
      {full && group("start_line", START_LINES, "Start")}
      {group("contact", CONTACTS, "Contact")}
      {error && <span className="error small">{error}</span>}
    </div>
  );
}
