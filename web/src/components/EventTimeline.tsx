import { PointerEvent as RPointerEvent, useRef, useState } from "react";
import { EVENT_LABELS, EventType, SwingEvent } from "../api";

export const REVIEW_THRESHOLD = 0.6;

interface Props {
  numFrames: number;
  frame: number;
  events: SwingEvent[];
  onSeek: (frame: number) => void;
  onCorrect: (event: EventType, frame: number) => void;
}

const SHORT: Record<EventType, string> = {
  address: "A",
  toe_up: "TU",
  mid_backswing: "MB",
  top: "T",
  mid_downswing: "MD",
  impact: "I",
  mid_follow_through: "MF",
  finish: "F",
};

/** Track with playhead + one draggable marker per event. Drag a marker to correct it. */
export default function EventTimeline({ numFrames, frame, events, onSeek, onCorrect }: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<{ event: EventType; frame: number } | null>(null);
  const last = Math.max(1, numFrames - 1);
  const pct = (f: number) => `${(f / last) * 100}%`;

  const frameAt = (clientX: number) => {
    const r = trackRef.current!.getBoundingClientRect();
    return Math.round(Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * last);
  };

  const onTrackDown = (e: RPointerEvent) => {
    if ((e.target as HTMLElement).dataset.marker) return;
    onSeek(frameAt(e.clientX));
  };

  const markerHandlers = (ev: SwingEvent) => ({
    onPointerDown: (e: RPointerEvent) => {
      e.stopPropagation();
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      setDrag({ event: ev.event_type, frame: ev.frame_index });
    },
    onPointerMove: (e: RPointerEvent) => {
      if (drag?.event !== ev.event_type) return;
      const f = frameAt(e.clientX);
      setDrag({ event: ev.event_type, frame: f });
      onSeek(f);
    },
    onPointerUp: () => {
      if (drag?.event !== ev.event_type) return;
      if (drag.frame !== ev.frame_index) onCorrect(ev.event_type, drag.frame);
      else onSeek(ev.frame_index);
      setDrag(null);
    },
  });

  return (
    <div className="timeline">
      <div className="track" ref={trackRef} onPointerDown={onTrackDown}>
        <div className="playhead" style={{ left: pct(frame) }} />
        {events.map((ev) => {
          const f = drag?.event === ev.event_type ? drag.frame : ev.frame_index;
          const low = !ev.corrected && (ev.confidence ?? 0) < REVIEW_THRESHOLD;
          return (
            <div
              key={ev.event_type}
              data-marker="1"
              className={`marker ${ev.corrected ? "corrected" : ""} ${low ? "low" : ""}`}
              style={{ left: pct(f) }}
              title={`${EVENT_LABELS[ev.event_type]} · frame ${f}${ev.confidence != null ? ` · conf ${ev.confidence.toFixed(2)}` : ""}${ev.corrected ? " · corrected" : ""}`}
              {...markerHandlers(ev)}
            >
              {SHORT[ev.event_type]}
            </div>
          );
        })}
      </div>
    </div>
  );
}
