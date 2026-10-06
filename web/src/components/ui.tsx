import { createContext, ReactNode, useCallback, useContext, useEffect, useRef, useState } from "react";

type ToastKind = "ok" | "error";
interface ToastItem { id: number; kind: ToastKind; text: string }
interface ConfirmReq {
  title: string;
  body?: string;
  confirmLabel: string;
  danger: boolean;
  resolve: (ok: boolean) => void;
}
interface Ctx {
  toast: (text: string, kind?: ToastKind) => void;
  confirm: (opts: { title: string; body?: string; confirmLabel?: string; danger?: boolean }) => Promise<boolean>;
}

const UiContext = createContext<Ctx | null>(null);

export function useUi(): Ctx {
  const ctx = useContext(UiContext);
  if (!ctx) throw new Error("useUi must be used inside <UiProvider>");
  return ctx;
}

function ConfirmDialog({ req, onClose }: { req: ConfirmReq; onClose: (ok: boolean) => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    ref.current?.showModal();
  }, []);
  return (
    <dialog ref={ref} className="confirm" onCancel={() => onClose(false)} onClose={() => onClose(false)}>
      <h2>{req.title}</h2>
      {req.body && <p className="muted">{req.body}</p>}
      <div className="confirm-actions">
        <button autoFocus onClick={() => onClose(false)}>Cancel</button>
        <button className={req.danger ? "danger" : "primary"} onClick={() => onClose(true)}>
          {req.confirmLabel}
        </button>
      </div>
    </dialog>
  );
}

/** Toasts (role=status / role=alert) and promise-based confirm dialogs. */
export function UiProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [req, setReq] = useState<ConfirmReq | null>(null);
  const nextId = useRef(1);

  const toast = useCallback((text: string, kind: ToastKind = "ok") => {
    const id = nextId.current++;
    setToasts((t) => [...t, { id, kind, text }]);
    window.setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === "error" ? 6000 : 3000);
  }, []);

  const confirm = useCallback<Ctx["confirm"]>(
    (o) =>
      new Promise((resolve) =>
        setReq({ title: o.title, body: o.body, confirmLabel: o.confirmLabel ?? "Confirm", danger: o.danger ?? false, resolve }),
      ),
    [],
  );

  const close = (ok: boolean) => {
    setReq((r) => {
      r?.resolve(ok);
      return null;
    });
  };

  return (
    <UiContext.Provider value={{ toast, confirm }}>
      {children}
      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`} role={t.kind === "error" ? "alert" : "status"}>
            {t.text}
          </div>
        ))}
      </div>
      {req && <ConfirmDialog req={req} onClose={close} />}
    </UiContext.Provider>
  );
}
