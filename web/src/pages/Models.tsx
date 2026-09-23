import { useEffect, useState } from "react";
import { api, ModelInfo } from "../api";

export default function Models() {
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listModels().then(setModels, (e) => setError(e.message));
  }, []);

  return (
    <>
      <h1>Models</h1>
      <p className="muted">
        Every pose, event and (later) club/fault output is tied to the model row that produced it. Models are
        registered here the first time they run; trained candidates will start as <em>experimental</em>.
      </p>
      {error && <p className="error">{error}</p>}
      {models && (
        <table className="table">
          <thead>
            <tr>
              <th>Task</th>
              <th>Name</th>
              <th>Version</th>
              <th>Status</th>
              <th>Eval</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.id}>
                <td>{m.task}</td>
                <td>{m.name}</td>
                <td>{m.version}</td>
                <td>
                  <span className={`chip ${m.status === "active" ? "ok" : ""}`}>{m.status}</span>
                </td>
                <td className="muted">{m.eval_metrics ? JSON.stringify(m.eval_metrics) : "—"}</td>
                <td className="muted">{m.notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}
