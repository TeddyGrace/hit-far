import { FormEvent, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, SessionSummary } from "../api";

function todayLocal(): string {
  const d = new Date();
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
  return d.toISOString().slice(0, 10);
}

export default function Sessions() {
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [date, setDate] = useState(todayLocal());
  const [name, setName] = useState("");
  const [location, setLocation] = useState("");
  const [club, setClub] = useState("");
  const [notes, setNotes] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    api.listSessions().then(setSessions, (e) => setError(e.message));
  }, []);

  async function create(e: FormEvent) {
    e.preventDefault();
    try {
      const s = await api.createSession({
        recorded_at: new Date(`${date}T12:00:00`).toISOString(),
        name: name || null,
        location: location || null,
        club_used: club || null,
        notes: notes || null,
      });
      navigate(`/sessions/${s.id}`);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <>
      <h1>Sessions</h1>
      {error && <p className="error">{error}</p>}
      <form className="card row-form" onSubmit={create}>
        <label>
          Date
          <input type="date" value={date} onChange={(e) => setDate(e.target.value)} required />
        </label>
        <label>
          Name
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Tuesday range" />
        </label>
        <label>
          Location
          <input value={location} onChange={(e) => setLocation(e.target.value)} placeholder="Range, sim, course…" />
        </label>
        <label>
          Club
          <input value={club} onChange={(e) => setClub(e.target.value)} placeholder="7i" />
        </label>
        <label className="grow">
          Notes
          <input value={notes} onChange={(e) => setNotes(e.target.value)} />
        </label>
        <button type="submit">New session</button>
      </form>

      {sessions === null ? (
        <p className="muted">Loading…</p>
      ) : sessions.length === 0 ? (
        <p className="muted">No sessions yet. Create one, then upload face-on swing videos to it.</p>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Name</th>
              <th>Location</th>
              <th>Club</th>
              <th>Videos</th>
              <th>Swings</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((s) => (
              <tr key={s.id}>
                <td>
                  <Link to={`/sessions/${s.id}`}>{new Date(s.recorded_at).toLocaleDateString()}</Link>
                </td>
                <td>{s.name ?? "—"}</td>
                <td>{s.location ?? "—"}</td>
                <td>{s.club_used ?? "—"}</td>
                <td>{s.num_videos}</td>
                <td>{s.num_swings}</td>
                <td className="muted">{s.notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}
