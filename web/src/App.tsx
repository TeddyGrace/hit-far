import { useEffect, useState } from "react";
import { Link, NavLink, Route, Routes, useNavigate } from "react-router-dom";
import { api, setUnauthorizedHandler } from "./api";
import Analysis from "./pages/Analysis";
import Login from "./pages/Login";
import Models from "./pages/Models";
import SessionDetail from "./pages/SessionDetail";
import Sessions from "./pages/Sessions";
import SwingView from "./pages/SwingView";

export default function App() {
  const [authed, setAuthed] = useState<boolean | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    setUnauthorizedHandler(() => setAuthed(false));
    api.me().then(
      () => setAuthed(true),
      () => setAuthed(false),
    );
  }, []);

  if (authed === null) return <div className="page muted">Loading…</div>;
  if (!authed) return <Login onLogin={() => setAuthed(true)} />;

  return (
    <>
      <header className="topbar">
        <Link to="/" className="brand">
          hit-far
        </Link>
        <nav>
          <NavLink to="/" end>
            Sessions
          </NavLink>
          <NavLink to="/analysis">Analysis</NavLink>
          <NavLink to="/models">Models</NavLink>
        </nav>
        <button
          className="link"
          onClick={async () => {
            await api.logout();
            setAuthed(false);
            navigate("/");
          }}
        >
          Log out
        </button>
      </header>
      <main className="page">
        <Routes>
          <Route path="/" element={<Sessions />} />
          <Route path="/sessions/:id" element={<SessionDetail />} />
          <Route path="/swings/:id" element={<SwingView />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/models" element={<Models />} />
          <Route path="*" element={<p>Not found.</p>} />
        </Routes>
      </main>
    </>
  );
}
