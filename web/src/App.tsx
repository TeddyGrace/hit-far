import { useEffect, useState } from "react";
import { Link, NavLink, Route, Routes, useNavigate } from "react-router-dom";
import { api, Me, setUnauthorizedHandler } from "./api";
import { UiProvider } from "./components/ui";
import Analysis from "./pages/Analysis";
import Login from "./pages/Login";
import Models from "./pages/Models";
import SessionDetail from "./pages/SessionDetail";
import Sessions from "./pages/Sessions";
import SwingView from "./pages/SwingView";
import Users from "./pages/Users";

export default function App() {
  // undefined while checking the session cookie, null when logged out.
  const [me, setMe] = useState<Me | null | undefined>(undefined);
  const navigate = useNavigate();

  useEffect(() => {
    setUnauthorizedHandler(() => setMe(null));
    api.me().then(setMe, () => setMe(null));
  }, []);

  if (me === undefined) return <div className="page muted">Loading…</div>;
  if (me === null) return <Login onLogin={setMe} />;

  return (
    <UiProvider>
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
          {me.is_admin && <NavLink to="/users">Users</NavLink>}
        </nav>
        <span className="muted small">{me.username}</span>
        <button
          className="link"
          onClick={async () => {
            await api.logout();
            setMe(null);
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
          {me.is_admin && <Route path="/users" element={<Users me={me} />} />}
          <Route path="*" element={<p>Not found. <Link to="/">Back to sessions</Link></p>} />
        </Routes>
      </main>
    </UiProvider>
  );
}
