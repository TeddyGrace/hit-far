import { useUi } from "../components/ui";
import { FormEvent, useCallback, useEffect, useState } from "react";
import { api, Me, UserInfo } from "../api";

// Admin-only: add golfers, reset passwords, remove users. There is no sign-up page.
export default function Users({ me }: { me: Me }) {
  const ui = useUi();
  const [users, setUsers] = useState<UserInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isAdmin, setIsAdmin] = useState(false);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    api.listUsers().then(setUsers, (e) => setError((e as Error).message));
  }, []);
  useEffect(load, [load]);

  async function run(action: () => Promise<unknown>, done: string) {
    setError(null);
    setNotice(null);
    try {
      await action();
      setNotice(done);
      load();
      return true;
    } catch (e) {
      setError((e as Error).message);
      return false;
    }
  }

  async function create(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    const name = username.trim().toLowerCase();
    const ok = await run(
      () => api.createUser({ username: name, password, is_admin: isAdmin }),
      `Added ${name}. They can sign in with that username and password now.`,
    );
    if (ok) {
      setUsername("");
      setPassword("");
      setIsAdmin(false);
    }
    setBusy(false);
  }

  function resetPassword(u: UserInfo) {
    const pw = window.prompt(`New password for ${u.username} (at least 8 characters):`);
    if (pw) run(() => api.updateUser(u.id, { password: pw }), `Password for ${u.username} changed.`);
  }

  function toggleAdmin(u: UserInfo) {
    run(
      () => api.updateUser(u.id, { is_admin: !u.is_admin }),
      u.is_admin ? `${u.username} is no longer an admin.` : `${u.username} is now an admin.`,
    );
  }

  async function remove(u: UserInfo) {
    if (await ui.confirm({ title: `Delete ${u.username}?`, body: "This can't be undone.", confirmLabel: "Delete", danger: true }))
      run(() => api.deleteUser(u.id), `Deleted ${u.username}.`);
  }

  return (
    <div>
      <h1>Users</h1>
      <form className="card row-form" onSubmit={create}>
        <label className="grow">
          Username
          <input autoCapitalize="none" autoComplete="off" value={username} placeholder="e.g. alex"
            onChange={(e) => setUsername(e.target.value)} />
        </label>
        <label className="grow">
          Password
          <input type="text" autoComplete="new-password" value={password} placeholder="at least 8 characters"
            onChange={(e) => setPassword(e.target.value)} />
        </label>
        <label className="inline">
          <input type="checkbox" checked={isAdmin} onChange={(e) => setIsAdmin(e.target.checked)} />
          Admin (can manage users)
        </label>
        <button type="submit" disabled={busy || !username.trim() || password.length < 8}>
          {busy ? "Adding…" : "Add user"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {notice && <p className="muted">{notice}</p>}
      {users === null ? (
        <p className="muted">Loading…</p>
      ) : (
        <div className="table-wrap"><table className="table">
          <thead>
            <tr>
              <th>Username</th>
              <th>Role</th>
              <th>Sessions</th>
              <th>Created</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {users.map((u) => {
              const self = u.username === me.username;
              return (
                <tr key={u.id}>
                  <td>
                    {u.username}
                    {self && <span className="muted"> (you)</span>}
                  </td>
                  <td>{u.is_admin ? <span className="chip ok">admin</span> : <span className="muted">golfer</span>}</td>
                  <td className="num">{u.num_sessions}</td>
                  <td>{new Date(u.created_at).toLocaleDateString()}</td>
                  <td className="actions">
                    <button className="link" onClick={() => resetPassword(u)}>
                      Reset password
                    </button>
                    {!self && (
                      <>
                        <button className="link" onClick={() => toggleAdmin(u)}>
                          {u.is_admin ? "Remove admin" : "Make admin"}
                        </button>
                        <button className="link danger" onClick={() => remove(u)}
                          disabled={u.num_sessions > 0}
                          title={u.num_sessions > 0 ? "Delete their sessions first" : undefined}>
                          Delete
                        </button>
                      </>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table></div>
      )}
    </div>
  );
}
