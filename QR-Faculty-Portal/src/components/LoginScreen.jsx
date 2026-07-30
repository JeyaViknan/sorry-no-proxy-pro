import { useState } from "react";

/**
 * Faculty sign-in.
 *
 * The access code is exchanged server-side for a signed, expiring token, and
 * the endpoint is rate limited to 10 attempts per 15 minutes per IP. Nothing
 * about session control was authenticated at all before this.
 */
export default function LoginScreen({ onLogin }) {
  const [accessCode, setAccessCode] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    if (!accessCode.trim() || busy) return;

    setBusy(true);
    setError("");
    try {
      await onLogin(accessCode.trim());
    } catch (err) {
      setError(
        err.code === "RATE_LIMITED_AUTH"
          ? "Too many attempts. Wait 15 minutes and try again."
          : err.message || "Sign in failed."
      );
      setBusy(false);
    }
  };

  return (
    <main className="shell shell-center">
      <form className="card card-narrow" onSubmit={submit}>
        <p className="eyebrow">Faculty</p>
        <h1>Attendance display</h1>
        <p className="subtext">
          Enter the access code for this semester to start a session.
        </p>

        <label className="field">
          <span>Access code</span>
          <input
            type="password"
            value={accessCode}
            onChange={(event) => setAccessCode(event.target.value)}
            autoComplete="current-password"
            autoFocus
            required
            aria-invalid={Boolean(error)}
            aria-describedby={error ? "login-error" : undefined}
          />
        </label>

        {error && (
          <p id="login-error" className="error-text" role="alert">
            {error}
          </p>
        )}

        <button type="submit" className="btn btn-primary" disabled={busy || !accessCode.trim()}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </main>
  );
}
