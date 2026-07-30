import { useState } from "react";

/**
 * Pre-session setup.
 *
 * The old setup screen asked for "speed per QR", "number of valid QRs" and
 * "shuffling duration". All three are gone, and that is a deliberate
 * simplification rather than a feature cut:
 *
 *   • Rotation speed and validity window are now protocol-level properties
 *     enforced by the server (QR_TOKEN_TTL_MS). A faculty member setting them
 *     per session could silently widen the window a proxy needs, which is
 *     exactly the parameter that should not be adjustable from the room.
 *
 *   • "Number of valid QRs" existed because the old design showed a valid
 *     code only at a few random instants. With signed, rotating slots every
 *     code on screen is valid for its own 4-second window, so a student can
 *     scan whenever they are ready instead of waiting to catch a lucky
 *     moment. That removes the single most frustrating part of the flow and
 *     loses nothing — a code is still useless seconds later, or outside the
 *     room.
 *
 * What remains is the one thing the server genuinely cannot know: which class
 * this is.
 */
export default function SetupScreen({ onStart, onSignOut }) {
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async (event) => {
    event.preventDefault();
    if (busy) return;

    setBusy(true);
    setError("");
    try {
      await onStart({ label: label.trim() });
    } catch (err) {
      setError(err.message || "Could not start the session.");
      setBusy(false);
    }
  };

  return (
    <main className="shell shell-center">
      <form className="card card-narrow" onSubmit={submit}>
        <p className="eyebrow">New session</p>
        <h1>Start attendance</h1>
        <p className="subtext">
          Project the next screen. Students scan the rotating code, then confirm
          their face.
        </p>

        <label className="field">
          <span>Class label (optional)</span>
          <input
            type="text"
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            placeholder="CSE-A · Lecture 3"
            maxLength={80}
            autoFocus
          />
        </label>

        {error && (
          <p className="error-text" role="alert">
            {error}
          </p>
        )}

        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? "Starting…" : "Start session"}
        </button>
        <button type="button" className="btn btn-ghost" onClick={onSignOut}>
          Sign out
        </button>
      </form>
    </main>
  );
}
