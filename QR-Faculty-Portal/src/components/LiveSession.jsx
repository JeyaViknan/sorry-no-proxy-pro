import { useCallback, useEffect, useState } from "react";
import { exportUrl } from "../lib/session.js";

/**
 * The projected view.
 *
 * Design constraint that drives everything here: this is read from the back
 * of a lecture hall, by a phone camera, often over a projector with poor
 * contrast. So the QR gets essentially the whole screen, on pure white, with
 * everything else demoted to a thin side rail that can be hidden entirely.
 *
 * The old layout put the QR in a decorated card sharing space with a stats
 * panel, at the library's default 128px. That is unreadable past the third
 * row.
 */
export default function LiveSession({ session, qrNode, status, summary, onEnd }) {
  const [focusMode, setFocusMode] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsed(Math.floor((Date.now() - session.startedAt) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [session.startedAt]);

  // Keep the projector awake — a screen blanking mid-session stops attendance.
  useEffect(() => {
    let sentinel = null;
    const acquire = async () => {
      try {
        sentinel = await navigator.wakeLock?.request("screen");
      } catch {
        /* unsupported or denied */
      }
    };
    acquire();
    const onVisible = () => {
      if (document.visibilityState === "visible") acquire();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      sentinel?.release?.().catch(() => {});
    };
  }, []);

  const toggleFullscreen = useCallback(async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch {
      /* denied — focus mode still helps */
    }
  }, []);

  useEffect(() => {
    const onKey = (event) => {
      if (event.key === "f") toggleFullscreen();
      if (event.key === "h") setFocusMode((value) => !value);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleFullscreen]);

  const minutes = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const seconds = String(elapsed % 60).padStart(2, "0");

  return (
    <main className={`shell shell-live ${focusMode ? "focus" : ""}`}>
      <section className="qr-stage">
        {/*
          Pure white background regardless of theme. QR contrast is the whole
          job of this surface, and a dark-mode inversion would break decoding
          on many phone cameras.
        */}
        <div className="qr-plate">{qrNode}</div>
      </section>

      {!focusMode && (
        <aside className="rail">
          <div className="rail-block">
            <p className="eyebrow">Live</p>
            <h2>{session.label || "Attendance"}</h2>
            <p className="mono session-id">{session.sessionId}</p>
          </div>

          <div className="rail-block">
            <p className="rail-label">Elapsed</p>
            <p className="rail-value mono">
              {minutes}:{seconds}
            </p>
          </div>

          <div className="rail-block">
            <p className="rail-label">Marked present</p>
            <p className="rail-value">{summary?.counts?.total ?? "—"}</p>
            {summary?.counts?.flagged > 0 && (
              <p className="rail-note warn">{summary.counts.flagged} need review</p>
            )}
          </div>

          {status.message && (
            <p className={`rail-note ${status.tone}`} role="status">
              {status.message}
            </p>
          )}

          <div className="rail-actions">
            <button type="button" className="btn btn-ghost btn-sm" onClick={toggleFullscreen}>
              Fullscreen <kbd>F</kbd>
            </button>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => setFocusMode(true)}
            >
              Hide panel <kbd>H</kbd>
            </button>
            <a className="btn btn-ghost btn-sm" href={exportUrl(session.sessionId)} download>
              Download CSV
            </a>
            <button type="button" className="btn btn-danger btn-sm" onClick={onEnd}>
              End session
            </button>
          </div>
        </aside>
      )}

      {focusMode && (
        <button
          type="button"
          className="focus-exit"
          onClick={() => setFocusMode(false)}
          aria-label="Show session panel"
        >
          ▸
        </button>
      )}
    </main>
  );
}
