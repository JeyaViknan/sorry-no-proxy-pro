import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { QRCodeSVG } from "qrcode.react";

import * as session from "./lib/session.js";
import LoginScreen from "./components/LoginScreen.jsx";
import SetupScreen from "./components/SetupScreen.jsx";
import LiveSession from "./components/LiveSession.jsx";
import "./App.css";

/**
 * Faculty QR display.
 *
 * WHAT CHANGED
 * ------------
 * • No secret in the client. Payloads are fetched pre-signed from the server
 *   (see lib/session.js). The old build did `jeycavbhakanadiyaz${hh}${mm}`
 *   locally, duplicating the scanner's logic and publishing the secret.
 *
 * • The display schedule is driven by SERVER time. The old code decided when
 *   a code was valid using the faculty laptop's clock, which had to agree
 *   with each student's phone clock. Both are now irrelevant: the server
 *   signs a slot, and both sides ask the server what time it is.
 *
 * • Decoys are format-identical to real payloads, so a valid code is no
 *   longer identifiable by its lower QR density.
 *
 * • Rendering runs on requestAnimationFrame with a display-time check rather
 *   than setInterval juggling. setInterval drifts, and when the tab is
 *   backgrounded browsers throttle it to once per second — which silently
 *   broke the rotation.
 */

const REFRESH_INTERVAL_MS = 400;
const SUMMARY_POLL_MS = 6000;

export default function App() {
  const [token, setToken] = useState(() => session.storedToken());
  const [live, setLive] = useState(null);
  const [qrValue, setQrValue] = useState("");
  const [status, setStatus] = useState({ tone: "idle", message: "" });
  const [summary, setSummary] = useState(null);

  const bufferRef = useRef(null);
  const rafRef = useRef(null);
  const lastPaintRef = useRef(0);
  const noiseSpecRef = useRef(null);

  // ── Display loop ───────────────────────────────────────────────────
  const paint = useCallback(() => {
    rafRef.current = requestAnimationFrame(paint);

    const now = performance.now();
    if (now - lastPaintRef.current < REFRESH_INTERVAL_MS) return;
    lastPaintRef.current = now;

    const buffer = bufferRef.current;
    if (!buffer) return;

    const valid = buffer.current();
    setQrValue(valid ? valid.payload : session.makeDecoy(noiseSpecRef.current, buffer.sessionId));

    if (buffer.needsRefill()) {
      buffer
        .refill()
        .then(() => setStatus({ tone: "ok", message: "" }))
        .catch(() =>
          setStatus({
            tone: "warn",
            message: "Reconnecting to the server — codes will resume automatically.",
          })
        );
    }
  }, []);

  useEffect(() => {
    if (!live) return undefined;
    rafRef.current = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(rafRef.current);
  }, [live, paint]);

  // ── Attendance summary polling ─────────────────────────────────────
  useEffect(() => {
    if (!live || !token) return undefined;

    let cancelled = false;
    const poll = async () => {
      try {
        const data = await session.fetchSummary({ token, sessionId: live.sessionId });
        if (!cancelled) setSummary(data);
      } catch {
        /* a failed poll is cosmetic; the display keeps running */
      }
    };

    poll();
    const timer = setInterval(poll, SUMMARY_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [live, token]);

  // ── Actions ────────────────────────────────────────────────────────
  const handleLogin = useCallback(async (accessCode) => {
    const issued = await session.login(accessCode);
    setToken(issued);
  }, []);

  const handleStart = useCallback(
    async ({ label }) => {
      const data = await session.startSession({ token, label });

      noiseSpecRef.current = data.qr.noiseSpec;
      const buffer = new session.TokenBuffer({
        token,
        sessionId: data.sessionId,
        slotTtlMs: data.qr.slotTtlMs,
        batchSpanMs: data.qr.batchSpanMs,
      });
      buffer.ingest(data.tokens);
      bufferRef.current = buffer;

      setLive({ sessionId: data.sessionId, label: data.label, startedAt: data.serverTime });
      setStatus({ tone: "ok", message: "" });
    },
    [token]
  );

  const handleEnd = useCallback(async () => {
    if (!live) return;
    try {
      await session.endSession({ token, sessionId: live.sessionId });
    } catch {
      /* ending locally is still correct if the call fails */
    }
    cancelAnimationFrame(rafRef.current);
    bufferRef.current = null;
    setLive(null);
    setQrValue("");
    setSummary(null);
  }, [live, token]);

  const handleSignOut = useCallback(() => {
    session.logout();
    setToken(null);
    setLive(null);
  }, []);

  const qrNode = useMemo(
    () =>
      qrValue ? (
        <QRCodeSVG
          value={qrValue}
          // Sized by CSS to fill the projected area. The old build passed no
          // size at all and inherited the 128px default — unreadable from the
          // back of a lecture hall.
          size={1024}
          level="M"
          marginSize={2}
          className="qr-svg"
          // The value changes several times a second; announcing it would
          // make a screen reader unusable.
          aria-hidden="true"
        />
      ) : null,
    [qrValue]
  );

  if (!token) return <LoginScreen onLogin={handleLogin} />;
  if (!live) return <SetupScreen onStart={handleStart} onSignOut={handleSignOut} />;

  return (
    <LiveSession
      session={live}
      qrNode={qrNode}
      status={status}
      summary={summary}
      token={token}
      onEnd={handleEnd}
    />
  );
}
