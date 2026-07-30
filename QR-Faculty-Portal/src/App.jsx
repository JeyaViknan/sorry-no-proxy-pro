import React, { useEffect, useRef, useState } from "react";
import { QRCodeSVG } from "qrcode.react";
import "./App.css";

const DEFAULTS = {
  speedMs: 500,
  validQrCount: 4,
  durationSec: 30,
};

const buildValidScheduleMs = (flashCount, durationMs, minDelayMs, minGapMs) => {
  const maxDelayMs = Math.max(minDelayMs, durationMs - minDelayMs);
  const delays = [];
  let attempts = 0;

  while (delays.length < flashCount && attempts < 5000) {
    attempts += 1;
    const candidate = Math.floor(Math.random() * (maxDelayMs - minDelayMs + 1)) + minDelayMs;
    if (delays.every((t) => Math.abs(t - candidate) >= minGapMs)) {
      delays.push(candidate);
    }
  }

  return delays.sort((a, b) => a - b);
};

const noisePayload = () => {
  const nonce = `${Math.random().toString(36).slice(2, 11)}-${Date.now().toString(36)}`;
  return JSON.stringify({ type: "attendance-noise", nonce, ts: Date.now() });
};

const App = () => {
  const [speedMs, setSpeedMs] = useState(DEFAULTS.speedMs);
  const [validQrCount, setValidQrCount] = useState(DEFAULTS.validQrCount);
  const [durationSec, setDurationSec] = useState(DEFAULTS.durationSec);

  const [running, setRunning] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [qrValue, setQrValue] = useState("");

  const timerRef = useRef(null);
  const qrIntervalRef = useRef(null);
  const qrTimeoutRefs = useRef([]);
  const validValueRef = useRef("");
  const showValidUntilRef = useRef(0);

  const stopAll = () => {
    clearInterval(timerRef.current);
    clearInterval(qrIntervalRef.current);
    qrTimeoutRefs.current.forEach((t) => clearTimeout(t));
    timerRef.current = null;
    qrIntervalRef.current = null;
    qrTimeoutRefs.current = [];
    showValidUntilRef.current = 0;
    setRunning(false);
  };

  const tickQr = () => {
    if (showValidUntilRef.current > Date.now()) {
      setQrValue(validValueRef.current);
      return;
    }
    setQrValue(noisePayload());
  };

  const startSession = () => {
    stopAll();

    const safeSpeed = Math.max(100, Number(speedMs) || DEFAULTS.speedMs);
    const safeDurationSec = Math.max(5, Number(durationSec) || DEFAULTS.durationSec);
    const safeDurationMs = safeDurationSec * 1000;
    const safeValidCount = Math.max(1, Number(validQrCount) || DEFAULTS.validQrCount);

    const now = new Date();
    const hh = String(now.getHours()).padStart(2, "0");
    const mm = String(now.getMinutes()).padStart(2, "0");
    const validValue = `jeycavbhakanadiyaz${hh}${mm}`;
    validValueRef.current = validValue;

    const schedule = buildValidScheduleMs(safeValidCount, safeDurationMs, 3000, 3000);

    setElapsedTime(0);
    setRunning(true);
    setQrValue(noisePayload());

    qrIntervalRef.current = setInterval(tickQr, safeSpeed);
    timerRef.current = setInterval(() => {
      setElapsedTime((prev) => {
        const next = prev + 1;
        if (next >= safeDurationSec) {
          stopAll();
          return safeDurationSec;
        }
        return next;
      });
    }, 1000);

    schedule.forEach((delay) => {
      const timeoutId = setTimeout(() => {
        showValidUntilRef.current = Date.now() + Math.max(500, safeSpeed);
        setQrValue(validValueRef.current);
      }, delay);
      qrTimeoutRefs.current.push(timeoutId);
    });
  };

  const resetDefaults = () => {
    setSpeedMs(DEFAULTS.speedMs);
    setValidQrCount(DEFAULTS.validQrCount);
    setDurationSec(DEFAULTS.durationSec);
  };

  useEffect(() => stopAll, []);

  const remaining = Math.max(0, durationSec - elapsedTime);

  return (
    <main className="app-shell">
      {!running ? (
        <section className="setup-card">
          <p className="eyebrow">Faculty Attendance</p>
          <h1>Configure Session</h1>
          <p className="subtext">
            Set the QR speed, number of valid QR appearances, and total shuffle duration.
          </p>

          <div className="settings-grid">
            <label className="field">
              <span>Speed per QR (seconds)</span>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={(speedMs / 1000).toFixed(1)}
                onChange={(e) =>
                  setSpeedMs(Math.max(100, Math.round(Number(e.target.value || 0.5) * 1000)))
                }
              />
            </label>

            <label className="field">
              <span>Number of valid QRs</span>
              <input
                type="number"
                min="1"
                step="1"
                value={validQrCount}
                onChange={(e) => setValidQrCount(Math.max(1, Number(e.target.value || 1)))}
              />
            </label>

            <label className="field">
              <span>Shuffling duration (seconds)</span>
              <input
                type="number"
                min="5"
                step="1"
                value={durationSec}
                onChange={(e) => setDurationSec(Math.max(5, Number(e.target.value || 5)))}
              />
            </label>
          </div>

          <p className="defaults-note">Default: 0.5s per QR, 4 valid QRs, 30s duration</p>

          <div className="setup-actions">
            <button className="ghost-btn" onClick={resetDefaults}>
              Reset Defaults
            </button>
            <button className="primary-btn" onClick={startSession}>
              Start Session
            </button>
          </div>
        </section>
      ) : (
        <section className="session-view">
          <div className="qr-stage">
            <div className="qr-frame">
              <QRCodeSVG value={qrValue} className="qr-svg" includeMargin={false} />
            </div>
          </div>
          <aside className="session-panel">
            <p className="eyebrow">Live Session</p>
            <div className="timer-box">
              <span className="timer-label">Time Left</span>
              <span className="timer-value">{remaining}s</span>
            </div>
            <div className="meta-line">
              Elapsed: <strong>{elapsedTime}s</strong>
            </div>
            <button className="danger-btn" onClick={stopAll}>
              End Session
            </button>
          </aside>
        </section>
      )}
    </main>
  );
};

export default App;
