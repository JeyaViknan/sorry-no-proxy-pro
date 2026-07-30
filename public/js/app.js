/**
 * Student scanner — orchestration.
 *
 * FLOW
 *   intro  -> tap Start (the user gesture the camera needs)
 *   scan   -> rear camera, decode continuously, relay candidates to the server
 *   capture-> front camera, live quality guidance, registration number
 *   verify -> freeze the exact frame, upload a burst, show real progress
 *   result -> confirm, or offer a retry that actually works
 *
 * The old page was a single 356-line HTML file with 310 lines of inline
 * script and six module-level mutable globals. Splitting it into modules is
 * not cosmetic: camera lifecycle, network retry and capture quality each had
 * bugs that were invisible when tangled together.
 */

import * as api from "./api.js";
import * as store from "./storage.js";
import * as ui from "./ui.js";
import { Camera, CameraError, checkSupport, watchVisibility } from "./camera.js";
import { QrScanner, looksLikeAttendancePayload } from "./scanner.js";
import { assessLiveFrame, captureBurst, freezeInto } from "./capture.js";

const REGNO_PATTERN = /^[0-9]{2}[A-Z]{3}[0-9]{4}$/;
const QUALITY_POLL_MS = 350;
/** Re-validate rather than submit if the token is about to expire. */
const TOKEN_REFRESH_MARGIN_MS = 15000;

const state = {
  deviceId: store.getDeviceId(),
  camera: null,
  scanner: null,
  attendanceToken: null,
  tokenExpiresAt: 0,
  session: null,
  qualityTimer: null,
  stopVisibilityWatch: null,
  wakeLock: null,
  inFlight: false,
  // Payloads already rejected this session. The display shows mostly decoys,
  // and without this the same one would be re-sent every frame.
  rejectedPayloads: new Set(),
};

// ── Helpers ──────────────────────────────────────────────────────────

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function tokenIsUsable() {
  return (
    state.attendanceToken && api.serverNow() < state.tokenExpiresAt - TOKEN_REFRESH_MARGIN_MS
  );
}

function stopQualityPolling() {
  if (state.qualityTimer) {
    clearInterval(state.qualityTimer);
    state.qualityTimer = null;
  }
}

async function teardown() {
  stopQualityPolling();
  state.scanner?.destroy();
  state.scanner = null;
  await state.camera?.stop();
  state.stopVisibilityWatch?.();
  try {
    await state.wakeLock?.release();
  } catch {
    /* already released */
  }
}

// ── Screen: intro ────────────────────────────────────────────────────

function initIntro() {
  const support = checkSupport();
  if (!support.supported) {
    ui.showResult({
      title: "This browser cannot be used",
      detail: support.message,
      tone: "error",
      actions: [{ label: "Reload", primary: true, onClick: () => location.reload() }],
    });
    return;
  }

  // Warm DNS/TCP/TLS and learn the server clock while the student is still
  // reading the intro. Entirely off the critical path.
  api.prefetchHello();

  const saved = store.getRegisterNumber();
  if (saved) {
    ui.$("#regno-input").value = saved;
    ui.$("#intro-saved").hidden = false;
    ui.$("#intro-saved-value").textContent = saved;
  }

  ui.$("#start-btn").addEventListener("click", startScanning, { once: false });
  ui.showScreen("intro");
}

// ── Screen: scanning ─────────────────────────────────────────────────

async function startScanning() {
  const button = ui.$("#start-btn");
  ui.setBusy(button, true, "Starting camera…");

  try {
    state.wakeLock = await ui.acquireWakeLock();

    state.camera = new Camera(ui.$("#scan-video"));
    state.camera.onInterrupted = () => {
      ui.showBanner("The camera stopped. Reopening…", { tone: "warn" });
      state.camera.switchTo(state.camera.facingMode || "environment").catch(() => {});
    };

    await state.camera.start("environment", { width: 1280, height: 720 });

    state.stopVisibilityWatch = watchVisibility(state.camera, {
      onLost: () => ui.showBanner("Reconnecting to the camera…", { tone: "warn" }),
      onRecover: () => ui.hideBanner(),
    });

    state.scanner = new QrScanner();
    await state.scanner.prepare();

    ui.showScreen("scan");
    ui.setScanStatus("Point at the screen at the front of the room");
    ui.setBusy(button, false);

    await state.scanner.start(ui.$("#scan-video"), onDecoded);
  } catch (error) {
    ui.setBusy(button, false);
    handleCameraError(error, startScanning);
  }
}

/**
 * Called for every decoded string. Most are decoys by design — the faculty
 * display shows far more invalid codes than valid ones — so this must stay
 * quiet and cheap rather than treating each as a failure.
 */
async function onDecoded(text) {
  if (state.inFlight) return;
  if (!looksLikeAttendancePayload(text)) return; // some unrelated QR
  if (state.rejectedPayloads.has(text)) return; // already ruled out

  state.inFlight = true;
  try {
    const result = await api.validateQr({
      payload: text,
      deviceId: state.deviceId,
      onRetry: () => ui.setScanStatus("Network is slow, retrying…", "warn"),
    });

    if (!result.valid) {
      // Expected for a decoy. Remember it so we never spend another request
      // on the same string, and say nothing to the student — showing
      // "Invalid QR" hundreds of times per session, as the old UI did, only
      // teaches them to distrust the app.
      state.rejectedPayloads.add(text);
      if (result.code === "SESSION_ENDED") {
        state.scanner.stop();
        ui.showResult({
          title: "Session ended",
          detail: "This attendance session is closed. Please speak to your faculty member.",
          tone: "warn",
          actions: [{ label: "Start over", primary: true, onClick: () => location.reload() }],
        });
      }
      return;
    }

    state.attendanceToken = result.attendanceToken;
    state.tokenExpiresAt = result.expiresAt;
    state.session = result.session;

    ui.vibrate(35);
    ui.setScanStatus("Verified", "success");
    state.scanner.stop();
    await goToCapture();
  } catch (error) {
    if (error.code === "OFFLINE" || error.code === "TIMEOUT") {
      ui.setScanStatus("Waiting for the network…", "warn");
    } else if (error.code?.startsWith("RATE_LIMITED")) {
      ui.setScanStatus(error.message, "warn");
      await sleep(3000);
    }
  } finally {
    state.inFlight = false;
  }
}

// ── Screen: capture ──────────────────────────────────────────────────

async function goToCapture() {
  ui.showScreen("capture");
  ui.$("#session-label").textContent = state.session?.label || "";

  const input = ui.$("#regno-input");
  const saved = store.getRegisterNumber();
  if (saved) input.value = saved;

  try {
    // One camera owner, so this is a plain awaited switch — no cross-library
    // contention, which is what made the old handoff fail unpredictably.
    await state.camera.switchTo("user", { width: 1280, height: 720 });
    // Mirror the preview: people expect a selfie view to behave like a mirror.
    ui.$("#capture-video").classList.add("mirrored");
  } catch (error) {
    handleCameraError(error, goToCapture);
    return;
  }

  startQualityPolling();
  validateRegnoInput();

  if (!saved) input.focus();
}

/**
 * Live camera guidance. This is the change that converts most failures into
 * a two-second self-correction instead of a server rejection the student
 * cannot interpret.
 */
function startQualityPolling() {
  stopQualityPolling();

  state.qualityTimer = setInterval(() => {
    if (!state.camera?.isLive) return;
    let assessment;
    try {
      assessment = assessLiveFrame(state.camera);
    } catch {
      return;
    }

    ui.setCameraHint(assessment.message, assessment.ok ? "success" : "warn");
    ui.$("#face-guide").dataset.state = assessment.ok ? "ready" : "adjust";
    validateRegnoInput(assessment.ok);
  }, QUALITY_POLL_MS);
}

function validateRegnoInput(cameraOk = null) {
  const input = ui.$("#regno-input");
  const value = input.value.trim().toUpperCase();
  const valid = REGNO_PATTERN.test(value);

  const error = ui.$("#regno-error");
  const touched = input.dataset.touched === "true";
  error.hidden = !(touched && value.length > 0 && !valid);
  input.setAttribute("aria-invalid", String(touched && value.length > 0 && !valid));

  // Camera readiness is advisory. Blocking Submit on it would strand anyone
  // whose lighting the heuristic dislikes but whose face the model would
  // still match — the server is the authority.
  ui.setSubmitEnabled(valid);
  return valid;
}

// ── Screen: verify ───────────────────────────────────────────────────

async function submit() {
  if (!validateRegnoInput()) {
    ui.$("#regno-input").dataset.touched = "true";
    validateRegnoInput();
    ui.$("#regno-input").focus();
    return;
  }

  const registerNumber = ui.$("#regno-input").value.trim().toUpperCase();
  store.setRegisterNumber(registerNumber);

  // Token may have expired while the student typed.
  if (!tokenIsUsable()) {
    ui.showBanner("Your scan expired. Scan the code again.", { tone: "warn", assertive: true });
    await returnToScanning();
    return;
  }

  stopQualityPolling();
  ui.showScreen("verify");
  ui.setProgress("Capturing…");

  try {
    // Freeze first, so the student sees the exact frame being evaluated
    // rather than the blank screen the old flow showed for 5-15 seconds.
    freezeInto(ui.$("#freeze-canvas"), state.camera);

    const profile = api.connectionProfile();
    const burst = await captureBurst(state.camera, {
      frames: profile.frames,
      onProgress: (n, total) => ui.setProgress(`Capturing… ${n}/${total}`),
    });

    // The camera has done its job; release it before the network wait so the
    // sensor is not held open during a slow upload.
    await state.camera.stop();

    ui.setProgress(
      profile.frames < 3 ? "Checking… (slow connection detected)" : "Checking your face…"
    );

    const result = await api.submitAttendance({
      attendanceToken: state.attendanceToken,
      deviceId: state.deviceId,
      registerNumber,
      frames: burst.frames,
      onRetry: ({ attempt, of }) =>
        ui.setProgress(`Network is slow — retrying (${attempt}/${of})…`),
    });

    showSuccess(result, registerNumber);
  } catch (error) {
    handleSubmitError(error, registerNumber);
  }
}

function showSuccess(result, registerNumber) {
  ui.vibrate([40, 60, 40]);

  const flagged = result.status === "flagged";
  ui.showResult({
    title: result.alreadyRecorded ? "Already marked present" : "You are marked present",
    detail: flagged
      ? `${registerNumber} — recorded, and flagged for your faculty member to confirm.`
      : `${registerNumber} — ${state.session?.label || "attendance recorded"}.`,
    tone: flagged ? "warn" : "success",
    actions: [{ label: "Done", primary: true, onClick: () => window.close() }],
  });
  teardown();
}

/**
 * Error handling with a real retry path.
 *
 * The old code ended the session on any verification failure ("DO NOT restore
 * video or allow retries") to deter brute force. It deterred nothing — there
 * was no server-side limit and reloading reset it — while permanently
 * locking out students whose single frame happened to be blurry. Retries are
 * now bounded server-side by a per-registration-number counter, so the
 * client can afford to be helpful.
 */
async function handleSubmitError(error, registerNumber) {
  const payload = error.payload || {};

  // A rejection the student can act on.
  if (payload.status === "rejected" || error.status === 200) {
    const canRetry = payload.canRetry !== false;
    ui.showResult({
      title: payload.code === "QUALITY" ? "Could not read your face" : "Face did not match",
      detail:
        payload.message +
        (canRetry && payload.attemptsRemaining
          ? ` (${payload.attemptsRemaining} attempt${payload.attemptsRemaining === 1 ? "" : "s"} left)`
          : ""),
      tone: "warn",
      actions: canRetry
        ? [
            { label: "Try again", primary: true, onClick: () => retryCapture(registerNumber) },
            { label: "Start over", onClick: () => location.reload() },
          ]
        : [{ label: "Start over", primary: true, onClick: () => location.reload() }],
    });
    return;
  }

  const recoverable = ["OFFLINE", "TIMEOUT", "VERIFIER_UNAVAILABLE", "BAD_RESPONSE"];
  if (recoverable.includes(error.code)) {
    ui.showResult({
      title: "Could not reach the server",
      detail:
        "Your attendance was not submitted. Check your connection and try again — nothing was lost.",
      tone: "warn",
      actions: [
        { label: "Try again", primary: true, onClick: () => retryCapture(registerNumber) },
        { label: "Start over", onClick: () => location.reload() },
      ],
    });
    return;
  }

  if (error.code?.startsWith("TOKEN_")) {
    ui.showResult({
      title: "Your scan expired",
      detail: "Scan the code on the screen again — it changes every few seconds.",
      tone: "warn",
      actions: [{ label: "Scan again", primary: true, onClick: returnToScanning }],
    });
    return;
  }

  ui.showResult({
    title: "Something went wrong",
    detail: error.message || "Please try again, or speak to your faculty member.",
    tone: "error",
    actions: [{ label: "Start over", primary: true, onClick: () => location.reload() }],
  });
}

/** Reopen the front camera and let the student re-submit. */
async function retryCapture(registerNumber) {
  if (!tokenIsUsable()) {
    await returnToScanning();
    return;
  }

  ui.showScreen("capture");
  ui.$("#regno-input").value = registerNumber;

  try {
    await state.camera.start("user", { width: 1280, height: 720 });
    startQualityPolling();
    validateRegnoInput();
  } catch (error) {
    handleCameraError(error, () => retryCapture(registerNumber));
  }
}

async function returnToScanning() {
  state.attendanceToken = null;
  state.tokenExpiresAt = 0;
  state.rejectedPayloads.clear();

  ui.showScreen("scan");
  ui.setScanStatus("Point at the screen at the front of the room");

  try {
    await state.camera.switchTo("environment", { width: 1280, height: 720 });
    if (!state.scanner) {
      state.scanner = new QrScanner();
      await state.scanner.prepare();
    }
    await state.scanner.start(ui.$("#scan-video"), onDecoded);
  } catch (error) {
    handleCameraError(error, returnToScanning);
  }
}

function handleCameraError(error, retryAction) {
  const isCameraError = error instanceof CameraError;
  const message = isCameraError ? error.message : "The camera could not be started.";

  ui.showResult({
    title: error.code === "PERMISSION_DENIED" ? "Camera permission needed" : "Camera problem",
    detail: message,
    tone: "error",
    actions: [
      ...(isCameraError && error.recoverable && retryAction
        ? [{ label: "Try again", primary: true, onClick: retryAction }]
        : []),
      { label: "Reload", primary: !isCameraError, onClick: () => location.reload() },
    ],
  });
}

// ── Wiring ───────────────────────────────────────────────────────────

function init() {
  ui.registerScreens(document);

  const input = ui.$("#regno-input");
  input.addEventListener("input", () => {
    // Uppercase as they type; the server normalises anyway but seeing the
    // canonical form avoids "did I type it wrong?" hesitation.
    const start = input.selectionStart;
    input.value = input.value.toUpperCase();
    input.setSelectionRange(start, start);
    validateRegnoInput();
  });
  input.addEventListener("blur", () => {
    input.dataset.touched = "true";
    validateRegnoInput();
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !ui.$("#submit-btn").disabled) submit();
  });

  ui.$("#submit-btn").addEventListener("click", submit);

  // Offline is a standing condition, not a passing event, so its banner is
  // sticky and survives screen transitions until connectivity returns.
  const offlineBanner = () =>
    ui.showBanner("No internet connection. Waiting to reconnect…", {
      tone: "error",
      sticky: true,
    });

  api.onConnectivityChange((online) => {
    if (online) ui.hideBanner();
    else offlineBanner();
  });
  if (api.isDefinitelyOffline()) offlineBanner();

  window.addEventListener("pagehide", teardown);

  initIntro();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
