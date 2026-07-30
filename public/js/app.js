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
import { QrScanner, looksLikeAttendancePayload, warmScannerBackend } from "./scanner.js";
import {
  assessLiveFrame,
  captureBurst,
  computeCaptureRegion,
  showFrozenFrame,
} from "./capture.js";

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
  // Backoff counter. If the faculty display's token buffer runs dry it shows
  // decoys continuously; without this every phone in the room would POST one
  // every ~400ms, amplifying the very outage that caused it.
  consecutiveInvalid: 0,
  backoffUntil: 0,
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
  // On browsers without BarcodeDetector this downloads the 368KB decoder now,
  // overlapping it with reading time instead of with "Starting camera…".
  warmScannerBackend();

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
      const facing = state.camera.facingMode || "environment";
      state.camera
        .switchTo(facing, {
          video: facing === "user" ? ui.$("#capture-video") : ui.$("#scan-video"),
        })
        .catch(() => {});
    };

    await state.camera.start("environment", {
      width: 1280,
      height: 720,
      video: ui.$("#scan-video"),
    });

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
  if (api.serverNow() < state.backoffUntil) return;

  state.inFlight = true;
  try {
    const result = await api.validateQr({
      payload: text,
      deviceId: state.deviceId,
      onRetry: () => ui.setScanStatus("Network is slow, retrying…", "warn"),
    });

    if (!result.valid) {
      state.consecutiveInvalid += 1;
      // Normally the display shows a currently-valid code, so the first scan
      // succeeds. A long run of invalid results means the faculty token
      // buffer has run dry — i.e. the server is already struggling. Backing
      // off stops several hundred phones hammering it every ~400ms and
      // amplifying the outage.
      if (state.consecutiveInvalid >= 5) {
        const wait = Math.min(1000 * 2 ** (state.consecutiveInvalid - 5), 15000);
        state.backoffUntil = api.serverNow() + wait;
        ui.setScanStatus("Waiting for the next code…", "warn");
      }
      // Expected for a decoy. Remember it so we never spend another request
      // on the same string, and say nothing to the student — showing
      // "Invalid QR" hundreds of times per session, as the old UI did, only
      // teaches them to distrust the app.
      if (state.rejectedPayloads.size > 500) state.rejectedPayloads.clear();
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

    state.consecutiveInvalid = 0;
    state.backoffUntil = 0;
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
    // Re-bind to the capture screen's own <video>. Without the explicit
    // element the stream stays attached to the (now hidden) scan element and
    // this screen renders black.
    await state.camera.switchTo("user", {
      width: 1280,
      height: 720,
      video: ui.$("#capture-video"),
    });
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
  }, QUALITY_POLL_MS);
}

function validateRegnoInput() {
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

  // Token may have expired while the student typed.
  if (!tokenIsUsable()) {
    ui.showBanner("Your scan expired. Scan the code again.", { tone: "warn", assertive: true });
    await returnToScanning();
    return;
  }

  // Geometry MUST be resolved while the capture screen is still visible —
  // getBoundingClientRect() on a hidden element is 0x0, which silently
  // collapses the crop to the full sensor frame and uploads a region the
  // student never framed.
  const region = computeCaptureRegion(state.camera);
  if (!region) {
    ui.showBanner("The camera is not ready yet. Give it a moment.", { tone: "warn" });
    return;
  }

  stopQualityPolling();
  ui.setSubmitEnabled(false);
  ui.setCameraHint("Hold still…", "neutral");

  try {
    // Capture while the preview is still on screen, so the student is looking
    // at the camera for the whole burst rather than at a transition.
    const profile = api.connectionProfile();
    const burst = await captureBurst(state.camera, {
      frames: profile.frames,
      region,
      onProgress: (n, total) => ui.setCameraHint(`Capturing ${n}/${total}…`, "neutral"),
    });

    // Now switch, showing the exact frame being judged.
    ui.showScreen("verify");
    // The live preview is mirrored (people expect a selfie to behave like a
    // mirror), so mirror the freeze too — otherwise the student's face
    // visibly flips at the moment of capture, which reads as a glitch. Only
    // the DISPLAY is flipped; the uploaded pixels stay unmirrored.
    ui.$("#freeze-canvas").classList.add("mirrored");
    showFrozenFrame(ui.$("#freeze-canvas"), burst.best);

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

    // A face mismatch is returned as HTTP 200 with ok:false, because it is a
    // legitimate answer rather than a transport failure. api.js therefore
    // resolves rather than throws, so this branch is REQUIRED — without it a
    // rejected student is shown "You are marked present", which defeats the
    // entire system. Automated tests missed this because the verifier returns
    // 503 (which does throw) when no model is loaded.
    if (result.ok === false || result.status === "rejected") {
      showRejection(result, registerNumber);
      return;
    }

    showSuccess(result, registerNumber);
  } catch (error) {
    handleSubmitError(error, registerNumber);
  }
}

function showSuccess(result, registerNumber) {
  // Only the register number is persisted, and only now that it has been
  // confirmed against a real face. Saving it earlier would remember a typo.
  store.setRegisterNumber(registerNumber);
  ui.vibrate([40, 60, 40]);

  const flagged = result.status === "flagged";
  ui.showResult({
    title: result.alreadyRecorded ? "Already marked present" : "You are marked present",
    detail: flagged
      ? `${registerNumber} — recorded, and flagged for your faculty member to confirm.`
      : `${registerNumber} — ${state.session?.label || "attendance recorded"}.`,
    tone: flagged ? "warn" : "success",
    // window.close() only works for script-opened windows, so it silently does
    // nothing here. Reloading back to the start is honest and actually useful
    // on a shared device.
    actions: [{ label: "Done", primary: true, onClick: () => location.reload() }],
  });
  teardown();
}

/**
 * A verification that completed and said no.
 *
 * Distinguishes "we could not read the photo" from "that is not you": the
 * first is recoverable in two seconds by moving into better light, the second
 * is not. Collapsing them, as the old build did, is why legitimate students
 * were told to go and speak to the professor.
 */
function showRejection(result, registerNumber) {
  ui.vibrate(200);

  const isQuality = result.code === "QUALITY" || result.code === "NO_FACE";
  const remaining = result.attemptsRemaining;
  const canRetry = result.canRetry !== false;

  const suffix =
    canRetry && typeof remaining === "number" && remaining > 0
      ? ` (${remaining} attempt${remaining === 1 ? "" : "s"} left)`
      : "";

  ui.showResult({
    title: isQuality ? "Could not read your face" : "That does not match",
    detail: `${result.message || "Please try again."}${suffix}`,
    tone: "warn",
    actions: canRetry
      ? [
          { label: "Try again", primary: true, onClick: () => retryCapture(registerNumber) },
          { label: "Start over", onClick: () => location.reload() },
        ]
      : [{ label: "Start over", primary: true, onClick: () => location.reload() }],
  });
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

  if (error.message === "CAPTURE_INTERRUPTED") {
    ui.showResult({
      title: "Capture interrupted",
      detail: "The app was closed or backgrounded mid-photo. Nothing was submitted — try again.",
      tone: "warn",
      actions: [
        { label: "Try again", primary: true, onClick: () => retryCapture(registerNumber) },
        { label: "Start over", onClick: () => location.reload() },
      ],
    });
    return;
  }

  if (error.message === "CAMERA_NOT_READY") {
    ui.showResult({
      title: "Camera not ready",
      detail: "The camera stopped before the photo was taken. Try again.",
      tone: "warn",
      actions: [
        { label: "Try again", primary: true, onClick: () => retryCapture(registerNumber) },
        { label: "Start over", onClick: () => location.reload() },
      ],
    });
    return;
  }

  // The attempt cap is enforced server-side and arrives as 403.
  if (error.code === "TOO_MANY_ATTEMPTS") {
    ui.showResult({
      title: "Too many attempts",
      detail: payload.message || error.message,
      tone: "error",
      actions: [{ label: "Start over", primary: true, onClick: () => location.reload() }],
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
    await state.camera.start("user", {
      width: 1280,
      height: 720,
      video: ui.$("#capture-video"),
    });
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
  state.consecutiveInvalid = 0;
  state.backoffUntil = 0;

  ui.showScreen("scan");
  ui.setScanStatus("Point at the screen at the front of the room");

  try {
    await state.camera.switchTo("environment", {
      width: 1280,
      height: 720,
      video: ui.$("#scan-video"),
    });
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
