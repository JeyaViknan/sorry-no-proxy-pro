/**
 * Local persistence.
 *
 * Two things are stored, both to remove work from the student:
 *   • deviceId          — stable identifier used for rate-limit fairness and
 *                         attendance-token binding. NOT a security boundary;
 *                         the server treats it as untrusted.
 *   • registerNumber    — so a returning student types nothing at all. This
 *                         is the single biggest UX win available: after the
 *                         first class the flow becomes scan → submit.
 *
 * Everything degrades gracefully when storage is unavailable — Safari private
 * mode, or a browser with cookies/storage blocked, both throw on access. The
 * old code would have taken an uncaught exception at startup.
 */

const DEVICE_KEY = "snp.deviceId";
const REGNO_KEY = "snp.registerNumber";

/** In-memory fallback so a session still works with storage disabled. */
const memory = new Map();

function safeGet(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return memory.get(key) ?? null;
  }
}

function safeSet(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    memory.set(key, value);
  }
}

function safeRemove(key) {
  try {
    window.localStorage.removeItem(key);
  } catch {
    memory.delete(key);
  }
}

function randomId() {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Stable per-browser identifier, created on first use. */
export function getDeviceId() {
  let id = safeGet(DEVICE_KEY);
  if (!id || id.length < 8) {
    id = randomId();
    safeSet(DEVICE_KEY, id);
  }
  return id;
}

export function getRegisterNumber() {
  return safeGet(REGNO_KEY) || "";
}

export function setRegisterNumber(value) {
  const normalised = String(value || "").trim().toUpperCase();
  if (normalised) safeSet(REGNO_KEY, normalised);
  else safeRemove(REGNO_KEY);
}

export function clearRegisterNumber() {
  safeRemove(REGNO_KEY);
}
