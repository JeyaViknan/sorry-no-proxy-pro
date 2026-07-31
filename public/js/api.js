/**
 * Network layer, built for an unreliable lecture-hall connection.
 *
 * WHAT THE OLD CODE DID
 * ---------------------
 * Two bare `fetch()` calls with no timeout, no retry and no offline handling.
 * Combined with a serialised Python queue on the server, a student's request
 * could hang indefinitely while the UI showed a static "Processing..." line.
 * That is the "app is frozen" experience, and it was entirely avoidable.
 *
 * WHAT THIS DOES
 * --------------
 *  • Every request has a deadline. Nothing hangs forever.
 *  • Transient failures (network drop, 5xx, 429) retry with exponential
 *    backoff plus jitter. Jitter matters here specifically: when 200 phones
 *    on one access point all fail at the same moment, un-jittered retries
 *    re-collide in lockstep and the recovery is worse than the outage.
 *  • Client errors (4xx) never retry — they will not succeed.
 *  • Retrying POST /api/attendance is safe because the endpoint is
 *    idempotent: a duplicate returns the existing record rather than
 *    double-recording. That property was designed for exactly this.
 *  • Server clock offset is tracked so the client can reason about token
 *    expiry using the server's notion of time, not the phone's.
 */

const DEFAULT_TIMEOUT_MS = 12000;
const MAX_ATTEMPTS = 3;
const BASE_BACKOFF_MS = 400;

/** Errors the caller can present, distinct from bugs. */
export class ApiError extends Error {
  constructor(code, message, { status = 0, retryable = false, payload = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.retryable = retryable;
    this.payload = payload;
  }
}

/** Milliseconds between the server's clock and this device's. */
let clockOffsetMs = 0;

export const serverNow = () => Date.now() + clockOffsetMs;
export const getClockOffsetMs = () => clockOffsetMs;

function noteServerTime(serverTime) {
  if (typeof serverTime !== "number" || !Number.isFinite(serverTime)) return;
  // Single sample, so it includes roughly half the round trip. Good enough:
  // this is used for "is my token about to expire", not for signing.
  clockOffsetMs = serverTime - Date.now();
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function backoffDelay(attempt) {
  const exponential = BASE_BACKOFF_MS * 2 ** (attempt - 1);
  // Full jitter. Prevents a whole classroom retrying in lockstep.
  return Math.random() * Math.min(exponential, 5000);
}

async function parseJson(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    throw new ApiError("BAD_RESPONSE", "The server sent an unreadable response.", {
      status: response.status,
      retryable: true,
    });
  }
}

/**
 * One HTTP attempt.
 * @param {string} path
 * @param {{method?: string, body?: object, timeoutMs?: number, signal?: AbortSignal}} options
 */
async function attempt(path, { method = "GET", body, timeoutMs, signal } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error("timeout")), timeoutMs);

  // Let a caller-supplied signal (e.g. user navigated away) cancel too.
  const onAbort = () => controller.abort(signal?.reason);
  signal?.addEventListener("abort", onAbort, { once: true });

  try {
    const response = await fetch(path, {
      method,
      headers: {
        ...(body ? { "Content-Type": "application/json" } : {}),
        // See the note in the faculty client: suppresses ngrok's HTML
        // interstitial so API responses are always JSON. No-op elsewhere.
        "ngrok-skip-browser-warning": "true",
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
      // Never let a stale cached response stand in for a live answer.
      cache: "no-store",
      keepalive: false,
    });

    const data = await parseJson(response);
    noteServerTime(data.serverTime);

    if (response.ok) return data;

    // 5xx and 429 are worth another go; 4xx are not.
    const retryable = response.status >= 500 || response.status === 429;
    throw new ApiError(
      data.code || `HTTP_${response.status}`,
      data.message || "The server could not complete that request.",
      { status: response.status, retryable, payload: data }
    );
  } catch (error) {
    if (error instanceof ApiError) throw error;

    if (error.name === "AbortError") {
      throw new ApiError("TIMEOUT", "The network is slow. Retrying…", { retryable: true });
    }
    // TypeError from fetch means the request never reached the server.
    throw new ApiError("OFFLINE", "Cannot reach the server.", { retryable: true });
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}

/**
 * Request with retry. `onRetry` lets the UI say "retrying…" instead of
 * looking frozen, which is the difference between a slow app and a broken one.
 */
export async function request(path, options = {}) {
  const {
    attempts = MAX_ATTEMPTS,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    onRetry = null,
    ...rest
  } = options;

  let lastError;

  for (let n = 1; n <= attempts; n += 1) {
    try {
      return await attempt(path, { ...rest, timeoutMs });
    } catch (error) {
      lastError = error;
      if (!error.retryable || n === attempts) break;

      const delay = backoffDelay(n);
      onRetry?.({ attempt: n, of: attempts, delayMs: delay, error });
      await sleep(delay);
    }
  }

  throw lastError;
}

// ── Endpoint helpers ─────────────────────────────────────────────────

/**
 * Warm the connection and learn the server clock while the camera is still
 * initialising. Fires DNS, TCP and TLS early so the first *meaningful*
 * request does not pay for them — worth 200-400ms on a cold mobile
 * connection, entirely off the critical path.
 */
export function prefetchHello({ signal } = {}) {
  return request("/api/hello", {
    attempts: 2,
    timeoutMs: 5000,
    signal,
  }).catch(() => null); // best effort; never block startup on it
}

export function validateQr({ payload, deviceId, onRetry, signal }) {
  return request("/api/qr/validate", {
    method: "POST",
    body: { payload, deviceId },
    // Short deadline: the payload is only valid for ~4s, so a slow response
    // is worthless anyway. Failing fast lets the scanner try the next frame.
    timeoutMs: 6000,
    attempts: 2,
    onRetry,
    signal,
  });
}

export function submitAttendance({ attendanceToken, deviceId, registerNumber, frames, onRetry, signal }) {
  return request("/api/attendance", {
    method: "POST",
    body: { attendanceToken, deviceId, registerNumber, frames },
    // Generous: this waits on face verification behind a possible queue.
    timeoutMs: 25000,
    attempts: 3,
    onRetry,
    signal,
  });
}

// ── Connectivity ─────────────────────────────────────────────────────

/**
 * `navigator.onLine` only reports whether an interface exists — a phone
 * associated with a captive-portal access point reports true while nothing
 * routes. Treat it as a hint that can prove offline, never as proof of
 * online.
 */
export function isDefinitelyOffline() {
  return navigator.onLine === false;
}

export function onConnectivityChange(handler) {
  const online = () => handler(true);
  const offline = () => handler(false);
  window.addEventListener("online", online);
  window.addEventListener("offline", offline);
  return () => {
    window.removeEventListener("online", online);
    window.removeEventListener("offline", offline);
  };
}

/**
 * Effective connection quality, where the browser exposes it. Used to decide
 * how many burst frames to upload: sending three 100KB frames over 2G costs
 * more in latency than the accuracy is worth.
 */
export function connectionProfile() {
  const connection =
    navigator.connection || navigator.mozConnection || navigator.webkitConnection;

  if (!connection) return { tier: "unknown", frames: 3 };

  const type = connection.effectiveType || "4g";
  if (connection.saveData) return { tier: "save-data", frames: 1 };

  switch (type) {
    case "slow-2g":
    case "2g":
      return { tier: type, frames: 1 };
    case "3g":
      return { tier: type, frames: 2 };
    default:
      return { tier: type, frames: 3 };
  }
}
