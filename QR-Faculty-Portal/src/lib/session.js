/**
 * Faculty session client — talks to the attendance backend.
 *
 * THE CHANGE
 * ----------
 * The old App.jsx built the valid payload itself:
 *
 *     const validValue = `jeycavbhakanadiyaz${hh}${mm}`;
 *
 * The same secret was duplicated in the student scanner, so two independent
 * implementations had to agree — and both shipped the secret to anyone who
 * opened DevTools. This module holds no secret at all. It asks the server for
 * pre-signed payloads and simply displays them on schedule.
 *
 * BATCHING (network reliability)
 * ------------------------------
 * The display rotates every ~500ms. Fetching per frame would stall visibly on
 * lecture-hall wifi. Instead we fetch ~60s of signed payloads at a time and
 * refill well before running out, so a brief outage is invisible to the room.
 */

const TOKEN_STORAGE_KEY = "snp.facultyToken";
const REFILL_MARGIN_MS = 20000;

export class SessionError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

/** Server time minus device time, learned from every response. */
let clockOffsetMs = 0;
export const serverNow = () => Date.now() + clockOffsetMs;

async function call(path, { method = "GET", body, token, timeoutMs = 10000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(path, {
      method,
      headers: {
        ...(body ? { "Content-Type": "application/json" } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
      cache: "no-store",
    });

    const text = await response.text();
    const data = text ? JSON.parse(text) : {};

    if (typeof data.serverTime === "number") {
      clockOffsetMs = data.serverTime - Date.now();
    }
    if (!response.ok) {
      throw new SessionError(data.code || `HTTP_${response.status}`, data.message || "Request failed");
    }
    return data;
  } catch (error) {
    if (error instanceof SessionError) throw error;
    if (error.name === "AbortError") {
      throw new SessionError("TIMEOUT", "The server did not respond. Check the connection.");
    }
    throw new SessionError("OFFLINE", "Cannot reach the attendance server.");
  } finally {
    clearTimeout(timer);
  }
}

// ── Auth ─────────────────────────────────────────────────────────────

export function storedToken() {
  try {
    return sessionStorage.getItem(TOKEN_STORAGE_KEY) || null;
  } catch {
    return null;
  }
}

function storeToken(token) {
  try {
    // sessionStorage, not localStorage: a faculty token should not outlive
    // the browser tab on a shared classroom machine.
    if (token) sessionStorage.setItem(TOKEN_STORAGE_KEY, token);
    else sessionStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch {
    /* storage disabled — token stays in memory only */
  }
}

export async function login(accessCode) {
  const data = await call("/api/faculty/login", {
    method: "POST",
    body: { accessCode },
  });
  storeToken(data.facultyToken);
  return data.facultyToken;
}

export function logout() {
  storeToken(null);
}

// ── Session ──────────────────────────────────────────────────────────

export async function startSession({ token, label }) {
  return call("/api/sessions", { method: "POST", body: { label }, token });
}

export async function fetchTokens({ token, sessionId, from, span }) {
  const params = new URLSearchParams();
  if (from) params.set("from", String(Math.round(from)));
  if (span) params.set("span", String(Math.round(span)));

  return call(`/api/sessions/${sessionId}/tokens?${params}`, { token });
}

export async function fetchSummary({ token, sessionId }) {
  return call(`/api/sessions/${sessionId}/summary`, { token, timeoutMs: 8000 });
}

export async function endSession({ token, sessionId }) {
  return call(`/api/sessions/${sessionId}/end`, { method: "POST", token });
}

export function exportUrl(sessionId) {
  return `/api/sessions/${sessionId}/export`;
}

// ── Token buffer ─────────────────────────────────────────────────────

/**
 * Holds signed payloads and hands out whichever is valid right now.
 *
 * Refills ahead of exhaustion rather than on empty, so a slow or dropped
 * request has ~20s of runway to recover in. If it never recovers the display
 * keeps showing decoys — students simply see no valid code, which is the
 * correct failure mode: never a false accept, only a delay.
 */
export class TokenBuffer {
  constructor({ token, sessionId, slotTtlMs, batchSpanMs }) {
    this.token = token;
    this.sessionId = sessionId;
    this.slotTtlMs = slotTtlMs;
    this.batchSpanMs = batchSpanMs;

    this.tokens = [];
    this.refilling = false;
    this.lastError = null;
    this.refillCount = 0;
  }

  ingest(tokens) {
    const now = serverNow();
    const known = new Set(this.tokens.map((t) => t.slot));
    for (const token of tokens) {
      if (!known.has(token.slot)) this.tokens.push(token);
    }
    // Drop anything already past, and keep the list ordered.
    this.tokens = this.tokens
      .filter((t) => t.notAfter > now)
      .sort((a, b) => a.slot - b.slot);
  }

  /** The payload valid at this instant, or null. */
  current() {
    const now = serverNow();
    return this.tokens.find((t) => now >= t.notBefore && now < t.notAfter) || null;
  }

  /** How much signed runway remains. */
  runwayMs() {
    const now = serverNow();
    const last = this.tokens[this.tokens.length - 1];
    return last ? Math.max(0, last.notAfter - now) : 0;
  }

  needsRefill() {
    return !this.refilling && this.runwayMs() < REFILL_MARGIN_MS;
  }

  async refill() {
    if (this.refilling) return;
    this.refilling = true;

    try {
      const from = serverNow() + this.runwayMs();
      const data = await fetchTokens({
        token: this.token,
        sessionId: this.sessionId,
        from,
        span: this.batchSpanMs,
      });
      this.ingest(data.tokens);
      this.lastError = null;
      this.refillCount += 1;
    } catch (error) {
      this.lastError = error;
      throw error;
    } finally {
      this.refilling = false;
    }
  }
}

// ── Decoys ───────────────────────────────────────────────────────────

const BASE32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
const BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

function randomFrom(alphabet, length) {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  let output = "";
  for (let i = 0; i < length; i += 1) output += alphabet[bytes[i] % alphabet.length];
  return output;
}

/**
 * Build a decoy indistinguishable from a real payload.
 *
 * This matters more than it looks. The old decoys were ~90-character JSON
 * blobs next to a 22-character valid string, so the valid QR had visibly
 * fewer modules — a student could pick it out by eye without decoding
 * anything. Matching the exact length and alphabet means every code on screen
 * renders as the same QR version with the same visual density.
 *
 * The signature is random, so a decoy can never pass server verification.
 */
export function makeDecoy(spec, sessionId) {
  const sid = sessionId || randomFrom(BASE32, spec.sessionIdChars);
  const slot = randomFrom(BASE36, spec.slotChars);
  const signature = randomFrom(BASE32, spec.signatureChars);
  return `${sid}.${slot}.${signature}`;
}
