"use strict";

/**
 * Rate limiting, shaped around one awkward fact about the deployment:
 *
 *   A CLASSROOM IS BEHIND ONE NAT. Five hundred students on the lecture-hall
 *   wifi share a single public IP. A conventional per-IP limiter would treat
 *   a normal class as a flood and lock out the entire room — turning a
 *   security control into an outage. This is the single easiest way to break
 *   this system in production, so the layering below is deliberate.
 *
 * Four layers, narrowest first:
 *
 *   1. Per-device  — the real control for student endpoints. deviceId is
 *                    client-generated and persisted, so it is not a security
 *                    boundary on its own; it is a fairness boundary that
 *                    stops one phone hammering the API.
 *   2. Per-regno   — enforced in the session store (FACE_MAX_ATTEMPTS). This
 *                    is what actually stops face brute-forcing, and it cannot
 *                    be evaded by clearing localStorage.
 *   3. Per-IP      — deliberately generous. Sized to let a large class
 *                    through while still stopping a single scripted attacker
 *                    from saturating the face-verification queue.
 *   4. Faculty     — strict per-IP, because the access code is a secret worth
 *                    brute-forcing and faculty are few.
 */

const rateLimit = require("express-rate-limit");
const { ipKeyGenerator } = require("express-rate-limit");

const jsonError = (code, message) => (req, res) =>
  res.status(429).json({ ok: false, code, message, retryAfterMs: 60_000 });

/**
 * Prefer the client device id; fall back to IP for callers that omit it.
 *
 * The IP fallback goes through `ipKeyGenerator`, which normalises IPv6 to its
 * /64 prefix. Using `req.ip` raw would be a real bypass, not a lint nit: a
 * residential IPv6 allocation is typically a whole /64, so a client could
 * pick a fresh address per request and never hit the limit.
 */
function deviceKey(req) {
  const deviceId = req.body?.deviceId || req.get("X-Device-Id");
  if (typeof deviceId === "string" && deviceId.length >= 8 && deviceId.length <= 128) {
    return `dev:${deviceId}`;
  }
  return `ip:${ipKeyGenerator(req.ip)}`;
}

const qrValidateLimiter = rateLimit({
  windowMs: 60_000,
  // A scanner posts at most one validate per decoded valid frame, and stops
  // on success. 30/min is far above normal use and far below useful abuse.
  limit: 30,
  keyGenerator: deviceKey,
  standardHeaders: "draft-7",
  legacyHeaders: false,
  handler: jsonError(
    "RATE_LIMITED_QR",
    "Too many scan attempts from this device. Wait a moment and try again."
  ),
});

/**
 * Must sit ABOVE the server-side attempt budgets, never below them.
 *
 * The quality budget allows 10 unreadable-photo retries and the identity
 * budget 3 mismatches. At the previous limit of 10/min a student fighting bad
 * lighting hit a bare 429 before the purpose-built cap ever applied — the
 * rate limiter, which exists to bound abuse, was silently doing the job of a
 * control designed to be helpful. Whenever those budgets change, this must
 * stay comfortably clear of their sum.
 */
const attendanceLimiter = rateLimit({
  windowMs: 60_000,
  limit: 25,
  keyGenerator: deviceKey,
  standardHeaders: "draft-7",
  legacyHeaders: false,
  handler: jsonError(
    "RATE_LIMITED_ATTENDANCE",
    "Too many submissions from this device. Wait a moment and try again."
  ),
});

/**
 * Backstop, sized from the actual per-student request budget.
 *
 * A student makes roughly: 1 x /api/hello, 1-2 x /api/qr/validate,
 * 1 x /api/attendance, plus a retry or two on a bad connection -> ~5 requests.
 * The whole class arrives inside a 2-3 minute window, from ONE NAT'd IP.
 *
 *     500 students  x 5 =  2,500 requests
 *   2,000 students  x 5 = 10,000 requests
 *
 * The previous value of 1200/min would therefore have rate-limited a normal
 * 500-student class part way through — turning a security control into an
 * outage, which is the single easiest way to break this system in production.
 *
 * Default 6000/min covers 500 students comfortably even if they all arrive in
 * the same minute. Raise RATE_LIMIT_GLOBAL_PER_MIN for larger cohorts; the
 * per-device limiters above remain the real fairness control.
 */
const globalLimiter = rateLimit({
  windowMs: 60_000,
  limit: Number(process.env.RATE_LIMIT_GLOBAL_PER_MIN) || 6000,
  standardHeaders: "draft-7",
  legacyHeaders: false,
  skip: (req) => req.path === "/healthz" || req.path === "/api/health",
  handler: jsonError(
    "RATE_LIMITED",
    "The server is receiving an unusual amount of traffic. Please try again shortly."
  ),
});

const facultyAuthLimiter = rateLimit({
  windowMs: 15 * 60_000,
  limit: 10,
  standardHeaders: "draft-7",
  legacyHeaders: false,
  skipSuccessfulRequests: true,
  handler: jsonError(
    "RATE_LIMITED_AUTH",
    "Too many failed access attempts. Try again in 15 minutes."
  ),
});

module.exports = {
  qrValidateLimiter,
  attendanceLimiter,
  globalLimiter,
  facultyAuthLimiter,
};
