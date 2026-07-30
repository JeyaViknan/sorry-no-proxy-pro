"use strict";

/**
 * Request logging with timing and a correlation id.
 *
 * THE GAP THIS CLOSES
 * -------------------
 * When a student says "it rejected me in class this morning", the only way to
 * investigate is to find that one request among thousands. Previously there
 * was no per-request record at all — no id, no duration, no outcome — so a
 * dispute was unresolvable and a slow verification was invisible until someone
 * noticed the queue backing up.
 *
 * Every response carries `X-Request-Id`, which the client surfaces on error
 * screens. A student can read it out and it maps to exactly one log line.
 *
 * DELIBERATELY NOT LOGGED: registration numbers, frames, tokens. Attendance
 * outcomes are logged separately in routes/attendance.js where the
 * registration number is genuinely needed; a raw access log has no business
 * carrying it, and logs are frequently shipped somewhere less protected than
 * the attendance store.
 */

const { randomUUID } = require("crypto");
const { logger } = require("../logger");

/** Endpoints too chatty to log at info level. */
const QUIET_PATHS = new Set(["/healthz", "/readyz", "/api/hello"]);

/** Anything slower than this is worth noticing even when it succeeds. */
const SLOW_REQUEST_MS = 3000;

function requestLog(req, res, next) {
  const startedAt = process.hrtime.bigint();
  const requestId = req.get("X-Request-Id") || randomUUID();

  req.id = requestId;
  res.setHeader("X-Request-Id", requestId);

  res.on("finish", () => {
    const durationMs = Number(process.hrtime.bigint() - startedAt) / 1e6;
    const quiet = QUIET_PATHS.has(req.path);
    const slow = durationMs > SLOW_REQUEST_MS;

    // Quiet endpoints only surface when they misbehave.
    if (quiet && res.statusCode < 400 && !slow) return;

    const entry = {
      id: requestId,
      method: req.method,
      path: req.path,
      status: res.statusCode,
      ms: Math.round(durationMs),
    };

    if (res.statusCode >= 500) logger.error("request failed", entry);
    else if (res.statusCode >= 400) logger.warn("request rejected", entry);
    else if (slow) logger.warn("slow request", entry);
    else logger.info("request", entry);
  });

  next();
}

module.exports = { requestLog };
