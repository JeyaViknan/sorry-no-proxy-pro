"use strict";

/**
 * Express application assembly.
 *
 * Separated from index.js (process lifecycle) so the app can be constructed
 * in a test without binding a port or spawning Python workers.
 */

const path = require("path");
const express = require("express");
const compression = require("compression");

const { logger } = require("./logger");
const { buildHelmet, buildCors, permissionsPolicy } = require("./middleware/security");
const { globalLimiter } = require("./middleware/rateLimit");
const { requestLog } = require("./middleware/requestLog");
const { errorHandler, notFoundHandler } = require("./middleware/errors");
const { FacultyAuth } = require("./middleware/auth");

const { QrTokenService } = require("./services/qrToken");
const { AttendanceTokenService } = require("./services/attendanceToken");
const { SessionStore } = require("./services/sessionStore");
const { SheetsExporter } = require("./services/sheets");

const { createHealthRouter } = require("./routes/health");
const { createSessionRouter } = require("./routes/session");
const { createQrRouter } = require("./routes/qr");
const { createAttendanceRouter } = require("./routes/attendance");

function createApp({ config, faceVerifier }) {
  const startedAt = Date.now();
  const app = express();

  // Behind Cloud Run / Render / HF Spaces there is always exactly one proxy.
  // Trusting precisely one hop lets rate limiting see the real client IP
  // without letting a client forge X-Forwarded-For.
  app.set("trust proxy", 1);
  app.disable("x-powered-by");

  // ── Services ──────────────────────────────────────────────────────
  const qrTokens = new QrTokenService({
    secret: config.secrets.qrSigning,
    ttlMs: config.qr.tokenTtlMs,
    clockSkewMs: config.qr.clockSkewMs,
  });

  const attendanceTokens = new AttendanceTokenService({
    secret: config.secrets.tokenSigning,
    ttlMs: config.attendance.tokenTtlMs,
  });

  const store = new SessionStore({
    sessionTtlMs: config.attendance.sessionTtlMs,
    attendanceTokenTtlMs: config.attendance.tokenTtlMs,
    qrTokenTtlMs: config.qr.tokenTtlMs,
  });

  const sheets = new SheetsExporter(config.sheets, logger);

  const facultyAuth = new FacultyAuth({
    secret: config.secrets.tokenSigning,
    accessCode: config.secrets.facultyAccessCode,
  });

  // ── Middleware ────────────────────────────────────────────────────
  // First, so every downstream log line and error response carries the id.
  app.use(requestLog);
  app.use(buildHelmet());
  app.use(permissionsPolicy);
  app.use(compression());
  app.use(buildCors(config.cors));
  app.use(globalLimiter);

  // Sized for the burst pipeline: 3 frames x 400KB, plus base64 overhead
  // and JSON envelope. Well below the old blanket 10MB, which combined with
  // a serialised worker queue made a trivial denial of service.
  app.use(express.json({ limit: "2mb" }));

  // ── Static assets ─────────────────────────────────────────────────
  // ONLY the public directory. The old `express.static(__dirname)` served the
  // entire application root, which meant GET /data/25BCE1276.png returned any
  // student's enrollment photograph, plus /server.js, /face_db.pkl and — had
  // one ever existed there — /.env with the Google private key.
  app.use(
    express.static(config.paths.publicDir, {
      index: "index.html",
      etag: true,
      lastModified: true,
      setHeaders(res, filePath) {
        const name = path.basename(filePath);

        if (filePath.endsWith(".html")) {
          // Always revalidate the entry point. It is ~2KB, so the cost is one
          // conditional request that usually answers 304.
          res.setHeader("Cache-Control", "no-cache, must-revalidate");
          return;
        }

        // Content-hashed filenames can never change meaning — cache hard.
        if (/[.-][0-9a-f]{8,}\./i.test(name)) {
          res.setHeader("Cache-Control", "public, max-age=31536000, immutable");
          return;
        }

        // Everything else (our unhashed CSS/JS, ~25KB gzipped in total).
        //
        // The previous `max-age=3600` meant a browser would not even ASK for
        // an updated file for an hour — so a fix pushed mid-class could not
        // reach the students in the room. This was not theoretical: it bit
        // during review, serving a stale stylesheet while the server had the
        // corrected one.
        //
        // `stale-while-revalidate` keeps loads instant (served from cache) but
        // refreshes in the background, so a fix propagates on the next visit
        // instead of after an hour.
        res.setHeader("Cache-Control", "public, max-age=60, stale-while-revalidate=86400");
      },
    })
  );

  // ── Routes ────────────────────────────────────────────────────────
  app.use(createHealthRouter({ faceVerifier, sheets, store, startedAt }));
  app.use("/api", createSessionRouter({ config, facultyAuth, qrTokens, store }));
  app.use("/api", createQrRouter({ qrTokens, attendanceTokens, store }));
  app.use(
    "/api",
    createAttendanceRouter({ config, attendanceTokens, store, faceVerifier, sheets })
  );

  app.use(notFoundHandler);
  app.use(errorHandler);

  app.locals.services = { qrTokens, attendanceTokens, store, sheets, facultyAuth };
  return app;
}

module.exports = { createApp };
