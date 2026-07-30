"use strict";

/**
 * Centralised, validated configuration.
 *
 * Design rule: the process refuses to start if a security-critical value
 * is missing or weak. The previous version silently degraded — an unset
 * GOOGLE_PRIVATE_KEY became the empty string and failures only surfaced
 * mid-class. Fail at boot, loudly, where it is cheap to fix.
 */

// quiet: dotenv v17 prints a promotional banner to stdout, which corrupts the
// first lines of a JSON log stream that an aggregator is trying to parse.
// On Cloud Run there is no .env file anyway — config comes from the
// environment and Secret Manager.
require("dotenv").config({ quiet: true });

const path = require("path");

const errors = [];
const warnings = [];

function required(name, { minLength = 1 } = {}) {
  const value = (process.env[name] || "").trim();
  if (!value) {
    errors.push(`${name} is required but not set`);
    return "";
  }
  if (value.length < minLength) {
    errors.push(`${name} must be at least ${minLength} characters (got ${value.length})`);
    return value;
  }
  return value;
}

function optional(name, fallback = "") {
  const value = (process.env[name] || "").trim();
  return value || fallback;
}

function integer(name, fallback, { min = 0, max = Number.MAX_SAFE_INTEGER } = {}) {
  const raw = (process.env[name] || "").trim();
  if (!raw) return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed)) {
    errors.push(`${name} must be an integer (got "${raw}")`);
    return fallback;
  }
  if (parsed < min || parsed > max) {
    errors.push(`${name} must be between ${min} and ${max} (got ${parsed})`);
    return fallback;
  }
  return parsed;
}

function decimal(name, fallback, { min = 0, max = 1 } = {}) {
  const raw = (process.env[name] || "").trim();
  if (!raw) return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < min || parsed > max) {
    errors.push(`${name} must be a number between ${min} and ${max} (got "${raw}")`);
    return fallback;
  }
  return parsed;
}

function list(name, fallback = []) {
  const raw = (process.env[name] || "").trim();
  if (!raw) return fallback;
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

const nodeEnv = optional("NODE_ENV", "development");
const isProduction = nodeEnv === "production";

// ── Secrets ─────────────────────────────────────────────────────────
// 24 chars is the floor for an HMAC secret that resists offline guessing.
const qrSigningSecret = required("QR_SIGNING_SECRET", { minLength: 24 });
const tokenSigningSecret = required("TOKEN_SIGNING_SECRET", { minLength: 24 });
const facultyAccessCode = required("FACULTY_ACCESS_CODE", { minLength: 12 });

if (qrSigningSecret && qrSigningSecret === tokenSigningSecret) {
  errors.push(
    "QR_SIGNING_SECRET and TOKEN_SIGNING_SECRET must differ — sharing them means " +
      "compromising the QR secret also grants the ability to mint attendance tokens"
  );
}

// ── CORS ────────────────────────────────────────────────────────────
const allowedOrigins = list("ALLOWED_ORIGINS");
if (isProduction && allowedOrigins.length === 0) {
  errors.push(
    "ALLOWED_ORIGINS must be set in production. The previous `origin: \"*\"` let " +
      "any website on the internet post attendance on a student's behalf."
  );
}
if (allowedOrigins.includes("*")) {
  errors.push('ALLOWED_ORIGINS must not contain "*" — list exact origins');
}

// ── Google Sheets (optional by design) ──────────────────────────────
const sheetsEmail = optional("GOOGLE_SERVICE_ACCOUNT_EMAIL");
const sheetsKeyRaw = optional("GOOGLE_PRIVATE_KEY");
const sheetId = optional("SHEET_ID");

// SHEET_ID alone is enough: on Cloud Run the credential comes from the
// instance metadata server, so no key needs to exist in the environment.
// An explicit key is only needed off-platform (local development).
const sheetsEnabled = Boolean(sheetId);

if (sheetsEnabled && !sheetsEmail && !sheetsKeyRaw) {
  warnings.push(
    "Sheets export will authenticate via the instance service account " +
      "(metadata server). Off Google Cloud this fails — set " +
      "GOOGLE_SERVICE_ACCOUNT_EMAIL and GOOGLE_PRIVATE_KEY for local use."
  );
}
if ((sheetsEmail && !sheetsKeyRaw) || (!sheetsEmail && sheetsKeyRaw)) {
  errors.push(
    "GOOGLE_SERVICE_ACCOUNT_EMAIL and GOOGLE_PRIVATE_KEY must be set together, or neither."
  );
}
if (!sheetsEnabled) {
  warnings.push(
    "Google Sheets export is OFF (SHEET_ID unset). Attendance is still recorded " +
      "server-side and exportable via /api/sessions/:id/export — a Sheets outage " +
      "never blocks a class."
  );
}

// Validate the gallery URI at boot rather than mid-startup.
const galleryGcsUri = optional("GALLERY_GCS_URI");
if (galleryGcsUri && !/^gs:\/\/[^/]+\/.+$/.test(galleryGcsUri)) {
  errors.push(
    `GALLERY_GCS_URI must look like gs://bucket-name/face_db.npz (got "${galleryGcsUri}")`
  );
}

// ── QR protocol timing ──────────────────────────────────────────────
const qrTokenTtlMs = integer("QR_TOKEN_TTL_MS", 4000, { min: 1000, max: 30000 });
const qrClockSkewMs = integer("QR_CLOCK_SKEW_MS", 2000, { min: 0, max: 10000 });
const attendanceTokenTtlMs = integer("ATTENDANCE_TOKEN_TTL_MS", 120000, {
  min: 15000,
  max: 600000,
});

// ── Face verification ───────────────────────────────────────────────
// DEFAULTS ARE 0.55 / 0.45, NOT the 0.50 / 0.42 this shipped with.
//
// 0.50 was never calibrated — it was inherited from a paper about a different
// dataset. Measured against the actual 67-student gallery, two students
// (25BRS1169 and 25BRS1286) score 0.5016 against each other, which at 0.50
// means each can mark the other present. That is a live false accept: the
// precise failure this system exists to prevent.
//
// 0.55 yields zero colliding pairs in that gallery. A default has to be SAFE
// when nobody sets the variable, so the safe value is the default.
//
// TODO(enrollment): replace both with the output of `npm run verify:gallery`
// once multi-image enrollment data exists — it can then measure false
// REJECTIONS too, which impostor data alone cannot.
const faceThresholdAccept = decimal("FACE_THRESHOLD_ACCEPT", 0.55, { min: 0.1, max: 0.99 });
const faceThresholdReview = decimal("FACE_THRESHOLD_REVIEW", 0.45, { min: 0.1, max: 0.99 });
if (faceThresholdReview > faceThresholdAccept) {
  errors.push(
    `FACE_THRESHOLD_REVIEW (${faceThresholdReview}) must be <= ` +
      `FACE_THRESHOLD_ACCEPT (${faceThresholdAccept})`
  );
}

const config = Object.freeze({
  nodeEnv,
  isProduction,
  port: integer("PORT", 7860, { min: 1, max: 65535 }),

  secrets: Object.freeze({
    qrSigning: qrSigningSecret,
    tokenSigning: tokenSigningSecret,
    facultyAccessCode,
  }),

  cors: Object.freeze({ allowedOrigins }),

  qr: Object.freeze({
    tokenTtlMs: qrTokenTtlMs,
    clockSkewMs: qrClockSkewMs,
    // How far ahead the faculty display may prefetch signed tokens. Larger
    // batches mean fewer round trips on flaky classroom wifi; too large and
    // a leaked batch stays useful for longer. 60s is the balance point.
    maxBatchSpanMs: integer("QR_MAX_BATCH_SPAN_MS", 60000, { min: 10000, max: 300000 }),
  }),

  attendance: Object.freeze({
    tokenTtlMs: attendanceTokenTtlMs,
    maxAttempts: integer("FACE_MAX_ATTEMPTS", 3, { min: 1, max: 10 }),
    sessionTtlMs: integer("SESSION_TTL_MS", 3 * 60 * 60 * 1000, {
      min: 60000,
      max: 12 * 60 * 60 * 1000,
    }),
    registerNumberPattern: optional("REGISTER_NUMBER_PATTERN", "^[0-9]{2}[A-Z]{3}[0-9]{4}$"),
  }),

  face: Object.freeze({
    thresholdAccept: faceThresholdAccept,
    thresholdReview: faceThresholdReview,
    maxFramesPerRequest: integer("FACE_MAX_FRAMES", 3, { min: 1, max: 5 }),
    // Per-frame ceiling. A 720p JPEG at q0.92 is ~180KB; 400KB leaves headroom
    // without letting a client tie up a worker with a huge upload.
    maxFrameBytes: integer("FACE_MAX_FRAME_BYTES", 400 * 1024, { min: 32 * 1024 }),
    workerCount: integer("FACE_WORKER_COUNT", 1, { min: 1, max: 8 }),
    requestTimeoutMs: integer("FACE_REQUEST_TIMEOUT_MS", 20000, { min: 3000, max: 60000 }),
  }),

  google: Object.freeze({
    clientEmail: sheetsEmail,
    // Secret managers and shells frequently deliver literal backslash-n.
    privateKey: sheetsKeyRaw.replace(/\\n/g, "\n"),
    scopes: Object.freeze([
      "https://www.googleapis.com/auth/spreadsheets",
      "https://www.googleapis.com/auth/devstorage.read_only",
    ]),
    galleryGcsUri,
  }),

  sheets: Object.freeze({
    enabled: sheetsEnabled,
    spreadsheetId: sheetId,
    range: optional("SHEET_RANGE", "Attendance!A:G"),
    flushIntervalMs: integer("SHEET_FLUSH_INTERVAL_MS", 5000, { min: 1000, max: 60000 }),
  }),

  paths: Object.freeze({
    root: path.resolve(__dirname, ".."),
    publicDir: path.resolve(__dirname, "..", "public"),
    galleryDir: path.resolve(__dirname, "..", optional("GALLERY_DIR", "./gallery")),
    pythonBin: optional("PYTHON_BIN", "python3"),
    verifierScript: path.resolve(__dirname, "..", "python", "verifier.py"),
  }),
});

function validateOrExit(logger = console) {
  for (const warning of warnings) logger.warn(`[config] ${warning}`);

  if (errors.length > 0) {
    logger.error("[config] Refusing to start — invalid configuration:");
    for (const error of errors) logger.error(`  • ${error}`);
    logger.error("");
    logger.error("  Copy .env.example to .env and fill in the values.");
    logger.error("  Generate a secret with:");
    logger.error(
      '    node -e "console.log(require(\'crypto\').randomBytes(32).toString(\'base64url\'))"'
    );
    process.exit(1);
  }
}

module.exports = { config, validateOrExit };
