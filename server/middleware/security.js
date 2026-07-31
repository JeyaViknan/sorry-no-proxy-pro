"use strict";

/**
 * Security headers and CORS.
 *
 * The old server had `cors({ origin: "*" })` and no headers at all, so any
 * page on the internet could call the attendance API from a student's
 * browser. Here the allow-list is explicit and comes from config, which
 * refuses to boot in production if it is empty.
 */

const helmet = require("helmet");

function buildHelmet() {
  return helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        // The scanner needs a worker for QR decoding and blob: for camera frames.
        scriptSrc: ["'self'", "blob:"],
        workerSrc: ["'self'", "blob:"],
        // Fonts are self-hosted now; the old stylesheet @import'ed them from
        // fonts.googleapis.com, which was a render-blocking third-party
        // request on classroom wifi and leaked every visitor to Google.
        styleSrc: ["'self'"],
        fontSrc: ["'self'"],
        imgSrc: ["'self'", "data:", "blob:"],
        mediaSrc: ["'self'", "blob:"],
        connectSrc: ["'self'"],
        objectSrc: ["'none'"],
        frameAncestors: ["'none'"],
        baseUri: ["'self'"],
        formAction: ["'self'"],
      },
    },
    // Camera access requires a secure context; HSTS enforces it after first visit.
    hsts: { maxAge: 31536000, includeSubDomains: true, preload: false },
    crossOriginEmbedderPolicy: false, // would break the camera preview
    referrerPolicy: { policy: "same-origin" },
  });
}

/**
 * Strict CORS. Requests with no Origin header (curl, native apps, same-origin
 * navigations) are allowed through because CORS is a browser mechanism and
 * blocking them here provides no protection — the real gate is the
 * attendance token, which those callers cannot forge.
 */
function buildCors({ allowedOrigins }) {
  const allowed = new Set(allowedOrigins);

  return function cors(req, res, next) {
    const origin = req.headers.origin;

    if (origin) {
      if (!allowed.has(origin)) {
        if (req.method === "OPTIONS") return res.status(403).end();
        return res.status(403).json({
          ok: false,
          code: "ORIGIN_NOT_ALLOWED",
          message: "This origin is not permitted to call the attendance API.",
        });
      }
      res.setHeader("Access-Control-Allow-Origin", origin);
      res.setHeader("Vary", "Origin");
      res.setHeader("Access-Control-Allow-Credentials", "false");
    }

    if (req.method === "OPTIONS") {
      res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
      // Every header the clients actually send must be listed here, or the
      // browser blocks the real request after the preflight — and `fetch`
      // reports it as a generic network failure, which reads to the user as
      // "the server is down" rather than "a header was not permitted".
      //
      // ngrok-skip-browser-warning is sent by both clients to suppress
      // ngrok's HTML interstitial. Omitting it here broke faculty sign-in
      // completely while the API itself was perfectly healthy.
      res.setHeader(
        "Access-Control-Allow-Headers",
        "Content-Type,Authorization,ngrok-skip-browser-warning"
      );
      res.setHeader("Access-Control-Max-Age", "86400");
      return res.status(204).end();
    }

    return next();
  };
}

/**
 * Permissions-Policy for the scanner page. Camera is needed; everything else
 * that could fingerprint or track is switched off explicitly.
 */
function permissionsPolicy(req, res, next) {
  res.setHeader(
    "Permissions-Policy",
    "camera=(self), microphone=(), geolocation=(), payment=(), usb=(), interest-cohort=()"
  );
  next();
}

module.exports = { buildHelmet, buildCors, permissionsPolicy };
