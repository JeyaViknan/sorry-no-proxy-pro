"use strict";

/**
 * Student-side QR validation — the security boundary that did not previously
 * exist. The scanner used to decide validity locally against a hardcoded
 * string and its own device clock; now it just relays what it decoded and
 * the server rules on it.
 *
 * On success the student receives a short-lived, single-use attendance token.
 * That token is the only way to reach POST /api/attendance, which closes the
 * `curl -X POST /register -d '{"registerNumber":"..."}'` hole entirely.
 */

const express = require("express");
const { ApiError } = require("../middleware/errors");
const { qrValidateLimiter } = require("../middleware/rateLimit");
const { logger } = require("../logger");

/** Distinct, actionable messages. The client also keys off `code`. */
const REJECTION = {
  MALFORMED: {
    code: "QR_NOT_RECOGNISED",
    message: "That is not an attendance code. Keep scanning.",
    keepScanning: true,
  },
  BAD_SIGNATURE: {
    code: "QR_NOT_RECOGNISED",
    message: "That is not an attendance code. Keep scanning.",
    keepScanning: true,
  },
  EXPIRED: {
    code: "QR_EXPIRED",
    message: "That code has already changed. Keep scanning.",
    keepScanning: true,
  },
  NOT_YET_VALID: {
    code: "QR_NOT_YET_VALID",
    message: "Check your device clock, then keep scanning.",
    keepScanning: true,
  },
};

function createQrRouter({ qrTokens, attendanceTokens, store }) {
  const router = express.Router();

  router.post("/qr/validate", qrValidateLimiter, (req, res, next) => {
    const { payload, deviceId } = req.body || {};

    // A missing device id is a client bug, not a scanning outcome — 4xx.
    if (typeof deviceId !== "string" || deviceId.length < 8 || deviceId.length > 128) {
      return next(
        ApiError.badRequest("DEVICE_ID_REQUIRED", "Your browser could not be identified.")
      );
    }

    /**
     * Every payload-level rejection answers with HTTP 200 and `valid:false`.
     *
     * Consistency matters here: an over-long payload previously returned 400
     * while a wrong-shaped one returned 200, so the client threw for one and
     * resolved for the other despite both meaning "that is not a valid code,
     * keep scanning". Decoding a decoy is the *expected* state during
     * scanning, not a transport error, and hundreds arrive per session.
     */
    const reject = (reason) => {
      const rejection = REJECTION[reason] || REJECTION.MALFORMED;
      return res.json({
        ok: false,
        valid: false,
        code: rejection.code,
        message: rejection.message,
        keepScanning: true,
      });
    };

    if (typeof payload !== "string" || payload.length > 128) {
      return reject("MALFORMED");
    }

    const now = Date.now();
    const result = qrTokens.verify(payload, now);

    if (!result.ok) return reject(result.reason);

    const session = store.getSession(result.sessionId, now);
    if (!session) {
      return res.json({
        ok: false,
        valid: false,
        code: "SESSION_ENDED",
        message: "This attendance session has ended.",
        keepScanning: false,
      });
    }

    // Idempotent by design — see SessionStore.recordScan.
    const { firstUse } = store.recordScan({
      sessionId: result.sessionId,
      slot: result.slot,
      deviceId,
      nowMs: now,
    });

    const issued = attendanceTokens.issue({
      sessionId: result.sessionId,
      deviceId,
      slot: result.slot,
      nowMs: now,
    });

    if (firstUse) {
      logger.info("qr validated", { sessionId: result.sessionId, slot: result.slot });
    }

    return res.json({
      ok: true,
      valid: true,
      attendanceToken: issued.token,
      expiresAt: issued.expiresAt,
      serverTime: now,
      session: { id: session.id, label: session.label },
    });
  });

  return router;
}

module.exports = { createQrRouter };
