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

    if (typeof deviceId !== "string" || deviceId.length < 8 || deviceId.length > 128) {
      return next(
        ApiError.badRequest("DEVICE_ID_REQUIRED", "Your browser could not be identified.")
      );
    }
    if (typeof payload !== "string" || payload.length > 128) {
      return next(REJECTION.MALFORMED.keepScanning
        ? ApiError.badRequest(REJECTION.MALFORMED.code, REJECTION.MALFORMED.message)
        : ApiError.badRequest("QR_NOT_RECOGNISED", "Unrecognised code."));
    }

    const now = Date.now();
    const result = qrTokens.verify(payload, now);

    if (!result.ok) {
      const rejection = REJECTION[result.reason] || REJECTION.MALFORMED;
      // Deliberately 200, not 4xx: decoding a decoy is the *expected* state
      // during scanning, not an error. Hundreds of these arrive per session
      // and treating them as failures pollutes logs and client error paths.
      return res.json({
        ok: false,
        valid: false,
        code: rejection.code,
        message: rejection.message,
        keepScanning: true,
      });
    }

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
