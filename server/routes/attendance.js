"use strict";

/**
 * Attendance submission — one atomic, authenticated transaction.
 *
 * REPLACES: POST /verify-face followed by POST /register, two unauthenticated
 * endpoints with nothing binding them. Face verification was optional
 * (`if (faceImage)`) and the client called /register without it, so the
 * entire system could be bypassed with a single curl.
 *
 * Now a submission requires an attendance token that can only be obtained by
 * presenting a server-signed QR payload within its 4-second window, and face
 * verification is unconditional.
 *
 * RETRY SEMANTICS (a deliberate change from the audit's first sketch)
 * ------------------------------------------------------------------
 * The old client made rejection terminal — "Please speak to the professor",
 * session over — nominally to stop brute force. It did not (no server-side
 * limit, and reloading the page reset it), while it *did* permanently lock
 * out a legitimate student whose one captured frame happened to be blurry.
 *
 * Here the token survives a retryable rejection so the student can simply try
 * again, and abuse is bounded by a per-registration-number attempt counter
 * held server-side, which clearing localStorage cannot reset. The token is
 * consumed only on a terminal outcome.
 */

const express = require("express");
const { ApiError } = require("../middleware/errors");
const { attendanceLimiter } = require("../middleware/rateLimit");
const { logger } = require("../logger");

const DATA_URI_PREFIX = /^data:image\/(jpeg|jpg|png|webp);base64,/;

/**
 * Separate, looser budget for unreadable photos. A student fighting bad
 * lighting should get several tries; this exists only so a client cannot
 * upload frames indefinitely.
 */
const QUALITY_ATTEMPT_CAP = 10;

function createAttendanceRouter({ config, attendanceTokens, store, faceVerifier, sheets }) {
  const router = express.Router();
  const regnoPattern = new RegExp(config.attendance.registerNumberPattern);

  function parseFrames(raw) {
    if (!Array.isArray(raw) || raw.length === 0) {
      throw ApiError.badRequest("FRAMES_REQUIRED", "No photo was captured. Please try again.");
    }
    if (raw.length > config.face.maxFramesPerRequest) {
      throw ApiError.badRequest(
        "TOO_MANY_FRAMES",
        `Send at most ${config.face.maxFramesPerRequest} images.`
      );
    }

    return raw.map((frame, index) => {
      if (typeof frame !== "string" || !DATA_URI_PREFIX.test(frame)) {
        throw ApiError.badRequest("FRAME_MALFORMED", "The captured photo was not readable.");
      }
      const payload = frame.slice(frame.indexOf(",") + 1);
      // base64 is 4/3 of the raw byte count.
      const approxBytes = Math.floor((payload.length * 3) / 4);
      if (approxBytes > config.face.maxFrameBytes) {
        throw ApiError.tooLarge(
          "FRAME_TOO_LARGE",
          "The captured photo is too large. Please try again."
        );
      }
      if (approxBytes < 2048) {
        throw ApiError.badRequest(
          "FRAME_EMPTY",
          "The camera did not produce an image. Check camera permissions and try again."
        );
      }
      return { index, base64: payload };
    });
  }

  router.post("/attendance", attendanceLimiter, async (req, res, next) => {
    const { attendanceToken, deviceId, registerNumber: rawRegno, frames: rawFrames } = req.body || {};
    const now = Date.now();

    // ── 1. Authenticate the submission ──────────────────────────────
    const tokenResult = attendanceTokens.verify(attendanceToken, { deviceId, nowMs: now });
    if (!tokenResult.ok) {
      const message =
        tokenResult.reason === "EXPIRED"
          ? "Your scan expired. Please scan the code again."
          : "Invalid submission. Please scan the code again.";
      return next(ApiError.unauthorized(`TOKEN_${tokenResult.reason}`, message));
    }

    const { claims } = tokenResult;
    const session = store.getSession(claims.sid, now);
    if (!session) {
      return next(
        ApiError.forbidden("SESSION_ENDED", "This attendance session has ended.")
      );
    }

    // ── 2. Validate inputs ──────────────────────────────────────────
    const registerNumber = String(rawRegno || "").trim().toUpperCase();
    if (!regnoPattern.test(registerNumber)) {
      return next(
        ApiError.badRequest(
          "REGNO_INVALID",
          "Check your registration number and try again."
        )
      );
    }

    let frames;
    try {
      frames = parseFrames(rawFrames);
    } catch (error) {
      return next(error);
    }

    // ── 3. Already marked? Idempotent, not an error ─────────────────
    const existing = store.getAttendance(session, registerNumber);
    if (existing) {
      return res.json({
        ok: true,
        status: existing.status,
        alreadyRecorded: true,
        message: "Your attendance is already recorded for this session.",
        recordedAt: existing.recordedAt,
      });
    }

    // ── 4. Attempt cap — the real brute-force control ───────────────
    const attempts = store.countAttempts(session, registerNumber);
    if (attempts >= config.attendance.maxAttempts) {
      logger.warn("identity attempt cap reached", { sessionId: session.id, registerNumber });
      return next(
        ApiError.forbidden(
          "TOO_MANY_ATTEMPTS",
          "Too many failed attempts. Please see your faculty member to be marked manually."
        )
      );
    }
    if (store.countQualityAttempts(session, registerNumber) >= QUALITY_ATTEMPT_CAP) {
      logger.warn("quality attempt cap reached", { sessionId: session.id, registerNumber });
      return next(
        ApiError.forbidden(
          "TOO_MANY_ATTEMPTS",
          "We could not get a clear photo. Please see your faculty member to be marked manually."
        )
      );
    }

    // ── 5. Face verification ────────────────────────────────────────
    let verification;
    try {
      verification = await faceVerifier.verify({ registerNumber, frames });
    } catch (error) {
      // A worker failure is our problem, not the student's — never consume
      // the token or count an attempt for it.
      logger.error("face verification unavailable", error);
      return next(
        ApiError.unavailable(
          "VERIFIER_UNAVAILABLE",
          "Verification is temporarily unavailable. Please try again in a moment."
        )
      );
    }

    if (!verification.ok) {
      // A QUALITY failure is not an identity claim — the photo was unreadable
      // (motion blur, backlight, no face in frame). Counting it against the
      // identity attempt budget means a student in bad lighting exhausts three
      // attempts without the model ever having compared a usable face, and is
      // then locked out of a class they are sitting in. Only genuine
      // mismatches consume the budget; quality failures get their own, looser
      // cap purely to bound abuse.
      const isQualityFailure =
        verification.reason === "QUALITY" || verification.reason === "NO_FACE";

      let remaining;
      if (isQualityFailure) {
        const qualityAttempts = store.incrementQualityAttempts(session, registerNumber);
        remaining = Math.max(0, QUALITY_ATTEMPT_CAP - qualityAttempts);
      } else {
        store.incrementAttempts(session, registerNumber);
        remaining = Math.max(0, config.attendance.maxAttempts - attempts - 1);
      }

      logger.info("verification rejected", {
        sessionId: session.id,
        registerNumber,
        reason: verification.reason,
        qualityCode: verification.qualityCode,
        similarity: verification.similarity,
        kind: isQualityFailure ? "quality" : "identity",
        attemptsRemaining: remaining,
      });

      // Retryable while attempts remain: token stays alive.
      return res.status(200).json({
        ok: false,
        status: "rejected",
        code: verification.reason,
        message: verification.message,
        canRetry: remaining > 0,
        attemptsRemaining: remaining,
      });
    }

    // ── 6. Terminal outcome — consume the token exactly once ────────
    if (!store.consumeToken(claims.jti, now)) {
      return next(
        ApiError.conflict(
          "TOKEN_ALREADY_USED",
          "This scan was already submitted. Please scan again if you need to retry."
        )
      );
    }

    const status = verification.similarity >= config.face.thresholdAccept ? "accepted" : "flagged";

    const record = {
      registerNumber,
      status,
      similarity: verification.similarity,
      quality: verification.quality,
      recordedAt: now,
      sessionId: session.id,
      slot: claims.slot,
    };
    store.recordAttendance(session, record);

    // Durable locally first; the spreadsheet is a downstream export that can
    // never block or fail a student's submission.
    sheets.enqueue([
      new Date(now).toISOString(),
      session.id,
      session.label || "",
      registerNumber,
      status,
      verification.similarity.toFixed(4),
      status === "flagged" ? "REVIEW" : "",
    ]);

    logger.info("attendance recorded", {
      sessionId: session.id,
      registerNumber,
      status,
      similarity: verification.similarity,
    });

    return res.json({
      ok: true,
      status,
      alreadyRecorded: false,
      message:
        status === "accepted"
          ? "Attendance recorded."
          : "Attendance recorded and flagged for faculty review.",
      recordedAt: now,
    });
  });

  return router;
}

module.exports = { createAttendanceRouter };
