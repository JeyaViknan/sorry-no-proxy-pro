"use strict";

/**
 * Faculty session control.
 *
 * The faculty display no longer knows how to make a valid QR — it asks the
 * server for pre-signed batches. That is the whole point of the redesign:
 * the signing key never leaves the backend, so reading the faculty app's
 * source (or its network traffic) does not let anyone mint a valid code.
 *
 * Batching is a network-reliability decision. The display rotates every
 * ~500ms; one request per frame would stall visibly on classroom wifi. One
 * request per ~60s of runway, fetched well ahead of need, means a 10-second
 * outage is invisible to the room.
 */

const express = require("express");
const { ApiError } = require("../middleware/errors");
const { facultyAuthLimiter } = require("../middleware/rateLimit");
const { randomBase32 } = require("../services/crypto");
const { logger } = require("../logger");

function createSessionRouter({ config, facultyAuth, qrTokens, store }) {
  const router = express.Router();
  const requireFaculty = facultyAuth.requireFaculty();

  // ── Sign in ───────────────────────────────────────────────────────
  router.post("/faculty/login", facultyAuthLimiter, (req, res, next) => {
    const { accessCode } = req.body || {};

    if (!facultyAuth.verifyAccessCode(accessCode)) {
      logger.warn("faculty login failed", { ip: req.ip });
      return next(ApiError.unauthorized("BAD_ACCESS_CODE", "Incorrect access code."));
    }

    const { token, expiresAt } = facultyAuth.issueToken();
    logger.info("faculty signed in", { ip: req.ip });

    return res.json({
      ok: true,
      facultyToken: token,
      expiresAt,
      serverTime: Date.now(),
    });
  });

  // ── Start a session ───────────────────────────────────────────────
  router.post("/sessions", requireFaculty, (req, res) => {
    const label = String(req.body?.label || "").slice(0, 80);
    const now = Date.now();

    // Session ids live inside the QR payload, so they must be base32 to keep
    // the symbol in QR alphanumeric mode. 5 bytes -> 8 chars -> 2^40 space,
    // which is ample for concurrent sessions and unguessable in practice.
    const sessionId = randomBase32(5);
    const session = store.createSession({ id: sessionId, label, nowMs: now });

    // Ship the first batch with the create response: the display can start
    // immediately without a second round trip.
    const tokens = qrTokens.batch(sessionId, now, config.qr.maxBatchSpanMs);

    logger.info("session started", { sessionId, label });

    return res.status(201).json({
      ok: true,
      sessionId,
      label,
      expiresAt: session.expiresAt,
      serverTime: now,
      qr: {
        slotTtlMs: config.qr.tokenTtlMs,
        batchSpanMs: config.qr.maxBatchSpanMs,
        noiseSpec: qrTokens.noiseSpec,
      },
      tokens,
    });
  });

  // ── Fetch the next batch of signed payloads ───────────────────────
  router.get("/sessions/:id/tokens", requireFaculty, (req, res, next) => {
    const session = store.getSession(req.params.id);
    if (!session) {
      return next(ApiError.notFound("SESSION_NOT_FOUND", "That session has ended or expired."));
    }

    const now = Date.now();
    const requestedFrom = Number(req.query.from);
    // Clamp into [now, now + span]. A client cannot pre-mint tokens for an
    // arbitrary future — that would hand it a reusable stockpile.
    const from = Number.isFinite(requestedFrom)
      ? Math.min(Math.max(requestedFrom, now), now + config.qr.maxBatchSpanMs)
      : now;

    const requestedSpan = Number(req.query.span);
    const span = Number.isFinite(requestedSpan)
      ? Math.min(Math.max(requestedSpan, config.qr.tokenTtlMs), config.qr.maxBatchSpanMs)
      : config.qr.maxBatchSpanMs;

    return res.json({
      ok: true,
      sessionId: session.id,
      // The client uses this to correct for its own clock drift. The faculty
      // device deciding *when* to show a token must agree with the server
      // about the current slot, or it will display expired codes.
      serverTime: now,
      slotTtlMs: config.qr.tokenTtlMs,
      tokens: qrTokens.batch(session.id, from, span),
    });
  });

  // ── Live status ───────────────────────────────────────────────────
  router.get("/sessions/:id/summary", requireFaculty, (req, res, next) => {
    const session = store.getSession(req.params.id);
    if (!session) {
      return next(ApiError.notFound("SESSION_NOT_FOUND", "That session has ended or expired."));
    }

    const records = [...session.attendance.values()];
    return res.json({
      ok: true,
      sessionId: session.id,
      label: session.label,
      startedAt: session.createdAt,
      serverTime: Date.now(),
      counts: {
        total: records.length,
        accepted: records.filter((r) => r.status === "accepted").length,
        flagged: records.filter((r) => r.status === "flagged").length,
      },
      records: records.map((r) => ({
        registerNumber: r.registerNumber,
        status: r.status,
        similarity: r.similarity,
        at: r.recordedAt,
      })),
    });
  });

  // ── CSV export ────────────────────────────────────────────────────
  // The safety net for the Sheets export: attendance is always recoverable
  // from the server itself, so a spreadsheet outage never loses a class.
  router.get("/sessions/:id/export", requireFaculty, (req, res, next) => {
    const session = store.sessions.get(req.params.id);
    if (!session) {
      return next(ApiError.notFound("SESSION_NOT_FOUND", "No such session."));
    }

    const header = "register_number,status,similarity,recorded_at,session_id,session_label\n";
    const rows = [...session.attendance.values()]
      .map((r) =>
        [
          r.registerNumber,
          r.status,
          r.similarity.toFixed(4),
          new Date(r.recordedAt).toISOString(),
          session.id,
          JSON.stringify(session.label || ""),
        ].join(",")
      )
      .join("\n");

    res.setHeader("Content-Type", "text/csv; charset=utf-8");
    res.setHeader(
      "Content-Disposition",
      `attachment; filename="attendance-${session.id}.csv"`
    );
    return res.send(header + rows + (rows ? "\n" : ""));
  });

  // ── End a session ─────────────────────────────────────────────────
  router.post("/sessions/:id/end", requireFaculty, (req, res, next) => {
    const session = store.endSession(req.params.id);
    if (!session) {
      return next(ApiError.notFound("SESSION_NOT_FOUND", "No such session."));
    }

    logger.info("session ended", {
      sessionId: session.id,
      recorded: session.attendance.size,
    });

    return res.json({
      ok: true,
      sessionId: session.id,
      recorded: session.attendance.size,
    });
  });

  return router;
}

module.exports = { createSessionRouter };
