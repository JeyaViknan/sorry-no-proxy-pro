"use strict";

/**
 * In-memory state for live attendance sessions.
 *
 * Holds four things:
 *   • sessions          — what the faculty started, and when it ends
 *   • scan ledger       — replay protection for (session, slot, device)
 *   • consumed tokens   — single-use enforcement for attendance tokens
 *   • attendance ledger — the authoritative record, plus attempt counters
 *
 * DELIBERATE CHOICE: this is the source of truth, and Google Sheets is a
 * downstream export (see sheets.js). The old code wrote straight to Sheets
 * inside the request, so a Sheets hiccup meant a student's attendance was
 * simply lost and they saw "Server error. Please try again." Recording
 * locally first makes the write path fast and the export retryable.
 *
 * SCALING NOTE: a Map is correct for a single instance, which comfortably
 * covers a few thousand students per class. Horizontal scaling (Cloud Run
 * with >1 instance) requires shared state — swap this class for a Redis
 * implementation exposing the same methods. Every consumer goes through
 * this interface precisely so that swap stays local.
 */

const SWEEP_INTERVAL_MS = 60_000;

class SessionStore {
  constructor({ sessionTtlMs, attendanceTokenTtlMs, qrTokenTtlMs }) {
    this.sessionTtlMs = sessionTtlMs;
    this.attendanceTokenTtlMs = attendanceTokenTtlMs;
    this.qrTokenTtlMs = qrTokenTtlMs;

    /** @type {Map<string, object>} */
    this.sessions = new Map();
    /** @type {Map<string, number>} key -> expiry ms */
    this.scanLedger = new Map();
    /** @type {Map<string, number>} jti -> expiry ms */
    this.consumedTokens = new Map();

    this.sweepTimer = setInterval(() => this.sweep(), SWEEP_INTERVAL_MS);
    // Never hold the event loop open just to run housekeeping.
    if (this.sweepTimer.unref) this.sweepTimer.unref();
  }

  // ── Sessions ──────────────────────────────────────────────────────

  createSession({ id, label = "", createdBy = "faculty", nowMs = Date.now() }) {
    const session = {
      id,
      label,
      createdBy,
      createdAt: nowMs,
      expiresAt: nowMs + this.sessionTtlMs,
      endedAt: null,
      /** @type {Map<string, object>} registerNumber -> record */
      attendance: new Map(),
      /** @type {Map<string, number>} registerNumber -> failed IDENTITY attempts */
      attempts: new Map(),
      /** @type {Map<string, number>} registerNumber -> unreadable-photo attempts */
      qualityAttempts: new Map(),
    };
    this.sessions.set(id, session);
    return session;
  }

  getSession(id, nowMs = Date.now()) {
    const session = this.sessions.get(id);
    if (!session) return null;
    if (session.endedAt !== null) return null;
    if (nowMs > session.expiresAt) return null;
    return session;
  }

  endSession(id, nowMs = Date.now()) {
    const session = this.sessions.get(id);
    if (!session) return null;
    session.endedAt = nowMs;
    return session;
  }

  // ── QR replay protection ──────────────────────────────────────────

  /**
   * A projected QR is scanned by the whole class at once, so a valid slot
   * cannot be globally single-use. It is single-use *per device* instead.
   *
   * Returns { firstUse } rather than rejecting on repeat: a camera fires
   * many frames per second and will legitimately decode the same slot
   * twice before the UI advances. Treating that as an attack produces
   * spurious failures; treating it as idempotent does not weaken anything,
   * because the caller reissues the same short-lived token.
   */
  recordScan({ sessionId, slot, deviceId, nowMs = Date.now() }) {
    const key = `${sessionId}:${slot}:${deviceId}`;
    const existing = this.scanLedger.get(key);
    if (existing !== undefined && existing > nowMs) {
      return { firstUse: false };
    }
    // Retain past the slot itself so a late duplicate is still recognised.
    this.scanLedger.set(key, nowMs + this.qrTokenTtlMs + this.attendanceTokenTtlMs);
    return { firstUse: true };
  }

  // ── Attendance token single-use ───────────────────────────────────

  /** @returns {boolean} true if this token had not been consumed before. */
  consumeToken(jti, nowMs = Date.now()) {
    const existing = this.consumedTokens.get(jti);
    if (existing !== undefined && existing > nowMs) return false;
    this.consumedTokens.set(jti, nowMs + this.attendanceTokenTtlMs);
    return true;
  }

  // ── Attendance ledger ─────────────────────────────────────────────

  getAttendance(session, registerNumber) {
    return session.attendance.get(registerNumber) || null;
  }

  recordAttendance(session, record) {
    const existing = session.attendance.get(record.registerNumber);
    if (existing) return { duplicate: true, record: existing };
    session.attendance.set(record.registerNumber, record);
    return { duplicate: false, record };
  }

  countAttempts(session, registerNumber) {
    return session.attempts.get(registerNumber) || 0;
  }

  incrementAttempts(session, registerNumber) {
    const next = this.countAttempts(session, registerNumber) + 1;
    session.attempts.set(registerNumber, next);
    return next;
  }

  // Quality failures are tracked separately so an unreadable photo never
  // consumes the identity budget. See routes/attendance.js.
  countQualityAttempts(session, registerNumber) {
    return session.qualityAttempts.get(registerNumber) || 0;
  }

  incrementQualityAttempts(session, registerNumber) {
    const next = this.countQualityAttempts(session, registerNumber) + 1;
    session.qualityAttempts.set(registerNumber, next);
    return next;
  }

  // ── Housekeeping ──────────────────────────────────────────────────

  sweep(nowMs = Date.now()) {
    let removed = 0;

    for (const [key, expiry] of this.scanLedger) {
      if (expiry <= nowMs) {
        this.scanLedger.delete(key);
        removed += 1;
      }
    }
    for (const [jti, expiry] of this.consumedTokens) {
      if (expiry <= nowMs) {
        this.consumedTokens.delete(jti);
        removed += 1;
      }
    }
    for (const [id, session] of this.sessions) {
      // Keep ended sessions around briefly so late requests get a clear
      // "session ended" rather than a confusing "not found".
      const graceExpiry = (session.endedAt || session.expiresAt) + this.sessionTtlMs;
      if (nowMs > graceExpiry) {
        this.sessions.delete(id);
        removed += 1;
      }
    }
    return removed;
  }

  stats() {
    let liveSessions = 0;
    const now = Date.now();
    for (const session of this.sessions.values()) {
      if (session.endedAt === null && now <= session.expiresAt) liveSessions += 1;
    }
    return {
      sessions: this.sessions.size,
      liveSessions,
      scanLedger: this.scanLedger.size,
      consumedTokens: this.consumedTokens.size,
    };
  }

  close() {
    clearInterval(this.sweepTimer);
  }
}

module.exports = { SessionStore };
