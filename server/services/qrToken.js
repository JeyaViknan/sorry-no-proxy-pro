"use strict";

/**
 * Rotating, server-signed QR payloads.
 *
 * WHAT CHANGED AND WHY
 * --------------------
 * The old scheme was `"jeycavbhakanadiyaz" + HH + MM`, computed identically
 * and independently by the faculty app (App.jsx) and the scanner (index.html).
 * Three fatal properties:
 *
 *   1. The secret shipped to every student in View Source.
 *   2. The value was derived from the *student's own device clock*, so
 *      setting the phone clock to class time validated from anywhere.
 *   3. The scanner accepted now-1min .. now+4min — six minutes of
 *      simultaneously-valid payloads, making the sub-second display
 *      rotation decorative.
 *
 * The replacement signs a (session, time-slot) pair with a server-held key.
 * Validity is decided by the *server's* clock. A payload is meaningful for
 * one slot (default 4s), so relaying a screenshot out of the room requires
 * real-time collusion rather than a leisurely copy-paste.
 *
 * PAYLOAD FORMAT  (fixed 34 chars, QR alphanumeric mode)
 *
 *     SSSSSSSS.TTTTTTTT.GGGGGGGGGGGGGGGG
 *     |        |        |
 *     |        |        +-- 16 chars base32: HMAC-SHA256 truncated to 10 bytes
 *     |        +----------- 8 chars base36 (upper), zero-padded: time slot
 *     +-------------------- 8 chars base32: session id
 *
 * Fixed width matters: the faculty display fills the gaps between valid
 * codes with noise of the *same* length and alphabet, so a valid QR is not
 * identifiable by having visibly fewer modules than the decoys. The old
 * noise was a ~90-char JSON blob next to a 22-char valid string — trivially
 * distinguishable by eye.
 *
 * 80 bits of truncated HMAC is far beyond forgeable within a 4-second window
 * while keeping the symbol small enough to scan from the back of a hall.
 */

const { base32Encode, hmac, safeEqual } = require("./crypto");

const SESSION_ID_CHARS = 8;
const SLOT_CHARS = 8;
const SIGNATURE_BYTES = 10;
const SIGNATURE_CHARS = 16;
const PAYLOAD_LENGTH = SESSION_ID_CHARS + 1 + SLOT_CHARS + 1 + SIGNATURE_CHARS;

const PAYLOAD_PATTERN = new RegExp(
  `^[A-Z2-7]{${SESSION_ID_CHARS}}\\.[0-9A-Z]{${SLOT_CHARS}}\\.[A-Z2-7]{${SIGNATURE_CHARS}}$`
);

class QrTokenService {
  /**
   * @param {object} options
   * @param {string} options.secret      HMAC key (QR_SIGNING_SECRET)
   * @param {number} options.ttlMs       slot width; how long one payload is valid
   * @param {number} options.clockSkewMs tolerance for faculty/server clock drift
   */
  constructor({ secret, ttlMs, clockSkewMs }) {
    if (!secret) throw new Error("QrTokenService requires a secret");
    this.secret = secret;
    this.ttlMs = ttlMs;
    this.clockSkewMs = clockSkewMs;
  }

  /** Time slot index for a timestamp. Slots are global, not per-session. */
  slotFor(timestampMs) {
    return Math.floor(timestampMs / this.ttlMs);
  }

  startOfSlot(slot) {
    return slot * this.ttlMs;
  }

  #signature(sessionId, slot) {
    const digest = hmac(this.secret, `${sessionId}:${slot}`, SIGNATURE_BYTES);
    return base32Encode(digest);
  }

  #encodeSlot(slot) {
    return slot.toString(36).toUpperCase().padStart(SLOT_CHARS, "0");
  }

  /** Build the signed payload for one (session, slot) pair. */
  sign(sessionId, slot) {
    return `${sessionId}.${this.#encodeSlot(slot)}.${this.#signature(sessionId, slot)}`;
  }

  /**
   * Pre-sign every slot in a time span so the faculty display can run without
   * a network round trip per QR. On unstable classroom wifi, one request per
   * 500ms frame would stall the display constantly; one request per minute
   * does not.
   */
  batch(sessionId, fromMs, spanMs) {
    const firstSlot = this.slotFor(fromMs);
    const lastSlot = this.slotFor(fromMs + spanMs);
    const tokens = [];

    for (let slot = firstSlot; slot <= lastSlot; slot += 1) {
      tokens.push({
        slot,
        payload: this.sign(sessionId, slot),
        notBefore: this.startOfSlot(slot),
        notAfter: this.startOfSlot(slot) + this.ttlMs,
      });
    }
    return tokens;
  }

  /**
   * Verify a scanned payload against the server clock.
   * Returns a discriminated result rather than throwing — every rejection
   * reason maps to a distinct student-facing message.
   */
  verify(payload, nowMs = Date.now()) {
    if (typeof payload !== "string" || payload.length !== PAYLOAD_LENGTH) {
      return { ok: false, reason: "MALFORMED" };
    }
    if (!PAYLOAD_PATTERN.test(payload)) {
      return { ok: false, reason: "MALFORMED" };
    }

    const [sessionId, slotRaw, signature] = payload.split(".");
    const slot = parseInt(slotRaw, 36);
    if (!Number.isSafeInteger(slot) || slot <= 0) {
      return { ok: false, reason: "MALFORMED" };
    }

    // Signature first: never leak whether a session id exists via timing or
    // ordering of checks.
    if (!safeEqual(signature, this.#signature(sessionId, slot))) {
      return { ok: false, reason: "BAD_SIGNATURE" };
    }

    const slotStart = this.startOfSlot(slot);
    const slotEnd = slotStart + this.ttlMs;

    if (nowMs < slotStart - this.clockSkewMs) {
      return { ok: false, reason: "NOT_YET_VALID", sessionId, slot };
    }
    if (nowMs > slotEnd + this.clockSkewMs) {
      return { ok: false, reason: "EXPIRED", sessionId, slot };
    }

    return { ok: true, sessionId, slot, expiresAt: slotEnd };
  }

  /**
   * Describes the payload shape so the faculty client can generate decoys
   * that are indistinguishable from real ones. The client never receives
   * anything that lets it *forge* a valid code — only the format.
   */
  get noiseSpec() {
    return {
      length: PAYLOAD_LENGTH,
      sessionIdChars: SESSION_ID_CHARS,
      slotChars: SLOT_CHARS,
      signatureChars: SIGNATURE_CHARS,
      alphabet: "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567",
      slotAlphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    };
  }
}

module.exports = { QrTokenService, PAYLOAD_LENGTH };
