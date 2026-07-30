"use strict";

/**
 * Short-lived, single-use tokens proving "this device scanned a valid QR
 * inside this session, on the server's clock, moments ago".
 *
 * This is the piece the old system had no equivalent of. Previously the
 * client called /verify-face and then separately called /register with only
 * a registration number — two unauthenticated endpoints with nothing binding
 * them together, so `curl -X POST /register -d '{"registerNumber":"..."}'`
 * marked attendance with no QR and no face. The token is what makes
 * attendance submission unforgeable: it can only be obtained by presenting a
 * payload the server itself signed, within its validity window.
 *
 * Verification is stateless (HMAC + expiry). Single-use enforcement is the
 * caller's job via the replay store — keeping the two separate means the
 * token service stays pure and testable, and the store can later move to
 * Redis for multi-instance deployments without touching this file.
 */

const crypto = require("crypto");
const { hmac, safeEqual, randomId } = require("./crypto");

const SIGNATURE_BYTES = 32;

class AttendanceTokenService {
  constructor({ secret, ttlMs }) {
    if (!secret) throw new Error("AttendanceTokenService requires a secret");
    this.secret = secret;
    this.ttlMs = ttlMs;
  }

  /**
   * Device ids come from the client and are therefore untrusted. Hashing
   * them before they enter a token means a stolen token cannot be replayed
   * against a *different* device id by inspecting the claim, and we never
   * persist a raw client-supplied identifier.
   */
  #bindDevice(deviceId) {
    return crypto
      .createHmac("sha256", this.secret)
      .update(`device:${deviceId}`)
      .digest("base64url")
      .slice(0, 16);
  }

  issue({ sessionId, deviceId, slot, nowMs = Date.now() }) {
    const claims = {
      v: 1,
      sid: sessionId,
      dev: this.#bindDevice(deviceId),
      slot,
      jti: randomId(12),
      iat: nowMs,
      exp: nowMs + this.ttlMs,
    };

    const body = Buffer.from(JSON.stringify(claims)).toString("base64url");
    const signature = hmac(this.secret, body, SIGNATURE_BYTES).toString("base64url");

    return {
      token: `${body}.${signature}`,
      jti: claims.jti,
      expiresAt: claims.exp,
    };
  }

  verify(token, { deviceId, nowMs = Date.now() } = {}) {
    if (typeof token !== "string" || token.length > 512) {
      return { ok: false, reason: "MALFORMED" };
    }

    const parts = token.split(".");
    if (parts.length !== 2) return { ok: false, reason: "MALFORMED" };

    const [body, signature] = parts;
    const expected = hmac(this.secret, body, SIGNATURE_BYTES).toString("base64url");
    if (!safeEqual(signature, expected)) {
      return { ok: false, reason: "BAD_SIGNATURE" };
    }

    let claims;
    try {
      claims = JSON.parse(Buffer.from(body, "base64url").toString("utf8"));
    } catch {
      return { ok: false, reason: "MALFORMED" };
    }

    if (claims.v !== 1) return { ok: false, reason: "UNSUPPORTED_VERSION" };
    if (typeof claims.exp !== "number" || nowMs > claims.exp) {
      return { ok: false, reason: "EXPIRED" };
    }

    // Re-binding the presented device id must reproduce the claim. Stops a
    // token captured on one device being replayed from another.
    if (deviceId !== undefined && claims.dev !== this.#bindDevice(deviceId)) {
      return { ok: false, reason: "DEVICE_MISMATCH" };
    }

    return { ok: true, claims };
  }
}

module.exports = { AttendanceTokenService };
