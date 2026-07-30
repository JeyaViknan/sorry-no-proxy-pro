"use strict";

/**
 * Faculty authentication.
 *
 * A shared access code is exchanged once for a signed, expiring bearer token
 * that authorises session control. Deliberately not a user database: this
 * system has no per-faculty state worth storing, and a code rotated each
 * semester is both sufficient and operationally simple. If per-faculty
 * accountability is ever needed, the token claims already carry a subject
 * field to hang it on.
 *
 * The code is compared in constant time and its endpoint is rate limited
 * (10 attempts / 15 min / IP) because it is the one secret a person might
 * plausibly guess.
 */

const { hmac, safeEqual, randomId } = require("../services/crypto");
const { ApiError } = require("./errors");

const SIGNATURE_BYTES = 32;
const FACULTY_TOKEN_TTL_MS = 6 * 60 * 60 * 1000; // one teaching day

class FacultyAuth {
  constructor({ secret, accessCode }) {
    this.secret = secret;
    this.accessCode = accessCode;
  }

  verifyAccessCode(candidate) {
    if (typeof candidate !== "string" || candidate.length === 0) return false;
    return safeEqual(candidate, this.accessCode);
  }

  issueToken({ subject = "faculty", nowMs = Date.now() } = {}) {
    const claims = {
      v: 1,
      sub: subject,
      jti: randomId(12),
      iat: nowMs,
      exp: nowMs + FACULTY_TOKEN_TTL_MS,
    };
    const body = Buffer.from(JSON.stringify(claims)).toString("base64url");
    const signature = hmac(this.secret, `faculty:${body}`, SIGNATURE_BYTES).toString("base64url");
    return { token: `${body}.${signature}`, expiresAt: claims.exp };
  }

  verifyToken(token, nowMs = Date.now()) {
    if (typeof token !== "string" || token.length > 512) return null;

    const parts = token.split(".");
    if (parts.length !== 2) return null;

    const [body, signature] = parts;
    const expected = hmac(this.secret, `faculty:${body}`, SIGNATURE_BYTES).toString("base64url");
    if (!safeEqual(signature, expected)) return null;

    try {
      const claims = JSON.parse(Buffer.from(body, "base64url").toString("utf8"));
      if (claims.v !== 1) return null;
      if (typeof claims.exp !== "number" || nowMs > claims.exp) return null;
      return claims;
    } catch {
      return null;
    }
  }

  /** Express middleware guarding faculty-only routes. */
  requireFaculty() {
    return (req, res, next) => {
      const header = req.get("Authorization") || "";
      const token = header.startsWith("Bearer ") ? header.slice(7) : null;

      if (!token) {
        return next(
          ApiError.unauthorized("FACULTY_AUTH_REQUIRED", "Sign in to control a session.")
        );
      }

      const claims = this.verifyToken(token);
      if (!claims) {
        return next(
          ApiError.unauthorized(
            "FACULTY_TOKEN_INVALID",
            "Your session has expired. Please sign in again."
          )
        );
      }

      req.faculty = claims;
      return next();
    };
  }
}

module.exports = { FacultyAuth, FACULTY_TOKEN_TTL_MS };
