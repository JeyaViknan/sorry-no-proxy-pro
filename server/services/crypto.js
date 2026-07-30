"use strict";

/**
 * Small crypto helpers shared by the token services.
 *
 * Base32 (RFC 4648, unpadded) is used for anything that ends up inside a QR
 * code. Its alphabet (A-Z, 2-7) is a subset of QR "alphanumeric mode", which
 * encodes at 5.5 bits per character instead of the 8 bits that byte mode
 * needs. In practice a 34-character base32 payload fits in a 25x25 QR
 * (version 2) while the same data in base64url — whose `-` and `_` force
 * byte mode — needs a denser symbol. Fewer modules means physically larger
 * squares on a projector, which is the difference between scanning from the
 * back row and not.
 */

const crypto = require("crypto");

const BASE32_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

function base32Encode(buffer) {
  let bits = 0;
  let value = 0;
  let output = "";

  for (const byte of buffer) {
    value = (value << 8) | byte;
    bits += 8;
    while (bits >= 5) {
      output += BASE32_ALPHABET[(value >>> (bits - 5)) & 31];
      bits -= 5;
    }
  }
  if (bits > 0) {
    output += BASE32_ALPHABET[(value << (5 - bits)) & 31];
  }
  return output;
}

function base32Decode(input) {
  let bits = 0;
  let value = 0;
  const bytes = [];

  for (const char of input) {
    const index = BASE32_ALPHABET.indexOf(char);
    if (index === -1) return null;
    value = (value << 5) | index;
    bits += 5;
    if (bits >= 8) {
      bytes.push((value >>> (bits - 8)) & 0xff);
      bits -= 8;
    }
  }
  return Buffer.from(bytes);
}

/** HMAC-SHA256, truncated to `bytes`. */
function hmac(secret, message, bytes = 32) {
  return crypto.createHmac("sha256", secret).update(message).digest().subarray(0, bytes);
}

/**
 * Constant-time string comparison that also tolerates length mismatch.
 * `crypto.timingSafeEqual` throws on differing lengths, and a naive
 * length check before it leaks length through timing. Hashing both sides
 * to a fixed width removes the branch entirely.
 */
function safeEqual(a, b) {
  const ha = crypto.createHash("sha256").update(String(a)).digest();
  const hb = crypto.createHash("sha256").update(String(b)).digest();
  return crypto.timingSafeEqual(ha, hb);
}

/** URL-safe random identifier. */
function randomId(bytes = 16) {
  return crypto.randomBytes(bytes).toString("base64url");
}

/** Random base32 identifier, for values that must live inside a QR code. */
function randomBase32(bytes = 5) {
  return base32Encode(crypto.randomBytes(bytes));
}

module.exports = {
  base32Encode,
  base32Decode,
  hmac,
  safeEqual,
  randomId,
  randomBase32,
};
