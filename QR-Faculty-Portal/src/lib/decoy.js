/**
 * Decoy payload generation.
 *
 * Deliberately separate from session.js, which reads `import.meta.env` — a
 * Vite-only construct that is `undefined` under plain Node. Keeping this pure
 * means `scripts/decoy-check.mjs` can import and verify it directly, without
 * a bundler and without stubbing anything. (It could not, until this split:
 * importing session.js from Node threw on `import.meta.env`.)
 *
 * WHY DECOYS MUST MATCH EXACTLY
 * -----------------------------
 * The old faculty build padded gaps with ~90-character JSON blobs while a
 * valid code was 22 characters. The two rendered at different QR versions, so
 * the valid one was identifiable by having visibly fewer modules — no
 * decoding required, just a glance. A decoy is only useful if it is
 * indistinguishable, so it must reproduce the real payload's exact length and
 * alphabet.
 *
 * The signature bytes are random, so a decoy can never pass server
 * verification. Verified by scripts/decoy-check.mjs.
 */

const BASE32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
const BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

function randomFrom(alphabet, length) {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  let output = "";
  for (let i = 0; i < length; i += 1) output += alphabet[bytes[i] % alphabet.length];
  return output;
}

/**
 * Build a decoy indistinguishable from a real payload.
 *
 * @param {{sessionIdChars: number, slotChars: number, signatureChars: number}} spec
 *        the `qr.noiseSpec` the server returned when the session started
 * @param {string} [sessionId] reuse the real session id so even that segment
 *        matches; only the signature is fabricated
 */
export function makeDecoy(spec, sessionId) {
  const sid = sessionId || randomFrom(BASE32, spec.sessionIdChars);
  const slot = randomFrom(BASE36, spec.slotChars);
  const signature = randomFrom(BASE32, spec.signatureChars);
  return `${sid}.${slot}.${signature}`;
}
