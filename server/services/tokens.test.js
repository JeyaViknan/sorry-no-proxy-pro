"use strict";

/**
 * Tests for the security-critical token logic.
 *
 * These are the pieces where a subtle bug means either "anyone can mark
 * attendance" or "nobody can", so they are the ones worth pinning down.
 * Run with: npm test
 */

const test = require("node:test");
const assert = require("node:assert/strict");

const { QrTokenService, PAYLOAD_LENGTH } = require("./qrToken");
const { AttendanceTokenService } = require("./attendanceToken");
const { SessionStore } = require("./sessionStore");
const { base32Encode, base32Decode, safeEqual } = require("./crypto");

const SECRET = "test-qr-secret-at-least-24-chars-long";
const OTHER_SECRET = "different-secret-at-least-24-chars-long";

function makeQr(overrides = {}) {
  return new QrTokenService({
    secret: SECRET,
    ttlMs: 4000,
    clockSkewMs: 2000,
    ...overrides,
  });
}

// ── base32 ───────────────────────────────────────────────────────────

test("base32 round-trips arbitrary bytes", () => {
  for (const length of [1, 5, 10, 32]) {
    const input = Buffer.from(Array.from({ length }, (_, i) => (i * 37) % 256));
    const decoded = base32Decode(base32Encode(input));
    assert.deepEqual(decoded.subarray(0, length), input);
  }
});

test("base32 output stays inside the QR alphanumeric charset", () => {
  const encoded = base32Encode(Buffer.from("attendance payload sample"));
  assert.match(encoded, /^[A-Z2-7]+$/);
});

// ── constant-time compare ────────────────────────────────────────────

test("safeEqual handles equal, unequal and differing-length inputs", () => {
  assert.equal(safeEqual("abc", "abc"), true);
  assert.equal(safeEqual("abc", "abd"), false);
  assert.equal(safeEqual("abc", "abcdefgh"), false, "must not throw on length mismatch");
  assert.equal(safeEqual("", ""), true);
});

// ── QR tokens ────────────────────────────────────────────────────────

test("a freshly signed payload verifies inside its slot", () => {
  const qr = makeQr();
  const now = Date.now();
  const payload = qr.sign("ABCDEFGH", qr.slotFor(now));

  const result = qr.verify(payload, now);
  assert.equal(result.ok, true);
  assert.equal(result.sessionId, "ABCDEFGH");
});

test("payload is fixed width so decoys are indistinguishable", () => {
  const qr = makeQr();
  for (const slot of [1, 1000, qr.slotFor(Date.now()), 99999999]) {
    assert.equal(qr.sign("ABCDEFGH", slot).length, PAYLOAD_LENGTH);
  }
});

test("a payload signed with another key is rejected", () => {
  const qr = makeQr();
  const forger = makeQr({ secret: OTHER_SECRET });
  const slot = qr.slotFor(Date.now());

  const result = qr.verify(forger.sign("ABCDEFGH", slot), Date.now());
  assert.equal(result.ok, false);
  assert.equal(result.reason, "BAD_SIGNATURE");
});

test("tampering with any field invalidates the payload", () => {
  const qr = makeQr();
  const now = Date.now();
  const payload = qr.sign("ABCDEFGH", qr.slotFor(now));
  const [sid, slot, sig] = payload.split(".");

  assert.equal(qr.verify(`ZZZZZZZZ.${slot}.${sig}`, now).ok, false);
  assert.equal(qr.verify(`${sid}.00000001.${sig}`, now).ok, false);
  assert.equal(qr.verify(`${sid}.${slot}.AAAAAAAAAAAAAAAA`, now).ok, false);
});

test("a payload expires once its slot plus skew has passed", () => {
  const qr = makeQr(); // ttl 4000, skew 2000
  // Anchor to a slot boundary so the assertions are exact rather than
  // depending on where inside the current slot the wall clock happens to sit.
  const slot = qr.slotFor(Date.now());
  const slotStart = qr.startOfSlot(slot);
  const payload = qr.sign("ABCDEFGH", slot);

  assert.equal(qr.verify(payload, slotStart).ok, true, "start of slot");
  assert.equal(qr.verify(payload, slotStart + 3999).ok, true, "end of slot");
  assert.equal(qr.verify(payload, slotStart + 5999).ok, true, "inside skew grace");

  const late = qr.verify(payload, slotStart + 6001);
  assert.equal(late.ok, false, "past slot + skew");
  assert.equal(late.reason, "EXPIRED");

  const early = qr.verify(payload, slotStart - 2001);
  assert.equal(early.ok, false, "before slot - skew");
  assert.equal(early.reason, "NOT_YET_VALID");
});

test("the old scheme's six-minute window is gone", () => {
  const qr = makeQr();
  const now = Date.now();
  const payload = qr.sign("ABCDEFGH", qr.slotFor(now));

  // The previous implementation accepted now-1min .. now+4min.
  assert.equal(qr.verify(payload, now + 60_000).ok, false, "1 minute later must fail");
  assert.equal(qr.verify(payload, now + 240_000).ok, false, "4 minutes later must fail");
});

test("verification uses the server clock, not a client-supplied time", () => {
  const qr = makeQr();
  const realNow = Date.now();
  // A student setting their device clock to class time cannot help them: the
  // server passes its own Date.now(), and a payload for a past slot fails.
  const stalePayload = qr.sign("ABCDEFGH", qr.slotFor(realNow - 3_600_000));
  assert.equal(qr.verify(stalePayload, realNow).ok, false);
});

test("batch covers a span contiguously and every token verifies in its slot", () => {
  const qr = makeQr();
  const from = Date.now();
  const tokens = qr.batch("ABCDEFGH", from, 20_000);

  assert.ok(tokens.length >= 5);
  for (let i = 1; i < tokens.length; i += 1) {
    assert.equal(tokens[i].slot, tokens[i - 1].slot + 1, "no gaps between slots");
  }
  for (const token of tokens) {
    const midSlot = token.notBefore + 1;
    assert.equal(qr.verify(token.payload, midSlot).ok, true);
  }
});

// ── Attendance tokens ────────────────────────────────────────────────

test("an issued attendance token verifies for the same device", () => {
  const svc = new AttendanceTokenService({ secret: SECRET, ttlMs: 120_000 });
  const { token } = svc.issue({ sessionId: "ABCDEFGH", deviceId: "device-1234", slot: 5 });

  const result = svc.verify(token, { deviceId: "device-1234" });
  assert.equal(result.ok, true);
  assert.equal(result.claims.sid, "ABCDEFGH");
});

test("a token captured on one device cannot be replayed from another", () => {
  const svc = new AttendanceTokenService({ secret: SECRET, ttlMs: 120_000 });
  const { token } = svc.issue({ sessionId: "ABCDEFGH", deviceId: "device-1234", slot: 5 });

  const result = svc.verify(token, { deviceId: "device-9999" });
  assert.equal(result.ok, false);
  assert.equal(result.reason, "DEVICE_MISMATCH");
});

test("an attendance token expires", () => {
  const svc = new AttendanceTokenService({ secret: SECRET, ttlMs: 1000 });
  const now = Date.now();
  const { token } = svc.issue({ sessionId: "ABCDEFGH", deviceId: "device-1234", slot: 5, nowMs: now });

  assert.equal(svc.verify(token, { deviceId: "device-1234", nowMs: now + 5000 }).reason, "EXPIRED");
});

test("claims cannot be rewritten without the signing key", () => {
  const svc = new AttendanceTokenService({ secret: SECRET, ttlMs: 120_000 });
  const { token } = svc.issue({ sessionId: "ABCDEFGH", deviceId: "device-1234", slot: 5 });
  const [body, signature] = token.split(".");

  const claims = JSON.parse(Buffer.from(body, "base64url").toString("utf8"));
  claims.exp = Date.now() + 10 ** 9; // try to extend lifetime
  const forgedBody = Buffer.from(JSON.stringify(claims)).toString("base64url");

  const result = svc.verify(`${forgedBody}.${signature}`, { deviceId: "device-1234" });
  assert.equal(result.ok, false);
  assert.equal(result.reason, "BAD_SIGNATURE");
});

test("garbage input is rejected without throwing", () => {
  const svc = new AttendanceTokenService({ secret: SECRET, ttlMs: 120_000 });
  for (const bad of ["", "x", "a.b.c", "....", "a".repeat(600), null, undefined, 42]) {
    const result = svc.verify(bad, { deviceId: "device-1234" });
    assert.equal(result.ok, false);
  }
});

// ── Session store ────────────────────────────────────────────────────

test("attendance tokens are single-use", () => {
  const store = new SessionStore({
    sessionTtlMs: 3600_000,
    attendanceTokenTtlMs: 120_000,
    qrTokenTtlMs: 4000,
  });

  assert.equal(store.consumeToken("jti-1"), true, "first use succeeds");
  assert.equal(store.consumeToken("jti-1"), false, "replay is refused");
  store.close();
});

test("the same device rescanning one slot is idempotent, not an error", () => {
  const store = new SessionStore({
    sessionTtlMs: 3600_000,
    attendanceTokenTtlMs: 120_000,
    qrTokenTtlMs: 4000,
  });

  // A camera decodes the same projected code several times per second.
  assert.equal(store.recordScan({ sessionId: "S", slot: 1, deviceId: "d1" }).firstUse, true);
  assert.equal(store.recordScan({ sessionId: "S", slot: 1, deviceId: "d1" }).firstUse, false);
  // A different device scanning the same projected code is normal — the whole
  // class does it at once.
  assert.equal(store.recordScan({ sessionId: "S", slot: 1, deviceId: "d2" }).firstUse, true);
  store.close();
});

test("attendance is recorded once per registration number", () => {
  const store = new SessionStore({
    sessionTtlMs: 3600_000,
    attendanceTokenTtlMs: 120_000,
    qrTokenTtlMs: 4000,
  });
  const session = store.createSession({ id: "ABCDEFGH" });

  const first = store.recordAttendance(session, { registerNumber: "25BCE1276", status: "accepted" });
  assert.equal(first.duplicate, false);

  const second = store.recordAttendance(session, { registerNumber: "25BCE1276", status: "accepted" });
  assert.equal(second.duplicate, true);
  store.close();
});

test("expired sessions stop resolving", () => {
  const store = new SessionStore({
    sessionTtlMs: 1000,
    attendanceTokenTtlMs: 120_000,
    qrTokenTtlMs: 4000,
  });
  const now = Date.now();
  store.createSession({ id: "ABCDEFGH", nowMs: now });

  assert.ok(store.getSession("ABCDEFGH", now));
  assert.equal(store.getSession("ABCDEFGH", now + 5000), null);
  store.close();
});

test("sweep releases expired entries", () => {
  const store = new SessionStore({
    sessionTtlMs: 1000,
    attendanceTokenTtlMs: 1000,
    qrTokenTtlMs: 1000,
  });
  const now = Date.now();

  store.consumeToken("jti-a", now);
  store.recordScan({ sessionId: "S", slot: 1, deviceId: "d1", nowMs: now });
  assert.ok(store.stats().consumedTokens + store.stats().scanLedger >= 2);

  store.sweep(now + 60_000);
  assert.equal(store.stats().consumedTokens, 0);
  assert.equal(store.stats().scanLedger, 0);
  store.close();
});
