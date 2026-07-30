"use strict";

/**
 * Attendance contract tests with a STUBBED verifier.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * The end-to-end suite runs without a face model, so every verification came
 * back 503 VERIFIER_UNAVAILABLE. That meant the accept / flag / reject
 * branches were never exercised, and a real bug hid there: a face MISMATCH is
 * returned as HTTP 200 with `ok:false` (it is a legitimate answer, not a
 * transport failure), so the client's fetch wrapper resolved instead of
 * throwing and the scanner showed "You are marked present" to a rejected
 * student — defeating the entire system.
 *
 * Injecting a fake verifier is exactly why createApp() is separate from
 * index.js. These tests pin the response contract the client depends on.
 */

const test = require("node:test");
const assert = require("node:assert/strict");

process.env.QR_SIGNING_SECRET ||= "test-qr-secret-at-least-24-chars-long";
process.env.TOKEN_SIGNING_SECRET ||= "test-token-secret-at-least-24-chars-ok";
process.env.FACULTY_ACCESS_CODE ||= "test-faculty-code-1234";
process.env.ALLOWED_ORIGINS ||= "http://localhost";
process.env.NODE_ENV = "test";
process.env.LOG_LEVEL ||= "error";

const { config } = require("../config");
const { createApp } = require("../app");

/** Verifier stub whose next answer the test controls. */
function makeVerifier() {
  const stub = {
    next: null,
    calls: [],
    status: () => ({ ready: true, workers: [], queued: 0, stats: {} }),
    async verify(input) {
      stub.calls.push(input);
      if (stub.next instanceof Error) throw stub.next;
      return stub.next;
    },
  };
  return stub;
}

/** Boot the app on an ephemeral port and return a small fetch helper. */
async function withServer(run) {
  const faceVerifier = makeVerifier();
  const app = createApp({ config, faceVerifier });
  const server = app.listen(0, "127.0.0.1");
  await new Promise((resolve) => server.once("listening", resolve));
  const base = `http://127.0.0.1:${server.address().port}`;

  const api = async (path, { method = "GET", body, token } = {}) => {
    const response = await fetch(`${base}${path}`, {
      method,
      headers: {
        ...(body ? { "Content-Type": "application/json" } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
    });
    const text = await response.text();
    return { status: response.status, data: text ? JSON.parse(text) : {} };
  };

  try {
    await run({ api, faceVerifier, app });
  } finally {
    await new Promise((resolve) => server.close(resolve));
    app.locals.services.store.close();
  }
}

/** Drive faculty login -> session -> QR scan -> attendance token. */
let deviceCounter = 0;

/**
 * Each test gets a fresh device id. The rate-limit stores are created at
 * module load, so they are shared across createApp() instances — a reused id
 * would leak limiter state between tests and produce spurious 429s.
 */
async function primeToken(api, { deviceId = `test-device-${String(++deviceCounter).padStart(6, "0")}` } = {}) {
  const login = await api("/api/faculty/login", {
    method: "POST",
    body: { accessCode: process.env.FACULTY_ACCESS_CODE },
  });
  const session = await api("/api/sessions", {
    method: "POST",
    body: { label: "Contract Test" },
    token: login.data.facultyToken,
  });

  const now = Date.now();
  const live =
    session.data.tokens.find((t) => now >= t.notBefore && now < t.notAfter) ||
    session.data.tokens[0];

  const validated = await api("/api/qr/validate", {
    method: "POST",
    body: { payload: live.payload, deviceId },
  });

  return {
    facultyToken: login.data.facultyToken,
    sessionId: session.data.sessionId,
    attendanceToken: validated.data.attendanceToken,
    deviceId,
  };
}

const FRAME = `data:image/jpeg;base64,${Buffer.alloc(4096, 0x41).toString("base64")}`;

const submit = (api, ctx, overrides = {}) =>
  api("/api/attendance", {
    method: "POST",
    body: {
      attendanceToken: ctx.attendanceToken,
      deviceId: ctx.deviceId,
      registerNumber: "25BCE1276",
      frames: [FRAME],
      ...overrides,
    },
  });

// ── The regression that shipped ──────────────────────────────────────

test("a face MISMATCH returns 200 with ok:false and status 'rejected'", async () => {
  await withServer(async ({ api, faceVerifier }) => {
    const ctx = await primeToken(api);
    faceVerifier.next = {
      ok: false,
      reason: "NO_MATCH",
      message: "We could not match your face to that registration number.",
      similarity: 0.21,
    };

    const response = await submit(api, ctx);

    // HTTP 200 is deliberate — but the body MUST make the rejection
    // unambiguous, because the client's fetch wrapper resolves on 2xx.
    assert.equal(response.status, 200);
    assert.equal(response.data.ok, false, "ok:false is what the client branches on");
    assert.equal(response.data.status, "rejected");
    assert.notEqual(response.data.status, "accepted");
    assert.equal(response.data.alreadyRecorded, undefined);
  });
});

test("a rejected student is NOT recorded as present", async () => {
  await withServer(async ({ api, faceVerifier }) => {
    const ctx = await primeToken(api);
    faceVerifier.next = { ok: false, reason: "NO_MATCH", message: "no", similarity: 0.1 };

    await submit(api, ctx);

    const summary = await api(`/api/sessions/${ctx.sessionId}/summary`, {
      token: ctx.facultyToken,
    });
    assert.equal(summary.data.counts.total, 0, "a rejection must not appear in the ledger");
  });
});

// ── Accept / flag ────────────────────────────────────────────────────

test("a match above the accept threshold is recorded as accepted", async () => {
  await withServer(async ({ api, faceVerifier }) => {
    const ctx = await primeToken(api);
    faceVerifier.next = { ok: true, similarity: 0.82, quality: {} };

    const response = await submit(api, ctx);
    assert.equal(response.status, 200);
    assert.equal(response.data.ok, true);
    assert.equal(response.data.status, "accepted");

    const summary = await api(`/api/sessions/${ctx.sessionId}/summary`, {
      token: ctx.facultyToken,
    });
    assert.equal(summary.data.counts.accepted, 1);
  });
});

test("a match in the review band is recorded but flagged", async () => {
  await withServer(async ({ api, faceVerifier }) => {
    const ctx = await primeToken(api);
    // Between FACE_THRESHOLD_REVIEW (0.42) and FACE_THRESHOLD_ACCEPT (0.50).
    faceVerifier.next = { ok: true, similarity: 0.46, quality: {} };

    const response = await submit(api, ctx);
    assert.equal(response.data.ok, true);
    assert.equal(response.data.status, "flagged");

    const summary = await api(`/api/sessions/${ctx.sessionId}/summary`, {
      token: ctx.facultyToken,
    });
    assert.equal(summary.data.counts.flagged, 1);
  });
});

test("resubmitting after success is idempotent, which is what makes retry safe", async () => {
  await withServer(async ({ api, faceVerifier }) => {
    const ctx = await primeToken(api);
    faceVerifier.next = { ok: true, similarity: 0.9, quality: {} };

    await submit(api, ctx);
    const second = await submit(api, ctx);

    assert.equal(second.data.ok, true);
    assert.equal(second.data.alreadyRecorded, true);

    const summary = await api(`/api/sessions/${ctx.sessionId}/summary`, {
      token: ctx.facultyToken,
    });
    assert.equal(summary.data.counts.total, 1, "no double-recording");
  });
});

// ── Attempt budgets ──────────────────────────────────────────────────

test("quality failures do NOT consume the identity attempt budget", async () => {
  await withServer(async ({ api, faceVerifier }) => {
    const ctx = await primeToken(api);
    faceVerifier.next = {
      ok: false,
      reason: "QUALITY",
      qualityCode: "TOO_BLURRY",
      message: "Hold the phone steady and try again.",
      similarity: 0,
    };

    // More unreadable photos than the identity cap (3) allows.
    for (let i = 0; i < 5; i += 1) {
      const response = await submit(api, ctx);
      assert.equal(response.status, 200, "a blurry photo is never a hard failure");
      assert.equal(response.data.canRetry, true, `attempt ${i + 1} must still be retryable`);
    }

    // A student in bad lighting must still be able to succeed once the photo
    // improves. Previously they were locked out before the model had ever
    // compared a usable face.
    faceVerifier.next = { ok: true, similarity: 0.85, quality: {} };
    const success = await submit(api, ctx);
    assert.equal(success.data.status, "accepted");
  });
});

test("identity mismatches DO consume the budget and then lock out", async () => {
  await withServer(async ({ api, faceVerifier }) => {
    const ctx = await primeToken(api);
    faceVerifier.next = { ok: false, reason: "NO_MATCH", message: "no", similarity: 0.1 };

    for (let i = 0; i < config.attendance.maxAttempts; i += 1) {
      await submit(api, ctx);
    }

    const blocked = await submit(api, ctx);
    assert.equal(blocked.status, 403);
    assert.equal(blocked.data.code, "TOO_MANY_ATTEMPTS");
  });
});

test("a verifier outage neither consumes the token nor counts an attempt", async () => {
  await withServer(async ({ api, faceVerifier }) => {
    const ctx = await primeToken(api);
    faceVerifier.next = new Error("worker died");

    const failed = await submit(api, ctx);
    assert.equal(failed.status, 503);
    assert.equal(failed.data.code, "VERIFIER_UNAVAILABLE");

    // The same token must still work once the verifier recovers — an outage
    // is our fault, and must not cost the student their scan.
    faceVerifier.next = { ok: true, similarity: 0.9, quality: {} };
    const recovered = await submit(api, ctx);
    assert.equal(recovered.data.status, "accepted");
  });
});

// ── Information leaks ────────────────────────────────────────────────

test("the response never leaks similarity, thresholds or the nearest identity", async () => {
  await withServer(async ({ api, faceVerifier }) => {
    const ctx = await primeToken(api);
    faceVerifier.next = {
      ok: false,
      reason: "NO_MATCH",
      message: "We could not match your face to that registration number.",
      similarity: 0.4812,
      // Even if the verifier volunteers it, the route must not forward it.
      bestOtherRegno: "25BRS1286",
    };

    const response = await submit(api, ctx);
    const body = JSON.stringify(response.data);

    assert.ok(!body.includes("25BRS1286"), "must not identify who the photo resembled");
    assert.ok(!body.includes("0.4812"), "must not leak the similarity score");
    assert.ok(!/threshold/i.test(body), "must not leak the operating point");
  });
});

test("an unknown registration number is indistinguishable from a mismatch", async () => {
  await withServer(async ({ api, faceVerifier }) => {
    const ctx = await primeToken(api);
    faceVerifier.next = {
      ok: false,
      reason: "NO_MATCH",
      message: "We could not match your face to that registration number.",
      similarity: 0,
    };

    const response = await submit(api, ctx, { registerNumber: "99ZZZ9999" });
    // Distinguishing them would let anyone enumerate who is enrolled.
    assert.equal(response.data.status, "rejected");
    assert.equal(response.data.code, "NO_MATCH");
  });
});
