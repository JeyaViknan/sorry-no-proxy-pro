#!/usr/bin/env node
/**
 * End-to-end protocol test against a running server.
 *
 *   node server/index.js &
 *   node scripts/integration-test.mjs
 *
 * Exercises the real HTTP surface the two apps use, including the attack
 * paths that worked against the previous version. Face verification is
 * expected to be unavailable in a bare dev environment (no model), so the
 * final step asserts the correct *failure* mode rather than a match.
 */

const BASE = process.env.BASE_URL || "http://localhost:7860";
const ACCESS_CODE = process.env.FACULTY_ACCESS_CODE || "dev-faculty-code-123";

let passed = 0;
let failed = 0;

function check(name, condition, detail = "") {
  if (condition) {
    passed += 1;
    console.log(`  \x1b[32mPASS\x1b[0m ${name}`);
  } else {
    failed += 1;
    console.log(`  \x1b[31mFAIL\x1b[0m ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

async function api(path, { method = "GET", body, token } = {}) {
  const response = await fetch(`${BASE}${path}`, {
    method,
    headers: {
      ...(body ? { "Content-Type": "application/json" } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text.slice(0, 200) };
  }
  return { status: response.status, data };
}

console.log(`\nIntegration test against ${BASE}\n`);

// ── Legacy attack surface ────────────────────────────────────────────
console.log("Legacy attack surface (must all be closed)");
{
  const bare = await api("/register", {
    method: "POST",
    body: { registerNumber: "25BCE1276" },
  });
  check("POST /register no longer exists", bare.status === 404, `got ${bare.status}`);

  const verify = await api("/verify-face", {
    method: "POST",
    body: { registerNumber: "25BCE1276", faceImage: "x" },
  });
  check("POST /verify-face no longer exists", verify.status === 404, `got ${verify.status}`);

  for (const path of ["/server.js", "/package.json", "/.env", "/gallery/face_db.pkl",
                      "/gallery/images/25BCE1276.png", "/python/verifier.py"]) {
    const response = await fetch(`${BASE}${path}`);
    check(`GET ${path} is not served`, response.status === 404, `got ${response.status}`);
  }

  const legacyQr = await api("/api/qr/validate", {
    method: "POST",
    body: { payload: "jeycavbhakanadiyaz1430", deviceId: "attacker-device-0001" },
  });
  check(
    "old hardcoded QR string is rejected",
    legacyQr.data.valid === false,
    JSON.stringify(legacyQr.data)
  );
}

// ── Faculty auth ─────────────────────────────────────────────────────
console.log("\nFaculty authentication");
let facultyToken;
{
  const bad = await api("/api/faculty/login", {
    method: "POST",
    body: { accessCode: "wrong-code" },
  });
  check("wrong access code is rejected", bad.status === 401, `got ${bad.status}`);

  const unauth = await api("/api/sessions", { method: "POST", body: { label: "x" } });
  check("session creation requires auth", unauth.status === 401, `got ${unauth.status}`);

  const good = await api("/api/faculty/login", {
    method: "POST",
    body: { accessCode: ACCESS_CODE },
  });
  check("correct access code issues a token", good.status === 200 && !!good.data.facultyToken);
  facultyToken = good.data.facultyToken;

  const forged = await api("/api/sessions", {
    method: "POST",
    body: { label: "x" },
    token: `${facultyToken}tampered`,
  });
  check("tampered faculty token is rejected", forged.status === 401, `got ${forged.status}`);
}

// ── Session + token batch ────────────────────────────────────────────
console.log("\nSession lifecycle");
let sessionId;
let tokens;
{
  const created = await api("/api/sessions", {
    method: "POST",
    body: { label: "Integration Test" },
    token: facultyToken,
  });
  check("session created", created.status === 201, `got ${created.status}`);
  sessionId = created.data.sessionId;
  tokens = created.data.tokens;

  check("first token batch ships with the create response", Array.isArray(tokens) && tokens.length > 1,
    `${tokens?.length} tokens`);
  check("payloads are fixed-width (decoys indistinguishable)",
    tokens.every((t) => t.payload.length === 34));
  check("payloads use QR alphanumeric charset only",
    tokens.every((t) => /^[A-Z2-7]{8}\.[0-9A-Z]{8}\.[A-Z2-7]{16}$/.test(t.payload)));
  check("slots are contiguous",
    tokens.every((t, i) => i === 0 || t.slot === tokens[i - 1].slot + 1));
  check("noise spec is published for decoy generation",
    created.data.qr?.noiseSpec?.length === 34);
}

// ── Student flow ─────────────────────────────────────────────────────
console.log("\nStudent flow");
let attendanceToken;
{
  const now = Date.now();
  const live = tokens.find((t) => now >= t.notBefore && now < t.notAfter) || tokens[0];

  const forgedSig = `${live.payload.slice(0, 18)}AAAAAAAAAAAAAAAA`;
  const forged = await api("/api/qr/validate", {
    method: "POST",
    body: { payload: forgedSig, deviceId: "student-device-0001" },
  });
  check("forged signature is rejected", forged.data.valid === false);

  const future = tokens[tokens.length - 1];
  const early = await api("/api/qr/validate", {
    method: "POST",
    body: { payload: future.payload, deviceId: "student-device-0001" },
  });
  check("a future slot is not yet valid", early.data.valid === false,
    JSON.stringify(early.data.code));

  const valid = await api("/api/qr/validate", {
    method: "POST",
    body: { payload: live.payload, deviceId: "student-device-0001" },
  });
  check("current payload validates", valid.data.valid === true, JSON.stringify(valid.data));
  attendanceToken = valid.data.attendanceToken;
  check("attendance token issued", !!attendanceToken);

  const again = await api("/api/qr/validate", {
    method: "POST",
    body: { payload: live.payload, deviceId: "student-device-0001" },
  });
  check("rescanning the same slot is idempotent, not an error", again.data.valid === true);

  const otherDevice = await api("/api/qr/validate", {
    method: "POST",
    body: { payload: live.payload, deviceId: "student-device-0002" },
  });
  check("a different device can scan the same projected code",
    otherDevice.data.valid === true);
}

// ── Attendance submission ────────────────────────────────────────────
console.log("\nAttendance submission");
{
  const tinyJpeg =
    "data:image/jpeg;base64," + Buffer.alloc(4096, 0x41).toString("base64");

  const noToken = await api("/api/attendance", {
    method: "POST",
    body: { deviceId: "student-device-0001", registerNumber: "25BCE1276", frames: [tinyJpeg] },
  });
  check("submission without a token is refused", noToken.status === 401, `got ${noToken.status}`);

  const wrongDevice = await api("/api/attendance", {
    method: "POST",
    body: {
      attendanceToken,
      deviceId: "student-device-9999",
      registerNumber: "25BCE1276",
      frames: [tinyJpeg],
    },
  });
  check("token stolen to another device is refused",
    wrongDevice.status === 401 && wrongDevice.data.code === "TOKEN_DEVICE_MISMATCH",
    JSON.stringify(wrongDevice.data.code));

  const badRegno = await api("/api/attendance", {
    method: "POST",
    body: { attendanceToken, deviceId: "student-device-0001", registerNumber: "hello", frames: [tinyJpeg] },
  });
  check("malformed registration number is refused",
    badRegno.status === 400 && badRegno.data.code === "REGNO_INVALID");

  const noFrames = await api("/api/attendance", {
    method: "POST",
    body: { attendanceToken, deviceId: "student-device-0001", registerNumber: "25BCE1276", frames: [] },
  });
  check("submission without frames is refused", noFrames.data.code === "FRAMES_REQUIRED");

  const tooMany = await api("/api/attendance", {
    method: "POST",
    body: {
      attendanceToken,
      deviceId: "student-device-0001",
      registerNumber: "25BCE1276",
      frames: Array(9).fill(tinyJpeg),
    },
  });
  check("frame count is capped", tooMany.data.code === "TOO_MANY_FRAMES");

  const real = await api("/api/attendance", {
    method: "POST",
    body: { attendanceToken, deviceId: "student-device-0001", registerNumber: "25BCE1276", frames: [tinyJpeg] },
  });
  // Without a model loaded the correct behaviour is a clean 503, never a
  // silent pass and never a stack trace.
  check("verifier unavailable degrades cleanly",
    real.status === 503 && real.data.code === "VERIFIER_UNAVAILABLE",
    `got ${real.status} ${real.data.code}`);
  check("no internal details leak in the error",
    !JSON.stringify(real.data).match(/Traceback|\/Users\/|node_modules|at Object/),
    JSON.stringify(real.data).slice(0, 120));
}

// ── Faculty reporting ────────────────────────────────────────────────
console.log("\nFaculty reporting");
{
  const summary = await api(`/api/sessions/${sessionId}/summary`, { token: facultyToken });
  check("summary is available", summary.status === 200);

  const unauth = await api(`/api/sessions/${sessionId}/summary`);
  check("summary requires auth", unauth.status === 401);

  const csv = await fetch(`${BASE}/api/sessions/${sessionId}/export`, {
    headers: { Authorization: `Bearer ${facultyToken}` },
  });
  check("CSV export works", csv.status === 200 &&
    (await csv.text()).startsWith("register_number,status"));

  const ended = await api(`/api/sessions/${sessionId}/end`, { method: "POST", token: facultyToken });
  check("session can be ended", ended.status === 200);

  const afterEnd = await api("/api/qr/validate", {
    method: "POST",
    body: { payload: tokens[0].payload, deviceId: "student-device-0001" },
  });
  check("codes stop working after the session ends", afterEnd.data.valid === false);
}

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed === 0 ? 0 : 1);
