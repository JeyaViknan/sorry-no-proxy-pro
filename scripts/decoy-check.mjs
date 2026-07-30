#!/usr/bin/env node
/**
 * Verifies the decoy invariant against a live server.
 *
 * The property under test: a decoy must be indistinguishable from a real
 * payload to the eye (same length, same alphabet, therefore the same QR
 * version and module density) while never being accepted by the server.
 *
 * This mattered in the old build: decoys were ~90-character JSON blobs next
 * to a 22-character valid string, so the valid QR had visibly fewer modules
 * and could be picked out without decoding anything.
 */

import { makeDecoy } from "../QR-Faculty-Portal/src/lib/session.js";

const BASE = process.env.BASE_URL || "http://localhost:7860";
const CODE = process.env.FACULTY_ACCESS_CODE || "dev-faculty-code-123";

const post = async (path, body, token) =>
  (
    await fetch(`${BASE}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(body),
    })
  ).json();

const login = await post("/api/faculty/login", { accessCode: CODE });
const session = await post("/api/sessions", { label: "decoy check" }, login.facultyToken);

const spec = session.qr.noiseSpec;
const real = session.tokens.map((t) => t.payload);
const decoys = Array.from({ length: 500 }, () => makeDecoy(spec, session.sessionId));

const SHAPE = /^[A-Z2-7]{8}\.[0-9A-Z]{8}\.[A-Z2-7]{16}$/;
const lengths = [...new Set([...real, ...decoys].map((p) => p.length))];

console.log(`\nreal payload   e.g.  ${real[0]}`);
console.log(`decoy          e.g.  ${decoys[0]}`);
console.log(`\ndistinct lengths across real + decoys : ${JSON.stringify(lengths)}`);
console.log(`all match the payload shape           : ${[...real, ...decoys].every((p) => SHAPE.test(p))}`);
console.log(`decoys are unique                     : ${new Set(decoys).size === decoys.length}`);
console.log(`no decoy collides with a real payload : ${decoys.every((d) => !real.includes(d))}`);

let accepted = 0;
for (const decoy of decoys.slice(0, 25)) {
  const result = await post("/api/qr/validate", { payload: decoy, deviceId: "decoy-test-device-1" });
  if (result.valid) accepted += 1;
}
console.log(`decoys accepted by the server         : ${accepted}  (must be 0)`);

const ok = lengths.length === 1 && accepted === 0;
console.log(`\n${ok ? "PASS" : "FAIL"}: decoys are visually identical and cryptographically useless\n`);
process.exit(ok ? 0 : 1);
