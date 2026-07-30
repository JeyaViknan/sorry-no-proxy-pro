# Attendance Protocol

The contract between the faculty display, the student scanner, and the
backend. Both clients are untrusted; the backend is the only authority.

## Why this replaced the old scheme

The previous design had the faculty app and the scanner each compute
`"jeycavbhakanadiyaz" + HH + MM` independently. Three consequences:

1. The secret shipped to every student in View Source.
2. Validity was decided by **the student's own device clock**, so setting the
   phone clock to class time validated from anywhere, at any later time.
3. The scanner accepted `now-1min … now+4min` — **six minutes** of
   simultaneously valid payloads, which made the sub-second display rotation
   purely decorative.

Net effect: attendance could be marked with one `curl`, from anywhere, with
no QR and no face. The protocol below closes that.

---

## Roles

| Party | Trusted? | Holds a secret? |
|---|---|---|
| Backend | yes — sole authority | `QR_SIGNING_SECRET`, `TOKEN_SIGNING_SECRET`, `FACULTY_ACCESS_CODE` |
| Faculty display | no | a short-lived bearer token, obtained by presenting the access code |
| Student scanner | no | nothing |

The faculty app **cannot mint a valid payload**. It requests pre-signed ones.
Reading its source or its network traffic yields nothing reusable.

---

## QR payload

Fixed 34 characters, entirely within the QR *alphanumeric* charset:

```
RT6KJQ6Q.007DQV9J.RIMRVRMM4BMZ6EU6
│        │        │
│        │        └─ 16 chars base32 — HMAC-SHA256(secret, "sid:slot"), 80 bits
│        └────────── 8 chars base36 upper — time slot, zero-padded
└─────────────────── 8 chars base32 — session id
```

**Why base32.** Its alphabet (`A-Z`, `2-7`) is a subset of QR alphanumeric
mode, which encodes at 5.5 bits/char rather than byte mode's 8. The payload
fits a **25×25 (version 2)** symbol. Base64url's `-` and `_` would force byte
mode and a denser grid. On a projector, fewer modules means physically larger
squares — the difference between scanning from the back row and not.

**Why fixed width.** The display fills the gaps between valid codes with
decoys of *exactly* this shape. The old decoys were ~90-character JSON blobs
beside a 22-character valid string, so the valid QR was identifiable by having
visibly fewer modules — no decoding required. Verified by
`scripts/decoy-check.mjs`.

**Slot.** `floor(epoch_ms / QR_TOKEN_TTL_MS)`, default 4 s. A payload is valid
only for its own slot, ±`QR_CLOCK_SKEW_MS`, judged by the **server's** clock.

---

## Flow

```
FACULTY                          BACKEND                        STUDENT
   │                                │                              │
   │ POST /api/faculty/login        │                              │
   │   {accessCode}                 │                              │
   │◀── facultyToken ───────────────│                              │
   │                                │                              │
   │ POST /api/sessions             │                              │
   │   {label}                      │                              │
   │◀── sessionId + FIRST BATCH ────│   ① one round trip           │
   │      + noiseSpec               │                              │
   │                                │                              │
   │ ┌── displays: decoy, decoy, VALID, decoy … ──────────────────▶│
   │ │                              │                              │
   │ │ GET /api/sessions/:id/tokens │   ② refilled ~60s ahead      │
   │ │◀── next batch ───────────────│                              │
   │                                │                              │
   │                                │◀─ POST /api/qr/validate ─────│
   │                                │     {payload, deviceId}      │
   │                                │── attendanceToken ──────────▶│
   │                                │     (single-use, 120s,       │
   │                                │      device-bound)           │
   │                                │                              │
   │                                │◀─ POST /api/attendance ──────│
   │                                │     {attendanceToken,        │
   │                                │      registerNumber,         │
   │                                │      frames[1..3]}           │
   │                                │── {status} ─────────────────▶│
   │                                │                              │
   │ GET /api/sessions/:id/summary  │  (polled, cosmetic)          │
   │ POST /api/sessions/:id/end     │                              │
```

### ① and ② are the network-reliability design

The display rotates every ~400 ms. One request per frame would stall visibly
on lecture-hall wifi. Instead:

- The **first batch ships with the session-create response** — zero extra
  round trips to start.
- Batches cover ~60 s and refill when **under 20 s of runway remains**, so a
  ten-second outage is invisible to the room.
- If refill never succeeds, the display keeps showing decoys. Students see no
  valid code — a delay, never a false accept. That is the correct failure
  direction.

---

## Endpoints

### `POST /api/faculty/login`
`{ accessCode }` → `{ facultyToken, expiresAt, serverTime }`
Constant-time comparison. Rate limited to 10 attempts / 15 min / IP.

### `POST /api/sessions` *(faculty)*
`{ label }` → `{ sessionId, tokens[], qr: { slotTtlMs, batchSpanMs, noiseSpec }, serverTime }`

### `GET /api/sessions/:id/tokens?from=&span=` *(faculty)*
→ `{ tokens[], serverTime, slotTtlMs }`
`from` is clamped into `[now, now + maxBatchSpan]` so a client cannot pre-mint
an arbitrary future stockpile.

### `POST /api/qr/validate` *(student, unauthenticated)*
`{ payload, deviceId }` → `{ valid, attendanceToken?, expiresAt?, session? }`

Returns **HTTP 200 with `valid:false`** for a decoy. Decoding a decoy is the
*expected* state during scanning, not an error — hundreds arrive per session,
and treating them as failures pollutes logs and client error paths.

Replay handling is deliberately **idempotent per device**: a projected code is
scanned by the whole class at once, so a slot cannot be globally single-use,
and a camera legitimately decodes the same slot several times per second.
Re-scanning reissues the same short-lived token rather than erroring.

### `POST /api/attendance` *(student, requires attendance token)*
`{ attendanceToken, deviceId, registerNumber, frames[] }` → `{ status, message }`

`status` ∈ `accepted` | `flagged` | `rejected`.

One atomic transaction. Face verification is **unconditional** — the old
`/register` made it optional (`if (faceImage)`) and the client never sent one.

- Token is verified, device-bound, and consumed **only on a terminal
  outcome**, so a retryable rejection does not force a re-scan.
- Abuse is bounded by a per-registration-number attempt counter held
  server-side, which clearing `localStorage` cannot reset.
- Already-recorded submissions return `alreadyRecorded: true` rather than
  erroring — this idempotency is what makes client-side retry safe.

### `GET /api/sessions/:id/summary` · `/export` · `POST /:id/end` *(faculty)*
Live counts, CSV download, session termination.

### `GET /healthz` · `/readyz` · `/api/hello`
Liveness; readiness (503 until the model is warm); a tiny endpoint the scanner
prefetches to warm DNS/TCP/TLS and learn the server clock while the camera
initialises.

---

## What this does and does not prevent

**Prevented**

| Attack | Mechanism |
|---|---|
| Marking attendance with `curl` | attendance token required; face verification unconditional |
| Computing a valid code offline | secret is server-side only |
| Setting the phone clock to class time | validity judged on the server's clock |
| Reusing a screenshot minutes later | 4-second slot |
| Replaying a token from another device | token bound to a hashed device id |
| Double-submitting | single-use `jti` + per-regno ledger |
| Spotting the valid QR by eye | decoys are format-identical |
| Learning who a photo resembles | `best_match_regno` never leaves the server |
| Brute-forcing the face check | server-side per-regno attempt cap |

**Not prevented**

- **Real-time relay.** A confederate in the room can screen-share the QR to
  someone outside within the 4-second window. Closing this requires the face
  layer, and ultimately liveness detection (see `docs/ENROLLMENT.md` §9).
- **Presentation attack.** Holding up a photo of another student still passes
  today — there is no liveness check yet. This is the largest remaining gap.
- **Genuine lookalikes.** Siblings and twins may exceed the threshold.
  `npm run verify:gallery` reports colliding pairs; it currently flags one in
  the existing gallery.

---

## Clock handling

Every response carries `serverTime`. Both clients track the offset and reason
about validity in server time. Neither client's clock affects any security
decision — it only affects *when they choose to act*, and acting at the wrong
moment simply fails.
