---
title: Sorry No Proxy
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# Sorry No Proxy

Classroom attendance that resists proxies: a rotating, server-signed QR code
on the projector, plus face verification on the student's own phone.

```
┌─ Faculty display ─┐      ┌─ Backend ────────┐      ┌─ Student scanner ─┐
│ React + Vite      │─────▶│ Node + Express   │◀─────│ Vanilla ES modules│
│ shows signed QR   │      │ signs QR payloads│      │ scans, then face  │
│ rotating ~400ms   │      │ issues tokens    │      │ 25KB gzipped      │
└───────────────────┘      │ Python verifier  │      └───────────────────┘
                           │ (warm ONNX pool) │
                           └──────────────────┘
```

## Quick start

```bash
cp .env.example .env
node -e "console.log(require('crypto').randomBytes(32).toString('base64url'))"  # for each secret

npm install
npm start                                  # backend  → http://localhost:7860
npm run dev --prefix QR-Faculty-Portal     # faculty  → http://localhost:5173
```

Students open the backend origin directly; the scanner is served from
`public/`.

> **HTTPS is required** for camera access outside `localhost`.

## How it works

The faculty display holds **no secret**. It requests batches of pre-signed
payloads from the backend and shows them on schedule, padding the gaps with
decoys that are format-identical to the real thing. A payload is valid for one
4-second slot, judged by the **server's** clock.

A student scans, the server validates, and issues a single-use, device-bound
attendance token. That token is the only way to reach `POST /api/attendance`,
which verifies a burst of face frames against the enrollment gallery before
recording anything.

Full contract, threat model and what is *not* prevented: **[docs/PROTOCOL.md](docs/PROTOCOL.md)**.

## Layout

```
server/          Node backend — routes, services, middleware
python/          Face pipeline; face_pipeline/ is pure NumPy except engine.py
public/          Student scanner (no build step, no framework)
QR-Faculty-Portal/  Faculty display (React + Vite)
gallery/         Enrollment data — gitignored, never web-served
docs/            Protocol, deployment, enrollment SOP
scripts/         Integration and decoy checks
```

## Testing

```bash
npm test                                          # 21 token/security tests
./.venv/bin/python python/tests/test_pipeline.py  # 35 pipeline tests
node scripts/integration-test.mjs                 # 37 protocol assertions
node scripts/decoy-check.mjs                      # decoy indistinguishability
```

## Enrollment

The gallery currently holds **one image per student**, which is the main cause
of false rejections, and contains one lookalike pair that collides at the
configured threshold. Both are documented with the fix in
**[docs/ENROLLMENT.md](docs/ENROLLMENT.md)**.

When new photographs arrive:

```bash
# drop images into gallery/images/ as {REGNO}_{NN}_{variant}.jpg
npm run build:gallery     # quality-gates every image, reports what to retake
npm run verify:gallery    # audits the gallery, calibrates the threshold
```

No code changes needed.

## Deployment

Google Cloud Run in `asia-south1` is recommended; rationale and alternatives
in **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

## Privacy

Student face images are sensitive personal data under the DPDP Act 2023. They
are gitignored, excluded from the Docker build context (only derived
embeddings enter the image), and never served over HTTP. See
[docs/ENROLLMENT.md](docs/ENROLLMENT.md) for retention and the outstanding
git-history cleanup.
