---
title: Sorry No Proxy
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# Sorry No Proxy

Classroom attendance that resists proxies: a rotating, server-signed QR code on
the projector, plus face verification on the student's own phone.

```
┌─ Faculty portal ──┐      ┌─ Backend ─────────────┐
│ React + Vite      │─────▶│ Node + Express        │
│ Cloudflare Pages  │      │ Hugging Face Space    │
│ shows signed QR   │      │  • signs QR payloads  │
│ rotating ~400ms   │      │  • issues tokens      │
└───────────────────┘      │  • Python verifier    │
                           │    (warm ONNX pool)   │
┌─ Student scanner ─┐      │  • serves the scanner │
│ Vanilla ES modules│◀────▶│                       │
│ 25 KB gzipped     │      └───────────┬───────────┘
│ same origin ──────┘                  │
└───────────────────┘      ┌───────────▼───────────┐
                           │ Private HF Dataset    │
                           │ face_db.npz           │
                           └───────────────────────┘
```

**Deploys free, with no credit card.** Hugging Face Spaces gives 16 GB RAM on
its free tier — and the verifier needs 564 MB, which rules out the usual
512 MB free allowances.

**→ [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** — free path, step by step.
**→ [docs/DEPLOYMENT-GCP.md](docs/DEPLOYMENT-GCP.md)** — Google Cloud Run, if you have billing.

---

## Run it locally

```bash
npm install
cp .env.example .env
```

Generate the three secrets `.env` needs:

```bash
node -e "console.log('QR_SIGNING_SECRET='+require('crypto').randomBytes(32).toString('base64url'))"
node -e "console.log('TOKEN_SIGNING_SECRET='+require('crypto').randomBytes(32).toString('base64url'))"
```

Set `FACULTY_ACCESS_CODE` to anything 12+ characters, then:

```bash
npm start                                  # backend + scanner → localhost:7860
npm run dev --prefix QR-Faculty-Portal     # faculty portal    → localhost:5173
```

The faculty dev server proxies `/api` to the backend, so no extra config.

> The backend refuses to start without a gallery. Build one with
> `npm run build:gallery`, or point `GALLERY_URL` at a private HF dataset
> (`hf://you/snp-gallery/face_db.npz`, plus `HF_TOKEN`).

> Camera access needs a secure context. `localhost` counts; a LAN IP does not —
> to test on a phone, use `npx localtunnel --port 7860` or deploy.

---

## How it works

The faculty display holds **no secret**. It requests batches of pre-signed
payloads from the backend and shows them on schedule, padding gaps with decoys
that are format-identical to the real thing. A payload is valid for one
4-second slot, judged by the **server's** clock.

A student scans; the server validates and issues a single-use, device-bound
attendance token. That token is the only way to reach `POST /api/attendance`,
which verifies a burst of face frames against the enrollment gallery before
recording anything.

Full contract, threat model, and what is explicitly *not* prevented:
**[docs/PROTOCOL.md](docs/PROTOCOL.md)**.

---

## Layout

```
server/              Node backend — routes, services, middleware
python/              Face pipeline; face_pipeline/ is pure NumPy except engine.py
public/              Student scanner (no build step, no framework)
QR-Faculty-Portal/   Faculty display (React + Vite)
gallery/             Enrollment data — gitignored, never web-served
scripts/             Setup, deploy, and operational scripts
docs/                Deployment, protocol, enrollment
```

## Scripts

| Command | What it does |
|---|---|
| `npm start` | Run the backend |
| `npm test` | 31 unit + contract tests |
| `npm run build:gallery` | Turn `gallery/images/` into `face_db.npz`, quality-gating each photo |
| `npm run verify:gallery` | Audit the gallery; recommend thresholds; flag lookalikes |
| `./scripts/deploy-hf.sh` | Deploy the backend to a Hugging Face Space (free) |
| `./scripts/deploy-faculty.sh` | Build + deploy the portal to Cloudflare Pages (free) |
| `./scripts/upload-gallery.sh` | Push the gallery to a private HF dataset, or to GCS |
| `./scripts/setup-gcp.sh` | Google Cloud setup — **billed path only** |
| `./scripts/deploy-backend.sh` | Deploy to Cloud Run — **billed path only** |
| `./scripts/warm.sh on\|off` | Keep a Cloud Run instance warm — **billed path only** |

## Testing

```bash
npm test                                          # 31 Node
./.venv/bin/python python/tests/test_pipeline.py  # 35 Python
node scripts/integration-test.mjs                 # 37 protocol + attack surface
node scripts/decoy-check.mjs                      # decoy indistinguishability
```

The integration suite asserts the historical attack paths stay closed —
`POST /register`, `GET /gallery/images/…`, the old hardcoded QR string.

---

## Known limitations

These are real and documented, not oversights:

- **No liveness detection.** A photo of another student held up to the camera
  still passes. Largest remaining gap.
- **Single-image gallery.** The current 67 students have one enrollment photo
  each — the main cause of false rejections, and why `25BRS1169`/`25BRS1286`
  collide at the old 0.50 threshold. See [docs/ENROLLMENT.md](docs/ENROLLMENT.md).
- **Thresholds are not calibrated** for your cohort. `npm run verify:gallery`
  computes them once multi-image enrollment data exists.
- **Session state is in-memory.** Correct for one instance (a few thousand
  students per class). Multiple instances need a shared store — swap
  `SessionStore` for a Redis implementation with the same interface.
- **A free Hugging Face Space sleeps after ~48 h idle** and wakes in ~40 s on
  the next request, and carries no SLA. Open it once before class. If you need
  guaranteed availability, use the Cloud Run path.

## Privacy

Student face images are sensitive personal data under the DPDP Act 2023. They
are gitignored, excluded from the container image (only derived embeddings ship),
stored in a **private** repository (HF dataset or GCS bucket), and never served
over HTTP. The Space that runs the app is public so students can reach the
scanner — which is precisely why the gallery lives somewhere else. See
[docs/ENROLLMENT.md](docs/ENROLLMENT.md) for retention and the outstanding
git-history cleanup.
