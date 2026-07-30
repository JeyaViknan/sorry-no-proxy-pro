# Deployment

## Recommended: Google Cloud Run, `asia-south1` (Mumbai)

Chosen for this specific workload:

- **Scale to zero.** Free outside class hours; a university runs this in
  bursts, not continuously.
- **`concurrency=1` maps onto a single-threaded ONNX worker.** Cloud Run's
  per-instance concurrency control matches the verifier's execution model
  exactly, and autoscaling then does the right thing under a class-sized
  burst.
- **`min-instances=1` around class times removes cold start**, which is the
  failure that would otherwise hit the first students of every lecture.
- **Mumbai region.** The previous host runs in US/EU — roughly 250–300 ms RTT
  from an Indian classroom before any compute happens. `asia-south1` cuts that
  to ~30 ms.

500 students over a 5-minute window is ~1.7 req/s sustained; a 2 vCPU instance
handles ~4–6 verifications/s, so 2–3 instances cover it with headroom.

```bash
PROJECT=your-project
REGION=asia-south1

gcloud run deploy snp-attendance \
  --source . \
  --region "$REGION" \
  --cpu 2 --memory 2Gi \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 60s \
  --set-env-vars "NODE_ENV=production,FACE_WORKER_COUNT=2" \
  --set-secrets "QR_SIGNING_SECRET=snp-qr-secret:latest,\
TOKEN_SIGNING_SECRET=snp-token-secret:latest,\
FACULTY_ACCESS_CODE=snp-faculty-code:latest,\
GOOGLE_PRIVATE_KEY=snp-sheets-key:latest"
```

Before a class, warm it:

```bash
gcloud run services update snp-attendance --region "$REGION" --min-instances 1
# afterwards
gcloud run services update snp-attendance --region "$REGION" --min-instances 0
```

## Alternatives

| Platform | Verdict |
|---|---|
| **Fly.io** (`bom`) | Strong second. Simpler model, always-on machine, Mumbai region. Less elastic. |
| **Hugging Face Spaces** | Where this runs today. Demo host: no SLA, no autoscaling, sleeps after inactivity, US/EU only. Fine for a pilot, wrong for a dependency. |
| **Render free** | **Worse than HF.** 512 MB RAM will not reliably hold insightface + onnxruntime + buffalo_l. Spins down at 15 min; 30–60 s cold start. |
| **Vercel / Netlify** | **Not viable** for the backend — no long-running ONNX, function timeouts, no persistent in-memory model. Fine for the faculty bundle. |
| **AWS App Runner / ECS** (`ap-south-1`) | Correct but heavier ops burden. |

## Split deployment

The two frontends are static and need no compute:

- **Faculty app** → any static host. `cd QR-Faculty-Portal && npm run build`,
  deploy `dist/`. Set `VITE_API_TARGET` or serve behind the same origin.
- **Student scanner** → served by the backend from `public/` (it is 25 KB
  gzipped, so this costs nothing), or push to a CDN.
- **Backend** → Cloud Run.

Add both frontend origins to `ALLOWED_ORIGINS`.

---

## Configuration

Copy `.env.example` to `.env` and fill it in. The server **refuses to start**
on a missing or weak secret rather than degrading silently.

```bash
node -e "console.log(require('crypto').randomBytes(32).toString('base64url'))"
```

Required: `QR_SIGNING_SECRET`, `TOKEN_SIGNING_SECRET` (must differ),
`FACULTY_ACCESS_CODE` (≥ 12 chars), `ALLOWED_ORIGINS` (no `*` — the old
`origin: "*"` let any website post attendance on a student's behalf).

Google Sheets is **optional by design**. Unset, attendance still records
server-side and exports via `GET /api/sessions/:id/export`. A Sheets outage
can never block a class.

### Gallery

`gallery/face_db.npz` must exist. Raw images are excluded from the build
context — only embeddings enter the image. Either bake the `.npz` at build
time or mount it:

```bash
gcloud run deploy ... --add-volume name=gallery,type=cloud-storage,bucket=snp-gallery \
                      --add-volume-mount volume=gallery,mount-path=/app/gallery
```

`/readyz` returns **503** until a worker has loaded it, so an orchestrator
holds traffic back rather than failing the first students of the class.

---

## HTTPS is mandatory

`getUserMedia` requires a secure context. Over plain HTTP the camera silently
never starts — the app detects this and says so, but there is no workaround.
All recommended platforms terminate TLS automatically.

---

## Operational checks

```bash
curl -s https://YOUR_HOST/healthz          # liveness
curl -s https://YOUR_HOST/readyz | jq      # readiness: model warm? sheets? queue depth?
```

`/readyz` reports per-worker state, restart counts, queue depth and Sheets
export health. Alert on `ready: false` for more than ~2 minutes, and on
`sheets.healthy: false`.

### Scaling notes

- **`FACE_WORKER_COUNT`** — one per vCPU. Each holds the model (~600 MB), so
  2 vCPU / 2 GB comfortably runs 2.
- **Session state is in-memory.** Correct for a single instance, which covers
  a few thousand students per class. For multiple instances, replace
  `SessionStore` with a Redis implementation exposing the same methods —
  every consumer already goes through that interface for exactly this reason.
- **Rate limiting is per-device, not per-IP.** A classroom is behind one NAT;
  a conventional per-IP limiter would treat a normal class as a flood and lock
  out the room. Preserve this if you change limits.
- **Google Sheets quota** (~60 writes/min/user) becomes the wall around 60
  students/minute. Writes are already batched off the request path; for a
  whole university, move to a database and export on a schedule.

---

## Verification

```bash
npm test                                   # 21 token/security unit tests
./.venv/bin/python python/tests/test_pipeline.py   # 35 pipeline tests
node scripts/integration-test.mjs          # 37 end-to-end protocol assertions
node scripts/decoy-check.mjs               # decoy indistinguishability
python3 python/verifier.py --selftest      # gallery + model load
```

The integration suite explicitly asserts the old attack paths are closed —
`POST /register`, `GET /gallery/images/…`, the hardcoded QR string.
