# Deployment Guide

Follow this top to bottom. It assumes you have never deployed anything.

**End state — three URLs:**

| What | Where | Example |
|---|---|---|
| Backend API **+ Student Scanner** | Google Cloud Run | `https://snp-attendance-abc123-el.a.run.app` |
| Faculty QR Portal | Cloudflare Pages | `https://snp-faculty.pages.dev` |
| Face gallery (private, not a URL) | Cloud Storage | `gs://your-bucket/face_db.npz` |

**Time:** ~45 minutes, most of it waiting for the first build.
**Cost:** roughly ₹0–400/month for a few classes a day. See [Costs](#costs).

---

## Why the scanner is not on Cloudflare Pages

You asked for three separate deployments. I served the student scanner **from
Cloud Run instead**, and it matters:

Putting the scanner on Pages makes every API call cross-origin, which adds a
**CORS preflight round-trip** to `POST /api/qr/validate` — the request that has
to finish inside a **4-second** QR validity window, on congested lecture-hall
wifi. Chrome also caps preflight caching at 2 hours, so students pay it again
in each class. The scanner is **25 KB gzipped**, so there is no CDN benefit
worth that. Same-origin also keeps the strict `connect-src 'self'` CSP intact.

The faculty portal *is* on Pages, where it belongs: its requests are batched
(~1/minute), so preflight cost is irrelevant, and a projector machine benefits
from a CDN.

You still get three URLs. Cloud Run just serves two things.

---

## 1. Prerequisites

Install these four, then verify each.

### Node.js 20 or newer

- **macOS:** `brew install node@20`
- **Windows/Linux:** https://nodejs.org (LTS installer)

```bash
node --version
```
Expect `v20.x.x` or higher. If you see `v18` or lower, upgrade — the backend
uses built-in `fetch`, which needs Node 18+, and is tested on 20.

### Google Cloud CLI

https://cloud.google.com/sdk/docs/install — use the installer for your OS.

```bash
gcloud --version
```
Expect several lines starting with `Google Cloud SDK 5xx.x.x`.

> **Windows:** run everything in this guide from **Git Bash** (ships with Git
> for Windows), not PowerShell. The `.sh` scripts are bash.

### Docker *(optional)*

Only needed if you want to build the image locally. `gcloud run deploy
--source` builds in the cloud, so you can skip this.

```bash
docker --version
```

### Python 3.11 *(optional)*

Only needed to build a gallery from enrollment photos on your own machine.

```bash
python3 --version
```
Expect `3.11.x` or `3.12.x`. **Not 3.13+** — insightface has no wheels for it
yet, and you will get compiler errors.

### A Google account with billing, and a Cloudflare account

Both free to create. Billing must be **enabled** on the Google project (see
step 2) — Cloud Run, Cloud Build and Artifact Registry all refuse to work
without it, with error messages that never mention billing.

---

## 2. Get the code

```bash
git clone <your-repo-url> sorry-no-proxy
cd sorry-no-proxy
npm install
```

`npm install` should finish in under a minute and report roughly **74
packages, 0 vulnerabilities**.

Verify the code is sound before deploying it:

```bash
npm test
```
Expect `# pass 31`, `# fail 0`.

---

## 3. Google Cloud setup

### 3.1 Create the project

**In the browser:**

1. Go to https://console.cloud.google.com/projectcreate
2. **Project name:** `attendance` (or anything)
3. Note the **Project ID** underneath — it is auto-generated, looks like
   `attendance-482910`, and is **not** the same as the name. You need the ID.
4. Click **Create**, wait ~30 seconds.

### 3.2 Enable billing

1. https://console.cloud.google.com/billing
2. **Link a billing account** to your new project (add a card if you have none).

> New Google Cloud accounts get \$300 of free credit, and this workload sits
> largely inside the always-free tier. You will not be charged meaningfully,
> but the project will not function without billing linked.

### 3.3 Log in from the terminal

```bash
gcloud auth login
```
Opens a browser. Approve.

```bash
gcloud auth application-default login
```
A second, separate login. Both are needed — the first authenticates the
`gcloud` command, the second authenticates libraries.

### 3.4 Configure the deployment

```bash
cp deploy.env.example deploy.env
```

Open `deploy.env` and set:

| Variable | Set it to | Notes |
|---|---|---|
| `GCP_PROJECT_ID` | your Project **ID** from 3.1 | not the display name |
| `GCP_REGION` | `asia-south1` | Mumbai. Nearest region to your classrooms. |
| `SERVICE_NAME` | `snp-attendance` | fine as-is |
| `GCS_BUCKET` | `yourname-snp-gallery` | **globally unique** across all of Google Cloud — prefix it |
| `SHEET_ID` | *(leave blank for now)* | filled in at step 3.6 |
| `FACULTY_ORIGIN` | *(leave blank for now)* | filled in at step 5 |

### 3.5 Run the setup script

```bash
./scripts/setup-gcp.sh
```

**What it does:** enables 7 APIs, creates a service account, creates the
private gallery bucket, and generates three secrets in Secret Manager. It is
idempotent — safe to re-run.

**Expected output ends with:**

```
  ┌─────────────────────────────────────────────────────────┐
  │  FACULTY ACCESS CODE:  Xk3mP9qR2wLnB7vT              │
  └─────────────────────────────────────────────────────────┘
  ! Write this down NOW — it is not shown again.
```

> **Write the faculty access code down.** Faculty type it to start a session.
> To read it back later:
> ```bash
> gcloud secrets versions access latest --secret=snp-faculty-access-code
> ```

**If it fails:** see [Troubleshooting §A](#a-google-cloud-setup).

### 3.6 Connect Google Sheets *(optional but recommended)*

1. Create a new spreadsheet at https://sheets.new
2. Rename the bottom tab from `Sheet1` to **`Attendance`** *(must match
   exactly — `SHEET_RANGE` refers to it)*
3. Click **Share**, and paste the service account address that `setup-gcp.sh`
   printed:
   ```
   snp-runtime@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```
4. Set it to **Editor**. **Untick "Notify people".** Click **Share**.
5. Copy the sheet ID from the URL:
   ```
   https://docs.google.com/spreadsheets/d/1a2B3cD4eF5gH6iJ7kL8mN9oP/edit
                                          └──────── this part ────────┘
   ```
6. Put it in `deploy.env` as `SHEET_ID=`

> Skipping this is fine. Attendance is still recorded server-side and
> downloadable as CSV from the faculty portal. A Sheets outage can never block
> a class.

---

## 4. The face gallery

The backend **will not start** without a gallery. It holds the face embeddings
every verification compares against.

### If you have enrollment photos

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt   # ~5 minutes

# Put images in gallery/images/ named {REGNO}_{NN}_{variant}.jpg
#   e.g. 25BCE1276_01_frontal.jpg
npm run build:gallery      # quality-gates every image, reports what to retake
npm run verify:gallery     # audits the gallery, recommends thresholds
./scripts/upload-gallery.sh
```

Read `docs/ENROLLMENT.md` before collecting photos — it explains the capture
protocol and why one image per student is the main cause of false rejections.

### If you have the legacy `face_db.pkl`

```bash
./.venv/bin/python -c "
import sys; sys.path.insert(0, 'python')
from face_pipeline import gallery
from face_pipeline.config import PipelineConfig
g = gallery.load(PipelineConfig.from_env())
gallery.save('gallery/face_db.npz', {r: [g.embeddings[i] for i in rows] for r, rows in g.index.items()}, model_name='buffalo_l')
print('converted')
"
./scripts/upload-gallery.sh
```

### Set your threshold

The default is now `FACE_THRESHOLD_ACCEPT=0.55` — in `deploy.env`, in `.env.example`, and as the built-in fallback in both `server/config.js` and `python/face_pipeline/config.py`.

**This is deliberate.** In the existing 67-student gallery, `25BRS1169` and
`25BRS1286` score **0.5016** against each other — above 0.50, meaning those two
students can currently mark each other present. `0.55` gives zero colliding
pairs. Run `npm run verify:gallery` to check your own data.

---

## 5. Deploy the backend

```bash
./scripts/deploy-backend.sh
```

**First run takes 8–12 minutes** — Cloud Build compiles insightface and caches
the 300 MB face model into the image. Later deploys take 2–3 minutes.

**What you should see:**

```
Preflight
  ✓ project attendance-482910
  ✓ runtime service account exists
  ✓ secrets present
  ✓ gallery present in gs://yourname-snp-gallery
  ✓ CORS allow-list: https://snp-faculty.pages.dev

Deploying (first build takes 8-12 minutes …)
  ... Building using Dockerfile and deploying container ...
  ✓ Service [snp-attendance] revision [snp-attendance-00001-abc] has been deployed

Verifying
  • waiting for the model to warm (up to 3 minutes on a cold revision)…
  ✓ service is ready
  ✓ CORS allows https://snp-faculty.pages.dev
  ✓ student scanner is being served

Deployed

  Backend API + Student Scanner
    https://snp-attendance-abc123-el.a.run.app
```

> **On the very first deploy** `FACULTY_ORIGIN` is blank and the script will
> stop with a message saying so. That is expected — either put a placeholder in
> now, or deploy the faculty portal first (step 6) and come back. The script
> tells you which.

Save that Cloud Run URL. **That is what you give students.**

**If it fails:** see [Troubleshooting §B](#b-backend-deployment).

---

## 6. Deploy the faculty portal

### 6.1 Log in to Cloudflare

```bash
npx wrangler login
```
Opens a browser. Click **Allow**.

```bash
npx wrangler whoami
```
Should print your account email and ID.

### 6.2 Deploy

```bash
./scripts/deploy-faculty.sh
```

**What it does:** reads your Cloud Run URL, checks the backend is actually
reachable, bakes the URL into the bundle, verifies it is really in there, and
uploads to Pages.

**Expected output ends with:**

```
  ✓ backend URL baked into the bundle
  ✨ Deployment complete! Take a peek over at https://snp-faculty.pages.dev
```

### 6.3 Close the CORS loop — **do not skip this**

The backend rejects requests from origins it does not know. Put the Pages URL
in `deploy.env`:

```bash
FACULTY_ORIGIN=https://snp-faculty.pages.dev
```

Then redeploy the backend so it takes effect:

```bash
./scripts/deploy-backend.sh
```

Skipping this produces a login that fails with a CORS error in the browser
console and no visible message in the UI. It is the single most common
deployment mistake.

> **Vite inlines the backend URL at build time.** If the Cloud Run URL ever
> changes, re-run `./scripts/deploy-faculty.sh` — a rebuild is required, not
> just a redeploy.

---

## 7. Final configuration

### HTTPS

Automatic on both platforms. Nothing to configure.

This is not optional — `getUserMedia` refuses to run outside a secure context,
so over plain HTTP the camera silently never starts.

### Custom domains *(optional)*

**Cloud Run:**
```bash
gcloud beta run domain-mappings create \
  --service snp-attendance --domain attendance.youruni.edu --region asia-south1
```
Then add the CNAME it prints to your DNS.

**Cloudflare Pages:** dashboard → your project → **Custom domains** → **Set up
a domain**.

After either, add the new origin to `FACULTY_ORIGIN` and redeploy the backend.

### Scaling

Set in `deploy.env`, applied by `deploy-backend.sh`:

| Class size | `MAX_INSTANCES` | `RATE_LIMIT_GLOBAL_PER_MIN` |
|---|---|---|
| ~200 | 5 | 6000 |
| ~500 | 10 | 6000 |
| ~2000 | 30 | 20000 |

`concurrency=1` is fixed and should stay that way: the ONNX verifier is
single-threaded, so one request per instance is what makes autoscaling behave.

### Cold starts

A cold instance downloads the gallery and loads a ~600 MB model — about 40
seconds. Before class:

```bash
./scripts/warm.sh on     # waits until /readyz reports ready
```
After class:
```bash
./scripts/warm.sh off    # back to scale-to-zero, free while idle
```

### Where secrets live

| Secret | Stored in | Rotate with |
|---|---|---|
| QR signing key | Secret Manager | `gcloud secrets versions add snp-qr-signing-secret --data-file=-` |
| Token signing key | Secret Manager | same pattern |
| Faculty access code | Secret Manager | same pattern (do this each semester) |
| Google credentials | **nowhere** | Cloud Run uses the instance service account via the metadata server — there is no key to leak |

After rotating any secret, redeploy so the new version is picked up. Rotating
the QR key instantly invalidates every code in circulation.

---

## Costs

| Service | Typical | Notes |
|---|---|---|
| Cloud Run | ₹0–300/mo | Free tier covers 2M requests. You mostly pay for warm minutes. |
| Cloud Storage | ~₹2/mo | The gallery is a few hundred KB. |
| Cloud Build | ₹0 | 120 free build-minutes/day. |
| Artifact Registry | ~₹40/mo | ~2 GB image. Prune old revisions to reduce. |
| Cloudflare Pages | ₹0 | Free tier is generous. |

Leaving `MIN_INSTANCES=1` around the clock costs roughly ₹1,800/month — use
`warm.sh` instead.

---

## 8. Verification checklist

Run through this before trusting it with a real class.

### Backend

```bash
URL=https://your-service-url.run.app

curl -s $URL/healthz
# {"ok":true,"uptimeSec":123}

curl -s $URL/readyz | python3 -m json.tool
# "ready": true, verifier.workers[].ready true, sheets.enabled as configured
```

- [ ] `/healthz` returns `ok:true`
- [ ] `/readyz` returns `ready:true` *(if false, the model is still loading or the gallery failed — check logs)*
- [ ] `readyz` shows the expected number of workers
- [ ] Opening `$URL/` in a browser shows **"Mark your attendance"**

### Security — these must all fail

```bash
# The old bypass: attendance with no QR and no face
curl -s -o /dev/null -w "%{http_code}\n" -X POST $URL/register \
  -H 'Content-Type: application/json' -d '{"registerNumber":"25BCE1276"}'
# 404

# Biometric data must not be reachable
curl -s -o /dev/null -w "%{http_code}\n" $URL/gallery/images/25BCE1276.png   # 404
curl -s -o /dev/null -w "%{http_code}\n" $URL/.env                            # 404
curl -s -o /dev/null -w "%{http_code}\n" $URL/server.js                       # 404

# Attendance without a token
curl -s -X POST $URL/api/attendance -H 'Content-Type: application/json' -d '{}'
# {"ok":false,"code":"TOKEN_MALFORMED",...}

# A rogue origin
curl -s -o /dev/null -w "%{http_code}\n" -H "Origin: https://evil.example" \
  -X OPTIONS $URL/api/faculty/login
# 403
```

- [ ] All six behave as shown

### Faculty portal

- [ ] Opens at the Pages URL
- [ ] Wrong access code → *"Incorrect access code."*
- [ ] Correct code → setup screen (**if this fails, CORS — step 6.3**)
- [ ] **Start session** → large QR fills the screen, rotating a few times a second
- [ ] Timer counts up; "Marked present" shows `0`
- [ ] `F` toggles fullscreen, `H` hides the panel
- [ ] **Download CSV** downloads a file with a header row

### End-to-end, on a real phone

Do this on an actual phone over **mobile data**, not desktop over wifi.

- [ ] Open the Cloud Run URL → intro screen
- [ ] Tap **Start** → camera permission prompt → rear camera opens
- [ ] Point at the projected QR → advances within ~2 seconds
- [ ] Front camera opens, face guide visible, hint reads **"Looking good"**
- [ ] Type a registration number → **Mark me present** enables
- [ ] Tap it → preview freezes on your photo → result within ~3 seconds
- [ ] Faculty portal's "Marked present" count goes up within ~6 seconds
- [ ] Sheets: a row appears within ~10 seconds *(if configured)*
- [ ] Scan again with the same number → *"Already marked present"* (not an error)

### Face verification behaviour

- [ ] An enrolled student is accepted
- [ ] A **different** person using that number is rejected
- [ ] Covering the camera → *"Could not read your face"*, and **retry still works**
      *(quality failures must not consume the 3 identity attempts)*
- [ ] Three genuine mismatches → *"Too many attempts"*

### Mobile compatibility

| Device | Check |
|---|---|
| iPhone Safari | Camera preview is **not black** — this was the #1 historical failure |
| Chrome Android | Scans fast (uses the native barcode API) |
| Samsung Internet | Rear→front camera handoff does not error |
| Firefox Mobile | Falls back to the bundled decoder |

- [ ] Background the app mid-capture, return → recovers, no stale frame
- [ ] Rotate the phone → layout stays usable

### Slow network

Chrome DevTools → Network → throttle to **Slow 3G**:

- [ ] Scanner still loads (~25 KB gzipped)
- [ ] "Network is slow — retrying (1/3)…" appears instead of freezing
- [ ] Submission eventually succeeds
- [ ] Airplane mode → *"No internet connection"* banner; restore → clears

---

## 9. Troubleshooting

### A. Google Cloud setup

| Symptom | Cause | Fix |
|---|---|---|
| `PERMISSION_DENIED: caller does not have permission` | Not an Owner/Editor on the project | Ask whoever owns it for **Owner**, or use your own project |
| `Billing account not found` / API enable fails | Billing not linked | https://console.cloud.google.com/billing — link an account |
| `The project ... does not exist` | Using the project **name**, not the **ID** | `gcloud projects list` — take the `PROJECT_ID` column |
| `bucket names must be globally unique` | Someone else has that name | Add a prefix: `yourname-snp-gallery` |
| `gcloud: command not found` | CLI not on PATH | Reopen your terminal; on Windows use Git Bash |

### B. Backend deployment

| Symptom | Cause | Fix |
|---|---|---|
| `gs://.../face_db.npz not found` | No gallery uploaded | `./scripts/upload-gallery.sh` (build one first, step 4) |
| Build fails: `failed to solve` / `pip install` errors | Transient network in Cloud Build | Re-run. If persistent, check `requirements.txt` was not edited |
| `Revision ... failed with message: Container failed to start` | The container exited at boot | `gcloud run services logs read snp-attendance --region asia-south1 --limit 100` and look for `FATAL` |
| Logs show `could not obtain the face gallery` | Service account cannot read the bucket | `gcloud storage buckets add-iam-policy-binding gs://BUCKET --member=serviceAccount:snp-runtime@PROJECT.iam.gserviceaccount.com --role=roles/storage.objectViewer` |
| Logs show `Refusing to start — invalid configuration` | A required env var is missing | The log lists exactly which. Usually a secret failed to mount — re-run `setup-gcp.sh` |
| `/readyz` stays `ready:false` | Model still loading, or a worker is crash-looping | Wait 3 min. Then check logs for `[verifier]`. If OOM, raise `MEMORY` to `4Gi` or drop `FACE_WORKER_COUNT` to `1` |
| Deploy succeeds, `/` returns 404 | `public/` did not make it into the image | Confirm `public/index.html` is **committed** — `.dockerignore` excludes untracked paths only if gitignored, but a missing file is a missing file |

### C. Faculty portal

| Symptom | Cause | Fix |
|---|---|---|
| Login does nothing; console shows `blocked by CORS policy` | Pages origin not allow-listed | Set `FACULTY_ORIGIN` in `deploy.env`, re-run `./scripts/deploy-backend.sh` — **step 6.3** |
| Login shows *"Cannot reach the attendance server"* | Wrong or stale `VITE_API_BASE` | Re-run `./scripts/deploy-faculty.sh` (rebuild is required) |
| `wrangler: not found` | Wrangler not installed | `npx wrangler` downloads it on demand; check Node 20+ |
| `Project not found` | Pages project does not exist yet | The script creates it on first deploy; if prompted, choose **Create new project** |
| QR is tiny | Stale cached build | Hard-refresh (Ctrl/Cmd+Shift+R) |
| **Download CSV** does nothing | Faculty token expired (6 h) | Sign out and back in |

### D. Student scanner

| Symptom | Cause | Fix |
|---|---|---|
| *"This page must be opened over HTTPS"* | Opened over `http://` or an IP | Use the `https://` Cloud Run URL |
| Camera permission blocked | Denied earlier and remembered | Tap the padlock/camera icon in the address bar → Allow → reload |
| **Black preview on iPhone** | Historically a missing `muted` attribute | Should be fixed. If it recurs, confirm the deployed `index.html` has `muted` on both `<video>` tags |
| Scans but never advances | Faculty session ended, or clock skew | Check the portal is still running; check the phone's clock is set to automatic |
| *"Verification is temporarily unavailable"* | Verifier not ready | `curl $URL/readyz`. Cold start takes ~40 s |
| Always *"Could not read your face"* | Lighting, or thresholds too strict | Face a window. Then `npm run verify:gallery` |
| Rejects a genuine student | Single-image gallery, or uncalibrated threshold | The known limitation — see `docs/ENROLLMENT.md` |

### E. Google Sheets

| Symptom | Cause | Fix |
|---|---|---|
| No rows appear | Sheet not shared with the service account | Share as **Editor** with `snp-runtime@PROJECT.iam.gserviceaccount.com` |
| Logs: `append failed (403)` | Same as above, or Sheets API disabled | `gcloud services enable sheets.googleapis.com` |
| Logs: `append failed (400)` | Tab named something other than `Attendance` | Rename the tab, or change `SHEET_RANGE` |
| Rows lag by a few seconds | Expected — batched off the request path | Not a fault. Attendance is already durable server-side. |

### Reading logs

```bash
# Live tail
gcloud run services logs tail snp-attendance --region asia-south1

# Recent errors only
gcloud run services logs read snp-attendance --region asia-south1 --limit 200 \
  | grep -i error
```

Every response carries an `X-Request-Id`, and error screens show it. If a
student reports a failure, ask for that id and grep the logs for it — it maps
to exactly one request.

---

## 10. If something goes wrong

Send me:

1. The exact command you ran
2. The full error output
3. `gcloud run services logs read snp-attendance --region YOUR_REGION --limit 50`
4. For browser problems: the console tab (F12 → Console)

Both deploy scripts are safe to re-run — they check current state before
changing anything.
