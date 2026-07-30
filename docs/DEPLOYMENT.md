# Deployment Guide — Free Tier

Deploy the whole system for **₹0, with no credit card**.

**End state — three URLs:**

| What | Where | Example |
|---|---|---|
| Backend API **+ Student Scanner** | Hugging Face Spaces | `https://you-snp-attendance.hf.space` |
| Faculty QR Portal | Cloudflare Pages | `https://snp-faculty.pages.dev` |
| Face gallery (private, not a URL) | Private HF Dataset | `hf://you/snp-gallery/face_db.npz` |

**Time:** ~1 hour, most of it waiting for the first build.
**Cost:** ₹0. **Credit card:** not required anywhere.

> Already have Google Cloud with billing? [docs/DEPLOYMENT-GCP.md](DEPLOYMENT-GCP.md)
> covers Cloud Run, which is faster and has an SLA.

---

## Why this architecture

### The constraint that decides everything: memory

I measured the running container:

| Configuration | Resident memory |
|---|---|
| 1 verifier worker, model warm, under inference | **564 MB** |
| 2 verifier workers | **790 MB** |

That is InsightFace's `buffalo_l` — a 17 MB detection model and a 174 MB
recognition model — plus ONNX Runtime, OpenCV and Node.

**This single number eliminates almost every free tier**, because the common
free allowance is 512 MB.

### Platform evaluation

| Platform | Node backend | Python worker | Free RAM | Card? | Verdict |
|---|---|---|---|---|---|
| **Hugging Face Spaces** | ✅ Docker | ✅ same container | **16 GB** | **No** | ✅ **Chosen** |
| Cloudflare Pages | static only | ❌ | — | No | ✅ for the faculty portal |
| Cloudflare Workers | ⚠️ V8 isolate, not Node | ❌ no native binaries | 128 MB | No | ❌ backend impossible |
| Render (free web service) | ✅ Docker | ✅ | 512 MB | No | ❌ **OOM at 564 MB** |
| Fly.io | ✅ Docker | ✅ | 256 MB×3 | **Yes** | ❌ card + too small |
| Railway | ✅ | ✅ | trial credit only | Yes | ❌ not free |
| Koyeb | ✅ Docker | ✅ | 512 MB | Yes | ❌ OOM + card |
| Vercel | ⚠️ serverless | ❌ 250 MB bundle cap | 1 GB | No | ❌ can't ship ONNX |
| Netlify | ⚠️ serverless | ❌ 50 MB zip cap | — | No | ❌ can't ship ONNX |
| GitHub Pages | ❌ static | ❌ | — | No | ❌ no backend |
| Supabase | ⚠️ Deno edge fns | ❌ | 256 MB | No | ❌ no Python |
| Firebase | ⚠️ Cloud Functions | ⚠️ Python runtime, no custom binaries | — | **Yes** | ❌ card |
| Oracle Cloud Always Free | ✅ VM | ✅ | 24 GB | Yes (verification) | ⚠️ card, manual ops |
| PythonAnywhere | ❌ no Node | ⚠️ | 512 MB | No | ❌ |

**Hugging Face Spaces wins decisively**: 2 vCPU and **16 GB RAM** free, no card,
Docker SDK that runs our existing `Dockerfile` unmodified, and HTTPS built in.
It is also where this project originally ran — the README frontmatter is still
there.

**Honest limitations of the free Space:**

- **Sleeps after ~48 hours idle**, waking on the next request (~40 s). Open it
  once before class.
- **No SLA.** It is free infrastructure; treat an outage as possible.
- **Ephemeral disk** — resets on restart. Irrelevant here: the gallery is
  re-downloaded at boot by design, and attendance is exported continuously.
- **Public Space.** Required so students can open the scanner. This is why the
  gallery lives in a *separate private dataset* and never in the Space repo.

### Could it fit in 512 MB?

Yes, by switching to `buffalo_s` (a ~13 MB recognition model instead of
174 MB), which would land around 400 MB. **I do not recommend it**: embeddings
are not comparable across models, so it requires re-enrolling everyone, and
accuracy drops noticeably on the degraded phone-camera images this system
works with. Given 16 GB is available for free, there is no reason to.

---

## 1. Prerequisites

### Node.js 20+

- macOS: `brew install node@20` · Windows/Linux: https://nodejs.org (LTS)

```bash
node --version
```
Expect `v20.x.x` or higher.

### Git

```bash
git --version
```

> **Windows:** run everything below in **Git Bash**, not PowerShell.

### Accounts (all free, no card)

| Account | Sign up | Used for |
|---|---|---|
| Hugging Face | https://huggingface.co/join | Backend + gallery |
| Cloudflare | https://dash.cloudflare.com/sign-up | Faculty portal |
| Google | you probably have one | Attendance spreadsheet |

### Python 3.11 *(only to build a gallery locally)*

```bash
python3 --version
```
`3.11.x` or `3.12.x`. **Not 3.13+** — insightface has no wheels for it.

---

## 2. Get the code

```bash
git clone <your-repo-url> sorry-no-proxy
cd sorry-no-proxy
npm install
npm test
```

Expect `# pass 31`, `# fail 0`.

```bash
cp deploy.env.example deploy.env
```

Edit `deploy.env` and set the free-path section:

```bash
HF_USERNAME=your-hf-username
HF_SPACE_NAME=snp-attendance
GALLERY_DATASET=snp-gallery
PAGES_PROJECT=snp-faculty
```

---

## 3. Hugging Face token

1. https://huggingface.co/settings/tokens → **New token**
2. Name: `snp-deploy`, Type: **Write**
3. Copy it (starts `hf_`). You will paste it as a git password shortly.

Also create a **Read** token named `snp-runtime` — the Space uses that one to
fetch the gallery. Never give the Space a write token.

---

## 4. The face gallery

The backend **will not start** without one.

### Build it

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt      # ~5 minutes

# Put images in gallery/images/ named {REGNO}_{NN}_{variant}.jpg
#   e.g. 25BCE1276_01_frontal.jpg
npm run build:gallery      # quality-gates every image, reports what to retake
npm run verify:gallery     # audits it, recommends thresholds
```

Read [docs/ENROLLMENT.md](ENROLLMENT.md) before collecting photos.

**Converting a legacy `face_db.pkl`:**

```bash
./.venv/bin/python -c "
import sys; sys.path.insert(0, 'python')
from face_pipeline import gallery
from face_pipeline.config import PipelineConfig
g = gallery.load(PipelineConfig.from_env())
gallery.save('gallery/face_db.npz',
  {r: [g.embeddings[i] for i in rows] for r, rows in g.index.items()},
  model_name='buffalo_l')
print('converted')
"
```

### Upload it to a PRIVATE dataset

```bash
./scripts/upload-gallery.sh
```

The script walks you through creating the dataset. **Visibility must be
Private** — this is derived biometric data. When git asks for credentials:
username = your HF username, password = your **write** token.

---

## 5. Attendance export *(optional, 5 minutes)*

No cloud project, no service account, no billing.

1. Create a spreadsheet: https://sheets.new
2. **Extensions → Apps Script**
3. Delete the placeholder, paste all of `scripts/apps-script/Code.gs`
4. Generate a secret and put it in `SHARED_SECRET` at the top:
   ```bash
   node -e "console.log(require('crypto').randomBytes(24).toString('base64url'))"
   ```
5. Save, then **Deploy → New deployment → Web app**
   - Execute as: **Me**
   - Who has access: **Anyone** ← required; the secret is the gate
6. **Authorize access** and approve the "unverified app" screen
7. Copy the **Web app URL** (ends in `/exec`)
8. Optional check: **Run → testAppend**, and confirm a row appears

Keep the URL and the secret — you set them on the Space next.

> Skipping this is fine. Attendance is still recorded and downloadable as CSV
> from the faculty portal.

---

## 6. Deploy the backend

```bash
./scripts/deploy-hf.sh
```

The script:
1. Refuses to run with uncommitted changes (a Space builds from a commit)
2. Checks the README frontmatter is intact
3. Adds the `space` git remote and pushes
4. **Generates your secrets and prints them once**
5. Waits for the build and verifies `/readyz` and the scanner

**When it pauses**, create the Space:
- https://huggingface.co/new-space
- Name matching `HF_SPACE_NAME`, SDK **Docker → Blank**, visibility **Public**

**When it prints secrets**, set them at
`https://huggingface.co/spaces/USER/SPACE/settings` → *Variables and secrets*:

| Kind | Name | Value |
|---|---|---|
| Secret | `QR_SIGNING_SECRET` | generated for you |
| Secret | `TOKEN_SIGNING_SECRET` | generated for you |
| Secret | `FACULTY_ACCESS_CODE` | generated for you — **write it down** |
| Secret | `HF_TOKEN` | your **read** token |
| Secret | `SHEET_WEBHOOK_SECRET` | same as `SHARED_SECRET` in Code.gs |
| Variable | `NODE_ENV` | `production` |
| Variable | `GALLERY_URL` | `hf://USER/snp-gallery/face_db.npz` |
| Variable | `ALLOWED_ORIGINS` | *(fill in after step 7)* |
| Variable | `SHEET_WEBHOOK_URL` | your `/exec` URL |
| Variable | `FACE_WORKER_COUNT` | `2` |

**The first build takes 10–20 minutes** — it compiles insightface and caches
the 190 MB model. Watch it under the Space's **Logs** tab.

Successful output ends with:

```
  ✓ Space is ready
  ✓ student scanner is being served

  Backend API + Student Scanner
    https://you-snp-attendance.hf.space
```

**That URL is what you give students.**

---

## 7. Deploy the faculty portal

```bash
npx wrangler login          # opens a browser
./scripts/deploy-faculty.sh
```

It reads the backend URL, checks it is reachable, bakes it into the bundle,
verifies it is really there, and uploads to Pages.

> `VITE_API_BASE` is inlined at **build** time. If the backend URL ever
> changes, re-run this script — a redeploy alone is not enough.

### Close the CORS loop — do not skip

Add the Pages URL to `ALLOWED_ORIGINS` on the Space, comma-separated with the
Space's own URL:

```
https://snp-faculty.pages.dev,https://you-snp-attendance.hf.space
```

Save. The Space restarts automatically.

Skipping this gives a login that silently fails with a CORS error in the
browser console. **It is the single most common deployment mistake.**

---

## 8. Verification

### Backend

```bash
URL=https://you-snp-attendance.hf.space

curl -s $URL/healthz          # {"ok":true,...}
curl -s $URL/readyz | python3 -m json.tool
```

- [ ] `/healthz` → `ok:true`
- [ ] `/readyz` → `ready:true` *(false = still loading, or the gallery failed)*
- [ ] `readyz` shows `sheets.mode: "apps-script-webhook"` if you set it up
- [ ] `$URL/` in a browser shows **"Mark your attendance"**

### Security — these must all fail

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST $URL/register \
  -H 'Content-Type: application/json' -d '{"registerNumber":"25BCE1276"}'   # 404

curl -s -o /dev/null -w "%{http_code}\n" $URL/gallery/images/25BCE1276.png   # 404
curl -s -o /dev/null -w "%{http_code}\n" $URL/.env                            # 404

curl -s -X POST $URL/api/attendance -H 'Content-Type: application/json' -d '{}'
# {"ok":false,"code":"TOKEN_MALFORMED",...}

curl -s -o /dev/null -w "%{http_code}\n" -H "Origin: https://evil.example" \
  -X OPTIONS $URL/api/faculty/login                                           # 403
```

- [ ] All five behave as shown

### Faculty portal

- [ ] Opens at the Pages URL
- [ ] Wrong access code → *"Incorrect access code."*
- [ ] Correct code → setup screen *(fails here = CORS, step 7)*
- [ ] **Start session** → large QR fills the screen, rotating
- [ ] `F` fullscreen, `H` hides the panel
- [ ] **Download CSV** produces a file with a header row

### End to end, on a real phone over mobile data

- [ ] Open the Space URL → intro screen
- [ ] **Start** → permission prompt → rear camera opens
- [ ] Point at the projected QR → advances within ~2 s
- [ ] Front camera opens, guide visible, hint says **"Looking good"**
- [ ] Enter registration number → **Mark me present**
- [ ] Preview freezes on your photo → result within ~3 s
- [ ] Faculty count increments within ~6 s
- [ ] A row appears in the spreadsheet within ~10 s
- [ ] Scan again → *"Already marked present"* (not an error)

### Face verification

- [ ] An enrolled student is accepted
- [ ] A different person using that number is rejected
- [ ] Cover the camera → *"Could not read your face"*, **and retry still works**
- [ ] Three genuine mismatches → *"Too many attempts"*

### Mobile

| Device | Check |
|---|---|
| iPhone Safari | Preview is **not black** — the historical #1 failure |
| Chrome Android | Fast (native barcode API, skips a 368 KB download) |
| Samsung Internet | Rear→front handoff does not error |
| Firefox Mobile | Falls back to the bundled decoder |

- [ ] Background mid-capture, return → recovers, no stale frame

### Slow network

DevTools → Network → **Slow 3G**:

- [ ] Scanner still loads (~25 KB gzipped)
- [ ] *"Network is slow — retrying (1/3)…"* instead of freezing
- [ ] Airplane mode → offline banner; restore → clears

---

## 9. Troubleshooting

### Hugging Face Space

| Symptom | Cause | Fix |
|---|---|---|
| Build fails immediately, "no Dockerfile" | Frontmatter missing/edited | `README.md` must start with `---` … `sdk: docker` … `app_port: 7860` |
| Logs: `Refusing to start — invalid configuration` | A secret is unset | The log names it. Add it under Settings, then **Restart this Space**. |
| Logs: `could not obtain the gallery` + `not found` | `HF_TOKEN` missing/wrong, or `GALLERY_URL` typo'd | HF returns **404, not 403**, for unauthorised private repos. Check the token is a *read* token on the right account. |
| Logs: `returned an HTML page, not a file` | `GALLERY_URL` points at a web page | Use `hf://owner/dataset/face_db.npz`, or a URL containing `/resolve/` |
| Space says "Sleeping" | ~48 h idle | Open the URL; it wakes in ~40 s |
| Push rejected | Wrong credentials | Username = HF username, password = **write** token (not your password) |
| `/readyz` stuck `ready:false` | Still loading, or a worker crash-looping | Wait 3 min. Then check logs for `[verifier]`. |
| Build succeeds, app 500s | Secrets added *after* the build | **Restart this Space** — secrets are injected at container start |

### Faculty portal

| Symptom | Cause | Fix |
|---|---|---|
| Login does nothing; console shows CORS | Pages origin not allow-listed | Add it to `ALLOWED_ORIGINS` on the Space — **step 7** |
| *"Cannot reach the attendance server"* | Stale `VITE_API_BASE` | Re-run `./scripts/deploy-faculty.sh` (rebuild required) |
| `wrangler: not found` | — | `npx wrangler login`; needs Node 20+ |
| **Download CSV** does nothing | Faculty token expired (6 h) | Sign out and back in |

### Student scanner

| Symptom | Cause | Fix |
|---|---|---|
| *"This page must be opened over HTTPS"* | Opened over `http://` | Use the `https://` Space URL |
| Camera blocked | Denied and remembered | Padlock icon → Allow → reload |
| Black preview on iPhone | Historical `muted` bug | Should be fixed; confirm both `<video>` tags have `muted` |
| Scans but never advances | Session ended, or phone clock wrong | Check the portal; set the phone clock to Automatic |
| *"Verification is temporarily unavailable"* | Space asleep or still warming | `curl $URL/readyz`; wake takes ~40 s |
| Rejects a genuine student | Single-image gallery / uncalibrated threshold | Known limitation — [ENROLLMENT.md](ENROLLMENT.md) |

### Attendance export

| Symptom | Cause | Fix |
|---|---|---|
| No rows appear | Wrong secret | `SHEET_WEBHOOK_SECRET` must equal `SHARED_SECRET` in Code.gs, exactly |
| Logs: `returned an HTML error page` | Deployment not set to "Anyone", or edited without redeploying | Deploy → Manage deployments → edit → **New version** → Deploy |
| Logs: `HTTP 401/403` | Deployment access is restricted | Set *Who has access* to **Anyone** |
| Rows lag a few seconds | Expected — batched off the request path | Not a fault |

### Reading logs

Space → **Logs** tab. Every response carries an `X-Request-Id`, shown on the
scanner's error screens — ask a student for it and search the logs.

---

## 10. Operating it

See [docs/RUNBOOK.md](RUNBOOK.md). The short version:

```bash
# Before class — wake the Space (it sleeps after ~48h idle)
curl -s https://you-snp-attendance.hf.space/readyz

# After re-enrolling students
npm run build:gallery && npm run verify:gallery && ./scripts/upload-gallery.sh
# then Restart the Space so it re-downloads
```

---

## 11. If something goes wrong

Send me:

1. The command you ran and its full output
2. The Space's **Logs** tab (last ~50 lines)
3. For browser issues: the console (F12 → Console)

All scripts are safe to re-run.
