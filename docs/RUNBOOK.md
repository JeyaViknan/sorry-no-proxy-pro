# Runbook

Day-to-day operation once deployed. For first-time setup see
[DEPLOYMENT.md](DEPLOYMENT.md).

---

## Before every class

```bash
./scripts/warm.sh on
```

Waits until the model is loaded and prints `Ready. Students can scan now.`
Takes about 40 seconds. Without it, the first student of the day waits through
a cold start while the container downloads the gallery and loads a 600 MB model.

Then open the faculty portal, sign in, and click **Start session**.

## After class

```bash
./scripts/warm.sh off
```

Back to scale-to-zero. The service costs nothing while idle.

> Forgetting this costs roughly ₹1,800/month if left on permanently. It is
> otherwise harmless.

---

## Checking health

```bash
URL=$(gcloud run services describe snp-attendance --region asia-south1 --format='value(status.url)')

curl -s $URL/readyz | python3 -m json.tool
```

What to look at:

| Field | Healthy | If not |
|---|---|---|
| `ready` | `true` | Model still loading, or all workers are down |
| `verifier.workers[].ready` | all `true` | A worker is crash-looping — check logs for `[verifier]` |
| `verifier.workers[].restarts` | `0` | Repeated restarts mean OOM or a bad gallery |
| `verifier.queued` | `0`–`2` | Sustained high means you need more `MAX_INSTANCES` |
| `sheets.healthy` | `true` | Sheet sharing or API problem — see below |
| `sheets.queued` | near `0` | Growing means the export is failing; attendance is still safe |

---

## Logs

```bash
# Live
gcloud run services logs tail snp-attendance --region asia-south1

# Errors in the last 200 lines
gcloud run services logs read snp-attendance --region asia-south1 --limit 200 | grep -i error
```

Every response carries an `X-Request-Id`, and the scanner shows it on error
screens. When a student reports a problem, ask for that id:

```bash
gcloud run services logs read snp-attendance --region asia-south1 --limit 500 \
  | grep "<the-request-id>"
```

It maps to exactly one request.

### What normal traffic looks like

```json
{"level":"info","msg":"session started","ctx":{"sessionId":"UXLG2UXF","label":"CSE-A"}}
{"level":"info","msg":"qr validated","ctx":{"sessionId":"UXLG2UXF","slot":446351248}}
{"level":"info","msg":"attendance recorded","ctx":{"registerNumber":"25BCE1276","status":"accepted","similarity":0.71}}
```

### Worth investigating

| Log line | Means |
|---|---|
| `verification rejected` with `kind:"identity"` | A face did not match. Occasional is normal; a cluster is not. |
| `verification rejected` with `kind:"quality"` | Lighting or camera problems in the room. |
| `identity attempt cap reached` | A student hit 3 mismatches — they need manual marking. |
| `slow request` | Something took >3s. A few during cold start is fine. |
| `[verifier] exited` | A worker crashed and is restarting. |
| `[sheets] export failing persistently` | Sheets is broken. Attendance is safe; use CSV. |

---

## A student cannot mark attendance

Work down this list:

1. **Is the session running?** The faculty portal must show the live QR screen.
2. **Is the service ready?** `curl -s $URL/readyz | grep ready`
3. **Their phone's clock.** If set manually and wrong by more than a few
   seconds, every code reads as expired. Settings → Date & Time → Automatic.
4. **Camera permission.** Padlock icon in the address bar → allow → reload.
5. **Attempt cap.** Three identity mismatches locks them out for the session —
   by design. Mark them manually and check their enrollment photo.
6. **Not enrolled.** If their registration number is not in the gallery they
   can never pass. Verify: `npm run verify:gallery` lists every enrolled id.

### Marking someone manually

There is deliberately no admin override in the app — an override endpoint is a
bypass of the entire system. Record it in your own register, or add the row to
the spreadsheet directly.

---

## Updating the gallery

After re-enrolling students:

```bash
npm run build:gallery      # quality-gates every photo, reports what to retake
npm run verify:gallery     # audit + threshold recommendation
./scripts/upload-gallery.sh
./scripts/warm.sh off && ./scripts/warm.sh on   # force a reload
```

The gallery is only read at container start, so a running instance keeps the
old one until it restarts.

### Rolling back a bad gallery

The bucket has versioning enabled:

```bash
gcloud storage ls -a gs://YOUR_BUCKET/face_db.npz          # list versions
gcloud storage cp gs://YOUR_BUCKET/face_db.npz#<GENERATION> gs://YOUR_BUCKET/face_db.npz
./scripts/warm.sh off && ./scripts/warm.sh on
```

---

## Rotating secrets

Do the faculty access code **every semester**, and the signing keys if you
suspect exposure.

```bash
# New faculty access code
node -e "console.log(require('crypto').randomBytes(15).toString('base64url'))" \
  | gcloud secrets versions add snp-faculty-access-code --data-file=-

# New QR signing key — invalidates every code in circulation immediately
node -e "console.log(require('crypto').randomBytes(32).toString('base64url'))" \
  | gcloud secrets versions add snp-qr-signing-secret --data-file=-

./scripts/deploy-backend.sh   # required: Cloud Run pins the version it started with
```

Read the current faculty code:

```bash
gcloud secrets versions access latest --secret=snp-faculty-access-code
```

---

## Rolling back a deploy

```bash
gcloud run revisions list --service snp-attendance --region asia-south1

gcloud run services update-traffic snp-attendance \
  --region asia-south1 --to-revisions REVISION_NAME=100
```

Traffic moves in seconds. The bad revision stays available to inspect.

---

## Scaling for a bigger cohort

Edit `deploy.env`, then `./scripts/deploy-backend.sh`:

| Students | `MAX_INSTANCES` | `RATE_LIMIT_GLOBAL_PER_MIN` |
|---|---|---|
| ~200 | 5 | 6000 |
| ~500 | 10 | 6000 |
| ~2000 | 30 | 20000 |

**Do not change `concurrency` from 1.** The ONNX verifier is single-threaded;
one request per instance is what makes autoscaling track real capacity.

### The limit you will hit first

Beyond roughly **2,000 students in one session**, session state — which lives
in memory on one instance — stops being viable, because Cloud Run will have
spread students across instances that cannot see each other's sessions.

Symptom: students get `SESSION_ENDED` for a session that is plainly running.

Fix: replace `SessionStore` (`server/services/sessionStore.js`) with a Redis
implementation exposing the same methods. Every consumer already goes through
that interface for exactly this reason.

> Below ~2,000 this is not a risk in practice, because `min-instances=1` plus
> steady traffic keeps one instance serving. It becomes real when autoscaling
> genuinely spins up a second instance mid-session.

---

## Cost control

```bash
# What is running right now
gcloud run services describe snp-attendance --region asia-south1 \
  --format='value(status.traffic,spec.template.metadata.annotations)'

# Prune old images (Artifact Registry is the main storage cost)
gcloud artifacts docker images list \
  asia-south1-docker.pkg.dev/YOUR_PROJECT/cloud-run-source-deploy \
  --include-tags --format='table(IMAGE,TAGS,CREATE_TIME)'
```

Set a budget alert at https://console.cloud.google.com/billing/budgets —
₹1,000/month is a sensible tripwire for this workload.

---

## Known limitations, restated

Do not let these surprise you mid-semester:

- **No liveness detection.** A photo held up to the camera passes.
- **Real-time relay is possible.** A confederate can screen-share the QR
  within its 4-second window. The face layer is what actually stops this, and
  only for people not physically present.
- **Lookalikes.** `npm run verify:gallery` reports pairs that collide at your
  configured threshold. Check it after every re-enrollment.
- **In-memory sessions.** See scaling above.
