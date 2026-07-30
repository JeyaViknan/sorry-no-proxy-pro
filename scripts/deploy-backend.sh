#!/usr/bin/env bash
#
# Deploy the backend to Cloud Run.
#
#   ./scripts/deploy-backend.sh
#
# Builds from source with Cloud Build (so you do not need a local amd64
# builder), wires secrets from Secret Manager, and verifies the result is
# actually serving before it claims success.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f deploy.env ]]; then
  echo "ERROR: deploy.env not found.  cp deploy.env.example deploy.env"
  exit 1
fi
# shellcheck disable=SC1091
source deploy.env

: "${GCP_PROJECT_ID:?set in deploy.env}"
: "${GCP_REGION:?set in deploy.env}"
: "${SERVICE_NAME:?set in deploy.env}"
: "${GCS_BUCKET:?set in deploy.env}"

SA_NAME="${SA_NAME:-snp-runtime}"
SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
info() { printf "  \033[36m•\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$*"; }

# ── Preflight ────────────────────────────────────────────────────────
bold "Preflight"

command -v gcloud >/dev/null || { fail "gcloud not found"; exit 1; }
gcloud config set project "$GCP_PROJECT_ID" >/dev/null 2>&1
ok "project $GCP_PROJECT_ID"

if ! gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1; then
  fail "service account $SA_EMAIL not found — run ./scripts/setup-gcp.sh first"
  exit 1
fi
ok "runtime service account exists"

for secret in snp-qr-signing-secret snp-token-signing-secret snp-faculty-access-code; do
  if ! gcloud secrets describe "$secret" >/dev/null 2>&1; then
    fail "secret '$secret' missing — run ./scripts/setup-gcp.sh first"
    exit 1
  fi
done
ok "secrets present"

# The service exits at boot without a gallery, so catch it here rather than
# after a five-minute build.
if ! gcloud storage ls "gs://${GCS_BUCKET}/face_db.npz" >/dev/null 2>&1; then
  fail "gs://${GCS_BUCKET}/face_db.npz not found."
  echo ""
  echo "  The backend cannot start without a gallery. Upload one:"
  echo "    npm run build:gallery                                  # creates gallery/face_db.npz"
  echo "    gcloud storage cp gallery/face_db.npz gs://${GCS_BUCKET}/face_db.npz"
  echo ""
  echo "  (No enrollment images yet? See docs/ENROLLMENT.md.)"
  exit 1
fi
ok "gallery present in gs://${GCS_BUCKET}"

# ── Build CORS allow-list ────────────────────────────────────────────
# The scanner is same-origin so it needs no entry. On the first deploy the
# service URL is unknown, so we add it on the second pass; harmless either way.
EXISTING_URL="$(gcloud run services describe "$SERVICE_NAME" --region "$GCP_REGION" \
  --format='value(status.url)' 2>/dev/null || true)"

ORIGINS="${FACULTY_ORIGIN:-}"
if [[ -n "$EXISTING_URL" ]]; then
  ORIGINS="${ORIGINS:+${ORIGINS},}${EXISTING_URL}"
fi
if [[ -z "$ORIGINS" ]]; then
  fail "FACULTY_ORIGIN is not set in deploy.env and no existing service URL was found."
  echo "  Set FACULTY_ORIGIN to your Cloudflare Pages URL, or leave it for now and"
  echo "  re-run this after the faculty portal is deployed."
  exit 1
fi
ok "CORS allow-list: $ORIGINS"

# ── Environment ──────────────────────────────────────────────────────
ENV_VARS="NODE_ENV=production"
ENV_VARS="${ENV_VARS},ALLOWED_ORIGINS=${ORIGINS}"
ENV_VARS="${ENV_VARS},GALLERY_GCS_URI=gs://${GCS_BUCKET}/face_db.npz"
ENV_VARS="${ENV_VARS},FACE_WORKER_COUNT=${FACE_WORKER_COUNT:-2}"
ENV_VARS="${ENV_VARS},FACE_THRESHOLD_ACCEPT=${FACE_THRESHOLD_ACCEPT:-0.55}"
ENV_VARS="${ENV_VARS},FACE_THRESHOLD_REVIEW=${FACE_THRESHOLD_REVIEW:-0.45}"
ENV_VARS="${ENV_VARS},RATE_LIMIT_GLOBAL_PER_MIN=${RATE_LIMIT_GLOBAL_PER_MIN:-6000}"
ENV_VARS="${ENV_VARS},LOG_LEVEL=${LOG_LEVEL:-info}"

if [[ -n "${SHEET_ID:-}" ]]; then
  ENV_VARS="${ENV_VARS},SHEET_ID=${SHEET_ID}"
  ENV_VARS="${ENV_VARS},SHEET_RANGE=${SHEET_RANGE:-Attendance!A:G}"
  ok "Sheets export enabled (sheet ${SHEET_ID:0:12}…)"
else
  info "Sheets export disabled (SHEET_ID blank) — CSV download still works"
fi

# Secrets are injected by Secret Manager, never placed in env vars here, so
# they never appear in shell history, CI logs or `gcloud run describe`.
SECRETS="QR_SIGNING_SECRET=snp-qr-signing-secret:latest"
SECRETS="${SECRETS},TOKEN_SIGNING_SECRET=snp-token-signing-secret:latest"
SECRETS="${SECRETS},FACULTY_ACCESS_CODE=snp-faculty-access-code:latest"

# ── Deploy ───────────────────────────────────────────────────────────
bold "Deploying (first build takes 8-12 minutes — it compiles insightface and caches the model)"

gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --region "$GCP_REGION" \
  --service-account "$SA_EMAIL" \
  --allow-unauthenticated \
  --cpu "${CPU:-2}" \
  --memory "${MEMORY:-2Gi}" \
  --concurrency 1 \
  --min-instances "${MIN_INSTANCES:-0}" \
  --max-instances "${MAX_INSTANCES:-10}" \
  --timeout 120 \
  --startup-probe "httpGet.path=/healthz,initialDelaySeconds=10,periodSeconds=5,failureThreshold=60,timeoutSeconds=5" \
  --set-env-vars "$ENV_VARS" \
  --set-secrets "$SECRETS"

SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" --region "$GCP_REGION" --format='value(status.url)')"

# ── Verify ───────────────────────────────────────────────────────────
bold "Verifying"

info "waiting for the model to warm (up to 3 minutes on a cold revision)…"
READY=""
for _ in $(seq 1 36); do
  if curl -fsS --max-time 10 "${SERVICE_URL}/readyz" 2>/dev/null | grep -q '"ready":true'; then
    READY=1
    break
  fi
  sleep 5
done

if [[ -z "$READY" ]]; then
  fail "the service did not become ready."
  echo ""
  echo "  Check the logs:"
  echo "    gcloud run services logs read $SERVICE_NAME --region $GCP_REGION --limit 100"
  echo ""
  echo "  Most likely causes, in order:"
  echo "    • gallery download failed  -> is gs://${GCS_BUCKET}/face_db.npz readable by ${SA_EMAIL}?"
  echo "    • model failed to load     -> look for 'FATAL during startup' in the logs"
  echo "    • out of memory            -> raise MEMORY, or lower FACE_WORKER_COUNT"
  exit 1
fi
ok "service is ready"

# CORS is the single most common post-deploy breakage, so assert it here.
if [[ -n "${FACULTY_ORIGIN:-}" ]]; then
  CORS_HEADER="$(curl -fsS -o /dev/null -D- --max-time 10 \
    -H "Origin: ${FACULTY_ORIGIN}" \
    -X OPTIONS "${SERVICE_URL}/api/faculty/login" 2>/dev/null \
    | grep -i '^access-control-allow-origin' | tr -d '\r' || true)"
  if [[ -n "$CORS_HEADER" ]]; then
    ok "CORS allows ${FACULTY_ORIGIN}"
  else
    fail "CORS did NOT allow ${FACULTY_ORIGIN} — the faculty portal will not be able to sign in"
  fi
fi

if curl -fsS --max-time 10 "${SERVICE_URL}/" | grep -q "Mark your attendance"; then
  ok "student scanner is being served"
else
  fail "the scanner did not load at ${SERVICE_URL}/"
fi

# ── Done ─────────────────────────────────────────────────────────────
bold "Deployed"
cat <<EOF

  Backend API + Student Scanner
    ${SERVICE_URL}

  Give students that URL. Point the faculty portal at it:
    VITE_API_BASE=${SERVICE_URL}

  Useful:
    gcloud run services logs tail ${SERVICE_NAME} --region ${GCP_REGION}
    curl -s ${SERVICE_URL}/readyz | python3 -m json.tool

EOF
