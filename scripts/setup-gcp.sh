#!/usr/bin/env bash
#
# One-time Google Cloud setup for Sorry No Proxy.
#
# Creates: APIs, service account, IAM bindings, GCS bucket, Secret Manager
# secrets. Safe to re-run — every step checks before creating.
#
#   ./scripts/setup-gcp.sh
#
# Everything is driven by deploy.env (copy deploy.env.example first).

set -euo pipefail

# ── Load configuration ───────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f deploy.env ]]; then
  echo "ERROR: deploy.env not found."
  echo "  cp deploy.env.example deploy.env"
  echo "  then edit it and run this again."
  exit 1
fi
# shellcheck disable=SC1091
source deploy.env

: "${GCP_PROJECT_ID:?set GCP_PROJECT_ID in deploy.env}"
: "${GCP_REGION:?set GCP_REGION in deploy.env}"
: "${SERVICE_NAME:?set SERVICE_NAME in deploy.env}"
: "${GCS_BUCKET:?set GCS_BUCKET in deploy.env}"

SA_NAME="${SA_NAME:-snp-runtime}"
SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
info() { printf "  \033[36m•\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }

# ── Preflight ────────────────────────────────────────────────────────
bold "Preflight"

command -v gcloud >/dev/null || { echo "ERROR: gcloud not found. See docs/DEPLOYMENT.md step 1."; exit 1; }
ok "gcloud $(gcloud version --format='value(\"Google Cloud SDK\")' 2>/dev/null | head -1)"

if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q .; then
  echo "ERROR: not logged in. Run: gcloud auth login"
  exit 1
fi
ok "authenticated as $(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -1)"

gcloud config set project "$GCP_PROJECT_ID" >/dev/null 2>&1
ok "project set to $GCP_PROJECT_ID"

# Billing must be live or API enablement and Cloud Build both fail with
# messages that do not mention billing.
if ! gcloud beta billing projects describe "$GCP_PROJECT_ID" \
      --format='value(billingEnabled)' 2>/dev/null | grep -qi true; then
  warn "Billing does not appear to be enabled for $GCP_PROJECT_ID."
  warn "Cloud Run, Cloud Build and Artifact Registry all require it."
  warn "Enable it at: https://console.cloud.google.com/billing/linkedaccount?project=$GCP_PROJECT_ID"
  read -r -p "  Continue anyway? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || exit 1
else
  ok "billing enabled"
fi

# ── APIs ─────────────────────────────────────────────────────────────
bold "Enabling APIs (this can take a couple of minutes the first time)"

REQUIRED_APIS=(
  run.googleapis.com                 # Cloud Run
  cloudbuild.googleapis.com          # builds the container from source
  artifactregistry.googleapis.com    # stores the built image
  secretmanager.googleapis.com       # QR/token secrets
  storage.googleapis.com             # gallery bucket
  sheets.googleapis.com              # attendance export
  iamcredentials.googleapis.com      # service-account token minting
)

ENABLED="$(gcloud services list --enabled --format='value(config.name)')"
TO_ENABLE=()
for api in "${REQUIRED_APIS[@]}"; do
  if grep -qx "$api" <<<"$ENABLED"; then ok "$api"; else TO_ENABLE+=("$api"); fi
done

if [[ ${#TO_ENABLE[@]} -gt 0 ]]; then
  info "enabling: ${TO_ENABLE[*]}"
  gcloud services enable "${TO_ENABLE[@]}"
  ok "enabled ${#TO_ENABLE[@]} API(s)"
fi

# ── Runtime service account ──────────────────────────────────────────
bold "Runtime service account"

if gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1; then
  ok "$SA_EMAIL already exists"
else
  gcloud iam service-accounts create "$SA_NAME" \
    --display-name="Sorry No Proxy runtime" \
    --description="Cloud Run identity: reads the gallery bucket, appends to the attendance sheet"
  ok "created $SA_EMAIL"
fi

# Deliberately narrow. This identity reads ONE bucket and writes ONE sheet;
# it is never given project-wide storage or editor roles.
info "granting roles/secretmanager.secretAccessor"
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor" \
  --condition=None >/dev/null
ok "secret access granted"

# ── Gallery bucket ───────────────────────────────────────────────────
bold "Gallery bucket"

if gcloud storage buckets describe "gs://${GCS_BUCKET}" >/dev/null 2>&1; then
  ok "gs://${GCS_BUCKET} already exists"
else
  # Uniform access + public access prevention: this bucket holds derived
  # biometric data and must never become world-readable by an ACL mistake.
  gcloud storage buckets create "gs://${GCS_BUCKET}" \
    --location="$GCP_REGION" \
    --uniform-bucket-level-access \
    --public-access-prevention
  ok "created gs://${GCS_BUCKET} (uniform access, public access prevented)"
fi

info "granting the runtime account read-only access to the bucket"
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectViewer" >/dev/null
ok "bucket read access granted"

# Versioning means a bad gallery upload is one command to undo.
gcloud storage buckets update "gs://${GCS_BUCKET}" --versioning >/dev/null
ok "object versioning enabled (a bad upload can be rolled back)"

# ── Secrets ──────────────────────────────────────────────────────────
bold "Secrets"

create_secret_if_missing() {
  local name="$1" value="$2"
  if gcloud secrets describe "$name" >/dev/null 2>&1; then
    ok "$name already exists (left untouched)"
  else
    printf '%s' "$value" | gcloud secrets create "$name" --data-file=- --replication-policy=automatic
    ok "created $name"
  fi
}

# Generated here so a strong value is the default and nobody has to invent one.
gen() { node -e "console.log(require('crypto').randomBytes(32).toString('base64url'))"; }

command -v node >/dev/null || { echo "ERROR: node not found (needed to generate secrets)."; exit 1; }

create_secret_if_missing "snp-qr-signing-secret"    "$(gen)"
create_secret_if_missing "snp-token-signing-secret" "$(gen)"

if gcloud secrets describe "snp-faculty-access-code" >/dev/null 2>&1; then
  ok "snp-faculty-access-code already exists (left untouched)"
else
  if [[ -n "${FACULTY_ACCESS_CODE:-}" ]]; then
    create_secret_if_missing "snp-faculty-access-code" "$FACULTY_ACCESS_CODE"
    warn "faculty access code taken from deploy.env — remove it from that file now"
  else
    GENERATED_CODE="$(gen | cut -c1-20)"
    create_secret_if_missing "snp-faculty-access-code" "$GENERATED_CODE"
    bold ""
    bold "  ┌─────────────────────────────────────────────────────────┐"
    printf "  │  FACULTY ACCESS CODE:  %-32s │\n" "$GENERATED_CODE"
    bold "  └─────────────────────────────────────────────────────────┘"
    warn "Write this down NOW — it is not shown again. Faculty type it to start a session."
    warn "To read it back later:  gcloud secrets versions access latest --secret=snp-faculty-access-code"
    bold ""
  fi
fi

# ── Sheets ───────────────────────────────────────────────────────────
bold "Google Sheets"
info "Share your attendance spreadsheet with this address, as Editor:"
printf "\n      \033[1m%s\033[0m\n\n" "$SA_EMAIL"
info "Then put the spreadsheet ID in deploy.env as SHEET_ID."
info "The ID is the long string in the URL:"
info "  https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit"

# ── Summary ──────────────────────────────────────────────────────────
bold "Setup complete"
cat <<EOF

  Project          $GCP_PROJECT_ID
  Region           $GCP_REGION
  Service account  $SA_EMAIL
  Gallery bucket   gs://${GCS_BUCKET}

  Next:
    1. Share the attendance sheet with $SA_EMAIL (Editor)
    2. Upload your gallery:
         gcloud storage cp gallery/face_db.npz gs://${GCS_BUCKET}/face_db.npz
    3. Deploy:
         ./scripts/deploy-backend.sh

EOF
