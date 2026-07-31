#!/usr/bin/env bash
#
# Build and deploy the faculty QR portal to Cloudflare Pages.
#
#   ./scripts/deploy-faculty.sh
#
# The backend URL is baked in at BUILD time (Vite inlines import.meta.env),
# so this must run after the backend exists — and must be re-run if the
# backend URL ever changes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f deploy.env ]]; then
  echo "ERROR: deploy.env not found.  cp deploy.env.example deploy.env"
  exit 1
fi
# shellcheck disable=SC1091
source deploy.env

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
info() { printf "  \033[36m•\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$*"; }

PAGES_PROJECT="${PAGES_PROJECT:-snp-faculty}"

# ── Resolve the backend URL ──────────────────────────────────────────
bold "Preflight"

# Resolve the backend URL, in order of specificity.
API_BASE="${VITE_API_BASE:-}"
[[ -z "$API_BASE" ]] && API_BASE="${BACKEND_URL:-}"
if [[ -z "$API_BASE" && -n "${NGROK_DOMAIN:-}" ]]; then
  API_BASE="https://${NGROK_DOMAIN}"
fi
if [[ -z "$API_BASE" ]] && command -v gcloud >/dev/null; then
  info "no BACKEND_URL set — trying Cloud Run"
  API_BASE="$(gcloud run services describe "${SERVICE_NAME:-snp-attendance}" \
    --region "${GCP_REGION:-asia-south1}" --format='value(status.url)' 2>/dev/null || true)"
fi

if [[ -z "$API_BASE" ]]; then
  fail "Could not determine the backend URL."
  echo "  Deploy the backend first (./scripts/deploy-backend.sh), or set"
  echo "  VITE_API_BASE=https://... in deploy.env."
  exit 1
fi
ok "backend: $API_BASE"

# A portal built against an unreachable backend looks fine and fails at login,
# so check now rather than in front of a class.
if curl -fsS --max-time 15 -H "ngrok-skip-browser-warning: true" "${API_BASE}/healthz" >/dev/null 2>&1; then
  ok "backend is reachable"
else
  fail "backend did not respond at ${API_BASE}/healthz"
  echo "  Deploy or fix the backend before building the portal against it."
  exit 1
fi

command -v npx >/dev/null || { fail "npx not found — install Node 20+"; exit 1; }

# ── Build ────────────────────────────────────────────────────────────
bold "Building"

cd QR-Faculty-Portal
npm ci --no-fund --no-audit
VITE_API_BASE="$API_BASE" npm run build
cd "$REPO_ROOT"

if ! grep -rq "$API_BASE" QR-Faculty-Portal/dist/assets/*.js; then
  fail "the built bundle does not contain the backend URL — VITE_API_BASE did not take effect"
  exit 1
fi
ok "backend URL baked into the bundle"

# ── Deploy ───────────────────────────────────────────────────────────
bold "Deploying to Cloudflare Pages"
info "a browser window will open for login on first use"

npx --yes wrangler@latest pages deploy QR-Faculty-Portal/dist \
  --project-name "$PAGES_PROJECT" \
  --branch main \
  --commit-dirty=true

cat <<EOF

$(bold "Deployed")

  Faculty portal is live. Cloudflare printed the URL above — it looks like:
    https://${PAGES_PROJECT}.pages.dev

  ┌─ ONE MORE STEP ────────────────────────────────────────────────┐
  │  The backend must allow that origin or sign-in will fail with   │
  │  a CORS error. Put it in deploy.env:                            │
  │                                                                 │
  │      FACULTY_ORIGIN=https://${PAGES_PROJECT}.pages.dev
  │                                                                 │
  │  then re-run:  ./scripts/deploy-backend.sh                      │
  └─────────────────────────────────────────────────────────────────┘

EOF
