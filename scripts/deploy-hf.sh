#!/usr/bin/env bash
#
# Deploy the backend to a Hugging Face Space (free, no credit card).
#
#   ./scripts/deploy-hf.sh
#
# A Space is a git repo that HF builds from a Dockerfile. This pushes the
# current commit to it and then waits for the build to come up.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

[[ -f deploy.env ]] || { echo "ERROR: deploy.env not found.  cp deploy.env.example deploy.env"; exit 1; }
# shellcheck disable=SC1091
source deploy.env

: "${HF_USERNAME:?set HF_USERNAME in deploy.env}"
: "${HF_SPACE_NAME:?set HF_SPACE_NAME in deploy.env}"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
info() { printf "  \033[36m•\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$*"; }

SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${HF_SPACE_NAME}"
APP_URL="https://${HF_USERNAME}-${HF_SPACE_NAME}.hf.space"

# ── Preflight ────────────────────────────────────────────────────────
bold "Preflight"

command -v git >/dev/null || { fail "git not found"; exit 1; }

if [[ -n "$(git status --porcelain)" ]]; then
  fail "You have uncommitted changes. The Space builds from a commit, so"
  echo "     anything uncommitted will NOT be deployed."
  echo ""
  git status --short | head -10 | sed 's/^/       /'
  echo ""
  read -r -p "  Commit them now? [y/N] " reply
  if [[ "$reply" =~ ^[Yy]$ ]]; then
    git add -A && git commit -m "Deploy to Hugging Face Space"
    ok "committed"
  else
    exit 1
  fi
else
  ok "working tree is clean"
fi

# The Space needs the README frontmatter (sdk: docker, app_port: 7860) to know
# how to build. Without it HF assumes a Gradio app and the build fails oddly.
if ! head -10 README.md | grep -q "sdk: docker"; then
  fail "README.md is missing the Hugging Face frontmatter (sdk: docker)."
  echo "     The Space will try to build this as a Gradio app and fail."
  exit 1
fi
ok "README frontmatter present (sdk: docker, port 7860)"

# ── Remote ───────────────────────────────────────────────────────────
bold "Space remote"

if git remote get-url space >/dev/null 2>&1; then
  ok "remote 'space' -> $(git remote get-url space | sed 's#//.*@#//#')"
else
  info "adding remote 'space' -> ${SPACE_URL}"
  git remote add space "${SPACE_URL}"
  ok "added"
fi

cat <<EOF

  If you have not created the Space yet, do it now:
    1. https://huggingface.co/new-space
    2. Owner: ${HF_USERNAME}    Name: ${HF_SPACE_NAME}
    3. License: whatever you prefer
    4. Select "Docker"  →  "Blank"
    5. Visibility: Public   (students must be able to open it)
    6. Create Space

EOF
read -r -p "  Press Enter once the Space exists… "

# ── Push ─────────────────────────────────────────────────────────────
bold "Pushing"
info "git will ask for credentials: username = ${HF_USERNAME}, password = your HF token"
info "create one at https://huggingface.co/settings/tokens (role: write)"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
# Spaces build from `main`. Map whatever the local branch is called onto it.
git push --force space "${BRANCH}:main"
ok "pushed ${BRANCH} -> space/main"

# ── Secrets reminder ─────────────────────────────────────────────────
bold "Secrets"
cat <<EOF

  Set these at:
    ${SPACE_URL}/settings   →   "Variables and secrets"

  ┌─ SECRETS (values are hidden after saving) ──────────────────────┐
EOF
printf "  │  %-24s %s\n" "QR_SIGNING_SECRET" "$(node -e "console.log(require('crypto').randomBytes(32).toString('base64url'))")"
printf "  │  %-24s %s\n" "TOKEN_SIGNING_SECRET" "$(node -e "console.log(require('crypto').randomBytes(32).toString('base64url'))")"
printf "  │  %-24s %s\n" "FACULTY_ACCESS_CODE" "$(node -e "console.log(require('crypto').randomBytes(15).toString('base64url'))")"
cat <<EOF
  │  HF_TOKEN                 <a READ token, for the gallery dataset>
  │  SHEET_WEBHOOK_SECRET     <the same value as in your Apps Script>
  └─────────────────────────────────────────────────────────────────┘

  ┌─ VARIABLES (not secret) ────────────────────────────────────────┐
  │  NODE_ENV                 production
  │  GALLERY_URL              hf://${HF_USERNAME}/${GALLERY_DATASET:-snp-gallery}/face_db.npz
  │  ALLOWED_ORIGINS          ${FACULTY_ORIGIN:-https://YOUR-PORTAL.pages.dev},${APP_URL}
  │  SHEET_WEBHOOK_URL        <your Apps Script /exec URL>
  │  FACE_WORKER_COUNT        2
  └─────────────────────────────────────────────────────────────────┘

  ⚠  Write the FACULTY_ACCESS_CODE down — faculty type it to start a session.
     These generated values are shown ONCE, here, and nowhere else.

EOF
read -r -p "  Press Enter once the secrets are saved (the Space will rebuild)… "

# ── Wait for the build ───────────────────────────────────────────────
bold "Waiting for the Space to come up"
info "first build takes 10-20 minutes (it compiles insightface and caches the model)"
info "watch progress at ${SPACE_URL}  →  'Logs'"

READY=""
for i in $(seq 1 120); do
  if curl -fsS --max-time 10 "${APP_URL}/readyz" 2>/dev/null | grep -q '"ready":true'; then
    READY=1; break
  fi
  if (( i % 6 == 0 )); then info "still building… ($((i/6)) min)"; fi
  sleep 10
done

if [[ -z "$READY" ]]; then
  fail "the Space did not become ready within 20 minutes."
  echo ""
  echo "  Check the build log:  ${SPACE_URL}  →  Logs"
  echo ""
  echo "  Most likely causes:"
  echo "    • Secrets not saved yet          -> the app exits at boot; set them, then Restart"
  echo "    • GALLERY_URL / HF_TOKEN wrong   -> log shows 'could not obtain the gallery'"
  echo "    • Still building                 -> the first build genuinely can take 20 min"
  exit 1
fi

ok "Space is ready"

if curl -fsS --max-time 10 "${APP_URL}/" | grep -q "Mark your attendance"; then
  ok "student scanner is being served"
else
  fail "the scanner did not load at ${APP_URL}/"
fi

bold "Deployed"
cat <<EOF

  Backend API + Student Scanner
    ${APP_URL}

  Give students that URL. Point the faculty portal at it:
    VITE_API_BASE=${APP_URL}

  Logs:  ${SPACE_URL}  →  Logs
  Note:  a free Space sleeps after ~48h idle and wakes on the next request
         (~40s). Open it once before class.

EOF
