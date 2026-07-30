#!/usr/bin/env bash
#
# Upload the face gallery to private storage.
#
#   ./scripts/upload-gallery.sh
#
# Supports both deployment paths:
#   • Hugging Face private dataset  (free, no billing)   — GALLERY_DATASET
#   • Google Cloud Storage          (needs billing)      — GCS_BUCKET
#
# Only the EMBEDDINGS are uploaded. Raw photographs never leave your machine —
# they are sensitive personal data and the runtime has no use for them.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -f deploy.env ]] || { echo "ERROR: deploy.env not found.  cp deploy.env.example deploy.env"; exit 1; }
# shellcheck disable=SC1091
source deploy.env

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
info() { printf "  \033[36m•\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$*"; }

if [[ ! -f gallery/face_db.npz ]]; then
  fail "gallery/face_db.npz not found."
  echo "     Build it first:  npm run build:gallery"
  echo "     (No enrollment images yet? See docs/ENROLLMENT.md.)"
  exit 1
fi

SIZE=$(wc -c < gallery/face_db.npz | tr -d ' ')
ok "gallery/face_db.npz (${SIZE} bytes)"

# ── Hugging Face dataset (the free path) ─────────────────────────────
if [[ -n "${GALLERY_DATASET:-}" ]]; then
  : "${HF_USERNAME:?set HF_USERNAME in deploy.env}"
  REPO="${HF_USERNAME}/${GALLERY_DATASET}"
  WORKDIR="$(mktemp -d)"
  trap 'rm -rf "$WORKDIR"' EXIT

  bold "Uploading to the private dataset ${REPO}"
  cat <<EOF

  If the dataset does not exist yet:
    1. https://huggingface.co/new-dataset
    2. Owner: ${HF_USERNAME}   Name: ${GALLERY_DATASET}
    3. Visibility: PRIVATE     ← this holds biometric data. Not negotiable.
    4. Create dataset

EOF
  read -r -p "  Press Enter once the PRIVATE dataset exists… "

  info "cloning (username = ${HF_USERNAME}, password = an HF token with write access)"
  git clone --depth 1 "https://huggingface.co/datasets/${REPO}" "$WORKDIR/repo" 2>&1 | sed 's/^/    /'

  cp gallery/face_db.npz "$WORKDIR/repo/face_db.npz"
  cd "$WORKDIR/repo"
  git add face_db.npz

  if git diff --cached --quiet; then
    ok "gallery is already up to date — nothing to push"
  else
    git -c user.email=noreply@localhost -c user.name="gallery upload" \
      commit -q -m "Update face gallery ($(date -u +%Y-%m-%dT%H:%MZ))"
    git push -q origin main 2>&1 | sed 's/^/    /'
    ok "pushed to ${REPO}"
  fi
  cd - >/dev/null

  cat <<EOF

  Set on your Space (Settings → Variables and secrets):
    GALLERY_URL   hf://${REPO}/face_db.npz     (variable)
    HF_TOKEN      <a READ token>                (secret)

  Then Restart the Space so it picks up the new gallery.

EOF
  exit 0
fi

# ── Google Cloud Storage (the billed path) ───────────────────────────
if [[ -n "${GCS_BUCKET:-}" ]]; then
  command -v gcloud >/dev/null || { fail "gcloud not found"; exit 1; }
  bold "Uploading to gs://${GCS_BUCKET}"
  gcloud storage cp gallery/face_db.npz "gs://${GCS_BUCKET}/face_db.npz"
  ok "uploaded"
  echo ""
  echo "  Restart Cloud Run to pick it up:"
  echo "    ./scripts/warm.sh off && ./scripts/warm.sh on"
  exit 0
fi

fail "Neither GALLERY_DATASET nor GCS_BUCKET is set in deploy.env."
echo "     Free path:   GALLERY_DATASET=snp-gallery  (+ HF_USERNAME)"
echo "     Google path: GCS_BUCKET=your-bucket-name"
exit 1
