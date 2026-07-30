#!/usr/bin/env bash
#
# Upload the face gallery to Cloud Storage.
#
#   ./scripts/upload-gallery.sh
#
# Run this after `npm run build:gallery`, and again whenever you re-enroll.
# Cloud Run picks up the new file on its next cold start — force one with:
#   ./scripts/warm.sh off && ./scripts/warm.sh on

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -f deploy.env ]] || { echo "ERROR: deploy.env not found"; exit 1; }
# shellcheck disable=SC1091
source deploy.env
: "${GCS_BUCKET:?set GCS_BUCKET in deploy.env}"

if [[ ! -f gallery/face_db.npz ]]; then
  echo "ERROR: gallery/face_db.npz not found."
  echo "  Build it first:  npm run build:gallery"
  echo "  (No enrollment images yet? See docs/ENROLLMENT.md.)"
  exit 1
fi

SIZE=$(wc -c < gallery/face_db.npz | tr -d ' ')
echo "Uploading gallery/face_db.npz (${SIZE} bytes) to gs://${GCS_BUCKET}/face_db.npz"

# Only the embeddings go up. Raw photographs stay on your machine — they are
# sensitive personal data and the runtime never needs them.
gcloud storage cp gallery/face_db.npz "gs://${GCS_BUCKET}/face_db.npz"

echo ""
echo "Uploaded. Verify what the server will load:"
echo "  npm run verify:gallery"
echo ""
echo "Restart Cloud Run to pick it up:"
echo "  ./scripts/warm.sh off && ./scripts/warm.sh on"
