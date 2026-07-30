#!/usr/bin/env bash
#
# Keep an instance warm around class time.
#
#   ./scripts/warm.sh on    # before class — removes the ~40s cold start
#   ./scripts/warm.sh off   # after class  — back to scale-to-zero (free)
#
# A cold Cloud Run instance must download the gallery and load a ~600MB ONNX
# model. The first student of the day would otherwise wait for all of it.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -f deploy.env ]] || { echo "ERROR: deploy.env not found"; exit 1; }
# shellcheck disable=SC1091
source deploy.env

ACTION="${1:-}"
case "$ACTION" in
  on)  INSTANCES=1 ;;
  off) INSTANCES=0 ;;
  *)   echo "usage: $0 on|off"; exit 1 ;;
esac

gcloud run services update "${SERVICE_NAME:-snp-attendance}" \
  --region "${GCP_REGION:-asia-south1}" \
  --min-instances "$INSTANCES" \
  --quiet

if [[ "$ACTION" == "on" ]]; then
  URL="$(gcloud run services describe "${SERVICE_NAME:-snp-attendance}" \
    --region "${GCP_REGION:-asia-south1}" --format='value(status.url)')"
  echo "Warming… (waiting for the model to load)"
  for _ in $(seq 1 36); do
    if curl -fsS --max-time 10 "${URL}/readyz" 2>/dev/null | grep -q '"ready":true'; then
      echo "Ready. Students can scan now."
      exit 0
    fi
    sleep 5
  done
  echo "Still not ready after 3 minutes — check: gcloud run services logs read ${SERVICE_NAME:-snp-attendance} --region ${GCP_REGION:-asia-south1}"
  exit 1
fi

echo "Scaled to zero. The service is free until the next request."
