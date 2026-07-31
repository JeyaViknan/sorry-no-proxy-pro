#!/usr/bin/env bash
#
# Run the backend on this machine and expose it publicly via ngrok.
#
#   ./scripts/start-local.sh          start
#   ./scripts/start-local.sh stop     stop
#   ./scripts/start-local.sh logs     follow the logs
#
# The gallery is mounted read-only from ./gallery, so no cloud storage and no
# HF token are involved — this is the simplest possible deployment that still
# gives students a real HTTPS URL.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -f deploy.env ]] || { echo "ERROR: deploy.env not found."; exit 1; }
# shellcheck disable=SC1091
source deploy.env

CONTAINER=snp-backend
IMAGE=snp-attendance:local
PORT=7860

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
info() { printf "  \033[36m•\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$*"; }

# ── stop / logs ──────────────────────────────────────────────────────
case "${1:-start}" in
  stop)
    docker rm -f "$CONTAINER" >/dev/null 2>&1 && ok "container stopped" || info "container was not running"
    pkill -f "ngrok http" 2>/dev/null && ok "ngrok stopped" || info "ngrok was not running"
    exit 0 ;;
  logs)
    exec docker logs -f "$CONTAINER" ;;
esac

# ── preflight ────────────────────────────────────────────────────────
bold "Preflight"

if ! docker info >/dev/null 2>&1; then
  fail "Docker is not running. Open Docker Desktop, wait for it to say 'Running', then retry."
  exit 1
fi
ok "docker is running"

command -v ngrok >/dev/null || { fail "ngrok not found"; exit 1; }
ok "ngrok found"

[[ -f gallery/face_db.npz ]] || { fail "gallery/face_db.npz missing — run: npm run build:gallery"; exit 1; }
ok "gallery present ($(wc -c < gallery/face_db.npz | tr -d ' ') bytes)"

: "${NGROK_DOMAIN:?set NGROK_DOMAIN in deploy.env}"
: "${QR_SIGNING_SECRET:?set QR_SIGNING_SECRET in deploy.env}"
: "${TOKEN_SIGNING_SECRET:?set TOKEN_SIGNING_SECRET in deploy.env}"
: "${FACULTY_ACCESS_CODE:?set FACULTY_ACCESS_CODE in deploy.env}"

BACKEND_URL="https://${NGROK_DOMAIN}"
ORIGINS="${BACKEND_URL}"
[[ -n "${FACULTY_ORIGIN:-}" ]] && ORIGINS="${ORIGINS},${FACULTY_ORIGIN}"
ok "public URL: ${BACKEND_URL}"

# ── build if needed ──────────────────────────────────────────────────
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  bold "Building the image (first time only — 10 to 20 minutes)"
  info "it compiles the face library and caches the 190MB model"
  docker build -t "$IMAGE" .
  ok "image built"
else
  ok "image already built (delete it with: docker rmi $IMAGE to force a rebuild)"
fi

# ── run ──────────────────────────────────────────────────────────────
bold "Starting the backend"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d --name "$CONTAINER" \
  -p "${PORT}:7860" \
  -v "$PWD/gallery:/app/gallery:ro" \
  -e NODE_ENV=production \
  -e QR_SIGNING_SECRET="$QR_SIGNING_SECRET" \
  -e TOKEN_SIGNING_SECRET="$TOKEN_SIGNING_SECRET" \
  -e FACULTY_ACCESS_CODE="$FACULTY_ACCESS_CODE" \
  -e ALLOWED_ORIGINS="$ORIGINS" \
  -e FACE_WORKER_COUNT="${FACE_WORKER_COUNT:-2}" \
  -e FACE_THRESHOLD_ACCEPT="${FACE_THRESHOLD_ACCEPT:-0.55}" \
  -e FACE_THRESHOLD_REVIEW="${FACE_THRESHOLD_REVIEW:-0.45}" \
  -e RATE_LIMIT_GLOBAL_PER_MIN="${RATE_LIMIT_GLOBAL_PER_MIN:-6000}" \
  ${SHEET_WEBHOOK_URL:+-e SHEET_WEBHOOK_URL="$SHEET_WEBHOOK_URL"} \
  ${SHEET_WEBHOOK_SECRET:+-e SHEET_WEBHOOK_SECRET="$SHEET_WEBHOOK_SECRET"} \
  --restart unless-stopped \
  "$IMAGE" >/dev/null

info "waiting for the face model to load (about 40 seconds)…"
for i in $(seq 1 40); do
  if curl -fsS --max-time 5 "http://localhost:${PORT}/readyz" 2>/dev/null | grep -q '"ready":true'; then
    ok "backend ready"
    break
  fi
  if ! docker ps --filter "name=$CONTAINER" --format '{{.Names}}' | grep -q .; then
    fail "the container exited. Logs:"
    docker logs --tail 30 "$CONTAINER"
    exit 1
  fi
  sleep 3
done

curl -fsS --max-time 5 "http://localhost:${PORT}/readyz" | grep -q '"ready":true' || {
  fail "backend did not become ready. Logs:"; docker logs --tail 40 "$CONTAINER"; exit 1; }

# ── expose ───────────────────────────────────────────────────────────
bold "Opening the public tunnel"
pkill -f "ngrok http" 2>/dev/null || true
sleep 1
nohup ngrok http "$PORT" --domain="$NGROK_DOMAIN" > /tmp/ngrok-snp.log 2>&1 &
sleep 4

if curl -fsS --max-time 15 "${BACKEND_URL}/healthz" \
     -H "ngrok-skip-browser-warning: true" 2>/dev/null | grep -q '"ok":true'; then
  ok "public URL is live"
else
  fail "the tunnel did not come up. Check /tmp/ngrok-snp.log"
  tail -5 /tmp/ngrok-snp.log
  exit 1
fi

bold "Running"
cat <<EOF

  Student scanner  ${BACKEND_URL}
  Faculty code     ${FACULTY_ACCESS_CODE}

  Keep this Mac awake and connected while class is running:
    caffeinate -dis &          (stops it sleeping until you close the terminal)

  Logs:   ./scripts/start-local.sh logs
  Stop:   ./scripts/start-local.sh stop

EOF
