# syntax=docker/dockerfile:1
#
# Sorry No Proxy — attendance backend.
#
# FIXES FROM THE PREVIOUS DOCKERFILE
# ---------------------------------
# • No .dockerignore existed, so `COPY . .` shipped .venv, .git, __pycache__
#   and — worst — the host's macOS-built node_modules straight over the Linux
#   `npm ci` result. The dependencies happened to be pure JS so it did not
#   break, but it was one native module away from a broken image, plus ~250MB
#   of dead layer.
# • It pre-warmed `buffalo_s` while the Python code requested `buffalo_l`.
#   Wasted bandwidth, and a trap: "fixing" the mismatch in the wrong direction
#   silently invalidates every stored embedding and rejects every student.
#   Only buffalo_l is fetched now, and gallery.py refuses to load a mismatch.
# • It ran a build-time embedding job that no longer exists and left a
#   `RUN ls -lh` debug line in a production image.
# • Ran as root.

# ── Stage 1: Node dependencies ───────────────────────────────────────
FROM node:20-bookworm-slim AS node-deps

WORKDIR /app
COPY package.json package-lock.json ./
# `npm ci --omit=dev` against the committed lockfile. The whole tree is now
# ~5.8MB (it was 229MB before googleapis was replaced with a direct REST
# client), so this layer is small and caches well.
RUN npm ci --omit=dev && npm cache clean --force


# ── Stage 2: Python dependencies + model ─────────────────────────────
FROM python:3.11-slim-bookworm AS python-deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    INSIGHTFACE_HOME=/opt/insightface

# Build toolchain is needed to compile insightface's Cython extensions, and
# is discarded with this stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

# Fetch the model at build time so the first request of the day does not sit
# behind a ~300MB download. This is buffalo_l — the same model the verifier
# and the gallery builder use.
RUN mkdir -p "${INSIGHTFACE_HOME}" && \
    python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection','recognition'], root='/opt/insightface'); \
app.prepare(ctx_id=-1); \
print('buffalo_l cached')"


# ── Stage 3: Runtime ─────────────────────────────────────────────────
FROM node:20-bookworm-slim AS runtime

ENV NODE_ENV=production \
    PYTHONUNBUFFERED=1 \
    INSIGHTFACE_HOME=/opt/insightface \
    PATH="/opt/venv/bin:$PATH" \
    PYTHON_BIN=/opt/venv/bin/python \
    GALLERY_DIR=/app/gallery \
    PORT=7860 \
    # Each verifier worker gets one core; parallelism comes from the pool.
    # Without these, ONNX Runtime and BLAS each grab every core and
    # oversubscribe the CPU, making the pool slower under load.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Runtime shared libraries only — no compilers, no X11. opencv-python-headless
# still needs libgomp and libglib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-minimal libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=python-deps /opt/venv /opt/venv
COPY --from=python-deps /opt/insightface /opt/insightface
COPY --from=node-deps  /app/node_modules ./node_modules

# Application code. .dockerignore keeps out node_modules, .venv, .git,
# gallery/images (raw biometrics) and local scratch.
COPY package.json ./
COPY server ./server
COPY python ./python
COPY public ./public

# Embeddings only — never the source photographs. If gallery/face_db.npz is
# absent at build time the image still starts; mount or bake it before
# serving traffic, and /readyz will report 503 until it is present.
COPY gallery* ./gallery/

# Drop privileges. The previous image ran everything as root.
RUN chown -R node:node /app
USER node

EXPOSE 7860

# Liveness only. Readiness (model warm) is /readyz — an orchestrator should
# gate traffic on that, not on this.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||7860)+'/healthz').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server/index.js"]
