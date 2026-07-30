# syntax=docker/dockerfile:1
#
# Sorry No Proxy — attendance backend.
#
# CRITICAL FIX vs the previous revision of this file
# --------------------------------------------------
# The Python venv was built in `python:3.11-slim-bookworm` (interpreter at
# /usr/local/bin/python3.11) and copied into a `node:20-bookworm-slim`
# runtime, where that path does not exist. A venv records its interpreter by
# absolute path in pyvenv.cfg and in bin/ symlinks, so /opt/venv/bin/python
# arrived as a DANGLING SYMLINK — every verifier spawn would have failed at
# runtime with ENOENT, after the image built and pushed cleanly.
#
# Both stages now share the same base and the same Debian interpreter at
# /usr/bin/python3.11, so the venv is portable between them.
#
# EARLIER FIXES RETAINED
# • .dockerignore added — `COPY . .` used to ship .venv, .git, __pycache__ and
#   the host's macOS-built node_modules over the Linux `npm ci` result.
# • Pre-warms buffalo_l, not buffalo_s. The mismatch was a trap: "fixing" it
#   the other way silently invalidates every embedding and rejects everyone.
# • No build-time embedding job (it no longer exists) and no `RUN ls` debug.
# • Runs as a non-root user.

ARG NODE_IMAGE=node:20-bookworm-slim

# ── Stage 1: Node dependencies ───────────────────────────────────────
FROM ${NODE_IMAGE} AS node-deps

WORKDIR /app
COPY package.json package-lock.json ./
# The tree is ~5.8MB (was 229MB before googleapis was replaced with a direct
# REST client), so this layer is small and caches well.
RUN npm ci --omit=dev && npm cache clean --force


# ── Stage 2: Python venv + model ─────────────────────────────────────
# Same base as the runtime so /usr/bin/python3.11 is identical in both.
FROM ${NODE_IMAGE} AS python-deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    INSIGHTFACE_HOME=/opt/insightface

# Build toolchain compiles insightface's Cython extensions and is discarded
# with this stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv build-essential \
        libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

# Fetch the model at build time so the first request of the day does not wait
# on a ~300MB download. buffalo_l — the same model the verifier and the
# gallery builder use.
RUN mkdir -p "${INSIGHTFACE_HOME}" && \
    python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection','recognition'], root='/opt/insightface'); \
app.prepare(ctx_id=-1); \
print('buffalo_l cached')"

# Fail the build here rather than at 9am in a lecture hall.
RUN /opt/venv/bin/python -c "import cv2, numpy, onnxruntime, insightface; print('python runtime OK')"


# ── Stage 3: Runtime ─────────────────────────────────────────────────
FROM ${NODE_IMAGE} AS runtime

ENV NODE_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INSIGHTFACE_HOME=/opt/insightface \
    PATH="/opt/venv/bin:$PATH" \
    PYTHON_BIN=/opt/venv/bin/python \
    GALLERY_DIR=/app/gallery \
    PORT=7860 \
    # One core per verifier worker; parallelism comes from the pool. Without
    # these, ONNX Runtime and BLAS each grab every core, oversubscribe the CPU
    # and make the pool slower under load.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    # albumentations (pulled in by insightface) phones home for a version
    # check on import. That is a network round trip on every worker start,
    # and it fails noisily in a container with no outbound CA trust.
    NO_ALBUMENTATIONS_UPDATE=1

# `python3` (not python3-minimal): the venv symlinks to the system stdlib, and
# the minimal package omits modules the verifier imports.
# opencv-python-headless still needs libgomp1 and libglib2.0-0.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --chown on COPY avoids a second full-size layer from a later `chown -R`.
COPY --from=python-deps --chown=node:node /opt/venv /opt/venv
COPY --from=python-deps --chown=node:node /opt/insightface /opt/insightface
COPY --from=node-deps  --chown=node:node /app/node_modules ./node_modules

COPY --chown=node:node package.json ./
COPY --chown=node:node server ./server
COPY --chown=node:node python ./python
COPY --chown=node:node public ./public

# Embeddings only — the raw photographs are excluded by .dockerignore. If
# gallery/ is absent at build time, create it and mount the gallery at
# runtime; /readyz reports 503 until a worker loads one.
COPY --chown=node:node gallery ./gallery

# Unit tests live beside the code for discoverability but have no business in
# a production image.
RUN find /app/server -name '*.test.js' -delete && \
    find /app/python -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

USER node

EXPOSE 7860

# Verify the venv actually resolves in THIS image — the exact failure the
# rebuild above fixes. Cheap insurance against it regressing.
RUN /opt/venv/bin/python -c "import sys, cv2, insightface; print('runtime python OK', sys.version)"

# Liveness only. Readiness (model warm) is /readyz — an orchestrator should
# gate traffic on that, not on this. Note: Cloud Run ignores Docker
# HEALTHCHECK; configure its startup probe against /readyz instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||7860)+'/healthz').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server/index.js"]
