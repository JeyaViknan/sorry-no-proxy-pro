FROM node:20-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/opt/insightface \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

# System libs required by OpenCV/DeepFace/RetinaFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Use virtualenv to avoid Debian's externally-managed Python restriction (PEP 668)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY package*.json ./
RUN npm ci --omit=dev

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

# Pre-download InsightFace model assets during build so runtime requests do not
# trigger model downloads and timeout behind a progress bar.
RUN mkdir -p "${INSIGHTFACE_HOME}" && python - <<'PY'
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_s", allowed_modules=["detection", "recognition"], root="/opt/insightface")
app.prepare(ctx_id=-1)
print("InsightFace model warmup complete")
PY

COPY data data/
COPY test test/
RUN python test/generate_embeddings.py
RUN ls -lh face_db.pkl

COPY . .

ENV NODE_ENV=production
EXPOSE 10000
CMD ["node", "server.js"]
