# gallery/

Enrollment data. **Nothing in here is committed** — see `.gitignore`.

| File | What it is | Where it comes from |
|---|---|---|
| `face_db.npz` | Face embeddings the verifier loads | `npm run build:gallery`, or downloaded from Cloud Storage at startup |
| `images/` | Raw enrollment photographs | You, after the enrollment session (`docs/ENROLLMENT.md`) |
| `rejected/` | Images that failed quality gates | Written by `npm run build:gallery` |

This directory is tracked (via `.gitkeep`) but empty, because the Docker build
copies it and would fail if it did not exist.

**In production the gallery is NOT in the image.** Cloud Run downloads
`face_db.npz` from a private GCS bucket at startup — set `GALLERY_GCS_URI`.
See `docs/DEPLOYMENT.md`.

Raw photographs are sensitive personal data under the DPDP Act 2023. They must
never be committed, never enter a container image, and never be served over
HTTP. The server refuses to serve this directory.
