"use strict";

/**
 * Fetch the enrollment gallery from Cloud Storage before workers start.
 *
 * WHY NOT BAKE IT INTO THE IMAGE
 * ------------------------------
 * Face embeddings are derived biometric data. Baking them into a container
 * image spreads them into the registry, into every cached layer on every
 * build machine, and into the deploy history — and updating the gallery would
 * mean a full rebuild and redeploy. Fetching at startup keeps exactly one
 * authoritative copy, in a private bucket with its own IAM and audit log, and
 * makes a gallery update a single file upload.
 *
 * WHY NOT GCSFUSE
 * ---------------
 * A mount adds cold-start latency and a second failure mode (a stalled mount
 * looks like a missing file). The gallery is a few hundred KB read exactly
 * once per container. A plain download is simpler and fails loudly.
 *
 * ORDERING: this runs BEFORE the verifier pool starts, and /readyz stays 503
 * until a worker has loaded the file — so an orchestrator holds traffic back
 * rather than failing the first students of the class.
 */

const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");

const STORAGE_HOST = "https://storage.googleapis.com";
const DOWNLOAD_TIMEOUT_MS = 60_000;
const MAX_ATTEMPTS = 3;

/** `gs://bucket/path/to/object` -> { bucket, object } */
function parseGcsUri(uri) {
  const match = /^gs:\/\/([^/]+)\/(.+)$/.exec(String(uri).trim());
  if (!match) {
    throw new Error(
      `GALLERY_GCS_URI must look like gs://bucket-name/face_db.npz (got "${uri}")`
    );
  }
  return { bucket: match[1], object: match[2] };
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Download the gallery if GALLERY_GCS_URI is configured.
 *
 * @returns {Promise<{source: string, path: string, bytes?: number}>}
 */
async function ensureGallery({ config, auth, logger }) {
  const uri = (process.env.GALLERY_GCS_URI || "").trim();
  const destination = path.join(config.paths.galleryDir, "face_db.npz");
  const legacy = path.join(config.paths.galleryDir, "face_db.pkl");

  if (!uri) {
    // No bucket configured: a local or baked-in gallery must already exist.
    if (fs.existsSync(destination) || fs.existsSync(legacy)) {
      logger.info("[gallery] using the local gallery", {
        dir: config.paths.galleryDir,
      });
      return { source: "local", path: destination };
    }
    throw new Error(
      `No gallery found in ${config.paths.galleryDir} and GALLERY_GCS_URI is not set.\n` +
        `  • Production: set GALLERY_GCS_URI=gs://your-bucket/face_db.npz\n` +
        `  • Local:      run "npm run build:gallery" to create gallery/face_db.npz\n` +
        `  See docs/DEPLOYMENT.md.`
    );
  }

  const { bucket, object } = parseGcsUri(uri);
  await fsp.mkdir(config.paths.galleryDir, { recursive: true });

  const url =
    `${STORAGE_HOST}/storage/v1/b/${encodeURIComponent(bucket)}` +
    `/o/${encodeURIComponent(object)}?alt=media`;

  let lastError;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt += 1) {
    try {
      const token = await auth.getAccessToken();

      const response = await fetch(url, {
        headers: { Authorization: `Bearer ${token}` },
        signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS),
      });

      if (response.status === 401 || response.status === 403) {
        auth.invalidate();
        throw new Error(
          `access denied (${response.status}). The service account needs ` +
            `roles/storage.objectViewer on gs://${bucket}.`
        );
      }
      if (response.status === 404) {
        // Not retryable — retrying a missing object just delays a clear error.
        throw Object.assign(
          new Error(
            `gs://${bucket}/${object} does not exist. Upload the gallery with:\n` +
              `  gcloud storage cp gallery/face_db.npz gs://${bucket}/${object}`
          ),
          { fatal: true }
        );
      }
      if (!response.ok) {
        throw new Error(`download failed with HTTP ${response.status}`);
      }

      const buffer = Buffer.from(await response.arrayBuffer());
      if (buffer.length < 128) {
        throw new Error(`downloaded object is only ${buffer.length} bytes — not a gallery`);
      }

      // Write to a temp path then rename, so a partial download can never be
      // observed as a valid gallery by a worker starting concurrently.
      const temp = `${destination}.download`;
      await fsp.writeFile(temp, buffer);
      await fsp.rename(temp, destination);

      logger.info("[gallery] downloaded from Cloud Storage", {
        uri,
        bytes: buffer.length,
        attempt,
      });
      return { source: uri, path: destination, bytes: buffer.length };
    } catch (error) {
      lastError = error;
      if (error.fatal || attempt === MAX_ATTEMPTS) break;
      const backoff = 500 * 2 ** (attempt - 1);
      logger.warn(`[gallery] attempt ${attempt} failed, retrying in ${backoff}ms`, {
        error: error.message,
      });
      await sleep(backoff);
    }
  }

  // Fall back to a gallery already on disk rather than refusing to serve:
  // a transient Cloud Storage failure should not cancel a class when a
  // perfectly good copy from the previous boot is present.
  if (fs.existsSync(destination) || fs.existsSync(legacy)) {
    logger.error(
      `[gallery] could not refresh from ${uri} (${lastError.message}) — ` +
        `continuing with the copy already on disk, which may be stale.`
    );
    return { source: "stale-local", path: destination };
  }

  throw new Error(`could not obtain the gallery from ${uri}: ${lastError.message}`);
}

module.exports = { ensureGallery, parseGcsUri };
