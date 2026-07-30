"use strict";

/**
 * Fetch the enrollment gallery before the verifier workers start.
 *
 * WHY NOT SHIP IT IN THE IMAGE
 * ----------------------------
 * Face embeddings are derived biometric data. Baking them into a container
 * image spreads them into the registry, into every cached build layer, and
 * into the deploy history — and updating them would mean a rebuild. Fetching
 * at startup keeps exactly one authoritative copy, in private storage with
 * its own access control, and makes an update a single file upload.
 *
 * It also matters on hosts with an ephemeral filesystem (Hugging Face Spaces,
 * Render, Fly): the disk resets on every restart, so re-fetching at boot is
 * the only thing that works anyway.
 *
 * TWO SOURCES, ONE MECHANISM
 * --------------------------
 *   GALLERY_URL   any HTTPS URL, with an optional bearer token. Covers a
 *                 private Hugging Face dataset, a private GitHub release
 *                 asset, an S3/R2 presigned URL — anything that speaks HTTP.
 *   GALLERY_GCS_URI  gs://bucket/object, using Google credentials.
 *
 * The generic URL path is the default for free-tier deployments, because it
 * needs no cloud provider and no billing account.
 */

const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");

const GCS_HOST = "https://storage.googleapis.com";
const DOWNLOAD_TIMEOUT_MS = 60_000;
const MAX_ATTEMPTS = 3;
const MIN_PLAUSIBLE_BYTES = 128;

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

/**
 * Convenience: turn `hf://owner/dataset/file` into the resolve URL, so nobody
 * has to remember Hugging Face's URL shape. A plain https:// URL passes
 * through untouched.
 */
function normaliseGalleryUrl(raw) {
  const value = String(raw).trim();

  const hf = /^hf:\/\/([^/]+)\/([^/]+)\/(.+)$/.exec(value);
  if (hf) {
    const [, owner, dataset, file] = hf;
    return `https://huggingface.co/datasets/${owner}/${dataset}/resolve/main/${file}`;
  }
  if (!/^https?:\/\//.test(value)) {
    throw new Error(
      `GALLERY_URL must be an https:// URL, or hf://owner/dataset/face_db.npz (got "${value}")`
    );
  }
  return value;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Write atomically so a partial download is never seen as a valid gallery. */
async function writeAtomically(destination, buffer) {
  const temp = `${destination}.download`;
  await fsp.writeFile(temp, buffer);
  await fsp.rename(temp, destination);
}

function describeAuthFailure(status, url) {
  if (url.includes("huggingface.co")) {
    return (
      `access denied (${status}). Check HF_TOKEN is set as a Space secret and ` +
      `has READ access to the dataset repo. A 404 from Hugging Face usually ` +
      `means "private and unauthorised" rather than "missing".`
    );
  }
  return `access denied (${status}). Check GALLERY_AUTH_TOKEN.`;
}

/** Download over plain HTTPS with an optional bearer token. */
async function fetchFromUrl({ url, token, destination, logger }) {
  const resolved = normaliseGalleryUrl(url);
  let lastError;

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt += 1) {
    try {
      const response = await fetch(resolved, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        redirect: "follow", // Hugging Face and GitHub both redirect to a CDN
        signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS),
      });

      if (response.status === 401 || response.status === 403) {
        throw Object.assign(new Error(describeAuthFailure(response.status, resolved)), {
          fatal: true,
        });
      }
      if (response.status === 404) {
        throw Object.assign(
          new Error(
            `not found at ${resolved}. If the repository is private, the token ` +
              `may be missing or lack read access — Hugging Face returns 404 ` +
              `rather than 403 for unauthorised private repos.`
          ),
          { fatal: true }
        );
      }
      if (!response.ok) throw new Error(`download failed with HTTP ${response.status}`);

      const buffer = Buffer.from(await response.arrayBuffer());

      // Check for HTML FIRST, and before the size check. A login wall or an
      // error page arrives with HTTP 200, and a short one would otherwise be
      // reported as "only 33 bytes" — true, but useless — and retried three
      // times even though no amount of retrying will fix a wrong URL.
      const head = buffer.subarray(0, 200).toString("utf8").trimStart().toLowerCase();
      if (head.startsWith("<!doctype") || head.startsWith("<html")) {
        throw Object.assign(
          new Error(
            `the URL returned an HTML page, not a file. It is probably a web ` +
              `page rather than a direct download — for Hugging Face use ` +
              `hf://owner/dataset/face_db.npz, or a URL containing /resolve/.`
          ),
          { fatal: true }
        );
      }
      if (buffer.length < MIN_PLAUSIBLE_BYTES) {
        throw Object.assign(
          new Error(
            `downloaded object is only ${buffer.length} bytes — too small to be ` +
              `a gallery. Check the URL points at face_db.npz itself.`
          ),
          { fatal: true }
        );
      }

      await writeAtomically(destination, buffer);
      logger.info("[gallery] downloaded", {
        source: resolved.replace(/\/\/[^@]*@/, "//"),
        bytes: buffer.length,
        attempt,
      });
      return { source: resolved, path: destination, bytes: buffer.length };
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
  throw lastError;
}

/** Download from Cloud Storage using Google credentials. */
async function fetchFromGcs({ uri, auth, destination, logger }) {
  const { bucket, object } = parseGcsUri(uri);
  const url =
    `${GCS_HOST}/storage/v1/b/${encodeURIComponent(bucket)}` +
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
        throw Object.assign(
          new Error(
            `gs://${bucket}/${object} does not exist. Upload it with:\n` +
              `  gcloud storage cp gallery/face_db.npz gs://${bucket}/${object}`
          ),
          { fatal: true }
        );
      }
      if (!response.ok) throw new Error(`download failed with HTTP ${response.status}`);

      const buffer = Buffer.from(await response.arrayBuffer());
      if (buffer.length < MIN_PLAUSIBLE_BYTES) {
        throw new Error(`downloaded object is only ${buffer.length} bytes — not a gallery`);
      }

      await writeAtomically(destination, buffer);
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
  throw lastError;
}

/**
 * Ensure a gallery is present on disk.
 *
 * @returns {Promise<{source: string, path: string, bytes?: number}>}
 */
async function ensureGallery({ config, auth, logger }) {
  const destination = path.join(config.paths.galleryDir, "face_db.npz");
  const legacy = path.join(config.paths.galleryDir, "face_db.pkl");
  const hasLocal = () => fs.existsSync(destination) || fs.existsSync(legacy);

  const url = (process.env.GALLERY_URL || "").trim();
  const gcsUri = (process.env.GALLERY_GCS_URI || "").trim();

  if (!url && !gcsUri) {
    if (hasLocal()) {
      logger.info("[gallery] using the local gallery", { dir: config.paths.galleryDir });
      return { source: "local", path: destination };
    }
    throw new Error(
      `No gallery found in ${config.paths.galleryDir}, and neither GALLERY_URL ` +
        `nor GALLERY_GCS_URI is set.\n` +
        `  • Free hosting: GALLERY_URL=hf://your-name/snp-gallery/face_db.npz (+ HF_TOKEN)\n` +
        `  • Google Cloud: GALLERY_GCS_URI=gs://your-bucket/face_db.npz\n` +
        `  • Local:        npm run build:gallery\n` +
        `  See docs/DEPLOYMENT.md.`
    );
  }

  await fsp.mkdir(config.paths.galleryDir, { recursive: true });

  try {
    if (url) {
      // HF_TOKEN is the conventional name on Hugging Face Spaces, so accept it
      // directly rather than making people duplicate it.
      const token =
        (process.env.GALLERY_AUTH_TOKEN || "").trim() || (process.env.HF_TOKEN || "").trim();
      return await fetchFromUrl({ url, token, destination, logger });
    }
    return await fetchFromGcs({ uri: gcsUri, auth, destination, logger });
  } catch (error) {
    // Prefer a stale copy over refusing to serve: a transient storage outage
    // should not cancel a class when a perfectly good gallery from the
    // previous boot is already on disk.
    if (hasLocal()) {
      logger.error(
        `[gallery] could not refresh (${error.message}) — continuing with the ` +
          `copy already on disk, which may be stale.`
      );
      return { source: "stale-local", path: destination };
    }
    throw new Error(`could not obtain the gallery: ${error.message}`);
  }
}

module.exports = { ensureGallery, parseGcsUri, normaliseGalleryUrl };
