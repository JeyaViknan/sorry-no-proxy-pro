"use strict";

/**
 * Google access tokens, from whichever credential source is available.
 *
 * TWO MODES, PREFERRED ORDER
 * --------------------------
 * 1. METADATA SERVER (Cloud Run, GCE, GKE). The platform vends a token for
 *    the service account attached to the revision. **No private key exists in
 *    the environment at all** — nothing to leak in an env dump, a log line, or
 *    a screenshot of the Cloud Run console. This is the production path.
 *
 * 2. EXPLICIT SERVICE-ACCOUNT KEY (local development, non-Google hosts).
 *    Signs an RFC 7523 JWT assertion and exchanges it for a token.
 *
 * Still no `googleapis` dependency: that package was 204MB of the project's
 * 229MB node_modules and carried seven unfixable high-severity advisories, to
 * do what these ~120 lines do.
 */

const crypto = require("crypto");

const METADATA_HOST = "http://metadata.google.internal";
const METADATA_TOKEN_PATH =
  "/computeMetadata/v1/instance/service-accounts/default/token";
const TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token";

/** Refresh this many seconds before expiry so a token never dies in flight. */
const EXPIRY_MARGIN_SEC = 60;

function base64url(input) {
  return Buffer.from(input).toString("base64url");
}

class GoogleAuth {
  /**
   * @param {object} options
   * @param {string} [options.clientEmail] service-account email (mode 2)
   * @param {string} [options.privateKey]  PEM private key (mode 2)
   * @param {string[]} options.scopes      OAuth scopes (mode 2 only; the
   *   metadata server issues a token with the service account's own scopes)
   * @param {object} logger
   */
  constructor({ clientEmail = "", privateKey = "", scopes = [] }, logger = console) {
    this.clientEmail = clientEmail;
    this.privateKey = privateKey;
    this.scopes = scopes;
    this.logger = logger;

    this.token = null;
    this.expiresAtSec = 0;
    this.mode = null;
    /** In-flight refresh, so concurrent callers share one request. */
    this.pending = null;
  }

  get hasExplicitKey() {
    return Boolean(this.clientEmail && this.privateKey);
  }

  /** Probe the metadata server. Short timeout: off-platform it never answers. */
  async #tryMetadata() {
    const response = await fetch(`${METADATA_HOST}${METADATA_TOKEN_PATH}`, {
      headers: { "Metadata-Flavor": "Google" },
      signal: AbortSignal.timeout(2000),
    });
    if (!response.ok) throw new Error(`metadata server returned ${response.status}`);

    const data = await response.json();
    if (!data.access_token) throw new Error("metadata server returned no access_token");
    return { token: data.access_token, expiresInSec: data.expires_in || 3600 };
  }

  async #trySignedJwt() {
    const now = Math.floor(Date.now() / 1000);

    const header = base64url(JSON.stringify({ alg: "RS256", typ: "JWT" }));
    const claims = base64url(
      JSON.stringify({
        iss: this.clientEmail,
        scope: this.scopes.join(" "),
        aud: TOKEN_ENDPOINT,
        iat: now,
        exp: now + 3600,
      })
    );

    const signer = crypto.createSign("RSA-SHA256");
    signer.update(`${header}.${claims}`);

    let signature;
    try {
      signature = signer.sign(this.privateKey, "base64url");
    } catch (error) {
      // Almost always a mangled key: literal "\n" not converted, or the
      // BEGIN/END armour stripped by a shell or a secret manager.
      throw new Error(
        `could not sign with GOOGLE_PRIVATE_KEY (${error.message}). ` +
          `The key must include the BEGIN/END PRIVATE KEY lines and real newlines.`
      );
    }

    const response = await fetch(TOKEN_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
        assertion: `${header}.${claims}.${signature}`,
      }),
      signal: AbortSignal.timeout(10_000),
    });

    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`token exchange failed (${response.status}): ${detail.slice(0, 200)}`);
    }

    const data = await response.json();
    return { token: data.access_token, expiresInSec: data.expires_in || 3600 };
  }

  async #refresh() {
    const errors = [];

    // Metadata first: on Cloud Run this is the credential we want, and it
    // needs no secret material.
    try {
      const result = await this.#tryMetadata();
      if (this.mode !== "metadata") {
        this.logger.info?.("[google-auth] using the instance service account (metadata server)");
      }
      this.mode = "metadata";
      return result;
    } catch (error) {
      errors.push(`metadata: ${error.message}`);
    }

    if (this.hasExplicitKey) {
      try {
        const result = await this.#trySignedJwt();
        if (this.mode !== "service-account-key") {
          this.logger.info?.("[google-auth] using an explicit service-account key");
        }
        this.mode = "service-account-key";
        return result;
      } catch (error) {
        errors.push(`service-account key: ${error.message}`);
      }
    } else {
      errors.push("service-account key: not configured");
    }

    throw new Error(`no usable Google credentials — ${errors.join("; ")}`);
  }

  /** Cached access token. Concurrent callers share a single refresh. */
  async getAccessToken() {
    const now = Math.floor(Date.now() / 1000);
    if (this.token && now < this.expiresAtSec - EXPIRY_MARGIN_SEC) {
      return this.token;
    }

    if (!this.pending) {
      this.pending = this.#refresh()
        .then(({ token, expiresInSec }) => {
          this.token = token;
          this.expiresAtSec = Math.floor(Date.now() / 1000) + expiresInSec;
          return token;
        })
        .finally(() => {
          this.pending = null;
        });
    }
    return this.pending;
  }

  /** Force a refresh on the next call (used after a 401). */
  invalidate() {
    this.token = null;
    this.expiresAtSec = 0;
  }

  status() {
    return {
      mode: this.mode || "unresolved",
      hasExplicitKey: this.hasExplicitKey,
      tokenCached: Boolean(this.token),
    };
  }
}

module.exports = { GoogleAuth };
