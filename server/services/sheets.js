"use strict";

/**
 * Google Sheets export — minimal, direct REST.
 *
 * WHY NOT `googleapis`
 * -------------------
 * The official client was 204 MB of the project's 229 MB node_modules and
 * carried 7 unfixable high-severity advisories through its dependency tree
 * (gaxios -> rimraf -> glob -> minimatch ReDoS, plus jws HMAC verification).
 * All of that to append a row to one spreadsheet. Service-account auth is
 * RFC 7523 — a signed JWT exchanged for a bearer token — which Node's built-in
 * crypto does in about thirty lines. Removing the dependency took the install
 * to 5.8 MB with zero advisories, and cut a large chunk of cold-start time
 * (`require("googleapis")` alone is famously slow, which matters on a
 * scale-to-zero host).
 *
 * WHY BATCHED AND OFF THE REQUEST PATH
 * ------------------------------------
 * The old code awaited a Sheets write inside POST /register. That put a
 * third-party network call between the student and their confirmation — the
 * slowest, least reliable link in the chain, on classroom wifi. It also hits
 * the per-user write quota (~60/min) at around 60 students/minute.
 *
 * Here, attendance is already durable in the session store before this runs.
 * Rows queue up and flush in batches, with retry and backoff. A total Sheets
 * outage degrades to "the spreadsheet lags behind" instead of "the class
 * cannot take attendance".
 */

const crypto = require("crypto");

const TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token";
const SCOPE = "https://www.googleapis.com/auth/spreadsheets";
const MAX_BATCH_ROWS = 200;
const MAX_RETRIES = 5;

function base64url(input) {
  return Buffer.from(input).toString("base64url");
}

class SheetsExporter {
  constructor({ enabled, clientEmail, privateKey, spreadsheetId, range, flushIntervalMs }, logger) {
    this.enabled = enabled;
    this.clientEmail = clientEmail;
    this.privateKey = privateKey;
    this.spreadsheetId = spreadsheetId;
    this.range = range;
    this.flushIntervalMs = flushIntervalMs;
    this.logger = logger;

    /** @type {Array<Array<string|number>>} */
    this.queue = [];
    this.accessToken = null;
    this.accessTokenExpiry = 0;
    this.flushing = false;
    this.consecutiveFailures = 0;
    this.droppedRows = 0;
    this.exportedRows = 0;

    if (this.enabled) {
      this.timer = setInterval(() => {
        this.flush().catch((err) => this.logger.error("[sheets] flush failed", err));
      }, this.flushIntervalMs);
      if (this.timer.unref) this.timer.unref();
    }
  }

  /** Fire-and-forget. Never awaited by a request handler. */
  enqueue(row) {
    if (!this.enabled) return;
    // Bound the queue so a long outage cannot grow memory without limit.
    if (this.queue.length >= 10_000) {
      this.droppedRows += 1;
      if (this.droppedRows === 1) {
        this.logger.error(
          "[sheets] queue full (10000 rows) — dropping. Attendance is still " +
            "recorded server-side; export via GET /api/session/:id/export."
        );
      }
      return;
    }
    this.queue.push(row);
  }

  async #getAccessToken() {
    const now = Math.floor(Date.now() / 1000);
    // 60s safety margin so a token never expires mid-flight.
    if (this.accessToken && now < this.accessTokenExpiry - 60) {
      return this.accessToken;
    }

    const header = base64url(JSON.stringify({ alg: "RS256", typ: "JWT" }));
    const claims = base64url(
      JSON.stringify({
        iss: this.clientEmail,
        scope: SCOPE,
        aud: TOKEN_ENDPOINT,
        iat: now,
        exp: now + 3600,
      })
    );

    const signer = crypto.createSign("RSA-SHA256");
    signer.update(`${header}.${claims}`);
    const signature = signer.sign(this.privateKey, "base64url");
    const assertion = `${header}.${claims}.${signature}`;

    const response = await fetch(TOKEN_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
        assertion,
      }),
      signal: AbortSignal.timeout(10_000),
    });

    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`token exchange failed (${response.status}): ${detail.slice(0, 200)}`);
    }

    const data = await response.json();
    this.accessToken = data.access_token;
    this.accessTokenExpiry = now + (data.expires_in || 3600);
    return this.accessToken;
  }

  async flush() {
    if (!this.enabled || this.flushing || this.queue.length === 0) return;

    this.flushing = true;
    const batch = this.queue.splice(0, MAX_BATCH_ROWS);

    try {
      const token = await this.#getAccessToken();
      const url =
        `https://sheets.googleapis.com/v4/spreadsheets/${encodeURIComponent(this.spreadsheetId)}` +
        `/values/${encodeURIComponent(this.range)}:append` +
        `?valueInputOption=RAW&insertDataOption=INSERT_ROWS`;

      const response = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ values: batch }),
        signal: AbortSignal.timeout(15_000),
      });

      if (!response.ok) {
        const detail = await response.text().catch(() => "");
        throw new Error(`append failed (${response.status}): ${detail.slice(0, 200)}`);
      }

      this.exportedRows += batch.length;
      this.consecutiveFailures = 0;
    } catch (error) {
      this.consecutiveFailures += 1;
      // Put the rows back at the front so ordering survives a transient failure.
      this.queue.unshift(...batch);

      if (this.consecutiveFailures <= MAX_RETRIES) {
        this.logger.warn(
          `[sheets] export attempt ${this.consecutiveFailures} failed, will retry: ${error.message}`
        );
      } else if (this.consecutiveFailures === MAX_RETRIES + 1) {
        this.logger.error(
          `[sheets] export failing persistently: ${error.message}. Attendance is ` +
            `safe server-side; retrying quietly in the background.`
        );
      }
      // Token may be the problem — force a refresh next attempt.
      this.accessToken = null;
    } finally {
      this.flushing = false;
    }
  }

  status() {
    return {
      enabled: this.enabled,
      queued: this.queue.length,
      exported: this.exportedRows,
      dropped: this.droppedRows,
      consecutiveFailures: this.consecutiveFailures,
      healthy: this.consecutiveFailures <= MAX_RETRIES,
    };
  }

  async close() {
    if (this.timer) clearInterval(this.timer);
    if (this.enabled) await this.flush().catch(() => {});
  }
}

module.exports = { SheetsExporter };
