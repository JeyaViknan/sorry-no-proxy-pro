"use strict";

/**
 * Attendance export via a Google Apps Script web app.
 *
 * WHY THIS EXISTS ALONGSIDE sheets.js
 * -----------------------------------
 * The service-account path (sheets.js) needs a Google Cloud project, an
 * enabled Sheets API, a service account and a JSON key. That is a lot of
 * setup, and Google increasingly prompts for a billing account during it.
 *
 * Apps Script needs none of that: it runs inside the spreadsheet's own Google
 * account. You paste ~20 lines into the sheet's script editor, deploy it as a
 * web app, and you have a URL. No cloud project, no service account, no
 * billing, no key material beyond one shared secret.
 *
 * For a college project this is strictly better. sheets.js remains for
 * institutional deployments that already have Google Cloud.
 *
 * SECURITY: the endpoint is a public URL (Apps Script offers no other option
 * for programmatic access), so every request carries a shared secret which
 * the script checks before writing. Without it, anyone who learned the URL
 * could append arbitrary rows to the attendance sheet. The secret never
 * reaches a browser — only this server sends it.
 */

const MAX_BATCH_ROWS = 200;
const MAX_RETRIES = 5;
const REQUEST_TIMEOUT_MS = 15_000;

class SheetsWebhookExporter {
  constructor({ enabled, webhookUrl, webhookSecret, flushIntervalMs }, logger) {
    this.enabled = enabled;
    this.webhookUrl = webhookUrl;
    this.webhookSecret = webhookSecret;
    this.flushIntervalMs = flushIntervalMs;
    this.logger = logger;

    /** @type {Array<Array<string|number>>} */
    this.queue = [];
    this.flushing = false;
    this.consecutiveFailures = 0;
    this.droppedRows = 0;
    this.exportedRows = 0;

    if (this.enabled) {
      this.timer = setInterval(() => {
        this.flush().catch((err) => this.logger.error("[sheets-webhook] flush failed", err));
      }, this.flushIntervalMs);
      if (this.timer.unref) this.timer.unref();
    }
  }

  /** Fire-and-forget. Never awaited by a request handler. */
  enqueue(row) {
    if (!this.enabled) return;
    if (this.queue.length >= 10_000) {
      this.droppedRows += 1;
      if (this.droppedRows === 1) {
        this.logger.error(
          "[sheets-webhook] queue full (10000 rows) — dropping. Attendance is " +
            "still recorded server-side; export via /api/sessions/:id/export."
        );
      }
      return;
    }
    this.queue.push(row);
  }

  async flush() {
    if (!this.enabled || this.flushing || this.queue.length === 0) return;

    this.flushing = true;
    const batch = this.queue.splice(0, MAX_BATCH_ROWS);

    try {
      const response = await fetch(this.webhookUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Apps Script web apps answer a 302 to script.googleusercontent.com;
        // without following it the body never arrives.
        redirect: "follow",
        body: JSON.stringify({ secret: this.webhookSecret, rows: batch }),
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });

      const text = await response.text();

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${text.slice(0, 200)}`);
      }
      // Apps Script returns 200 with an HTML error page when the script itself
      // throws, so a status check alone is not enough.
      if (!text.includes('"ok":true')) {
        throw new Error(
          text.toLowerCase().includes("<!doctype")
            ? "the script returned an HTML error page — check the Apps Script " +
              "deployment is set to 'Anyone' access and redeployed after edits"
            : `unexpected response: ${text.slice(0, 200)}`
        );
      }

      this.exportedRows += batch.length;
      this.consecutiveFailures = 0;
    } catch (error) {
      this.consecutiveFailures += 1;
      this.queue.unshift(...batch); // preserve ordering across a transient failure

      if (this.consecutiveFailures <= MAX_RETRIES) {
        this.logger.warn(
          `[sheets-webhook] attempt ${this.consecutiveFailures} failed, will retry: ${error.message}`
        );
      } else if (this.consecutiveFailures === MAX_RETRIES + 1) {
        this.logger.error(
          `[sheets-webhook] failing persistently: ${error.message}. Attendance is ` +
            `safe server-side; retrying quietly in the background.`
        );
      }
    } finally {
      this.flushing = false;
    }
  }

  status() {
    return {
      enabled: this.enabled,
      mode: "apps-script-webhook",
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

/** No-op exporter, so callers never branch on whether export is configured. */
class NullExporter {
  constructor() {
    this.enabled = false;
  }
  enqueue() {}
  async flush() {}
  status() {
    return { enabled: false, mode: "disabled" };
  }
  async close() {}
}

module.exports = { SheetsWebhookExporter, NullExporter };
