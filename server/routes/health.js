"use strict";

/**
 * Health checks that can actually fail.
 *
 * The old /healthz hardcoded `verifierReady: true, verifierLastError: null`.
 * A health check that structurally cannot report unhealthy is worse than
 * none, because the platform's restart and load-balancing logic trusts it and
 * keeps routing traffic to a broken instance.
 *
 * Two endpoints, because they answer different questions:
 *   /healthz  — liveness: is the process up? Cheap, always 200 if serving.
 *   /readyz   — readiness: can it serve a verification right now? 503 while
 *               the model is still warming, so an orchestrator holds traffic
 *               back instead of failing the first students of the class.
 */

const express = require("express");

function createHealthRouter({ faceVerifier, sheets, store, startedAt }) {
  const router = express.Router();

  router.get("/healthz", (req, res) => {
    res.json({
      ok: true,
      uptimeSec: Math.round((Date.now() - startedAt) / 1000),
    });
  });

  router.get("/readyz", (req, res) => {
    const verifier = faceVerifier.status();
    const ready = verifier.ready;

    res.status(ready ? 200 : 503).json({
      ok: ready,
      ready,
      verifier,
      sheets: sheets.status(),
      store: store.stats(),
      uptimeSec: Math.round((Date.now() - startedAt) / 1000),
    });
  });

  // Small, unauthenticated surface the scanner uses to warm the connection
  // and learn the server clock before a student starts scanning. Prefetching
  // this while the camera initialises removes a round trip from the critical
  // path, and lets the client detect a badly-set device clock early.
  router.get("/api/hello", (req, res) => {
    res.set("Cache-Control", "no-store");
    res.json({
      ok: true,
      serverTime: Date.now(),
      ready: faceVerifier.status().ready,
    });
  });

  return router;
}

module.exports = { createHealthRouter };
