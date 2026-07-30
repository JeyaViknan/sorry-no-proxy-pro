"use strict";

/**
 * Process entry point: validate config, start workers, bind the port,
 * shut down cleanly.
 *
 * Ordering matters. The old server called app.listen() before registering
 * /healthz and /, and started accepting traffic with no idea whether the
 * verifier worked. Here the port opens immediately (so the platform's health
 * probe succeeds and the container is not killed during model load) but
 * /readyz reports 503 until a worker is warm, which is the signal an
 * orchestrator should use to hold traffic back.
 */

const { config, validateOrExit } = require("./config");
const { logger } = require("./logger");
const { createApp } = require("./app");
const { FaceVerifierPool } = require("./services/faceWorker");
const { GoogleAuth } = require("./services/googleAuth");
const { ensureGallery } = require("./services/galleryBootstrap");

validateOrExit(logger);

const googleAuth = new GoogleAuth(config.google, logger);
const faceVerifier = new FaceVerifierPool({ config, logger });
const app = createApp({ config, faceVerifier, googleAuth });

const server = app.listen(config.port, "0.0.0.0", () => {
  logger.info("server listening", {
    port: config.port,
    env: config.nodeEnv,
    workers: config.face.workerCount,
    sheets: config.sheets.enabled ? "enabled" : "disabled",
    origins: config.cors.allowedOrigins,
  });
});

// Classroom wifi produces plenty of half-open connections. Keep-alive slightly
// above the typical proxy idle timeout avoids races where the proxy reuses a
// socket the server is closing.
server.keepAliveTimeout = 72_000;
server.headersTimeout = 75_000;

/**
 * Startup order matters.
 *
 * The port is already open (above) so the platform's health probe succeeds and
 * the container is not killed while the model loads — but /readyz stays 503
 * until a worker is warm, which is the signal an orchestrator should gate
 * traffic on.
 *
 * The gallery must land on disk BEFORE the workers spawn: a worker that
 * starts without one exits and enters restart backoff, turning a 3-second
 * download into a minute of flapping.
 */
(async () => {
  try {
    const gallery = await ensureGallery({ config, auth: googleAuth, logger });
    logger.info("gallery ready", gallery);
  } catch (error) {
    logger.error(
      "FATAL: could not obtain the face gallery. The service cannot verify " +
        "anyone. See docs/DEPLOYMENT.md#gallery.",
      error
    );
    // Exit rather than serve: a running instance that rejects every student
    // is worse than one the orchestrator will restart and alert on.
    process.exit(1);
  }

  faceVerifier.start();
  faceVerifier
    .waitUntilReady()
    .then(() => logger.info("face verifier warm — accepting verifications"))
    .catch((error) => {
      logger.error(
        "face verifier did not become ready. Verification requests will fail " +
          "with VERIFIER_UNAVAILABLE until a worker recovers.",
        error
      );
    });
})();

// ── Graceful shutdown ───────────────────────────────────────────────
let shuttingDown = false;

async function shutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;
  logger.info(`received ${signal}, shutting down`);

  const force = setTimeout(() => {
    logger.error("shutdown timed out, forcing exit");
    process.exit(1);
  }, 15_000);
  if (force.unref) force.unref();

  server.close();
  try {
    // Flush queued attendance rows before the process disappears.
    await app.locals.services.sheets.close();
    app.locals.services.store.close();
    await faceVerifier.stop();
  } catch (error) {
    logger.error("error during shutdown", error);
  }

  clearTimeout(force);
  process.exit(0);
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));

process.on("unhandledRejection", (reason) => {
  logger.error("unhandled promise rejection", reason instanceof Error ? reason : { reason });
});
process.on("uncaughtException", (error) => {
  logger.error("uncaught exception — exiting", error);
  shutdown("uncaughtException");
});
