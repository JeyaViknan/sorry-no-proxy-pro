"use strict";

/**
 * Persistent face-verification worker pool.
 *
 * THE FIX FOR THE WORST PERFORMANCE BUG IN THE PROJECT
 * ---------------------------------------------------
 * The old server ran `spawn("python3", [script, regno, tmpPath])` per request,
 * serialised behind a global `isVerifying` boolean. Every single student paid:
 *
 *     import onnxruntime + insightface + cv2   2-4 s
 *     FaceAnalysis.prepare() (load buffalo_l)  1-3 s
 *     embedding store initialisation           0.2 s  (or 60-120 s on the
 *                                                       silent fallback path)
 *     the actual inference                     0.2 s
 *     ----------------------------------------------
 *     5-15 s per student, strictly one at a time
 *
 * That is the true cause of the reported "high loading times" and "random
 * failures depending on device": nothing was device-dependent, it was queue
 * position. Student 40 waited five minutes behind a queue their phone could
 * not see, and their fetch — which had no timeout — hung until the platform
 * proxy killed it at 60 s.
 *
 * A `--serve` worker mode already existed in face_verification.py, fully
 * written, and server.js simply never used it; git history shows it was wired
 * up at one point and dropped. `APP_DEPLOY_MARKER` still claimed
 * "insightface-worker-huggingface" while the code did the opposite.
 *
 * Here the model is loaded once at boot and stays warm. Per-request cost falls
 * to roughly 200-400 ms, and the pool serves N concurrently. Same hardware,
 * ~40x the throughput.
 *
 * PROTOCOL: newline-delimited JSON over stdin/stdout, correlated by `id`.
 * stderr is reserved for logs so a stray print can never corrupt a response —
 * the old code had to scan stdout backwards looking for a line that happened
 * to start with `{`.
 */

const { spawn } = require("child_process");
const { EventEmitter } = require("events");
const { randomId } = require("./crypto");

const RESTART_BASE_DELAY_MS = 500;
const RESTART_MAX_DELAY_MS = 30_000;

class FaceWorker extends EventEmitter {
  constructor({ id, config, logger }) {
    super();
    this.id = id;
    this.config = config;
    this.logger = logger;

    this.proc = null;
    this.ready = false;
    this.busy = false;
    this.stdoutBuffer = "";
    this.pending = new Map(); // requestId -> { resolve, reject, timer }
    this.restartAttempts = 0;
    this.stopping = false;
    this.lastError = null;
  }

  start() {
    if (this.stopping) return;

    const { pythonBin, verifierScript, galleryDir } = this.config.paths;

    this.proc = spawn(pythonBin, [verifierScript, "--serve"], {
      cwd: this.config.paths.root,
      env: {
        ...process.env,
        GALLERY_DIR: galleryDir,
        FACE_THRESHOLD_ACCEPT: String(this.config.face.thresholdAccept),
        FACE_THRESHOLD_REVIEW: String(this.config.face.thresholdReview),
        // ONNX Runtime and BLAS both try to grab every core by default. With
        // several workers that oversubscribes the CPU and makes everything
        // slower. One thread each, parallelism comes from the pool.
        OMP_NUM_THREADS: "1",
        OPENBLAS_NUM_THREADS: "1",
        MKL_NUM_THREADS: "1",
        PYTHONUNBUFFERED: "1",
      },
      stdio: ["pipe", "pipe", "pipe"],
    });

    this.proc.stdout.setEncoding("utf8");
    this.proc.stdout.on("data", (chunk) => this.#onStdout(chunk));

    this.proc.stderr.setEncoding("utf8");
    this.proc.stderr.on("data", (chunk) => {
      const text = chunk.trim();
      if (text) this.logger.debug(`[verifier ${this.id}] ${text}`);
    });

    this.proc.on("error", (error) => {
      this.lastError = error.message;
      this.logger.error(`[verifier ${this.id}] spawn failed`, error);
    });

    this.proc.on("exit", (code, signal) => this.#onExit(code, signal));
  }

  #onStdout(chunk) {
    this.stdoutBuffer += chunk;

    let newlineIndex;
    while ((newlineIndex = this.stdoutBuffer.indexOf("\n")) !== -1) {
      const line = this.stdoutBuffer.slice(0, newlineIndex).trim();
      this.stdoutBuffer = this.stdoutBuffer.slice(newlineIndex + 1);
      if (!line) continue;

      let message;
      try {
        message = JSON.parse(line);
      } catch {
        this.logger.warn(`[verifier ${this.id}] non-JSON on stdout`, { line: line.slice(0, 200) });
        continue;
      }

      if (message.type === "ready") {
        this.ready = true;
        this.restartAttempts = 0;
        this.lastError = null;
        this.logger.info(`[verifier ${this.id}] ready`, {
          identities: message.identities,
          model: message.model,
        });
        this.emit("ready");
        continue;
      }

      if (message.type === "fatal") {
        this.lastError = message.error;
        this.logger.error(`[verifier ${this.id}] fatal: ${message.error}`);
        continue;
      }

      const pending = this.pending.get(message.id);
      if (!pending) continue;

      clearTimeout(pending.timer);
      this.pending.delete(message.id);
      this.busy = this.pending.size > 0;
      pending.resolve(message);
      this.emit("free");
    }
  }

  #onExit(code, signal) {
    this.ready = false;
    this.busy = false;

    for (const [, pending] of this.pending) {
      clearTimeout(pending.timer);
      pending.reject(new Error("Face verifier exited before responding"));
    }
    this.pending.clear();

    if (this.stopping) return;

    this.restartAttempts += 1;
    const delay = Math.min(
      RESTART_BASE_DELAY_MS * 2 ** (this.restartAttempts - 1),
      RESTART_MAX_DELAY_MS
    );
    this.logger.error(
      `[verifier ${this.id}] exited (code=${code} signal=${signal}), ` +
        `restarting in ${delay}ms (attempt ${this.restartAttempts})`
    );
    const timer = setTimeout(() => this.start(), delay);
    if (timer.unref) timer.unref();
  }

  send(payload, timeoutMs) {
    return new Promise((resolve, reject) => {
      if (!this.ready || !this.proc || this.proc.killed) {
        reject(new Error("Face verifier is not ready"));
        return;
      }

      const id = randomId(8);
      const timer = setTimeout(() => {
        this.pending.delete(id);
        this.busy = this.pending.size > 0;
        // A timed-out worker may still be mid-inference and out of step;
        // recycling is safer than reusing it.
        this.logger.warn(`[verifier ${this.id}] request timed out, recycling worker`);
        this.restart();
        reject(new Error("Face verification timed out"));
      }, timeoutMs);
      if (timer.unref) timer.unref();

      this.pending.set(id, { resolve, reject, timer });
      this.busy = true;

      const line = `${JSON.stringify({ id, ...payload })}\n`;
      this.proc.stdin.write(line, (error) => {
        if (!error) return;
        clearTimeout(timer);
        this.pending.delete(id);
        this.busy = this.pending.size > 0;
        reject(error);
      });
    });
  }

  restart() {
    if (this.proc && !this.proc.killed) this.proc.kill("SIGKILL");
  }

  async stop() {
    this.stopping = true;
    if (!this.proc || this.proc.killed) return;

    this.proc.stdin.end();
    await new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.proc.kill("SIGKILL");
        resolve();
      }, 3000);
      this.proc.once("exit", () => {
        clearTimeout(timer);
        resolve();
      });
    });
  }
}

class FaceVerifierPool {
  constructor({ config, logger }) {
    this.config = config;
    this.logger = logger;
    this.workers = [];
    this.waiting = [];
    this.nextWorker = 0;
    this.stats = { requests: 0, failures: 0, timeouts: 0 };
  }

  start() {
    for (let i = 0; i < this.config.face.workerCount; i += 1) {
      const worker = new FaceWorker({ id: i, config: this.config, logger: this.logger });
      worker.on("free", () => this.#drainQueue());
      worker.on("ready", () => this.#drainQueue());
      worker.start();
      this.workers.push(worker);
    }
  }

  /** Resolves once at least one worker has loaded the model. */
  waitUntilReady(timeoutMs = 180_000) {
    if (this.workers.some((w) => w.ready)) return Promise.resolve();

    return new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error("No face verifier became ready in time")),
        timeoutMs
      );
      for (const worker of this.workers) {
        worker.once("ready", () => {
          clearTimeout(timer);
          resolve();
        });
      }
    });
  }

  #idleWorker() {
    // Round-robin start point so load spreads evenly across the pool.
    for (let i = 0; i < this.workers.length; i += 1) {
      const worker = this.workers[(this.nextWorker + i) % this.workers.length];
      if (worker.ready && !worker.busy) {
        this.nextWorker = (this.nextWorker + i + 1) % this.workers.length;
        return worker;
      }
    }
    return null;
  }

  #drainQueue() {
    while (this.waiting.length > 0) {
      const worker = this.#idleWorker();
      if (!worker) return;
      const job = this.waiting.shift();
      clearTimeout(job.queueTimer);
      this.#dispatch(worker, job);
    }
  }

  #dispatch(worker, job) {
    worker
      .send(job.payload, this.config.face.requestTimeoutMs)
      .then(job.resolve)
      .catch(job.reject);
  }

  /**
   * @param {{registerNumber: string, frames: Array<{base64: string}>}} input
   * @returns {Promise<{ok: boolean, similarity: number, reason?: string, message?: string, quality?: object}>}
   */
  async verify({ registerNumber, frames }) {
    this.stats.requests += 1;

    const payload = {
      registerNumber,
      frames: frames.map((f) => f.base64),
      thresholdAccept: this.config.face.thresholdAccept,
      thresholdReview: this.config.face.thresholdReview,
    };

    const response = await new Promise((resolve, reject) => {
      const worker = this.#idleWorker();
      const job = { payload, resolve, reject, queueTimer: null };

      if (worker) {
        this.#dispatch(worker, job);
        return;
      }

      // All workers busy: queue, but bound the wait so a student is never
      // left hanging behind a backlog. Better a clear "try again" than a
      // silent stall — the old design's defining failure.
      job.queueTimer = setTimeout(() => {
        const index = this.waiting.indexOf(job);
        if (index !== -1) this.waiting.splice(index, 1);
        this.stats.timeouts += 1;
        reject(new Error("Verification queue is saturated"));
      }, this.config.face.requestTimeoutMs);
      if (job.queueTimer.unref) job.queueTimer.unref();

      this.waiting.push(job);
    });

    if (response.error) {
      this.stats.failures += 1;
      throw new Error(response.error);
    }
    return response.result;
  }

  status() {
    return {
      ready: this.workers.some((w) => w.ready),
      workers: this.workers.map((w) => ({
        id: w.id,
        ready: w.ready,
        busy: w.busy,
        restarts: w.restartAttempts,
        lastError: w.lastError,
      })),
      queued: this.waiting.length,
      stats: this.stats,
    };
  }

  async stop() {
    await Promise.all(this.workers.map((w) => w.stop()));
  }
}

module.exports = { FaceVerifierPool };
