"use strict";

/**
 * Minimal structured logger — no dependency, JSON in production so a log
 * aggregator can parse it, human-readable in development.
 *
 * The audit found the only record of a verification decision was ephemeral
 * stderr on Hugging Face Spaces, which made "I was there and it rejected me"
 * impossible to investigate. Verification outcomes are logged here at info
 * level with a stable shape so they survive into whatever log sink the
 * deployment uses.
 */

const LEVELS = { debug: 10, info: 20, warn: 30, error: 40 };

const configuredLevel = (process.env.LOG_LEVEL || "info").toLowerCase();
const threshold = LEVELS[configuredLevel] ?? LEVELS.info;
const useJson = process.env.NODE_ENV === "production";

/** Values that must never reach a log line. */
const REDACT_KEYS = new Set([
  "faceImage",
  "frames",
  "attendanceToken",
  "facultyToken",
  "accessCode",
  "privateKey",
  "authorization",
]);

function redact(value, depth = 0) {
  if (depth > 4 || value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return `[${value.length} items]`;

  const output = {};
  for (const [key, item] of Object.entries(value)) {
    output[key] = REDACT_KEYS.has(key) ? "[redacted]" : redact(item, depth + 1);
  }
  return output;
}

function emit(level, message, context) {
  if (LEVELS[level] < threshold) return;

  const stream = LEVELS[level] >= LEVELS.warn ? process.stderr : process.stdout;

  if (useJson) {
    stream.write(
      `${JSON.stringify({
        ts: new Date().toISOString(),
        level,
        msg: message,
        ...(context ? { ctx: redact(context) } : {}),
      })}\n`
    );
    return;
  }

  const time = new Date().toISOString().slice(11, 23);
  const tail = context ? ` ${JSON.stringify(redact(context))}` : "";
  stream.write(`${time} ${level.toUpperCase().padEnd(5)} ${message}${tail}\n`);
}

const logger = {
  debug: (message, context) => emit("debug", message, context),
  info: (message, context) => emit("info", message, context),
  warn: (message, context) => emit("warn", message, context),
  error: (message, context) => {
    if (context instanceof Error) {
      emit("error", message, { error: context.message, stack: context.stack });
    } else {
      emit("error", message, context);
    }
  },
};

module.exports = { logger };
