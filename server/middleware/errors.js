"use strict";

/**
 * Error handling and the shared response shape.
 *
 * Every API response — success or failure — carries a stable machine-readable
 * `code`. The client maps codes to messages and recovery actions, so error
 * text can be rewritten without breaking behaviour, and the student never
 * sees a raw exception.
 *
 * The old server leaked internals in two ways this replaces:
 *   • `"Face does not match registration number (best match: 25BCE1431)"`
 *     turned an unauthenticated endpoint into a biometric identification
 *     oracle — upload any photo, learn whose face it is.
 *   • Stack traces and Python stderr reached the client on 500s.
 */

const { logger } = require("../logger");

class ApiError extends Error {
  constructor(status, code, message, details = undefined) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
    this.expose = true;
  }

  static badRequest(code, message, details) {
    return new ApiError(400, code, message, details);
  }
  static unauthorized(code, message) {
    return new ApiError(401, code, message);
  }
  static forbidden(code, message) {
    return new ApiError(403, code, message);
  }
  static notFound(code, message) {
    return new ApiError(404, code, message);
  }
  static conflict(code, message) {
    return new ApiError(409, code, message);
  }
  static tooLarge(code, message) {
    return new ApiError(413, code, message);
  }
  static unavailable(code, message) {
    return new ApiError(503, code, message);
  }
}

function notFoundHandler(req, res) {
  res.status(404).json({
    ok: false,
    code: "NOT_FOUND",
    message: "No such endpoint.",
  });
}

// eslint-disable-next-line no-unused-vars -- Express identifies error middleware by arity
function errorHandler(err, req, res, next) {
  if (err instanceof ApiError) {
    logger.debug("request rejected", {
      path: req.path,
      code: err.code,
      status: err.status,
    });
    return res.status(err.status).json({
      ok: false,
      code: err.code,
      message: err.message,
      ...(err.details ? { details: err.details } : {}),
    });
  }

  // Body parser rejections arrive as generic errors.
  if (err.type === "entity.too.large") {
    return res.status(413).json({
      ok: false,
      code: "PAYLOAD_TOO_LARGE",
      message: "The submitted images are too large. Try again in better light.",
    });
  }
  if (err.type === "entity.parse.failed") {
    return res.status(400).json({
      ok: false,
      code: "MALFORMED_JSON",
      message: "Malformed request.",
    });
  }

  // Anything unrecognised is a bug. Log it fully; tell the client nothing.
  logger.error("unhandled error", err);
  return res.status(500).json({
    ok: false,
    code: "INTERNAL_ERROR",
    message: "Something went wrong on our side. Please try again.",
  });
}

module.exports = { ApiError, errorHandler, notFoundHandler };
