/**
 * Camera lifecycle.
 *
 * Every fix in this file maps to a specific failure students actually hit.
 *
 * 1. iOS SAFARI SHOWED A BLACK BOX
 *    `<video autoplay playsinline>` without `muted` will not autoplay on iOS,
 *    even for a MediaStream. The element silently refused to start and the
 *    old code then captured from it anyway. We set `muted` before assigning
 *    the stream and call play() explicitly, surfacing the rejection instead
 *    of swallowing it.
 *
 * 2. "RANDOM FAILURES DEPENDING ON DEVICE" WAS A RACE
 *    The old handoff did `html5QrCode.stop()` (a promise, not awaited) then
 *    immediately requested the front camera. On many Android builds the rear
 *    camera is still held and the second getUserMedia rejects with
 *    NotReadableError. Whether it failed depended on teardown speed, which is
 *    why it looked device-specific and random. Teardown is now fully awaited,
 *    every track is explicitly stopped, and the hardware gets a settle delay.
 *
 * 3. WRONG CAMERA SELECTED
 *    The old code matched `label.includes("back")`. Labels are empty until
 *    permission is granted on Firefox/Safari and localised elsewhere
 *    ("Rückkamera", "背面カメラ"). The fallback `cameras[0]` is the FRONT
 *    camera on many Androids, so students were scanning the projector with
 *    their selfie camera. We use facingMode constraints and let the browser
 *    choose.
 *
 * 4. STALE FRAME AFTER BACKGROUNDING
 *    iOS suspends tracks when the tab backgrounds; on return the element
 *    holds a frozen frame. Nothing handled visibilitychange, so a capture
 *    could silently encode a frame from minutes earlier. We detect it and
 *    re-acquire.
 *
 * 5. BLANK-FRAME FALLBACK
 *    The old captureFaceImage() had `if (width === 0) { width = 320; }` under
 *    a "prevent crash" comment. It did not prevent a crash — it manufactured
 *    a black 320x240 JPEG, shipped it to the server, and produced an
 *    unexplainable rejection. Readiness is now a hard precondition.
 */

const HARDWARE_SETTLE_MS = 250;
const PLAY_TIMEOUT_MS = 8000;
const READY_TIMEOUT_MS = 8000;

export class CameraError extends Error {
  constructor(code, message, { recoverable = false, cause = null } = {}) {
    super(message);
    this.name = "CameraError";
    this.code = code;
    this.recoverable = recoverable;
    this.cause = cause;
  }
}

/** Map a getUserMedia rejection to something a student can act on. */
function classify(error) {
  switch (error?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return new CameraError(
        "PERMISSION_DENIED",
        "Camera access was blocked. Tap the camera icon in your browser's address bar to allow it, then try again.",
        { recoverable: true, cause: error }
      );
    case "NotFoundError":
    case "DevicesNotFoundError":
      return new CameraError("NO_CAMERA", "No camera was found on this device.", {
        cause: error,
      });
    case "NotReadableError":
    case "TrackStartError":
      return new CameraError(
        "CAMERA_BUSY",
        "The camera is in use by another app. Close other camera apps and try again.",
        { recoverable: true, cause: error }
      );
    case "OverconstrainedError":
      return new CameraError("UNSUPPORTED_MODE", "This camera does not support the requested mode.", {
        recoverable: true,
        cause: error,
      });
    case "AbortError":
      return new CameraError("CAMERA_ABORTED", "The camera stopped unexpectedly.", {
        recoverable: true,
        cause: error,
      });
    default:
      return new CameraError(
        "CAMERA_FAILED",
        "The camera could not be started. Try reloading the page.",
        { recoverable: true, cause: error }
      );
  }
}

/** Secure context is a hard requirement for getUserMedia. */
export function checkSupport() {
  if (!window.isSecureContext) {
    return {
      supported: false,
      code: "INSECURE_CONTEXT",
      message: "This page must be opened over HTTPS for the camera to work.",
    };
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return {
      supported: false,
      code: "NO_MEDIA_DEVICES",
      message:
        "This browser cannot access the camera. Try Chrome or Safari, and make sure you are not in a private/in-app browser.",
    };
  }
  return { supported: true };
}

function stopStream(stream) {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    try {
      track.stop();
    } catch {
      /* already stopped */
    }
  }
}

/**
 * Wait until the element genuinely has pixels. `readyState >= HAVE_CURRENT_DATA`
 * plus non-zero dimensions is the condition the old code never checked.
 */
function waitForFrames(video, timeoutMs = READY_TIMEOUT_MS) {
  if (video.readyState >= 2 && video.videoWidth > 0) return Promise.resolve();

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      reject(new CameraError("CAMERA_NO_FRAMES", "The camera did not produce a picture.", {
        recoverable: true,
      }));
    }, timeoutMs);

    const check = () => {
      if (video.readyState >= 2 && video.videoWidth > 0) {
        cleanup();
        resolve();
      }
    };
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener("loadeddata", check);
      video.removeEventListener("canplay", check);
      video.removeEventListener("playing", check);
    };

    video.addEventListener("loadeddata", check);
    video.addEventListener("canplay", check);
    video.addEventListener("playing", check);
    check();
  });
}

/**
 * Manages exactly one active stream. Constructing a second Camera for the
 * other facing mode is not how this is used — call `switchTo()` so teardown
 * is guaranteed to complete before acquisition begins.
 */
export class Camera {
  constructor(videoElement) {
    this.video = videoElement;
    this.stream = null;
    this.facingMode = null;
    this.onInterrupted = null;
    this.#prepareElement();
  }

  #prepareElement() {
    const video = this.video;
    // All four are required for reliable inline autoplay across iOS Safari,
    // Chrome Android and Samsung Internet. `muted` is the one whose absence
    // broke iPhones.
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    video.setAttribute("playsinline", "");
    video.setAttribute("webkit-playsinline", "");
    video.setAttribute("muted", "");
    video.setAttribute("autoplay", "");
    video.disablePictureInPicture = true;
  }

  get isLive() {
    return Boolean(this.stream?.getVideoTracks().some((t) => t.readyState === "live"));
  }

  get dimensions() {
    return { width: this.video.videoWidth || 0, height: this.video.videoHeight || 0 };
  }

  /**
   * Acquire a camera. Must be called from a user gesture on first use — iOS
   * and Firefox Mobile deny or mis-handle an unprompted request, which is why
   * the old `window.addEventListener("load", initializeScanner)` was unreliable.
   */
  async start(facingMode = "environment", { width = 1280, height = 720 } = {}) {
    const support = checkSupport();
    if (!support.supported) {
      throw new CameraError(support.code, support.message);
    }

    await this.stop();

    // `ideal`, not `exact`. With `exact`, a tablet with only one camera
    // rejects outright; with `ideal` the browser gives us the closest match,
    // which is always better than no camera at all.
    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: width },
        height: { ideal: height },
        frameRate: { ideal: 30, max: 30 },
      },
    };

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      // A device that cannot satisfy the resolution hints should still work.
      if (error?.name === "OverconstrainedError") {
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: { facingMode: { ideal: facingMode } },
          });
        } catch (retryError) {
          throw classify(retryError);
        }
      } else {
        throw classify(error);
      }
    }

    this.stream = stream;
    this.facingMode = facingMode;
    this.video.srcObject = stream;

    // Surface a stream that dies underneath us (unplugged, revoked, OS took
    // the camera) rather than showing a frozen frame forever.
    for (const track of stream.getVideoTracks()) {
      track.addEventListener("ended", () => {
        if (this.stream === stream) this.onInterrupted?.("TRACK_ENDED");
      });
    }

    try {
      await Promise.race([
        this.video.play(),
        new Promise((_, reject) =>
          setTimeout(
            () => reject(new CameraError("PLAY_TIMEOUT", "The camera preview did not start.", {
              recoverable: true,
            })),
            PLAY_TIMEOUT_MS
          )
        ),
      ]);
    } catch (error) {
      if (error instanceof CameraError) {
        await this.stop();
        throw error;
      }
      await this.stop();
      throw new CameraError(
        "AUTOPLAY_BLOCKED",
        "The camera preview could not start. Tap the screen and try again.",
        { recoverable: true, cause: error }
      );
    }

    await waitForFrames(this.video);
    return this;
  }

  /**
   * Switch facing mode with a fully-awaited teardown.
   * This is the fix for the rear-to-front handoff race.
   */
  async switchTo(facingMode, options) {
    await this.stop();
    // Give the OS a moment to actually release the sensor. Without this,
    // Samsung Internet and several Android builds reject the next
    // getUserMedia with NotReadableError.
    await new Promise((resolve) => setTimeout(resolve, HARDWARE_SETTLE_MS));

    try {
      return await this.start(facingMode, options);
    } catch (error) {
      // One retry specifically for the busy case — it is usually a slow
      // release rather than a genuine conflict.
      if (error.code === "CAMERA_BUSY") {
        await new Promise((resolve) => setTimeout(resolve, 700));
        return this.start(facingMode, options);
      }
      throw error;
    }
  }

  async stop() {
    stopStream(this.stream);
    this.stream = null;
    this.facingMode = null;

    if (this.video.srcObject) {
      this.video.srcObject = null;
      // Some WebKit builds keep the sensor warm until the element is reset.
      try {
        this.video.load();
      } catch {
        /* non-fatal */
      }
    }
  }

  /**
   * Draw the current frame. Throws rather than inventing a blank canvas —
   * a capture that cannot happen must be an error the caller handles, not a
   * black JPEG the server has to reject.
   */
  drawTo(canvas, { maxWidth = 0 } = {}) {
    const { width: sourceWidth, height: sourceHeight } = this.dimensions;

    if (!this.isLive) {
      throw new CameraError("CAMERA_NOT_LIVE", "The camera is not running.", { recoverable: true });
    }
    if (sourceWidth === 0 || sourceHeight === 0 || this.video.readyState < 2) {
      throw new CameraError("CAMERA_NOT_READY", "The camera is still starting.", {
        recoverable: true,
      });
    }

    let targetWidth = sourceWidth;
    let targetHeight = sourceHeight;
    if (maxWidth > 0 && sourceWidth > maxWidth) {
      targetHeight = Math.round(sourceHeight * (maxWidth / sourceWidth));
      targetWidth = maxWidth;
    }

    canvas.width = targetWidth;
    canvas.height = targetHeight;

    const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
    context.drawImage(this.video, 0, 0, targetWidth, targetHeight);
    return { width: targetWidth, height: targetHeight, context };
  }
}

/**
 * Re-acquire the stream when the page returns to the foreground.
 *
 * iOS suspends MediaStream tracks on background. On return the video element
 * holds the last frame, which looks live but is not — and the old code would
 * happily capture from it.
 */
export function watchVisibility(camera, { onRecover, onLost } = {}) {
  const handler = async () => {
    if (document.visibilityState === "hidden") return;
    if (!camera.facingMode) return;

    // A live track plus flowing frames means nothing was lost.
    if (camera.isLive && camera.video.readyState >= 2) {
      try {
        await camera.video.play();
        return;
      } catch {
        /* fall through to re-acquire */
      }
    }

    onLost?.();
    try {
      await camera.switchTo(camera.facingMode || "user");
      onRecover?.();
    } catch (error) {
      onLost?.(error);
    }
  };

  document.addEventListener("visibilitychange", handler);
  window.addEventListener("pageshow", handler);

  return () => {
    document.removeEventListener("visibilitychange", handler);
    window.removeEventListener("pageshow", handler);
  };
}
