/**
 * QR decoding from a camera we already own.
 *
 * ARCHITECTURAL CHANGE FROM THE OLD CODE
 * --------------------------------------
 * html5-qrcode used to create and own its own <video> element and
 * MediaStream. The face step then opened a *second* stream from a different
 * owner, and the two fought over the hardware — that is the root of the
 * rear-to-front handoff race that produced NotReadableError on Samsung
 * Internet and much of Android.
 *
 * Here there is exactly one camera owner (camera.js). This module only reads
 * frames from the video element it is handed, so switching from the rear to
 * the front camera is an ordinary awaited call with no cross-library
 * contention.
 *
 * TWO BACKENDS
 * ------------
 *   • BarcodeDetector — native, hardware-accelerated, zero download.
 *     Available on Chrome/Edge Android, which is most students.
 *   • html5-qrcode    — lazily imported ONLY when the native API is absent
 *     (Safari/iOS, Firefox). Chrome users never pay the 375KB.
 *
 * The vendored bundle is 375KB — larger than the rest of the app combined —
 * so not loading it by default is one of the biggest payload wins available,
 * and it matters most on the slow connections this system has to tolerate.
 */

/** Native decode runs cheaply; the fallback re-encodes, so it runs slower. */
const NATIVE_INTERVAL_MS = 120;
const FALLBACK_INTERVAL_MS = 260;

/** A projected QR needs nothing like full resolution to decode reliably. */
const DECODE_WIDTH = 540;

const decodeCanvas = document.createElement("canvas");
const decodeContext = decodeCanvas.getContext("2d", { alpha: false, willReadFrequently: true });

let fallbackLoader = null;

/** Load the heavy library once, on demand. */
function loadFallbackLibrary() {
  if (fallbackLoader) return fallbackLoader;

  fallbackLoader = new Promise((resolve, reject) => {
    if (window.__Html5QrcodeLibrary__?.Html5Qrcode) {
      resolve(window.__Html5QrcodeLibrary__.Html5Qrcode);
      return;
    }
    const script = document.createElement("script");
    script.src = "/vendor/html5-qrcode.min.js";
    script.async = true;
    script.onload = () => {
      const library = window.__Html5QrcodeLibrary__?.Html5Qrcode || window.Html5Qrcode;
      if (library) resolve(library);
      else reject(new Error("QR library loaded but did not register"));
    };
    script.onerror = () => reject(new Error("Could not load the QR scanner library"));
    document.head.appendChild(script);
  });

  return fallbackLoader;
}

async function createNativeDetector() {
  if (!("BarcodeDetector" in window)) return null;
  try {
    const formats = await window.BarcodeDetector.getSupportedFormats();
    if (!formats.includes("qr_code")) return null;
    return new window.BarcodeDetector({ formats: ["qr_code"] });
  } catch {
    return null;
  }
}

/** Downscale the current frame into the shared decode canvas. */
function drawForDecode(video) {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  if (!sourceWidth || !sourceHeight) return false;

  const scale = Math.min(1, DECODE_WIDTH / sourceWidth);
  decodeCanvas.width = Math.round(sourceWidth * scale);
  decodeCanvas.height = Math.round(sourceHeight * scale);
  decodeContext.drawImage(video, 0, 0, decodeCanvas.width, decodeCanvas.height);
  return true;
}

function canvasToFile() {
  return new Promise((resolve) => {
    decodeCanvas.toBlob(
      (blob) => resolve(blob ? new File([blob], "frame.jpg", { type: "image/jpeg" }) : null),
      "image/jpeg",
      0.75
    );
  });
}

/**
 * Kick off the fallback download early, on browsers that will need it.
 *
 * `prepare()` is awaited during camera startup, so on iOS the student was
 * watching "Starting camera…" for the length of a 368KB download on lecture
 * wifi. Calling this from the intro screen overlaps that with reading time
 * and a user's thumb travelling to the button. No-op where BarcodeDetector
 * exists, which is most of Android.
 */
export async function warmScannerBackend() {
  if (await createNativeDetector()) return "native";
  try {
    await loadFallbackLibrary();
    return "fallback";
  } catch {
    return "deferred"; // prepare() will retry and surface the error properly
  }
}

export class QrScanner {
  constructor() {
    this.running = false;
    this.backend = null;
    this.detector = null;
    this.fallback = null;
    this.timer = null;
    this.decoding = false;
  }

  /** Which backend is active — surfaced for diagnostics, not behaviour. */
  get backendName() {
    return this.backend || "none";
  }

  async prepare() {
    this.detector = await createNativeDetector();
    if (this.detector) {
      this.backend = "native";
      return this.backend;
    }

    const Html5Qrcode = await loadFallbackLibrary();
    // A detached host element: we only use the file-decoding path, never the
    // library's camera handling.
    const host = document.createElement("div");
    host.id = `qr-fallback-${Date.now()}`;
    host.style.display = "none";
    document.body.appendChild(host);

    this.fallbackHost = host;
    this.fallback = new Html5Qrcode(host.id, { verbose: false });
    this.backend = "fallback";
    return this.backend;
  }

  /**
   * Begin decoding frames from `video`.
   *
   * @param {HTMLVideoElement} video
   * @param {(text: string) => void} onDecode called for every decoded string,
   *        valid or not — validity is the server's decision, not ours.
   */
  async start(video, onDecode) {
    if (!this.backend) await this.prepare();
    this.running = true;

    const interval = this.backend === "native" ? NATIVE_INTERVAL_MS : FALLBACK_INTERVAL_MS;

    const tick = async () => {
      if (!this.running) return;

      // Never queue work behind a slow decode; drop the frame instead. This
      // is what keeps scanning responsive on low-end phones.
      if (this.decoding) {
        this.timer = setTimeout(tick, interval);
        return;
      }

      this.decoding = true;
      try {
        const text = await this.#decodeOnce(video);
        if (text && this.running) onDecode(text);
      } catch {
        /* an undecodable frame is the normal case, not an error */
      } finally {
        this.decoding = false;
      }

      if (this.running) this.timer = setTimeout(tick, interval);
    };

    tick();
  }

  async #decodeOnce(video) {
    if (video.readyState < 2 || !video.videoWidth) return null;

    if (this.backend === "native") {
      // BarcodeDetector reads the element directly — no canvas round trip.
      const results = await this.detector.detect(video);
      return results?.[0]?.rawValue || null;
    }

    if (!drawForDecode(video)) return null;
    const file = await canvasToFile();
    if (!file) return null;

    try {
      const result = await this.fallback.scanFileV2(file, /* showImage */ false);
      return result?.decodedText || null;
    } catch {
      return null; // no code in this frame
    }
  }

  stop() {
    this.running = false;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  destroy() {
    this.stop();
    try {
      this.fallback?.clear?.();
    } catch {
      /* already torn down */
    }
    this.fallbackHost?.remove();
    this.fallback = null;
    this.detector = null;
  }
}

/**
 * Cheap local pre-filter.
 *
 * The faculty display shows mostly decoys, so most decodes are not worth a
 * round trip. Payloads are a fixed 34-character shape, and decoys are
 * generated to match it exactly — so this rejects unrelated QR codes (a
 * poster on the wall, someone's wifi code) without ever being able to
 * distinguish a real payload from a decoy. Validity remains entirely the
 * server's decision; this only avoids obviously pointless requests on a
 * congested network.
 */
const PAYLOAD_SHAPE = /^[A-Z2-7]{8}\.[0-9A-Z]{8}\.[A-Z2-7]{16}$/;

export function looksLikeAttendancePayload(text) {
  return typeof text === "string" && text.length === 34 && PAYLOAD_SHAPE.test(text);
}
