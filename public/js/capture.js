/**
 * Burst capture with client-side quality scoring.
 *
 * THREE PROBLEMS THIS SOLVES
 * --------------------------
 * 1. SINGLE-FRAME ROULETTE. The old code captured one frame at the instant
 *    Submit was pressed. A blink or a hand tremor produced a garbage
 *    embedding and an unexplainable rejection. Capturing a short burst and
 *    keeping the sharpest frames removes that coin flip. It is the highest
 *    accuracy-per-line change available: false rejections drop sharply while
 *    false acceptances barely move, because several frames of an impostor
 *    are still an impostor.
 *
 * 2. WHAT YOU SEE IS NOT WHAT YOU SEND. The preview was styled
 *    `object-fit: cover` (cropped) while capture did
 *    `drawImage(video, 0, 0, w, h)` (full frame, squashed to the canvas
 *    aspect). The student framed their face against one picture and the
 *    server received a different, distorted one. We replicate the cover
 *    geometry exactly, so the captured pixels are the previewed pixels.
 *
 * 3. RESOLUTION THROWN AWAY. Capture was downscaled to 480px wide and
 *    encoded at JPEG quality 0.7. At arm's length that leaves a face roughly
 *    90px across, which the recogniser then upsamples into its 112x112
 *    input — feeding it interpolated detail that was never photographed.
 *    Here we crop to the framing guide first and then size, so the face
 *    occupies most of the payload instead of the ceiling and the student's
 *    shoulders.
 */

/** Where the face guide sits, as a fraction of the previewed area. */
const GUIDE = { width: 0.82, height: 0.82, centerY: 0.46 };

/** Longest edge of an uploaded frame. Cropped to the guide, this keeps the
 *  face well above the 55px inter-ocular floor while staying ~80-120KB. */
const OUTPUT_MAX_EDGE = 720;
const JPEG_QUALITY = 0.88;

/** Burst shape. ~90ms spacing spans hand tremor without feeling slow. */
const BURST_FRAMES = 5;
const BURST_INTERVAL_MS = 90;

/** Small canvas used only for scoring — full-resolution analysis is wasteful. */
const SCORE_WIDTH = 160;

const scoreCanvas = document.createElement("canvas");
const scoreContext = scoreCanvas.getContext("2d", { alpha: false, willReadFrequently: true });

const outputCanvas = document.createElement("canvas");
const outputContext = outputCanvas.getContext("2d", { alpha: false });

/**
 * The source rectangle actually visible under `object-fit: cover`.
 * Mirrors what the browser does to render the preview.
 */
function coverRect(sourceWidth, sourceHeight, displayWidth, displayHeight) {
  if (!displayWidth || !displayHeight) {
    return { sx: 0, sy: 0, sw: sourceWidth, sh: sourceHeight };
  }
  const scale = Math.max(displayWidth / sourceWidth, displayHeight / sourceHeight);
  const sw = Math.min(sourceWidth, displayWidth / scale);
  const sh = Math.min(sourceHeight, displayHeight / scale);
  return { sx: (sourceWidth - sw) / 2, sy: (sourceHeight - sh) / 2, sw, sh };
}

/** Narrow the visible rectangle to the on-screen framing guide. */
function guideRect(visible) {
  const sw = visible.sw * GUIDE.width;
  const sh = visible.sh * GUIDE.height;
  const sx = visible.sx + (visible.sw - sw) / 2;
  // Faces sit slightly above centre when someone holds a phone at eye level.
  const sy = visible.sy + Math.max(0, visible.sh * GUIDE.centerY - sh / 2);
  return { sx, sy, sw, sh: Math.min(sh, visible.sy + visible.sh - sy) };
}

/**
 * Variance of the 4-neighbour Laplacian — the same blur proxy the server
 * uses, so client and server agree on what "too blurry" means. Computed on a
 * 160px grayscale copy: enough signal to rank frames, cheap enough to run
 * five times without dropping the preview.
 */
function sharpness(imageData) {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);

  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }

  let sum = 0;
  let sumSquares = 0;
  let count = 0;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const p = y * width + x;
      const value =
        gray[p - width] + gray[p + width] + gray[p - 1] + gray[p + 1] - 4 * gray[p];
      sum += value;
      sumSquares += value * value;
      count += 1;
    }
  }

  if (count === 0) return { variance: 0, brightness: 0 };

  const mean = sum / count;
  let brightness = 0;
  for (let i = 0; i < gray.length; i += 1) brightness += gray[i];

  return {
    variance: sumSquares / count - mean * mean,
    brightness: brightness / gray.length,
  };
}

/** Score one frame without allocating a full-resolution bitmap. */
function scoreFrame(video, rect) {
  const aspect = rect.sh / rect.sw;
  scoreCanvas.width = SCORE_WIDTH;
  scoreCanvas.height = Math.max(1, Math.round(SCORE_WIDTH * aspect));

  scoreContext.drawImage(
    video,
    rect.sx, rect.sy, rect.sw, rect.sh,
    0, 0, scoreCanvas.width, scoreCanvas.height
  );

  const imageData = scoreContext.getImageData(0, 0, scoreCanvas.width, scoreCanvas.height);
  return sharpness(imageData);
}

/** Encode the guide region at full working resolution. */
function encodeFrame(video, rect) {
  const aspect = rect.sh / rect.sw;
  const width = Math.min(OUTPUT_MAX_EDGE, Math.round(rect.sw));
  const height = Math.max(1, Math.round(width * aspect));

  outputCanvas.width = width;
  outputCanvas.height = height;
  outputContext.drawImage(video, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, width, height);

  return outputCanvas.toDataURL("image/jpeg", JPEG_QUALITY);
}

/**
 * Resolve the source rectangle that the student is actually looking at.
 *
 * MUST be called while the capture screen is still VISIBLE.
 * `getBoundingClientRect()` on a hidden element returns 0x0, which silently
 * collapses `coverRect` to the full sensor frame — a completely different
 * crop from the one the student framed. Capturing after switching screens
 * therefore uploads the wrong region, which is exactly the sort of quiet
 * accuracy loss that is impossible to diagnose from a rejection message.
 * Callers compute this first and pass it through.
 */
export function computeCaptureRegion(camera) {
  const { width, height } = camera.dimensions;
  if (!camera.isLive || width === 0) return null;

  const displayRect = camera.video.getBoundingClientRect();
  if (!displayRect.width || !displayRect.height) return null;

  return guideRect(coverRect(width, height, displayRect.width, displayRect.height));
}

/** Client-side gate. Catches the obvious cases before spending a round trip. */
export function assessLiveFrame(camera) {
  const { width, height } = camera.dimensions;
  if (!camera.isLive || width === 0) {
    return { ok: false, code: "CAMERA_NOT_READY", message: "Starting camera…" };
  }

  const rect = computeCaptureRegion(camera);
  if (!rect) return { ok: false, code: "CAMERA_NOT_READY", message: "Starting camera…" };

  const { variance, brightness } = scoreFrame(camera.video, rect);

  // Thresholds are on the 160px scoring canvas, deliberately loose: this is
  // guidance to help the student, not the authority. The server decides.
  if (brightness < 45) {
    return { ok: false, code: "TOO_DARK", message: "Too dark — face a window or a light", variance, brightness };
  }
  if (brightness > 215) {
    return { ok: false, code: "TOO_BRIGHT", message: "Too bright — move out of direct light", variance, brightness };
  }
  if (variance < 8) {
    return { ok: false, code: "TOO_BLURRY", message: "Hold steady", variance, brightness };
  }
  return { ok: true, code: "READY", message: "Looking good", variance, brightness };
}

/**
 * Wait for the next painted frame, with a hard ceiling.
 *
 * requestAnimationFrame never fires while the page is hidden (and is throttled
 * to ~1Hz in some background states), so awaiting it bare can block forever.
 * Every wait in the capture path is bounded.
 */
function nextFrame(timeoutMs = 250) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    requestAnimationFrame(finish);
    setTimeout(finish, timeoutMs);
  });
}

/** Retain one burst frame as a bitmap so it can be ranked, then encoded. */
function grabFrame(video, rect) {
  const aspect = rect.sh / rect.sw;
  const width = Math.min(OUTPUT_MAX_EDGE, Math.round(rect.sw));
  const height = Math.max(1, Math.round(width * aspect));

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas
    .getContext("2d", { alpha: false })
    .drawImage(video, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, width, height);
  return canvas;
}

/** Score an already-captured canvas (not the live element). */
function scoreCanvasFrame(source) {
  const aspect = source.height / source.width;
  scoreCanvas.width = SCORE_WIDTH;
  scoreCanvas.height = Math.max(1, Math.round(SCORE_WIDTH * aspect));
  scoreContext.drawImage(source, 0, 0, scoreCanvas.width, scoreCanvas.height);
  return sharpness(scoreContext.getImageData(0, 0, scoreCanvas.width, scoreCanvas.height));
}

/** Free a canvas's backing store immediately rather than waiting for GC. */
function release(canvas) {
  canvas.width = 0;
  canvas.height = 0;
}

/**
 * Capture a burst and return the sharpest frames, best first.
 *
 * Server-side matching takes the max similarity across frames, and the
 * verifier stops early on the first accept — so ordering makes the typical
 * request one frame of work instead of three.
 *
 * Each frame is RETAINED as it is captured, then ranked, then only the
 * winners are encoded. An earlier version scored the burst and afterwards
 * encoded whatever happened to be on screen, which made the whole selection
 * a no-op and sent several near-identical frames — the exact opposite of the
 * diversity a burst is supposed to provide.
 *
 * @param {import("./camera.js").Camera} camera
 * @param {{ frames?: number, region?: object, onProgress?: (n: number, total: number) => void }} options
 */
export async function captureBurst(camera, { frames = 3, region = null, onProgress } = {}) {
  if (!camera.isLive || camera.dimensions.width === 0) {
    throw new Error("CAMERA_NOT_READY");
  }

  // Prefer the region resolved while the preview was on screen.
  const rect = region || computeCaptureRegion(camera);
  if (!rect) throw new Error("CAMERA_NOT_READY");

  const captured = [];
  try {
    for (let i = 0; i < BURST_FRAMES; i += 1) {
      // Align to the compositor so we sample distinct decoded frames rather
      // than the same one several times — but never BLOCK on it. Browsers
      // suspend requestAnimationFrame while a page is hidden, so a student
      // who switches apps mid-capture would otherwise hang on "Capturing…"
      // forever with no timeout anywhere in the chain.
      await nextFrame();

      // If the page went away, stop rather than collecting stale frames: the
      // element holds whatever was last decoded, which may be seconds old.
      if (document.hidden) {
        if (captured.length === 0) throw new Error("CAPTURE_INTERRUPTED");
        break;
      }

      const canvas = grabFrame(camera.video, rect);
      captured.push({ index: i, canvas, ...scoreCanvasFrame(canvas) });
      onProgress?.(i + 1, BURST_FRAMES);

      if (i < BURST_FRAMES - 1) {
        await new Promise((resolve) => setTimeout(resolve, BURST_INTERVAL_MS));
      }
    }

    captured.sort((a, b) => b.variance - a.variance);
    const keep = captured.slice(0, Math.max(1, frames));

    // Encoding is the expensive step, so only the winners pay it.
    const encoded = keep.map((frame) => frame.canvas.toDataURL("image/jpeg", JPEG_QUALITY));

    return {
      frames: encoded,
      // The sharpest frame, kept alive for the freeze preview so the student
      // sees the image actually being evaluated.
      best: keep[0].canvas,
      metrics: keep.map(({ index, variance, brightness }) => ({ index, variance, brightness })),
      region: rect,
      approxBytes: encoded.reduce((total, frame) => total + Math.floor(frame.length * 0.75), 0),
    };
  } finally {
    // Release every frame except the one handed back for display.
    const winner = captured.length
      ? captured.slice().sort((a, b) => b.variance - a.variance)[0].canvas
      : null;
    for (const frame of captured) {
      if (frame.canvas !== winner) release(frame.canvas);
    }
  }
}

/**
 * Show the frame that is actually being verified.
 *
 * Takes the winning burst canvas rather than re-sampling the camera, so the
 * student sees precisely the image the server is judging. The old flow set
 * `display: none` on everything and left them staring at a blank screen for
 * the 5-15 seconds verification took.
 *
 * @param {HTMLCanvasElement} canvas destination, on screen
 * @param {HTMLCanvasElement} source the `best` canvas from captureBurst
 */
export function showFrozenFrame(canvas, source) {
  if (!source || !source.width) return false;

  canvas.width = source.width;
  canvas.height = source.height;
  canvas.getContext("2d", { alpha: false }).drawImage(source, 0, 0);

  // The winner is no longer needed once it is on screen.
  release(source);
  return true;
}
