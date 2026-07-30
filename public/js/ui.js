/**
 * View layer — all DOM access lives here so app.js stays pure orchestration.
 *
 * ACCESSIBILITY NOTES
 * -------------------
 * The old page used `alert()` for every error (blocking, unstyled, and on iOS
 * capable of interrupting an active MediaStream), had no labels, no live
 * regions, and set `user-scalable=no` — a WCAG 1.4.4 violation that iOS has
 * ignored since iOS 10 anyway.
 *
 * Status text now lives in polite/assertive live regions so screen readers
 * announce state changes, every control has an accessible name, and pinch
 * zoom works.
 */

const screens = new Map();
let statusTimer = null;

export function registerScreens(root) {
  for (const element of root.querySelectorAll("[data-screen]")) {
    screens.set(element.dataset.screen, element);
  }
}

export function showScreen(name) {
  for (const [key, element] of screens) {
    const active = key === name;
    element.hidden = !active;
    element.setAttribute("aria-hidden", String(!active));
  }
  // Drop advice about the previous screen; keep conditions that are still
  // true (offline). Without this a "your scan expired" warning can sit above
  // a success screen that has plainly superseded it.
  hideBanner({ includeSticky: false });
  // Move focus to the new screen's heading so keyboard and screen-reader
  // users are not left behind on a hidden element.
  const heading = screens.get(name)?.querySelector("h1, h2, [data-autofocus]");
  if (heading) {
    heading.setAttribute("tabindex", "-1");
    heading.focus({ preventScroll: true });
  }
}

export const $ = (selector) => document.querySelector(selector);

/**
 * Camera-guidance line. Debounced because the quality gate samples several
 * times a second and unthrottled updates make the text strobe unreadably.
 */
export function setCameraHint(text, tone = "neutral") {
  const element = $("#camera-hint");
  if (!element) return;
  if (element.textContent === text && element.dataset.tone === tone) return;

  clearTimeout(statusTimer);
  statusTimer = setTimeout(() => {
    element.textContent = text;
    element.dataset.tone = tone;
  }, 120);
}

export function setScanStatus(text, tone = "neutral") {
  const element = $("#scan-status");
  if (!element) return;
  element.textContent = text;
  element.dataset.tone = tone;
}

/**
 * Banner for connectivity and recoverable problems.
 *
 * Banners are either STICKY or TRANSIENT.
 *   sticky    — a condition that is still true (offline). Only whatever set
 *               it may clear it.
 *   transient — advice about a moment that has passed ("scan expired",
 *               "reconnecting to camera"). Cleared automatically on the next
 *               screen change, so stale advice never sits on top of a screen
 *               that has already moved on and contradicted it.
 *
 * `assertive` interrupts a screen reader; reserve it for things that block.
 */
export function showBanner(text, { tone = "info", assertive = false, sticky = false } = {}) {
  const banner = $("#banner");
  if (!banner) return;
  banner.textContent = text;
  banner.dataset.tone = tone;
  banner.dataset.sticky = String(sticky);
  banner.setAttribute("aria-live", assertive ? "assertive" : "polite");
  banner.hidden = false;
}

export function hideBanner({ includeSticky = true } = {}) {
  const banner = $("#banner");
  if (!banner) return;
  if (!includeSticky && banner.dataset.sticky === "true") return;
  banner.hidden = true;
}

/** Progress line during verification, so a slow network never looks frozen. */
export function setProgress(text, { spinner = true } = {}) {
  const element = $("#verify-status");
  if (element) element.textContent = text;
  const spin = $("#verify-spinner");
  if (spin) spin.hidden = !spinner;
}

export function setSubmitEnabled(enabled) {
  const button = $("#submit-btn");
  if (!button) return;
  button.disabled = !enabled;
  button.setAttribute("aria-disabled", String(!enabled));
}

export function setBusy(button, busy, busyLabel = "Working…") {
  if (!button) return;
  if (busy) {
    button.dataset.idleLabel = button.textContent;
    button.textContent = busyLabel;
    button.disabled = true;
  } else {
    if (button.dataset.idleLabel) button.textContent = button.dataset.idleLabel;
    button.disabled = false;
  }
}

/** Terminal result screen. */
export function showResult({ title, detail, tone = "success", actions = [] }) {
  $("#result-icon").dataset.tone = tone;
  $("#result-icon").textContent = tone === "success" ? "✓" : tone === "warn" ? "!" : "×";
  $("#result-title").textContent = title;
  $("#result-detail").textContent = detail || "";

  const container = $("#result-actions");
  container.replaceChildren();
  for (const action of actions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = action.primary ? "btn btn-primary" : "btn btn-ghost";
    button.textContent = action.label;
    button.addEventListener("click", action.onClick);
    container.appendChild(button);
  }

  showScreen("result");
}

/** Haptic acknowledgement. Silently absent on iOS, which does not support it. */
export function vibrate(pattern) {
  try {
    navigator.vibrate?.(pattern);
  } catch {
    /* unsupported */
  }
}

/**
 * Keep the screen awake during a session. Students hold the phone still while
 * scanning, which is exactly when the display times out and the stream stops.
 */
export async function acquireWakeLock() {
  try {
    if (!("wakeLock" in navigator)) return null;
    const sentinel = await navigator.wakeLock.request("screen");
    // Re-acquire after backgrounding, or it stays released.
    document.addEventListener("visibilitychange", async () => {
      if (document.visibilityState === "visible" && sentinel.released) {
        try {
          await navigator.wakeLock.request("screen");
        } catch {
          /* best effort */
        }
      }
    });
    return sentinel;
  } catch {
    return null;
  }
}
