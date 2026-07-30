import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * The faculty app is a static bundle that talks to the attendance backend.
 *
 * In development it proxies /api to the local server so the browser sees one
 * origin — no CORS in the loop while iterating, and the same relative URLs
 * work unchanged in production when the bundle is served from the backend's
 * own origin or a CDN pointed at it.
 */
export default defineConfig({
  plugins: [react()],

  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://localhost:7860",
        changeOrigin: true,
      },
    },
  },

  build: {
    outDir: "dist",
    // The projector machine is not a phone; a slightly larger, well-cached
    // bundle is fine. Sourcemaps stay off so the build is not shipping
    // readable internals to a classroom machine.
    sourcemap: false,
    target: "es2020",
  },
});
