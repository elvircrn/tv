// Trunk's custom-initializer hook (see CLAUDE.md / the `data-initializer`
// attribute on the `<link data-trunk rel="rust">` tag in index.html and
// index-mt.html) — this runs BEFORE the wasm module has loaded at all, so
// it can't rely on anything from the Rust side. It draws a small progress
// bar over the otherwise-blank page while the (multi-MB) wasm binary
// downloads, then hands off to the real app once it's ready.
export default function trunkInitializer() {
  const overlay = document.createElement("div");
  overlay.id = "tv-loading";
  overlay.innerHTML = `
    <div class="tv-loading-bar-track">
      <div class="tv-loading-bar-fill"></div>
    </div>
    <div class="tv-loading-label">Loading…</div>
  `;
  document.body.appendChild(overlay);
  const fill = overlay.querySelector(".tv-loading-bar-fill");
  const label = overlay.querySelector(".tv-loading-label");

  return {
    onProgress: ({ current, total }) => {
      if (total) {
        const pct = Math.min(100, Math.round((current / total) * 100));
        fill.classList.remove("indeterminate");
        fill.style.width = pct + "%";
        label.textContent = `Loading… ${pct}%`;
      } else {
        // No Content-Length from the server — show a moving indicator and
        // fall back to a running byte count instead of a stalled 0%.
        fill.classList.add("indeterminate");
        label.textContent = `Loading… ${(current / (1024 * 1024)).toFixed(1)} MB`;
      }
    },
    onSuccess: () => {
      // The wasm side draws its own (dark-background) splash once it takes
      // over, and the page background already matches, so a plain removal
      // here is enough — no fade needed to avoid a visible seam.
      overlay.remove();
    },
    onFailure: (error) => {
      fill.classList.add("failed");
      label.textContent = "Failed to load: " + (error?.message ?? error);
    },
  };
}
