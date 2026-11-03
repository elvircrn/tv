// Trunk's custom-initializer hook (see CLAUDE.md / the `data-initializer`
// attribute on the `<link data-trunk rel="rust">` tag in index.html and
// index-mt.html) — this runs BEFORE the wasm module has loaded at all, so
// it can't rely on anything from the Rust side. It draws the app splash
// (vLLM mark + progress bar + label) over the otherwise-blank page while
// the (multi-MB) wasm binary downloads.
//
// For a share-link open (?src=) this overlay IS the whole load
// screen: it stays up after the wasm boots, and the Rust side drives the
// same bar/label on through the trace download and the parse
// (update_boot_overlay in main.rs), removing the overlay only when the
// trace is ready — one continuous loading bar from link to timeline,
// instead of an app bar that fills and then a second one resets when the
// trace download begins. The Rust-side weights (boot_weights in main.rs)
// must match WASM_SHARE below.
export default function trunkInitializer() {
  const overlay = document.createElement("div");
  overlay.id = "tv-loading";
  // The same mark the app draws (draw_vllm_logo in ui.rs — exact path data
  // from the official vLLM-Logo.svg, same colors), so the wasm->app
  // handoff has no visual seam.
  overlay.innerHTML = `
    <svg width="96" height="96" viewBox="0 0 32 32" aria-hidden="true">
      <polygon points="13.417,7.835 13.417,30.286 2.191,7.835" fill="#fdb515"/>
      <polygon points="13.416,30.286 22.237,30.286 29.809,1.714 19.427,7.179" fill="#30a2ff"/>
    </svg>
    <div class="tv-loading-bar-track">
      <div class="tv-loading-bar-fill" id="tv-boot-fill"></div>
    </div>
    <div class="tv-loading-label" id="tv-boot-label">Loading...</div>
  `;
  document.body.appendChild(overlay);
  const fill = overlay.querySelector("#tv-boot-fill");
  const label = overlay.querySelector("#tv-boot-label");

  // A share-link open keeps this overlay as its single load screen (the
  // Rust side takes the bar over and removes it); a plain app open has
  // nothing left to load once the wasm is here.
  const params = new URLSearchParams(location.search);
  const share = params.has("src");
  // How much of the whole link-to-timeline load this phase owns. The rest
  // (50% trace download, 25% parse) is driven from Rust — see
  // boot_weights in main.rs.
  const WASM_SHARE = share ? 0.25 : 1.0;

  return {
    onProgress: ({ current, total }) => {
      if (total) {
        const pct = Math.min(100, Math.round((current / total) * 100 * WASM_SHARE));
        fill.classList.remove("indeterminate");
        fill.style.width = pct + "%";
        label.textContent = (share ? "Loading trace... " : "Loading... ") + pct + "%";
      } else {
        // No Content-Length from the server — show a moving indicator and
        // fall back to a running byte count instead of a stalled 0%.
        fill.classList.add("indeterminate");
        label.textContent = `Loading... ${(current / (1024 * 1024)).toFixed(1)} MB`;
      }
    },
    onSuccess: () => {
      if (!share) {
        // The wasm side draws its own (dark-background) splash once it
        // takes over, and the page background already matches, so a plain
        // removal here is enough — no fade needed to avoid a visible seam.
        overlay.remove();
      }
      // Else: update_boot_overlay (main.rs) continues driving #tv-boot-fill
      // / #tv-boot-label from here and removes the overlay when the trace
      // (or its error) is on screen.
    },
    onFailure: (error) => {
      fill.classList.add("failed");
      label.textContent = "Failed to load: " + (error?.message ?? error);
    },
  };
}
