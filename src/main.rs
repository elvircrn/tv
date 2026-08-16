mod parse;
mod types;
#[cfg(not(target_arch = "wasm32"))]
mod renderer;
#[cfg(target_arch = "wasm32")]
mod renderer_web;
#[cfg(target_arch = "wasm32")]
mod wasm_libc_shims;
mod loader;
mod state;
mod ui;
mod diff;
mod time;

use parse::{skip_value, parse_args_flat, FnvMap};
use types::*;
#[cfg(not(target_arch = "wasm32"))]
use renderer::MetalRenderer as PlatformRenderer;
#[cfg(target_arch = "wasm32")]
use renderer_web::WebGl2Renderer as PlatformRenderer;
use state::*;
use ui::*;

use imgui::{ClipboardBackend, Condition, InputTextCallback, InputTextCallbackHandler, StyleColor, StyleVar, TextCallbackData, WindowFlags};
use std::fmt::Write as FmtWrite;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Write as IoWrite;
use crate::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, Touch, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

#[cfg(target_os = "macos")]
extern "C" {
    fn objc_autoreleasePoolPush() -> *mut std::ffi::c_void;
    fn objc_autoreleasePoolPop(pool: *mut std::ffi::c_void);
}

/// Compact rank descriptor for the status bar, e.g. "rank 0/32 · tp0 pp0 dp0 ep0".
/// `rank`/`world` come from the trace's `distributedInfo`; the parallelism
/// coordinates are parsed from the vLLM filename (`..dp0_pp0_tp0_dcp0_ep0_rank0..`).
/// Returns "" when nothing useful is known (non-distributed / non-vLLM traces).
fn rank_summary(fname: &str, dist_rank: i32, dist_world: i32) -> String {
    let mut out = String::new();
    if dist_rank >= 0 && dist_world > 0 {
        let _ = write!(out, "rank {}/{}", dist_rank, dist_world);
    } else if dist_rank >= 0 {
        let _ = write!(out, "rank {}", dist_rank);
    } else if dist_world > 0 {
        let _ = write!(out, "{} ranks", dist_world);
    }

    // Parallelism coordinates from the underscore/dot-delimited filename tokens.
    // Fixed display order; dcp is omitted when 0 (the common case) to cut noise.
    let mut mesh = String::new();
    for pfx in ["tp", "pp", "dp", "ep", "dcp"] {
        for tok in fname.split(|c| c == '_' || c == '.') {
            if let Some(rest) = tok.strip_prefix(pfx) {
                if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
                    if pfx == "dcp" && rest == "0" { break; }
                    if !mesh.is_empty() { mesh.push(' '); }
                    let _ = write!(mesh, "{}{}", pfx, rest);
                    break;
                }
            }
        }
    }
    if !mesh.is_empty() {
        if !out.is_empty() { out.push_str(" · "); }
        out.push_str(&mesh);
    }
    out
}

/// Short label for a pane's tab / "Diff against" entry: the dropped folder's
/// name, the trace file's basename, or "Untitled" for a pane with nothing
/// loaded yet. Multi-rank panes store `trace_path` as "N ranks: prefix" (see
/// `Pane::open_multi`) — this shows just the prefix, not the rank count,
/// since the count belongs in the status bar, not a short tab label.
fn tab_label(pane: &Pane) -> &str {
    if let Some(dir) = &pane.reload_dir {
        if let Some(name) = std::path::Path::new(dir).file_name().and_then(|f| f.to_str()) {
            return name;
        }
    }
    if pane.trace_path.is_empty() {
        return "Untitled";
    }
    let shown = pane.trace_path.split_once("ranks: ").map(|(_, p)| p).unwrap_or(&pane.trace_path);
    shown.rsplit('/').next().unwrap_or(shown)
}

struct App {
    window: Option<Window>,
    imgui: Option<imgui::Context>,
    renderer: Option<PlatformRenderer>,
    state: AppState,
    last_frame: Instant,
    // Actual measured height of the toolbar window from the last frame it
    // rendered — the toolbar's own box height is content-fit (see
    // render_frame), so the tab strip below it needs the *real* number, not
    // the old fixed TOOLBAR_H, or a few px of leftover box height between
    // the menu bar and the tab row reads as a visible gap. Falls back to
    // TOOLBAR_H before the first frame renders it.
    toolbar_h: f32,
    scale_factor: f64,
    scroll_accum: [f32; 2],
    pinch_accum: f32,
    mod_super: bool,
    mod_ctrl: bool,
    mod_shift: bool,
    pending_files: Vec<String>,
    pending_drops: Vec<String>,
    last_mouse_x: f32,
    nav_keys: u8,
    // Touch support (mobile browsers): imgui only understands a mouse
    // position + button state, so a single active touch is translated into
    // synthesized CursorMoved/MouseInput events through the exact same IO
    // calls real mouse input uses — every existing mouse-driven interaction
    // (pan, select, resize dividers) works unmodified once fed that way. A
    // second simultaneous touch means the user is pinching, not dragging,
    // so it releases the synthesized button and switches to feeding
    // `pinch_accum` (macOS's native PinchGesture never fires on web) from
    // the frame-to-frame relative change in inter-touch distance, matching
    // the same "relative growth factor" semantics PinchGesture's `delta`
    // already has (see the zoom-factor computation in ui.rs).
    active_touches: std::collections::HashMap<u64, (f32, f32)>,
    primary_touch_id: Option<u64>,
    last_pinch_dist: Option<f32>,
    nav_pan_vel: f64,
    nav_zoom_vel: f64,
    last_reload: Instant,
    // winit's web backend never resizes the canvas on its own when the
    // browser window/viewport changes — unlike native windows, a <canvas>
    // has no OS-level "this window got resized" notion, so nothing calls
    // `request_inner_size` for us. We listen for the DOM `resize` event
    // ourselves (registered in `resumed()`) and use this proxy to hop back
    // onto the winit event loop and issue that resize request from
    // `user_event()`, where `self.window` is reachable.
    #[cfg(target_arch = "wasm32")]
    event_loop_proxy: Option<winit::event_loop::EventLoopProxy<()>>,
    // winit's web backend doesn't implement WindowEvent::DroppedFile at all —
    // there's no OS-level file-drop concept for a <canvas> the way there is
    // for a real window. We register our own "dragover"/"drop" DOM listeners
    // (see `resumed()`) and read each dropped File's bytes asynchronously
    // (File::array_buffer() is JS-Promise-based); completed (name, bytes)
    // pairs land here, then get drained and routed into panes from
    // `user_event()` once the async read(s) finish and wake the event loop.
    #[cfg(target_arch = "wasm32")]
    pending_web_files: std::rc::Rc<std::cell::RefCell<Vec<(String, Vec<u8>)>>>,
    // imgui's ClipboardBackend::get is synchronous, but the Clipboard API's
    // readText() is Promise-based — there's no way to serve a paste request
    // through that trait at all. Instead we listen for the browser's own
    // native "paste" event on the canvas (registered in `resumed()`), which
    // hands over the pasted text synchronously as part of the event, and
    // feed it into imgui's IO directly from `user_event()` (same hop-via-
    // proxy pattern as the resize/drop workarounds) as if it had been typed.
    #[cfg(target_arch = "wasm32")]
    pending_paste: std::rc::Rc<std::cell::RefCell<Option<String>>>,
    // Shareable-link loading: `?src=<url>` is fetched once at startup (see
    // `resumed()`). A successful fetch's bytes go through the same
    // `pending_web_files` queue a dropped file would; a failed fetch (bad
    // URL, 404, CORS rejection) has nothing to hand `open_from_bytes`, so it
    // needs its own slot to report the error back once the async fetch
    // settles — same Rc<RefCell> + proxy-wake hop as `pending_paste`.
    #[cfg(target_arch = "wasm32")]
    pending_src_error: std::rc::Rc<std::cell::RefCell<Option<String>>>,
    // Populated during a `?src=`/`?gist=` download (see `fetch_bytes_with_progress`)
    // so the empty-pane placeholder can show a live "Downloading: X / Y" line
    // instead of the page looking frozen for as long as the fetch takes.
    // Cleared once the fetch settles either way (bytes land in
    // `pending_web_files`, or an error lands in `pending_src_error`).
    #[cfg(target_arch = "wasm32")]
    pending_download_progress: std::rc::Rc<std::cell::RefCell<Option<DownloadProgress>>>,
}

impl App {
    fn new(files: Vec<String>) -> Self {
        Self {
            window: None,
            imgui: None,
            renderer: None,
            state: AppState {
                panes: vec![Pane::new()],
                active: 0,
                pending_active_tab: None,
                buf: DrawBuf::default(),
                bottom_h: DETAIL_H,
                drag: DragKind::None,
                show_diff: false,
                diff_popup_open: false,
                diff_result: None,
                diff_bar_scroll: 0.0,
                diff_bar_zoom: 1.0,
                diff_pane_indices: None,
            },
            last_frame: Instant::now(),
            toolbar_h: TOOLBAR_H,
            scale_factor: 1.0,
            scroll_accum: [0.0; 2],
            pinch_accum: 0.0,
            mod_super: false,
            mod_ctrl: false,
            mod_shift: false,
            pending_files: files,
            pending_drops: Vec::new(),
            last_mouse_x: 0.0,
            nav_keys: 0,
            nav_pan_vel: 0.0,
            nav_zoom_vel: 0.0,
            last_reload: Instant::now(),
            active_touches: std::collections::HashMap::new(),
            primary_touch_id: None,
            last_pinch_dist: None,
            #[cfg(target_arch = "wasm32")]
            event_loop_proxy: None,
            #[cfg(target_arch = "wasm32")]
            pending_web_files: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
            #[cfg(target_arch = "wasm32")]
            pending_paste: std::rc::Rc::new(std::cell::RefCell::new(None)),
            #[cfg(target_arch = "wasm32")]
            pending_src_error: std::rc::Rc::new(std::cell::RefCell::new(None)),
            #[cfg(target_arch = "wasm32")]
            pending_download_progress: std::rc::Rc::new(std::cell::RefCell::new(None)),
        }
    }
}

const NAV_W: u8 = 1;
const NAV_A: u8 = 2;
const NAV_S: u8 = 4;
const NAV_D: u8 = 8;
const NAV_UP: u8 = 16;
const NAV_DOWN: u8 = 32;
const NAV_LEFT: u8 = 64;
const NAV_RIGHT: u8 = 128;

#[cfg(not(target_arch = "wasm32"))]
struct MacClipboard;
#[cfg(not(target_arch = "wasm32"))]
impl imgui::ClipboardBackend for MacClipboard {
    fn get(&mut self) -> Option<String> {
        std::process::Command::new("pbpaste")
            .output().ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
    }
    fn set(&mut self, value: &str) {
        if let Ok(mut child) = std::process::Command::new("pbcopy")
            .stdin(std::process::Stdio::piped()).spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(value.as_bytes()).ok();
            }
            child.wait().ok();
        }
    }
}

// Browser clipboard access is async (Clipboard API) and paste can't be
// pulled synchronously through imgui's ClipboardBackend::get; real copy/paste
// wiring (writeText + a canvas paste-event listener) lands in a later phase.
#[cfg(target_arch = "wasm32")]
struct WebClipboard;
#[cfg(target_arch = "wasm32")]
impl imgui::ClipboardBackend for WebClipboard {
    // The Clipboard API's readText()/writeText() are both Promise-based, but
    // ClipboardBackend::get is synchronous — there's no way to actually wait
    // for a read here. Paste is handled separately (see the canvas "paste"
    // listener in resumed(), which gets pasted text synchronously from the
    // browser's native paste event instead of round-tripping through this
    // trait at all).
    fn get(&mut self) -> Option<String> { None }
    fn set(&mut self, value: &str) {
        let Some(clipboard) = web_sys::window().map(|w| w.navigator().clipboard()) else { return };
        let value = value.to_string();
        wasm_bindgen_futures::spawn_local(async move {
            let _ = wasm_bindgen_futures::JsFuture::from(clipboard.write_text(&value)).await;
        });
    }
}

// The File and Directory Entries API (webkitGetAsEntry and friends) predates
// Promises on the web platform, so every step is success/error-callback
// based. Wrapping each callback in a one-shot `js_sys::Promise` lets the
// directory walk below read as ordinary (if boxed, for the recursive case)
// async Rust instead of a hand-rolled callback pyramid.
#[cfg(target_arch = "wasm32")]
async fn read_all_directory_entries(reader: &web_sys::FileSystemDirectoryReader) -> Vec<web_sys::FileSystemEntry> {
    use wasm_bindgen::JsCast;
    // readEntries() only returns a batch at a time (spec-mandated, to bound
    // memory for huge directories) — empty result means "no more entries",
    // not "empty directory" necessarily, so this must loop until empty.
    let mut all = Vec::new();
    loop {
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            let cb = wasm_bindgen::closure::Closure::once(move |entries: js_sys::Array| {
                let _ = resolve.call1(&wasm_bindgen::JsValue::NULL, &entries);
            });
            let _ = reader.read_entries_with_callback(cb.as_ref().unchecked_ref());
            cb.forget();
        });
        let Ok(result) = wasm_bindgen_futures::JsFuture::from(promise).await else { break };
        let arr: js_sys::Array = result.unchecked_into();
        if arr.length() == 0 { break; }
        for i in 0..arr.length() {
            all.push(arr.get(i).unchecked_into::<web_sys::FileSystemEntry>());
        }
    }
    all
}

/// Formats a live `DownloadProgress` for the empty-pane placeholder text
/// (see `render_frame`) — a byte count alone when the server didn't send
/// `Content-Length` (chunked/unknown-length responses), a percentage too
/// when it did.
#[cfg(target_arch = "wasm32")]
fn download_progress_text(p: &DownloadProgress) -> String {
    let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
    match p.total {
        Some(total) if total > 0 => format!(
            "Downloading: {:.1} MB / {:.1} MB ({:.0}%)",
            mb(p.received), mb(total), 100.0 * p.received as f64 / total as f64,
        ),
        _ => format!("Downloading: {:.1} MB", mb(p.received)),
    }
}

/// Bytes received so far and, if the server sent `Content-Length`, the
/// total — the total is `None` for chunked/unknown-length responses, in
/// which case the UI shows a byte count instead of a percentage.
#[cfg(target_arch = "wasm32")]
pub struct DownloadProgress {
    pub received: u64,
    pub total: Option<u64>,
}

/// Fetches `url` and returns its response body as bytes — used for the
/// `?src=<url>`/`?gist=<id>` shareable-link feature (see `resumed()`).
/// `url` must serve permissive CORS headers (`Access-Control-Allow-Origin`)
/// since this is a cross-origin `fetch()` from the page's own JS context; a
/// gist raw URL or a repo's `raw.githubusercontent.com` file both do this
/// by default. Reads the response body incrementally via its
/// `ReadableStream` rather than waiting for the whole thing to land in one
/// `array_buffer()` Promise, so `on_progress` can report bytes-received as
/// they arrive instead of the download being an opaque black box until
/// it's entirely done.
#[cfg(target_arch = "wasm32")]
async fn fetch_bytes_with_progress(
    win: &web_sys::Window, url: &str, on_progress: Option<&dyn Fn(DownloadProgress)>,
) -> Result<Vec<u8>, String> {
    use wasm_bindgen::JsCast;
    let resp_value = wasm_bindgen_futures::JsFuture::from(win.fetch_with_str(url))
        .await
        .map_err(|_| "network request failed (bad URL, CORS, or offline)".to_string())?;
    let resp: web_sys::Response = resp_value.unchecked_into();
    if !resp.ok() {
        return Err(format!("server returned HTTP {}", resp.status()));
    }

    let total: Option<u64> = resp.headers().get("content-length").ok().flatten()
        .and_then(|s| s.parse().ok());

    let body = match resp.body() {
        Some(b) => b,
        None => return Ok(Vec::new()),
    };
    let reader: web_sys::ReadableStreamDefaultReader = body.get_reader().unchecked_into();

    let mut bytes: Vec<u8> = Vec::new();
    loop {
        let chunk_value = wasm_bindgen_futures::JsFuture::from(reader.read())
            .await
            .map_err(|_| "failed reading response stream".to_string())?;
        let done = js_sys::Reflect::get(&chunk_value, &"done".into())
            .ok().and_then(|v| v.as_bool()).unwrap_or(true);
        if done { break; }
        if let Ok(value) = js_sys::Reflect::get(&chunk_value, &"value".into()) {
            let chunk: js_sys::Uint8Array = value.unchecked_into();
            let start = bytes.len();
            bytes.resize(start + chunk.length() as usize, 0);
            chunk.copy_to(&mut bytes[start..]);
            if let Some(f) = on_progress {
                f(DownloadProgress { received: bytes.len() as u64, total });
            }
        }
    }
    Ok(bytes)
}

/// Resolves a bare gist ID (from `?gist=<id>`) to its one file's raw URL via
/// GitHub's gist API, which serves permissive CORS and — unlike the raw
/// content host — knows the filename without it being spelled out in the
/// link. Lets a shareable link be the short `?gist=<id>` instead of a fully
/// spelled-out, percent-encoded `?src=<raw-url>`. Picks the first file if a
/// gist somehow has more than one (this app always creates single-file
/// gists via "Export GPU (web)").
#[cfg(target_arch = "wasm32")]
async fn resolve_gist_raw_url(win: &web_sys::Window, gist_id: &str) -> Result<String, String> {
    use wasm_bindgen::JsCast;
    let api_url = format!("https://api.github.com/gists/{gist_id}");
    let resp_value = wasm_bindgen_futures::JsFuture::from(win.fetch_with_str(&api_url))
        .await
        .map_err(|_| "gist lookup failed (network or bad gist id)".to_string())?;
    let resp: web_sys::Response = resp_value.unchecked_into();
    if !resp.ok() {
        return Err(format!("gist lookup returned HTTP {} (check the gist id)", resp.status()));
    }
    let text_promise = resp.text().map_err(|_| "gist response has no body".to_string())?;
    let text_value = wasm_bindgen_futures::JsFuture::from(text_promise)
        .await
        .map_err(|_| "failed reading gist response body".to_string())?;
    let text = text_value.as_string().ok_or_else(|| "gist response wasn't text".to_string())?;
    let json = js_sys::JSON::parse(&text).map_err(|_| "gist response wasn't valid JSON".to_string())?;
    let files = js_sys::Reflect::get(&json, &"files".into())
        .map_err(|_| "gist response missing 'files'".to_string())?;
    let keys = js_sys::Object::keys(files.unchecked_ref::<js_sys::Object>());
    let first_key = keys.get(0);
    if first_key.is_undefined() {
        return Err("gist has no files".to_string());
    }
    let file_info = js_sys::Reflect::get(&files, &first_key)
        .map_err(|_| "gist file entry missing".to_string())?;
    js_sys::Reflect::get(&file_info, &"raw_url".into())
        .ok()
        .and_then(|v| v.as_string())
        .ok_or_else(|| "gist file entry missing raw_url".to_string())
}

#[cfg(target_arch = "wasm32")]
async fn file_from_entry(entry: &web_sys::FileSystemFileEntry) -> Option<web_sys::File> {
    use wasm_bindgen::JsCast;
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let cb = wasm_bindgen::closure::Closure::once(move |file: web_sys::File| {
            let _ = resolve.call1(&wasm_bindgen::JsValue::NULL, &file);
        });
        entry.file_with_callback(cb.as_ref().unchecked_ref());
        cb.forget();
    });
    wasm_bindgen_futures::JsFuture::from(promise).await.ok().map(|v| v.unchecked_into())
}

/// Recursively walks a dropped `FileSystemEntry` (a file, or a folder that
/// may itself contain rank files and/or subfolders), collecting (name,
/// bytes) for every trace-like file found. `.tvcache` *directories* are
/// skipped (native's disk-cache dirs — meaningless on wasm, see
/// `loader::load_cache`'s stub, and not real trace data); a `.tvcache`
/// *file* is still accepted, same as `loader::is_trace_file`.
/// Boxed because async fns can't recurse directly — the resulting future's
/// size would depend on its own size.
#[cfg(target_arch = "wasm32")]
fn walk_dropped_entry(entry: web_sys::FileSystemEntry) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<(String, Vec<u8>)>>>> {
    use wasm_bindgen::JsCast;
    Box::pin(async move {
        if entry.is_file() {
            let file_entry: web_sys::FileSystemFileEntry = entry.unchecked_into();
            let Some(file) = file_from_entry(&file_entry).await else { return Vec::new() };
            let name = file.name();
            if !crate::loader::is_trace_file(&name) { return Vec::new(); }
            let Ok(buf) = wasm_bindgen_futures::JsFuture::from(file.array_buffer()).await else { return Vec::new() };
            vec![(name, js_sys::Uint8Array::new(&buf).to_vec())]
        } else if entry.is_directory() {
            let name = entry.name();
            if name.ends_with(".tvcache") { return Vec::new(); }
            let dir_entry: web_sys::FileSystemDirectoryEntry = entry.unchecked_into();
            let reader = dir_entry.create_reader();
            let mut out = Vec::new();
            for sub_entry in read_all_directory_entries(&reader).await {
                out.extend(walk_dropped_entry(sub_entry).await);
            }
            out
        } else {
            Vec::new()
        }
    })
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Native gets a fixed initial size (there's no "browser viewport" to
        // match). On web, opening at a hardcoded 1400x800 regardless of the
        // actual browser window would leave the canvas mismatched with the
        // page until the first manual resize, so seed it from the real
        // viewport size when we can read one.
        #[cfg(not(target_arch = "wasm32"))]
        let (init_w, init_h) = (INITIAL_WIN_W, INITIAL_WIN_H);
        #[cfg(target_arch = "wasm32")]
        let (init_w, init_h) = {
            let win = web_sys::window();
            let w = win.as_ref().and_then(|w| w.inner_width().ok()).and_then(|v| v.as_f64());
            let h = win.as_ref().and_then(|w| w.inner_height().ok()).and_then(|v| v.as_f64());
            (w.unwrap_or(INITIAL_WIN_W as f64) as f32, h.unwrap_or(INITIAL_WIN_H as f64) as f32)
        };
        let attrs = Window::default_attributes()
            .with_title("Trace Viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(init_w, init_h));
        #[cfg(target_arch = "wasm32")]
        let attrs = {
            use winit::platform::web::WindowAttributesExtWebSys;
            // Without a tabindex, a <canvas> is not a focusable element, so
            // it can never become `document.activeElement` and therefore
            // never receives "keydown"/"keyup" DOM events at all (they're
            // only dispatched to the focused element or its ancestors).
            // winit's web backend registers its keyboard listeners directly
            // on the canvas (see Canvas::on_keyboard_press/release), so
            // without this, WindowEvent::KeyboardInput never fires — no
            // amount of correct event-loop/control-flow handling on our end
            // can compensate. `with_focusable(true)` makes winit set
            // `tabindex="0"` on the canvas at creation time.
            attrs.with_append(true).with_focusable(true)
        };
        let window = event_loop.create_window(attrs).unwrap();
        self.scale_factor = window.scale_factor();

        // Unlike a native OS window, a <canvas> has no concept of being
        // resized by the user dragging its edges — winit's web backend only
        // resizes the canvas when *we* tell it to (via `request_inner_size`,
        // see `user_event` below). Nothing does that on its own, so without
        // this listener the canvas would stay locked at its creation size
        // forever, and the app would never re-layout when the browser
        // window/tab is resized. `resize` fires on `window`, not the canvas,
        // so this can't be wired through winit's per-window event
        // registration — we listen ourselves and hop back onto the winit
        // event loop via a proxy, since `self` isn't reachable from this
        // 'static JS closure.
        #[cfg(target_arch = "wasm32")]
        if let Some(proxy) = self.event_loop_proxy.clone() {
            use wasm_bindgen::JsCast;
            if let Some(win) = web_sys::window() {
                let closure = wasm_bindgen::closure::Closure::<dyn FnMut(_)>::new(
                    move |_e: web_sys::Event| {
                        let _ = proxy.send_event(());
                    },
                );
                let _ = win.add_event_listener_with_callback(
                    "resize",
                    closure.as_ref().unchecked_ref(),
                );
                // Leaked deliberately: this listener must outlive `resumed`
                // and live for the app's entire lifetime (there's exactly
                // one `App` per page load, never torn down).
                closure.forget();
            }
        }

        // Drag-and-drop file loading. winit's web backend has no
        // WindowEvent::DroppedFile at all (see the App::pending_web_files
        // field comment), so this is wired entirely by hand: a "dragover"
        // listener suppresses the browser's default (reject-drop) behavior
        // — without `prevent_default()`, "drop" never fires at all — and a
        // "drop" listener reads each dropped File's bytes (async: File's
        // `array_buffer()` is a JS Promise) and, once ready, stashes them in
        // the shared queue and wakes the winit event loop via the same
        // proxy the resize listener uses, so `user_event` can route them
        // into panes on the next tick.
        #[cfg(target_arch = "wasm32")]
        if let Some(proxy) = self.event_loop_proxy.clone() {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowExtWebSys;
            if let Some(canvas) = window.canvas() {
                let dragover = wasm_bindgen::closure::Closure::<dyn FnMut(_)>::new(
                    |e: web_sys::DragEvent| { e.prevent_default(); },
                );
                let _ = canvas.add_event_listener_with_callback(
                    "dragover", dragover.as_ref().unchecked_ref(),
                );
                dragover.forget();

                let paste_proxy = proxy.clone();
                let pending = self.pending_web_files.clone();
                let drop_listener = wasm_bindgen::closure::Closure::<dyn FnMut(_)>::new(
                    move |e: web_sys::DragEvent| {
                        e.prevent_default();
                        let Some(dt) = e.data_transfer() else { return };
                        let items = dt.items();
                        // webkitGetAsEntry() must be called synchronously,
                        // here, while the DataTransfer is still valid (it's
                        // invalidated as soon as this handler returns) — the
                        // resulting FileSystemEntry handles stay valid for
                        // the async directory walk below, which is what
                        // needs to happen to support dropping a *folder*
                        // (dt.files() alone can't see into one at all: a
                        // bare File object can't represent a directory).
                        let mut entries = Vec::new();
                        for i in 0..items.length() {
                            let Some(item) = items.get(i) else { continue };
                            if let Ok(Some(entry)) = item.webkit_get_as_entry() {
                                entries.push(entry);
                            }
                        }
                        let pending = pending.clone();
                        let proxy = proxy.clone();
                        wasm_bindgen_futures::spawn_local(async move {
                            let mut files = Vec::new();
                            for entry in entries {
                                files.extend(walk_dropped_entry(entry).await);
                            }
                            if !files.is_empty() {
                                pending.borrow_mut().extend(files);
                                let _ = proxy.send_event(());
                            }
                        });
                    },
                );
                let _ = canvas.add_event_listener_with_callback(
                    "drop", drop_listener.as_ref().unchecked_ref(),
                );
                drop_listener.forget();

                // Native's Cmd+V handling (see `WindowEvent::KeyboardInput`
                // below) calls `copy_selection_text`/`PlatformClipboard`
                // directly — there's no equivalent for paste since imgui's
                // `ClipboardBackend::get` can't do anything async. The
                // browser's own "paste" event hands over the clipboard text
                // synchronously (unlike the Clipboard API's readText()), so
                // we take it straight from there instead.
                let pending_paste = self.pending_paste.clone();
                let paste_listener = wasm_bindgen::closure::Closure::<dyn FnMut(_)>::new(
                    move |e: web_sys::ClipboardEvent| {
                        let Some(dt) = e.clipboard_data() else { return };
                        let Ok(text) = dt.get_data("text/plain") else { return };
                        if text.is_empty() { return }
                        e.prevent_default();
                        *pending_paste.borrow_mut() = Some(text);
                        let _ = paste_proxy.send_event(());
                    },
                );
                let _ = canvas.add_event_listener_with_callback(
                    "paste", paste_listener.as_ref().unchecked_ref(),
                );
                paste_listener.forget();
            }
        }

        // Shareable-link loading: either a `?src=<url>` query param (e.g. a
        // repo raw-file URL — anything serving permissive CORS headers) or
        // the shorter `?gist=<id>` (resolved to its raw file URL via
        // resolve_gist_raw_url first) is fetched once at startup and
        // dropped into the same `pending_web_files` queue a real file drop
        // would use, so it's routed into a pane by the exact same code in
        // `user_event` below. A failed fetch (bad URL/id, 404, CORS
        // rejection) has no bytes to hand that queue, so it reports through
        // `pending_src_error` instead.
        #[cfg(target_arch = "wasm32")]
        if let Some(proxy) = self.event_loop_proxy.clone() {
            if let Some(win) = web_sys::window() {
                let search = win.location().search().unwrap_or_default();
                let params = web_sys::UrlSearchParams::new_with_str(&search).ok();
                let src = params.as_ref().and_then(|p| p.get("src"));
                let gist = params.as_ref().and_then(|p| p.get("gist"));
                if src.is_some() || gist.is_some() {
                    let pending = self.pending_web_files.clone();
                    let pending_err = self.pending_src_error.clone();
                    let pending_progress = self.pending_download_progress.clone();
                    wasm_bindgen_futures::spawn_local(async move {
                        let url_result = match (src, gist) {
                            (Some(url), _) => Ok(url),
                            (None, Some(id)) => resolve_gist_raw_url(&win, &id).await,
                            (None, None) => unreachable!(),
                        };
                        let result = match url_result {
                            Ok(url) => {
                                let progress_proxy = proxy.clone();
                                let progress_slot = pending_progress.clone();
                                let on_progress = move |p: DownloadProgress| {
                                    *progress_slot.borrow_mut() = Some(p);
                                    let _ = progress_proxy.send_event(());
                                };
                                fetch_bytes_with_progress(&win, &url, Some(&on_progress)).await
                                    .map(|bytes| (url, bytes))
                            }
                            Err(e) => Err(e),
                        };
                        *pending_progress.borrow_mut() = None;
                        match result {
                            Ok((url, bytes)) => {
                                let name = url.rsplit('/').next()
                                    .filter(|s| !s.is_empty())
                                    .unwrap_or("shared-trace")
                                    .to_string();
                                pending.borrow_mut().push((name, bytes));
                            }
                            Err(e) => {
                                *pending_err.borrow_mut() = Some(format!("shareable link failed to load: {e}"));
                            }
                        }
                        let _ = proxy.send_event(());
                    });
                }
            }
        }

        let mut imgui = imgui::Context::create();
        imgui.io_mut().config_mac_os_behaviors = true;
        #[cfg(not(target_arch = "wasm32"))]
        imgui.set_clipboard_backend(MacClipboard);
        #[cfg(target_arch = "wasm32")]
        imgui.set_clipboard_backend(WebClipboard);
        let renderer = PlatformRenderer::new(&window, &mut imgui, self.scale_factor);

        self.window = Some(window);
        self.imgui = Some(imgui);
        self.renderer = Some(renderer);

        let cli_dirs: Vec<String> = self.pending_files.iter()
            .filter(|p| std::path::Path::new(p).is_dir())
            .cloned().collect();
        let cli_cache_dir = if cli_dirs.len() == 1 && !cli_dirs[0].ends_with(".tvcache") {
            Some(crate::loader::cache_dir_for_folder(&cli_dirs[0]))
        } else { None };
        let (rank_groups, standalone) = crate::loader::detect_rank_groups(&self.pending_files);
        let mut pi = 0;
        for group in rank_groups {
            if pi >= self.state.panes.len() {
                self.state.panes.push(Pane::new());
            }
            if cli_dirs.len() == 1 {
                self.state.panes[pi].reload_dir = Some(cli_dirs[0].clone());
                self.state.panes[pi].cache_dir = cli_cache_dir.clone();
            }
            self.state.panes[pi].open_multi(group);
            pi += 1;
        }
        for path in standalone {
            if pi >= self.state.panes.len() {
                self.state.panes.push(Pane::new());
            }
            if cli_dirs.len() == 1 {
                self.state.panes[pi].reload_dir = Some(cli_dirs[0].clone());
                self.state.panes[pi].cache_dir = cli_cache_dir.clone();
            }
            self.state.panes[pi].open(path);
            pi += 1;
        }
        self.pending_files.clear();
    }

    // Fired when the browser-side `resize` listener installed in `resumed()`
    // hops back onto the winit event loop through the proxy. Re-reads the
    // current viewport and asks winit to match it; winit's own resize
    // machinery (ResizeObserver -> WindowEvent::Resized, handled below) takes
    // it from there, same as a real OS window resize on native.
    #[cfg(target_arch = "wasm32")]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {
        if let (Some(window), Some(win)) = (self.window.as_ref(), web_sys::window()) {
            let w = win.inner_width().ok().and_then(|v| v.as_f64()).unwrap_or(0.0);
            let h = win.inner_height().ok().and_then(|v| v.as_f64()).unwrap_or(0.0);
            if w > 0.0 && h > 0.0 {
                let _ = window.request_inner_size(winit::dpi::LogicalSize::new(w, h));
            }
        }

        let files = std::mem::take(&mut *self.pending_web_files.borrow_mut());
        if !files.is_empty() {
            // A dropped folder's files land here as one flat batch (see
            // walk_dropped_entry) — group same-run rank files (e.g.
            // "...-rank-0.json.gz", "...-rank-1.json.gz") into a single
            // merged multi-rank pane instead of opening each rank as its
            // own separate trace, matching native's drag-drop behavior.
            let (rank_groups, standalone) = crate::loader::group_by_rank_bytes(files);
            for group in rank_groups {
                let empty = self.state.panes.iter().position(|p| !p.has_trace() && p.loading.is_none());
                let target = empty.unwrap_or_else(|| self.state.add_pane());
                self.state.active = target;
                self.state.pending_active_tab = Some(target);
                self.state.panes[target].open_multi_from_bytes(group);
            }
            for (name, bytes) in standalone {
                let empty = self.state.panes.iter().position(|p| !p.has_trace() && p.loading.is_none());
                let target = empty.unwrap_or_else(|| self.state.add_pane());
                self.state.active = target;
                self.state.pending_active_tab = Some(target);
                self.state.panes[target].open_from_bytes(name, bytes);
            }
            if let Some(w) = &self.window { w.request_redraw(); }
        }

        if let Some(text) = self.pending_paste.borrow_mut().take() {
            if let Some(imgui) = self.imgui.as_mut() {
                let io = imgui.io_mut();
                for ch in text.chars() {
                    if ch >= ' ' && ch != '\x7f' {
                        io.add_input_character(ch);
                    }
                }
            }
            if let Some(w) = &self.window { w.request_redraw(); }
        }

        if let Some(msg) = self.pending_src_error.borrow_mut().take() {
            let empty = self.state.panes.iter().position(|p| !p.has_trace() && p.loading.is_none());
            let target = empty.unwrap_or(self.state.active);
            self.state.panes[target].error = Some(msg);
            if let Some(w) = &self.window { w.request_redraw(); }
        }

        // Each downloaded chunk wakes this via the proxy purely to get a
        // redraw of the "Downloading: X / Y" placeholder text — there's no
        // state here to drain/consume, just a live progress value to reflect.
        if self.pending_download_progress.borrow().is_some() {
            if let Some(w) = &self.window { w.request_redraw(); }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _wid: WindowId, event: WindowEvent) {
        let imgui = match self.imgui.as_mut() {
            Some(ctx) => ctx,
            None => return,
        };
        let io = imgui.io_mut();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(r) = &self.renderer {
                    r.resize(size.width, size.height);
                }
                let s = self.scale_factor as f32;
                io.display_size = [size.width as f32 / s, size.height as f32 / s];
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor = scale_factor;
                io.display_framebuffer_scale = [scale_factor as f32; 2];
            }

            WindowEvent::CursorMoved { position, .. } => {
                let s = self.scale_factor as f32;
                let x = position.x as f32 / s;
                io.add_mouse_pos_event([x, position.y as f32 / s]);
                self.last_mouse_x = x;
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let btn = match button {
                    winit::event::MouseButton::Left => imgui::MouseButton::Left,
                    winit::event::MouseButton::Right => imgui::MouseButton::Right,
                    winit::event::MouseButton::Middle => imgui::MouseButton::Middle,
                    _ => return,
                };
                io.add_mouse_button_event(btn, state == ElementState::Pressed);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let (h, v) = match delta {
                    MouseScrollDelta::LineDelta(h, v) => (h * LINE_SCROLL_PX, v * LINE_SCROLL_PX),
                    MouseScrollDelta::PixelDelta(p) => (p.x as f32, p.y as f32),
                };
                self.scroll_accum[0] += h;
                self.scroll_accum[1] += v;
                io.add_mouse_wheel_event([h / LINE_SCROLL_PX, v / LINE_SCROLL_PX]);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if let Some(key) = winit_to_imgui(code) {
                        io.add_key_event(key, event.state == ElementState::Pressed);
                    }
                    let nav_bit = match code {
                        KeyCode::KeyW => NAV_W,
                        KeyCode::KeyA => NAV_A,
                        KeyCode::KeyS => NAV_S,
                        KeyCode::KeyD => NAV_D,
                        KeyCode::ArrowUp => NAV_UP,
                        KeyCode::ArrowDown => NAV_DOWN,
                        KeyCode::ArrowLeft => NAV_LEFT,
                        KeyCode::ArrowRight => NAV_RIGHT,
                        _ => 0,
                    };
                    if nav_bit != 0 {
                        if event.state == ElementState::Pressed {
                            if self.nav_keys == 0 {
                                self.last_frame = Instant::now();
                            }
                            self.nav_keys |= nav_bit;
                        } else {
                            self.nav_keys &= !nav_bit;
                        }
                    }
                    if event.state == ElementState::Pressed && (self.mod_ctrl || self.mod_super) {
                        if code == KeyCode::KeyA {
                            self.state.panes[self.state.active].select_all_pending = true;
                        }
                        if code == KeyCode::KeyC {
                            if let Some(text) = self.state.panes[self.state.active].copy_selection_text() {
                                #[cfg(not(target_arch = "wasm32"))]
                                MacClipboard.set(&text);
                                #[cfg(target_arch = "wasm32")]
                                WebClipboard.set(&text);
                            }
                        }
                    }
                }
                if event.state == ElementState::Pressed && !self.mod_super && !self.mod_ctrl {
                    if let Some(text) = &event.text {
                        for ch in text.chars() {
                            if ch >= ' ' && ch != '\x7f' {
                                io.add_input_character(ch);
                            }
                        }
                    }
                }
            }

            WindowEvent::ModifiersChanged(mods) => {
                let s = mods.state();
                self.mod_ctrl = s.control_key();
                self.mod_super = s.super_key();
                self.mod_shift = s.shift_key();
                io.add_key_event(imgui::Key::LeftCtrl, s.control_key());
                io.add_key_event(imgui::Key::ModCtrl, s.control_key());
                io.add_key_event(imgui::Key::LeftShift, s.shift_key());
                io.add_key_event(imgui::Key::ModShift, s.shift_key());
                io.add_key_event(imgui::Key::LeftAlt, s.alt_key());
                io.add_key_event(imgui::Key::ModAlt, s.alt_key());
                io.add_key_event(imgui::Key::LeftSuper, s.super_key());
                io.add_key_event(imgui::Key::ModSuper, s.super_key());
            }

            WindowEvent::Focused(_) => {}

            WindowEvent::PinchGesture { delta, .. } => {
                self.pinch_accum += delta as f32;
            }

            WindowEvent::Touch(Touch { phase, location, id, .. }) => {
                let s = self.scale_factor as f32;
                let pos = (location.x as f32 / s, location.y as f32 / s);
                match phase {
                    TouchPhase::Started => {
                        self.active_touches.insert(id, pos);
                        if self.active_touches.len() == 1 {
                            self.primary_touch_id = Some(id);
                            io.add_mouse_pos_event([pos.0, pos.1]);
                            io.add_mouse_button_event(imgui::MouseButton::Left, true);
                            self.last_mouse_x = pos.0;
                        } else if self.active_touches.len() == 2 {
                            // A second finger means this is a pinch, not a
                            // drag — release the synthesized button so it
                            // doesn't keep panning/selecting underneath the
                            // zoom gesture.
                            if self.primary_touch_id.take().is_some() {
                                io.add_mouse_button_event(imgui::MouseButton::Left, false);
                            }
                            self.last_pinch_dist = self.two_touch_distance();
                        }
                    }
                    TouchPhase::Moved => {
                        self.active_touches.insert(id, pos);
                        if self.active_touches.len() == 1 && self.primary_touch_id == Some(id) {
                            io.add_mouse_pos_event([pos.0, pos.1]);
                            self.last_mouse_x = pos.0;
                        } else if self.active_touches.len() == 2 {
                            if let Some(dist) = self.two_touch_distance() {
                                if let Some(last) = self.last_pinch_dist {
                                    if last > 0.0 {
                                        self.pinch_accum += dist / last - 1.0;
                                    }
                                }
                                self.last_pinch_dist = Some(dist);
                            }
                        }
                    }
                    TouchPhase::Ended | TouchPhase::Cancelled => {
                        self.active_touches.remove(&id);
                        if self.primary_touch_id == Some(id) {
                            self.primary_touch_id = None;
                            io.add_mouse_button_event(imgui::MouseButton::Left, false);
                        }
                        if self.active_touches.len() < 2 {
                            self.last_pinch_dist = None;
                        }
                        // If a pinch just ended and exactly one finger is
                        // still down, resume dragging from where it already
                        // is instead of requiring a fresh tap.
                        if self.active_touches.len() == 1 && self.primary_touch_id.is_none() {
                            let (&remaining_id, &remaining_pos) = self.active_touches.iter().next().unwrap();
                            self.primary_touch_id = Some(remaining_id);
                            io.add_mouse_pos_event([remaining_pos.0, remaining_pos.1]);
                            io.add_mouse_button_event(imgui::MouseButton::Left, true);
                            self.last_mouse_x = remaining_pos.0;
                        }
                    }
                }
            }

            WindowEvent::DroppedFile(path) => {
                self.pending_drops.push(path.to_string_lossy().into());
            }

            WindowEvent::RedrawRequested => {
                #[cfg(target_os = "macos")]
                let pool = unsafe { objc_autoreleasePoolPush() };
                self.render_frame();
                #[cfg(target_os = "macos")]
                unsafe { objc_autoreleasePoolPop(pool); }
                return;
            }

            _ => return,
        }

        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let any_watching = self.state.panes.iter().any(|p| p.auto_reload);
        if any_watching && self.last_reload.elapsed().as_secs_f32() >= 2.0 {
            for p in &mut self.state.panes {
                if p.auto_reload && p.loading.is_none() && p.trace.is_some() {
                    p.reload();
                }
            }
            self.last_reload = Instant::now();
        }
        // On web, the canvas reports a 0x0 size for a frame or two after
        // creation — winit's real size only arrives once its ResizeObserver
        // fires, asynchronously. Keep polling until a real size shows up, or
        // a Wait-mode app would render one empty frame and then never redraw
        // again (see WindowEvent::Resized below and WebGl2Renderer::render's
        // zero-size early-return).
        #[cfg(target_arch = "wasm32")]
        let awaiting_real_size = self.window.as_ref()
            .is_some_and(|w| { let s = w.inner_size(); s.width == 0 || s.height == 0 });
        #[cfg(not(target_arch = "wasm32"))]
        let awaiting_real_size = false;

        let needs_poll = self.state.panes.iter().any(|p| p.loading.is_some())
            || self.nav_keys != 0
            || self.nav_pan_vel.abs() > 1e-6
            || self.nav_zoom_vel.abs() > 1e-6
            || any_watching
            || awaiting_real_size
            || self.state.panes.iter().any(|p| p.view.anim.is_some());
        if needs_poll {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        } else {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
        }
    }
}

impl App {
    /// Distance between the two active touches, or `None` unless exactly
    /// two are down. Pairs them by touch id (not raw `HashMap` iteration
    /// order, which isn't guaranteed stable) so frame-to-frame distance
    /// comparisons in the `WindowEvent::Touch` handler always diff the same
    /// two physical fingers.
    fn two_touch_distance(&self) -> Option<f32> {
        if self.active_touches.len() != 2 { return None; }
        let mut pts: Vec<(u64, (f32, f32))> = self.active_touches.iter().map(|(&k, &v)| (k, v)).collect();
        pts.sort_by_key(|(id, _)| *id);
        let (a, b) = (pts[0].1, pts[1].1);
        Some(((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt())
    }

    fn render_frame(&mut self) {
        let imgui = match self.imgui.as_mut() {
            Some(ctx) => ctx,
            None => return,
        };
        let window = self.window.as_ref().unwrap();
        let phys = window.inner_size();
        let s = self.scale_factor as f32;

        let io = imgui.io_mut();
        io.display_size = [phys.width as f32 / s, phys.height as f32 / s];
        io.display_framebuffer_scale = [s, s];
        let now = Instant::now();
        let dt = self.last_frame.elapsed().as_secs_f32().max(0.0001);
        io.delta_time = dt;
        self.last_frame = now;

        if !self.pending_drops.is_empty() {
            let drops = std::mem::take(&mut self.pending_drops);
            let dropped_dirs: Vec<String> = drops.iter()
                .filter(|p| std::path::Path::new(p).is_dir())
                .cloned().collect();
            let drop_cache_dir = if dropped_dirs.len() == 1 && !dropped_dirs[0].ends_with(".tvcache") {
                Some(crate::loader::cache_dir_for_folder(&dropped_dirs[0]))
            } else { None };
            let (rank_groups, standalone) = crate::loader::detect_rank_groups(&drops);
            for group in rank_groups {
                let empty = self.state.panes.iter().position(|p| !p.has_trace() && p.loading.is_none());
                let target = empty.unwrap_or_else(|| self.state.add_pane());
                self.state.active = target;
                self.state.pending_active_tab = Some(target);
                if dropped_dirs.len() == 1 {
                    self.state.panes[target].reload_dir = Some(dropped_dirs[0].clone());
                    self.state.panes[target].cache_dir = drop_cache_dir.clone();
                }
                self.state.panes[target].open_multi(group);
            }
            for path in standalone {
                let empty = self.state.panes.iter().position(|p| !p.has_trace() && p.loading.is_none());
                let target = empty.unwrap_or_else(|| self.state.add_pane());
                self.state.active = target;
                self.state.pending_active_tab = Some(target);
                if dropped_dirs.len() == 1 {
                    self.state.panes[target].reload_dir = Some(dropped_dirs[0].clone());
                    self.state.panes[target].cache_dir = drop_cache_dir.clone();
                }
                self.state.panes[target].open(path);
            }
        }

        for p in &mut self.state.panes {
            p.poll_loading();
            #[cfg(not(target_arch = "wasm32"))]
            p.poll_share_upload();
        }

        let scroll = self.scroll_accum;
        self.scroll_accum = [0.0; 2];
        let pinch = self.pinch_accum;
        self.pinch_accum = 0.0;
        let ctrl = self.mod_ctrl || self.mod_super;
        let shift = self.mod_shift;
        let nav_keys = self.nav_keys;

        let ui = imgui.new_frame();
        let display = ui.io().display_size;
        let mouse_pos = ui.io().mouse_pos;
        let mouse_delta = ui.io().mouse_delta;

        // Measured at the end of last frame's toolbar window (see below) —
        // its box is now content-fit, not the old fixed TOOLBAR_H, so
        // anything positioned relative to "below the toolbar" needs the
        // real number or a few px of leftover box height shows up as a gap.
        let toolbar_h = self.toolbar_h;
        let state = &mut self.state;

        // ---- Tab strip: one tab per open trace ----
        let mut close_pane: Option<usize> = None;
        {
            let _pad = ui.push_style_var(StyleVar::WindowPadding([8.0, 4.0]));
            // Dear ImGui floors every non-child, non-auto-resize window at
            // style.WindowMinSize (32x32 by default) regardless of an
            // explicit `.size()` — without this override a 24px-tall strip
            // gets silently grown to 32px and overlaps whatever's below it
            // (see CalcWindowSizeAfterConstraint).
            let _min = ui.push_style_var(StyleVar::WindowMinSize([1.0, 1.0]));
            ui.window("##tracetabs")
                .position([0.0, toolbar_h], Condition::Always)
                .size([display[0], TAB_BAR_H], Condition::Always)
                .flags(
                    WindowFlags::NO_DECORATION
                        | WindowFlags::NO_MOVE
                        | WindowFlags::NO_SAVED_SETTINGS
                        | WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS,
                )
                .build(|| {
                    if let Some(_bar) = ui.tab_bar("##tracetabbar") {
                        let pending = state.pending_active_tab.take();
                        for pi in 0..state.panes.len() {
                            let tab_id = format!("{}##tab{}", tab_label(&state.panes[pi]), pi);
                            let mut open = true;
                            let flags = if pending == Some(pi) {
                                imgui::TabItemFlags::SET_SELECTED
                            } else { imgui::TabItemFlags::empty() };
                            if let Some(_t) = imgui::TabItem::new(&tab_id).opened(&mut open).flags(flags).begin(&ui) {
                                state.active = pi;
                            }
                            if !open {
                                close_pane = Some(pi);
                            }
                        }
                    }
                });
        }
        if let Some(pi) = close_pane {
            if state.panes.len() > 1 {
                state.remove_pane(pi);
            } else {
                state.panes[0] = Pane::new();
            }
            window.request_redraw();
        }

        let ai = state.active;
        let has_trace = state.panes[ai].has_trace();
        let bottom_h = if has_trace { state.bottom_h } else { 0.0 };
        let status_h = if has_trace { STATUS_H } else { 0.0 };
        let content_top = toolbar_h + TAB_BAR_H;

        let mut t_section = Instant::now();
        macro_rules! mark {
            ($name:expr, $t:expr) => {
                let e = $t.elapsed().as_secs_f64() * 1000.0;
                if e > 20.0 { eprintln!("  {}:{:.0} ", $name, e); }
                $t = Instant::now();
            }
        }

        // ---- Drag handling (bottom divider, label divider) ----
        let selecting = (shift && ui.io().mouse_down[0]) || state.panes[ai].selection_dirty;
        if has_trace && !state.diff_popup_open {
            let divider_y = display[1] - bottom_h - status_h;
            // Small area around the lock icon itself: a plain single click
            // there also toggles pin, on top of double-clicking anywhere
            // else on the divider — same raw hit-test technique, excluded
            // from the resize-drag strip so clicking the icon can't also
            // start a resize.
            let near_icon = (mouse_pos[0] - display[0] * 0.5).abs() < LOCK_ICON_RADIUS * 2.2
                && (mouse_pos[1] - divider_y).abs() < DIVIDER_GRAB_PX;
            let near_h = !selecting && !near_icon
                && (mouse_pos[1] - divider_y).abs() < DIVIDER_GRAB_PX && mouse_pos[1] > content_top;

            // Double-clicking the divider (or a single click right on the
            // icon) toggles "even spacing" for the active pane — every
            // visible track scales to evenly fill the space from the ruler
            // down to this divider (see draw_timeline's even_spacing
            // handling), so it reads as the tracks "auto-fitting" to the
            // available height. This is the same toggle already reachable
            // by double-clicking the last track row/empty space beneath it;
            // the icon just makes it discoverable, especially in the merged
            // multi-rank view where a handful of rank rows otherwise leave a
            // lot of dead space below them.
            let toggle_fit = (near_h && ui.is_mouse_double_clicked(imgui::MouseButton::Left))
                || (near_icon && ui.is_mouse_clicked(imgui::MouseButton::Left));
            if toggle_fit {
                let pane = &mut state.panes[ai];
                pane.even_spacing = !pane.even_spacing;
            }
            if near_icon {
                ui.set_mouse_cursor(Some(imgui::MouseCursor::Hand));
                let tip = if state.panes[ai].even_spacing {
                    "Tracks are fit to the visible height — click to go back to their normal (scrollable) sizes."
                } else {
                    "Fit all tracks to evenly fill the space down to this divider — handy for the merged view, where a few rank rows otherwise leave a lot of dead space. Double-clicking the divider does the same thing."
                };
                ui.tooltip_text(tip);
            }

            if (near_h && !state.drag.is_active()) || state.drag == DragKind::BottomDivider {
                ui.set_mouse_cursor(Some(imgui::MouseCursor::ResizeNS));
            }

            if ui.io().mouse_down[0] && !state.drag.is_active() && near_h {
                state.drag = DragKind::BottomDivider;
            }

            if ui.io().mouse_down[0] && state.drag == DragKind::BottomDivider {
                state.bottom_h -= mouse_delta[1];
                state.bottom_h = state.bottom_h.clamp(MIN_BOTTOM_H, display[1] - content_top - status_h - MIN_BOTTOM_H);
            } else if state.drag == DragKind::BottomDivider && !ui.io().mouse_down[0] {
                state.drag = DragKind::None;
            }
        }

        // Label divider (active pane only)
        if !state.diff_popup_open && has_trace {
            let divider_y = display[1] - bottom_h - status_h;
            let mut near_v = false;
            if !selecting {
                let label_x = state.panes[ai].label_w;
                near_v = (mouse_pos[0] - label_x).abs() < DIVIDER_GRAB_PX
                    && mouse_pos[1] > content_top && mouse_pos[1] < divider_y;
            }

            if (near_v && !state.drag.is_active()) || matches!(state.drag, DragKind::LabelDivider(_)) {
                ui.set_mouse_cursor(Some(imgui::MouseCursor::ResizeEW));
            }
            if ui.io().mouse_down[0] && !state.drag.is_active() && near_v {
                state.drag = DragKind::LabelDivider(ai);
            }
            if let DragKind::LabelDivider(pi) = state.drag {
                if ui.io().mouse_down[0] {
                    state.panes[pi].label_w += mouse_delta[0];
                    let max_w = display[0] * 0.5;
                    state.panes[pi].label_w = state.panes[pi].label_w.clamp(MIN_LABEL_W, max_w);
                } else {
                    state.drag = DragKind::None;
                }
            }
        }

        mark!("drag", t_section);
        // ---- Toolbar (active pane) ----
        let mut search_changed = false;
        let mut diff_clicked_against: Option<usize> = None;
        let active_has_sel = !state.panes[ai].selection_stats.is_empty();
        let diff_targets: Vec<(usize, String)> = if active_has_sel {
            (0..state.panes.len())
                .filter(|&pi| pi != ai && !state.panes[pi].selection_stats.is_empty())
                .map(|pi| (pi, tab_label(&state.panes[pi]).to_string()))
                .collect()
        } else {
            Vec::new()
        };
        {
            let _pad = ui.push_style_var(StyleVar::WindowPadding([8.0, 6.0]));
            // Auto-fit height (0.0) instead of the old fixed TOOLBAR_H: the
            // tab strip sits directly below this window now, so its box must
            // match its actual content height exactly or the leftover slack
            // between the menu bar and the window's bottom edge reads as a
            // gap above the tabs. WindowMinSize override for the same reason
            // as ##tracetabs — a content-fit menu bar can be under 32px.
            let _min = ui.push_style_var(StyleVar::WindowMinSize([1.0, 1.0]));
            ui.window("##toolbar")
                .position([0.0, 0.0], Condition::Always)
                .size([display[0], 0.0], Condition::Always)
                .flags(
                    WindowFlags::NO_DECORATION
                        | WindowFlags::NO_MOVE
                        | WindowFlags::NO_SAVED_SETTINGS
                        | WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS
                        | WindowFlags::MENU_BAR,
                )
                .build(|| {
                    self.toolbar_h = ui.window_size()[1];
                    let pane = &mut state.panes[ai];
                    if pane.trace.is_some() {
                        if let Some(_mb) = ui.begin_menu_bar() {
                            // ---- File menu: export/share/sync. Export/Share
                            // are native-only (no filesystem, no `gh` CLI on
                            // wasm); Sync Clocks works everywhere — sources
                            // rank/filename pairs from either reload_paths
                            // (native directory opens) or the trace's own
                            // persisted rank_paths (works after loading an
                            // already-merged .tvcache, native or web — see
                            // loader::sync_multi_rank_clocks).
                            #[cfg(not(target_arch = "wasm32"))]
                            let has_export_share = !pane.reload_paths.is_empty()
                                && pane.trace.as_ref().is_some_and(|t| t.tracks.iter().any(|tr| tr.gpu));
                            let multi_rank = pane.reload_paths.len() >= 2
                                || pane.trace.as_ref().is_some_and(|t| t.rank_paths.len() >= 2);
                            #[cfg(not(target_arch = "wasm32"))]
                            if has_export_share {
                                ui.menu("File", || {
                                    let uploading = pane.share_link_job.is_some();
                                    let include_cpu = pane.share_include_cpu;
                                    if ui.menu_item_config("Share via Gist").enabled(!uploading).build() {
                                        pane.start_share_upload(include_cpu);
                                    }
                                    if ui.is_item_hovered() {
                                        let what = if include_cpu { "the whole trace (CPU and GPU, args intact)" } else { "GPU-only timings (no args)" };
                                        ui.tooltip_text(&format!("Export {what}, then upload it to a new secret GitHub gist via your locally logged-in `gh` CLI, and show a copyable ?gist= link — no browser login or pasted token, but it does need `gh auth login` already done on this machine"));
                                    }
                                    ui.menu_item_config("Include CPU data##sharecpu").build_with_ref(&mut pane.share_include_cpu);
                                    if ui.is_item_hovered() {
                                        ui.tooltip_text("Upload the whole trace (all CPU and GPU tracks, args left intact, not stripped) instead of just GPU-only timings — much larger, but complete");
                                    }
                                    match &pane.share_link_result {
                                        Some((true, link)) if link.starts_with("https://") => {
                                            ui.text_colored([0.4, 0.85, 0.4, 1.0], link);
                                            ui.same_line();
                                            if ui.small_button("Copy##sharelink") {
                                                #[cfg(not(target_arch = "wasm32"))]
                                                MacClipboard.set(link);
                                            }
                                        }
                                        Some((ok, msg)) => {
                                            let color = if *ok { [0.8, 0.8, 0.4, 1.0] } else { [0.9, 0.4, 0.4, 1.0] };
                                            ui.text_colored(color, msg);
                                        }
                                        None => {}
                                    }
                                });
                            }

                            // ---- View menu: display toggles + cross-pane actions.
                            ui.menu("View", || {
                                ui.menu_item_config("Show CPU trace").build_with_ref(&mut pane.show_cpu);
                                ui.menu_item_config("Merge Streams").build_with_ref(&mut pane.merge_gpu);
                                // vLLM traces emit a per-generation `execute_context_N(N)_generation_M(M)`
                                // span on every stream — one toggle hides/shows them all. Only shown
                                // when the trace actually contains such names (computed once at load,
                                // in `poll_loading`, not recomputed every toolbar frame).
                                if !pane.exec_context_names.is_empty() {
                                    let all_hidden = pane.exec_context_names.iter()
                                        .all(|&i| pane.hidden_names.get(i).copied().unwrap_or(false));
                                    let label = if all_hidden { "Show Execute Context" } else { "Hide Execute Context" };
                                    if ui.menu_item(label) {
                                        for &i in &pane.exec_context_names {
                                            if let Some(h) = pane.hidden_names.get_mut(i) { *h = !all_hidden; }
                                        }
                                        if !pane.search.is_empty() { pane.rebuild_search(); }
                                    }
                                }
                                if pane.reload_dir.is_some() || !pane.reload_paths.is_empty() {
                                    ui.menu_item_config("Watch").build_with_ref(&mut pane.auto_reload);
                                }
                                if multi_rank {
                                    ui.separator();
                                    if ui.menu_item("Sync Clocks") {
                                        pane.sync_message = Some(match pane.sync_clocks() {
                                            Ok(msg) => (true, msg),
                                            Err(e) => (false, format!("Sync failed: {e}")),
                                        });
                                    }
                                    if ui.is_item_hovered() {
                                        ui.tooltip_text("Correct inter-node clock skew across DP groups (in memory only) — aligns each DP group's first DeepEP combine kernel, matching sync_traces.py");
                                    }
                                    if let Some((ok, msg)) = &pane.sync_message {
                                        let color = if *ok { [0.4, 0.85, 0.4, 1.0] } else { [0.9, 0.4, 0.4, 1.0] };
                                        ui.text_colored(color, msg.lines().next().unwrap_or(msg));
                                        if ui.is_item_hovered() && msg.contains('\n') {
                                            ui.tooltip_text(msg);
                                        }
                                    }
                                }
                                if !diff_targets.is_empty() {
                                    ui.separator();
                                    ui.menu("Diff against", || {
                                        for (other, label) in &diff_targets {
                                            if ui.menu_item(label) {
                                                diff_clicked_against = Some(*other);
                                            }
                                        }
                                    });
                                }
                            });

                            // ---- Help menu: keyboard/mouse reference.
                            {
                                ui.menu("Help", || {
                                    ui.text("Navigation");
                                    ui.separator();
                                    ui.text("W / Up            Zoom in (at view center)");
                                    ui.text("S / Down          Zoom out");
                                    ui.text("A / Left          Pan left");
                                    ui.text("D / Right         Pan right");
                                    ui.text("Home              Fit whole trace to view");
                                    ui.text("Scroll            Scroll tracks up / down");
                                    ui.text("Shift+Scroll      Pan left / right");
                                    ui.text("Ctrl+Scroll       Zoom at cursor");
                                    ui.separator();
                                    ui.text("Search");
                                    ui.separator();
                                    ui.text("/  or  Ctrl+F     Focus search box");
                                    ui.text("Enter             Select & frame all matches");
                                    ui.text("N                 Jump to next match");
                                    ui.text("Shift+N           Jump to previous match");
                                    ui.text("Escape            Clear search & selection");
                                    ui.separator();
                                    ui.text("Selection");
                                    ui.separator();
                                    ui.text("Click event       Select + show detail & flow arrows");
                                    ui.text("Double-click ev.  Highlight all same-name events");
                                    ui.text("Shift+Drag        Select a time range");
                                    ui.text("Right-click ev.   Copy kernel name to clipboard");
                                    ui.text("Ctrl+C            Copy selected kernel name");
                                    ui.separator();
                                    ui.text("Tracks");
                                    ui.separator();
                                    ui.text("Drag label        Reorder tracks");
                                    ui.text("Drag border       Resize track height");
                                    ui.text("Dbl-click below   Even-fill lane heights (toggle)");
                                    ui.separator();
                                    ui.text("Tabs");
                                    ui.separator();
                                    ui.text("Drop file         Open in a new tab");
                                    ui.text("Click tab         Switch trace");
                                    ui.text("Click tab's x     Close trace");
                                    ui.text("View > Diff against   Compare two tabs' selections");
                                    ui.separator();
                                    ui.text("Files");
                                    ui.separator();
                                    ui.text("Drop file         Open trace");
                                    ui.text("Drop folder       Merge multi-rank traces");
                                });
                            }

                            // ---- Search: stays inline in the menu bar itself
                            // (not a dropdown) since it's used constantly.
                            if pane.search_focus {
                                ui.set_keyboard_focus_here();
                                pane.search_focus = false;
                            }
                            ui.set_next_item_width(SEARCH_W);
                            let prev_len = pane.prev_search.len();
                            struct SelectAllCb<'a>(&'a mut bool);
                            impl InputTextCallbackHandler for SelectAllCb<'_> {
                                fn on_always(&mut self, mut data: TextCallbackData) {
                                    if *self.0 { data.select_all(); *self.0 = false; }
                                }
                            }
                            let enter = ui.input_text("##search", &mut pane.search)
                                .hint("Search (/) ")
                                .flags(imgui::InputTextFlags::ENTER_RETURNS_TRUE
                                    | imgui::InputTextFlags::AUTO_SELECT_ALL)
                                .callback(InputTextCallback::ALWAYS, SelectAllCb(&mut pane.select_all_pending))
                                .build();
                            if pane.search.len() != prev_len || pane.search != pane.prev_search {
                                search_changed = true;
                                pane.prev_search.clear();
                                pane.prev_search.push_str(&pane.search);
                            }
                            if enter && !pane.search.trim().is_empty() {
                                if search_changed {
                                    pane.rebuild_search();
                                    search_changed = false;
                                }
                                pane.select_from_search(&mut state.buf);
                                pane.zoom_to_search();
                                pane.pending_tab = Some(BottomTab::Selection);
                            }
                            let search_active = pane.search_mask.iter().any(|&m| m);
                            if search_active {
                                let has_matches = !pane.search_nav.is_empty();
                                ui.same_line_with_spacing(0.0, 6.0);
                                ui.enabled(has_matches, || {
                                    if ui.small_button("<##prevmatch") { pane.nav_search(false); }
                                });
                                if ui.is_item_hovered() { ui.tooltip_text("Previous match (Shift+N)"); }
                                ui.same_line_with_spacing(0.0, 2.0);
                                ui.enabled(has_matches, || {
                                    if ui.small_button(">##nextmatch") { pane.nav_search(true); }
                                });
                                if ui.is_item_hovered() { ui.tooltip_text("Next match (N)"); }
                                ui.same_line_with_spacing(0.0, 6.0);
                                ui.align_text_to_frame_padding();
                                state.buf.fmt.clear();
                                write!(state.buf.fmt, "{} matches", pane.search_nav.len()).unwrap();
                                ui.text_colored([0.6, 0.8, 1.0, 1.0], &state.buf.fmt);
                            }
                        }
                    } else if pane.loading.is_some() {
                        ui.align_text_to_frame_padding();
                        ui.text(&pane.loading_progress_text());
                    } else {
                        ui.align_text_to_frame_padding();
                        #[cfg(target_arch = "wasm32")]
                        let placeholder = self.pending_download_progress.borrow().as_ref()
                            .map(download_progress_text)
                            .unwrap_or_else(|| "Drop a trace file here, or: tv <file.json[.gz]>".to_string());
                        #[cfg(not(target_arch = "wasm32"))]
                        let placeholder = "Drop a trace file here, or: tv <file.json[.gz]>".to_string();
                        ui.text(&placeholder);
                    }
                    if let Some(e) = &pane.error {
                        let _c = ui.push_style_color(StyleColor::Text, [1.0, 0.4, 0.4, 1.0]);
                        ui.text(e);
                    }
                });
        }

        mark!("toolbar", t_section);
        // ---- Diff trigger ----
        if let Some(other) = diff_clicked_against {
            let seq_a = state.panes[ai].extract_selection_events();
            let seq_b = state.panes[other].extract_selection_events();
            state.diff_result = Some(diff::compute_diff(&seq_a, &seq_b));
            state.diff_bar_scroll = 0.0;
            state.diff_bar_zoom = 1.0;
            state.show_diff = true;
            state.diff_pane_indices = Some([ai, other]);
        }

        // ---- Divider line (skip when diff popup covers everything) ----
        if !state.diff_popup_open && has_trace {
            let divider_y = display[1] - bottom_h - status_h;
            let dl = ui.get_foreground_draw_list();
            let near = !selecting && ((mouse_pos[1] - divider_y).abs() < DIVIDER_GRAB_PX || state.drag == DragKind::BottomDivider);
            let col = if near { col32(120, 120, 120, 255) } else { col32(60, 60, 60, 255) };
            dl.add_line([0.0, divider_y], [display[0], divider_y], col).build();
            draw_lock_icon(&dl, display[0] * 0.5, divider_y - 1.0, state.panes[ai].even_spacing);
        }

        mark!("dividers", t_section);
        // ---- Bottom panel (active pane) ----
        if state.panes[ai].has_trace() {
            let _pad = ui.push_style_var(StyleVar::WindowPadding([8.0, 6.0]));
            ui.window("##bottom")
                .position([0.0, display[1] - bottom_h - status_h], Condition::Always)
                .size([display[0], bottom_h], Condition::Always)
                .flags(
                    WindowFlags::NO_DECORATION
                        | WindowFlags::NO_MOVE
                        | WindowFlags::NO_SAVED_SETTINGS
                        | WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS,
                )
                .build(|| {
                    let pane = &mut state.panes[ai];
                    let trace = pane.trace.as_ref().unwrap();
                    if let Some(_tab_bar) = ui.tab_bar("##bottomtabs") {
                        let pending = pane.pending_tab.take();
                        let detail_flags = if pending == Some(BottomTab::Detail) {
                            imgui::TabItemFlags::SET_SELECTED
                        } else { imgui::TabItemFlags::empty() };
                        if let Some(_t) = imgui::TabItem::new("Detail").flags(detail_flags).begin(&ui) {
                            // Resolve the selection defensively: after a reload/merge
                            // the trace can have fewer tracks/events than when the
                            // EventRef was captured, so a stale index must not panic.
                            let sel_ev = pane.selected.and_then(|sel| {
                                trace.tracks.get(sel.track_idx as usize).and_then(|t| {
                                    t.events.get(sel.event_idx as usize).map(|ev| (t, ev))
                                })
                            });
                            if let Some((track, ev)) = sel_ev {
                                let name = &trace.names[ev.name as usize];
                                let is_hidden = pane.hidden_names.get(ev.name as usize).copied().unwrap_or(false);
                                state.buf.fmt.clear();
                                if is_hidden {
                                    write!(state.buf.fmt, "Show##hideev").unwrap();
                                } else {
                                    write!(state.buf.fmt, "Hide##hideev").unwrap();
                                }
                                if ui.small_button(&state.buf.fmt) {
                                    if (ev.name as usize) < pane.hidden_names.len() {
                                        pane.hidden_names[ev.name as usize] = !is_hidden;
                                    }
                                }
                                state.buf.detail_buf.clear();
                                write!(state.buf.detail_buf, "{}\n", name).unwrap();
                                state.buf.detail_buf.push_str("Dur: ");
                                write_time(&mut state.buf.detail_buf, ev.dur);
                                state.buf.detail_buf.push_str("  |  Start: +");
                                write_time(&mut state.buf.detail_buf, ev.ts);
                                write!(state.buf.detail_buf, "\nCat: {}  |  Track: {}", trace.cats[ev.cat as usize], track.label).unwrap();
                                if ev.args_off > 0 {
                                    let raw = &trace.raw_bufs[track.raw_buf_idx as usize];
                                    let off = ev.args_off as usize;
                                    if off < raw.len() {
                                        let end = skip_value(raw, off);
                                        let mut strs = Vec::new();
                                        let mut idx = FnvMap::default();
                                        let mut pairs = Vec::new();
                                        parse_args_flat(&raw[off..end], &mut strs, &mut idx, &mut pairs);
                                        if !pairs.is_empty() {
                                            state.buf.detail_buf.push('\n');
                                            for &[k, v] in &pairs {
                                                write!(state.buf.detail_buf, "\n{}: {}", strs[k as usize], strs[v as usize]).unwrap();
                                            }
                                        }
                                    }
                                }
                                let avail = ui.content_region_avail();
                                let mut text_h = avail[1];
                                if track.gpu {
                                    let drew = draw_duration_histogram(
                                        &ui, trace, pane.show_cpu, ev.name, ev.dur, &mut state.buf, DETAIL_HIST_H,
                                    );
                                    if drew {
                                        ui.spacing();
                                        text_h = ui.content_region_avail()[1];
                                    }
                                }
                                ui.input_text_multiline("##detail_text", &mut state.buf.detail_buf, [avail[0], text_h])
                                    .flags(imgui::InputTextFlags::READ_ONLY)
                                    .build();
                            } else {
                                ui.text_colored([0.5, 0.5, 0.5, 1.0], "Click an event to see details");
                            }
                        }

                        let sel_flags = if pending == Some(BottomTab::Selection) {
                            imgui::TabItemFlags::SET_SELECTED
                        } else { imgui::TabItemFlags::empty() };
                        if let Some(_t) = imgui::TabItem::new("Selection").flags(sel_flags).begin(&ui) {
                            if !pane.selection_stats.is_empty() {
                                ui.checkbox("Aggregate", &mut pane.sel_aggregate);
                                ui.same_line();
                                if ui.button("Hide Selected") {
                                    for se in &pane.selection_stats {
                                        if let Some(h) = pane.hidden_names.get_mut(se.name as usize) { *h = true; }
                                    }
                                }
                                draw_hidden_clear(&ui, 10.0, &mut pane.hidden_names, &mut state.buf.fmt);
                                ui.same_line();
                                let sel_total_count: u32 = pane.selection_stats.iter().map(|s| s.count).sum();
                                let sel_total_dur: f64 = pane.selection_stats.iter().map(|s| s.total_dur).sum();
                                state.buf.fmt.clear();
                                write!(state.buf.fmt, "{} events, ", sel_total_count).unwrap();
                                write_time(&mut state.buf.fmt, sel_total_dur);
                                write!(state.buf.fmt, " total, ").unwrap();
                                write_time(&mut state.buf.fmt, pane.sel_median);
                                write!(state.buf.fmt, " median").unwrap();
                                ui.text_colored([0.6, 0.6, 0.6, 1.0], &state.buf.fmt);
                                // Fine-grained timing (separate from the coarse "bottom"
                                // mark! bucket) so a slow Selection tab shows exactly
                                // which piece is responsible instead of one lump sum.
                                let t_hist = Instant::now();
                                draw_selection_histogram(&ui, trace, &pane.selection_stats, pane.sel_aggregate, &mut state.buf);
                                let hist_ms = t_hist.elapsed().as_secs_f64() * 1000.0;
                                if hist_ms > 10.0 { eprintln!("  histogram:{:.0} ({} entries)", hist_ms, pane.selection_stats.iter().map(|s| s.durations.len()).sum::<usize>()); }
                                ui.separator();
                                let t_tbl = Instant::now();
                                if pane.sel_aggregate {
                                    draw_stats_table(&ui, trace, &pane.sel_agg_stats, None, pane.sel_generation, &mut pane.search, &mut search_changed, &mut pane.sort_col, &mut pane.sort_asc, &mut state.buf, "##selstats");
                                } else {
                                    draw_stats_table(&ui, trace, &pane.sel_individual, Some(&pane.sel_individual_refs), pane.sel_generation, &mut pane.search, &mut search_changed, &mut pane.sort_col, &mut pane.sort_asc, &mut state.buf, "##selstats");
                                }
                                let tbl_ms = t_tbl.elapsed().as_secs_f64() * 1000.0;
                                let tbl_rows = if pane.sel_aggregate { pane.sel_agg_stats.len() } else { pane.sel_individual.len() };
                                if tbl_ms > 10.0 { eprintln!("  stats_table:{:.0} ({} rows, aggregate={})", tbl_ms, tbl_rows, pane.sel_aggregate); }
                            } else {
                                ui.text_colored([0.5, 0.5, 0.5, 1.0], "Shift+drag to select a time range");
                                draw_hidden_clear(&ui, 16.0, &mut pane.hidden_names, &mut state.buf.fmt);
                            }
                        }
                    }
                });
        }

        mark!("bottom", t_section);
        // ---- Status bar (active pane) ----
        if state.panes[ai].has_trace() {
            let pane = &state.panes[ai];
            let t = pane.trace.as_ref().unwrap();
            let _pad = ui.push_style_var(StyleVar::WindowPadding([8.0, 2.0]));
            ui.window("##status")
                .position([0.0, display[1] - status_h], Condition::Always)
                .size([display[0], status_h], Condition::Always)
                .flags(
                    WindowFlags::NO_DECORATION
                        | WindowFlags::NO_MOVE
                        | WindowFlags::NO_SAVED_SETTINGS
                        | WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS,
                )
                .build(|| {
                    let dl = ui.get_window_draw_list();
                    let win_pos = ui.window_pos();
                    let win_size = ui.window_size();
                    let text_h = ui.calc_text_size("X")[1];
                    let cy = win_pos[1] + (status_h - text_h) / 2.0;

                    state.buf.fmt.clear();
                    let fname = pane.trace_path.rsplit('/').next().unwrap_or(&pane.trace_path);
                    let ranks = rank_summary(fname, t.dist_rank, t.dist_world);
                    if ranks.is_empty() {
                        write!(state.buf.fmt, "{} | {} events | {} tracks | {:.1}ms", fname, t.total_events, t.tracks.len(), dt * 1000.0).unwrap();
                    } else {
                        write!(state.buf.fmt, "{} | {} | {} events | {} tracks | {:.1}ms", fname, ranks, t.total_events, t.tracks.len(), dt * 1000.0).unwrap();
                    }
                    dl.add_text([win_pos[0] + 8.0, cy], col32(153, 153, 153, 255), &state.buf.fmt);

                    let logo_scale = text_h / VLLM_LOGO_GRID;
                    let logo_w = VLLM_LOGO_GRID * logo_scale;
                    let mut right_x = win_pos[0] + win_size[0] - 8.0;
                    if !t.device.is_empty() {
                        let dev_size = ui.calc_text_size(&t.device);
                        right_x -= dev_size[0];
                        dl.add_text([right_x, cy], device_name_color(), &t.device);
                        right_x -= 6.0;
                        let on_size = ui.calc_text_size("on");
                        right_x -= on_size[0];
                        dl.add_text([right_x, cy], col32(120, 120, 120, 255), "on");
                        right_x -= 6.0;
                    }
                    // vLLM version/commit, when the trace carries it, sits just
                    // right of the logo: `[logo] 0.26.1rc1.dev528+g...  on  Device`.
                    if !t.vllm_version.is_empty() {
                        let ver_size = ui.calc_text_size(&t.vllm_version);
                        right_x -= ver_size[0];
                        dl.add_text([right_x, cy], col32(120, 120, 120, 255), &t.vllm_version);
                        right_x -= 6.0;
                    }
                    right_x -= logo_w;
                    draw_vllm_logo(&dl, right_x, cy, logo_scale);
                });
        }

        mark!("status", t_section);
        // ---- Timeline (active pane) ----
        let mut hover_result: Option<EventRef> = None;
        let mut click_result: Option<EventRef> = None;
        let mut new_selection: Option<Option<[f64; 4]>> = None;
        let mut double_click = false;

        {
            if !state.panes[ai].has_trace() {
                let tl_top = content_top;
                let tl_h = display[1] - content_top;
                let _pad = ui.push_style_var(StyleVar::WindowPadding([0.0, 0.0]));
                ui.window("##splash")
                    .position([0.0, tl_top], Condition::Always)
                    .size([display[0], tl_h], Condition::Always)
                    .flags(
                        WindowFlags::NO_DECORATION
                            | WindowFlags::NO_MOVE
                            | WindowFlags::NO_SAVED_SETTINGS
                            | WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS,
                    )
                    .build(|| {
                        let dl = ui.get_window_draw_list();
                        let win_pos = ui.window_pos();
                        let avail = ui.content_region_avail();
                        // Same on-screen size as before (16*6 == 32*3 == 96px):
                        // only the grid's resolution doubled, not the logo's footprint.
                        let logo_scale = 3.0;
                        let logo_w = VLLM_LOGO_GRID * logo_scale;
                        let logo_h = VLLM_LOGO_GRID * logo_scale;
                        let cx = win_pos[0] + (avail[0] - logo_w) * 0.5;
                        let mut cy = win_pos[1] + (avail[1] - logo_h) * 0.5;
                        if state.panes[ai].error.is_some() {
                            cy -= 30.0;
                        }
                        draw_vllm_logo(&dl, cx, cy, logo_scale);
                        drop(dl);
                        if let Some(e) = &state.panes[ai].error {
                            let wrap_w = (avail[0] * 0.8).min(600.0);
                            let text_size = ui.calc_text_size_with_opts(e, false, wrap_w);
                            let tx = (avail[0] - text_size[0].min(wrap_w)) * 0.5;
                            let ty = cy - win_pos[1] + logo_h + 16.0;
                            ui.set_cursor_pos([tx, ty]);
                            let _c = ui.push_style_color(StyleColor::Text, [1.0, 0.4, 0.4, 1.0]);
                            let _wrap = ui.push_text_wrap_pos_with_pos(ui.cursor_pos()[0] + wrap_w);
                            ui.text_wrapped(e);
                        }
                    });
            } else {
            let tl_top = content_top;
            let tl_h = display[1] - content_top - bottom_h - status_h;

            let _pad = ui.push_style_var(StyleVar::WindowPadding([0.0, 0.0]));
            ui.window("##timeline")
                .position([0.0, tl_top], Condition::Always)
                .size([display[0], tl_h], Condition::Always)
                .flags(
                    WindowFlags::NO_DECORATION
                        | WindowFlags::NO_MOVE
                        | WindowFlags::NO_SAVED_SETTINGS
                        | WindowFlags::NO_SCROLL_WITH_MOUSE
                        | WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS,
                )
                .build(|| {
                    let pos = ui.cursor_screen_pos();
                    let avail = ui.content_region_avail();
                    ui.invisible_button("##canvas", avail);
                    let hovered = ui.is_item_hovered();
                    let clicked = ui.is_item_clicked();
                    let double_clicked = ui.is_mouse_double_clicked(imgui::MouseButton::Left) && hovered;
                    let active = ui.is_item_active();

                    let pane = &mut state.panes[ai];
                    let trace = pane.trace.as_ref().unwrap();
                    let rect = [pos[0], pos[1], pos[0] + avail[0], pos[1] + avail[1]];
                    let (h, c, sel) = draw_timeline(
                        &ui,
                        trace,
                        &mut pane.view,
                        pane.show_cpu,
                        &mut state.buf,
                        rect,
                        ai,
                        hovered,
                        clicked,
                        active,
                        mouse_pos,
                        mouse_delta,
                        scroll,
                        pinch,
                        ctrl,
                        shift,
                        &pane.search_mask,
                        pane.selection,
                        &pane.finished_sel_events,
                        &mut pane.collapsed,
                        &pane.hidden_names,
                        pane.selected,
                        pane.multi_select_name,
                        &pane.sel_mask,
                        pane.label_w,
                        &mut pane.track_scales,
                        &mut pane.even_spacing,
                        &mut pane.geom,
                        &mut pane.track_order,
                        &mut state.drag,
                        pane.merge_gpu,
                        dt,
                        &mut pane.pending_focus,
                    );
                    hover_result = h;
                    click_result = c;
                    new_selection = sel;
                    double_click = double_clicked;
                });
            }
        }

        mark!("timeline", t_section);

        // ---- Process click/selection results (active pane) ----
        {
            if let Some(c) = click_result {
                let pane = &mut state.panes[ai];
                let trace = pane.trace.as_ref().unwrap();
                let ev = &trace.tracks[c.track_idx as usize].events[c.event_idx as usize];
                let double = double_click;
                pane.multi_select_name = if double { Some(ev.name) } else { None };
                pane.selected = Some(c);
                // Clicking a single event to inspect it (Detail tab) must not
                // blow away an active region selection's Selection-tab table —
                // that used to happen unconditionally here, so switching back
                // to Selection after clicking any event elsewhere showed
                // nothing. A double-click still replaces the Selection tab's
                // contents via rebuild_multi_select_stats below.
                if double {
                    // Double-click replaces any drag-region selection with a
                    // by-name multi-select, so clear the region's visual
                    // rectangle (rebuild_multi_select_stats overwrites the
                    // Selection tab's data itself, but not this highlight).
                    pane.selection = None;
                    pane.finished_sel = None;
                    pane.sel_mask.clear();
                    // Double-click selects every event of this name: populate the
                    // Selection tab with that kernel's aggregate + distribution.
                    pane.rebuild_multi_select_stats();
                    pane.pending_tab = Some(BottomTab::Selection);
                } else {
                    pane.pending_tab = Some(BottomTab::Detail);
                }
            }

            if let Some(sel) = new_selection {
                let pane = &mut state.panes[ai];
                pane.selection = sel;
                if sel.is_some() {
                    pane.selected = None;
                    pane.multi_select_name = None;
                    pane.selection_dirty = true;
                    // A region shift-drag supersedes an active search: clear the
                    // search box and its highlight (once, at drag start).
                    if !pane.search.is_empty() || pane.search_mask.iter().any(|&m| m) {
                        pane.clear_search();
                    }
                } else {
                    pane.clear_selection();
                }
            }

            if state.panes[ai].selection_dirty && !ui.io().mouse_down[0] {
                state.panes[ai].finish_selection(&mut state.buf);
                state.panes[ai].pending_tab = Some(BottomTab::Selection);
                state.panes[ai].selection_dirty = false;
            }
        }

        let needs_extra_frame = click_result.is_some() || new_selection.is_some();
        if needs_extra_frame {
            window.request_redraw();
        }

        mark!("clicks", t_section);
        // ---- Hover tooltip (active pane) ----
        if !state.diff_popup_open {
            if let Some(r) = &hover_result {
                if let Some(trace) = &state.panes[ai].trace {
                    let track = &trace.tracks[r.track_idx as usize];
                    let ev = &track.events[r.event_idx as usize];
                    ui.tooltip(|| {
                        ui.text(&trace.names[ev.name as usize]);
                        state.buf.fmt.clear();
                        state.buf.fmt.push_str("Dur: ");
                        write_time(&mut state.buf.fmt, ev.dur);
                        ui.text(&state.buf.fmt);
                        state.buf.fmt.clear();
                        state.buf.fmt.push_str("Start: +");
                        write_time(&mut state.buf.fmt, ev.ts);
                        ui.text(&state.buf.fmt);
                        state.buf.fmt.clear();
                        write!(state.buf.fmt, "Cat: {}", trace.cats[ev.cat as usize]).unwrap();
                        ui.text(&state.buf.fmt);
                        state.buf.fmt.clear();
                        write!(state.buf.fmt, "Track: {}", track.label).unwrap();
                        ui.text(&state.buf.fmt);
                    });
                }
            }
        }

        mark!("tooltip", t_section);
        // ---- Diff window ----
        state.diff_popup_open = false;
        if state.show_diff {
            let mut opened = true;
            let size_cond = if state.diff_result.is_some() { Condition::Once } else { Condition::Never };
            if let Some(_token) = ui.window("Diff")
                .size([display[0] * 0.7, display[1] * 0.75], size_cond)
                .position([display[0] * 0.15, display[1] * 0.12], size_cond)
                .resizable(true)
                .collapsible(false)
                .opened(&mut opened)
                .flags(WindowFlags::NO_SCROLLBAR | WindowFlags::NO_SCROLL_WITH_MOUSE)
                .begin()
            {
                state.diff_popup_open = true;
                if let Some(diff) = &state.diff_result {
                    let [da, db] = state.diff_pane_indices.unwrap_or([0, 1]);
                    let na = state.panes.get(da).map(|p| std::path::Path::new(&p.trace_path)
                        .file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()).unwrap_or_default();
                    let nb = state.panes.get(db).map(|p| std::path::Path::new(&p.trace_path)
                        .file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()).unwrap_or_default();
                    let (scroll, zoom) = draw_diff_popup(&ui, diff, &mut state.buf, display, &na, &nb,
                        state.diff_bar_scroll, state.diff_bar_zoom);
                    state.diff_bar_scroll = scroll;
                    state.diff_bar_zoom = zoom;
                }
                if ui.is_key_pressed(imgui::Key::Escape) {
                    opened = false;
                }
            }
            if !opened {
                state.show_diff = false;
            }
        }

        mark!("diff", t_section);
        // ---- Keybinds (active pane) ----
        let any_text_focused = ui.is_any_item_active();

        if ui.is_key_pressed(imgui::Key::Home) {
            if let Some(max_ts) = state.panes[ai].trace.as_ref().map(|t| t.max_ts) {
                let pad = max_ts * FIT_PAD_FRAC;
                state.panes[ai].view.t0 = -pad;
                state.panes[ai].view.t1 = max_ts + pad;
                state.panes[ai].view.scroll_y = 0.0;
                state.panes[ai].view.anim = None;
            }
        }
        if ui.is_key_pressed(imgui::Key::Escape) {
            state.panes[ai].search.clear();
            state.panes[ai].clear_selection();
            search_changed = true;
        }
        if !any_text_focused && ctrl && ui.is_key_pressed(imgui::Key::C) {
            let pane = &state.panes[ai];
            if let Some(sel) = &pane.selected {
                if let Some(trace) = &pane.trace {
                    if let Some(ev) = trace.tracks.get(sel.track_idx as usize)
                        .and_then(|t| t.events.get(sel.event_idx as usize))
                    {
                        ui.set_clipboard_text(&trace.names[ev.name as usize]);
                    }
                }
            }
        }
        {
            let nav_dt = (dt.min(0.05)) as f64;
            let accel = 40.0;
            let decel = 40.0;

            let zoom_in = nav_keys & (NAV_W | NAV_UP) != 0;
            let zoom_out = nav_keys & (NAV_S | NAV_DOWN) != 0;
            let zoom_target = if !any_text_focused && !state.diff_popup_open {
                zoom_in as i32 - zoom_out as i32
            } else { 0 } as f64;
            if zoom_target != 0.0 {
                self.nav_zoom_vel += (zoom_target - self.nav_zoom_vel) * (1.0 - (-accel * nav_dt).exp());
            } else {
                self.nav_zoom_vel *= (-decel * nav_dt).exp();
                if self.nav_zoom_vel.abs() < 1e-6 { self.nav_zoom_vel = 0.0; }
            }

            let pan_right = nav_keys & (NAV_D | NAV_RIGHT) != 0;
            let pan_left = nav_keys & (NAV_A | NAV_LEFT) != 0;
            let pan_target = if !any_text_focused && !state.diff_popup_open {
                pan_right as i32 - pan_left as i32
            } else { 0 } as f64;
            if pan_target != 0.0 {
                self.nav_pan_vel += (pan_target - self.nav_pan_vel) * (1.0 - (-accel * nav_dt).exp());
            } else {
                self.nav_pan_vel *= (-decel * nav_dt).exp();
                if self.nav_pan_vel.abs() < 1e-6 { self.nav_pan_vel = 0.0; }
            }

            let pane = &mut state.panes[ai];
            if pane.trace.is_some() {
                // A manual keyboard zoom/pan supersedes an in-flight search zoom.
                if (zoom_target != 0.0 || pan_target != 0.0) && pane.view.anim.is_some() {
                    pane.view.anim = None;
                }
                let range = pane.view.t1 - pane.view.t0;
                if self.nav_zoom_vel.abs() > 1e-6 {
                    let factor = ZOOM_STEP.powf(nav_dt * 20.0 * self.nav_zoom_vel);
                    let center = (pane.view.t0 + pane.view.t1) / 2.0;
                    pane.view.t0 = center + (pane.view.t0 - center) / factor;
                    pane.view.t1 = center + (pane.view.t1 - center) / factor;
                }
                if self.nav_pan_vel.abs() > 1e-6 {
                    let dt_pan = range * 1.5 * nav_dt * self.nav_pan_vel;
                    pane.view.t0 += dt_pan;
                    pane.view.t1 += dt_pan;
                }
            }
        }
        if !any_text_focused {
            if ui.is_key_pressed(imgui::Key::Slash) || (ctrl && ui.is_key_pressed(imgui::Key::F)) {
                state.panes[ai].search_focus = true;
            }
            if ui.is_key_pressed(imgui::Key::N) {
                state.panes[ai].nav_search(!shift);
            }
        }

        if search_changed {
            state.panes[ai].rebuild_search();
        }

        mark!("keybinds", t_section);
        let draw_data = imgui.render();
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.render(draw_data);
        }
        mark!("render", t_section);
        let frame_ms = now.elapsed().as_secs_f64() * 1000.0;
        if frame_ms > 50.0 {
            eprintln!("\nSLOW FRAME: {:.0}ms", frame_ms);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(|a| a.as_str()) == Some("--bench") {
        let bench_args: Vec<String> = args[1..].to_vec();
        let t0 = Instant::now();
        let (rank_groups, standalone) = loader::detect_rank_groups(&bench_args);
        let all_files: Vec<&str> = rank_groups.iter()
            .flat_map(|g| g.iter().map(|(_, p)| p.as_str()))
            .chain(standalone.iter().map(|p| p.as_str()))
            .collect();
        eprintln!("bench: {} files found in {:.3}s", all_files.len(), t0.elapsed().as_secs_f64());
        for f in &all_files { eprintln!("  {f}"); }

        if rank_groups.is_empty() && standalone.len() <= 1 {
            let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let path = standalone.first().map(|s| s.as_str()).unwrap_or(&bench_args[0]);
            match loader::load_trace(path, &counter, 0, None) {
                Ok(t) => eprintln!("  ok: {} events, {} tracks, {:.2}s", t.total_events, t.tracks.len(), t0.elapsed().as_secs_f64()),
                Err(e) => eprintln!("  err: {e}"),
            }
        } else {
            let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let mut all_paths: Vec<(usize, String)> = Vec::new();
            for group in &rank_groups { all_paths.extend(group.clone()); }
            for (i, p) in standalone.iter().enumerate() { all_paths.push((all_paths.len() + i, p.clone())); }

            let tpf = (std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4) / all_paths.len()).max(2);
            eprintln!("bench: {} threads per file", tpf);
            let t_load = Instant::now();
            use rayon::prelude::*;
            let results: Vec<_> = all_paths.par_iter().filter_map(|(rank, path)| {
                let r = *rank;
                let ctr = counter.clone();
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let t = Instant::now();
                    let res = loader::load_trace(path, &ctr, tpf, None);
                    (res, t.elapsed().as_secs_f64())
                })) {
                    Ok((res, elapsed)) => Some((r, res, elapsed)),
                    Err(e) => {
                        let msg = e.downcast_ref::<&str>().map(|s| s.to_string())
                            .or_else(|| e.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "unknown panic".to_string());
                        eprintln!("  rank {r} load thread panicked: {msg}");
                        None
                    }
                }
            }).collect();

            let mut traces = Vec::new();
            let mut max_file_time = 0.0f64;
            for (rank, result, elapsed) in results {
                match result {
                    Ok(t) => {
                        eprintln!("  rank {rank}: {:.2}s, {} events, {} tracks",
                            elapsed, t.total_events, t.tracks.len());
                        max_file_time = max_file_time.max(elapsed);
                        traces.push((rank, t));
                    }
                    Err(e) => eprintln!("  rank {rank}: err: {e}"),
                }
            }
            eprintln!("  parallel load: {:.2}s wall, {:.2}s slowest file",
                t_load.elapsed().as_secs_f64(), max_file_time);

            if !traces.is_empty() {
                let t_merge = Instant::now();
                let merged = loader::merge_traces(traces);
                eprintln!("  merge: {:.2}s, {} events, {} tracks",
                    t_merge.elapsed().as_secs_f64(), merged.total_events, merged.tracks.len());
            }
            eprintln!("  total: {:.2}s", t0.elapsed().as_secs_f64());
        }
        return;
    }
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(args);
    event_loop.run_app(&mut app).unwrap();
}

// Re-exporting this (rather than only calling it below) is what makes
// wasm-bindgen actually emit the `init_thread_pool`/`initThreadPool` glue and
// the worker-entry exports (`wbg_rayon_start_worker` etc.) into the compiled
// module at all — wasm-bindgen's CLI post-processing pass discovers
// `#[wasm_bindgen]` items via reachability from the crate, and a bin crate's
// `main` alone doesn't reach into an otherwise-unused dependency's items just
// by calling one function from it the way a `pub use` guarantees.
#[cfg(all(target_arch = "wasm32", feature = "mt"))]
pub use wasm_bindgen_rayon::init_thread_pool;

// `run_app` blocks the calling thread until the event loop exits, which is
// how the native app runs its whole lifetime on the main thread. That's not
// allowed on the web (there is no blocking the browser's main thread) —
// `spawn_app` instead hands the app to the browser's own event loop and
// returns immediately.
#[cfg(target_arch = "wasm32")]
fn build_and_run_app() {
    use winit::platform::web::EventLoopExtWebSys;
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(Vec::new());
    // `create_proxy` only exists on the owning `EventLoop`, not the
    // `ActiveEventLoop` passed into `resumed()` — grab it here and stash it
    // for `resumed()` to hand to the browser-resize listener it installs.
    app.event_loop_proxy = Some(event_loop.create_proxy());
    event_loop.spawn_app(app);
}

// `fn main()` on wasm32 can't just be `async fn main()` — wasm-bindgen's
// generated glue for a `[[bin]]` crate's `data-type="main"` auto-invokes the
// plain synchronous `main` export during module instantiation, with no hook
// for awaiting anything first. Rayon needs its Web Worker pool up and
// running *before* any `par_iter`/`scope`/`spawn` call, or those calls
// silently execute on the calling thread alone (rayon's wasm fallback for
// "no pool yet" is sequential, not an error) — so the entire existing
// startup sequence gets deferred into a `spawn_local` future that awaits
// `initThreadPool` first. `main` itself returns immediately after kicking
// that future off, same as it always effectively did once `spawn_app` handed
// control back to the browser's own event loop.
#[cfg(all(target_arch = "wasm32", feature = "mt"))]
fn main() {
    console_error_panic_hook::set_once();
    wasm_bindgen_futures::spawn_local(async {
        // `navigator.hardwareConcurrency` is 0/unavailable in vanishingly
        // rare embeddings (per spec it's a `f64`, not guaranteed >= 1) —
        // fall back to a small fixed pool rather than requesting a
        // zero-thread pool, which `wasm_bindgen_rayon::init_thread_pool`
        // hard-panics on.
        let threads = web_sys::window()
            .map(|w| w.navigator().hardware_concurrency())
            .filter(|n| *n >= 1.0)
            .map(|n| n as usize)
            .unwrap_or(4);
        if let Err(e) = wasm_bindgen_futures::JsFuture::from(init_thread_pool(threads)).await {
            // Not fatal: every rayon call site on wasm32 already tolerates
            // running sequentially (that was the *only* mode before this
            // phase) — worst case without a working pool is the old
            // single-threaded behavior, not a crash.
            web_sys::console::error_1(&e);
        }
        build_and_run_app();
    });
}

// Plain stable-toolchain build (scripts/build-wasm.sh / serve-wasm.sh):
// no thread pool to wait on, so this can stay fully synchronous exactly as
// before the `mt` feature existed.
#[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
fn main() {
    console_error_panic_hook::set_once();
    build_and_run_app();
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
