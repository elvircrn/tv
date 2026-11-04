/// eprintln! that survives the web build: wasm32-unknown-unknown's std
/// stdio is a hard no-op (see sys/pal's unsupported stdio — writes succeed
/// into a void), so on wasm this forwards to the browser console instead.
/// Same call sites either way: loader phase timings, slow-frame marks.
#[macro_export]
macro_rules! tlog {
    ($($arg:tt)*) => {{
        #[cfg(target_arch = "wasm32")]
        {
            web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&format!($($arg)*)));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            eprintln!($($arg)*);
        }
    }};
}

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
mod ktz;
// wasm: nothing in here is reachable (the Source tab is native-only), so
// the types are dead code on that target.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
mod source;
#[cfg(target_os = "macos")]
mod dock_menu;

use parse::{skip_value, parse_args_flat, InternTable};
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

/// Status-bar frame-time indicator padded to a fixed 7-cell field (`12.3ms`,
/// ` 1.23s`), so the unit suffix and everything after it on the bar stay put as
/// the digit count changes. Millis below a second, seconds above.
fn fmt_frame_dur(dt: f32) -> String {
    let ms = dt * 1000.0;
    if ms < 1000.0 {
        format!("{:>5.1}ms", ms)
    } else {
        format!("{:>6.2}s", ms / 1000.0)
    }
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

/// Tab-strip label: `tab_label` middle-ellipsized to `max` chars. Real vLLM
/// run-directory names run 70+ chars, which at full length makes every tab
/// wider than the toolbar and pushes the strip into its scroll arrows. The
/// head carries the model/run prefix and the tail the prefill/decode
/// suffix — the two parts that actually tell tabs apart — and the full name
/// goes in the tab's tooltip instead.
fn strip_tab_label(full: &str, max: usize) -> String {
    let n = full.chars().count();
    if n <= max { return full.to_string(); }
    let head = max * 2 / 3;
    let tail = max - head - 3; // -3 for the "..."
    let mut out = String::with_capacity(max);
    out.extend(full.chars().take(head));
    out.push_str("...");
    out.extend(full.chars().skip(n - tail));
    out
}

/// Whether the tab strip is worth a row at all — hidden for the single
/// default empty pane (nothing opened yet, or everything was closed back
/// down to it), since a lone "Untitled" tab over the splash screen is just
/// clutter. Shown as soon as there's more than one pane, or the one pane
/// has ever had something opened into it (`trace_path` is set immediately
/// in `Pane::open`/`open_multi`, before loading even finishes).
pub(crate) fn should_show_tab_strip(panes: &[Pane]) -> bool {
    panes.len() > 1 || !panes[0].trace_path.is_empty()
}

/// One row of the toolbar's editor picker: the editor's brand mark, its
/// display name, and a check mark when it's the current choice. The row is
/// a plain span-width selectable whose label is empty — icon and text are
/// then drawn at exact pixel offsets, so they line up identically whatever
/// the font's space advance is (reserving room with leading spaces in the
/// label, the usual trick, drifts between SF Mono and DejaVu). Returns true
/// when the row was clicked.
#[cfg(not(target_arch = "wasm32"))]
fn editor_picker_row(ui: &imgui::Ui, editor: &str, label: &str, selected: bool) -> bool {
    let row_h = ui.text_line_height() + 8.0;
    let icon_sz = (row_h - 8.0).min(16.0);
    let text_w = ui.calc_text_size(label)[0];
    let row_w = icon_sz + 8.0 + text_w + 26.0 + 12.0;
    let clicked = ui.selectable_config(format!("##edrow-{editor}"))
        .selected(selected)
        .size([row_w, row_h])
        .build();
    let rmin = ui.item_rect_min();
    let rmax = ui.item_rect_max();
    let cy = (rmin[1] + rmax[1]) * 0.5;
    let text_y = cy - ui.text_line_height() * 0.5;
    let dl = ui.get_window_draw_list();
    draw_editor_icon(&dl, editor, rmin[0] + 6.0, cy - icon_sz * 0.5, icon_sz);
    let text_col = if selected { col32(255, 255, 255, 255) } else { col32(220, 220, 220, 255) };
    dl.add_text([rmin[0] + 6.0 + icon_sz + 8.0, text_y], text_col, label);
    if selected {
        let cx = rmin[0] + row_w - 16.0;
        let c = col32(120, 200, 255, 255);
        dl.add_line([cx - 3.5, cy], [cx - 0.5, cy + 3.0], c).thickness(1.8).build();
        dl.add_line([cx - 0.5, cy + 3.0], [cx + 4.0, cy - 3.5], c).thickness(1.8).build();
    }
    drop(dl);
    clicked
}

/// Wires up a freshly-opened pane's `reload_dir` when its files came from a
/// directory argument (CLI or drop) — `dir` is `None` for bare file
/// arguments, which have nothing to watch.
fn set_reload_dir(pane: &mut Pane, dir: Option<&str>) {
    if let Some(dir) = dir {
        pane.reload_dir = Some(dir.to_string());
    }
}

/// Formats one arg's value for the Detail tab's free-text panel. A nested
/// JSON object (CUPTI's `occupancy` dict is the common one — limitingFactors,
/// blockLimitRegs, etc.) is expanded one key per indented line instead of
/// being dumped as a single compact-JSON line, which otherwise runs off the
/// edge of the panel: Dear ImGui's `InputTextMultiline` has no word-wrap
/// option (it's a text editor widget, not a label), so a line that's too
/// wide is just invisible past the right edge rather than wrapping.
/// Anything that isn't a `{...}` object (numbers, strings, arrays) is
/// written through unchanged. Recursion is capped at a few levels as a
/// defensive bound, not because args actually nest that deep in practice.
fn format_arg_value(buf: &mut String, val: &str, depth: usize) {
    if depth < 4 && val.starts_with('{') && val.ends_with('}') {
        let mut strs = InternTable::new();
        let mut pairs = Vec::new();
        parse_args_flat(val.as_bytes(), &mut strs, &mut pairs);
        if !pairs.is_empty() {
            for &[k, v] in &pairs {
                buf.push('\n');
                for _ in 0..=depth { buf.push_str("  "); }
                write!(buf, "{}: ", strs.str(k)).unwrap();
                format_arg_value(buf, strs.str(v), depth + 1);
            }
            return;
        }
    }
    buf.push_str(val);
}

struct App {
    window: Option<Window>,
    imgui: Option<imgui::Context>,
    renderer: Option<PlatformRenderer>,
    state: AppState,
    last_frame: Instant,
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
    /// Any imgui popup/menu open at the end of the last frame. Feeds
    /// `about_to_wait`'s Poll decision: imgui auto-resizes a newly opened
    /// popup over its first frames (AutoFitFramesX/Y), and this loop
    /// otherwise sleeps until the next OS event - so without this a menu
    /// opened by a click sat at its provisional size until the mouse
    /// moved, which read as the toolbar being unresponsive.
    popup_open: bool,
    /// When the one "settle" frame after the current input burst is due
    /// (see `INPUT_SETTLE`). `about_to_wait` sleeps until then instead of
    /// indefinitely, so widgets imgui activates during an input frame get
    /// their follow-up frame even if the mouse stops right afterwards.
    /// Native only (and `std`'s Instant, which is what winit's WaitUntil
    /// takes): the browser drives the wasm loop with its own frame clock.
    #[cfg(not(target_arch = "wasm32"))]
    input_settle: Option<std::time::Instant>,
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
    // Populated during a `?src=` download (see `fetch_bytes_with_progress`)
    // so the empty-pane placeholder can show a live "Downloading: X / Y" line
    // instead of the page looking frozen for as long as the fetch takes.
    // Cleared once the fetch settles either way (bytes land in
    // `pending_web_files`, or an error lands in `pending_src_error`).
    #[cfg(target_arch = "wasm32")]
    pending_download_progress: std::rc::Rc<std::cell::RefCell<Option<DownloadProgress>>>,
}

impl App {
    fn new(files: Vec<String>) -> Self {
        // Zero-setup source cross-referencing: if the machine has a vLLM
        // checkout at the conventional place (~/vllm), prefill the Source
        // tab with it. An index holds nothing until the first jump, so
        // this is free.
        #[cfg(not(target_arch = "wasm32"))]
        let (src_path, src_idx) = match source::default_checkout() {
            Some((p, idx)) => (p, Some(idx)),
            None => (String::new(), None),
        };
        #[cfg(target_arch = "wasm32")]
        let (src_path, src_idx) = (String::new(), None);
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
                tabs_w: 0.0,
                drag: DragKind::None,
                show_diff: false,
                diff_popup_open: false,
                diff_result: None,
                diff_bar_scroll: 0.0,
                diff_bar_zoom: 1.0,
                diff_pane_indices: None,
                kernel_ctx_menu: None,
                src_path,
                src_editor: "gram".to_string(),
                src_idx,
                #[cfg(not(target_arch = "wasm32"))]
                src_clone: None,
                #[cfg(not(target_arch = "wasm32"))]
                src_clone_done: None,
                #[cfg(not(target_arch = "wasm32"))]
                src_clone_fail: None,
                src_pinned: false,
                src_build: String::new(),
            },
            last_frame: Instant::now(),
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
            popup_open: false,
            #[cfg(not(target_arch = "wasm32"))]
            input_settle: None,
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

    /// Opens the argv trace paths into panes: directory arguments expand to
    /// multi-rank groups, bare files to standalone panes. Pure fs + channels
    /// + `rayon::spawn` — no window/renderer/imgui needed — so `main` calls
    /// it BEFORE `EventLoop::new` (etc.) and the parse overlaps all the GUI
    /// init instead of waiting for `resumed` to finish it (measured ~130ms
    /// of macOS/ImGui/renderer setup on an M4 Pro that loading used to sit
    /// behind). `resumed` calls it again as a no-op fallback (it drains
    /// `pending_files`), which is what actually opens files on wasm, where
    /// the launch paths arrive after the loop is already running.
    fn open_launch_files(&mut self) {
        let (rank_groups, standalone) = crate::loader::detect_rank_groups(&self.pending_files);
        let mut pi = 0;
        for (group, source_dir) in rank_groups {
            if pi >= self.state.panes.len() {
                self.state.panes.push(Pane::new());
            }
            set_reload_dir(&mut self.state.panes[pi], source_dir.as_deref());
            self.state.panes[pi].open_multi(group);
            pi += 1;
        }
        for (path, source_dir) in standalone {
            if pi >= self.state.panes.len() {
                self.state.panes.push(Pane::new());
            }
            set_reload_dir(&mut self.state.panes[pi], source_dir.as_deref());
            self.state.panes[pi].open(path);
            pi += 1;
        }
        self.pending_files.clear();
    }
}

/// How long after the last input event the settle frame is drawn. imgui
/// hides a window for one frame when it is created or reactivated (new
/// tooltips and reopened popups set `HiddenFramesCannotSkipItems`, and
/// auto-sized popups measure on that frame), so UI requested during an
/// input frame only becomes visible on the frame after it. Under
/// `ControlFlow::Wait` that frame waits for the next OS event - e.g. a
/// tooltip needed a mouse nudge after you stopped on a button. Waiting
/// just past the last event draws exactly that one follow-up frame. Long
/// enough that a burst of trackpad moves has finished (each event pushes
/// the deadline out), short enough not to read as a tooltip delay.
#[cfg(not(target_arch = "wasm32"))]
const INPUT_SETTLE: std::time::Duration = std::time::Duration::from_millis(40);

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
        // 0 bytes and no total yet = the fetch-headers phase
        // before the first chunk arrives (DNS/TLS/headers can take a
        // second on a cold connection) — "0.0 MB" would read as broken.
        _ if p.received == 0 => "Downloading trace...".to_string(),
        _ => format!("Downloading: {:.1} MB", mb(p.received)),
    }
}

/// Bytes received so far and, if the server sent `Content-Length`, the
/// total — the total is `None` for chunked/unknown-length responses, in
/// which case the UI shows a byte count instead of a percentage.
#[derive(Clone, Copy)]
pub struct DownloadProgress {
    pub received: u64,
    pub total: Option<u64>,
}

/// How the web boot overlay's single bar (web/initializer.mjs) splits the
/// whole link-to-timeline load into phases, in percent. The initializer
/// owns the first slice while the wasm binary downloads; `update_boot_overlay`
/// continues the same bar through the trace download and parse, and removes
/// the overlay when nothing is left in flight. Must stay in sync with
/// WASM_SHARE in web/initializer.mjs.
#[cfg(target_arch = "wasm32")]
mod boot_weights {
    pub const WASM: f32 = 25.0;
    pub const DOWNLOAD: f32 = 50.0;
    pub const PARSE: f32 = 25.0;
}

/// Fetches `url` and returns its response body as bytes — used for the
/// `?src=<url>` shareable-link feature (see `resumed()`).
/// `url` must serve permissive CORS headers (`Access-Control-Allow-Origin`)
/// since this is a cross-origin `fetch()` from the page's own JS context; a
/// repo's `raw.githubusercontent.com` file does this
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
/// bytes) for every trace-like (.ktz) file found.
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
                        // Fallback for drops whose items don't expose
                        // webkitGetAsEntry(): a real OS drag always yields
                        // entries, but a scripted/re-dispatched DataTransfer
                        // (e.g. an embedding page forwarding a drop into this
                        // app) returns null there. Like the entry scan above,
                        // the File objects must be grabbed here, synchronously,
                        // while the DataTransfer is still valid.
                        let flat_files: Vec<web_sys::File> = if entries.is_empty() {
                            dt.files().map(|files| {
                                (0..files.length()).filter_map(|i| files.get(i)).collect()
                            }).unwrap_or_default()
                        } else {
                            Vec::new()
                        };
                        let pending = pending.clone();
                        let proxy = proxy.clone();
                        wasm_bindgen_futures::spawn_local(async move {
                            let mut files = Vec::new();
                            for entry in entries {
                                files.extend(walk_dropped_entry(entry).await);
                            }
                            for file in flat_files {
                                let name = file.name();
                                if !crate::loader::is_trace_file(&name) { continue; }
                                let Ok(buf) = wasm_bindgen_futures::JsFuture::from(file.array_buffer()).await else { continue };
                                files.push((name, js_sys::Uint8Array::new(&buf).to_vec()));
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

        // Shareable-link loading: a `?src=<url>` query param (e.g. a repo
        // raw-file URL — anything serving permissive CORS headers) is
        // fetched once at startup and dropped into the same
        // `pending_web_files` queue a real file drop would use, so it's
        // routed into a pane by the exact same code in `user_event` below.
        // A failed fetch (bad URL, 404, CORS rejection) has no bytes to
        // hand that queue, so it reports through `pending_src_error`
        // instead.
        #[cfg(target_arch = "wasm32")]
        if let Some(proxy) = self.event_loop_proxy.clone() {
            if let Some(win) = web_sys::window() {
                let search = win.location().search().unwrap_or_default();
                let params = web_sys::UrlSearchParams::new_with_str(&search).ok();
                let src = params.as_ref().and_then(|p| p.get("src"));
                if let Some(url) = src {
                    let pending = self.pending_web_files.clone();
                    let pending_err = self.pending_src_error.clone();
                    let pending_progress = self.pending_download_progress.clone();
                    // Announce the download before the first byte moves: the
                    // fetch's own progress ticks only fire once response
                    // chunks arrive (~1s of DNS/TLS/header time on a cold
                    // connection) — without this the splash is a bare,
                    // silent logo for that whole stretch, which reads as a
                    // hang.
                    *pending_progress.borrow_mut() = Some(DownloadProgress { received: 0, total: None });
                    wasm_bindgen_futures::spawn_local(async move {
                        let progress_proxy = proxy.clone();
                        let progress_slot = pending_progress.clone();
                        let on_progress = move |p: DownloadProgress| {
                            *progress_slot.borrow_mut() = Some(p);
                            let _ = progress_proxy.send_event(());
                        };
                        let result = fetch_bytes_with_progress(&win, &url, Some(&on_progress))
                            .await.map(|bytes| (url, bytes));
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

        self.open_launch_files();
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
            // "...-rank-0.pt.trace.json.ktz", "...-rank-1.pt.trace.json.ktz")
            // into a single merged multi-rank pane instead of opening each
            // rank as its own separate trace, matching native's drag-drop
            // behavior.
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

            // Leaving the window sends no further CursorMoved, so without
            // this the item under the cursor stayed hovered - highlight and
            // tooltip frozen on screen - until the next event. The invalid
            // position is imgui's own "mouse disappeared" idiom; it also
            // zeroes MouseDelta for the first frame back so the re-entry
            // doesn't read as a huge jump (imgui.cpp, the IsMousePosValid
            // guard above the MouseDelta assignment).
            WindowEvent::CursorLeft { .. } => {
                io.add_mouse_pos_event([-f32::MAX, -f32::MAX]);
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
                // A dropped directory is not a trace: treat it as the source
                // checkout for the Source cross-reference (drag a vllm
                // checkout onto the window to use it). Files go through the
                // trace pipeline.
                #[cfg(not(target_arch = "wasm32"))]
                if path.is_dir() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name == "vllm" || path.join("vllm").is_dir() {
                            self.state.src_path = path.to_string_lossy().into_owned();
                            self.state.src_idx = Some(Box::new(source::SourceIndex::new(path)));
                            self.state.src_pinned = true;
                            if let Some(pane) = self.state.panes.get_mut(self.state.active) {
                                pane.src_msg = Some((SrcNote::Ok,
                                    format!("source checkout: {}", self.state.src_path)));
                            }
                            return;
                        }
                    }
                }
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

        // Every event handled above is input, and each one pushes the settle
        // frame back: the follow-up is drawn once the burst ends, not
        // between its events. `std`'s Instant because that is the one
        // `ControlFlow::WaitUntil` compares against.
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.input_settle = Some(std::time::Instant::now() + INPUT_SETTLE);
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

        // A finished exact-build clone: stash the outcome for the next
        // frame (the rejump needs ui access) — and keep the loop in Poll
        // while one runs, exactly like a trace load, so a Wait-mode loop
        // doesn't sleep past the clone finishing.
        #[cfg(not(target_arch = "wasm32"))]
        let src_cloning = self.state.src_clone.is_some();
        #[cfg(target_arch = "wasm32")]
        let src_cloning = false;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(task) = self.state.src_clone.take() {
            if task.handle.is_finished() {
                let res = task.handle.join()
                    .unwrap_or_else(|_| Err("clone thread panicked".to_string()));
                self.state.src_clone_done = Some((task.rejump, task.build_id, res));
            } else {
                self.state.src_clone = Some(task);
            }
        }

        let needs_poll = self.state.panes.iter().any(|p| p.loading.is_some())
            || self.nav_keys != 0
            || self.nav_pan_vel.abs() > 1e-6
            || self.nav_zoom_vel.abs() > 1e-6
            || any_watching
            || awaiting_real_size
            || self.popup_open
            || self.state.panes.iter().any(|p| p.view.anim.is_some())
            || src_cloning;
        if needs_poll {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        } else {
            // Wake for the settle frame just after the input stops, then
            // draw it and go back to sleeping until the next event.
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(deadline) = self.input_settle {
                if std::time::Instant::now() < deadline {
                    event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(deadline));
                    return;
                }
                self.input_settle = None;
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
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

    /// Drives the HTML boot overlay (web/initializer.mjs) that a share-link
    /// open (?src=) keeps alive across the whole load, so the user
    /// sees ONE bar from link to timeline instead of an app bar that resets
    /// when the trace download begins. The initializer owns the wasm slice
    /// (0-25%); this continues the same bar through the trace download
    /// (25-75%, by Content-Length) and the parse (75-100%, by the stepped
    /// plan's phase progress), and removes the overlay once nothing is in
    /// flight — the frame that removes it is the frame the timeline (or the
    /// error splash, drawn by the canvas) first composits. Called every
    /// frame from render_frame; a plain app open has no overlay left to
    /// find and this is a no-op. A free function (not a method) so
    /// render_frame's live `imgui` field borrow still splits. Weights:
    /// `boot_weights` (must match the initializer's WASM_SHARE).
    #[cfg(target_arch = "wasm32")]
    fn update_boot_overlay(
        state: &AppState,
        pending_dl: &std::rc::Rc<std::cell::RefCell<Option<DownloadProgress>>>,
        pending_files: &std::rc::Rc<std::cell::RefCell<Vec<(String, Vec<u8>)>>>,
    ) {
        use wasm_bindgen::JsCast;
        let doc = match web_sys::window().and_then(|w| w.document()) {
            Some(d) => d,
            None => return,
        };
        let overlay = match doc.get_element_by_id("tv-loading") {
            Some(e) => e,
            None => return,
        };

        // What's still in flight, in bar order: the download itself, the
        // staging gap between fetch completion and the file landing in a
        // pane (hold at the download boundary), then any pane's parse/load.
        let dl = *pending_dl.borrow();
        let parse = if dl.is_some() {
            None
        } else if !pending_files.borrow().is_empty() {
            // Files staged, user_event hasn't routed them into a pane yet.
            Some(None)
        } else {
            state.panes.iter().filter_map(|p| p.boot_parse_progress()).next()
        };

        let (pct, label, indeterminate) = if let Some(p) = dl {
            match p.total.filter(|t| *t > 0) {
                Some(total) => (
                    boot_weights::WASM + boot_weights::DOWNLOAD
                        * (p.received as f32 / total as f32).min(1.0),
                    None,
                    false,
                ),
                // 0 bytes and no total yet = response headers
                // still pending — hold where the wasm slice ended.
                None if p.received == 0 => (boot_weights::WASM, None, false),
                None => (boot_weights::WASM, Some(format!(
                    "Loading trace... {:.1} MB", p.received as f64 / (1024.0 * 1024.0),
                )), true),
            }
        } else if let Some(progress) = parse {
            match progress {
                Some(f) => (
                    (boot_weights::WASM + boot_weights::DOWNLOAD + boot_weights::PARSE * f)
                        .min(99.0),
                    None,
                    false,
                ),
                None => (boot_weights::WASM + boot_weights::DOWNLOAD, None, false),
            }
        } else {
            // Nothing in flight: whatever final state exists (trace, error)
            // is already behind the overlay — hand the screen over.
            if let Some(parent) = overlay.parent_node() {
                let _ = parent.remove_child(&overlay);
            }
            return;
        };

        let label = label.unwrap_or_else(|| format!("Loading trace... {:.0}%", pct));
        if let Some(el) = doc.get_element_by_id("tv-boot-fill") {
            // class_list is Element's; style needs the HtmlElement cast.
            let _ = el.class_list().toggle_with_force("indeterminate", indeterminate);
            if let Some(he) = el.dyn_ref::<web_sys::HtmlElement>() {
                let _ = he.style().set_property("width", &format!("{:.0}%", pct));
            }
        }
        if let Some(el) = doc.get_element_by_id("tv-boot-label") {
            el.set_text_content(Some(&label));
        }
    }

    fn render_frame(&mut self) {
        let imgui = match self.imgui.as_mut() {
            Some(ctx) => ctx,
            None => return,
        };
        let window = self.window.as_ref().unwrap();
        let phys = window.inner_size();
        let s = self.scale_factor as f32;
        // The 2x-size atlas font (see renderer.rs), for oversized labels.
        let big_font = self.renderer.as_ref().map(|r| r.big_font);

        let io = imgui.io_mut();
        io.display_size = [phys.width as f32 / s, phys.height as f32 / s];
        io.display_framebuffer_scale = [s, s];
        let now = Instant::now();
        let dt = self.last_frame.elapsed().as_secs_f32().max(0.0001);
        io.delta_time = dt;
        self.last_frame = now;

        if !self.pending_drops.is_empty() {
            let drops = std::mem::take(&mut self.pending_drops);
            let (rank_groups, standalone) = crate::loader::detect_rank_groups(&drops);
            for (group, source_dir) in rank_groups {
                let empty = self.state.panes.iter().position(|p| !p.has_trace() && p.loading.is_none());
                let target = empty.unwrap_or_else(|| self.state.add_pane());
                self.state.active = target;
                self.state.pending_active_tab = Some(target);
                set_reload_dir(&mut self.state.panes[target], source_dir.as_deref());
                self.state.panes[target].open_multi(group);
            }
            for (path, source_dir) in standalone {
                let empty = self.state.panes.iter().position(|p| !p.has_trace() && p.loading.is_none());
                let target = empty.unwrap_or_else(|| self.state.add_pane());
                self.state.active = target;
                self.state.pending_active_tab = Some(target);
                set_reload_dir(&mut self.state.panes[target], source_dir.as_deref());
                self.state.panes[target].open(path);
            }
        }

        for p in &mut self.state.panes {
            // Single-threaded wasm: run a staged parse now, one frame after
            // it was staged (see Pane::run_pending_parse) — before
            // poll_loading, so the result lands in this same frame.
            #[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
            p.run_pending_parse();
            p.poll_loading();
        }

        // Before any UI builds: a share-link open's boot overlay comes off
        // in the same frame the trace (or its error) lands, so the first
        // thing composited without it is the timeline itself. A free
        // function taking the fields (not a &self method) so the borrow
        // splits around the live `imgui` field borrow above.
        #[cfg(target_arch = "wasm32")]
        Self::update_boot_overlay(&self.state, &self.pending_download_progress, &self.pending_web_files);

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

        // Exact height needed for the toolbar's single content row (menu bar
        // when a trace is loaded, or plain text otherwise), computed from
        // real style/font metrics instead of a guessed constant — the tab
        // strip sits directly below this, so any slack here shows up as a
        // gap above the tabs. `text_line_height` + this window's own
        // WindowPadding top+bottom (see below) covers both cases: a menu
        // bar's natural height (FontSize + FramePadding.y*2) is smaller
        // than this and just gets a little breathing room.
        let toolbar_h = ui.text_line_height() + 12.0;
        let state = &mut self.state;

        let ai = state.active;
        let has_trace = state.panes[ai].has_trace();
        let bottom_h = if has_trace { state.bottom_h } else { 0.0 };
        let status_h = if has_trace { STATUS_H } else { 0.0 };
        let content_top = toolbar_h;

        let mut t_section = Instant::now();
        macro_rules! mark {
            ($name:expr, $t:expr) => {
                let e = $t.elapsed().as_secs_f64() * 1000.0;
                if e > 20.0 { tlog!("  {}:{:.0} ", $name, e); }
                $t = Instant::now();
            }
        }

        // A finished exact-build clone lands here (stashed by
        // about_to_wait): swap it in as the checkout and re-run the jump
        // that started it. On failure, remember the ref so every jump
        // doesn't re-curl a stamp that matches no repo, then still jump
        // against whatever checkout exists (the drift warning rides it).
        #[cfg(not(target_arch = "wasm32"))]
        if let Some((rejump, build_id, res)) = state.src_clone_done.take() {
            match res {
                Ok((path, desc)) => {
                    let disp = path.to_string_lossy().into_owned();
                    state.src_idx = Some(Box::new(source::SourceIndex::new(path)));
                    state.src_pinned = false;
                    if let Some(p) = state.panes.get_mut(state.active) {
                        p.src_msg = Some((SrcNote::Ok,
                            format!("exact vllm build {} ({})", desc, disp)));
                    }
                }
                Err(e) => {
                    state.src_clone_fail = Some((build_id.clone(), e.clone()));
                    if let Some(p) = state.panes.get_mut(state.active) {
                        p.src_msg = Some((SrcNote::Warn, e));
                    }
                }
            }
            if let Some((pi, r)) = rejump {
                if pi < state.panes.len() && state.panes[pi].has_trace() {
                    let (act, warn) = resolve_event_source(&mut state.src_idx, &state.panes[pi], r);
                    let pane = &mut state.panes[pi];
                    apply_src_action(ui, &mut pane.src_msg, &mut pane.src_hits,
                        &state.src_editor, act, warn);
                }
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
                    "Tracks fill the visible height. Click to restore scrollable track sizes."
                } else {
                    "Fit all tracks between the timeline and this divider. Double-clicking the divider toggles it too."
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
            // WindowMinSize override: dear ImGui floors every non-child,
            // non-auto-resize window at style.WindowMinSize (32x32 by
            // default) regardless of an explicit `.size()` — toolbar_h can
            // be under 32px depending on font size.
            let _min = ui.push_style_var(StyleVar::WindowMinSize([1.0, 1.0]));
            ui.window("##toolbar")
                .position([0.0, 0.0], Condition::Always)
                .size([display[0].max(120.0), toolbar_h], Condition::Always)
                .flags(
                    WindowFlags::NO_DECORATION
                        | WindowFlags::NO_MOVE
                        | WindowFlags::NO_SAVED_SETTINGS
                        | WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS
                        | WindowFlags::MENU_BAR,
                )
                .build(|| {
                    let pane = &mut state.panes[ai];
                    if pane.trace.is_some() {
                        if let Some(_mb) = ui.begin_menu_bar() {
                            let multi_rank = pane.reload_paths.len() >= 2
                                || pane.trace.as_ref().is_some_and(|t| t.rank_paths.len() >= 2);

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
                                        .all(|&i| pane.hidden_names.get(i).unwrap_or(false));
                                    let label = if all_hidden { "Show Execute Context" } else { "Hide Execute Context" };
                                    if ui.menu_item(label) {
                                        for &i in &pane.exec_context_names {
                                            pane.hidden_names.set(i, !all_hidden);
                                        }
                                        if !pane.search.is_empty() { pane.rebuild_search(); }
                                    }
                                }
                                if pane.reload_dir.is_some() || !pane.reload_paths.is_empty() {
                                    ui.menu_item_config("Watch").build_with_ref(&mut pane.auto_reload);
                                }
                                // ---- Color section: how timeline events get
                                // their fill. Name-hash (identity) or a
                                // duration heat map.
                                ui.separator();
                                ui.menu("Color", || {
                                    if ui.menu_item_config("By Name").selected(!pane.color_by_dur).build() {
                                        pane.color_by_dur = false;
                                    }
                                    if ui.menu_item_config("By Duration").selected(pane.color_by_dur).build() {
                                        pane.color_by_dur = true;
                                    }
                                    if ui.is_item_hovered() {
                                        ui.tooltip_text("Heat-map events by duration (blue = short, red = long) instead of by name.\nThe scale runs from the 5th percentile of GPU kernel/memcpy durations up to the\nslowest kernel, so the warmest color is exactly the slowest one. Annotation\nspans are excluded. Fixed per trace.\nThe legend at the bottom-right of the timeline shows the actual durations.");
                                    }
                                });
                                if multi_rank {
                                    ui.separator();
                                    if ui.menu_item("Sync Clocks") {
                                        pane.sync_message = Some(match pane.sync_clocks() {
                                            Ok(msg) => (true, msg),
                                            Err(e) => (false, format!("Sync failed: {e}")),
                                        });
                                    }
                                    if ui.is_item_hovered() {
                                        ui.tooltip_text("Correct inter-node clock skew across DP groups (in memory only) by aligning each DP group's first DeepEP combine kernel. Same alignment as sync_traces.py.");
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
                                    ui.text("[ / ]             Previous / next vLLM step");
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

                            // ---- Editor picker: which CLI the "View
                            // Source" jumps launch. A toolbar entry showing
                            // the current editor's mark; the dropdown
                            // repeats the marks per row so the choice reads
                            // at a glance (all vector geometry, no image
                            // assets - see `draw_editor_icon`). The text
                            // field takes any other editor command, the
                            // same state the Source setup popup edits.
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                let label = editor_label(&state.src_editor);
                                let icon_sz = (ui.text_line_height() - 1.0).min(16.0);
                                let text_w = ui.calc_text_size(label)[0];
                                let btn_w = 7.0 + icon_sz + 6.0 + text_w + 8.0 + 9.0 + 7.0;
                                let clicked = ui.button_with_size(
                                    "##editorpick", [btn_w, ui.frame_height()]);
                                let rmin = ui.item_rect_min();
                                let rmax = ui.item_rect_max();
                                let cy = (rmin[1] + rmax[1]) * 0.5;
                                {
                                    let dl = ui.get_window_draw_list();
                                    draw_editor_icon(&dl, &state.src_editor,
                                        rmin[0] + 7.0, cy - icon_sz * 0.5, icon_sz);
                                    dl.add_text(
                                        [rmin[0] + 7.0 + icon_sz + 6.0,
                                         cy - ui.text_line_height() * 0.5],
                                        col32(228, 228, 228, 255), label);
                                    // Down chevron: "this opens a picker".
                                    let ax = rmax[0] - 10.0;
                                    let ac = col32(166, 166, 166, 255);
                                    dl.add_line([ax - 3.5, cy - 1.5], [ax, cy + 2.0], ac)
                                        .thickness(1.4).build();
                                    dl.add_line([ax, cy + 2.0], [ax + 3.5, cy - 1.5], ac)
                                        .thickness(1.4).build();
                                }
                                if ui.is_item_hovered() {
                                    ui.tooltip_text("Editor used by View Source jumps.\nClick to pick one; any other editor command can be typed in the picker.");
                                }
                                if clicked { ui.open_popup("##editorpick"); }
                                // The popup belongs to this window's ID
                                // stack (the menu bar pushes "##menubar"),
                                // so open and draw it here rather than
                                // outside; anchoring it under the button
                                // beats ImGui's default "at the mouse".
                                unsafe {
                                    imgui_sys::igSetNextWindowPos(
                                        [rmin[0], rmax[1] + 2.0].into(),
                                        Condition::Appearing as i32,
                                        [0.0, 0.0].into(),
                                    );
                                }
                                ui.popup("##editorpick", || {
                                    for e in KNOWN_EDITORS {
                                        if editor_picker_row(ui, e, editor_label(e),
                                                             state.src_editor == e)
                                        {
                                            state.src_editor = e.to_string();
                                        }
                                    }
                                    ui.separator();
                                    ui.text_disabled("Other command:");
                                    ui.set_next_item_width(180.0);
                                    ui.input_text("##editorcustom", &mut state.src_editor)
                                        .hint("code-insiders, nvim, ...")
                                        .build();
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
                            let search_active = pane.search_mask.any();
                            if search_active {
                                // Live count from the load-time per-name
                                // stats — the nav index itself is built
                                // lazily on the first Enter/N/arrow use.
                                let n_matches = pane.search_match_count();
                                ui.same_line_with_spacing(0.0, 6.0);
                                ui.enabled(n_matches > 0, || {
                                    if ui.small_button("<##prevmatch") { pane.nav_search(false); }
                                });
                                if ui.is_item_hovered() { ui.tooltip_text("Previous match (Shift+N)"); }
                                ui.same_line_with_spacing(0.0, 2.0);
                                ui.enabled(n_matches > 0, || {
                                    if ui.small_button(">##nextmatch") { pane.nav_search(true); }
                                });
                                if ui.is_item_hovered() { ui.tooltip_text("Next match (N)"); }
                                ui.same_line_with_spacing(0.0, 6.0);
                                ui.align_text_to_frame_padding();
                                state.buf.fmt.clear();
                                write!(state.buf.fmt, "{} matches", n_matches).unwrap();
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
                            .unwrap_or_else(|| "Drop a trace file here, or: tv <file.ktz>".to_string());
                        #[cfg(not(target_arch = "wasm32"))]
                        let placeholder = "Drop a trace file here, or: tv <file.ktz>".to_string();
                        ui.text(&placeholder);
                    }
                    if let Some(e) = &pane.error {
                        let _c = ui.push_style_color(StyleColor::Text, [1.0, 0.4, 0.4, 1.0]);
                        ui.text(e);
                    }
                });
        }

        mark!("toolbar", t_section);

        // ---- Tab strip: one tab per open trace ----
        // Inline with the toolbar row (same y, at its right end) instead of
        // a separate row below it — saves a whole row of vertical space.
        // Hidden entirely when there's nothing open yet (or everything was
        // closed back to the single default pane) — a lone "Untitled" tab
        // for a pane with no trace just reads as clutter over the splash;
        // once has_trace() is possible for any pane the strip is worth its
        // space again, even at one real tab.
        //
        // Built AFTER the toolbar and overlaid on its right end (the
        // toolbar spans the full width underneath): sizing the toolbar
        // against the strip's width coupled the two across frames —
        // auto-resize converges one frame late, so every tab open/close/
        // relabel made the toolbar's right edge jump a beat after the
        // strip's left edge did, which read as janky. Overlaying decouples
        // them: the strip's width is now nobody else's business.
        let show_tabs = should_show_tab_strip(&state.panes);
        // The strip must never crowd the toolbar's content — File/View/Help
        // + the search box, its match-nav buttons and the "N matches" text
        // need roughly this much room themselves.
        let min_toolbar_w = 520.0f32;
        let mut close_pane: Option<usize> = None;
        let mut strip_switched = false;
        let mut tabs_w = state.tabs_w;
        if show_tabs {
            // 3px vertical padding (not 4): the tab bar's height is
            // FontSize + FramePadding.y*2 (=6), so 2*3 + that exactly equals
            // toolbar_h (text_line_height + 12) at ANY font size — the strip
            // row lines up with the toolbar row instead of overflowing it
            // (the old fixed-height window was 2px short of the tab bar's
            // real height, which clipped tabs / surfaced a scrollbar).
            let _pad = ui.push_style_var(StyleVar::WindowPadding([8.0, 3.0]));
            let _min = ui.push_style_var(StyleVar::WindowMinSize([1.0, 1.0]));
            ui.window("##tracetabs")
                // Right-glued via pivot instead of positioning at
                // (display - estimated_width): the old label+32-per-tab
                // estimate diverged from imgui's real per-tab overhead
                // (close button + paddings + the window's own padding), so
                // the fixed-size window was narrower than its content —
                // tabs got squeezed by the ResizeDown fitting policy and
                // the strip read as "not lined up from the right". An
                // auto-resizing window pivoted at its right edge is
                // exactly as wide as its content, always flush with the
                // right edge, and capped so it can't crowd the toolbar.
                // NO_BACKGROUND: the toolbar's own background shows through
                // where the strip is empty, so the tabs read as part of the
                // chrome row instead of a floating grey pill on top of it.
                .position([display[0], 0.0], Condition::Always)
                .position_pivot([1.0, 0.0])
                .size_constraints([1.0, 1.0], [(display[0] - min_toolbar_w).max(120.0), toolbar_h])
                .flags(
                    WindowFlags::NO_DECORATION
                        | WindowFlags::NO_MOVE
                        | WindowFlags::NO_SAVED_SETTINGS
                        // NO_BRING_TO_FRONT_ON_FOCUS is deliberately NOT set
                        // here (unlike the other chrome windows): imgui
                        // z-orders windows by creation/focus order, not
                        // per-frame submission order, and ##toolbar is
                        // created on frame 1 — earlier in the array = in
                        // front. Since the strip now overlays the
                        // full-width toolbar, it must be brought to front
                        // when it appears (and on tab clicks) or the
                        // toolbar's opaque background draws straight over
                        // it and the tabs vanish.
                        | WindowFlags::NO_BACKGROUND
                        | WindowFlags::NO_SCROLLBAR
                        | WindowFlags::ALWAYS_AUTO_RESIZE,
                )
                .build(|| {
                    if let Some(_bar) = ui.tab_bar_with_flags(
                        "##tracetabbar",
                        imgui::TabBarFlags::FITTING_POLICY_SCROLL,
                    ) {
                        let pending = state.pending_active_tab.take();
                        for pi in 0..state.panes.len() {
                            let full = tab_label(&state.panes[pi]).to_string();
                            let tab_id = format!("{}##tab{}", strip_tab_label(&full, 28), pi);
                            let mut open = true;
                            let flags = if pending == Some(pi) {
                                imgui::TabItemFlags::SET_SELECTED
                            } else { imgui::TabItemFlags::empty() };
                            if let Some(_t) = imgui::TabItem::new(&tab_id).opened(&mut open).flags(flags).begin(&ui) {
                                // The strip renders after the toolbar/bottom
                                // panel, so a switch (or close) lands one
                                // frame late for them — strip_switched
                                // requests the follow-up frame that applies
                                // it everywhere.
                                if state.active != pi {
                                    state.active = pi;
                                    strip_switched = true;
                                }
                            }
                            if ui.is_item_hovered() && full.chars().count() > 28 {
                                ui.tooltip_text(&full);
                            }
                            if !open {
                                close_pane = Some(pi);
                            }
                        }
                    }
                    // Auto-resize converges over frames; `tabs_w_changed`
                    // requests the follow-up frame so the converged width
                    // isn't left stale in Wait mode.
                    tabs_w = ui.window_size()[0];
                });
        }
        // Track the strip's rendered width so a width change (tab opened/
        // closed/relabeled) requests the follow-up frame the auto-resize
        // needs to converge — the event loop is Wait-driven, so nobody else
        // will.
        let tabs_w_changed = show_tabs && (tabs_w - state.tabs_w).abs() > 0.5;
        if !show_tabs { tabs_w = 0.0; }
        state.tabs_w = tabs_w;
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
        // Step row clicked in the Steps tab (index into pane.steps), filled
        // by the bottom panel below and applied after it so frame_step's
        // view.anim isn't clobbered by the remaining view handling later in
        // the frame. gap_clicked is the same pattern for the Gaps tab
        // (index into pane.gaps).
        let mut step_clicked: Option<usize> = None;
        let mut gap_clicked: Option<usize> = None;
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
                    // No panel-wide `trace` binding: Detail (its main
                    // consumer) is the LAST tab now, and a borrow live
                    // across the later whole-pane &mut calls
                    // (ensure_sel_overlap_totals, ensure_gaps) no longer
                    // ends before them. Each tab re-borrows locally.
                    if let Some(_tab_bar) = ui.tab_bar("##bottomtabs") {
                        let pending = pane.pending_tab.take();
                        let sel_flags = if pending == Some(BottomTab::Selection) {
                            imgui::TabItemFlags::SET_SELECTED
                        } else { imgui::TabItemFlags::empty() };
                        if let Some(_t) = imgui::TabItem::new("Selection").flags(sel_flags).begin(&ui) {
                            pane.bottom_tab = BottomTab::Selection;
                            if !pane.selection_stats.is_empty() {
                                ui.checkbox("Aggregate", &mut pane.sel_aggregate);
                                ui.same_line();
                                // Overlap-aware totals only exist for
                                // aggregated rows (a per-event row is one
                                // launch), so the toggle is inert in
                                // individual mode.
                                let _dis = ui.begin_disabled(!pane.sel_aggregate);
                                ui.checkbox("Overlap aware", &mut pane.sel_overlap);
                                if ui.is_item_hovered() {
                                    ui.tooltip_text(
                                        "Total splits every instant of wall time equally\n\
                                        among the selected kernels running it, per rank:\n\
                                        a kernel always alongside one other is worth half\n\
                                        its sum, a solo kernel keeps it all. Totals sum\n\
                                        to the selection's busy wall time, so concurrency\n\
                                        is never double-counted. Mean/Median/Max/Min\n\
                                        stay per-launch.",
                                    );
                                }
                                drop(_dis);
                                ui.same_line();
                                if ui.button("Hide Selected") {
                                    for se in &pane.selection_stats {
                                        pane.hidden_names.set(se.name as usize, true);
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
                                draw_selection_histogram(&ui, pane.trace.as_ref().unwrap(), &pane.selection_stats, pane.sel_aggregate, &mut state.buf);
                                let hist_ms = t_hist.elapsed().as_secs_f64() * 1000.0;
                                if hist_ms > 10.0 { tlog!("  histogram:{:.0} ({} entries)", hist_ms, pane.selection_stats.iter().map(|s| s.durations.len()).sum::<usize>()); }
                                ui.separator();
                                let t_tbl = Instant::now();
                                if pane.sel_aggregate {
                                    // Runs after the checkbox above (so a
                                    // same-frame toggle is covered) and takes
                                    // &mut pane, so the table below re-borrows
                                    // the trace instead of using the panel-wide
                                    // `trace` binding.
                                    if pane.sel_overlap { pane.ensure_sel_overlap_totals(); }
                                    let overlap = if pane.sel_overlap { Some(&pane.sel_overlap_totals[..]) } else { None };
                                    draw_stats_table(&ui, pane.trace.as_ref().unwrap(), &pane.sel_agg_stats, None, overlap, Some(&pane.sel_agg_starts), pane.sel_generation, &mut pane.search, &mut search_changed, &mut pane.sort_col, &mut pane.sort_asc, &mut pane.sort_cache_key, &mut pane.sort_idx, &mut state.buf, "##selstats");
                                } else {
                                    draw_stats_table(&ui, pane.trace.as_ref().unwrap(), &pane.sel_individual, Some(&pane.sel_individual_refs), None, None, pane.sel_generation, &mut pane.search, &mut search_changed, &mut pane.sort_col, &mut pane.sort_asc, &mut pane.sort_cache_key, &mut pane.sort_idx, &mut state.buf, "##selstats");
                                }
                                let tbl_ms = t_tbl.elapsed().as_secs_f64() * 1000.0;
                                let tbl_rows = if pane.sel_aggregate { pane.sel_agg_stats.len() } else { pane.sel_individual.len() };
                                if tbl_ms > 10.0 { tlog!("  stats_table:{:.0} ({} rows, aggregate={})", tbl_ms, tbl_rows, pane.sel_aggregate); }
                            } else {
                                // Big centered hint: with nothing selected the
                                // tab is a blank rectangle, which reads as
                                // broken. Falls back to plain text if the 2x
                                // atlas font isn't available (no renderer yet).
                                let avail = ui.content_region_avail();
                                let cur = ui.cursor_pos();
                                let label = "Shift+drag to select events";
                                if let Some(f) = big_font {
                                    let _t = ui.push_font(f);
                                    let sz = ui.calc_text_size(label);
                                    ui.set_cursor_pos([
                                        cur[0] + ((avail[0] - sz[0]) * 0.5).max(0.0),
                                        cur[1] + ((avail[1] - sz[1]) * 0.5).max(0.0),
                                    ]);
                                    ui.text_colored([0.5, 0.5, 0.5, 1.0], label);
                                } else {
                                    ui.text_colored([0.5, 0.5, 0.5, 1.0], label);
                                }
                                // Top-left corner of the tab, like the
                                // controls row in the non-empty case. Not
                                // `draw_hidden_clear`: its `same_line` would
                                // snap the indicator to the right edge of
                                // the big hint above instead, top-aligned
                                // against a 2x line.
                                ui.set_cursor_pos(cur);
                                draw_hidden_clear_here(&ui, &mut pane.hidden_names, &mut state.buf.fmt);
                            }
                        }

                        // ---- Steps tab: per-vLLM-step summary, click a row to
                        // frame that step. Only registered as a tab when the
                        // trace actually has steps, so non-vLLM traces are
                        // unaffected.
                        if !pane.steps.is_empty() {
                            let steps_flags = if pending == Some(BottomTab::Steps) {
                                imgui::TabItemFlags::SET_SELECTED
                            } else { imgui::TabItemFlags::empty() };
                            if let Some(_t) = imgui::TabItem::new("Steps").flags(steps_flags).begin(&ui) {
                            pane.bottom_tab = BottomTab::Steps;
                                // Summary line first: totals over all steps,
                                // plus the DP-group count when the trace's
                                // batches are per-group (each DP group runs
                                // its own scheduler, so group-local batches
                                // are summed into the totals).
                                let n = pane.steps.len();
                                let total_idle: f64 = pane.steps.iter().map(|s| s.idle_before).sum();
                                let span = pane.steps.last().map(|s| s.t1).unwrap_or(0.0)
                                    - pane.steps.first().map(|s| s.t0).unwrap_or(0.0);
                                state.buf.fmt.clear();
                                write!(state.buf.fmt, "{n} steps, ").unwrap();
                                write_time(&mut state.buf.fmt, span);
                                write!(state.buf.fmt, " span, ").unwrap();
                                write_time(&mut state.buf.fmt, total_idle);
                                write!(state.buf.fmt, " idle ({:.0}%)", total_idle / span.max(1e-9) * 100.0).unwrap();
                                let n_groups = pane.steps.iter()
                                    .map(|s| s.group_batches.len()).max().unwrap_or(0);
                                if n_groups > 1 {
                                    write!(state.buf.fmt, ", {n_groups} DP groups (batch summed per group; hover cells for breakdown)").unwrap();
                                }
                                ui.text_colored([0.6, 0.6, 0.6, 1.0], &state.buf.fmt);
                                draw_steps_table(&ui, &pane.steps, &mut state.buf, &mut step_clicked);
                            }
                        }

                        // ---- Gaps tab: top GPU idle gaps per rank, annotated
                        // with what the rank's CPU threads were doing during
                        // each (launch-bound vs truly idle). Click a row to
                        // zoom to the gap. Only registered as a tab when the
                        // trace has GPU tracks; computation is lazy
                        // (ensure_gaps) so loading pays nothing for it.
                        // The gate re-borrows pane.trace rather than using the
                        // panel-wide `trace` binding — that one must not stay
                        // live across ensure_sel_overlap_totals above.
                        let has_gpu_tracks = pane.trace.as_ref()
                            .is_some_and(|t| t.tracks.iter().any(|t| t.gpu));
                        if has_gpu_tracks {
                            let gaps_flags = if pending == Some(BottomTab::Gaps) {
                                imgui::TabItemFlags::SET_SELECTED
                            } else { imgui::TabItemFlags::empty() };
                            if let Some(_t) = imgui::TabItem::new("Gaps").flags(gaps_flags).begin(&ui) {
                            pane.bottom_tab = BottomTab::Gaps;
                                // Filter row: real traces carry seconds-long
                                // profiler-artifact gaps that fill the whole
                                // top-N list. Zoom into the known-good region
                                // and press "Limit to view" to restrict
                                // detection to it; changing it invalidates
                                // the cached gaps and the row focus so they
                                // re-detect. While a selection is active it
                                // takes over the range (and scopes the
                                // ranks), so the button is hidden — it'd be
                                // silently overridden.
                                let view_range = (pane.view.t0, pane.view.t1);
                                let sel_range = pane.gap_sel_window();
                                let mut filter_changed = false;
                                // align_text_to_frame_padding: the row mixes
                                // plain text with the "Limit to view" button,
                                // and raw text sits at text-line height while
                                // framed widgets center on FramePadding —
                                // without the align, "Range:" rides high.
                                ui.align_text_to_frame_padding();
                                ui.text("Range:");
                                ui.same_line();
                                ui.align_text_to_frame_padding();
                                state.buf.fmt.clear();
                                match sel_range.or(pane.gap_filter.window) {
                                    Some((w0, w1)) => {
                                        write_time(&mut state.buf.fmt, w0);
                                        state.buf.fmt.push_str(" - ");
                                        write_time(&mut state.buf.fmt, w1);
                                        if sel_range.is_some() {
                                            state.buf.fmt.push_str(" (selection)");
                                        }
                                    }
                                    None => state.buf.fmt.push_str("whole trace"),
                                }
                                ui.text_colored([0.6, 0.6, 0.6, 1.0], &state.buf.fmt);
                                if sel_range.is_some() && ui.is_item_hovered() {
                                    ui.tooltip_text("Gaps are limited to the active selection's time range and ranks. Clear the selection to use a manual range.");
                                }
                                if sel_range.is_none() {
                                    ui.same_line();
                                    if ui.button("Limit to view") {
                                        pane.gap_filter.window = Some(view_range);
                                        filter_changed = true;
                                    }
                                    if ui.is_item_hovered() {
                                        ui.tooltip_text(
                                            "Look for gaps only inside the visible time range.");
                                    }
                                }
                                // Selection-scoped idle, per rank: when a range
                                // selection exists, each rank's empty space
                                // inside it — leftmost-to-rightmost selected
                                // kernel span minus the busy union (see
                                // selection_gpu_summary). The Gaps tab is the
                                // idle-analysis home, so it lives here, not in
                                // the Selection tab; always a table, even for
                                // one rank, so the layout doesn't jump between
                                // selections.
                                if !pane.sel_gpu.is_empty() {
                                    state.buf.fmt.clear();
                                    state.buf.fmt.push_str("In selection ");
                                    write_time(&mut state.buf.fmt, pane.sel_gpu.iter().map(|g| g.t0).fold(f64::INFINITY, f64::min));
                                    state.buf.fmt.push_str(" - ");
                                    write_time(&mut state.buf.fmt, pane.sel_gpu.iter().map(|g| g.t1).fold(f64::NEG_INFINITY, f64::max));
                                    state.buf.fmt.push(':');
                                    ui.text_colored([0.6, 0.6, 0.6, 1.0], &state.buf.fmt);
                                    use imgui::{TableColumnSetup, TableFlags};
                                    const GH: [&str; 4] = ["Rank", "GPU idle", "Span", "Idle %"];
                                    let cols: [TableColumnSetup<&str>; 4] =
                                        std::array::from_fn(|i| TableColumnSetup::new(GH[i]));
                                    let flags = TableFlags::ROW_BG
                                        | TableFlags::BORDERS_INNER_V
                                        | TableFlags::SIZING_FIXED_FIT
                                        | TableFlags::NO_SAVED_SETTINGS;
                                    if let Some(_t) = ui.begin_table_header_with_sizing(
                                        "##selgpu", cols, flags, [0.0, 0.0], 0.0,
                                    ) {
                                        for g in &pane.sel_gpu {
                                            let span = g.t1 - g.t0;
                                            let idle = span - g.busy;
                                            let pct = if span > 0.0 { idle / span * 100.0 } else { 0.0 };
                                            ui.table_next_row();
                                            set_table_column(ui, 0);
                                            match g.rank {
                                                Some(r) => ui.text(format!("rank {r}")),
                                                None => ui.text("trace"),
                                            }
                                            set_table_column(ui, 1);
                                            state.buf.fmt.clear();
                                            write_time(&mut state.buf.fmt, idle);
                                            ui.text(&state.buf.fmt);
                                            set_table_column(ui, 2);
                                            state.buf.fmt.clear();
                                            write_time(&mut state.buf.fmt, span);
                                            ui.text(&state.buf.fmt);
                                            set_table_column(ui, 3);
                                            ui.text(format!("{pct:.0}%"));
                                        }
                                    }
                                }
                                if filter_changed {
                                    pane.gaps = None;
                                    pane.focused_gap = None;
                                }
                                pane.ensure_gaps();
                                let gaps = pane.gaps.as_deref().unwrap_or(&[]);
                                state.buf.fmt.clear();
                                if gaps.is_empty() {
                                    write!(state.buf.fmt, "No GPU idle gaps >= ").unwrap();
                                    write_time(&mut state.buf.fmt, MIN_GAP_US);
                                    state.buf.fmt.push_str(" found");
                                    ui.text_colored([0.5, 0.5, 0.5, 1.0], &state.buf.fmt);
                                } else {
                                    let total: f64 = gaps.iter().map(|g| g.t1 - g.t0).sum();
                                    write!(state.buf.fmt, "{} gaps, ", gaps.len()).unwrap();
                                    write_time(&mut state.buf.fmt, total);
                                    write!(state.buf.fmt, " total idle. Per rank (streams unioned), top {} by length. Click a row to zoom.", GAP_TOP_N).unwrap();
                                    ui.text_colored([0.6, 0.6, 0.6, 1.0], &state.buf.fmt);
                                }
                                draw_gaps_table(&ui, pane.trace.as_ref().unwrap(), gaps, pane.focused_gap, &mut state.buf, &mut gap_clicked);
                            }
                        }

                        let detail_flags = if pending == Some(BottomTab::Detail) {
                            imgui::TabItemFlags::SET_SELECTED
                        } else { imgui::TabItemFlags::empty() };
                        if let Some(_t) = imgui::TabItem::new("Detail").flags(detail_flags).begin(&ui) {
                            pane.bottom_tab = BottomTab::Detail;
                            let trace = pane.trace.as_ref().unwrap();
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
                                let is_hidden = pane.hidden_names.get(ev.name as usize).unwrap_or(false);
                                state.buf.fmt.clear();
                                if is_hidden {
                                    write!(state.buf.fmt, "Show##hideev").unwrap();
                                } else {
                                    write!(state.buf.fmt, "Hide##hideev").unwrap();
                                }
                                if ui.small_button(&state.buf.fmt) {
                                    if (ev.name as usize) < pane.hidden_names.len() {
                                        pane.hidden_names.set(ev.name as usize, !is_hidden);
                                    }
                                }
                                state.buf.detail_buf.clear();
                                write!(state.buf.detail_buf, "{}\n", name).unwrap();
                                state.buf.detail_buf.push_str("Dur: ");
                                write_time(&mut state.buf.detail_buf, ev.dur);
                                state.buf.detail_buf.push_str("  |  Start: +");
                                write_time(&mut state.buf.detail_buf, ev.ts);
                                write!(state.buf.detail_buf, "\nCat: {}  |  Track: {}", trace.cats[ev.cat as usize], track.label).unwrap();
                                // Args include CallFrom (kineto with_stack):
                                // the exact python file:line a frame was
                                // called from. Rendered here, consumed by the
                                // right-click View Source resolution.
                                if ev.args_off > 0 {
                                    let raw = &trace.raw_bufs[track.raw_buf_idx as usize];
                                    let off = ev.args_off as usize;
                                    if off < raw.len() {
                                        let end = skip_value(raw, off);
                                        let mut strs = InternTable::new();
                                        let mut pairs = Vec::new();
                                        parse_args_flat(&raw[off..end], &mut strs, &mut pairs);
                                        if !pairs.is_empty() {
                                            state.buf.detail_buf.push('\n');
                                            for &[k, v] in &pairs {
                                                write!(state.buf.detail_buf, "\n{}: ", strs.str(k)).unwrap();
                                                format_arg_value(&mut state.buf.detail_buf, strs.str(v), 0);
                                            }
                                        }
                                    }
                                }
                                let avail = ui.content_region_avail();
                                let mut text_h = avail[1];
                                if track.gpu {
                                    let drew = draw_duration_histogram(
                                        &ui, trace, pane.show_cpu, ev.name, ev.dur,
                                        &mut pane.detail_hist_key, &mut pane.detail_hist_bins,
                                        &mut pane.detail_hist_min, &mut pane.detail_hist_max,
                                        &mut pane.detail_hist_mean, &mut pane.detail_hist_median,
                                        &mut state.buf, DETAIL_HIST_H,
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
                    }
                });
        }

        mark!("bottom", t_section);
        // A Steps-tab row click: move the cursor and frame that step.
        if let Some(i) = step_clicked.take() {
            let pane = &mut state.panes[ai];
            if i < pane.steps.len() {
                pane.step_cursor = i;
                pane.frame_step(i);
            }
        }
        // A Gaps-tab row click: highlight that row (focused_gap drives the
        // table selection and the timeline's red band) and zoom to the gap,
        // scrolling its GPU track into view. The (t0, t1, focus_track) triple
        // is copied out first — frame_gap takes &mut pane, which the gap
        // borrow would block.
        if let Some(i) = gap_clicked.take() {
            let pane = &mut state.panes[ai];
            pane.focused_gap = Some(i);
            let gap = pane.gaps.as_ref().and_then(|v| v.get(i))
                .map(|g| (g.t0, g.t1, g.focus_track));
            if let Some((t0, t1, ft)) = gap {
                pane.frame_gap(t0, t1, ft);
            }
        }
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
                        write!(state.buf.fmt, "{} | {} events | {} tracks | {}", fname, t.total_events, t.tracks.len(), fmt_frame_dur(dt)).unwrap();
                    } else {
                        write!(state.buf.fmt, "{} | {} | {} events | {} tracks | {}", fname, ranks, t.total_events, t.tracks.len(), fmt_frame_dur(dt)).unwrap();
                    }
                    dl.add_text([win_pos[0] + 8.0, cy], col32(153, 153, 153, 255), &state.buf.fmt);
                    // A source-jump outcome rides the status bar's left side —
                    // the one feedback place for the right-click View Source
                    // gesture: green where it landed, yellow when the
                    // checkout drifts from the trace's build, red why it
                    // didn't land at all.
                    if let Some((note, e)) = &pane.src_msg {
                        let left_w = ui.calc_text_size(&state.buf.fmt)[0];
                        let msg = e.lines().next().unwrap_or(e);
                        let c = match note {
                            SrcNote::Ok => col32(140, 200, 150, 255),
                            SrcNote::Warn => col32(225, 185, 80, 255),
                            SrcNote::Err => col32(220, 90, 90, 255),
                        };
                        dl.add_text([win_pos[0] + 8.0 + left_w + 14.0, cy], c, msg);
                    }

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
                    // Clicking it opens the source-setup popup — the version
                    // is what the setup exists to match, so the affordance
                    // lives there instead of another toolbar button.
                    if !t.vllm_version.is_empty() {
                        let ver_size = ui.calc_text_size(&t.vllm_version);
                        right_x -= ver_size[0];
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            ui.set_cursor_screen_pos([right_x, cy]);
                            ui.invisible_button("##srcver", ver_size);
                            if ui.is_item_clicked() {
                                ui.open_popup("##src_setup");
                            }
                            if ui.is_item_hovered() {
                                ui.tooltip_text("source cross-reference setup");
                            }
                        }
                        dl.add_text([right_x, cy], col32(120, 120, 120, 255), &t.vllm_version);
                        right_x -= 6.0;
                    }
                    right_x -= logo_w;
                    draw_vllm_logo(&dl, right_x, cy, logo_scale);
                });
        }

        mark!("status", t_section);
        // ---- Source cross-reference popups ----
        // No Source tab and no per-event button: the one gesture is
        // right-click -> View Source, the one feedback place is the status
        // bar. These popups exist only when a jump needs them (or the
        // status-bar version text was clicked).
        #[cfg(not(target_arch = "wasm32"))]
        {
            ui.popup("##src_setup", || {
                ui.text("Source cross-reference");
                ui.separator();
                // Version banner, computed before the input widgets take
                // their &mut borrows: the trace's build commit vs the
                // checkout's git describe. A mismatch means line numbers
                // can drift — the one way this feature silently lies.
                let trace_ver = state.panes[ai].trace.as_ref()
                    .map(|t| t.vllm_version.clone()).unwrap_or_default();
                let git_desc = state.src_idx.as_mut().and_then(|i| i.git_desc());
                if !trace_ver.is_empty() {
                    state.buf.fmt.clear();
                    match (source::trace_build_ref(&trace_ver), git_desc.as_deref()) {
                        (Some(b), Some(desc)) if b.matches_desc(desc) => {
                            write!(state.buf.fmt, "checkout matches trace build {}", b.id()).unwrap();
                            ui.text_colored([0.4, 0.8, 0.5, 1.0], &state.buf.fmt);
                        }
                        (Some(b), Some(desc)) => {
                            write!(state.buf.fmt, "trace built at {} but checkout is at {} - line numbers may drift",
                                b.id(), desc).unwrap();
                            ui.text_colored([0.9, 0.7, 0.3, 1.0], &state.buf.fmt);
                        }
                        _ => {
                            write!(state.buf.fmt, "trace vllm_version: {}", trace_ver).unwrap();
                            ui.text_colored([0.6, 0.6, 0.6, 1.0], &state.buf.fmt);
                        }
                    }
                }
                // Applying is free — nothing is indexed until a jump, so
                // switching checkouts (or branches) is instant and always
                // current. An explicit Set pins: auto-materialization
                // stands down until a different trace's build asks again.
                let mut go = false;
                ui.set_next_item_width(320.0);
                if ui.input_text("##srcpath", &mut state.src_path)
                    .hint("/path/to/vllm (or drop the folder on the window)")
                    .build() { go = true; }
                ui.same_line();
                if ui.button("Set") { go = true; }
                if go {
                    let p = std::path::PathBuf::from(&state.src_path);
                    if p.is_dir() {
                        state.src_idx = Some(Box::new(source::SourceIndex::new(p)));
                        state.src_pinned = true;
                        state.panes[ai].src_msg =
                            Some((SrcNote::Ok, format!("source checkout: {}", state.src_path)));
                        ui.close_current_popup();
                    } else {
                        state.panes[ai].src_msg =
                            Some((SrcNote::Err, format!("not a directory: {}", state.src_path)));
                    }
                }
                // Manual exact-build: for traces whose version carries a
                // pipeline stamp (matches no repo), paste the real build
                // commit or tag (from the image label, the PR, ...) and tv
                // materializes it like the automatic path would have.
                let mut fetch = false;
                ui.set_next_item_width(320.0);
                if ui.input_text("##srcbuild", &mut state.src_build)
                    .hint("exact build: commit sha or tag (e.g. 40941f2c4b, v0.11.0)")
                    .build() { fetch = true; }
                ui.same_line();
                if ui.button("Fetch") { fetch = true; }
                if fetch && state.src_clone.is_none() && !state.src_build.trim().is_empty() {
                    // A '.' means version tag (a sha never has one).
                    let t = state.src_build.trim();
                    let b = if t.contains('.') {
                        source::BuildRef::Tag(format!("v{}", t.trim_start_matches('v')))
                    } else {
                        source::BuildRef::Commit(t.to_string())
                    };
                    let bid = b.id().to_string();
                    state.src_clone = Some(SrcCloneTask {
                        handle: std::thread::spawn(move || source::materialize_build(&b)),
                        rejump: None,
                        build_id: bid.clone(),
                    });
                    state.panes[ai].src_msg = Some((SrcNote::Ok,
                        format!("fetching exact vllm build {} ...", bid)));
                    ui.close_current_popup();
                }
                // The known-editor quick-pick lives in the toolbar's editor
                // button; this field is for any other CLI (and always
                // reflects that button's current choice).
                ui.set_next_item_width(90.0);
                ui.input_text("##srceditor", &mut state.src_editor)
                    .hint("editor")
                    .build();
                if ui.is_item_hovered() {
                    ui.tooltip_text("Editor command for View Source jumps. Gram, VS Code, Cursor and\nZed are wired for file:line positioning; anything else is launched\nwith `-g file:line`. Pick a known one from the toolbar's editor button.");
                }
            });
            // Multi-hit picker: a name that resolved to more than one
            // definition lists them; click a row to open file:line.
            ui.popup("##src_hits", || {
                ui.text_disabled("multiple definitions - pick one");
                ui.separator();
                let mut picked: Option<source::SourceHit> = None;
                let hits = std::mem::take(&mut state.panes[ai].src_hits);
                for h in &hits {
                    state.buf.fmt.clear();
                    write!(state.buf.fmt, "{}:{}", h.rel, h.line).unwrap();
                    if ui.selectable(&state.buf.fmt) { picked = Some(h.clone()); }
                }
                state.panes[ai].src_hits = hits;
                if let Some(h) = picked {
                    state.panes[ai].src_hits.clear();
                    ui.close_current_popup();
                    match source::open_in_editor(&state.src_editor, &h) {
                        Err(e) => state.panes[ai].src_msg = Some((SrcNote::Err, e)),
                        Ok(()) => state.panes[ai].src_msg =
                            Some((SrcNote::Ok, format!("opened {}:{}", h.rel, h.line))),
                    }
                }
            });
        }
        // ---- Timeline (active pane) ----
        let mut hover_result: Option<EventRef> = None;
        let mut click_result: Option<EventRef> = None;
        let mut new_selection: Option<Option<[f64; 4]>> = None;
        let mut double_click = false;

        {
            if !state.panes[ai].has_trace() {
                let tl_top = content_top;
                let tl_h = display[1] - content_top;
                // What the active pane is waiting on, drawn centered under
                // the logo: a ?src= download (web — normally hidden
                // behind the boot overlay, which owns the download's
                // progress bar; this text is the fallback if the overlay
                // isn't there), a staged parse (single-threaded wasm), or a
                // running load's event count.
                #[cfg(target_arch = "wasm32")]
                let dl_status = self.pending_download_progress.borrow().as_ref()
                    .map(download_progress_text);
                #[cfg(not(target_arch = "wasm32"))]
                let dl_status: Option<String> = None;
                #[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
                let parse_staged = state.panes[ai].parse_announced();
                #[cfg(not(all(target_arch = "wasm32", not(feature = "mt"))))]
                let parse_staged = false;
                let splash_status = dl_status.or_else(|| {
                    let pane = &state.panes[ai];
                    if pane.error.is_some() {
                        None
                    } else if parse_staged {
                        Some("Parsing trace...".to_string())
                    } else if pane.loading.is_some() {
                        Some(pane.loading_progress_text())
                    } else {
                        None
                    }
                });
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
                        } else if let Some(status) = &splash_status {
                            let sz = ui.calc_text_size(status);
                            let tx = (avail[0] - sz[0]) * 0.5;
                            let ty = cy - win_pos[1] + logo_h + 16.0;
                            ui.set_cursor_pos([tx, ty]);
                            ui.text_colored([0.65, 0.65, 0.65, 1.0], status);
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
                    // Heat bounds are trace-derived; ensure them before the
                    // immutable `trace` borrow below pins the pane for the
                    // rest of this block. Same for the focused gap's
                    // (t0, t1, rank), copied out for the same reason.
                    if pane.color_by_dur { pane.ensure_heat_bounds(); }
                    let focused_gap = pane.focused_gap.and_then(|i| {
                        pane.gaps.as_ref().and_then(|v| v.get(i)).map(|g| (g.t0, g.t1, g.rank))
                    });
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
                        pane.label_w,
                        &mut pane.track_scales,
                        &mut pane.even_spacing,
                        &mut pane.geom,
                        &mut pane.track_order,
                        &mut state.drag,
                        pane.merge_gpu,
                        dt,
                        &mut pane.pending_focus,
                        &mut pane.merge_cache_key,
                        &mut pane.merged_gpu_groups,
                        &pane.steps,
                        pane.step_cursor,
                        pane.color_by_dur,
                        pane.heat_bounds,
                        focused_gap,
                    );
                    hover_result = h;
                    click_result = c;
                    new_selection = sel;
                    double_click = double_clicked;

                    // Right-click a kernel instance to sync every rank's clock so
                    // that rank's own first occurrence of the SAME kernel name
                    // ends at the same time as this one — a generalized, ad-hoc
                    // version of "Sync Clocks" (which hardcodes the DeepEP combine
                    // marker) driven by whatever the user actually clicked.
                    let right_clicked = ui.is_mouse_clicked(imgui::MouseButton::Right) && hovered;
                    if right_clicked {
                        if let Some(r) = h {
                            state.kernel_ctx_menu = Some((ai, r));
                            ui.open_popup("##kernel_ctx_menu");
                        }
                    }
                    if let Some((pi, r)) = state.kernel_ctx_menu {
                        if pi == ai {
                            let name_opt: Option<String> = state.panes[ai].trace.as_ref().and_then(|t| {
                                t.tracks.get(r.track_idx as usize)
                                    .and_then(|tr| tr.events.get(r.event_idx as usize))
                                    .map(|ev| t.names[ev.name as usize].clone())
                            });
                            ui.popup("##kernel_ctx_menu", || {
                                // No name row: demangled kernel names are
                                // hundreds of chars wide, and the hover
                                // tooltip already shows the name. Menu items
                                // only, so the menu stays as narrow as its
                                // longest label.
                                let Some(name) = &name_opt else {
                                    ui.close_current_popup();
                                    return;
                                };
                                if ui.menu_item("Sync Ranks to End of This Event") {
                                    let pane = &mut state.panes[ai];
                                    pane.sync_message = Some(match pane.sync_clocks_on(name) {
                                        Ok(msg) => (true, msg),
                                        Err(e) => (false, format!("Sync failed: {e}")),
                                    });
                                }
                                // Right-click -> source: the one gesture for
                                // the whole feature. When the trace names
                                // its build commit/tag and the checkout on
                                // hand isn't it, first go get the exact
                                // tree (background clone into /tmp) and
                                // re-run this jump when it lands — jumps
                                // against a drifted checkout are how "right
                                // file, wrong line" happens. Stamped dev
                                // versions infer their commit from repo
                                // history (the dev count is a tag distance).
                                // Stands down when the user pinned a
                                // checkout, a clone is already running, or
                                // this build already failed to resolve.
                                #[cfg(not(target_arch = "wasm32"))]
                                if ui.menu_item("View Source") {
                                    let build = state.panes[ai].trace.as_ref()
                                        .and_then(|t| source::trace_build_ref(&t.vllm_version));
                                    let clone = match &build {
                                        Some(b) if !state.src_pinned
                                            && state.src_clone.is_none()
                                            && state.src_clone_fail.as_ref()
                                                .map(|(id, _)| id != &b.id()).unwrap_or(true)
                                            && !state.src_idx.as_mut()
                                                .and_then(|i| i.git_desc())
                                                .is_some_and(|d| b.matches_desc(d.as_str())) =>
                                        {
                                            let bid = b.id().to_string();
                                            state.src_clone = Some(SrcCloneTask {
                                                handle: std::thread::spawn({
                                                    let b = b.clone();
                                                    move || source::materialize_build(&b)
                                                }),
                                                rejump: Some((ai, r)),
                                                build_id: bid.clone(),
                                            });
                                            state.panes[ai].src_msg = Some((SrcNote::Ok,
                                                format!("fetching exact vllm build {} ...", bid)));
                                            true
                                        }
                                        _ => false,
                                    };
                                    if !clone && state.src_clone.is_some() {
                                        state.panes[ai].src_msg = Some((SrcNote::Ok,
                                            "still fetching exact vllm build ...".to_string()));
                                    } else if !clone {
                                        let (act, warn) = resolve_event_source(
                                            &mut state.src_idx, &state.panes[ai], r);
                                        let pane = &mut state.panes[ai];
                                        apply_src_action(
                                            ui,
                                            &mut pane.src_msg, &mut pane.src_hits,
                                            &state.src_editor, act, warn);
                                    }
                                }
                            });
                        }
                    }
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
                    if !pane.search.is_empty() || pane.search_mask.any() {
                        pane.clear_search();
                    }
                } else {
                    pane.clear_selection();
                }
            }

            if state.panes[ai].selection_dirty && !ui.io().mouse_down[0] {
                // Auto-switch to Selection only from Detail — that tab shows
                // the single clicked event, which a region selection just
                // cleared, so without the switch it looks like nothing
                // happened. Gaps/Steps/Selection are left alone: yanking the
                // user off a tab they're working in (Gaps especially, which
                // reacts to the selection itself) is its own annoyance.
                if state.panes[ai].bottom_tab == BottomTab::Detail {
                    state.panes[ai].pending_tab = Some(BottomTab::Selection);
                }
                state.panes[ai].finish_selection(&mut state.buf);
                state.panes[ai].selection_dirty = false;
            }
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
            if !state.diff_popup_open {
                if ui.is_key_pressed(imgui::Key::RightBracket) {
                    state.panes[ai].nav_step(true);
                }
                if ui.is_key_pressed(imgui::Key::LeftBracket) {
                    state.panes[ai].nav_step(false);
                }
            }
        }

        if search_changed {
            state.panes[ai].rebuild_search();
        }

        // A tab close was clicked in the strip above: remove the pane HERE,
        // at the very end of the frame — `ai` was copied before the strip
        // ran and the canvas/bottom panel/tooltip/keybind blocks between
        // still index `state.panes[ai]`, so removing any earlier (as an
        // older version did, after just the canvas) left them indexing a
        // dead slot and panicked when the closed tab was the active one.
        let closed_tab = close_pane.is_some();
        if let Some(pi) = close_pane {
            if state.panes.len() > 1 {
                state.remove_pane(pi);
            } else {
                state.panes[0] = Pane::new();
            }
        }

        let needs_extra_frame = click_result.is_some() || new_selection.is_some() || tabs_w_changed || strip_switched || closed_tab;
        if needs_extra_frame {
            window.request_redraw();
        }

        mark!("keybinds", t_section);
        let draw_data = imgui.render();
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.render(draw_data);
        }
        // Read after render(): the popup stack is still valid until the
        // next `new_frame`. Null id + AnyPopup* is the "is anything open"
        // form (see `IsPopupOpen` - it never dereferences the string id
        // when AnyPopupId is set). `about_to_wait` turns this into Poll so
        // an open menu gets the frames its auto-resize needs.
        self.popup_open = unsafe {
            imgui_sys::igIsPopupOpen(
                std::ptr::null(),
                (imgui_sys::ImGuiPopupFlags_AnyPopupId | imgui_sys::ImGuiPopupFlags_AnyPopupLevel) as i32,
            )
        };
        mark!("render", t_section);
        // mark!'s trailing reassignment is dead on the final call — read
        // t_section once so the macro can stay uniform across call sites.
        let _ = t_section;
        let frame_ms = now.elapsed().as_secs_f64() * 1000.0;
        if frame_ms > 50.0 {
            tlog!("\nSLOW FRAME: {:.0}ms", frame_ms);
        }
    }
}

/// What `resolve_event_source` decided to do with an event's source
/// linkage; `apply_src_action` turns it into UI effects.
#[cfg(not(target_arch = "wasm32"))]
enum SrcAction {
    /// Surface this message in the Source tab's error line.
    Error(String),
    /// Exactly one hit: launch the editor at it.
    Open(source::SourceHit),
    /// Several hits: list them in the Source tab and switch to it.
    List(Vec<source::SourceHit>),
    /// No checkout indexed yet: switch to the Source tab for setup.
    NeedCheckout,
}

/// What kineto's launch plumbing knows about where a GPU kernel came from.
/// `op` is the name of the CPU-side event carrying the same `External id`
/// (the RecordFunction that launched the kernel, e.g. `vllm::foo`), and
/// `at` is the (track, event) of the CPU launch event itself, whose
/// enclosing python frames name the file:line that issued the launch.
#[cfg(not(target_arch = "wasm32"))]
struct LaunchLink {
    op: Option<String>,
    at: Option<(u32, u32)>,
}

/// Follow a kernel's `External id` / `correlation` args to its CPU-side
/// launch. This is the only route to source for kernels whose name carries
/// nothing: CUPTI reports CuTe DSL's JIT-loaded cubins as plain `Kernel`,
/// so the name cannot resolve, but every launch still correlates to the
/// event that issued it. Only the categories that ever carry a correlation
/// are parsed (`cpu_op` for External id, `cuda_driver`/`cuda_runtime` for
/// the launch call), which keeps the scan at ~0.2s even on a 15M-event
/// trace; python_function events are 98% of a kineto trace and never carry
/// either key.
#[cfg(not(target_arch = "wasm32"))]
fn launch_link(trace: &Trace, track_idx: u32, event_idx: u32) -> Option<LaunchLink> {
    let track = trace.tracks.get(track_idx as usize)?;
    let ev = track.events.get(event_idx as usize)?;
    if ev.args_off == 0 { return None; }
    let raw = &trace.raw_bufs[track.raw_buf_idx as usize];
    let off = ev.args_off as usize;
    if off >= raw.len() { return None; }
    let end = skip_value(raw, off);
    let mut strs = InternTable::new();
    let mut pairs = Vec::new();
    parse_args_flat(&raw[off..end], &mut strs, &mut pairs);
    let (mut extid, mut corr) = (None, None);
    for &[k, v] in &pairs {
        match strs.str(k) {
            "External id" => extid = Some(strs.str(v).to_string()),
            "correlation" => corr = Some(strs.str(v).to_string()),
            _ => {}
        }
    }
    if extid.is_none() && corr.is_none() { return None; }
    // A merged trace interleaves ranks whose correlation ids are independent
    // counters, so a match from another rank's tracks is a false positive.
    // Prefer the same "Rank N" prefix when the kernel track carries one.
    let want_rank = rank_tag(&track.label);
    let mut link = LaunchLink { op: None, at: None };
    let mut other_strs = InternTable::new();
    let mut other_pairs = Vec::new();
    for (tj, tjr) in trace.tracks.iter().enumerate() {
        if tj as u32 == track_idx { continue; }
        if let (Some(want), Some(got)) = (&want_rank, rank_tag(&tjr.label)) {
            if want != &got { continue; }
        }
        let traw = &trace.raw_bufs[tjr.raw_buf_idx as usize];
        for (ej, e2) in tjr.events.iter().enumerate() {
            let cat = &trace.cats[e2.cat as usize];
            if !(cat == "cpu_op" || cat == "cuda_driver" || cat == "cuda_runtime") { continue; }
            if e2.args_off == 0 { continue; }
            let eoff = e2.args_off as usize;
            if eoff >= traw.len() { continue; }
            let eend = skip_value(traw, eoff);
            other_pairs.truncate(0);
            parse_args_flat(&traw[eoff..eend], &mut other_strs, &mut other_pairs);
            for &[k, v] in &other_pairs {
                let (key, val) = (other_strs.str(k), other_strs.str(v));
                if key == "External id" && extid.as_deref() == Some(val) {
                    link.op = Some(trace.names[e2.name as usize].clone());
                    link.at.get_or_insert((tj as u32, ej as u32));
                } else if key == "correlation" && corr.as_deref() == Some(val) {
                    link.at = Some((tj as u32, ej as u32));
                }
            }
            if link.op.is_some() && link.at.is_some() { return Some(link); }
        }
    }
    (link.op.is_some() || link.at.is_some()).then_some(link)
}

/// `"  Rank 3 stream 19"` -> `Some("Rank 3")`; used to keep a merged
/// trace's cross-rank correlation ids from matching each other.
#[cfg(not(target_arch = "wasm32"))]
fn rank_tag(label: &str) -> Option<String> {
    let mut it = label.trim().strip_prefix("Rank ")?.split_whitespace();
    Some(format!("Rank {}", it.next()?))
}

/// `vllm/v1/worker/gpu/model_runner.py(1813): execute_model` ->
/// (`vllm/v1/worker/gpu/model_runner.py`, 1813). Also accepts kineto's
/// older `execute_model(vllm/...py:1813)` spelling. None for frames that
/// carry no file:line (`<built-in function launch>`, `<module>`).
#[cfg(not(target_arch = "wasm32"))]
fn parse_frame_site(name: &str) -> Option<(&str, u32)> {
    if let Some(i) = name.find(".py(") {
        let rest = &name[i + 4..];
        let close = rest.find(')')?;
        let line = rest[..close].parse().ok()?;
        return Some((&name[..i + 3], line));
    }
    let i = name.find(".py:")?;
    let start = name[..i].rfind('(').map_or(0, |p| p + 1);
    let rest = &name[i + 4..];
    let close = rest.find(')').unwrap_or(rest.len());
    let line = rest[..close].parse().ok()?;
    Some((&name[start..i + 3], line))
}

/// The innermost enclosing python frame of `(track, event)` that resolves
/// in the checkout. Frames nest by time on the same track, so the shortest
/// enclosing span is the innermost one; walking outward past torch/triton
/// frames (which don't exist under the checkout root) lands on the vLLM
/// frame that actually issued the operation.
#[cfg(not(target_arch = "wasm32"))]
fn enclosing_frame(
    trace: &Trace,
    at: (u32, u32),
    idx: &source::SourceIndex,
) -> Option<source::SourceHit> {
    let track = trace.tracks.get(at.0 as usize)?;
    let ev = track.events.get(at.1 as usize)?;
    let py = trace.cats.iter().position(|c| c == "python_function")? as u32;
    let mut cands: Vec<(f64, String)> = track.events.iter()
        .filter(|e| e.cat == py && e.ts <= ev.ts && ev.ts <= e.ts + e.dur)
        .map(|e| (e.dur, trace.names[e.name as usize].clone()))
        .collect();
    cands.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, name) in cands {
        let Some((file, line)) = parse_frame_site(&name) else { continue };
        let Some(hit) = idx.resolve_callfrom(&format!("{file}:{line}")) else { continue };
        return Some(hit);
    }
    None
}

/// Event -> source resolution, shared by the Detail tab's "View Source"
/// button and the timeline right-click menu. Takes `pane` by shared
/// reference and the index as a separate borrow, so it can be called from
/// tab-body scopes that hold `&mut pane` — the disjoint-field borrow split
/// those bodies already rely on. Returns the action plus an optional drift
/// warning to show even on success: a checkout that isn't the trace's build
/// commit resolves the right FILES but shifted LINES.
#[cfg(not(target_arch = "wasm32"))]
fn resolve_event_source(
    idx: &mut Option<Box<source::SourceIndex>>,
    pane: &Pane,
    r: EventRef,
) -> (SrcAction, Option<String>) {
    let Some(trace) = pane.trace.as_ref() else {
        return (SrcAction::Error("no trace".into()), None);
    };
    // Resolve defensively: an EventRef can outlive a reload/merge that
    // shrank the trace (same reasoning as the Detail tab's sel_ev).
    let Some(track) = trace.tracks.get(r.track_idx as usize) else {
        return (SrcAction::Error("stale event reference".into()), None);
    };
    let Some(ev) = track.events.get(r.event_idx as usize) else {
        return (SrcAction::Error("stale event reference".into()), None);
    };
    let name = &trace.names[ev.name as usize];
    let cat = &trace.cats[ev.cat as usize];
    // CallFrom (kineto with_stack) is the exact python file:line.
    let mut callfrom: Option<String> = None;
    if ev.args_off > 0 {
        let raw = &trace.raw_bufs[track.raw_buf_idx as usize];
        let off = ev.args_off as usize;
        if off < raw.len() {
            let end = skip_value(raw, off);
            let mut strs = InternTable::new();
            let mut pairs = Vec::new();
            parse_args_flat(&raw[off..end], &mut strs, &mut pairs);
            for &[k, v] in &pairs {
                if strs.str(k) == "CallFrom" {
                    callfrom = Some(strs.str(v).to_string());
                    break;
                }
            }
        }
    }
    if callfrom.is_none() {
        // Name resolution covers GPU kernels AND cpu custom ops — `vllm::`
        // ops appear as cpu_op events on CPU tracks, so those must link too.
        if !(track.gpu || cat == "kernel" || cat == "cpu_op") {
            return (SrcAction::Error(format!(
                "no source linkage for `{}` ({} events carry none)", name, cat)), None);
        }
        if source::is_torch_internal(name) {
            return (SrcAction::Error(format!(
                "`{}` is a torch op - its source lives in pytorch, not the vllm checkout", name)), None);
        }
    }
    let Some(idx) = idx.as_mut() else { return (SrcAction::NeedCheckout, None) };
    // Drift check, before the borrows get tangled: the trace's build ref
    // vs the checkout's git describe. A mismatch is the one way a jump can
    // land in the right file at the wrong line, so it must ride along with
    // every success, not just sit in the Source tab's banner.
    let git_desc = idx.git_desc();
    let warn = source::trace_build_ref(&trace.vllm_version).filter(|b| {
        !git_desc.as_deref().is_some_and(|d| b.matches_desc(d))
    }).map(|b| format!(
        "checkout {} is not the trace's build {} - line numbers may be off",
        git_desc.as_deref().unwrap_or("?"), b.id()));
    // A generic name (`Kernel`) must not be scanned: it word-matches
    // unrelated C++ symbols and would hand back a junk list.
    let generic = source::is_generic_kernel_name(name);
    let mut hits = if let Some(cf) = &callfrom {
        idx.resolve_callfrom(cf).into_iter().collect::<Vec<_>>()
    } else if generic {
        Vec::new()
    } else {
        idx.resolve_kernel(name)
    };
    // No name-based source: a launched kernel still knows where it was
    // launched. Follow External id / correlation to the CPU event and use
    // its op name, else the innermost enclosing python frame. This is what
    // resolves CuTe DSL JIT kernels, which CUPTI only ever names `Kernel`.
    let mut via = None;
    if hits.is_empty() && callfrom.is_none() && (track.gpu || cat == "kernel") {
        if let Some(link) = launch_link(trace, r.track_idx, r.event_idx) {
            if let Some(op) = link.op.as_deref() {
                if !source::is_torch_internal(op) {
                    hits = idx.resolve_kernel(op);
                    if !hits.is_empty() { via = Some(op.to_string()); }
                }
            }
            if hits.is_empty() {
                if let Some(at) = link.at {
                    if let Some(h) = enclosing_frame(trace, at, idx.as_ref()) {
                        via = Some(format!("{}:{}", h.rel, h.line));
                        hits = vec![h];
                    }
                }
            }
        }
    }
    // The launch-site note rides the jump, so a generic `Kernel` event
    // says where it came from instead of looking like the kernel's def.
    // A drift warning still matters (launch-site lines drift too), so both
    // are kept.
    let warn = match (warn, via) {
        (Some(w), Some(v)) => Some(format!("{w}; launch site of `{name}`: {v}")),
        (None, Some(v)) => Some(format!("launch site of `{name}`: {v}")),
        (w, None) => w,
    };
    let act = match hits.len() {
        0 => SrcAction::Error(if generic {
            format!("`{}` is CUPTI's placeholder for a kernel it could not \
                     symbolicate, and this trace carries no launch correlation \
                     to resolve it", name)
        } else { match source::external_kernel_family(name) {
            Some(why) => format!("`{}` is {}", name, why),
            None => format!(
                "no source found for `{}` in the checkout (aten/CUB/CUTLASS kernels live outside vllm)", name),
        } }),
        1 => SrcAction::Open(hits.into_iter().next().unwrap()),
        _ => SrcAction::List(hits),
    };
    (act, warn)
}

/// Apply a resolution: open the editor, open the multi-hit picker popup,
/// open the setup popup (no checkout yet), or park the outcome in
/// `src_msg` for the status bar. Takes the affected Pane fields
/// individually, not `&mut Pane` — the context-menu call site holds a
/// shared borrow of the pane's trace across it.
#[cfg(not(target_arch = "wasm32"))]
fn apply_src_action(
    ui: &imgui::Ui,
    msg: &mut Option<(SrcNote, String)>,
    hits: &mut Vec<source::SourceHit>,
    editor: &str,
    act: SrcAction,
    warn: Option<String>,
) {
    match act {
        SrcAction::Error(e) => *msg = Some((SrcNote::Err, e)),
        SrcAction::Open(h) => match source::open_in_editor(editor, &h) {
            Err(e) => *msg = Some((SrcNote::Err, e)),
            // Where the jump landed, or the drift warning when the
            // checkout isn't the trace's build (the jump did happen).
            Ok(()) => *msg = Some(match warn {
                Some(w) => (SrcNote::Warn, w),
                None => (SrcNote::Ok, format!("opened {}:{}", h.rel, h.line)),
            }),
        },
        SrcAction::List(hs) => {
            *hits = hs;
            ui.open_popup("##src_hits");
            *msg = warn.map(|w| (SrcNote::Warn, w));
        }
        // No checkout yet: the setup popup is where one gets set up.
        SrcAction::NeedCheckout => ui.open_popup("##src_setup"),
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
            .flat_map(|(g, _)| g.iter().map(|(_, p)| p.as_str()))
            .chain(standalone.iter().map(|(p, _)| p.as_str()))
            .collect();
        eprintln!("bench: {} files found in {:.3}s", all_files.len(), t0.elapsed().as_secs_f64());
        for f in &all_files { eprintln!("  {f}"); }

        if rank_groups.is_empty() && standalone.len() <= 1 {
            let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let path = standalone.first().map(|(s, _)| s.as_str()).unwrap_or(&bench_args[0]);
            match loader::load_trace(path, &counter) {
                Ok(t) => eprintln!("  ok: {} events, {} tracks, {:.2}s", t.total_events, t.tracks.len(), t0.elapsed().as_secs_f64()),
                Err(e) => eprintln!("  err: {e}"),
            }
        } else {
            let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let mut all_paths: Vec<(usize, String)> = Vec::new();
            for (group, _) in &rank_groups { all_paths.extend(group.clone()); }
            for (i, (p, _)) in standalone.iter().enumerate() { all_paths.push((all_paths.len() + i, p.clone())); }

            let t_load = Instant::now();
            use rayon::prelude::*;
            let results: Vec<_> = all_paths.par_iter().filter_map(|(rank, path)| {
                let r = *rank;
                let ctr = counter.clone();
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let t = Instant::now();
                    let res = loader::load_trace(path, &ctr);
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
    let mut app = App::new(args);
    // Start loading argv traces before any macOS/winit/ImGui/renderer init —
    // the load job is fs + channels + rayon only, so the parse overlaps all
    // of it instead of waiting for `resumed` to finish (see
    // `open_launch_files`).
    app.open_launch_files();
    let event_loop = EventLoop::new().unwrap();
    #[cfg(target_os = "macos")]
    dock_menu::install();
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
            // single-threaded behavior, not a crash. That fallback is why
            // MT_POOL_READY stays false here: `rayon::spawn` with no live
            // pool would never be picked up (see spawn_load_job), so trace
            // loads fall back to running inline instead of hanging.
            web_sys::console::error_1(&e);
        } else {
            crate::state::MT_POOL_READY.store(true, std::sync::atomic::Ordering::Relaxed);
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
