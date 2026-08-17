use imgui::ImColor32;
use std::collections::HashMap;
use std::sync::Arc;

pub enum ArgsBuf {
    Heap(Vec<u8>),
}

impl std::ops::Deref for ArgsBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            ArgsBuf::Heap(v) => v,
        }
    }
}

pub const RULER_H: f32 = 28.0;
pub const SUB_LANE_H: f32 = 20.0;
pub const LABEL_W: f32 = 200.0;
pub const MIN_EV_PX: f32 = 1.0;
pub const TEXT_MIN_PX: f32 = 60.0;
pub const TOOLBAR_ROW: f32 = 24.0;
/// Height of the tab strip (open traces, one per tab), directly below the
/// menu-bar/search toolbar row (whose height is computed from real font/style
/// metrics in `render_frame`, not a fixed constant — see `toolbar_h` there).
pub const TAB_BAR_H: f32 = TOOLBAR_ROW;
pub const STATUS_H: f32 = TOOLBAR_ROW + 4.0;
pub const DETAIL_H: f32 = 200.0;
pub const INITIAL_BUF: usize = 256 * 1024;

pub const INITIAL_WIN_W: f32 = 1400.0;
pub const INITIAL_WIN_H: f32 = 800.0;
pub const LINE_SCROLL_PX: f32 = 20.0;
pub const SCROLL_ZOOM_SENSITIVITY: f64 = 200.0;
pub const TIMELINE_PAD_FRAC: f64 = 0.05;
pub const MIN_TIME_RANGE: f64 = 0.001;
pub const FIT_PAD_FRAC: f64 = 0.02;
pub const DIVIDER_GRAB_PX: f32 = 7.0;
pub const MIN_BOTTOM_H: f32 = 60.0;
pub const MIN_LABEL_W: f32 = 60.0;
pub const LANE_GAP: f32 = 4.0;
pub const EV_INSET: f32 = 2.0;
pub const EV_ROUNDING: f32 = 2.0;
pub const SWATCH_W: f32 = 10.0;
pub const SWATCH_PAD: f32 = SWATCH_W + 4.0;
pub const SEARCH_W: f32 = 200.0;
pub const DIFF_BAR_H: f32 = 22.0;
pub const DIFF_BAR_GAP: f32 = 4.0;
pub const DIFF_SEG_GAP: f32 = 3.0;
pub const ZOOM_STEP: f64 = 1.15;
pub const MAX_ZOOM: f64 = 200.0;
pub const TRACK_SCALE_MIN: f32 = 0.5;
pub const TRACK_SCALE_MAX: f32 = 30.0;
pub const RESIZE_GRAB_H: f32 = 8.0;
pub const ZOOM_ANIM_DUR: f32 = 0.35;
pub const SEARCH_ZOOM_FILL: f64 = 0.8; // matches fill this fraction of the width
pub const ROW_PAD: f32 = 4.0;
pub const HISTOGRAM_BAR_H: f32 = 18.0;
pub const DETAIL_HIST_H: f32 = 70.0;
/// Floor for the dynamically shrunk font size used when a lane/row is
/// squashed thinner than one line of text at the default size (deep call
/// nesting, a scaled-down track, or an even-spacing collapse). Below this
/// glyphs stop being legible either way, so the text is drawn at this size
/// and allowed to overflow its row slightly rather than being hidden
/// outright — a label peeking into its neighbor reads better than a blank row.
pub const MIN_TEXT_PX: f32 = 6.0;

pub fn track_height(max_depth: u16, collapsed: bool, scale: f32) -> f32 {
    let base = if collapsed { SUB_LANE_H } else { max_depth.max(1) as f32 * SUB_LANE_H };
    base * scale
}

pub fn bisect_overlap(events: &[Event], prefix_max_dur: &[f64], t: f64) -> usize {
    let (mut lo, mut hi) = (0, events.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if events[mid].ts + prefix_max_dur[mid] < t {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

pub const PALETTE: &[u32] = &[
    0x4E79A7, 0xF28E2B, 0xE15759, 0x76B7B2, 0x59A14F, 0xEDC948, 0xB07AA1, 0xFF9DA7, 0x9C755F,
    0x86BCB6, 0x8CD17D, 0xB6992D, 0xF1CE63, 0xA0CBE8, 0xFFBE7D, 0xD4A6C8,
];

// ---- Unified chrome colors ----
// A single accent (the logo blue) replaces the several mismatched blues that
// used to be scattered across selection rects, tooltips and outlines.
pub const ACCENT_LINE: ImColor32 = ImColor32::from_rgba(48, 162, 255, 200); // selection/tooltip borders
pub const ACCENT_FILL: ImColor32 = ImColor32::from_rgba(48, 162, 255, 40); // range-select fill
pub const ACCENT_SOFT: ImColor32 = ImColor32::from_rgba(48, 162, 255, 180); // block selection/search outline

// Neutral greys used across the timeline chrome (values unchanged, just named).
pub const BG_TIMELINE: ImColor32 = ImColor32::from_rgba(24, 24, 24, 255);
pub const BG_LABELS: ImColor32 = ImColor32::from_rgba(20, 20, 20, 255);
pub const DIVIDER: ImColor32 = ImColor32::from_rgba(50, 50, 50, 255);
pub const GRID: ImColor32 = ImColor32::from_rgba(40, 40, 40, 255);
pub const ROW_BG_A: ImColor32 = ImColor32::from_rgba(28, 28, 28, 255);
pub const ROW_BG_B: ImColor32 = ImColor32::from_rgba(32, 32, 32, 255);
pub const RULER_BG: ImColor32 = ImColor32::from_rgba(18, 18, 18, 255);
pub const RULER_TICK: ImColor32 = ImColor32::from_rgba(60, 60, 60, 255);
pub const RULER_TEXT: ImColor32 = ImColor32::from_rgba(160, 160, 160, 255);

pub struct Trace {
    pub tracks: Vec<Track>,
    pub names: Vec<String>,
    pub cats: Vec<String>,
    pub raw_bufs: Vec<Arc<ArgsBuf>>,
    pub stats: Vec<KernelStats>,
    pub max_ts: f64,
    pub min_ts: f64,
    pub total_events: usize,
    pub device: String,
    /// vLLM version/commit string from the trace header (e.g.
    /// "0.26.1rc1.dev528+gf8d03e774"). Empty when the trace omits it.
    pub vllm_version: String,
    /// This rank's id from `distributedInfo`, or -1 if the trace has none
    /// (non-distributed run, or a merged multi-rank trace).
    pub dist_rank: i32,
    /// Total ranks in the job from `distributedInfo.world_size`, or 0 if absent.
    pub dist_world: i32,
    pub flow_pairs: Vec<FlowPair>,
    /// `(rank, source file path/name)` for each rank folded into this trace
    /// by `merge_traces` — empty for a single-rank trace. Persisted in the
    /// on-disk/exported cache format so `Pane::sync_clocks` can still group
    /// ranks by DP/TP (parsed from these filenames) after reopening a
    /// `.tvcache` directly, when `Pane::reload_paths` no longer has the
    /// original per-rank paths (e.g. it points at just the one cache file).
    pub rank_paths: Vec<(usize, String)>,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct FlowPair {
    pub src_track: u32,
    pub dst_track: u32,
    pub src_ts: f64,
    pub dst_ts: f64,
}

#[derive(Clone)]
pub struct Track {
    pub label: String,
    pub gpu: bool,
    pub events: Vec<Event>,
    pub max_depth: u16,
    pub prefix_max_dur: Vec<f64>,
    pub raw_buf_idx: u8,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Event {
    pub ts: f64,
    pub dur: f64,
    pub name: u32,
    pub cat: u32,
    pub args_off: u32,
    pub depth: u16,
    pub args_len: u16,
}

#[derive(Clone, Copy)]
pub struct EventRef {
    pub track_idx: u32,
    pub event_idx: u32,
}

pub struct View {
    pub t0: f64,
    pub t1: f64,
    pub scroll_y: f32,
    /// In-flight smooth zoom (e.g. from search-Enter). `None` when idle.
    pub anim: Option<ViewAnim>,
}

impl Default for View {
    fn default() -> Self {
        Self { t0: 0.0, t1: 1.0, scroll_y: 0.0, anim: None }
    }
}

/// A smooth zoom between two time ranges. The center eases linearly while the
/// range eases geometrically (log space), which reads as a natural zoom rather
/// than the visible content rushing when endpoints are lerped directly.
pub struct ViewAnim {
    pub from_t0: f64,
    pub from_t1: f64,
    pub to_t0: f64,
    pub to_t1: f64,
    pub from_scroll: f32,
    pub to_scroll: f32,
    pub elapsed: f32,
    pub dur: f32,
}

impl View {
    /// Advance an in-flight zoom by `dt` seconds, writing the eased t0/t1.
    /// Returns true while the animation is still running.
    pub fn tick_anim(&mut self, dt: f32) -> bool {
        let a = match self.anim.as_mut() {
            Some(a) => a,
            None => return false,
        };
        // Clamp per-tick dt: the first frame after an idle Wait can carry a large
        // elapsed time, which would blow past dur and snap the zoom instead of
        // gliding. While animating we run in Poll (~60fps), so capping keeps it smooth.
        a.elapsed += dt.min(0.05);
        let f = (a.elapsed / a.dur).clamp(0.0, 1.0);
        if f >= 1.0 {
            self.t0 = a.to_t0;
            self.t1 = a.to_t1;
            self.scroll_y = a.to_scroll;
            self.anim = None;
            return false;
        }
        // smoothstep ease-in-out
        let e = (f * f * (3.0 - 2.0 * f)) as f64;
        let c_from = (a.from_t0 + a.from_t1) / 2.0;
        let c_to = (a.to_t0 + a.to_t1) / 2.0;
        let r_from = (a.from_t1 - a.from_t0).max(1e-12);
        let r_to = (a.to_t1 - a.to_t0).max(1e-12);
        let c = c_from + (c_to - c_from) * e;
        let r = r_from * (r_to / r_from).powf(e);
        self.t0 = c - r / 2.0;
        self.t1 = c + r / 2.0;
        self.scroll_y = a.from_scroll + (a.to_scroll - a.from_scroll) * e as f32;
        true
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct KernelStats {
    pub name: u32,
    pub count: u32,
    pub total_dur: f64,
    pub median_dur: f64,
    pub max_dur: f64,
    pub min_dur: f64,
}

pub struct SelectionEntry {
    pub name: u32,
    pub count: u32,
    pub total_dur: f64,
    pub durations: Vec<f64>,
    /// (track_idx, event_idx) parallel to `durations`, so the individual-rows
    /// stats table can look each event's raw args back up (e.g. to show its
    /// CUDA occupancy-limiting factor). Empty for aggregation paths that don't
    /// need it (there are none currently, but keep it optional-shaped).
    pub event_refs: Vec<(u32, u32)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DiffKind { Same, Added, Removed }

pub struct DiffLine {
    pub kind: DiffKind,
    pub name: String,
    pub dur_a: Option<f64>,
    pub dur_b: Option<f64>,
}

pub struct DiffResult {
    pub lines: Vec<DiffLine>,
    pub count_a: u32,
    pub count_b: u32,
    pub total_dur_a: f64,
    pub total_dur_b: f64,
}

#[derive(Clone, Copy, PartialEq)]
pub enum BottomTab { Detail, Selection }

#[derive(Clone, Copy, PartialEq)]
pub enum DragKind {
    None,
    BottomDivider,
    LabelDivider(usize),
    TrackResize(usize, usize),
    TrackDrag(usize, usize, f32),
}

impl DragKind {
    pub fn is_active(self) -> bool { self != DragKind::None }
}

pub struct MergedGpuGroup {
    pub tracks: Vec<usize>,
    pub events: Vec<(u32, u32, u16)>,
    pub max_depth: u16,
    pub vi: usize,
    pub label: String,
}

/// Per-pane snapshot of the final row layout produced by `draw_timeline`.
///
/// `DrawBuf` (the render scratch) is shared across panes, so after the per-pane
/// render loop it only holds the LAST-drawn pane's geometry. Any per-pane
/// operation that runs later — selection stats, diff extraction, clipboard copy
/// — must read from this pane-owned snapshot instead, or it would compute
/// against the wrong pane's tracks.
#[derive(Default)]
pub struct PaneGeom {
    pub visible: Vec<usize>,
    pub heights: Vec<f32>,
    pub y_offsets: Vec<f32>,
    pub merged: Vec<MergedGeom>,
}

pub struct MergedGeom {
    pub vi: usize,
    /// Packed `(track_idx, event_idx, packed_depth)` triples — exactly the events
    /// drawn in the merged row. The Tetris packing in `draw_timeline` already
    /// stripped grandparent wrappers (whole-stream spans) and hidden names, so a
    /// selection that iterates these matches the rendered row precisely instead of
    /// sweeping in ghost events that were never drawn.
    pub events: Vec<(u32, u32, u16)>,
}

#[derive(Default)]
pub struct DrawBuf {
    pub visible: Vec<usize>,
    pub heights: Vec<f32>,
    pub y_offsets: Vec<f32>,
    pub last_px: Vec<i32>,
    pub fmt: String,
    pub sel_map: HashMap<u32, (u32, f64, Vec<f64>, Vec<(u32, u32)>)>,
    pub sel_bars: Vec<(f64, u32)>,
    pub detail_buf: String,
    pub merged_gpu_groups: Vec<MergedGpuGroup>,
}
