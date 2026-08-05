use imgui::ImColor32;
use std::collections::HashMap;
use std::sync::Arc;

pub enum ArgsBuf {
    Heap(Vec<u8>),
    Mmap { mmap: memmap2::Mmap, offset: usize, len: usize },
}

impl std::ops::Deref for ArgsBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            ArgsBuf::Heap(v) => v,
            ArgsBuf::Mmap { mmap, offset, len } => &mmap[*offset..*offset + *len],
        }
    }
}

pub const RULER_H: f32 = 28.0;
pub const SUB_LANE_H: f32 = 20.0;
pub const LABEL_W: f32 = 200.0;
pub const MIN_EV_PX: f32 = 1.0;
pub const TEXT_MIN_PX: f32 = 60.0;
pub const TOOLBAR_ROW: f32 = 24.0;
pub const TOOLBAR_H: f32 = TOOLBAR_ROW + 8.0;
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
pub const DIVIDER_GRAB_PX: f32 = 4.0;
pub const MIN_BOTTOM_H: f32 = 60.0;
pub const MIN_LABEL_W: f32 = 60.0;
pub const MIN_SPLIT_W: f32 = 200.0;
pub const LANE_GAP: f32 = 4.0;
pub const EV_INSET: f32 = 2.0;
pub const EV_ROUNDING: f32 = 2.0;
pub const SWATCH_W: f32 = 10.0;
pub const SWATCH_PAD: f32 = SWATCH_W + 4.0;
pub const STATS_COL_W: f32 = 80.0;
pub const SEARCH_W: f32 = 200.0;
pub const DIFF_BAR_H: f32 = 22.0;
pub const DIFF_BAR_GAP: f32 = 4.0;
pub const DIFF_SEG_GAP: f32 = 3.0;
pub const ZOOM_STEP: f64 = 1.15;
pub const MAX_ZOOM: f64 = 200.0;
pub const TRACK_SCALE_MIN: f32 = 0.5;
pub const TRACK_SCALE_MAX: f32 = 5.0;
pub const RESIZE_GRAB_H: f32 = 6.0;
pub const ROW_PAD: f32 = 4.0;
pub const HISTOGRAM_BAR_H: f32 = 18.0;
pub const LABEL_NAME_W: f32 = 160.0;

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

pub const LABEL_PALETTE: &[u32] = &[
    0x2196F3, 0x4CAF50, 0xFF9800, 0xE91E63, 0x9C27B0, 0x00BCD4, 0xFFEB3B, 0x795548,
];

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
    pub flow_pairs: Vec<FlowPair>,
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
}

impl Default for View {
    fn default() -> Self {
        Self { t0: 0.0, t1: 1.0, scroll_y: 0.0 }
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
}

pub struct SelectionEntry {
    pub name: u32,
    pub count: u32,
    pub total_dur: f64,
    pub durations: Vec<f64>,
}

pub struct Label {
    pub name: String,
    pub color: ImColor32,
    pub pattern: Vec<u32>,
}

pub struct LabelStats {
    pub label_idx: u8,
    pub total_dur: f64,
    pub count: u32,
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
pub enum BottomTab { Detail, Stats, Selection, Labels }

#[derive(Clone, Copy, PartialEq)]
pub enum DragKind {
    None,
    BottomDivider,
    LabelDivider(usize),
    TrackResize(usize, usize),
    TrackDrag(usize, usize, f32),
    SplitDivider(usize),
}

impl DragKind {
    pub fn is_active(self) -> bool { self != DragKind::None }
}

#[derive(Default)]
pub struct DrawBuf {
    pub visible: Vec<usize>,
    pub heights: Vec<f32>,
    pub y_offsets: Vec<f32>,
    pub last_px: Vec<i32>,
    pub fmt: String,
    pub sort_idx: Vec<usize>,
    pub sel_map: HashMap<u32, (u32, f64, Vec<f64>)>,
    pub sel_bars: Vec<(f64, u32)>,
    pub detail_buf: String,
    pub col_widths: [f32; 7],
    pub col_widths_total: f32,
    pub merged_gpu_tracks: Vec<usize>,
    pub merged_gpu_vi: Option<usize>,
    pub merged_gpu_events: Vec<(u32, u32, u16)>,
    pub merged_gpu_max_depth: u16,
}
