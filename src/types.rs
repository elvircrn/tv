use imgui::ImColor32;
use std::collections::HashMap;

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

pub fn track_height(max_depth: u16, collapsed: bool, scale: f32) -> f32 {
    let base = if collapsed { SUB_LANE_H } else { max_depth.max(1) as f32 * SUB_LANE_H };
    base * scale
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
    pub arg_strs: Vec<String>,
    pub arg_pairs: Vec<[u32; 2]>,
    pub stats: Vec<KernelStats>,
    pub max_ts: f64,
    pub total_events: usize,
    pub device: String,
}

pub struct Track {
    pub label: String,
    pub gpu: bool,
    pub events: Vec<Event>,
    pub max_depth: u16,
    pub max_dur: f64,
}

#[derive(Clone, Copy)]
pub struct Event {
    pub ts: f64,
    pub dur: f64,
    pub name: u32,
    pub cat: u32,
    pub args_start: u32,
    pub args_count: u16,
    pub depth: u16,
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
    LabelDivider,
    TrackResize(usize),
    SplitDivider,
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
}
