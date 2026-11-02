use imgui::ImColor32;
use std::collections::HashMap;
use std::sync::Arc;

pub enum ArgsBuf {
    Heap(Vec<u8>),
    /// The mmap'd source .json itself, kept alive zero-copy. `into_vec`-ing a
    /// mapped file cost a full-file heap copy on every plain .json load
    /// (0.36s and ~1.6GB of anon RSS on a 1.6GB file) that `compact_args`
    /// then paid for a second time when it copied the args ranges out — so
    /// events now hold `args_off` straight into the file's pages. The mapping
    /// is file-backed (clean, reclaimable under pressure) and short-lived:
    /// `compact_args` swaps in one Heap buffer and every live `Arc<ArgsBuf>`
    /// to the mapping drops. Native-only, mirroring `RawData::Mmap`.
    #[cfg(not(target_arch = "wasm32"))]
    Mmap(memmap2::Mmap),
}

impl std::ops::Deref for ArgsBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            ArgsBuf::Heap(v) => v,
            #[cfg(not(target_arch = "wasm32"))]
            ArgsBuf::Mmap(m) => m,
        }
    }
}

pub const RULER_H: f32 = 20.0;
pub const SUB_LANE_H: f32 = 20.0;
pub const LABEL_W: f32 = 200.0;
pub const MIN_EV_PX: f32 = 1.0;
pub const TEXT_MIN_PX: f32 = 60.0;
pub const TOOLBAR_ROW: f32 = 24.0;
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
/// Slack either side of a step when framing it via `[`/`]` or the Steps tab —
/// fraction of the final view width, so the inter-step gaps stay visible.
pub const STEP_PAD: f64 = 0.05;
/// A GPU idle gap shorter than this isn't a bubble worth listing: kernel
/// launch spacing alone leaves sub-µs holes between back-to-back kernels.
pub const MIN_GAP_US: f64 = 1.0;
/// How many gaps (biggest first) the Gaps tab keeps — the per-gap CPU
/// annotation is the expensive part, so it runs on the top slice only.
pub const GAP_TOP_N: usize = 500;
pub const ROW_PAD: f32 = 4.0;
pub const HISTOGRAM_BAR_H: f32 = 18.0;
pub const DETAIL_HIST_H: f32 = 90.0;
/// Floor for the dynamically shrunk font size used when a lane/row is
/// squashed thinner than one line of text at the default size (deep call
/// nesting, a scaled-down track, or an even-spacing collapse). Below this
/// glyphs stop being legible either way, so the text is drawn at this size
/// and allowed to overflow its row slightly rather than being hidden
/// outright — a label peeking into its neighbor reads better than a blank row.
pub const MIN_TEXT_PX: f32 = 6.0;

/// A packed bit set over `len` indexed slots — one bit per slot in `u64`
/// words instead of one byte per slot. Reads go through `get` or `Index`;
/// writes through `set`/`push`/`fill` (there is no `&mut bool` into packed
/// bits). Bits at index `>= len` are kept clear, so equality, `iter`, and
/// `count_ones` see exactly the live slots.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct BitSet {
    words: Vec<u64>,
    len: usize,
}

impl BitSet {
    pub fn new() -> Self {
        Self { words: Vec::new(), len: 0 }
    }

    /// An all-clear set with `n` slots.
    pub fn with_len(n: usize) -> Self {
        Self { words: vec![0u64; n.div_ceil(64)], len: n }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Drops every slot (`len` becomes 0), like `Vec::clear`.
    pub fn clear(&mut self) {
        self.words.clear();
        self.len = 0;
    }

    /// Grows or shrinks to `n` slots. Bits below `n` keep their value; new
    /// slots start clear.
    pub fn resize(&mut self, n: usize) {
        self.words.resize(n.div_ceil(64), 0);
        let rem = n % 64;
        if rem != 0 {
            if let Some(w) = self.words.last_mut() { *w &= (1u64 << rem) - 1; }
        }
        self.len = n;
    }

    pub fn get(&self, i: usize) -> Option<bool> {
        if i < self.len { Some(self[i]) } else { None }
    }

    pub fn set(&mut self, i: usize, v: bool) {
        assert!(i < self.len, "BitSet set at {i} but len is {}", self.len);
        let w = &mut self.words[i / 64];
        let bit = 1u64 << (i % 64);
        if v { *w |= bit; } else { *w &= !bit; }
    }

    /// Appends one slot, like `Vec::push`.
    pub fn push(&mut self, v: bool) {
        if self.len % 64 == 0 { self.words.push(0); }
        if v {
            let i = self.len;
            self.words[i / 64] |= 1u64 << (i % 64);
        }
        self.len += 1;
    }

    /// Sets every slot to `v`.
    pub fn fill(&mut self, v: bool) {
        let all = if v { !0u64 } else { 0 };
        for w in &mut self.words { *w = all; }
        let rem = self.len % 64;
        if rem != 0 {
            if let Some(w) = self.words.last_mut() { *w &= (1u64 << rem) - 1; }
        }
    }

    /// Whether any slot is set.
    pub fn any(&self) -> bool {
        self.words.iter().any(|&w| w != 0)
    }

    /// How many slots are set.
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.len).map(move |i| self.words[i / 64] & (1u64 << (i % 64)) != 0)
    }
}

impl FromIterator<bool> for BitSet {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let mut bs = BitSet::new();
        for v in iter { bs.push(v); }
        bs
    }
}

impl std::ops::Index<usize> for BitSet {
    type Output = bool;
    fn index(&self, i: usize) -> &bool {
        assert!(i < self.len, "BitSet index {i} but len is {}", self.len);
        if self.words[i / 64] & (1u64 << (i % 64)) != 0 { &true } else { &false }
    }
}

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

/// Median of `durs` via partial reordering (`select_nth_unstable`) instead
/// of a full sort — callers only ever need the middle value(s), so this is
/// an O(n) average instead of O(n log n), which matters at the scale a
/// selection or a whole-trace kernel-name group can reach (hundreds of
/// thousands of durations sharing one float comparator). Leaves `durs`
/// reordered but NOT fully sorted as a side effect; only call this with a
/// buffer the caller doesn't need sorted afterward.
pub fn median_inplace(durs: &mut [f64]) -> f64 {
    let n = durs.len();
    if n == 0 { return 0.0; }
    let mid = n / 2;
    durs.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    if n % 2 == 1 {
        durs[mid]
    } else {
        let hi = durs[mid];
        let lo = durs[..mid].iter().copied().fold(f64::MIN, f64::max);
        (lo + hi) / 2.0
    }
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
// vLLM step bands on the ruler (see draw_ruler): even steps get a faint tint
// over RULER_BG, and the step the [`/`] cursor is framed on gets a stronger
// one so consecutive steps stay distinguishable from each other AND from the
// cursor's step.
pub const STEP_BAND: ImColor32 = ImColor32::from_rgba(38, 38, 44, 255);
pub const STEP_BAND_CUR: ImColor32 = ImColor32::from_rgba(66, 66, 90, 255);
pub const STEP_EDGE: ImColor32 = ImColor32::from_rgba(90, 90, 105, 255);
// Focused-gap highlight (a Gaps-tab row click): a red-family band — idle
// time, matching the Why column's "CPU idle (starved)" — over the ruler and
// the canvas, drawn like a range selection.
pub const GAP_BAND_FILL: ImColor32 = ImColor32::from_rgba(235, 85, 65, 36);
pub const GAP_BAND_LINE: ImColor32 = ImColor32::from_rgba(235, 85, 65, 190);

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

/// Selection-tab GPU idle summary, PER RANK scope: over the selection's
/// GPU-work events on that rank's streams (clipped to its time range), the
/// span from the rank's leftmost to rightmost kernel and how much of it
/// any of them covers. `t1 - t0 - busy` is that rank's empty space between
/// the selected kernels. One entry per rank present in the selection
/// (None = unlabeled single-rank trace), sorted by rank. Computed by
/// `state::selection_gpu_summary` when the drag finishes.
pub struct SelGpuSummary {
    pub rank: Option<usize>,
    pub t0: f64,
    pub t1: f64,
    pub busy: f64,
}

/// Batch composition parsed from a vLLM execute-annotation span name.
/// Two formats exist (see vLLM `gpu_worker.py:annotate_profile`):
/// legacy `execute_context_1(4)_generation_1(1)` and detailed
/// `execute_512_context_1(sq512sk388352sqsq262144sqsk198836224)_generation_0(sq0sk0sqsq0sqsk0)`
/// (emitted with `profiler_config.detailed_trace_annotation`). The `kv`
/// fields are None in the legacy format.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepBatch {
    /// Total scheduled tokens across all requests (detailed format only;
    /// falls back to ctx+gen tokens for legacy).
    pub total_tokens: u64,
    /// Context (prefill / chunked-prefill) request count.
    pub n_ctx: u64,
    /// Context-phase scheduled tokens.
    pub ctx_tokens: u64,
    /// Sum of sequence lengths across context requests (total KV written).
    pub ctx_kv: Option<u64>,
    /// Generation (decode) request count.
    pub n_gen: u64,
    /// Generation-phase scheduled tokens (== n_gen for plain decode).
    pub gen_tokens: u64,
    /// Sum of sequence lengths across generation requests.
    pub gen_kv: Option<u64>,
}

impl StepBatch {
    /// Sum with another batch — used to aggregate the per-DP-group local
    /// batches of one scheduler step into the job-wide batch. With external
    /// data parallelism each DP group's workers annotate only their own
    /// group's slice, so the displayed batch is the sum over groups. KV sums
    /// stay None if either side lacks them (legacy format).
    pub fn sum(&self, other: &StepBatch) -> StepBatch {
        StepBatch {
            total_tokens: self.total_tokens + other.total_tokens,
            n_ctx: self.n_ctx + other.n_ctx,
            ctx_tokens: self.ctx_tokens + other.ctx_tokens,
            ctx_kv: match (self.ctx_kv, other.ctx_kv) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
            n_gen: self.n_gen + other.n_gen,
            gen_tokens: self.gen_tokens + other.gen_tokens,
            gen_kv: match (self.gen_kv, other.gen_kv) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
        }
    }
}

/// One vLLM scheduler step, derived from the `execute_context_*`
/// `gpu_user_annotation` spans (CPU-side `user_annotation` as fallback).
/// All spans share the same interned name — steps are distinguishable only
/// by temporal order, so a "reference track" defines the canonical sequence
/// and every other rank's span is assigned to the interval it overlaps most.
/// The reference is the track with the most such spans, ties broken by total
/// span duration then track index (see `compute_steps` for why the
/// tie-breaks matter); roughly the busiest CUDA stream that brackets each
/// step's full GPU extent.
/// Batch size / #tokens aren't recorded by these traces (`record_shapes=0`),
/// so the summary is timing/counts only.
pub struct StepInfo {
    /// Step start: min span start across assigned spans (covers cross-rank
    /// skew).
    pub t0: f64,
    /// Step end: max span end across assigned spans.
    pub t1: f64,
    /// Median across ranks of each rank's GPU-busy time for this step —
    /// the union of that rank's assigned projection spans (parallel
    /// streams overlap, so they're merged, not summed). Robust against
    /// one straggler rank stretching t1.
    pub gpu_dur: f64,
    /// Gap from the previous step's end (0.0 for the first step).
    pub idle_before: f64,
    /// Number of distinct ranks with a span assigned to this step (via
    /// `parse_rank` on track labels; 1 for single-rank traces).
    pub n_ranks: u32,
    /// `cat == "kernel"` events inside [t0, t1) across GPU tracks.
    pub kernel_count: u32,
    /// Batch composition parsed from the span name (token/req counts). With
    /// external data parallelism, each DP group's spans carry only that
    /// group's local batch, so this is the sum over DP groups (all groups'
    /// members share one scheduler output within a group, hence one parsed
    /// batch per group). None when no assigned span's name matched either
    /// annotation format.
    pub batch: Option<StepBatch>,
    /// The per-DP-group local batches that `batch` aggregates:
    /// `(dp index, local batch)`, one entry per distinct DP group with a
    /// span assigned to this step. Empty for single-scheduler traces (no
    /// DP split) — `batch` is then the whole batch already. Displayed as a
    /// per-group breakdown so a straggler group is visible rather than
    /// averaged away.
    pub group_batches: Vec<(u32, StepBatch)>,
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
pub enum BottomTab { Detail, Selection, Steps, Gaps }

/// One GPU idle gap ("bubble"): a wall-time interval where a rank's GPU was
/// doing no work at all, plus the CPU-side annotation that explains it.
/// Computed by `state::compute_gaps`; see `Pane::gaps` for the cache.
pub struct GapInfo {
    /// Rank scope the gap belongs to (`parse_rank` of the GPU tracks'
    /// labels; a multi-rank merge labels every track "  Rank N ...").
    /// None = unlabeled tracks, i.e. a single-rank trace's one shared scope.
    pub rank: Option<usize>,
    pub t0: f64,
    pub t1: f64,
    /// Fraction of [t0, t1] covered by ANY event on this rank's CPU tracks,
    /// as a union — nested call stacks (cpu_op ⊃ python_function) must not
    /// double-count. 0.0 when the CPU was doing nothing.
    pub cpu_busy_frac: f64,
    /// Fraction of [t0, t1] covered by CUDA-runtime events (launch calls,
    /// blocking syncs — both are `cat == "runtime"`).
    pub runtime_frac: f64,
    /// Whether the trace has ANY CPU tracks for this gap's scope. Without
    /// one, the fractions above are 0 for lack of data, not because the CPU
    /// was idle — the "Why" column must not claim starved.
    pub has_cpu: bool,
    /// Top CPU ops overlapping the gap, by covered time desc:
    /// (name id, overlapping event count, covered time — per-name union).
    pub cpu_top: Vec<(u32, u32, f64)>,
    /// Name of the GPU work that ends at `t0` (what stalled).
    pub prev_name: Option<u32>,
    /// Name of the GPU work that starts at `t1` (what resumes).
    pub next_name: Option<u32>,
    /// A GPU track of this scope, for scroll-into-view on row click.
    pub focus_track: u32,
}

/// Gaps-tab detection filter (per pane): real traces often carry
/// seconds-long idle spans that are profiler artifacts (ramp-up, ranks
/// parked for the whole capture), which would otherwise fill the entire
/// top-N list. All fields are applied in `compute_gaps_in` BEFORE the
/// top-N cut, so filtered-out artifacts don't consume list slots.
#[derive(Clone, PartialEq, Default)]
pub struct GapFilter {
    /// Restrict detection to [t0, t1]: gaps are clipped to the window and
    /// ones falling outside it are dropped. Set from the current view in
    /// the Gaps tab ("Limit to view"), so zooming into the known-good
    /// region then re-detecting is the intended flow. None = whole trace.
    /// While a timeline selection is active it is overridden by the
    /// selection's own range (see `Pane::ensure_gaps`).
    pub window: Option<(f64, f64)>,
    /// Restrict detection to these rank scopes, keyed like
    /// `compute_gaps_in` scopes them (`parse_rank(...).unwrap_or(MAX)`).
    /// Set from a finished timeline selection while it's active; None =
    /// every rank.
    pub scopes: Option<Vec<usize>>,
}

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

/// One merged rank-group's Tetris packing over its events, in ts-sorted
/// parallel arrays. View-INDEPENDENT: packed once over the group's whole
/// time span (keyed only on hidden_names/track_order, see
/// `Pane::merge_cache_key`), so panning and zooming never re-pack — the
/// render loop binary-searches `ts`/`prefix_max_dur` for the visible slice
/// instead. That view-independence is also what fixes left-edge popping
/// during a pan: a window-relative pack re-depthed every event straddling
/// the left edge as its off-screen influencers (events overlapping it but
/// ending before view.t0) dropped out of the collect window, visibly
/// re-laning and re-stretching them frame after frame.
#[derive(Default)]
pub struct MergedPack {
    /// Event start times, ascending (the sort key of every other array).
    pub ts: Vec<f64>,
    /// ts + dur, parallel to `ts` (render x1 and overlap checks).
    pub ends: Vec<f64>,
    /// Running max of durations over `ts[..=i]` — same shape as
    /// `Track::prefix_max_dur`, feeds the bisect_overlap-style lower bound
    /// when windowing the visible slice.
    pub prefix_max_dur: Vec<f64>,
    /// (track_idx, event_idx) back-references into the trace. Only needed
    /// for the wide-event render path (hover/selection/full event data) —
    /// the sub-pixel path reads `name` below instead of chasing these into
    /// track memory.
    pub refs: Vec<(u32, u32)>,
    /// Interned event-name index per event, parallel to `ts`. Duplicated
    /// from the referenced events so the sub-pixel render pass (the common
    /// case at wide zoom, millions of events) touches this compact array
    /// sequentially instead of pointer-chasing `refs` into scattered track
    /// storage just to pick a color.
    pub name: Vec<u32>,
    /// Packed Tetris depth per event, parallel to `ts`.
    pub depth: Vec<u16>,
    /// (lo, hi) stretched depth range per event (see `stretch_bounds`),
    /// parallel to `ts`.
    pub stretch: Vec<(u16, u16)>,
    /// Max packed depth over the whole group (>= 1) — drives row height.
    pub max_depth: u16,
}

impl MergedPack {
    /// First index whose event could overlap `t` (conservative lower bound:
    /// `prefix_max_dur` accounts for the longest event seen up to each
    /// index, not that event's own duration — same caveat as
    /// `types::bisect_overlap`, callers skip non-overlapping entries).
    pub fn overlap_start(&self, t: f64) -> usize {
        let (mut lo, mut hi) = (0, self.ts.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.ts[mid] + self.prefix_max_dur[mid] < t { lo = mid + 1; } else { hi = mid; }
        }
        lo
    }

    /// One-past-last index whose event starts at or before `t`.
    pub fn ts_end(&self, t: f64) -> usize {
        self.ts.partition_point(|&ts| ts <= t)
    }
}

pub struct MergedGpuGroup {
    pub tracks: Vec<usize>,
    /// The view-independent packed representation (see `MergedPack`).
    /// Shared behind an Arc with the per-pane `PaneGeom` snapshot — the
    /// snapshot is a refcount bump, and the (rare) rebuild swaps in a fresh
    /// Arc rather than mutating under live readers.
    pub pack: Arc<MergedPack>,
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
    /// The view-independent packed events exactly as drawn in the merged
    /// row (Tetris packing already stripped grandparent wrappers and hidden
    /// names, so a selection iterating these matches the rendered row
    /// instead of sweeping in ghost events). Shares the same Arc as
    /// `MergedGpuGroup::pack` — snapshotting is a refcount bump, and the
    /// selection read (windowed just like the render loop) sees exactly
    /// what the render loop drew.
    pub pack: Arc<MergedPack>,
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
}
