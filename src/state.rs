use crate::loader::{load_trace_progressive, load_multi_progressive};
use crate::types::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

/// Fires off a trace-load job that reports its result via an mpsc channel
/// (polled every frame by `poll_loading`), mirroring the old
/// `std::thread::spawn(move || { ... })` background-job shape.
///
/// Native: `rayon::spawn` onto rayon's global pool — real worker threads
/// continuously service that pool's injector queue, so this runs
/// asynchronously in the background exactly like the `std::thread::spawn` it
/// replaces.
///
/// wasm32: NOT `rayon::spawn`. Confirmed empirically (a throwaway
/// wasm-bindgen + Node harness) that on wasm's no-real-threads fallback, a
/// bare `rayon::spawn` job is never picked up — not synchronously, and not
/// even as a side effect of later, unrelated rayon calls — because nothing
/// ever runs a worker-thread loop to service the global queue there (unlike
/// `join`/`scope`/`par_iter`, where the *calling* thread itself executes the
/// work as part of the blocking call). Shipping `rayon::spawn` verbatim here
/// would silently hang every wasm trace load forever (the channel never
/// receives anything, so `poll_loading` waits indefinitely). Until real wasm
/// threading lands (a later phase — wasm-bindgen-rayon + an atomics build),
/// there's no way to background this on wasm at all, so run it synchronously
/// inline instead: blocks the tab for the load's duration, but at least
/// completes and reports a result instead of hanging.
fn spawn_load_job(job: impl FnOnce() + Send + 'static) {
    #[cfg(not(target_arch = "wasm32"))]
    rayon::spawn(job);
    #[cfg(target_arch = "wasm32")]
    job();
}

/// Interned name indices matching vLLM's per-generation `execute_context_*`
/// wrapper span. Pulled out as a pure function so it's testable without
/// spinning up the loader pipeline, and so `poll_loading` can compute it once
/// per trace load instead of every toolbar frame.
pub(crate) fn find_exec_context_names(names: &[String]) -> Vec<usize> {
    names.iter().enumerate()
        .filter(|(_, n)| n.contains("execute_context"))
        .map(|(i, _)| i)
        .collect()
}

pub(crate) fn parse_rank(label: &str) -> Option<usize> {
    let l = label.trim_start();
    if l.starts_with("Rank ") {
        l[5..].split_once(' ').and_then(|(n, _)| n.parse().ok())
    } else {
        None
    }
}

/// Default view-level track order: a stable sort by numeric rank (via
/// `parse_rank`), not by track index or label text. Deliberately computed
/// here rather than relying on `merge_traces` writing tracks in the right
/// order to begin with — this way the *view* is always correct regardless
/// of how the underlying `Trace` was produced, including a `.tvcache`
/// export that was itself built and saved before a rank-ordering fix
/// existed (the label text it stored still has the numeric rank in it;
/// only the on-disk *track order* predates the fix). A trace with no
/// "Rank N" labels at all (not a multi-rank merge) gets the same `None`
/// key for every track, so the stable sort leaves it in its original order.
pub(crate) fn default_track_order(tracks: &[Track]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..tracks.len()).collect();
    order.sort_by_key(|&i| parse_rank(&tracks[i].label).unwrap_or(usize::MAX));
    order
}

pub struct Pane {
    pub trace: Option<Trace>,
    pub view: View,
    pub loading: Option<mpsc::Receiver<Result<Trace, String>>>,
    pub error: Option<String>,
    pub trace_path: String,
    pub show_cpu: bool,
    pub merge_gpu: bool,
    pub selected: Option<EventRef>,
    pub multi_select_name: Option<u32>,
    pub search: String,
    pub search_focus: bool,
    pub select_all_pending: bool,
    pub search_mask: Vec<bool>,
    pub search_nav: Vec<(f64, u32, u32)>,
    pub search_cursor: usize,
    pub prev_search: String,
    pub selection: Option<[f64; 4]>,
    pub finished_sel: Option<[f64; 4]>,
    /// The concrete `(track_idx, event_idx)` set `finished_sel` refers to,
    /// resolved once when the drag finishes (see `capture_sel_events`). Every
    /// consumer of `finished_sel` after that point (stats, extract, copy, the
    /// rendered highlight) reads this frozen set instead of re-deriving track
    /// membership from `finished_sel`'s raw Y range against whatever the
    /// CURRENT layout happens to be — otherwise toggling Show CPU, reordering
    /// tracks, or resizing the bottom panel (which can change track heights)
    /// silently changes which events are "selected".
    pub finished_sel_events: std::collections::HashSet<(u32, u32)>,
    pub selection_stats: Vec<SelectionEntry>,
    pub selection_dirty: bool,
    pub sel_mask: Vec<bool>,
    pub collapsed: Vec<bool>,
    pub track_scales: Vec<f32>,
    pub even_spacing: bool,
    pub geom: PaneGeom,
    pub hidden_names: Vec<bool>,
    /// (view.t0 bits, view.t1 bits, hidden_names snapshot, track_order
    /// snapshot) from the last time the merged multi-rank view's per-group
    /// Tetris packing (`build_merged_group_events`) actually ran for this
    /// pane. Rebuilding it is O(events in view) per rank group — measured at
    /// ~11ms for a 28-rank, 468K-event trace fully zoomed out — and it reran
    /// unconditionally on every redraw (i.e. every mouse-move) even when
    /// nothing about the view had changed. Owned per-pane, not on the shared
    /// `DrawBuf`, for the same reason `sort_cache_key`/`detail_hist_key`
    /// moved there: only one pane renders per frame (see the tab strip in
    /// main.rs), so a shared cache would compare against whichever *other*
    /// pane last rendered.
    pub merge_cache_key: Option<(u64, u64, Vec<bool>, Vec<usize>)>,
    /// The merged multi-rank view's per-rank-group row data (packed events +
    /// max depth), validated against `merge_cache_key` above. This must live
    /// here, not on the shared `DrawBuf` (where it lived before it was
    /// cached) — once frame-to-frame reuse was introduced, a pane whose own
    /// `merge_cache_key` still matched could otherwise silently read back
    /// whatever a *different* pane's render had just overwritten this with,
    /// panicking on (track_idx, event_idx) pairs that don't exist in this
    /// pane's own trace. Harmless before caching, since it was always fully
    /// rebuilt within the same frame it was read, regardless of which pane's
    /// turn it was.
    pub merged_gpu_groups: Vec<MergedGpuGroup>,
    /// Seconds since the merged view's Tetris packing was last actually
    /// rebuilt for a view-range-only change (an active pan/zoom, which can
    /// invalidate `merge_cache_key` every single frame). A continuous zoom
    /// still costs a full rebuild each time this crosses `MERGE_REBUILD_THROTTLE_S`
    /// in `draw_timeline`, but not on every frame in between — see the
    /// merge_cache_valid computation there for the full reasoning. Reset to
    /// 0 whenever a rebuild actually happens (throttled or not), so it only
    /// ever measures "time since we were last accurate".
    pub merge_throttle_elapsed: f32,
    /// Interned name indices matching "execute_context" — vLLM's per-generation
    /// wrapper span. Computed once per trace load (see `poll_loading`), not
    /// per toolbar frame, since it only depends on `trace.names`.
    pub exec_context_names: Vec<usize>,
    pub pending_tab: Option<BottomTab>,
    /// Track index whose row draw_timeline should scroll into view as part of an
    /// in-flight search zoom. Consumed (cleared) on the next timeline draw.
    pub pending_focus: Option<u32>,
    pub sort_col: usize,
    pub sort_asc: bool,
    /// (generation, sort_col, sort_asc, row count, is_individual_mode) from
    /// the last time `sort_idx` was actually recomputed for THIS pane. Owned
    /// per-pane, not on the shared `DrawBuf` — only one pane renders per
    /// frame (see the tab strip in main.rs), so a shared cache would compare
    /// against whichever *other* pane last rendered and could spuriously
    /// "match" and show that pane's stale rows instead of recomputing.
    pub sort_cache_key: Option<(u64, usize, bool, usize, bool)>,
    pub sort_idx: Vec<usize>,
    /// (event name, show_cpu, bucket count) the Detail tab's
    /// duration-distribution histogram was last computed for, and the
    /// cached bucket counts — same per-pane-ownership reasoning as
    /// `sort_cache_key` (a name id is only unique within this pane's own
    /// trace, so two different open traces routinely reuse the same low
    /// ids). Bucket count is part of the key because it now scales with the
    /// panel's width (see `DETAIL_HIST_TARGET_BAR_W`), so a resize needs a
    /// rebucket too, not just a name/visibility change.
    pub detail_hist_key: Option<(u32, bool, usize)>,
    pub detail_hist_bins: Vec<u32>,
    pub detail_hist_min: f64,
    pub detail_hist_max: f64,
    pub detail_hist_mean: f64,
    pub detail_hist_median: f64,
    pub sel_aggregate: bool,
    pub label_w: f32,
    pub sel_median: f64,
    pub sel_agg_stats: Vec<KernelStats>,
    pub sel_individual: Vec<KernelStats>,
    /// (track_idx, event_idx) parallel to `sel_individual`, so the stats table
    /// can look up each row's raw args (e.g. CUDA occupancy-limiting factor).
    pub sel_individual_refs: Vec<(u32, u32)>,
    /// Bumped every time `compute_aggregates` rebuilds `sel_agg_stats` /
    /// `sel_individual`. `draw_stats_table` uses this to skip re-sorting on
    /// redraws where the selection hasn't actually changed — the table
    /// otherwise re-sorted unconditionally on every redraw (which fires on
    /// every mouse-move event, not just clicks), measured at ~120ms for a
    /// 1M-row selection even on a plain numeric column.
    pub sel_generation: u64,
    pub track_order: Vec<usize>,
    pub auto_reload: bool,
    pub reload_paths: Vec<(usize, String)>,
    pub reload_dir: Option<String>,
    pub cache_dir: Option<String>,
    pub loading_events: Arc<AtomicUsize>,
    /// Result of the last "Export GPU" click (success message, or an
    /// error) — shown next to the button until the next click replaces it.
    #[cfg(not(target_arch = "wasm32"))]
    pub export_message: Option<(bool, String)>,
    /// Background `gh`/`git`-based gist upload started by "Share via Gist"
    /// (see `start_share_upload`) — `None` once no upload is in flight.
    #[cfg(not(target_arch = "wasm32"))]
    pub share_link_job: Option<mpsc::Receiver<Result<String, String>>>,
    /// Result of the last "Share via Gist" click (the finished `?gist=`
    /// link, or an error) — shown next to the button until replaced.
    #[cfg(not(target_arch = "wasm32"))]
    pub share_link_result: Option<(bool, String)>,
    /// "Include CPU data" checkbox next to "Share via Gist" — persisted
    /// across frames like any other UI toggle. When set, uploads the whole
    /// trace (`export_full_web`) instead of the GPU-only, args-stripped
    /// default (`export_gpu_only_web`).
    #[cfg(not(target_arch = "wasm32"))]
    pub share_include_cpu: bool,
    /// Result of the last "Sync Clocks" click (success summary, or an
    /// error) — shown next to the button until the next click replaces it.
    pub sync_message: Option<(bool, String)>,
}

impl Pane {
    pub fn new() -> Self {
        Self {
            trace: None,
            view: View::default(),
            loading: None,
            error: None,
            trace_path: String::new(),
            show_cpu: false,
            merge_gpu: true,
            selected: None,
            multi_select_name: None,
            search: String::with_capacity(256),
            search_focus: false,
            select_all_pending: false,
            search_mask: Vec::new(),
            search_nav: Vec::new(),
            search_cursor: 0,
            prev_search: String::new(),
            selection: None,
            finished_sel: None,
            finished_sel_events: std::collections::HashSet::new(),
            selection_stats: Vec::new(),
            selection_dirty: false,
            sel_mask: Vec::new(),
            collapsed: Vec::new(),
            track_scales: Vec::new(),
            even_spacing: false,
            geom: PaneGeom::default(),
            hidden_names: Vec::new(),
            merge_cache_key: None,
            merged_gpu_groups: Vec::new(),
            merge_throttle_elapsed: 0.0,
            exec_context_names: Vec::new(),
            pending_tab: Some(BottomTab::Detail),
            pending_focus: None,
            sort_col: 2,
            sort_asc: false,
            sort_cache_key: None,
            sort_idx: Vec::new(),
            detail_hist_key: None,
            detail_hist_bins: Vec::new(),
            detail_hist_min: 0.0,
            detail_hist_max: 0.0,
            detail_hist_mean: 0.0,
            detail_hist_median: 0.0,
            sel_aggregate: true,
            label_w: LABEL_W,
            sel_median: 0.0,
            sel_agg_stats: Vec::new(),
            sel_individual: Vec::new(),
            sel_individual_refs: Vec::new(),
            sel_generation: 0,
            track_order: Vec::new(),
            auto_reload: false,
            reload_paths: Vec::new(),
            reload_dir: None,
            cache_dir: None,
            loading_events: Arc::new(AtomicUsize::new(0)),
            #[cfg(not(target_arch = "wasm32"))]
            export_message: None,
            #[cfg(not(target_arch = "wasm32"))]
            share_link_job: None,
            #[cfg(not(target_arch = "wasm32"))]
            share_link_result: None,
            #[cfg(not(target_arch = "wasm32"))]
            share_include_cpu: false,
            sync_message: None,
        }
    }

    pub fn has_trace(&self) -> bool { self.trace.is_some() }

    /// Exports this pane's GPU tracks (timings only, args stripped — see
    /// `loader::export_gpu_only`) to a sibling "<name>-gpu-only/" folder next
    /// to wherever the trace was actually opened from. `reload_paths` (set
    /// by every native open path — `open`, `open_multi`, and CLI/drag-drop
    /// in main.rs) always has at least one real file path even for a
    /// multi-rank trace, unlike `trace_path`, which is a synthetic "N ranks:
    /// ..." label in that case, not a real filesystem path.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn export_gpu_only(&self) -> Result<String, String> {
        let trace = self.trace.as_ref().ok_or_else(|| "no trace loaded".to_string())?;
        let sample_path = self.reload_paths.first().map(|(_, p)| p.as_str())
            .ok_or_else(|| "no source file path available for this trace".to_string())?;
        let base = std::path::Path::new(sample_path);
        let parent = base.parent().unwrap_or_else(|| std::path::Path::new("."));

        let stem = match &self.reload_dir {
            Some(dir) => std::path::Path::new(dir).file_name()
                .and_then(|s| s.to_str()).unwrap_or("trace").to_string(),
            None => {
                let s = base.file_stem().and_then(|s| s.to_str()).unwrap_or("trace");
                // file_stem() on "trace.json.gz" only strips ".gz" -> "trace.json";
                // peel one more layer so the export folder name reads cleanly.
                s.strip_suffix(".json").or_else(|| s.strip_suffix(".tar")).unwrap_or(s).to_string()
            }
        };

        let out_path = parent.join(format!("{stem}-gpu-only")).join("gpu-only.tvcache.xz");
        let out_str = out_path.to_str().ok_or_else(|| "non-UTF-8 destination path".to_string())?;
        crate::loader::export_gpu_only(trace, out_str)?;
        Ok(out_path.to_string_lossy().into_owned())
    }

    /// Sibling-folder naming shared by `export_gpu_only_web`/`export_full_web`:
    /// derived from the source path (or the merge directory's own name for a
    /// multi-rank trace, since `reload_paths` for those points at individual
    /// rank files, not the folder the user actually opened).
    #[cfg(not(target_arch = "wasm32"))]
    fn export_web_paths(&self) -> Result<(std::path::PathBuf, String), String> {
        let sample_path = self.reload_paths.first().map(|(_, p)| p.as_str())
            .ok_or_else(|| "no source file path available for this trace".to_string())?;
        let base = std::path::Path::new(sample_path);
        let parent = base.parent().unwrap_or_else(|| std::path::Path::new(".")).to_path_buf();

        let stem = match &self.reload_dir {
            Some(dir) => std::path::Path::new(dir).file_name()
                .and_then(|s| s.to_str()).unwrap_or("trace").to_string(),
            None => {
                let s = base.file_stem().and_then(|s| s.to_str()).unwrap_or("trace");
                s.strip_suffix(".json").or_else(|| s.strip_suffix(".tar")).unwrap_or(s).to_string()
            }
        };
        Ok((parent, stem))
    }

    /// Same as `export_gpu_only`, but via `loader::export_gpu_only_web`
    /// (plain layout, gzip instead of xz) so the result can be uploaded
    /// somewhere with permissive CORS and opened through the web build's
    /// `?src=<url>` shareable-link loader, which has no LZMA decoder and
    /// can't reverse `export_gpu_only`'s columnar/kernel-grouped encoding.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn export_gpu_only_web(&self) -> Result<String, String> {
        let trace = self.trace.as_ref().ok_or_else(|| "no trace loaded".to_string())?;
        let (parent, stem) = self.export_web_paths()?;
        let out_path = parent.join(format!("{stem}-gpu-only")).join("gpu-only-web.tvcache.gz");
        let out_str = out_path.to_str().ok_or_else(|| "non-UTF-8 destination path".to_string())?;
        crate::loader::export_gpu_only_web(trace, out_str)?;
        Ok(out_path.to_string_lossy().into_owned())
    }

    /// Exports the whole trace (CPU and GPU tracks, args left intact — see
    /// `loader::export_full_web`) rather than `export_gpu_only_web`'s
    /// deliberately stripped-down version. Much larger, but complete —
    /// for when you want to share the actual trace, not just GPU timings.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn export_full_web(&self) -> Result<String, String> {
        let trace = self.trace.as_ref().ok_or_else(|| "no trace loaded".to_string())?;
        let (parent, stem) = self.export_web_paths()?;
        let out_path = parent.join(format!("{stem}-full-export")).join("full-trace.tvcache");
        let out_str = out_path.to_str().ok_or_else(|| "non-UTF-8 destination path".to_string())?;
        crate::loader::export_full_web(trace, out_str)?;
        Ok(out_path.to_string_lossy().into_owned())
    }

    /// Corrects inter-node clock skew across this pane's merged multi-rank
    /// trace — see `loader::sync_multi_rank_clocks` for the full algorithm
    /// (a Rust port of `sync_traces.py`, applied in memory instead of
    /// writing shifted files to disk). Only meaningful for an actual
    /// multi-rank merge. Deliberately leaves the view's pan/zoom untouched —
    /// the user is likely already looking at the region they care about
    /// (e.g. the very marker event this aligns on) and a shift on the order
    /// of hundreds of microseconds shouldn't visibly move anything at
    /// whatever zoom level made that visible in the first place.
    ///
    /// Prefers `reload_paths` (the live per-rank source files, when this
    /// pane was opened as a directory/rank list) but falls back to the
    /// merged trace's own `rank_paths` — the same rank/filename pairs,
    /// persisted into the `.tvcache` format at export time — so this still
    /// works after reopening an already-merged cache file directly, when
    /// there's no directory of per-rank JSON to re-derive them from — the
    /// latter is the only path available on wasm (`reload_paths` is always
    /// empty there), but it's enough: any `.tvcache` merged natively and
    /// then opened in the browser carries its `rank_paths` with it.
    pub fn sync_clocks(&mut self) -> Result<String, String> {
        self.sync_clocks_on(crate::loader::DEFAULT_SYNC_MARKER)
    }

    /// Same alignment as `sync_clocks`, but on an arbitrary marker instead of
    /// the hardcoded DeepEP combine kernel — used by the timeline's
    /// right-click "sync ranks to this event" context menu, where `marker`
    /// is the exact name of whichever kernel instance the user clicked.
    pub fn sync_clocks_on(&mut self, marker: &str) -> Result<String, String> {
        let rank_paths: Vec<(usize, String)> = if self.reload_paths.len() >= 2 {
            self.reload_paths.clone()
        } else {
            let trace = self.trace.as_ref().ok_or_else(|| "no trace loaded".to_string())?;
            if trace.rank_paths.len() < 2 {
                return Err("not a multi-rank trace".to_string());
            }
            trace.rank_paths.clone()
        };
        let trace = self.trace.as_mut().ok_or_else(|| "no trace loaded".to_string())?;
        let result = crate::loader::sync_multi_rank_clocks(trace, &rank_paths, marker);
        // Shifts event timestamps in place (same events/indices, so this
        // isn't the out-of-bounds risk a reload is) but the merged view's
        // cached Tetris packing is order-dependent on those timestamps, and
        // the cache key doesn't otherwise change just because clocks synced.
        if result.is_ok() { self.merge_cache_key = None; }
        result
    }

    /// Exports (via `export_gpu_only_web`, or `export_full_web` when
    /// `include_cpu` is set) and, on a background thread, uploads the
    /// result to a new secret GitHub gist through the locally authenticated
    /// `gh` CLI — no browser OAuth flow or pasted token involved, since
    /// this only runs on native where `gh`'s own stored credentials are
    /// already available. The export itself is fast (local disk only) and
    /// stays synchronous; only the network upload (`loader::upload_gist`)
    /// is backgrounded, since that can take a few seconds and shouldn't
    /// freeze the UI.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn start_share_upload(&mut self, include_cpu: bool) {
        if self.share_link_job.is_some() { return; }
        let path = match if include_cpu { self.export_full_web() } else { self.export_gpu_only_web() } {
            Ok(p) => p,
            Err(e) => { self.share_link_result = Some((false, e)); return; }
        };
        let (tx, rx) = mpsc::channel();
        self.share_link_job = Some(rx);
        self.share_link_result = Some((true, "Uploading to gist...".to_string()));
        rayon::spawn(move || {
            let _ = tx.send(crate::loader::upload_gist(&path));
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn poll_share_upload(&mut self) {
        let rx = match &self.share_link_job {
            Some(rx) => rx,
            None => return,
        };
        match rx.try_recv() {
            Ok(Ok(link)) => {
                self.share_link_result = Some((true, link));
                self.share_link_job = None;
            }
            Ok(Err(e)) => {
                self.share_link_result = Some((false, e));
                self.share_link_job = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => { self.share_link_job = None; }
        }
    }

    pub fn loading_progress_text(&self) -> String {
        let n = self.loading_events.load(Ordering::Relaxed);
        if n == 0 {
            "Loading: reading file...".to_string()
        } else if n < 1_000 {
            format!("Loading: {} events...", n)
        } else if n < 1_000_000 {
            format!("Loading: {:.1}K events...", n as f64 / 1_000.0)
        } else {
            format!("Loading: {:.2}M events...", n as f64 / 1_000_000.0)
        }
    }

    pub fn clear_selection(&mut self) {
        self.selection = None;
        self.finished_sel = None;
        self.finished_sel_events.clear();
        self.selection_stats.clear();
        self.sel_mask.clear();
        self.sel_median = 0.0;
        self.sel_agg_stats.clear();
        self.sel_individual.clear();
        self.sel_individual_refs.clear();
    }

    pub fn clear_search(&mut self) {
        self.search.clear();
        self.prev_search.clear();
        self.search_mask.clear();
        self.search_nav.clear();
        self.search_cursor = 0;
    }

    pub fn finish_selection(&mut self, buf: &mut DrawBuf) {
        self.finished_sel = self.selection;
        self.finished_sel_events = self.selection.map(|s| self.capture_sel_events(s).into_iter().collect()).unwrap_or_default();
        self.selection = None;
        self.rebuild_selection_stats(buf);
        self.sel_mask.clear();
    }

    /// Resolve a raw pixel-space selection rectangle into the concrete set of
    /// `(track_idx, event_idx)` pairs it covers, using `self.geom`/
    /// `hidden_names`/`show_cpu` exactly as they are right now (drag-finish
    /// time) — full precision, including which merged-row depth lane the drag
    /// covered. The result is frozen from this point on: nothing that happens
    /// afterward (Show CPU toggle, track reorder/resize, hiding a name) can
    /// retroactively add, remove, or reassign what this selection refers to.
    /// That freezing is the fix — the old code re-derived "what's selected"
    /// from the raw Y range against whatever the CURRENT layout was on every
    /// read, which silently changed the answer as the layout changed.
    pub(crate) fn capture_sel_events(&self, sel: [f64; 4]) -> Vec<(u32, u32)> {
        let trace = match &self.trace { Some(t) => t, None => return Vec::new() };
        let (s0, s1) = if sel[0] <= sel[1] { (sel[0], sel[1]) } else { (sel[1], sel[0]) };
        let (y0, y1) = if sel[2] <= sel[3] { (sel[2] as f32, sel[3] as f32) } else { (sel[3] as f32, sel[2] as f32) };
        let mut out = Vec::new();
        for vi in 0..self.geom.visible.len() {
            let track_top = self.geom.y_offsets[vi];
            let track_h = self.geom.heights[vi];
            let track_bot = track_top + track_h;
            if track_bot < y0 || track_top > y1 { continue; }

            if let Some(group) = self.geom.merged.iter().find(|g| g.vi == vi) {
                // Match the rendered merged row: iterate the packed events
                // (wrappers stripped) and apply the renderer's depth/y test.
                let max_depth = group.events.iter().map(|&(_, _, d)| d).max().map(|d| d + 1).unwrap_or(1);
                let sub_h = track_h / max_depth.max(1) as f32;
                for &(ti32, ei32, depth) in &group.events {
                    let ev = &trace.tracks[ti32 as usize].events[ei32 as usize];
                    if !(ev.ts + ev.dur >= s0 && ev.ts <= s1) { continue; }
                    let ev_top = track_top + depth as f32 * sub_h;
                    let ev_bot = ev_top + sub_h;
                    if ev_bot < y0 || ev_top > y1 { continue; }
                    out.push((ti32, ei32));
                }
            } else {
                let ti = self.geom.visible[vi];
                let track = &trace.tracks[ti];
                if !self.show_cpu && !track.gpu { continue; }
                let sub_h = track_h / track.max_depth.max(1) as f32;
                let start = bisect_overlap(&track.events, &track.prefix_max_dur, s0);
                let end = track.events.partition_point(|e| e.ts <= s1).max(start);
                let mut ancestor_sel = vec![false; track.max_depth as usize + 1];
                for (local_i, ev) in track.events[start..end].iter().enumerate() {
                    if self.hidden_names.get(ev.name as usize).copied().unwrap_or(false) { continue; }
                    let ev_top = track_top + ev.depth as f32 * sub_h;
                    let ev_bot = ev_top + sub_h;
                    for d in ev.depth as usize..ancestor_sel.len() { ancestor_sel[d] = false; }
                    if ev_bot < y0 || ev_top > y1 { continue; }
                    if ev.ts + ev.dur >= s0 && ev.ts <= s1 {
                        ancestor_sel[ev.depth as usize] = true;
                        if (0..ev.depth as usize).any(|d| ancestor_sel[d]) { continue; }
                        out.push((ti as u32, (start + local_i) as u32));
                    }
                }
            }
        }
        out
    }

    pub fn open(&mut self, path: String) {
        let (tx, rx) = mpsc::channel();
        self.loading = Some(rx);
        self.error = None;
        self.trace_path = path.clone();
        self.reload_paths = vec![(0, path.clone())];
        self.loading_events = Arc::new(AtomicUsize::new(0));
        let counter = self.loading_events.clone();
        let cd = self.cache_dir.clone();
        spawn_load_job(move || {
            load_trace_progressive(&path, &counter, 0, &tx, cd.as_deref());
        });
    }

    /// wasm equivalent of `open`: takes bytes already read into memory (e.g.
    /// via the browser's drag-and-drop/file-picker File API) instead of a
    /// filesystem path — there's no mmap/filesystem on wasm32 to open a path
    /// against. `name` is the dropped/picked file's name (e.g.
    /// `File::name()`), used only to sniff the format (.tvcache/.tar.gz/.gz
    /// vs plain JSON), same role a real path's suffix plays natively.
    #[cfg(target_arch = "wasm32")]
    pub fn open_from_bytes(&mut self, name: String, bytes: Vec<u8>) {
        let (tx, rx) = mpsc::channel();
        self.loading = Some(rx);
        self.error = None;
        self.trace_path = name.clone();
        self.loading_events = Arc::new(AtomicUsize::new(0));
        let counter = self.loading_events.clone();
        spawn_load_job(move || {
            crate::loader::load_trace_from_bytes_progressive(&name, bytes, &counter, &tx);
        });
    }

    /// wasm equivalent of `open_multi`: each rank's bytes are already in
    /// memory (e.g. from walking a dropped folder via the browser's File and
    /// Directory Entries API — see main.rs), so there's no path to re-read
    /// from, and no `reload_dir`/`reload_paths` equivalent since a browser
    /// drop can't be "re-opened" the way a filesystem directory can.
    #[cfg(target_arch = "wasm32")]
    pub fn open_multi_from_bytes(&mut self, rank_named_bytes: Vec<(usize, String, Vec<u8>)>) {
        let (tx, rx) = mpsc::channel();
        let n = rank_named_bytes.len();
        let fname = rank_named_bytes[0].1.rsplit('/').next().unwrap_or(&rank_named_bytes[0].1);
        let prefix = fname.find("-rank-").map(|p| fname[..p].to_string())
            .unwrap_or_else(|| "multi-rank".to_string());
        self.trace_path = format!("{} ranks: {}", n, prefix);
        self.loading = Some(rx);
        self.error = None;
        self.loading_events = Arc::new(AtomicUsize::new(0));
        let counter = self.loading_events.clone();
        spawn_load_job(move || {
            crate::loader::load_multi_from_bytes_progressive(rank_named_bytes, &counter, &tx);
        });
    }

    pub fn open_multi(&mut self, rank_paths: Vec<(usize, String)>) {
        let (tx, rx) = mpsc::channel();
        let n = rank_paths.len();
        let prefix = std::path::Path::new(&rank_paths[0].1)
            .file_name()
            .and_then(|f| f.to_str())
            .and_then(|f| f.find("-rank-").map(|p| &f[..p]))
            .unwrap_or("multi-rank")
            .to_string();
        self.trace_path = format!("{} ranks: {}", n, prefix);
        self.reload_paths = rank_paths.clone();
        self.loading = Some(rx);
        self.error = None;
        self.loading_events = Arc::new(AtomicUsize::new(0));
        let counter = self.loading_events.clone();
        let tpf = (std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4) / n).max(2);
        let cd = self.cache_dir.clone();
        spawn_load_job(move || {
            load_multi_progressive(rank_paths, &counter, tpf, &tx, cd.as_deref(), false);
        });
    }

    pub fn reload(&mut self) {
        if self.loading.is_some() { return; }
        if let Some(dir) = &self.reload_dir {
            let (groups, standalone) = crate::loader::detect_rank_groups(&[dir.clone()]);
            let mut all_paths: Vec<(usize, String)> = Vec::new();
            for (group, _dir) in groups {
                all_paths.extend(group);
            }
            for (i, (path, _dir)) in standalone.into_iter().enumerate() {
                let rank = all_paths.len() + i;
                all_paths.push((rank, path));
            }
            if all_paths.is_empty() { return; }
            self.reload_paths = all_paths;
        }
        if self.reload_paths.is_empty() { return; }
        let (tx, rx) = mpsc::channel();
        self.loading = Some(rx);
        self.loading_events = Arc::new(AtomicUsize::new(0));
        let counter = self.loading_events.clone();
        let paths = self.reload_paths.clone();
        let cd = self.cache_dir.clone();
        if paths.len() == 1 {
            let path = paths[0].1.clone();
            spawn_load_job(move || {
                load_trace_progressive(&path, &counter, 0, &tx, cd.as_deref());
            });
        } else {
            let tpf = (std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4) / paths.len()).max(1);
            spawn_load_job(move || {
                load_multi_progressive(paths, &counter, tpf, &tx, cd.as_deref(), true);
            });
        }
    }

    pub fn poll_loading(&mut self) {
        let rx = match &self.loading {
            Some(rx) => rx,
            None => return,
        };
        match rx.try_recv() {
            Ok(Ok(trace)) => {
                let is_reload = self.trace.is_some();
                if is_reload {
                    let n_tracks = trace.tracks.len();
                    let n_names = trace.names.len();
                    self.collapsed.resize(n_tracks, false);
                    self.track_scales.resize(n_tracks, 1.0);
                    if self.track_order.len() != n_tracks {
                        self.track_order = default_track_order(&trace.tracks);
                    }
                    self.hidden_names.resize(n_names, false);
                    self.search_mask.clear();
                    self.search_nav.clear();
                    self.search_cursor = 0;
                    // The reloaded trace may have fewer events; a retained
                    // EventRef would index out of bounds in the Detail panel.
                    self.selected = None;
                    self.selection_stats.clear();
                    // Same reasoning: the merged-view Tetris-packing cache
                    // stores raw (track_idx, event_idx) pairs keyed on
                    // (view range, hidden names, track order) — none of
                    // which necessarily change on a reload, so a stale cache
                    // hit would keep pointing at indices that no longer
                    // exist in the new trace's (possibly smaller) tracks.
                    self.merge_cache_key = None;
                    let old_stats = std::mem::take(&mut self.trace.as_mut().unwrap().stats);
                    self.trace = Some(trace);
                    self.trace.as_mut().unwrap().stats = old_stats;
                    if !self.search.is_empty() { self.rebuild_search(); }
                } else {
                    let pad = trace.max_ts * 0.02;
                    self.view.t0 = -pad;
                    self.view.t1 = trace.max_ts + pad;
                    self.view.scroll_y = 0.0;
                    self.collapsed = vec![false; trace.tracks.len()];
                    self.track_scales = vec![1.0; trace.tracks.len()];
                    self.track_order = default_track_order(&trace.tracks);
                    self.hidden_names = vec![false; trace.names.len()];
                    self.trace = Some(trace);
                }
                // Computed once per load instead of every toolbar frame — the
                // "Hide Execute Context" button's name list only ever needs
                // this trace's `names`, which don't change until the next load.
                self.exec_context_names = self.trace.as_ref()
                    .map(|t| find_exec_context_names(&t.names))
                    .unwrap_or_default();
            }
            Ok(Err(e)) => {
                self.error = Some(e);
                self.loading = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.loading = None;
            }
        }
    }

    pub fn rebuild_search(&mut self) {
        let trace = match &self.trace {
            Some(t) => t,
            None => return,
        };
        let q = self.search.trim().to_ascii_lowercase();
        self.search_mask.clear();
        self.search_mask.resize(trace.names.len(), false);
        self.search_nav.clear();
        self.search_cursor = 0;

        if q.is_empty() { return; }

        for (i, name) in trace.names.iter().enumerate() {
            if name.to_ascii_lowercase().contains(&q) {
                self.search_mask[i] = true;
            }
        }

        for (ti, track) in trace.tracks.iter().enumerate() {
            for (ei, ev) in track.events.iter().enumerate() {
                if self.search_mask[ev.name as usize] {
                    self.search_nav.push((ev.ts, ti as u32, ei as u32));
                }
            }
        }
        self.search_nav.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    pub fn rebuild_selection_stats(&mut self, buf: &mut DrawBuf) {
        let t_start = crate::time::Instant::now();
        if self.trace.is_none() { return; }
        if self.finished_sel.is_none() {
            self.selection_stats.clear(); self.sel_agg_stats.clear(); self.sel_individual.clear(); self.sel_individual_refs.clear(); self.sel_median = 0.0;
            return;
        }
        // finished_sel_events is the frozen (track_idx, event_idx) set
        // resolved when the drag finished (see capture_sel_events) — reading
        // it directly instead of re-deriving track membership from the raw
        // selection rectangle keeps this stable across later layout changes
        // (Show CPU toggle, track reorder/resize).
        let trace = self.trace.as_ref().unwrap();

        let map = &mut buf.sel_map;
        for v in map.values_mut() { v.0 = 0; v.1 = 0.0; v.2.clear(); v.3.clear(); }

        for &(ti, ei) in &self.finished_sel_events {
            let ev = &trace.tracks[ti as usize].events[ei as usize];
            let e = map.entry(ev.name).or_insert((0, 0.0, Vec::new(), Vec::new()));
            e.0 += 1;
            e.1 += ev.dur;
            e.2.push(ev.dur);
            e.3.push((ti, ei));
        }
        self.selection_stats.clear();
        for (&name, (count, total_dur, durations, event_refs)) in map.iter_mut() {
            if *count == 0 { continue; }
            self.selection_stats.push(SelectionEntry {
                name, count: *count, total_dur: *total_dur,
                durations: std::mem::take(durations),
                event_refs: std::mem::take(event_refs),
            });
        }
        self.selection_stats.sort_unstable_by(|a, b| b.total_dur.partial_cmp(&a.total_dur).unwrap().then(a.name.cmp(&b.name)));
        let ev_count: u32 = self.selection_stats.iter().map(|e| e.count).sum();
        eprintln!("  select: {:.1}ms ({} events, {} names)", t_start.elapsed().as_secs_f64() * 1000.0, ev_count, self.selection_stats.len());

        let t_agg = crate::time::Instant::now();
        self.compute_aggregates();
        eprintln!("  aggregate: {:.1}ms ({} agg, {} individual)", t_agg.elapsed().as_secs_f64() * 1000.0, self.sel_agg_stats.len(), self.sel_individual.len());
    }

    pub fn extract_selection_events(&self) -> Vec<(String, f64)> {
        let trace = match &self.trace {
            Some(t) => t,
            None => return Vec::new(),
        };
        if self.finished_sel.is_none() { return Vec::new(); }

        let mut events: Vec<(f64, String, f64)> = self.finished_sel_events.iter().map(|&(ti, ei)| {
            let ev = &trace.tracks[ti as usize].events[ei as usize];
            (ev.ts, trace.names[ev.name as usize].clone(), ev.dur)
        }).collect();
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        events.into_iter().map(|(_, name, dur)| (name, dur)).collect()
    }

    pub fn copy_selection_text(&self) -> Option<String> {
        let trace = self.trace.as_ref()?;
        let mut events: Vec<(f64, &str, f64)> = Vec::new();

        if self.finished_sel.is_some() {
            for &(ti, ei) in &self.finished_sel_events {
                let ev = &trace.tracks[ti as usize].events[ei as usize];
                events.push((ev.ts, &trace.names[ev.name as usize], ev.dur));
            }
        } else if let Some(name_id) = self.multi_select_name {
            for track in &trace.tracks {
                if !self.show_cpu && !track.gpu { continue; }
                for ev in &track.events {
                    if ev.name == name_id {
                        events.push((ev.ts, &trace.names[ev.name as usize], ev.dur));
                    }
                }
            }
        } else if let Some(sel) = self.selected {
            if let Some(ev) = trace.tracks.get(sel.track_idx as usize)
                .and_then(|t| t.events.get(sel.event_idx as usize))
            {
                events.push((ev.ts, &trace.names[ev.name as usize], ev.dur));
            }
        }

        if events.is_empty() { return None; }
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut out = String::new();
        for (_, name, dur) in &events {
            use crate::ui::write_time;
            out.push_str(name);
            out.push_str(" (");
            write_time(&mut out, *dur);
            out.push_str(")\n");
        }
        Some(out)
    }

    pub fn select_from_search(&mut self, buf: &mut DrawBuf) {
        let trace = match &self.trace {
            Some(t) => t,
            None => return,
        };
        if !self.search_mask.iter().any(|&m| m) { return; }

        self.selected = None;
        self.multi_select_name = None;
        self.selection = None;
        self.selection_stats.clear();
        self.sel_mask.clear();

        let map = &mut buf.sel_map;
        for v in map.values_mut() { v.0 = 0; v.1 = 0.0; v.2.clear(); v.3.clear(); }

        for (ti, track) in trace.tracks.iter().enumerate() {
            if !self.show_cpu && !track.gpu { continue; }
            for (ei, ev) in track.events.iter().enumerate() {
                if self.hidden_names.get(ev.name as usize).copied().unwrap_or(false) { continue; }
                if self.search_mask[ev.name as usize] {
                    let e = map.entry(ev.name).or_insert((0, 0.0, Vec::new(), Vec::new()));
                    e.0 += 1;
                    e.1 += ev.dur;
                    e.2.push(ev.dur);
                    e.3.push((ti as u32, ei as u32));
                }
            }
        }
        self.selection_stats.clear();
        for (&name, (count, total_dur, durations, event_refs)) in map.iter_mut() {
            if *count == 0 { continue; }
            self.selection_stats.push(SelectionEntry {
                name, count: *count, total_dur: *total_dur,
                durations: std::mem::take(durations),
                event_refs: std::mem::take(event_refs),
            });
        }
        self.selection_stats.sort_unstable_by(|a, b| b.total_dur.partial_cmp(&a.total_dur).unwrap().then(a.name.cmp(&b.name)));
        self.compute_aggregates();
    }

    /// Build the Selection-tab stats for a double-click multi-select: every
    /// event whose name matches `multi_select_name`, across the visible tracks
    /// (respecting the CPU/GPU filter), so the tab shows that kernel's count,
    /// totals and duration distribution — matching the timeline highlight.
    pub fn rebuild_multi_select_stats(&mut self) {
        let name_id = match self.multi_select_name {
            Some(n) => n,
            None => return,
        };
        let trace = match &self.trace {
            Some(t) => t,
            None => return,
        };
        let mut count = 0u32;
        let mut total_dur = 0.0f64;
        let mut durations = Vec::new();
        let mut event_refs = Vec::new();
        for (ti, track) in trace.tracks.iter().enumerate() {
            if !self.show_cpu && !track.gpu { continue; }
            for (ei, ev) in track.events.iter().enumerate() {
                if ev.name != name_id { continue; }
                count += 1;
                total_dur += ev.dur;
                durations.push(ev.dur);
                event_refs.push((ti as u32, ei as u32));
            }
        }
        self.selection_stats.clear();
        if count > 0 {
            self.selection_stats.push(SelectionEntry { name: name_id, count, total_dur, durations, event_refs });
        }
        self.compute_aggregates();
    }

    /// Start a smooth zoom to the FIRST (earliest) search match, sized so that
    /// event fills `SEARCH_ZOOM_FILL` (80%) of the timeline width, and record
    /// its track as the pending vertical focus so draw_timeline can scroll it
    /// into view. Uses the same visibility filters as `select_from_search`.
    /// Whether the match at `(track_idx, event_idx)` is currently on screen —
    /// i.e. its track isn't CPU-hidden and its name isn't hidden. Search
    /// navigation skips matches that fail this, so it never frames an event on a
    /// track the user can't see.
    fn nav_match_visible(&self, ti: u32, ei: u32) -> bool {
        let trace = match &self.trace { Some(t) => t, None => return false };
        let track = match trace.tracks.get(ti as usize) { Some(t) => t, None => return false };
        if !self.show_cpu && !track.gpu { return false; }
        match track.events.get(ei as usize) {
            Some(ev) => !self.hidden_names.get(ev.name as usize).copied().unwrap_or(false),
            None => false,
        }
    }

    /// Frame the first visible match and sync the nav cursor to it, so a
    /// subsequent prev/next continues from the framed event rather than from a
    /// stale cursor (which could step into a hidden CPU match and appear to do
    /// nothing).
    pub fn zoom_to_search(&mut self) {
        let idx = self.search_nav.iter()
            .position(|&(_, ti, ei)| self.nav_match_visible(ti, ei));
        if let Some(idx) = idx {
            self.search_cursor = idx;
            self.zoom_to_nav_cursor();
        }
    }

    fn zoom_to_nav_cursor(&mut self) {
        let (ts, ti, ei) = self.search_nav[self.search_cursor];
        self.selected = Some(EventRef { track_idx: ti, event_idx: ei });
        let dur = self.trace.as_ref().unwrap().tracks[ti as usize].events[ei as usize].dur;
        self.start_zoom_to(ts, dur, ti as usize);
    }

    /// Start a smooth zoom that frames a single event (`ts`, `dur`) at
    /// `SEARCH_ZOOM_FILL` of the width and requests a vertical scroll to its
    /// track. The vertical target is resolved in `draw_timeline` from layout.
    pub fn start_zoom_to(&mut self, ts: f64, dur: f64, track_idx: usize) {
        let range = (dur / crate::types::SEARCH_ZOOM_FILL).max(crate::types::MIN_TIME_RANGE);
        let center = ts + dur / 2.0;
        let to_t0 = center - range / 2.0;
        let to_t1 = center + range / 2.0;
        self.view.anim = Some(crate::types::ViewAnim {
            from_t0: self.view.t0,
            from_t1: self.view.t1,
            to_t0,
            to_t1,
            from_scroll: self.view.scroll_y,
            to_scroll: self.view.scroll_y, // resolved in draw_timeline from layout
            elapsed: 0.0,
            dur: crate::types::ZOOM_ANIM_DUR,
        });
        self.pending_focus = Some(track_idx as u32);
    }

    /// Step the search-match cursor by one and smooth-zoom to that match.
    /// `forward` advances, `!forward` goes back; both wrap around. Selects the
    /// event and switches the bottom panel to the Detail tab.
    pub fn nav_search(&mut self, forward: bool) {
        let n = self.search_nav.len();
        if n == 0 { return; }
        // Step to the next/previous *visible* match, skipping hidden CPU-track
        // and hidden-name matches, wrapping around. If none are visible, do
        // nothing rather than jump to an off-screen event.
        let mut i = self.search_cursor;
        for _ in 0..n {
            i = if forward { (i + 1) % n } else { (i + n - 1) % n };
            let (_, ti, ei) = self.search_nav[i];
            if self.nav_match_visible(ti, ei) {
                self.search_cursor = i;
                self.pending_tab = Some(BottomTab::Detail);
                self.zoom_to_nav_cursor();
                return;
            }
        }
    }

    fn compute_aggregates(&mut self) {
        self.sel_generation = self.sel_generation.wrapping_add(1);
        let mut all_durs: Vec<f64> = self.selection_stats.iter().flat_map(|s| s.durations.iter().copied()).collect();
        self.sel_median = median_inplace(&mut all_durs);

        self.sel_agg_stats = self.selection_stats.iter().map(|s| {
            let mut sorted = s.durations.clone();
            let median = median_inplace(&mut sorted);
            KernelStats {
                name: s.name, count: s.count, total_dur: s.total_dur,
                median_dur: median,
                max_dur: s.durations.iter().copied().fold(0.0f64, f64::max),
                min_dur: s.durations.iter().copied().fold(f64::MAX, f64::min),
            }
        }).collect();

        // One row per event, in whatever order selection_stats yields them —
        // NOT pre-sorted by duration. draw_stats_table maintains its own
        // sort_idx keyed on whichever column the user actually sorted by
        // (see ui.rs) and never assumes its input arrives sorted, so an
        // eager sort here was pure wasted work: for a 380k-event selection
        // it cost as much as the rest of this function combined, just to be
        // immediately thrown away and re-derived by sort_idx anyway.
        self.sel_individual.clear();
        self.sel_individual_refs.clear();
        for se in &self.selection_stats {
            for (i, &d) in se.durations.iter().enumerate() {
                let r = se.event_refs.get(i).copied().unwrap_or((u32::MAX, u32::MAX));
                self.sel_individual.push(KernelStats { name: se.name, count: 1, total_dur: d, median_dur: d, max_dur: d, min_dur: d });
                self.sel_individual_refs.push(r);
            }
        }
    }

}

pub struct AppState {
    pub panes: Vec<Pane>,
    pub active: usize,
    /// One-shot "force this tab selected next frame" signal, set right after
    /// opening a new trace so the tab strip visually catches up with
    /// `active`. Mirrors `Pane.pending_tab`.
    pub pending_active_tab: Option<usize>,
    pub buf: DrawBuf,
    pub bottom_h: f32,
    pub drag: DragKind,
    pub show_diff: bool,
    pub diff_popup_open: bool,
    pub diff_result: Option<DiffResult>,
    pub diff_bar_scroll: f64,
    pub diff_bar_zoom: f64,
    pub diff_pane_indices: Option<[usize; 2]>,
    /// Target of the "right-click a kernel -> sync ranks to it" context menu,
    /// set at right-click time and read when the menu's item is clicked.
    /// `(pane_idx, event)` — pane index guards against the popup surviving a
    /// tab switch and firing against the wrong pane's trace.
    pub kernel_ctx_menu: Option<(usize, EventRef)>,
}

impl AppState {
    pub fn add_pane(&mut self) -> usize {
        self.panes.push(Pane::new());
        self.panes.len() - 1
    }

    pub fn remove_pane(&mut self, pi: usize) {
        self.panes.remove(pi);
        if self.active > pi {
            self.active -= 1;
        } else if self.active >= self.panes.len() {
            self.active = self.panes.len().saturating_sub(1);
        }
        self.drag = DragKind::None;
    }
}
