use crate::loader::{load_trace_progressive, load_multi_progressive};
use crate::types::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

pub(crate) fn parse_rank(label: &str) -> Option<usize> {
    let l = label.trim_start();
    if l.starts_with("Rank ") {
        l[5..].split_once(' ').and_then(|(n, _)| n.parse().ok())
    } else {
        None
    }
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
    pub selection_stats: Vec<SelectionEntry>,
    pub selection_dirty: bool,
    pub sel_mask: Vec<bool>,
    pub collapsed: Vec<bool>,
    pub track_scales: Vec<f32>,
    pub even_spacing: bool,
    pub geom: PaneGeom,
    pub hidden_names: Vec<bool>,
    pub pending_tab: Option<BottomTab>,
    /// Track index whose row draw_timeline should scroll into view as part of an
    /// in-flight search zoom. Consumed (cleared) on the next timeline draw.
    pub pending_focus: Option<u32>,
    pub sort_col: usize,
    pub sort_asc: bool,
    pub sel_aggregate: bool,
    pub label_w: f32,
    pub sel_median: f64,
    pub sel_agg_stats: Vec<KernelStats>,
    pub sel_individual: Vec<KernelStats>,
    pub track_order: Vec<usize>,
    pub auto_reload: bool,
    pub reload_paths: Vec<(usize, String)>,
    pub reload_dir: Option<String>,
    pub cache_dir: Option<String>,
    pub loading_events: Arc<AtomicUsize>,
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
            merge_gpu: false,
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
            selection_stats: Vec::new(),
            selection_dirty: false,
            sel_mask: Vec::new(),
            collapsed: Vec::new(),
            track_scales: Vec::new(),
            even_spacing: false,
            geom: PaneGeom::default(),
            hidden_names: Vec::new(),
            pending_tab: Some(BottomTab::Detail),
            pending_focus: None,
            sort_col: 2,
            sort_asc: false,
            sel_aggregate: true,
            label_w: LABEL_W,
            sel_median: 0.0,
            sel_agg_stats: Vec::new(),
            sel_individual: Vec::new(),
            track_order: Vec::new(),
            auto_reload: false,
            reload_paths: Vec::new(),
            reload_dir: None,
            cache_dir: None,
            loading_events: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn has_trace(&self) -> bool { self.trace.is_some() }

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
        self.selection_stats.clear();
        self.sel_mask.clear();
        self.sel_median = 0.0;
        self.sel_agg_stats.clear();
        self.sel_individual.clear();
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
        self.selection = None;
        self.rebuild_selection_stats(buf);
        self.sel_mask.clear();
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
        std::thread::spawn(move || {
            load_trace_progressive(&path, &counter, 0, &tx, cd.as_deref());
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
        std::thread::spawn(move || {
            load_multi_progressive(rank_paths, &counter, tpf, &tx, cd.as_deref(), false);
        });
    }

    pub fn reload(&mut self) {
        if self.loading.is_some() { return; }
        if let Some(dir) = &self.reload_dir {
            let (groups, standalone) = crate::loader::detect_rank_groups(&[dir.clone()]);
            let mut all_paths: Vec<(usize, String)> = Vec::new();
            for group in groups {
                all_paths.extend(group);
            }
            for (i, path) in standalone.into_iter().enumerate() {
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
            std::thread::spawn(move || {
                load_trace_progressive(&path, &counter, 0, &tx, cd.as_deref());
            });
        } else {
            let tpf = (std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4) / paths.len()).max(1);
            std::thread::spawn(move || {
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
                        self.track_order = (0..n_tracks).collect();
                    }
                    self.hidden_names.resize(n_names, false);
                    self.search_mask.clear();
                    self.search_nav.clear();
                    self.search_cursor = 0;
                    // The reloaded trace may have fewer events; a retained
                    // EventRef would index out of bounds in the Detail panel.
                    self.selected = None;
                    self.selection_stats.clear();
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
                    self.track_order = (0..trace.tracks.len()).collect();
                    self.hidden_names = vec![false; trace.names.len()];
                    self.trace = Some(trace);
                }
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
        let t_start = std::time::Instant::now();
        let trace = match &self.trace {
            Some(t) => t,
            None => return,
        };
        let sel = match self.selection.or(self.finished_sel) {
            Some(s) => s,
            None => { self.selection_stats.clear(); self.sel_agg_stats.clear(); self.sel_individual.clear(); self.sel_median = 0.0; return; }
        };
        let (s0, s1) = if sel[0] <= sel[1] { (sel[0], sel[1]) } else { (sel[1], sel[0]) };
        let (y0, y1) = if sel[2] <= sel[3] { (sel[2] as f32, sel[3] as f32) } else { (sel[3] as f32, sel[2] as f32) };

        let map = &mut buf.sel_map;
        for v in map.values_mut() { v.0 = 0; v.1 = 0.0; v.2.clear(); }

        let mut total_scanned = 0usize;
        for vi in 0..self.geom.visible.len() {
            let track_top = self.geom.y_offsets[vi];
            let track_h = self.geom.heights[vi];
            let track_bot = track_top + track_h;
            if track_bot < y0 || track_top > y1 { continue; }

            if let Some(group) = self.geom.merged.iter().find(|g| g.vi == vi) {
                // Iterate the packed events that were actually drawn in the merged
                // row (grandparent wrappers already stripped), and apply the same
                // depth/y test the renderer uses for its selection highlight, so
                // the stats match the visible rectangle exactly.
                let sub_h = track_h / group.max_depth.max(1) as f32;
                total_scanned += group.events.len();
                for &(ti32, ei32, depth) in &group.events {
                    let ev = &trace.tracks[ti32 as usize].events[ei32 as usize];
                    if !(ev.ts + ev.dur >= s0 && ev.ts <= s1) { continue; }
                    let ev_top = track_top + depth as f32 * sub_h;
                    let ev_bot = ev_top + sub_h;
                    if ev_bot < y0 || ev_top > y1 { continue; }
                    let e = map.entry(ev.name).or_insert((0, 0.0, Vec::new()));
                    e.0 += 1;
                    e.1 += ev.dur;
                    e.2.push(ev.dur);
                }
            } else {
                let ti = self.geom.visible[vi];
                let track = &trace.tracks[ti];
                if !self.show_cpu && !track.gpu { continue; }
                let sub_h = track_h / track.max_depth.max(1) as f32;
                let start = bisect_overlap(&track.events, &track.prefix_max_dur, s0);
                let end = track.events.partition_point(|e| e.ts <= s1).max(start);
                total_scanned += end - start;
                let mut ancestor_sel = vec![false; track.max_depth as usize + 1];
                for ev in &track.events[start..end] {
                    if self.hidden_names.get(ev.name as usize).copied().unwrap_or(false) { continue; }
                    let ev_top = track_top + ev.depth as f32 * sub_h;
                    let ev_bot = ev_top + sub_h;
                    for d in ev.depth as usize..ancestor_sel.len() { ancestor_sel[d] = false; }
                    if ev_bot < y0 || ev_top > y1 { continue; }
                    if ev.ts + ev.dur >= s0 && ev.ts <= s1 {
                        ancestor_sel[ev.depth as usize] = true;
                        if (0..ev.depth as usize).any(|d| ancestor_sel[d]) { continue; }
                        let e = map.entry(ev.name).or_insert((0, 0.0, Vec::new()));
                        e.0 += 1;
                        e.1 += ev.dur;
                        e.2.push(ev.dur);
                    }
                }
            }
        }
        self.selection_stats.clear();
        for (&name, (count, total_dur, durations)) in map.iter_mut() {
            if *count == 0 { continue; }
            self.selection_stats.push(SelectionEntry {
                name, count: *count, total_dur: *total_dur,
                durations: std::mem::take(durations),
            });
        }
        self.selection_stats.sort_unstable_by(|a, b| b.total_dur.partial_cmp(&a.total_dur).unwrap().then(a.name.cmp(&b.name)));
        let ev_count: u32 = self.selection_stats.iter().map(|e| e.count).sum();
        eprintln!("  select: {:.1}ms ({} events, {} names, {} scanned)", t_start.elapsed().as_secs_f64() * 1000.0, ev_count, self.selection_stats.len(), total_scanned);

        let t_agg = std::time::Instant::now();
        self.compute_aggregates();
        eprintln!("  aggregate: {:.1}ms ({} agg, {} individual)", t_agg.elapsed().as_secs_f64() * 1000.0, self.sel_agg_stats.len(), self.sel_individual.len());
    }

    pub fn extract_selection_events(&self) -> Vec<(String, f64)> {
        let trace = match &self.trace {
            Some(t) => t,
            None => return Vec::new(),
        };
        let sel = match self.finished_sel {
            Some(s) => s,
            None => return Vec::new(),
        };
        let (s0, s1) = if sel[0] <= sel[1] { (sel[0], sel[1]) } else { (sel[1], sel[0]) };
        let (y0, y1) = if sel[2] <= sel[3] { (sel[2] as f32, sel[3] as f32) } else { (sel[3] as f32, sel[2] as f32) };

        let mut events: Vec<(f64, String, f64)> = Vec::new();
        for vi in 0..self.geom.visible.len() {
            let track_top = self.geom.y_offsets[vi];
            let track_h = self.geom.heights[vi];
            let track_bot = track_top + track_h;
            if track_bot < y0 || track_top > y1 { continue; }

            if let Some(group) = self.geom.merged.iter().find(|g| g.vi == vi) {
                // Match the rendered merged row: iterate the packed events (wrappers
                // stripped) and apply the renderer's depth/y test.
                let sub_h = track_h / group.max_depth.max(1) as f32;
                for &(ti32, ei32, depth) in &group.events {
                    let ev = &trace.tracks[ti32 as usize].events[ei32 as usize];
                    if !(ev.ts + ev.dur >= s0 && ev.ts <= s1) { continue; }
                    let ev_top = track_top + depth as f32 * sub_h;
                    let ev_bot = ev_top + sub_h;
                    if ev_bot < y0 || ev_top > y1 { continue; }
                    events.push((ev.ts, trace.names[ev.name as usize].clone(), ev.dur));
                }
            } else {
                let ti = self.geom.visible[vi];
                let track = &trace.tracks[ti];
                if !self.show_cpu && !track.gpu { continue; }
                let sub_h = track_h / track.max_depth.max(1) as f32;
                let start = bisect_overlap(&track.events, &track.prefix_max_dur, s0);
                let end = track.events.partition_point(|e| e.ts <= s1).max(start);
                let mut ancestor_sel = vec![false; track.max_depth as usize + 1];
                for ev in &track.events[start..end] {
                    if self.hidden_names.get(ev.name as usize).copied().unwrap_or(false) { continue; }
                    let ev_top = track_top + ev.depth as f32 * sub_h;
                    let ev_bot = ev_top + sub_h;
                    for d in ev.depth as usize..ancestor_sel.len() { ancestor_sel[d] = false; }
                    if ev_bot < y0 || ev_top > y1 { continue; }
                    if ev.ts + ev.dur >= s0 && ev.ts <= s1 {
                        ancestor_sel[ev.depth as usize] = true;
                        if (0..ev.depth as usize).any(|d| ancestor_sel[d]) { continue; }
                        events.push((ev.ts, trace.names[ev.name as usize].clone(), ev.dur));
                    }
                }
            }
        }
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        events.into_iter().map(|(_, name, dur)| (name, dur)).collect()
    }

    pub fn copy_selection_text(&self) -> Option<String> {
        let trace = self.trace.as_ref()?;
        let mut events: Vec<(f64, &str, f64)> = Vec::new();

        if self.finished_sel.is_some() {
            let sel = self.finished_sel.unwrap();
            let (s0, s1) = if sel[0] <= sel[1] { (sel[0], sel[1]) } else { (sel[1], sel[0]) };
            let (y0, y1) = if sel[2] <= sel[3] { (sel[2] as f32, sel[3] as f32) } else { (sel[3] as f32, sel[2] as f32) };
            for vi in 0..self.geom.visible.len() {
                let track_top = self.geom.y_offsets[vi];
                let track_h = self.geom.heights[vi];
                let track_bot = track_top + track_h;
                if track_bot < y0 || track_top > y1 { continue; }

                if let Some(group) = self.geom.merged.iter().find(|g| g.vi == vi) {
                    // Match the rendered merged row: iterate the packed events
                    // (wrappers stripped) and apply the renderer's depth/y test.
                    let sub_h = track_h / group.max_depth.max(1) as f32;
                    for &(ti32, ei32, depth) in &group.events {
                        let ev = &trace.tracks[ti32 as usize].events[ei32 as usize];
                        if !(ev.ts + ev.dur >= s0 && ev.ts <= s1) { continue; }
                        let ev_top = track_top + depth as f32 * sub_h;
                        let ev_bot = ev_top + sub_h;
                        if ev_bot < y0 || ev_top > y1 { continue; }
                        events.push((ev.ts, &trace.names[ev.name as usize], ev.dur));
                    }
                } else {
                    let ti = self.geom.visible[vi];
                    let track = &trace.tracks[ti];
                    if !self.show_cpu && !track.gpu { continue; }
                    let sub_h = track_h / track.max_depth.max(1) as f32;
                    let start = bisect_overlap(&track.events, &track.prefix_max_dur, s0);
                    let end = track.events.partition_point(|e| e.ts <= s1).max(start);
                    let mut ancestor_sel = vec![false; track.max_depth as usize + 1];
                    for ev in &track.events[start..end] {
                        if self.hidden_names.get(ev.name as usize).copied().unwrap_or(false) { continue; }
                        let ev_top = track_top + ev.depth as f32 * sub_h;
                        let ev_bot = ev_top + sub_h;
                        for d in ev.depth as usize..ancestor_sel.len() { ancestor_sel[d] = false; }
                        if ev_bot < y0 || ev_top > y1 { continue; }
                        if ev.ts + ev.dur >= s0 && ev.ts <= s1 {
                            ancestor_sel[ev.depth as usize] = true;
                            if (0..ev.depth as usize).any(|d| ancestor_sel[d]) { continue; }
                            events.push((ev.ts, &trace.names[ev.name as usize], ev.dur));
                        }
                    }
                }
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
        for v in map.values_mut() { v.0 = 0; v.1 = 0.0; v.2.clear(); }

        for track in &trace.tracks {
            if !self.show_cpu && !track.gpu { continue; }
            for ev in &track.events {
                if self.hidden_names.get(ev.name as usize).copied().unwrap_or(false) { continue; }
                if self.search_mask[ev.name as usize] {
                    let e = map.entry(ev.name).or_insert((0, 0.0, Vec::new()));
                    e.0 += 1;
                    e.1 += ev.dur;
                    e.2.push(ev.dur);
                }
            }
        }
        self.selection_stats.clear();
        for (&name, (count, total_dur, durations)) in map.iter_mut() {
            if *count == 0 { continue; }
            self.selection_stats.push(SelectionEntry {
                name, count: *count, total_dur: *total_dur,
                durations: std::mem::take(durations),
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
        for track in &trace.tracks {
            if !self.show_cpu && !track.gpu { continue; }
            for ev in &track.events {
                if ev.name != name_id { continue; }
                count += 1;
                total_dur += ev.dur;
                durations.push(ev.dur);
            }
        }
        self.selection_stats.clear();
        if count > 0 {
            self.selection_stats.push(SelectionEntry { name: name_id, count, total_dur, durations });
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
        let mut all_durs: Vec<f64> = self.selection_stats.iter().flat_map(|s| s.durations.iter().copied()).collect();
        all_durs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let n = all_durs.len();
        self.sel_median = if n == 0 { 0.0 } else if n % 2 == 1 { all_durs[n / 2] } else { (all_durs[n / 2 - 1] + all_durs[n / 2]) / 2.0 };

        self.sel_agg_stats = self.selection_stats.iter().map(|s| {
            let mut sorted = s.durations.clone();
            sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let n = sorted.len();
            let median = if n == 0 { 0.0 } else if n % 2 == 1 { sorted[n / 2] } else { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 };
            KernelStats {
                name: s.name, count: s.count, total_dur: s.total_dur,
                median_dur: median,
                max_dur: s.durations.iter().copied().fold(0.0f64, f64::max),
            }
        }).collect();

        self.sel_individual.clear();
        for se in &self.selection_stats {
            for &d in &se.durations {
                self.sel_individual.push(KernelStats { name: se.name, count: 1, total_dur: d, median_dur: d, max_dur: d });
            }
        }
        self.sel_individual.sort_unstable_by(|a, b| b.total_dur.partial_cmp(&a.total_dur).unwrap());
    }

}

pub struct AppState {
    pub panes: Vec<Pane>,
    pub active: usize,
    pub divider_xs: Vec<f32>,
    pub buf: DrawBuf,
    pub bottom_h: f32,
    pub drag: DragKind,
    pub show_diff: bool,
    pub diff_popup_open: bool,
    pub diff_result: Option<DiffResult>,
    pub diff_bar_scroll: f64,
    pub diff_bar_zoom: f64,
    pub diff_pane_indices: Option<[usize; 2]>,
}

impl AppState {
    pub fn recompute_dividers(&mut self, width: f32) {
        let n = self.panes.len();
        self.divider_xs.clear();
        for i in 1..n {
            self.divider_xs.push(width * i as f32 / n as f32);
        }
    }

    pub fn pane_x(&self, pi: usize, _width: f32) -> f32 {
        if pi == 0 { 0.0 } else { self.divider_xs[pi - 1] }
    }

    pub fn pane_w(&self, pi: usize, width: f32) -> f32 {
        let x0 = if pi == 0 { 0.0 } else { self.divider_xs[pi - 1] };
        let x1 = if pi < self.divider_xs.len() { self.divider_xs[pi] } else { width };
        x1 - x0
    }

    pub fn add_pane(&mut self, width: f32) {
        self.panes.push(Pane::new());
        self.recompute_dividers(width);
    }

    pub fn remove_pane(&mut self, pi: usize, width: f32) {
        self.panes.remove(pi);
        if self.active > pi {
            self.active -= 1;
        } else if self.active >= self.panes.len() {
            self.active = self.panes.len().saturating_sub(1);
        }
        self.drag = DragKind::None;
        self.recompute_dividers(width);
    }

    pub fn pane_at_x(&self, x: f32) -> usize {
        for (i, &dx) in self.divider_xs.iter().enumerate() {
            if x < dx { return i; }
        }
        self.panes.len() - 1
    }
}
