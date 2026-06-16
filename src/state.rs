use crate::loader::{load_trace_progressive, load_multi_progressive};
use crate::parse::json_unescape;
use crate::types::*;
use imgui::ImColor32;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

pub(crate) fn parse_rank(label: &str) -> Option<usize> {
    if label.starts_with("[rank ") {
        label[6..].find(']').and_then(|p| label[6..6 + p].parse().ok())
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
    pub labels: Vec<Label>,
    pub event_labels: Vec<Vec<Option<u8>>>,
    pub label_input: String,
    pub label_stats: Vec<LabelStats>,
    pub hidden_names: Vec<bool>,
    pub pending_tab: Option<BottomTab>,
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
            labels: Vec::new(),
            event_labels: Vec::new(),
            label_input: String::with_capacity(64),
            label_stats: Vec::new(),
            hidden_names: Vec::new(),
            pending_tab: Some(BottomTab::Stats),
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
            load_multi_progressive(rank_paths, &counter, tpf, &tx, cd.as_deref());
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
                load_multi_progressive(paths, &counter, tpf, &tx, cd.as_deref());
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
                    self.event_labels = trace.tracks.iter()
                        .map(|t| vec![None; t.events.len()])
                        .collect();
                    self.hidden_names.resize(n_names, false);
                    self.search_mask.clear();
                    self.search_nav.clear();
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
                    self.event_labels = trace.tracks.iter()
                        .map(|t| vec![None; t.events.len()])
                        .collect();
                    self.labels.clear();
                    self.label_stats.clear();
                    self.hidden_names = vec![false; trace.names.len()];
                    self.trace = Some(trace);
                    self.load_labels();
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

        let mut cum_y = 0.0f32;
        let mut total_scanned = 0usize;
        for (ti, track) in trace.tracks.iter().enumerate() {
            if !self.show_cpu && !track.gpu { continue; }
            let track_h = track_height(
                track.max_depth,
                self.collapsed.get(ti).copied().unwrap_or(false),
                self.track_scales.get(ti).copied().unwrap_or(1.0),
            );
            let sub_h = track_h / track.max_depth.max(1) as f32;
            let track_top = cum_y;
            cum_y += track_h;
            let track_bot = cum_y;
            if track_bot < y0 || track_top > y1 { continue; }
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
        let mut cum_y = 0.0f32;
        for (ti, track) in trace.tracks.iter().enumerate() {
            if !self.show_cpu && !track.gpu { continue; }
            let track_h = track_height(
                track.max_depth,
                self.collapsed.get(ti).copied().unwrap_or(false),
                self.track_scales.get(ti).copied().unwrap_or(1.0),
            );
            let sub_h = track_h / track.max_depth.max(1) as f32;
            let track_top = cum_y;
            cum_y += track_h;
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
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        events.into_iter().map(|(_, name, dur)| (name, dur)).collect()
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

    pub fn apply_label(&mut self, name: &str) {
        if name.is_empty() { return; }
        let trace = match &self.trace {
            Some(t) => t,
            None => return,
        };
        let sel = match &self.selection {
            Some(s) => *s,
            None => return,
        };
        let (s0, s1) = if sel[0] <= sel[1] { (sel[0], sel[1]) } else { (sel[1], sel[0]) };

        let hidden = &self.hidden_names;
        let is_visible = |e: &Event| !hidden.get(e.name as usize).copied().unwrap_or(false);

        let mut pattern: Vec<u32> = Vec::new();
        for track in &trace.tracks {
            if !self.show_cpu && !track.gpu { continue; }
            let p: Vec<u32> = track.events.iter()
                .filter(|e| e.depth == 0 && is_visible(e) && e.ts + e.dur >= s0 && e.ts <= s1)
                .map(|e| e.name)
                .collect();
            if !p.is_empty() {
                pattern = p;
                break;
            }
        }
        if pattern.is_empty() {
            for track in &trace.tracks {
                if !self.show_cpu && !track.gpu { continue; }
                let p: Vec<u32> = track.events.iter()
                    .filter(|e| is_visible(e) && e.ts + e.dur >= s0 && e.ts <= s1)
                    .map(|e| e.name)
                    .collect();
                if !p.is_empty() {
                    pattern = p;
                    break;
                }
            }
        }
        if pattern.is_empty() { return; }

        let li = if let Some(i) = self.labels.iter().position(|l| l.name == name) {
            self.labels[i].pattern = pattern.clone();
            i
        } else {
            let ci = self.labels.len() % LABEL_PALETTE.len();
            let c = LABEL_PALETTE[ci];
            self.labels.push(Label {
                name: name.to_string(),
                color: ImColor32::from_rgba((c >> 16) as u8, ((c >> 8) & 0xFF) as u8, (c & 0xFF) as u8, 255),
                pattern: pattern.clone(),
            });
            self.labels.len() - 1
        };

        self.compact_labels();
        self.rebuild_event_labels();
        self.rebuild_label_stats();
        self.save_labels();
        eprintln!("  label \"{name}\": pattern len {}, label idx {li}", pattern.len());
    }

    pub fn delete_label(&mut self, idx: usize) {
        if idx >= self.labels.len() { return; }
        self.labels.remove(idx);
        self.rebuild_event_labels();
        self.rebuild_label_stats();
        self.save_labels();
    }

    fn compact_labels(&mut self) {
        self.labels.retain(|l| !l.pattern.is_empty());
    }

    pub fn rebuild_event_labels(&mut self) {
        for track_labels in &mut self.event_labels {
            for v in track_labels.iter_mut() { *v = None; }
        }
        let trace = match &self.trace {
            Some(t) => t,
            None => return,
        };
        for (li, label) in self.labels.iter().enumerate() {
            if label.pattern.is_empty() { continue; }
            for (ti, track) in trace.tracks.iter().enumerate() {
                let seq: Vec<usize> = track.events.iter().enumerate()
                    .filter(|(_, e)| !self.hidden_names.get(e.name as usize).copied().unwrap_or(false))
                    .map(|(i, _)| i)
                    .collect();
                let seq_names: Vec<u32> = seq.iter().map(|&i| track.events[i].name).collect();
                if label.pattern.len() > seq_names.len() { continue; }
                for i in 0..=seq_names.len() - label.pattern.len() {
                    if seq_names[i..i + label.pattern.len()] == label.pattern[..] {
                        for j in 0..label.pattern.len() {
                            let ei = seq[i + j];
                            if ti < self.event_labels.len() && ei < self.event_labels[ti].len() {
                                self.event_labels[ti][ei] = Some(li as u8);
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn rebuild_label_stats(&mut self) {
        self.label_stats.clear();
        let trace = match &self.trace {
            Some(t) => t,
            None => return,
        };
        let mut by_label: Vec<(f64, u32)> = vec![(0.0, 0); self.labels.len()];
        for (ti, track) in trace.tracks.iter().enumerate() {
            for (ei, ev) in track.events.iter().enumerate() {
                if let Some(li) = self.event_labels.get(ti).and_then(|t| t.get(ei)).copied().flatten() {
                    by_label[li as usize].0 += ev.dur;
                    by_label[li as usize].1 += 1;
                }
            }
        }
        for (li, &(total_dur, count)) in by_label.iter().enumerate() {
            self.label_stats.push(LabelStats { label_idx: li as u8, total_dur, count });
        }
        self.label_stats.sort_by(|a, b| b.total_dur.partial_cmp(&a.total_dur).unwrap());
    }

    pub fn save_labels(&self) {
        if self.trace_path.is_empty() { return; }
        let trace = match &self.trace {
            Some(t) => t,
            None => return,
        };
        let path = format!("{}.labels.json", self.trace_path.trim_end_matches(".gz"));
        let mut out = String::from("{\"labels\":[\n");
        for (i, label) in self.labels.iter().enumerate() {
            if i > 0 { out.push_str(",\n"); }
            out.push_str(&format!("  {{\"name\":{},\"pattern\":[",
                json_escape(&label.name)));
            for (j, &kn) in label.pattern.iter().enumerate() {
                if j > 0 { out.push(','); }
                out.push_str(&json_escape(&trace.names[kn as usize]));
            }
            out.push_str("]}");
        }
        out.push_str("\n]}\n");
        let _ = std::fs::write(&path, &out);
    }

    pub fn load_labels(&mut self) {
        if self.trace_path.is_empty() { return; }
        let trace = match &self.trace {
            Some(t) => t,
            None => return,
        };
        let path = format!("{}.labels.json", self.trace_path.trim_end_matches(".gz"));
        let data = match std::fs::read_to_string(&path) {
            Ok(d) => d,
            Err(_) => return,
        };

        let mut name_to_idx: HashMap<&str, u32> = HashMap::new();
        for (i, name) in trace.names.iter().enumerate() {
            name_to_idx.insert(name, i as u32);
        }

        self.labels.clear();

        let mut pos = 0;
        let bytes = data.as_bytes();
        while pos < bytes.len() {
            if let Some(p) = data[pos..].find("\"name\":\"") {
                let name_start = pos + p + 8;
                let name_end = match data[name_start..].find('"') {
                    Some(e) => name_start + e,
                    None => break,
                };
                let label_name = json_unescape(&data[name_start..name_end]);

                let rest_start = name_end + 1;
                let arr_start = match data[rest_start..].find('[') {
                    Some(a) => rest_start + a + 1,
                    None => break,
                };
                let arr_end = match data[arr_start..].find(']') {
                    Some(a) => arr_start + a,
                    None => break,
                };
                let arr = &data[arr_start..arr_end];

                let mut pattern = Vec::new();
                for part in arr.split(',') {
                    let s = part.trim().trim_matches('"');
                    if !s.is_empty() {
                        if let Some(&idx) = name_to_idx.get(s) {
                            pattern.push(idx);
                        }
                    }
                }

                if !pattern.is_empty() {
                    let ci = self.labels.len() % LABEL_PALETTE.len();
                    let c = LABEL_PALETTE[ci];
                    self.labels.push(Label {
                        name: label_name,
                        color: ImColor32::from_rgba((c >> 16) as u8, ((c >> 8) & 0xFF) as u8, (c & 0xFF) as u8, 255),
                        pattern,
                    });
                }

                pos = arr_end + 1;
            } else {
                break;
            }
        }
        self.rebuild_event_labels();
        self.rebuild_label_stats();
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

pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}
