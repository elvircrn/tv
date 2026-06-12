use crate::loader::{load_trace, merge_traces};
use crate::parse::json_unescape;
use crate::types::*;
use imgui::ImColor32;
use std::collections::HashMap;
use std::sync::mpsc;

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
    pub align_ranks: bool,
    pub straggler_mask: Vec<Vec<bool>>,
    pub time_aligned: bool,
    pub rank_time_offsets: Vec<f64>,
    pub step_aligned: bool,
    pub step_align_offsets: Vec<Vec<f64>>,
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
            align_ranks: false,
            straggler_mask: Vec::new(),
            time_aligned: false,
            rank_time_offsets: Vec::new(),
            step_aligned: false,
            step_align_offsets: Vec::new(),
        }
    }

    pub fn has_trace(&self) -> bool { self.trace.is_some() }

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
        std::thread::spawn(move || {
            tx.send(load_trace(&path)).ok();
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
        self.loading = Some(rx);
        self.error = None;
        std::thread::spawn(move || {
            let results: Vec<_> = std::thread::scope(|s| {
                let handles: Vec<_> = rank_paths.iter().map(|(rank, path)| {
                    let r = *rank;
                    s.spawn(move || (r, load_trace(path)))
                }).collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mut traces = Vec::new();
            for (rank, result) in results {
                match result {
                    Ok(t) => traces.push((rank, t)),
                    Err(e) => {
                        tx.send(Err(format!("rank {rank}: {e}"))).ok();
                        return;
                    }
                }
            }
            tx.send(Ok(merge_traces(traces))).ok();
        });
    }

    pub fn poll_loading(&mut self) {
        let rx = match &self.loading {
            Some(rx) => rx,
            None => return,
        };
        match rx.try_recv() {
            Ok(Ok(trace)) => {
                let pad = trace.max_ts * 0.02;
                self.view.t0 = -pad;
                self.view.t1 = trace.max_ts + pad;
                self.view.scroll_y = 0.0;
                self.collapsed = vec![false; trace.tracks.len()];
                self.track_scales = vec![1.0; trace.tracks.len()];
                self.event_labels = trace.tracks.iter()
                    .map(|t| vec![None; t.events.len()])
                    .collect();
                self.labels.clear();
                self.label_stats.clear();
                self.hidden_names = vec![false; trace.names.len()];
                self.trace = Some(trace);
                self.loading = None;
                self.load_labels();
            }
            Ok(Err(e)) => {
                self.error = Some(e);
                self.loading = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(_) => {
                self.error = Some("Load thread crashed".into());
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

    pub fn align_rank_times(&mut self) {
        let trace = match &mut self.trace {
            Some(t) => t,
            None => return,
        };
        let mut rank_min: HashMap<usize, f64> = HashMap::new();
        let mut track_rank: Vec<Option<usize>> = Vec::new();
        for track in trace.tracks.iter() {
            let rank = parse_rank(&track.label);
            if let Some(r) = rank {
                if track.gpu {
                    let min = track.events.iter().map(|e| e.ts).fold(f64::MAX, f64::min);
                    let entry = rank_min.entry(r).or_insert(f64::MAX);
                    *entry = entry.min(min);
                }
            }
            track_rank.push(rank);
        }
        self.rank_time_offsets = vec![0.0; trace.tracks.len()];
        for (ti, rank) in track_rank.iter().enumerate() {
            if let Some(r) = rank {
                let offset = rank_min[r];
                self.rank_time_offsets[ti] = offset;
                for ev in &mut trace.tracks[ti].events {
                    ev.ts -= offset;
                }
            }
        }
        trace.max_ts = trace.tracks.iter()
            .flat_map(|t| t.events.iter().map(|e| e.ts + e.dur))
            .fold(0.0f64, f64::max);
    }

    pub fn unalign_rank_times(&mut self) {
        let trace = match &mut self.trace {
            Some(t) => t,
            None => return,
        };
        for (ti, &offset) in self.rank_time_offsets.iter().enumerate() {
            if offset != 0.0 {
                for ev in &mut trace.tracks[ti].events {
                    ev.ts += offset;
                }
            }
        }
        self.rank_time_offsets.clear();
        trace.max_ts = trace.tracks.iter()
            .flat_map(|t| t.events.iter().map(|e| e.ts + e.dur))
            .fold(0.0f64, f64::max);
    }

    pub fn align_per_step(&mut self) {
        let trace = match &mut self.trace {
            Some(t) => t,
            None => return,
        };

        let mut rank_gpu_tracks: HashMap<usize, Vec<usize>> = HashMap::new();
        for (ti, track) in trace.tracks.iter().enumerate() {
            if !track.gpu { continue; }
            if let Some(r) = parse_rank(&track.label) {
                rank_gpu_tracks.entry(r).or_default().push(ti);
            }
        }
        if rank_gpu_tracks.len() < 2 { return; }

        let mut ranks_sorted: Vec<usize> = rank_gpu_tracks.keys().copied().collect();
        ranks_sorted.sort();

        let mut exec_times: Vec<Vec<f64>> = Vec::new();
        for &rank in &ranks_sorted {
            let tis = &rank_gpu_tracks[&rank];
            let mut execs: Vec<f64> = Vec::new();
            for &ti in tis {
                for ev in &trace.tracks[ti].events {
                    if trace.names[ev.name as usize].starts_with("execute") {
                        execs.push(ev.ts);
                    }
                }
            }
            execs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            exec_times.push(execs);
        }

        let min_count = exec_times.iter().map(|e| e.len()).min().unwrap_or(0);
        if min_count == 0 { return; }

        let ref_times: Vec<f64> = (0..min_count).map(|k| exec_times[0][k]).collect();

        let mut rank_segments: Vec<Vec<(f64, f64)>> = Vec::new();
        for (ri, _rank) in ranks_sorted.iter().enumerate() {
            let mut segs = Vec::new();
            for k in 0..min_count {
                let offset = ref_times[k] - exec_times[ri][k];
                segs.push((exec_times[ri][k], offset));
            }
            rank_segments.push(segs);
        }

        self.step_align_offsets = vec![Vec::new(); trace.tracks.len()];
        for (ti, track) in trace.tracks.iter_mut().enumerate() {
            let rank = match parse_rank(&track.label) {
                Some(r) => r,
                None => continue,
            };
            let ri = match ranks_sorted.iter().position(|&r| r == rank) {
                Some(i) => i,
                None => continue,
            };
            let segs = &rank_segments[ri];
            let mut ev_offsets = Vec::with_capacity(track.events.len());
            for ev in &mut track.events {
                let offset = match segs.binary_search_by(|s| s.0.partial_cmp(&ev.ts).unwrap()) {
                    Ok(i) => segs[i].1,
                    Err(0) => segs[0].1,
                    Err(i) if i >= segs.len() => segs[segs.len() - 1].1,
                    Err(i) => {
                        let (t0, o0) = segs[i - 1];
                        let (t1, o1) = segs[i];
                        let frac = (ev.ts - t0) / (t1 - t0);
                        o0 + frac * (o1 - o0)
                    }
                };
                ev_offsets.push(offset);
                ev.ts += offset;
            }
            self.step_align_offsets[ti] = ev_offsets;
        }

        trace.max_ts = trace.tracks.iter()
            .flat_map(|t| t.events.iter().map(|e| e.ts + e.dur))
            .fold(0.0f64, f64::max);
        eprintln!("  step-align: {} steps matched across {} ranks", min_count, ranks_sorted.len());
    }

    pub fn unalign_per_step(&mut self) {
        let trace = match &mut self.trace {
            Some(t) => t,
            None => return,
        };
        for (ti, offsets) in self.step_align_offsets.iter().enumerate() {
            if offsets.len() != trace.tracks[ti].events.len() { continue; }
            for (ei, ev) in trace.tracks[ti].events.iter_mut().enumerate() {
                ev.ts -= offsets[ei];
            }
        }
        self.step_align_offsets.clear();
        trace.max_ts = trace.tracks.iter()
            .flat_map(|t| t.events.iter().map(|e| e.ts + e.dur))
            .fold(0.0f64, f64::max);
    }

    pub fn detect_stragglers(&mut self) {
        let t0 = std::time::Instant::now();
        let trace = match &self.trace {
            Some(t) => t,
            None => return,
        };
        self.straggler_mask = trace.tracks.iter()
            .map(|t| vec![false; t.events.len()])
            .collect();

        let mut rank_tracks: HashMap<usize, Vec<usize>> = HashMap::new();
        for (ti, track) in trace.tracks.iter().enumerate() {
            if !track.gpu { continue; }
            if let Some(rank) = parse_rank(&track.label) {
                rank_tracks.entry(rank).or_default().push(ti);
            }
        }
        let n_ranks = rank_tracks.len();
        if n_ranks < 2 { return; }

        let mut ranks_sorted: Vec<usize> = rank_tracks.keys().copied().collect();
        ranks_sorted.sort();

        // Single pass: group events by (name, rank) → Vec<(ti, ei, dur)> sorted by ts
        // Key: (event_name, rank_index_in_sorted_order)
        let mut by_name_rank: HashMap<u32, Vec<Vec<(usize, usize, f64)>>> = HashMap::new();
        for (ri, &rank) in ranks_sorted.iter().enumerate() {
            let tis = &rank_tracks[&rank];
            let mut events: Vec<(u32, usize, usize, f64, f64)> = Vec::new();
            for &ti in tis {
                for (ei, ev) in trace.tracks[ti].events.iter().enumerate() {
                    events.push((ev.name, ti, ei, ev.dur, ev.ts));
                }
            }
            events.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap());
            for (name, ti, ei, dur, _ts) in events {
                let entry = by_name_rank.entry(name).or_insert_with(|| vec![Vec::new(); n_ranks]);
                entry[ri].push((ti, ei, dur));
            }
        }

        let mut n_collectives = 0u32;
        let mut flagged = 0usize;
        for (_name, rank_seqs) in &by_name_rank {
            if rank_seqs.iter().any(|s| s.is_empty()) { continue; }
            let min_len = rank_seqs.iter().map(|s| s.len()).min().unwrap();
            n_collectives += 1;

            for k in 0..min_len {
                let mut durs: Vec<f64> = rank_seqs.iter().map(|s| s[k].2).collect();
                durs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let n = durs.len();
                let median = if n % 2 == 1 { durs[n / 2] } else { (durs[n / 2 - 1] + durs[n / 2]) / 2.0 };
                let threshold = median * 2.0;

                for seq in rank_seqs {
                    let (ti, ei, dur) = seq[k];
                    if dur > threshold {
                        self.straggler_mask[ti][ei] = true;
                        flagged += 1;
                    }
                }
            }
        }

        eprintln!("  stragglers: {:.1}ms, {} flagged across {} collective names, {} ranks",
            t0.elapsed().as_secs_f64() * 1000.0, flagged, n_collectives, n_ranks);
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
