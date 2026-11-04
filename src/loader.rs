use crate::ktz::{self, KtzDecoder, KtzTrace};
use crate::parse::*;
use crate::types::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use crate::time::Instant;

fn kernel_stats_from_dur_map(dur_map: HashMap<u32, Vec<f64>>) -> Vec<KernelStats> {
    let mut stats: Vec<KernelStats> = dur_map.into_iter().map(|(name, mut durs)| {
        let count = durs.len() as u32;
        let total_dur: f64 = durs.iter().sum();
        let max_dur = durs.iter().copied().fold(0.0f64, f64::max);
        let min_dur = durs.iter().copied().fold(f64::MAX, f64::min);
        let median_dur = crate::types::median_inplace(&mut durs);
        KernelStats { name, count, total_dur, median_dur, max_dur, min_dur }
    }).collect();
    stats.sort_by(|a, b| b.total_dur.partial_cmp(&a.total_dur).unwrap());
    stats
}

/// `kernel_stats_from_dur_map` straight from a trace's tracks — the shared
/// body behind the tests' `make_trace` helper.
#[cfg(test)]
pub(crate) fn compute_kernel_stats(tracks: &[Track]) -> Vec<KernelStats> {
    let mut dur_map: HashMap<u32, Vec<f64>> = HashMap::new();
    for t in tracks {
        for ev in &t.events {
            dur_map.entry(ev.name).or_default().push(ev.dur);
        }
    }
    kernel_stats_from_dur_map(dur_map)
}

/// The fold-half of `build_trace`: merges one decoded `KtzTrace` (the
/// direct columnar decode of one .ktz file — or one rank of a multi-rank
/// load) into the shared tables/lists. Held out as a struct so the
/// single-threaded wasm build can fold decode/assemble one slice per frame
/// (`WasmParsePlan`) while native runs the same merge in a tight loop.
/// Field-for-field the locals of the old per-chunk loop.
pub(crate) struct TraceScan {
    names: InternTable,
    cats: InternTable,
    track_map: HashMap<(u64, u64), Vec<Event>>,
    thread_names: HashMap<(u64, u64), String>,
    flow_events: Vec<(u64, u64, u64, f64, bool)>,
    min_ts: f64,
    max_ts: f64,
    total_events: usize,
    scan_count: u64,
}

impl TraceScan {
    fn new() -> Self {
        Self {
            names: InternTable::new(),
            cats: InternTable::new(),
            track_map: HashMap::new(),
            thread_names: HashMap::new(),
            flow_events: Vec::new(),
            min_ts: f64::MAX, max_ts: f64::MIN,
            total_events: 0, scan_count: 0,
        }
    }

    /// Fold one decoded trace into the shared tables/lists. The args buffer
    /// is NOT part of the scan — the caller takes it out of the `KtzTrace`
    /// first (it becomes the finished `Trace`'s one `raw_bufs` entry).
    fn merge_chunk(&mut self, chunk: KtzTrace) {
        let KtzTrace { events, names, cats, thread_names, flows, min_ts, max_ts, total_events, scan_count, args: _ } = chunk;
        let name_remap = self.names.merge_from(&names);
        let cat_remap = self.cats.merge_from(&cats);

        for (pid, tid, mut ev) in events {
            ev.name = name_remap[ev.name as usize];
            ev.cat = cat_remap[ev.cat as usize];
            self.track_map.entry((pid, tid)).or_default().push(ev);
        }

        for (key, value) in thread_names {
            self.thread_names.entry(key).or_insert(value);
        }

        self.flow_events.extend(flows.iter().copied());

        self.min_ts = self.min_ts.min(min_ts);
        self.max_ts = self.max_ts.max(max_ts);
        self.total_events += total_events;
        self.scan_count += scan_count;
    }

    /// Drain the per-(pid,tid) event lists for track assembly, into a Vec
    /// so both the native par_iter and the wasm one-track-per-slice loop
    /// walk the same entries.
    fn take_track_entries(&mut self) -> Vec<((u64, u64), Vec<Event>)> {
        self.track_map.drain().collect()
    }
}

/// One track's assembly from its raw event list: normalize timestamps to
/// the trace min, sort (exploiting the mostly-sorted input), Tetris-pack
/// depths, prefix-max durations, classify CPU vs GPU. A pure function of
/// its inputs so it runs per-track on rayon natively and once per frame
/// slice on wasm.
fn assemble_track(
    pid: u64,
    tid: u64,
    mut evs: Vec<Event>,
    min_ts: f64,
    cats: &InternTable,
    thread_names: &HashMap<(u64, u64), String>,
) -> Track {
    for ev in evs.iter_mut() { ev.ts -= min_ts; }
    let mut sorted_end = evs.len();
    for i in 1..evs.len() {
        if evs[i].ts < evs[i - 1].ts {
            sorted_end = i;
            break;
        }
    }
    if sorted_end < evs.len() {
        if evs.len() - sorted_end < evs.len() / 100 {
            let mut tail = evs.split_off(sorted_end);
            tail.sort_unstable_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap());
            let prefix = std::mem::take(&mut evs);
            evs.reserve(prefix.len() + tail.len());
            let (mut i, mut j) = (0, 0);
            while i < prefix.len() && j < tail.len() {
                if prefix[i].ts <= tail[j].ts {
                    evs.push(prefix[i]); i += 1;
                } else {
                    evs.push(tail[j]); j += 1;
                }
            }
            if i < prefix.len() { evs.extend_from_slice(&prefix[i..]); }
            if j < tail.len() { evs.extend_from_slice(&tail[j..]); }
        } else {
            evs.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap());
        }
    }
    let mut lanes: Vec<f64> = Vec::new();
    let mut max_depth: u16 = 1;
    for ev in evs.iter_mut() {
        let d = lanes.iter().position(|&end| end <= ev.ts)
            .unwrap_or_else(|| { lanes.push(0.0); lanes.len() - 1 });
        lanes[d] = ev.ts + ev.dur;
        ev.depth = d as u16;
        max_depth = max_depth.max(d as u16 + 1);
    }
    let mut prefix_max_dur = Vec::with_capacity(evs.len());
    let mut running_max = 0.0f64;
    for ev in &evs {
        running_max = running_max.max(ev.dur);
        prefix_max_dur.push(running_max);
    }
    let gpu_count = evs.iter().filter(|e| {
        let c = cats.str(e.cat);
        c == "kernel" || c.starts_with("gpu_")
    }).count();
    let gpu = gpu_count > evs.len() / 2;
    let label = thread_names.get(&(pid, tid)).cloned().unwrap_or_else(|| {
        if gpu { format!("GPU {tid}") } else { format!("Thread {tid}") }
    });    evs.shrink_to_fit();
    Track { label, gpu, events: evs, max_depth, prefix_max_dur, raw_buf_idx: 0 }
}

/// Everything a load does once every track is assembled: metadata scans
/// (over the ktz head/foot text, or — for a JSON trace — the raw JSON
/// itself when `meta` is `None`), track ordering, flow pairing, kernel
/// stats, and the `Trace` itself. `raw_buf` is the buffer events' args
/// offsets index into: the decoder-written compact args text for `.ktz`,
/// the whole decompressed JSON for a direct JSON load. Shared by the native
/// path and the wasm plan's assemble phase.
fn finish_trace(
    meta: Option<&[u8]>,
    raw_buf: Arc<ArgsBuf>,
    scan: TraceScan,
    keyed_tracks: Vec<((u64, u64), Track)>,
    t0: &Instant,
) -> Result<Trace, String> {
    let meta: &[u8] = meta.unwrap_or(&raw_buf[..]);
    let TraceScan { names, cats, track_map: _, thread_names: _, mut flow_events, min_ts, mut max_ts, total_events, scan_count } = scan;
    let mut names = names.into_strings();
    let mut cats = cats.into_strings();

    let mut device = String::new();
    if let Some(dp) = find_key(meta, b"deviceProperties") {
        let mut p = dp + "\"deviceProperties\"".len();
        p = skip_ws(meta, p);
        if p < meta.len() && meta[p] == b':' { p += 1; }
        p = skip_ws(meta, p);
        if let Some(np) = find_key(&meta[p..], b"name") {
            let mut q = p + np + "\"name\"".len();
            q = skip_ws(meta, q);
            if q < meta.len() && meta[q] == b':' { q += 1; }
            q = skip_ws(meta, q);
            if q < meta.len() && meta[q] == b'"' {
                let end = skip_string(meta, q);
                device = json_unescape(std::str::from_utf8(&meta[q + 1..end - 1]).unwrap_or(""));
            }
        }
    }

    // vLLM traces embed a top-level `vllm_version` string (e.g.
    // "0.26.1rc1.dev528+gf8d03e774"). Plenty of traces lack it — leave it empty.
    let mut vllm_version = String::new();
    if let Some(vp) = find_key(meta, b"vllm_version") {
        let mut q = vp + "\"vllm_version\"".len();
        q = skip_ws(meta, q);
        if q < meta.len() && meta[q] == b':' { q += 1; }
        q = skip_ws(meta, q);
        if q < meta.len() && meta[q] == b'"' {
            let end = skip_string(meta, q);
            vllm_version = json_unescape(std::str::from_utf8(&meta[q + 1..end - 1]).unwrap_or(""));
        }
    }

    // `distributedInfo.{rank,world_size}` identify this rank within the job.
    // `"rank"` (singular) only appears at the top of the distributedInfo object;
    // the pg_config arrays use `"ranks"` (plural), which find_key won't match.
    let (mut dist_rank, mut dist_world) = (-1i32, 0i32);
    if let Some(di) = find_key(meta, b"distributedInfo") {
        if let Some(r) = find_int_field(meta, di, b"rank") { dist_rank = r as i32; }
        if let Some(w) = find_int_field(meta, di, b"world_size") { dist_world = w as i32; }
    }
    tlog!("  scan: {:.3}s ({} objects, {} events, {} names)",
        t0.elapsed().as_secs_f64(), scan_count, total_events, names.len());

    if min_ts == f64::MAX {
        return Err("no duration events found".into());
    }

    let t2 = Instant::now();
    let mut keyed_tracks = keyed_tracks;

    keyed_tracks.sort_by(|a, b| b.1.gpu.cmp(&a.1.gpu).then_with(|| b.1.events.len().cmp(&a.1.events.len())));

    let ptid_to_track: HashMap<(u64, u64), usize> = keyed_tracks.iter().enumerate()
        .map(|(i, (key, _))| (*key, i)).collect();
    let tracks: Vec<Track> = keyed_tracks.into_iter().map(|(_, t)| t).collect();

    max_ts -= min_ts;

    let mut flow_pairs: Vec<FlowPair> = Vec::new();
    flow_events.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.3.partial_cmp(&b.3).unwrap()));
    let mut i = 0;
    while i < flow_events.len() {
        let cur_id = flow_events[i].0;
        let id_start = i;
        while i < flow_events.len() && flow_events[i].0 == cur_id { i += 1; }
        let id_end = i;

        let starts: Vec<_> = flow_events[id_start..id_end].iter().filter(|e| e.4).collect();
        let ends: Vec<_> = flow_events[id_start..id_end].iter().filter(|e| !e.4).collect();
        if starts.is_empty() || ends.is_empty() { continue; }

        for s in &starts {
            let src_ti = match ptid_to_track.get(&(s.1, s.2)) { Some(&v) => v, None => continue };
            let s_adj = s.3 - min_ts;
            for f in &ends {
                let dst_ti = match ptid_to_track.get(&(f.1, f.2)) { Some(&v) => v, None => continue };
                let f_adj = f.3 - min_ts;
                flow_pairs.push(FlowPair { src_track: src_ti as u32, dst_track: dst_ti as u32, src_ts: s_adj, dst_ts: f_adj });
                flow_pairs.push(FlowPair { src_track: dst_ti as u32, dst_track: src_ti as u32, src_ts: f_adj, dst_ts: s_adj });
            }
        }
    }
    drop(flow_events);
    flow_pairs.sort_unstable_by(|a, b| a.src_track.cmp(&b.src_track)
        .then_with(|| a.src_ts.partial_cmp(&b.src_ts).unwrap()));
    flow_pairs.dedup_by(|a, b| a.src_track == b.src_track && a.src_ts == b.src_ts
        && a.dst_track == b.dst_track && a.dst_ts == b.dst_ts);
    drop(ptid_to_track);

    let mut dur_map: HashMap<u32, Vec<f64>> = HashMap::new();
    for t in &tracks {
        for ev in &t.events {
            dur_map.entry(ev.name).or_default().push(ev.dur);
        }
    }
    let stats = kernel_stats_from_dur_map(dur_map);

    names.shrink_to_fit();
    cats.shrink_to_fit();

    tlog!("  lanes: {:.3}s ({} tracks, {} flow_pairs)", t2.elapsed().as_secs_f64(), tracks.len(), flow_pairs.len());
    Ok(Trace { tracks, names, cats, raw_bufs: vec![raw_buf], stats, max_ts, min_ts, total_events, device, vllm_version, dist_rank, dist_world, flow_pairs, rank_paths: Vec::new() })
}

/// The scan -> assemble -> finish half every decoder shares. `raw_buf` is
/// the args buffer the decoder's events index into; `meta` is the small
/// text `finish_trace` scans for metadata, or `None` to scan `raw_buf`
/// itself (a JSON trace's top-level keys are already in there).
fn assemble_and_finish(
    meta: Option<&[u8]>,
    raw_buf: Arc<ArgsBuf>,
    mut scan: TraceScan,
    t0: &Instant,
) -> Result<Trace, String> {
    let min_ts = scan.min_ts;
    let entries = scan.take_track_entries();
    let cats = &scan.cats;
    let thread_names = &scan.thread_names;
    let keyed_tracks: Vec<((u64, u64), Track)> = entries
        .into_par_iter()
        .map(|((pid, tid), evs)| ((pid, tid), assemble_track(pid, tid, evs, min_ts, cats, thread_names)))
        .collect();
    finish_trace(meta, raw_buf, scan, keyed_tracks, t0)
}

/// Decode a .ktz file's bytes into a `Trace`: `KtzDecoder` walks the
/// columnar sections straight into a `KtzTrace` (events, interned
/// names/cats, args text — no JSON is ever materialized), then the events
/// go through the scan -> assemble -> finish pipeline every load path
/// shares. `meta` is the ktz head/foot text `finish_trace` scans for
/// deviceProperties/vllm_version/distributedInfo.
fn build_trace(meta: Vec<u8>, mut ktz: KtzTrace, t0: &Instant) -> Result<Trace, String> {
    let args = std::mem::take(&mut ktz.args);
    let mut scan = TraceScan::new();
    scan.merge_chunk(ktz);
    assemble_and_finish(Some(&meta), Arc::new(ArgsBuf(args)), scan, t0)
}

/// Build a `Trace` from decompressed Chrome/kineto JSON. The raw JSON
/// becomes the trace's args buffer (events reference their args by offset,
/// never by copy), which is also what `finish_trace` scans for metadata.
fn build_json_trace(raw: Vec<u8>, counter: &AtomicUsize, t0: &Instant) -> Result<Trace, String> {
    let parsed = crate::json_trace::parse(raw, counter)?;
    let mut scan = TraceScan::new();
    for chunk in parsed.chunks { scan.merge_chunk(chunk); }
    let raw_buf = Arc::new(ArgsBuf(parsed.raw));
    assemble_and_finish(None, raw_buf, scan, t0)
}

/// Shared body of every load entry point: sniff the container, decode,
/// build. `.ktz` is the primary format; a gzip stream or a JSON
/// object/array is a Chrome/kineto trace loaded directly, through the same
/// ktz-shaped pipeline.
fn decode_any(data: Vec<u8>, counter: &AtomicUsize, t0: &Instant) -> Result<Trace, String> {
    if ktz::is_ktz(&data) {
        let mut dec = KtzDecoder::new(&data)?;
        let n = dec.n_events();
        let mut out = KtzTrace::with_events(n);
        while !dec.step(&mut out, 65_536, counter)? {}
        let meta = dec.into_head_foot();
        return build_trace(meta, out, t0);
    }
    if crate::json_trace::is_gzip(&data) {
        let raw = crate::json_trace::decompress_gz(&data)?;
        return build_json_trace(raw, counter, t0);
    }
    if crate::json_trace::looks_like_json(&data) {
        return build_json_trace(data, counter, t0);
    }
    Err("unsupported file type - expected a .ktz trace or a Chrome/kineto JSON trace (.json, .json.gz)".into())
}

/// `decode_any` with the dropped file's name prefixed onto errors.
#[cfg(target_arch = "wasm32")]
fn decode_any_named(name: &str, data: Vec<u8>, counter: &AtomicUsize, t0: &Instant) -> Result<Trace, String> {
    decode_any(data, counter, t0).map_err(|e| format!("{name}: {e}"))
}

pub fn load_trace(path: &str, counter: &Arc<AtomicUsize>) -> Result<Trace, String> {
    let t0 = Instant::now();
    let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let trace = decode_any(data, counter, &t0).map_err(|e| format!("{path}: {e}"))?;
    let event_bytes: usize = trace.tracks.iter().map(|t| t.events.len() * std::mem::size_of::<Event>()).sum();
    let str_bytes: usize = trace.names.iter().chain(trace.cats.iter()).map(|s| s.len() + std::mem::size_of::<String>()).sum();
    let args_bytes: usize = trace.raw_bufs.iter().map(|b| b.len()).sum();
    tlog!("  memory: events={}MB strings={}MB args={}MB total={}MB",
        event_bytes / 1024 / 1024, str_bytes / 1024 / 1024, args_bytes / 1024 / 1024,
        (event_bytes + str_bytes + args_bytes) / 1024 / 1024);
    tlog!("  total: {:.3}s", t0.elapsed().as_secs_f64());
    Ok(trace)
}

pub fn load_trace_progressive(
    path: &str,
    counter: &Arc<AtomicUsize>,
    tx: &std::sync::mpsc::Sender<Result<Trace, String>>,
) {
    match load_trace(path, counter) {
        Ok(trace) => { let _ = tx.send(Ok(trace)); }
        Err(e) => { let _ = tx.send(Err(e)); }
    }
}

/// Files a directory walk / folder drop picks up: the primary `.ktz`
/// container, plus raw Chrome/kineto traces (`.json`, and the
/// `.pt.trace.json.gz` spelling PyTorch's profiler writes). The name is
/// only a filter — `decode_any` still sniffs the bytes. Plain `.json`
/// files additionally have to look like traces (see
/// `is_json_trace_file`); profile folders also ship `manifest.json` and
/// friends, which are valid JSON but not traces.
pub(crate) fn is_trace_file(name: &str) -> bool {
    name.ends_with(".ktz") || name.ends_with(".json.gz") || name.ends_with(".json")
}

/// The name-only variant of the folder-walk filter, for the wasm drop walk
/// (which can't peek at file bytes synchronously): a plain `.json` has to
/// carry a trace-ish name, so a folder's `manifest.json` isn't opened as a
/// trace. `.json.gz` needs no check.
#[cfg(target_arch = "wasm32")]
pub(crate) fn is_trace_file_by_name(name: &str) -> bool {
    if name.ends_with(".json") {
        return name.to_ascii_lowercase().contains("trace");
    }
    is_trace_file(name)
}

/// A raw trace transcoded next to itself (`trace.pt.trace.json.gz` ->
/// `trace.pt.trace.json.gz.ktz`) otherwise opens the folder as two copies
/// of every rank. When the `.ktz` sibling exists, the folder walk keeps
/// only it — the columnar container is the primary format.
fn has_ktz_sibling(path: &std::path::Path) -> bool {
    if path.extension().and_then(|e| e.to_str()) == Some("ktz") { return false; }
    let mut with_suffix = path.as_os_str().to_owned();
    with_suffix.push(".ktz");
    std::path::Path::new(&with_suffix).is_file()
}

/// Whether a plain (uncompressed) `.json` file found by a directory walk is
/// a trace. `.json.gz` is unambiguous by extension, so only this case needs
/// a look. Accepts a trace-shaped filename outright, otherwise peeks at the
/// head for the `traceEvents` key (every producer we care about writes it
/// before any multi-megabyte metadata).
fn is_json_trace_file(path: &std::path::Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name.to_ascii_lowercase().contains("trace") { return true; }
    use std::io::Read;
    const PEEK: usize = 1 << 20;
    let Ok(mut f) = std::fs::File::open(path) else { return false };
    let mut head = vec![0u8; PEEK];
    let Ok(n) = f.read(&mut head) else { return false };
    head.truncate(n);
    crate::parse::find_key(&head, b"traceEvents").is_some()
}

fn expand_dirs(paths: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    for path in paths {
        let p = std::path::Path::new(path);
        if p.is_dir() {
            dirs.push(p.to_path_buf());
        } else {
            out.push(path.clone());
        }
    }
    while let Some(dir) = dirs.pop() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let ep = entry.path();
                if ep.is_dir() {
                    dirs.push(ep);
                } else if let Some(name) = ep.file_name().and_then(|n| n.to_str()) {
                    let plain_json = name.ends_with(".json");
                    if is_trace_file(name) && (!plain_json || is_json_trace_file(&ep))
                        && !has_ktz_sibling(&ep)
                    {
                        out.push(ep.to_string_lossy().into());
                    }
                }
            }
        }
    }
    out
}

fn extract_rank(fname: &str) -> Option<(String, usize)> {
    if let Some(pos) = fname.find("-rank-") {
        let prefix = &fname[..pos];
        let after = &fname[pos + 6..];
        if let Some(dot) = after.find('.') {
            if let Ok(rank) = after[..dot].parse::<usize>() {
                return Some((prefix.to_string(), rank));
            }
        }
    }
    for part in fname.split('_') {
        if part.starts_with("ep") {
            if let Ok(rank) = part[2..].parse::<usize>() {
                return Some(("ep-group".to_string(), rank));
            }
        }
    }
    None
}

/// Groups an already-expanded (no directories left) flat file list by
/// `extract_rank`'s key, splitting singleton "groups" back out to
/// `standalone`. Shared by `detect_rank_groups`, which calls this once per
/// top-level directory argument (see there for why: `extract_rank`'s keys —
/// especially the `"ep-group"` fallback — aren't globally unique across
/// unrelated runs, only within one directory's own files).
fn group_paths(expanded: &[String]) -> (Vec<Vec<(usize, String)>>, Vec<String>) {
    let mut groups: HashMap<String, Vec<(usize, String)>> = HashMap::new();
    let mut standalone = Vec::new();

    for path in expanded {
        let fname = std::path::Path::new(path)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or(path.as_str());
        if let Some((key, rank)) = extract_rank(fname) {
            groups.entry(key).or_default().push((rank, path.clone()));
        } else {
            standalone.push(path.clone());
        }
    }

    let mut rank_groups: Vec<Vec<(usize, String)>> = Vec::new();
    for mut group in groups.into_values() {
        if group.len() > 1 {
            group.sort_by_key(|(rank, _)| *rank);
            rank_groups.push(group);
        } else {
            standalone.push(group.remove(0).1);
        }
    }

    (rank_groups, standalone)
}

/// Detects multi-rank groups among the given top-level paths (files and/or
/// directories — e.g. CLI args, or everything dropped onto the window at
/// once). Each *directory* argument is expanded and grouped in total
/// isolation from every other argument (own `expand_dirs` + `group_paths`
/// call), so opening N sibling sweep-result folders together always yields N
/// separate opens instead of silently merging them — `extract_rank`'s keys
/// (filename prefix before `-rank-N`, or the literal `"ep-group"` fallback)
/// routinely collide across different runs that happen to share a naming
/// convention, so pooling every directory's files into one grouping pass
/// would misattribute one folder's ranks to another's.
///
/// Bare file arguments (not directories) are still pooled together and
/// grouped as one batch, same as before — that's the "select several rank
/// files and drop them all at once" case, where they're *meant* to merge.
///
/// The `Option<String>` alongside each result is the directory argument it
/// was expanded from (`None` for bare file arguments), so the caller can set
/// up per-pane `reload_dir`/watch support for each directory independently.
pub fn detect_rank_groups(paths: &[String]) -> (Vec<(Vec<(usize, String)>, Option<String>)>, Vec<(String, Option<String>)>) {
    let mut rank_groups = Vec::new();
    let mut standalone = Vec::new();
    let mut loose_files = Vec::new();

    for path in paths {
        if std::path::Path::new(path).is_dir() {
            let (dir_groups, dir_standalone) = group_paths(&expand_dirs(std::slice::from_ref(path)));
            rank_groups.extend(dir_groups.into_iter().map(|g| (g, Some(path.clone()))));
            standalone.extend(dir_standalone.into_iter().map(|p| (p, Some(path.clone()))));
        } else {
            loose_files.push(path.clone());
        }
    }

    if !loose_files.is_empty() {
        let (loose_groups, loose_standalone) = group_paths(&expand_dirs(&loose_files));
        rank_groups.extend(loose_groups.into_iter().map(|g| (g, None)));
        standalone.extend(loose_standalone.into_iter().map(|p| (p, None)));
    }

    (rank_groups, standalone)
}

pub fn merge_traces(traces: Vec<(usize, Trace)>) -> Trace {
    let global_min = traces.iter().map(|(_, t)| t.min_ts).fold(f64::MAX, f64::min);

    let mut names = InternTable::new();
    let mut cats = InternTable::new();
    let mut device = String::new();
    let mut vllm_version = String::new();
    // A merged trace spans many ranks, so there's no single rank id; keep the
    // shared world_size (from the first rank that reports one) for context.
    let mut dist_world = 0i32;

    let mut remap_info: Vec<(Vec<u32>, Vec<u32>, f64)> = Vec::with_capacity(traces.len());
    let mut all_raw_bufs: Vec<Arc<ArgsBuf>> = Vec::new();

    for (_, trace) in &traces {
        let time_offset = trace.min_ts - global_min;

        let name_remap = names.merge_strings(&trace.names);
        let cat_remap = cats.merge_strings(&trace.cats);

        if device.is_empty() && !trace.device.is_empty() {
            device = trace.device.clone();
        }
        if vllm_version.is_empty() && !trace.vllm_version.is_empty() {
            vllm_version = trace.vllm_version.clone();
        }
        if dist_world == 0 && trace.dist_world > 0 {
            dist_world = trace.dist_world;
        }

        remap_info.push((name_remap, cat_remap, time_offset));
    }

    // Buffer base of each rank inside the merged `raw_bufs`: each rank keeps
    // its own decoder-written args buffer (one per .ktz file), so a rank's
    // tracks must be rebased by how many buffers the ranks before it
    // contributed, not blanket-assigned one buffer per rank. raw_buf_idx is
    // u8, capping a merge at 256 args buffers (one per rank) — far past the
    // 32-rank merges this exists for.
    let mut buf_bases: Vec<usize> = Vec::with_capacity(traces.len());
    let mut base = 0usize;
    for (_, t) in &traces {
        buf_bases.push(base);
        base += t.raw_bufs.len();
    }

    let remapped: Vec<_> = traces.into_iter().zip(remap_info).enumerate()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(trace_idx, ((rank, mut trace), (name_remap, cat_remap, time_offset)))| {
            let buf_base = buf_bases[trace_idx];
            let mut max_ts: f64 = 0.0;
            let mut total_events = 0usize;
            let mut local_dur: HashMap<u32, Vec<f64>> = HashMap::new();
            for track in &mut trace.tracks {
                track.label = format!("  Rank {} {}", rank, track.label);
                track.raw_buf_idx = (buf_base + track.raw_buf_idx as usize) as u8;
                for ev in &mut track.events {
                    ev.ts += time_offset;
                    ev.name = name_remap[ev.name as usize];
                    ev.cat = cat_remap[ev.cat as usize];
                    max_ts = max_ts.max(ev.ts + ev.dur);
                    local_dur.entry(ev.name).or_default().push(ev.dur);
                }
                total_events += track.events.len();
            }

            for f in &mut trace.flow_pairs {
                f.src_ts += time_offset;
                f.dst_ts += time_offset;
            }

            (trace.tracks, trace.raw_bufs, max_ts, total_events, local_dur, trace.flow_pairs)
        })
        .collect();

    let mut all_tracks: Vec<Track> = Vec::new();
    let mut all_flow_pairs: Vec<FlowPair> = Vec::new();
    let mut total_events = 0;
    let mut max_ts: f64 = 0.0;
    let mut dur_map: HashMap<u32, Vec<f64>> = HashMap::new();

    for (tracks, raw_bufs, mt, te, local_dur, mut fps) in remapped {
        let offset = all_tracks.len() as u32;
        for f in &mut fps {
            f.src_track += offset;
            f.dst_track += offset;
        }
        all_flow_pairs.extend(fps);
        all_tracks.extend(tracks);
        all_raw_bufs.extend(raw_bufs);
        total_events += te;
        max_ts = max_ts.max(mt);
        for (name, durs) in local_dur {
            dur_map.entry(name).or_default().extend(durs);
        }
    }

    // Stable sort by rank *number* only (not the whole label string, which
    // would put "Rank 10" before "Rank 2"): tracks already arrive in a
    // sensible per-rank order (GPU-first, busiest first — each individual
    // trace's own sort in build_trace), and a stable sort preserves that
    // within each rank while fixing the cross-rank order to be numeric.
    // Uses the same crate::state::parse_rank the GPU-merge-by-rank display
    // feature already relies on, rather than a second rank-parsing
    // function — same label shape, same job.
    let mut sort_perm: Vec<usize> = (0..all_tracks.len()).collect();
    sort_perm.sort_by_key(|&i| crate::state::parse_rank(&all_tracks[i].label).unwrap_or(usize::MAX));
    let mut old_to_new = vec![0u32; all_tracks.len()];
    for (new_i, &old_i) in sort_perm.iter().enumerate() {
        old_to_new[old_i] = new_i as u32;
    }
    let sorted_tracks: Vec<Track> = sort_perm.into_iter().map(|i| std::mem::replace(&mut all_tracks[i], Track {
        label: String::new(), gpu: false, events: Vec::new(), max_depth: 0,
        prefix_max_dur: Vec::new(), raw_buf_idx: 0,
    })).collect();

    for f in &mut all_flow_pairs {
        f.src_track = old_to_new[f.src_track as usize];
        f.dst_track = old_to_new[f.dst_track as usize];
    }
    all_flow_pairs.sort_unstable_by(|a, b| a.src_track.cmp(&b.src_track)
        .then_with(|| a.src_ts.partial_cmp(&b.src_ts).unwrap()));
    all_flow_pairs.dedup_by(|a, b| a.src_track == b.src_track && a.src_ts == b.src_ts
        && a.dst_track == b.dst_track && a.dst_ts == b.dst_ts);

    let stats = kernel_stats_from_dur_map(dur_map);

    let mut names = names.into_strings();
    let mut cats = cats.into_strings();
    names.shrink_to_fit();
    cats.shrink_to_fit();

    Trace {
        tracks: sorted_tracks, names, cats, raw_bufs: all_raw_bufs, stats,
        max_ts, min_ts: global_min, total_events, device, vllm_version,
        dist_rank: -1, dist_world, flow_pairs: all_flow_pairs,
        rank_paths: Vec::new(),
    }
}

/// Default marker substring used to align DP groups — see
/// `sync_multi_rank_clocks`. A DeepEP `combine` kernel launch is a good
/// sync point because every rank in a run reaches it early and reliably.
pub const DEFAULT_SYNC_MARKER: &str = "deep_ep::elastic::combine_impl";

/// Extracts `(dp, tp)` from a rank's source filename, e.g.
/// `dp2_pp0_tp5_dcp0_ep17_rank1....json` -> `(Some(2), Some(5))`. Mirrors
/// `sync_traces.py`'s `parse_dp_tp` regex (`\bdp(\d+)`, `\btp(\d+)`): a
/// literal "dp"/"tp" not immediately preceded by an alphanumeric
/// character, followed by one or more digits. Pure string parsing, so this
/// (and `sync_multi_rank_clocks`) builds and runs on wasm too — `Pane::
/// sync_clocks` sources rank/filename pairs from `reload_paths`
/// (native directory opens) or the merged trace's own persisted
/// `rank_paths`.
pub(crate) fn parse_dp_tp(fname: &str) -> (Option<u32>, Option<u32>) {
    fn find(fname: &str, pfx: &str) -> Option<u32> {
        let bytes = fname.as_bytes();
        let pb = pfx.as_bytes();
        let mut i = 0;
        while i + pb.len() < bytes.len() {
            let boundary_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            if boundary_ok && &bytes[i..i + pb.len()] == pb && bytes[i + pb.len()].is_ascii_digit() {
                let start = i + pb.len();
                let mut end = start;
                while end < bytes.len() && bytes[end].is_ascii_digit() { end += 1; }
                return fname[start..end].parse().ok();
            }
            i += 1;
        }
        None
    }
    (find(fname, "dp"), find(fname, "tp"))
}

/// Corrects INTER-node clock skew across an already-merged multi-rank
/// trace, entirely in memory — a Rust port of `sync_traces.py`, which does
/// the same computation but writes shifted copies of the source JSON
/// files to disk. The skew is inter-node (inter-DP-group): each DP pod
/// runs on its own machine with its own clock, so traces from different DP
/// groups start at different absolute times. It is NOT intra-node: the TP
/// ranks inside one pod share the same host clock, so their relative
/// timing is already exact and must not be adjusted independently — only
/// one shift per DP group, applied uniformly to every TP member, corrects
/// the inter-node skew without distorting correct intra-node relationships.
///
/// Finds each rank's first (by ts) event whose interned name contains
/// `marker`, groups ranks by the `dp` index parsed from `rank_paths`'
/// filenames (a rank with no parseable `dp` becomes its own singleton
/// group, so it's still aligned rather than ignored — matching
/// `sync_traces.py`'s `solo:<filename>` fallback key), and takes each
/// group's reference as its lowest-`tp` member's marker-event end time
/// (`ts + dur`). All groups are anchored to the *latest* group reference,
/// so every shift is >= 0, matching the script exactly.
///
/// Mutates every affected track's events in place, then the flow pairs
/// (whose `src_ts`/`dst_ts` need the shift of whichever track they
/// reference, not necessarily the same shift as each other), then
/// recomputes `trace.max_ts` (used directly for the view's zoom bounds) —
/// `min_ts` is left alone, since nothing outside this file's own
/// merge/serialize logic reads it. A uniform per-track shift changes
/// nothing about event order or duration, so `prefix_max_dur` (built from
/// durations, not timestamps) stays valid without recomputation.
pub fn sync_multi_rank_clocks(trace: &mut Trace, rank_paths: &[(usize, String)], marker: &str) -> Result<String, String> {
    let matching_names: std::collections::HashSet<u32> = trace.names.iter().enumerate()
        .filter(|(_, n)| n.contains(marker))
        .map(|(i, _)| i as u32)
        .collect();
    if matching_names.is_empty() {
        return Err(format!("no event name contains {marker:?} in this trace"));
    }

    struct RankInfo { rank: usize, dp: Option<u32>, tp: Option<u32>, end: f64 }
    let mut infos: Vec<RankInfo> = Vec::new();
    for (rank, path) in rank_paths {
        let fname = std::path::Path::new(path).file_name().and_then(|s| s.to_str()).unwrap_or(path);
        let (dp, tp) = parse_dp_tp(fname);
        let prefix = format!("  Rank {rank} ");
        let mut best: Option<(f64, f64)> = None;
        for t in &trace.tracks {
            if !t.label.starts_with(&prefix) { continue; }
            for e in &t.events {
                if matching_names.contains(&e.name) {
                    let is_earlier = match best { None => true, Some((bts, _)) => e.ts < bts };
                    if is_earlier { best = Some((e.ts, e.dur)); }
                }
            }
        }
        if let Some((ts, dur)) = best {
            infos.push(RankInfo { rank: *rank, dp, tp, end: ts + dur });
        }
    }
    if infos.is_empty() {
        return Err(format!("no rank had an event containing {marker:?}"));
    }

    #[derive(PartialEq, Eq, Hash, Clone, Copy)]
    enum GroupKey { Dp(u32), Solo(usize) }
    let key_of = |info: &RankInfo| match info.dp {
        Some(dp) => GroupKey::Dp(dp),
        None => GroupKey::Solo(info.rank),
    };

    let mut groups: HashMap<GroupKey, Vec<usize>> = HashMap::new();
    for (i, info) in infos.iter().enumerate() {
        groups.entry(key_of(info)).or_default().push(i);
    }
    let mut group_ref: HashMap<GroupKey, f64> = HashMap::new();
    for (key, idxs) in &groups {
        let rep = idxs.iter().min_by_key(|&&i| infos[i].tp.unwrap_or(0)).unwrap();
        group_ref.insert(*key, infos[*rep].end);
    }
    let target = group_ref.values().copied().fold(f64::MIN, f64::max);

    let mut rank_shift: HashMap<usize, f64> = HashMap::new();
    let mut summary = String::new();
    for key in groups.keys() {
        use std::fmt::Write;
        let shift = target - group_ref[key];
        let members: Vec<&RankInfo> = groups[key].iter().map(|&i| &infos[i]).collect();
        let label = match key { GroupKey::Dp(d) => format!("dp{d}"), GroupKey::Solo(_) => "solo".to_string() };
        let _ = writeln!(summary, "{label}: shift={shift:+.1}us ({} rank{})",
            members.len(), if members.len() == 1 { "" } else { "s" });
        for m in members {
            rank_shift.insert(m.rank, shift);
        }
    }

    let mut track_shift = vec![0.0f64; trace.tracks.len()];
    for (ti, t) in trace.tracks.iter().enumerate() {
        if let Some(r) = crate::state::parse_rank(&t.label) {
            if let Some(&shift) = rank_shift.get(&r) {
                track_shift[ti] = shift;
            }
        }
    }
    for (ti, t) in trace.tracks.iter_mut().enumerate() {
        let shift = track_shift[ti];
        if shift != 0.0 {
            for e in &mut t.events { e.ts += shift; }
        }
    }
    for f in &mut trace.flow_pairs {
        f.src_ts += track_shift[f.src_track as usize];
        f.dst_ts += track_shift[f.dst_track as usize];
    }

    let mut max_ts = f64::MIN;
    for t in &trace.tracks {
        for e in &t.events {
            max_ts = max_ts.max(e.ts + e.dur);
        }
    }
    trace.max_ts = max_ts;

    Ok(format!("Aligned {} ranks across {} DP group{} on {marker:?}:\n{summary}",
        infos.len(), groups.len(), if groups.len() == 1 { "" } else { "s" }))
}

/// Transient-heap budget for one wave of concurrent rank loads in
/// `load_multi_progressive`: the decompressed section payloads of the
/// in-flight ranks (each alive until its decoder finishes) plus the
/// `KtzTrace`s being built. 8GB keeps a 32-rank c16 .ktz merge (~1-2GB
/// estimated per rank) to a handful of ranks per wave instead of
/// everything at once, while small traces still load together in one wave.
const WAVE_MEM_BUDGET: u64 = 8 << 30;

/// How many of `estimates` (per-rank transient-heap estimates from
/// `estimated_load_heap`) fit in one wave: greedily pack ranks until the
/// next one would overspend the budget, always at least one so a single
/// oversized rank still loads (alone, exactly like before waves existed).
pub(crate) fn wave_count(estimates: &[u64]) -> usize {
    let mut n = 0usize;
    let mut budget = WAVE_MEM_BUDGET as i64;
    for &e in estimates {
        if n > 0 && budget < e as i64 { break; }
        budget -= e as i64;
        n += 1;
    }
    n.max(1)
}

/// Estimated transient heap for loading one rank: the decompressed section
/// payload (exact, from the KTZ3/4 frame table — alive until the decoder
/// finishes) plus the `KtzTrace` being built (~224 bytes/event: the events
/// Vec at 48B/event, the args text, and track assembly). When the header
/// can't be read (or a KTZ2 file, whose payload size lives inside the zstd
/// frame), fall back to file*256 (ktz's zstd ratio on columns runs well
/// past 100x) plus the event-count hint.
fn estimated_load_heap(path: &str) -> u64 {
    let file = std::fs::metadata(path).map(|m| m.len()).unwrap_or(96 << 20);
    if path.ends_with(".gz") {
        // A direct JSON trace keeps its whole decompressed text as the args
        // buffer, so that text is the retained payload — read the exact size
        // from the gzip trailer rather than guessing from the compressed
        // bytes. The extra quarter is the Event/track side.
        let raw = crate::json_trace::gz_uncompressed_size(path)
            .unwrap_or_else(|| file.saturating_mul(25));
        return raw.saturating_add(raw / 4);
    }
    if path.ends_with(".json") {
        return file.saturating_add(file / 4);
    }
    let payload = crate::ktz::payload_size_hint(path)
        .unwrap_or_else(|| file.saturating_mul(256));
    payload.saturating_add(
        crate::ktz::event_count_hint(path).unwrap_or(0).saturating_mul(224),
    )
}

pub fn load_multi_progressive(
    rank_paths: Vec<(usize, String)>,
    counter: &Arc<AtomicUsize>,
    tx: &std::sync::mpsc::Sender<Result<Trace, String>>,
) {
    let t0 = Instant::now();

    // Preserve the previous std::thread::scope + JoinHandle::join().ok()
    // behavior: a rank whose load panics is dropped (with a message) rather
    // than taking the whole multi-rank load down with it.
    //
    // Ranks load in waves bounded by estimated heap, not all at once: each
    // rank's decompressed section payload stays on the heap until its
    // decoder finishes, so dozens of concurrent loads transiently need
    // multiples of that. A wave keeps the in-flight payloads under
    // WAVE_MEM_BUDGET; completed ranks hold only their finished traces.
    // Waves run in order, so results arrive in the same rank order the old
    // all-at-once par_iter gave.
    let estimates: Vec<u64> = rank_paths.iter().map(|(_, p)| estimated_load_heap(p)).collect();
    let wave = wave_count(&estimates);
    tlog!("  loading {} ranks in waves of {}", rank_paths.len(), wave);
    let mut results: Vec<(usize, Result<Trace, String>)> = Vec::new();
    for wave_paths in rank_paths.chunks(wave) {
        let wave_results: Vec<(usize, Result<Trace, String>)> = wave_paths.par_iter().filter_map(|(rank, path)| {
            let r = *rank;
            let ctr = counter.clone();
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load_trace(path, &ctr))) {
                Ok(result) => Some((r, result)),
                Err(e) => {
                    let msg = e.downcast_ref::<&str>().map(|s| s.to_string())
                        .or_else(|| e.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "unknown panic".to_string());
                    tlog!("  rank {r} load thread panicked: {msg}");
                    None
                }
            }
        }).collect();
        results.extend(wave_results);
    }
    let mut traces = Vec::new();
    for (rank, result) in results {
        match result {
            Ok(t) => traces.push((rank, t)),
            Err(e) => { tlog!("  rank {rank}: {e}"); }
        }
    }
    if traces.is_empty() {
        let _ = tx.send(Err("all ranks failed to load".into()));
        return;
    }
    let succeeded: std::collections::HashSet<usize> = traces.iter().map(|(r, _)| *r).collect();
    let merged_rank_paths: Vec<(usize, String)> =
        rank_paths.into_iter().filter(|(r, _)| succeeded.contains(r)).collect();
    let mut trace = merge_traces(traces);
    trace.rank_paths = merged_rank_paths;
    tlog!("  merged: {:.3}s ({} events, {} flow_pairs)", t0.elapsed().as_secs_f64(), trace.total_events, trace.flow_pairs.len());
    let _ = tx.send(Ok(trace));
}

/// wasm equivalent of `load_trace_progressive`: takes bytes already read into
/// memory (there's no filesystem path to open on wasm32) instead of a path.
/// No background thread on the single-threaded build — see
/// `state::spawn_load_job` / `Pane::run_pending_parse`.
#[cfg(all(target_arch = "wasm32", feature = "mt"))]
pub fn load_trace_from_bytes_progressive(
    name: &str, raw_input: Vec<u8>, counter: &Arc<AtomicUsize>,
    tx: &std::sync::mpsc::Sender<Result<Trace, String>>,
) {
    let t0 = Instant::now();
    match decode_any_named(name, raw_input, counter, &t0) {
        Ok(trace) => {
            tlog!("  ready: {:.3}s ({} tracks)", t0.elapsed().as_secs_f64(), trace.tracks.len());
            let _ = tx.send(Ok(trace));
        }
        Err(e) => { let _ = tx.send(Err(e)); }
    }
}

/// wasm equivalent of `detect_rank_groups`'s grouping logic (the part after
/// `expand_dirs`, which is fs-based and doesn't apply here — the caller
/// already resolved a dropped folder into a flat file list via the
/// browser's File and Directory Entries API, see main.rs's directory-entry
/// walk). Reuses the same `extract_rank` filename convention natively uses.
#[cfg(target_arch = "wasm32")]
pub fn group_by_rank_bytes(
    files: Vec<(String, Vec<u8>)>,
) -> (Vec<Vec<(usize, String, Vec<u8>)>>, Vec<(String, Vec<u8>)>) {
    let mut groups: HashMap<String, Vec<(usize, String, Vec<u8>)>> = HashMap::new();
    let mut standalone = Vec::new();

    for (name, bytes) in files {
        let fname = name.rsplit('/').next().unwrap_or(&name);
        if let Some((key, rank)) = extract_rank(fname) {
            groups.entry(key).or_default().push((rank, name, bytes));
        } else {
            standalone.push((name, bytes));
        }
    }

    let mut rank_groups: Vec<Vec<(usize, String, Vec<u8>)>> = Vec::new();
    for mut group in groups.into_values() {
        if group.len() > 1 {
            group.sort_by_key(|(rank, _, _)| *rank);
            rank_groups.push(group);
        } else {
            let (_, name, bytes) = group.remove(0);
            standalone.push((name, bytes));
        }
    }

    (rank_groups, standalone)
}

/// wasm equivalent of `load_multi_progressive`: takes each rank's bytes
/// already in memory instead of a path. Each rank decodes to its own trace
/// (own args buffer — see `merge_traces`), then the merge concatenates the
/// buffers as-is.
#[cfg(all(target_arch = "wasm32", feature = "mt"))]
pub fn load_multi_from_bytes_progressive(
    rank_named_bytes: Vec<(usize, String, Vec<u8>)>,
    counter: &Arc<AtomicUsize>,
    tx: &std::sync::mpsc::Sender<Result<Trace, String>>,
) {
    let t0 = Instant::now();

    let results: Vec<(usize, Result<Trace, String>)> = rank_named_bytes.into_par_iter()
        .map(|(rank, name, bytes)| (rank, decode_any_named(&name, bytes, counter, &t0)))
        .collect();

    let mut traces = Vec::new();
    for (rank, result) in results {
        match result {
            Ok(t) => traces.push((rank, t)),
            Err(e) => tlog!("  rank {rank}: {e}"),
        }
    }
    if traces.is_empty() {
        let _ = tx.send(Err("all ranks failed to load".into()));
        return;
    }
    let trace = merge_traces(traces);
    tlog!("  merged: {:.3}s ({} events, {} flow_pairs)", t0.elapsed().as_secs_f64(), trace.total_events, trace.flow_pairs.len());
    let _ = tx.send(Ok(trace));
}

/// Per-call work budget for `WasmParsePlan::step`: loop work units until
/// this much time elapses (at least one unit always runs, so progress is
/// guaranteed), then hand the main thread back to the browser so it can
/// render the splash. 10ms leaves room for that render inside a 60fps
/// frame.
#[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
const STEP_BUDGET_MS: u64 = 10;

/// Decode-phase unit size: events per `step_unit`, small enough that
/// several units fit inside a frame's step budget so the event counter
/// visibly animates at requestAnimationFrame cadence.
#[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
const STEP_EVENTS: usize = 8192;

/// What `WasmParsePlan::prepare` / `WasmMultiParsePlan::prepare` hand back:
/// either a plan to step through, or a finished result (a prepare failure)
/// that the caller sends down the channel as-is.
#[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
pub enum Prepared {
    Plan(SteppedParse),
    Done(Result<Trace, String>),
}

/// `load_trace_from_bytes_progressive` split into resumable phases for the
/// single-threaded wasm build (driven by `Pane::run_pending_parse`). Every
/// phase runs the same helpers the monolithic version runs, in the same
/// order — results are identical, only the pacing differs. `step` returns
/// to the caller every STEP_BUDGET_MS so the browser keeps compositing
/// frames (a frozen splash reads as a hang); without this there is no
/// thread to background the load on. The mt build doesn't need it (the
/// whole job runs on the rayon Worker pool).
///
/// Phases: Decode (`KtzDecoder::step`, 8192 events per unit — the header
/// parse + zstd decompression in `prepare` is the one deliberately-blocking
/// call left, same as it always was) and Assemble (`assemble_track` per
/// track, then `finish_trace`). Multi-rank member plans run the same
/// phases; `merge_traces` then concatenates each rank's args buffer as-is.
#[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
pub struct WasmParsePlan {
    phase: PlanPhase,
    name: String,
    /// Decode-phase state; Some only while `phase == Decode`.
    dec: Option<KtzDecoder>,
    out: KtzTrace,
    n_events: u64,
    /// head++foot text, taken from the decoder when Decode completes
    /// (`finish_trace` scans it for metadata).
    meta: Vec<u8>,
    /// The decoded args buffer, taken out of `out` alongside `meta`.
    args: Vec<u8>,
    scan: TraceScan,
    /// Assemble-phase input/output.
    track_entries: Vec<((u64, u64), Vec<Event>)>,
    keyed_tracks: Vec<((u64, u64), Track)>,
    /// The plan's answer, set when the last phase (or an error) resolves.
    result: Option<Result<Trace, String>>,
    counter: Arc<AtomicUsize>,
    t0: Instant,
}

#[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
#[derive(PartialEq)]
enum PlanPhase { Decode, Assemble, Done }

/// The single- and multi-rank plans behind one `step`/`finish` interface.
#[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
pub enum SteppedParse {
    Single(WasmParsePlan),
    Multi(WasmMultiParsePlan),
}

#[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
impl SteppedParse {
    /// Advance by up to one frame's worth of work. Returns true when the
    /// load is complete and `finish` can run.
    pub fn step(&mut self) -> bool {
        match self {
            SteppedParse::Single(p) => p.step(),
            SteppedParse::Multi(p) => p.step(),
        }
    }

    /// Splash text for the phases the event counter can't describe; None
    /// means "use the event counter text" (the Decode phase).
    pub fn status(&self) -> Option<String> {
        match self {
            SteppedParse::Single(p) => p.status(),
            SteppedParse::Multi(p) => p.status(),
        }
    }

    /// 0..1 progress through the plan (see `WasmParsePlan::progress`), for
    /// the boot overlay's parse slice.
    pub fn progress(&self) -> f32 {
        match self {
            SteppedParse::Single(p) => p.progress(),
            SteppedParse::Multi(p) => p.progress(),
        }
    }

    /// Send the completed result down the channel (merge first for
    /// multi-rank). Only call once `step` reported true.
    pub fn finish(self, tx: &std::sync::mpsc::Sender<Result<Trace, String>>) {
        match self {
            SteppedParse::Single(mut p) => {
                if let Some(r) = p.take_result() {
                    if let Ok(t) = &r {
                        tlog!("  ready: {:.3}s ({} tracks)", p.t0.elapsed().as_secs_f64(), t.tracks.len());
                    }
                    let _ = tx.send(r);
                }
            }
            SteppedParse::Multi(p) => p.finish(tx),
        }
    }
}

#[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
impl WasmParsePlan {
    /// Everything `load_trace_from_bytes_progressive` does up to (but not
    /// including) the decode loop. A non-ktz input or a bad container
    /// resolves entirely in here and comes back as `Prepared::Done`.
    pub fn prepare(name: &str, raw_input: Vec<u8>, counter: &Arc<AtomicUsize>) -> Prepared {
        let t0 = Instant::now();
        if !crate::ktz::is_ktz(&raw_input) {
            // A direct JSON trace has no resumable decoder: its args offsets
            // index the whole decompressed text, so the text must exist in
            // full before any event can be assembled. Decode it in one
            // prepare step (the same blocking shape as a .ktz header/frame
            // decompression) and hand the finished trace straight back.
            if crate::json_trace::is_gzip(&raw_input) || crate::json_trace::looks_like_json(&raw_input) {
                return Prepared::Done(decode_any_named(name, raw_input, counter, &t0));
            }
            return Prepared::Done(Err(format!(
                "{name}: unsupported file type - expected a .ktz trace or a Chrome/kineto JSON trace (.json, .json.gz)"
            )));
        }
        let dec = match KtzDecoder::new(&raw_input) {
            Ok(d) => d,
            Err(e) => return Prepared::Done(Err(format!("{name}: {e}"))),
        };
        let n_events = dec.n_events();
        Prepared::Plan(SteppedParse::Single(WasmParsePlan {
            phase: PlanPhase::Decode,
            name: name.to_string(),
            dec: Some(dec),
            out: KtzTrace::with_events(n_events),
            n_events,
            meta: Vec::new(),
            args: Vec::new(),
            scan: TraceScan::new(),
            track_entries: Vec::new(),
            keyed_tracks: Vec::new(),
            result: None,
            counter: counter.clone(),
            t0,
        }))
    }

    /// A plan that is born finished — a direct JSON rank, which `prepare`
    /// decodes synchronously and has nothing left to step.
    fn from_result(result: Result<Trace, String>, counter: &Arc<AtomicUsize>, t0: Instant) -> Self {
        Self {
            phase: PlanPhase::Done,
            name: String::new(),
            dec: None,
            out: KtzTrace::with_events(0),
            n_events: 0,
            meta: Vec::new(),
            args: Vec::new(),
            scan: TraceScan::new(),
            track_entries: Vec::new(),
            keyed_tracks: Vec::new(),
            result: Some(result),
            counter: counter.clone(),
            t0,
        }
    }

    pub fn is_done(&self) -> bool {
        self.phase == PlanPhase::Done
    }

    /// Advance by up to one frame's worth of work: loop work units until
    /// STEP_BUDGET_MS elapses (a single unit may overshoot — progress is
    /// guaranteed regardless). Returns true when the plan is complete and
    /// `take_result` holds the answer.
    pub fn step(&mut self) -> bool {
        let start = Instant::now();
        loop {
            if !self.step_unit() {
                return true;
            }
            if start.elapsed().as_millis() as u64 >= STEP_BUDGET_MS {
                return false;
            }
        }
    }

    /// One unit of the current phase. Returns false only when Done.
    /// pub(crate) so the multi-rank plan can interleave rank units inside
    /// its own budget.
    pub(crate) fn step_unit(&mut self) -> bool {
        match self.phase {
            PlanPhase::Done => false,
            PlanPhase::Decode => self.step_decode(),
            PlanPhase::Assemble => self.step_assemble(),
        }
    }

    /// Decode STEP_EVENTS more events. On completion, hand the head/foot
    /// text and args buffer over to the plan, fold the decoded trace into
    /// the scan, and move on to Assemble.
    fn step_decode(&mut self) -> bool {
        let mut dec = match self.dec.take() {
            Some(d) => d,
            None => { self.phase = PlanPhase::Assemble; return true; }
        };
        let done = match dec.step(&mut self.out, STEP_EVENTS, &self.counter) {
            Ok(d) => d,
            Err(e) => {
                self.result = Some(Err(format!("{}: {e}", self.name)));
                self.phase = PlanPhase::Done;
                return true;
            }
        };
        if !done {
            self.dec = Some(dec);
            return true;
        }
        self.meta = dec.into_head_foot();
        let mut ktz = std::mem::replace(&mut self.out, KtzTrace::with_events(0));
        self.args = std::mem::take(&mut ktz.args);
        self.scan.merge_chunk(ktz);
        self.track_entries = self.scan.take_track_entries();
        self.phase = PlanPhase::Assemble;
        true
    }

    /// Assemble one track per unit. The unit after the last runs
    /// finish_trace (metadata scans, flow pairing, stats — the glue that
    /// needs every track present).
    fn step_assemble(&mut self) -> bool {
        if let Some(((pid, tid), evs)) = self.track_entries.pop() {
            let track = assemble_track(
                pid, tid, evs, self.scan.min_ts, &self.scan.cats, &self.scan.thread_names,
            );
            self.keyed_tracks.push(((pid, tid), track));
            return true;
        }
        let scan = std::mem::replace(&mut self.scan, TraceScan::new());
        let args = std::mem::take(&mut self.args);
        let keyed_tracks = std::mem::take(&mut self.keyed_tracks);
        let meta = std::mem::take(&mut self.meta);
        self.result = Some(finish_trace(
            Some(&meta), Arc::new(ArgsBuf(args)), scan, keyed_tracks, &self.t0,
        ));
        self.phase = PlanPhase::Done;
        true
    }

    /// The plan's answer once `step` reports done.
    pub fn take_result(&mut self) -> Option<Result<Trace, String>> {
        self.result.take()
    }

    /// Splash text for the phases the event counter can't describe; None
    /// during Decode (the counter covers it) and Done.
    pub fn status(&self) -> Option<String> {
        match self.phase {
            PlanPhase::Decode => None,
            PlanPhase::Assemble => Some(format!(
                "Building tracks... {}/{}",
                self.keyed_tracks.len(),
                self.keyed_tracks.len() + self.track_entries.len(),
            )),
            PlanPhase::Done => None,
        }
    }

    /// Coarse 0..1 progress through the plan's phases, for the boot
    /// overlay's parse slice (update_boot_overlay in main.rs). The
    /// within-phase position is real (events decoded, tracks built); the
    /// phase boundaries are fixed weights, so a phase that runs heavier
    /// than its weight only makes the bar slow, never wrong.
    pub fn progress(&self) -> f32 {
        let frac = |a: u64, b: u64| if b == 0 { 1.0 } else { a.min(b) as f32 / b as f32 };
        match self.phase {
            PlanPhase::Decode => 0.65 * frac(self.out.total_events as u64, self.n_events),
            PlanPhase::Assemble => 0.65 + 0.35 * frac(
                self.keyed_tracks.len() as u64,
                (self.keyed_tracks.len() + self.track_entries.len()) as u64,
            ),
            PlanPhase::Done => 1.0,
        }
    }
}

/// Multi-rank stepped parse: per-rank `WasmParsePlan`s round-robin stepped
/// so the shared counter grows across all ranks instead of one rank
/// finishing before the next starts. Mirrors `load_multi_from_bytes_progressive`
/// phase for phase.
#[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
pub struct WasmMultiParsePlan {
    plans: Vec<(usize, WasmParsePlan)>,
    next: usize,
    t0: Instant,
}

#[cfg(all(target_arch = "wasm32", not(feature = "mt")))]
impl WasmMultiParsePlan {
    pub fn prepare(
        rank_named_bytes: Vec<(usize, String, Vec<u8>)>, counter: &Arc<AtomicUsize>,
    ) -> Prepared {
        let t0 = Instant::now();
        let mut plans = Vec::new();
        for (rank, name, bytes) in rank_named_bytes {
            match WasmParsePlan::prepare(&name, bytes, counter) {
                Prepared::Plan(SteppedParse::Single(p)) => plans.push((rank, p)),
                Prepared::Plan(SteppedParse::Multi(_)) => unreachable!("single prepare cannot return a multi plan"),
                Prepared::Done(Ok(t)) => plans.push((rank, WasmParsePlan::from_result(Ok(t), counter, t0))),
                Prepared::Done(Err(e)) => tlog!("  rank {rank}: {e}"),
            }
        }
        if plans.is_empty() {
            return Prepared::Done(Err("all ranks failed to load".into()));
        }
        Prepared::Plan(SteppedParse::Multi(WasmMultiParsePlan { plans, next: 0, t0 }))
    }

    /// Step work units from the ranks round-robin, one frame's budget
    /// shared across ranks (so the counter grows across all ranks instead
    /// of one rank finishing before the next starts). Returns true when
    /// every rank is fully done.
    pub fn step(&mut self) -> bool {
        let start = Instant::now();
        loop {
            if self.plans.iter().all(|(_, p)| p.is_done()) {
                return true;
            }
            let n = self.plans.len();
            for k in 0..n {
                let i = (self.next + k) % n;
                if !self.plans[i].1.is_done() {
                    self.next = i + 1;
                    self.plans[i].1.step_unit();
                    break;
                }
            }
            if start.elapsed().as_millis() as u64 >= STEP_BUDGET_MS {
                return false;
            }
        }
    }

    /// Splash text: the next rank's phase, or the merge once all are done.
    pub fn status(&self) -> Option<String> {
        for (_, p) in &self.plans {
            if !p.is_done() {
                return p.status();
            }
        }
        Some("Merging ranks...".to_string())
    }

    /// Mean of the member plans' progress, so the bar advances evenly
    /// across ranks instead of rank by rank. See `WasmParsePlan::progress`.
    pub fn progress(&self) -> f32 {
        let n = self.plans.len();
        if n == 0 { 1.0 } else { self.plans.iter().map(|(_, p)| p.progress()).sum::<f32>() / n as f32 }
    }

    /// Collect each rank's result, merge, and send. `merge_traces` is the
    /// one deliberately-blocking call left: it has no counter to animate,
    /// so slicing it buys nothing.
    fn finish(self, tx: &std::sync::mpsc::Sender<Result<Trace, String>>) {
        let mut traces = Vec::new();
        for (rank, mut plan) in self.plans {
            match plan.take_result() {
                Some(Ok(t)) => traces.push((rank, t)),
                Some(Err(e)) => tlog!("  rank {rank}: {e}"),
                None => tlog!("  rank {rank}: no result"),
            }
        }
        if traces.is_empty() {
            let _ = tx.send(Err("all ranks failed to load".into()));
            return;
        }
        let trace = merge_traces(traces);
        tlog!("  merged: {:.3}s ({} events, {} flow_pairs)",
            self.t0.elapsed().as_secs_f64(), trace.total_events, trace.flow_pairs.len());
        let _ = tx.send(Ok(trace));
    }
}
