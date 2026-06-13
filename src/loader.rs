use crate::parse::*;
use crate::types::*;
use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::time::Instant;

pub enum RawData {
    Vec(Vec<u8>),
    Mmap(memmap2::Mmap),
}

impl std::ops::Deref for RawData {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            RawData::Vec(v) => v,
            RawData::Mmap(m) => m,
        }
    }
}

struct ChunkState {
    names: Vec<String>,
    name_idx: HashMap<u64, u32>,
    cats: Vec<String>,
    cat_idx: HashMap<u64, u32>,
    arg_strs: Vec<String>,
    arg_str_idx: HashMap<u64, u32>,
    arg_pairs: Vec<[u32; 2]>,
    arg_dedup: HashMap<u64, (u32, u16)>,
    events: Vec<(u64, u64, Event)>,
    thread_names: HashMap<(u64, u64), String>,
    min_ts: f64,
    max_ts: f64,
    total_events: usize,
    scan_count: u64,
}

impl ChunkState {
    fn new() -> Self {
        let mut name_idx = HashMap::new();
        name_idx.insert(0, 0);
        let mut cat_idx = HashMap::new();
        cat_idx.insert(0, 0);
        let mut arg_str_idx = HashMap::new();
        arg_str_idx.insert(0, 0);
        Self {
            names: vec![String::new()],
            name_idx,
            cats: vec![String::new()],
            cat_idx,
            arg_strs: vec![String::new()],
            arg_str_idx,
            arg_pairs: Vec::new(),
            arg_dedup: HashMap::new(),
            events: Vec::new(),
            thread_names: HashMap::new(),
            min_ts: f64::MAX,
            max_ts: f64::MIN,
            total_events: 0,
            scan_count: 0,
        }
    }
}

fn parse_chunk(raw: &[u8], start: usize, chunk_end: usize, state: &mut ChunkState) {
    let mut pos = start;
    loop {
        pos = skip_ws_comma(raw, pos);
        if pos >= chunk_end || pos >= raw.len() || raw[pos] == b']' { break; }
        state.scan_count += 1;
        if raw[pos] != b'{' {
            pos = skip_value(raw, pos);
            continue;
        }

        pos += 1;
        let mut depth = 1u32;
        let mut ph: u8 = 0;
        let mut ts: f64 = 0.0;
        let mut dur: f64 = 0.0;
        let mut has_dur = false;
        let mut tid: u64 = 0;
        let mut pid: u64 = 0;
        let mut name: u32 = 0;
        let mut cat: u32 = 0;
        let mut args_start: u32 = 0;
        let mut args_count: u16 = 0;

        while depth > 0 && pos < raw.len() {
            match raw[pos] {
                b'"' if depth == 1 => {
                    let ks = pos + 1;
                    pos = skip_string(raw, pos);
                    let ke = pos - 1;
                    pos = skip_ws(raw, pos);
                    if pos < raw.len() && raw[pos] == b':' { pos += 1; }
                    pos = skip_ws(raw, pos);

                    let klen = ke - ks;
                    if klen >= 2 && klen <= 4 && pos < raw.len() {
                        match (klen, raw[ks]) {
                            (2, b'p') if raw[ks + 1] == b'h' => {
                                if raw[pos] == b'"' {
                                    ph = raw[pos + 1];
                                    pos = skip_string(raw, pos);
                                } else { pos = skip_value(raw, pos); }
                            }
                            (2, b't') if raw[ks + 1] == b's' => {
                                let s = pos;
                                pos = skip_number(raw, pos);
                                ts = parse_f64(&raw[s..pos]);
                            }
                            (3, b'd') if raw[ks + 1] == b'u' => {
                                let s = pos;
                                pos = skip_number(raw, pos);
                                dur = parse_f64(&raw[s..pos]);
                                has_dur = true;
                            }
                            (3, b't') if raw[ks + 1] == b'i' => {
                                if raw[pos] == b'"' {
                                    let s = pos + 1;
                                    pos = skip_string(raw, pos);
                                    tid = fnv1a(&raw[s..pos - 1]);
                                } else {
                                    let s = pos;
                                    pos = skip_number(raw, pos);
                                    tid = parse_f64(&raw[s..pos]) as u64;
                                }
                            }
                            (3, b'p') if raw[ks + 1] == b'i' => {
                                if raw[pos] == b'"' {
                                    let s = pos + 1;
                                    pos = skip_string(raw, pos);
                                    pid = fnv1a(&raw[s..pos - 1]);
                                } else {
                                    let s = pos;
                                    pos = skip_number(raw, pos);
                                    pid = parse_f64(&raw[s..pos]) as u64;
                                }
                            }
                            (4, b'n') if raw[ks + 1] == b'a' && raw[ks + 2] == b'm' => {
                                if raw[pos] == b'"' {
                                    let s = pos + 1;
                                    pos = skip_string(raw, pos);
                                    name = intern(&raw[s..pos - 1], &mut state.names, &mut state.name_idx);
                                } else { pos = skip_value(raw, pos); }
                            }
                            (4, b'a') if raw[ks + 1] == b'r' => {
                                let s = pos;
                                let (end, hash) = skip_value_hashed(raw, pos);
                                pos = end;
                                if let Some(&(st, ct)) = state.arg_dedup.get(&hash) {
                                    args_start = st;
                                    args_count = ct;
                                } else {
                                    let pair_start = state.arg_pairs.len() as u32;
                                    parse_args_flat(&raw[s..pos], &mut state.arg_strs, &mut state.arg_str_idx, &mut state.arg_pairs);
                                    let ct = (state.arg_pairs.len() as u32 - pair_start) as u16;
                                    state.arg_dedup.insert(hash, (pair_start, ct));
                                    args_start = pair_start;
                                    args_count = ct;
                                }
                            }
                            (3, b'c') if raw[ks + 1] == b'a' => {
                                if raw[pos] == b'"' {
                                    let s = pos + 1;
                                    pos = skip_string(raw, pos);
                                    cat = intern(&raw[s..pos - 1], &mut state.cats, &mut state.cat_idx);
                                } else { pos = skip_value(raw, pos); }
                            }
                            _ => { pos = skip_value(raw, pos); }
                        }
                    } else { pos = skip_value(raw, pos); }
                }
                b'"' => pos = skip_string(raw, pos),
                b'{' | b'[' => { depth += 1; pos += 1; }
                b'}' | b']' => { depth -= 1; pos += 1; }
                _ => pos += 1,
            }
        }

        if ph == b'X' && has_dur {
            state.min_ts = state.min_ts.min(ts);
            state.max_ts = state.max_ts.max(ts + dur);
            state.events.push((pid, tid, Event {
                ts, dur, name, cat, args_start, args_count, depth: 0,
            }));
            state.total_events += 1;
        } else if ph == b'M' {
            let name_str = &state.names[name as usize];
            if name_str == "thread_name" {
                for i in args_start as usize..(args_start as usize + args_count as usize) {
                    let [k, v] = state.arg_pairs[i];
                    if state.arg_strs[k as usize] == "name" {
                        state.thread_names.insert((pid, tid), state.arg_strs[v as usize].clone());
                        break;
                    }
                }
            }
        }
    }
}

fn merge_intern_table(
    global: &mut Vec<String>,
    global_idx: &mut HashMap<u64, u32>,
    local: &[String],
    local_idx: &HashMap<u64, u32>,
) -> Vec<u32> {
    let mut remap = vec![0u32; local.len()];
    for (&hash, &local_i) in local_idx {
        if let Some(&global_i) = global_idx.get(&hash) {
            remap[local_i as usize] = global_i;
        } else {
            let global_i = global.len() as u32;
            global.push(local[local_i as usize].clone());
            global_idx.insert(hash, global_i);
            remap[local_i as usize] = global_i;
        }
    }
    remap
}

pub fn load_trace(path: &str) -> Result<Trace, String> {
    let t0 = Instant::now();
    let raw = read_bytes(path)?;
    eprintln!("  read: {:.2}s ({}MB)", t0.elapsed().as_secs_f64(), raw.len() / 1024 / 1024);

    let te = find_key(&raw, b"traceEvents").ok_or("no traceEvents found")?;
    let mut pos = te + "\"traceEvents\"".len();
    pos = skip_ws(&raw, pos);
    if pos < raw.len() && raw[pos] == b':' { pos += 1; }
    pos = skip_ws(&raw, pos);
    if pos >= raw.len() || raw[pos] != b'[' {
        return Err("malformed traceEvents".into());
    }
    let array_start = pos + 1;

    let n_threads = if raw.len() < 10 * 1024 * 1024 {
        1
    } else {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            .clamp(2, 8)
    };

    let t1 = Instant::now();
    let split_points = find_split_points(&raw, array_start, n_threads);
    let n_chunks = split_points.len() - 1;
    if n_chunks > 1 {
        eprintln!("  split: {} chunks in {:.3}s", n_chunks, t1.elapsed().as_secs_f64());
    }

    let chunks: Vec<ChunkState> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..n_chunks).map(|i| {
            let start = split_points[i];
            let end = split_points[i + 1];
            let raw_ref = &raw;
            s.spawn(move || {
                let mut state = ChunkState::new();
                parse_chunk(raw_ref, start, end, &mut state);
                state
            })
        }).collect();
        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });

    let mut names: Vec<String> = vec![String::new()];
    let mut name_idx: HashMap<u64, u32> = HashMap::new();
    name_idx.insert(0, 0);
    let mut cats: Vec<String> = vec![String::new()];
    let mut cat_idx: HashMap<u64, u32> = HashMap::new();
    cat_idx.insert(0, 0);
    let mut arg_strs: Vec<String> = vec![String::new()];
    let mut arg_str_idx: HashMap<u64, u32> = HashMap::new();
    arg_str_idx.insert(0, 0);
    let mut arg_pairs: Vec<[u32; 2]> = Vec::new();

    let mut track_map: HashMap<(u64, u64), Vec<Event>> = HashMap::new();
    let mut thread_names: HashMap<(u64, u64), String> = HashMap::new();
    let mut min_ts = f64::MAX;
    let mut max_ts = f64::MIN;
    let mut total_events: usize = 0;
    let mut scan_count: u64 = 0;

    for chunk in chunks {
        let name_remap = merge_intern_table(&mut names, &mut name_idx, &chunk.names, &chunk.name_idx);
        let cat_remap = merge_intern_table(&mut cats, &mut cat_idx, &chunk.cats, &chunk.cat_idx);
        let arg_str_remap = merge_intern_table(&mut arg_strs, &mut arg_str_idx, &chunk.arg_strs, &chunk.arg_str_idx);

        let pair_offset = arg_pairs.len() as u32;
        for &[k, v] in &chunk.arg_pairs {
            arg_pairs.push([arg_str_remap[k as usize], arg_str_remap[v as usize]]);
        }

        for (pid, tid, mut ev) in chunk.events {
            ev.name = name_remap[ev.name as usize];
            ev.cat = cat_remap[ev.cat as usize];
            ev.args_start += pair_offset;
            track_map.entry((pid, tid)).or_default().push(ev);
        }

        for (key, value) in chunk.thread_names {
            thread_names.entry(key).or_insert(value);
        }

        min_ts = min_ts.min(chunk.min_ts);
        max_ts = max_ts.max(chunk.max_ts);
        total_events += chunk.total_events;
        scan_count += chunk.scan_count;
    }

    let mut device = String::new();
    if let Some(dp) = find_key(&raw, b"deviceProperties") {
        let mut p = dp + "\"deviceProperties\"".len();
        p = skip_ws(&raw, p);
        if p < raw.len() && raw[p] == b':' { p += 1; }
        p = skip_ws(&raw, p);
        if let Some(np) = find_key(&raw[p..], b"name") {
            let mut q = p + np + "\"name\"".len();
            q = skip_ws(&raw, q);
            if q < raw.len() && raw[q] == b':' { q += 1; }
            q = skip_ws(&raw, q);
            if q < raw.len() && raw[q] == b'"' {
                let end = skip_string(&raw, q);
                device = json_unescape(std::str::from_utf8(&raw[q + 1..end - 1]).unwrap_or(""));
            }
        }
    }

    eprintln!("  scan: {:.2}s ({} objects, {} events, {} names, {}x parallel)",
        t0.elapsed().as_secs_f64(), scan_count, total_events, names.len(), n_chunks);
    drop(raw);
    drop(name_idx);
    drop(cat_idx);
    drop(arg_str_idx);

    if min_ts == f64::MAX {
        return Err("no duration events found".into());
    }

    let t2 = Instant::now();
    let mut tracks: Vec<Track> = Vec::new();

    std::thread::scope(|s| {
        let cat_ref = &cats;
        let tn_ref = &thread_names;
        let handles: Vec<_> = track_map
            .into_iter()
            .map(|((pid, tid), mut evs)| {
                s.spawn(move || {
                    for ev in evs.iter_mut() { ev.ts -= min_ts; }
                    evs.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap());
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
                        let c = &cat_ref[e.cat as usize];
                        c == "kernel" || c.starts_with("gpu_")
                    }).count();
                    let gpu = gpu_count > evs.len() / 2;
                    let label = tn_ref.get(&(pid, tid)).cloned().unwrap_or_else(|| {
                        if gpu { format!("GPU {tid}") } else { format!("Thread {tid}") }
                    });
                    evs.shrink_to_fit();
                    Track { label, gpu, events: evs, max_depth, prefix_max_dur }
                })
            })
            .collect();
        tracks = handles.into_iter().map(|h| h.join().unwrap()).collect();
    });

    tracks.sort_by(|a, b| b.gpu.cmp(&a.gpu).then_with(|| b.events.len().cmp(&a.events.len())));
    max_ts -= min_ts;

    let mut dur_map: HashMap<u32, Vec<f64>> = HashMap::new();
    for t in &tracks {
        for ev in &t.events {
            dur_map.entry(ev.name).or_default().push(ev.dur);
        }
    }
    let mut stats: Vec<KernelStats> = dur_map.into_iter().map(|(name, mut durs)| {
        let count = durs.len() as u32;
        let total_dur: f64 = durs.iter().sum();
        let max_dur = durs.iter().copied().fold(0.0f64, f64::max);
        durs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let n = durs.len();
        let median_dur = if n % 2 == 1 { durs[n / 2] } else { (durs[n / 2 - 1] + durs[n / 2]) / 2.0 };
        KernelStats { name, count, total_dur, median_dur, max_dur }
    }).collect();
    stats.sort_by(|a, b| b.total_dur.partial_cmp(&a.total_dur).unwrap());

    names.shrink_to_fit();
    cats.shrink_to_fit();
    arg_strs.shrink_to_fit();
    arg_pairs.shrink_to_fit();

    let event_bytes: usize = tracks.iter().map(|t| t.events.len() * std::mem::size_of::<Event>()).sum();
    let str_bytes: usize = names.iter().chain(cats.iter()).chain(arg_strs.iter()).map(|s| s.len() + std::mem::size_of::<String>()).sum();
    let pair_bytes = arg_pairs.len() * std::mem::size_of::<[u32; 2]>();
    eprintln!("  sort+lanes: {:.2}s ({} tracks)", t2.elapsed().as_secs_f64(), tracks.len());
    eprintln!("  memory: events={}MB strings={}MB args={}MB total={}MB",
        event_bytes / 1024 / 1024, str_bytes / 1024 / 1024, pair_bytes / 1024 / 1024,
        (event_bytes + str_bytes + pair_bytes) / 1024 / 1024);
    eprintln!("  total: {:.2}s", t0.elapsed().as_secs_f64());
    Ok(Trace { tracks, names, cats, arg_strs, arg_pairs, stats, max_ts, min_ts, total_events, device })
}

fn expand_dirs(paths: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for path in paths {
        let p = std::path::Path::new(path);
        if p.is_dir() {
            if let Ok(entries) = std::fs::read_dir(p) {
                for entry in entries.flatten() {
                    let ep = entry.path();
                    if ep.is_file() {
                        if let Some(name) = ep.file_name().and_then(|n| n.to_str()) {
                            if name.ends_with(".json") || name.ends_with(".json.gz")
                                || name.ends_with(".tar.gz") || name.ends_with(".tgz") {
                                out.push(ep.to_string_lossy().into());
                            }
                        }
                    }
                }
            }
        } else {
            out.push(path.clone());
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

pub fn detect_rank_groups(paths: &[String]) -> (Vec<Vec<(usize, String)>>, Vec<String>) {
    let expanded = expand_dirs(paths);
    let mut groups: HashMap<String, Vec<(usize, String)>> = HashMap::new();
    let mut standalone = Vec::new();

    for path in &expanded {
        let fname = std::path::Path::new(path)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or(path);
        if let Some((key, rank)) = extract_rank(fname) {
            groups.entry(key).or_default().push((rank, path.clone()));
        } else {
            standalone.push(path.clone());
        }
    }

    let mut rank_groups: Vec<Vec<(usize, String)>> = groups.into_values()
        .filter(|g| g.len() > 1)
        .collect();
    for group in &mut rank_groups {
        group.sort_by_key(|(rank, _)| *rank);
    }
    for group in rank_groups.iter() {
        if group.len() == 1 {
            standalone.push(group[0].1.clone());
        }
    }
    rank_groups.retain(|g| g.len() > 1);

    (rank_groups, standalone)
}

pub fn merge_traces(traces: Vec<(usize, Trace)>) -> Trace {
    let global_min = traces.iter().map(|(_, t)| t.min_ts).fold(f64::MAX, f64::min);

    let mut names: Vec<String> = vec![String::new()];
    let mut name_idx: HashMap<u64, u32> = HashMap::new();
    name_idx.insert(0, 0);
    let mut cats: Vec<String> = vec![String::new()];
    let mut cat_idx: HashMap<u64, u32> = HashMap::new();
    cat_idx.insert(0, 0);
    let mut arg_strs: Vec<String> = vec![String::new()];
    let mut arg_str_idx: HashMap<u64, u32> = HashMap::new();
    arg_str_idx.insert(0, 0);
    let mut arg_pairs: Vec<[u32; 2]> = Vec::new();
    let mut all_tracks: Vec<Track> = Vec::new();
    let mut total_events = 0;
    let mut max_ts: f64 = 0.0;
    let mut device = String::new();

    for (rank, trace) in traces {
        let time_offset = trace.min_ts - global_min;

        let mut local_name_idx: HashMap<u64, u32> = HashMap::new();
        for (i, s) in trace.names.iter().enumerate() {
            local_name_idx.insert(crate::parse::fnv1a(s.as_bytes()), i as u32);
        }
        let name_remap = merge_intern_table(&mut names, &mut name_idx, &trace.names, &local_name_idx);

        let mut local_cat_idx: HashMap<u64, u32> = HashMap::new();
        for (i, s) in trace.cats.iter().enumerate() {
            local_cat_idx.insert(crate::parse::fnv1a(s.as_bytes()), i as u32);
        }
        let cat_remap = merge_intern_table(&mut cats, &mut cat_idx, &trace.cats, &local_cat_idx);

        let mut local_arg_str_idx: HashMap<u64, u32> = HashMap::new();
        for (i, s) in trace.arg_strs.iter().enumerate() {
            local_arg_str_idx.insert(crate::parse::fnv1a(s.as_bytes()), i as u32);
        }
        let arg_str_remap = merge_intern_table(&mut arg_strs, &mut arg_str_idx, &trace.arg_strs, &local_arg_str_idx);

        let pair_offset = arg_pairs.len() as u32;
        for &[k, v] in &trace.arg_pairs {
            arg_pairs.push([arg_str_remap[k as usize], arg_str_remap[v as usize]]);
        }

        for mut track in trace.tracks {
            track.label = format!("[rank {}] {}", rank, track.label);
            for ev in &mut track.events {
                ev.ts += time_offset;
                ev.name = name_remap[ev.name as usize];
                ev.cat = cat_remap[ev.cat as usize];
                ev.args_start += pair_offset;
            }
            total_events += track.events.len();
            for ev in &track.events {
                max_ts = max_ts.max(ev.ts + ev.dur);
            }
            all_tracks.push(track);
        }

        if device.is_empty() && !trace.device.is_empty() {
            device = trace.device;
        }
    }

    all_tracks.sort_by(|a, b| {
        a.label.cmp(&b.label)
    });

    let mut dur_map: HashMap<u32, Vec<f64>> = HashMap::new();
    for t in &all_tracks {
        for ev in &t.events {
            dur_map.entry(ev.name).or_default().push(ev.dur);
        }
    }
    let mut stats: Vec<KernelStats> = dur_map.into_iter().map(|(name, mut durs)| {
        let count = durs.len() as u32;
        let total_dur: f64 = durs.iter().sum();
        let max_dur = durs.iter().copied().fold(0.0f64, f64::max);
        durs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let n = durs.len();
        let median_dur = if n % 2 == 1 { durs[n / 2] } else { (durs[n / 2 - 1] + durs[n / 2]) / 2.0 };
        KernelStats { name, count, total_dur, median_dur, max_dur }
    }).collect();
    stats.sort_by(|a, b| b.total_dur.partial_cmp(&a.total_dur).unwrap());

    names.shrink_to_fit();
    cats.shrink_to_fit();
    arg_strs.shrink_to_fit();
    arg_pairs.shrink_to_fit();

    Trace {
        tracks: all_tracks, names, cats, arg_strs, arg_pairs, stats,
        max_ts, min_ts: global_min, total_events, device,
    }
}

fn read_gz_tolerant<R: std::io::Read>(reader: R) -> Result<Vec<u8>, String> {
    let mut decoder = flate2::read::GzDecoder::new(reader);
    let mut buf = Vec::new();
    let mut tmp = [0u8; 64 * 1024];
    loop {
        match decoder.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
            Err(e) if buf.is_empty() => return Err(format!("decompress: {e}")),
            Err(e) => {
                eprintln!("  truncated gz stream after {} bytes: {e}", buf.len());
                break;
            }
        }
    }
    Ok(buf)
}

pub fn read_bytes(path: &str) -> Result<RawData, String> {
    if path.ends_with(".tar.gz") || path.ends_with(".tgz") {
        let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let gz = flate2::read::GzDecoder::new(BufReader::new(file));
        let mut archive = tar::Archive::new(gz);
        let entries = archive.entries().map_err(|e| format!("tar {path}: {e}"))?;
        for entry in entries {
            let mut entry = entry.map_err(|e| format!("tar entry: {e}"))?;
            let name = entry.path().map_err(|e| format!("tar path: {e}"))?.to_string_lossy().to_string();
            if name.ends_with(".json.gz") {
                let buf = read_gz_tolerant(&mut entry)
                    .map_err(|e| format!("{name}: {e}"))?;
                eprintln!("  extracted: {name}");
                return Ok(RawData::Vec(buf));
            } else if name.ends_with(".json") {
                let mut buf = Vec::new();
                match entry.read_to_end(&mut buf) {
                    Ok(_) => {}
                    Err(e) if buf.is_empty() => return Err(format!("read {name}: {e}")),
                    Err(e) => eprintln!("  truncated tar entry after {} bytes: {e}", buf.len()),
                }
                eprintln!("  extracted: {name}");
                return Ok(RawData::Vec(buf));
            }
        }
        Err(format!("no .json or .json.gz file found in {path}"))
    } else if path.ends_with(".gz") {
        let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let buf = read_gz_tolerant(BufReader::new(file))
            .map_err(|e| format!("{path}: {e}"))?;
        Ok(RawData::Vec(buf))
    } else {
        let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("mmap {path}: {e}"))?;
        mmap.advise(memmap2::Advice::Sequential).ok();
        Ok(RawData::Mmap(mmap))
    }
}
