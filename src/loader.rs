use crate::parse::*;
use crate::types::*;
use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
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

impl RawData {
    fn into_vec(self) -> Vec<u8> {
        match self {
            RawData::Vec(v) => v,
            RawData::Mmap(m) => m.to_vec(),
        }
    }
}

struct ChunkState {
    names: Vec<String>,
    name_idx: FnvMap<u32>,
    cats: Vec<String>,
    cat_idx: FnvMap<u32>,
    events: Vec<(u64, u64, Event)>,
    thread_names: HashMap<(u64, u64), String>,
    min_ts: f64,
    max_ts: f64,
    total_events: usize,
    scan_count: u64,
}

impl ChunkState {
    fn new() -> Self {
        let mut name_idx = FnvMap::default();
        name_idx.insert(0, 0);
        let mut cat_idx = FnvMap::default();
        cat_idx.insert(0, 0);
        Self {
            names: vec![String::new()],
            name_idx,
            cats: vec![String::new()],
            cat_idx,
            events: Vec::new(),
            thread_names: HashMap::new(),
            min_ts: f64::MAX,
            max_ts: f64::MIN,
            total_events: 0,
            scan_count: 0,
        }
    }
}

fn parse_chunk(raw: &[u8], start: usize, chunk_end: usize, state: &mut ChunkState, counter: &AtomicUsize) {
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
        let mut ph: u8 = 0;
        let mut ts: f64 = 0.0;
        let mut dur: f64 = 0.0;
        let mut has_dur = false;
        let mut tid: u64 = 0;
        let mut pid: u64 = 0;
        let mut name: u32 = 0;
        let mut cat: u32 = 0;
        let mut args_off: u32 = 0;
        let mut args_len: u16 = 0;

        loop {
            pos = skip_ws_comma(raw, pos);
            if pos >= raw.len() || raw[pos] == b'}' { pos += 1; break; }
            if raw[pos] != b'"' { pos = skip_value(raw, pos); continue; }

            macro_rules! skip_colon {
                () => {{
                    pos = skip_ws(raw, pos);
                    if pos < raw.len() && raw[pos] == b':' { pos += 1; }
                    pos = skip_ws(raw, pos);
                }};
            }

            if pos + 4 < raw.len() && raw[pos + 3] == b'"' {
                let k = [raw[pos + 1], raw[pos + 2]];
                pos += 4;
                skip_colon!();
                match k {
                    [b'p', b'h'] => {
                        if pos < raw.len() && raw[pos] == b'"' {
                            ph = raw[pos + 1];
                            pos = skip_string(raw, pos);
                            if ph != b'X' && ph != b'M' {
                                pos = skip_to_closing(raw, pos);
                                break;
                            }
                        } else { pos = skip_value(raw, pos); }
                    }
                    [b't', b's'] => {
                        let s = pos;
                        pos = skip_number(raw, pos);
                        ts = parse_f64(&raw[s..pos]);
                    }
                    _ => { pos = skip_value(raw, pos); }
                }
            } else if pos + 5 < raw.len() && raw[pos + 4] == b'"' {
                let k = [raw[pos + 1], raw[pos + 2]];
                pos += 5;
                skip_colon!();
                match k {
                    [b'd', b'u'] => {
                        let s = pos;
                        pos = skip_number(raw, pos);
                        dur = parse_f64(&raw[s..pos]);
                        has_dur = true;
                    }
                    [b't', b'i'] => {
                        if pos < raw.len() && raw[pos] == b'"' {
                            let s = pos + 1;
                            pos = skip_string(raw, pos);
                            tid = fnv1a(&raw[s..pos - 1]);
                        } else {
                            let s = pos;
                            pos = skip_number(raw, pos);
                            tid = parse_f64(&raw[s..pos]) as u64;
                        }
                    }
                    [b'p', b'i'] => {
                        if pos < raw.len() && raw[pos] == b'"' {
                            let s = pos + 1;
                            pos = skip_string(raw, pos);
                            pid = fnv1a(&raw[s..pos - 1]);
                        } else {
                            let s = pos;
                            pos = skip_number(raw, pos);
                            pid = parse_f64(&raw[s..pos]) as u64;
                        }
                    }
                    [b'c', b'a'] => {
                        if pos < raw.len() && raw[pos] == b'"' {
                            let s = pos + 1;
                            pos = skip_string(raw, pos);
                            cat = intern(&raw[s..pos - 1], &mut state.cats, &mut state.cat_idx);
                        } else { pos = skip_value(raw, pos); }
                    }
                    _ => { pos = skip_value(raw, pos); }
                }
            } else if pos + 6 < raw.len() && raw[pos + 5] == b'"' {
                let k = [raw[pos + 1], raw[pos + 2]];
                pos += 6;
                skip_colon!();
                match k {
                    [b'n', b'a'] => {
                        if pos < raw.len() && raw[pos] == b'"' {
                            let s = pos + 1;
                            pos = skip_string(raw, pos);
                            name = intern(&raw[s..pos - 1], &mut state.names, &mut state.name_idx);
                        } else { pos = skip_value(raw, pos); }
                    }
                    [b'a', b'r'] => {
                        args_off = pos as u32;
                        pos = skip_value(raw, pos);
                        let len = pos - args_off as usize;
                        args_len = if len <= u16::MAX as usize { len as u16 } else { 0 };
                    }
                    _ => { pos = skip_value(raw, pos); }
                }
            } else {
                pos = skip_string(raw, pos);
                skip_colon!();
                pos = skip_value(raw, pos);
            }
        }

        if ph == b'X' && has_dur {
            state.min_ts = state.min_ts.min(ts);
            state.max_ts = state.max_ts.max(ts + dur);
            state.events.push((pid, tid, Event {
                ts, dur, name, cat, args_off, args_len, depth: 0,
            }));
            state.total_events += 1;
            counter.fetch_add(1, Ordering::Relaxed);
        } else if ph == b'M' {
            let name_str = &state.names[name as usize];
            if name_str == "thread_name" && args_off > 0 {
                let end = skip_value(raw, args_off as usize);
                let mut tmp_strs = Vec::new();
                let mut tmp_idx = FnvMap::default();
                let mut tmp_pairs = Vec::new();
                parse_args_flat(&raw[args_off as usize..end], &mut tmp_strs, &mut tmp_idx, &mut tmp_pairs);
                for &[k, v] in &tmp_pairs {
                    if tmp_strs.get(k as usize).map_or(false, |s| s == "name") {
                        if let Some(s) = tmp_strs.get(v as usize) {
                            state.thread_names.insert((pid, tid), s.clone());
                        }
                        break;
                    }
                }
            }
        }
    }
}

fn merge_intern_table(
    global: &mut Vec<String>,
    global_idx: &mut FnvMap<u32>,
    local: &[String],
    local_idx: &FnvMap<u32>,
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

fn calc_n_threads(data_len: usize, max_parse_threads: usize) -> usize {
    if max_parse_threads == 1 || data_len < 10 * 1024 * 1024 { return 1; }
    let avail = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
    let cap = if max_parse_threads > 1 { max_parse_threads } else { 8 };
    avail.clamp(2, cap)
}

fn collect_chunks(handles: Vec<std::thread::ScopedJoinHandle<'_, ChunkState>>) -> Vec<ChunkState> {
    handles.into_iter().filter_map(|h| match h.join() {
        Ok(state) => Some(state),
        Err(e) => {
            let msg = e.downcast_ref::<&str>().map(|s| s.to_string())
                .or_else(|| e.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            eprintln!("  parse thread panicked: {msg}");
            None
        }
    }).collect()
}

fn decompress_parse_seq(
    path: &str, counter: &Arc<AtomicUsize>, max_parse_threads: usize, t0: &Instant,
) -> Result<(RawData, Vec<ChunkState>, usize), String> {
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
    let n_threads = calc_n_threads(raw.len(), max_parse_threads);

    let split_points = find_split_points(&raw, array_start, n_threads);
    let n_chunks = split_points.len() - 1;

    let chunks: Vec<ChunkState> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..n_chunks).map(|i| {
            let start = split_points[i];
            let end = split_points[i + 1];
            let raw_ref = &raw;
            let ctr = &*counter;
            s.spawn(move || {
                let mut state = ChunkState::new();
                parse_chunk(raw_ref, start, end, &mut state, ctr);
                state
            })
        }).collect();
        collect_chunks(handles)
    });

    Ok((raw, chunks, n_chunks))
}

fn try_libdeflate(compressed: &[u8], estimated: usize) -> Option<Vec<u8>> {
    let mut decompressor = libdeflater::Decompressor::new();
    let mut buf = vec![0u8; estimated];
    match decompressor.gzip_decompress(compressed, &mut buf) {
        Ok(actual) => { buf.truncate(actual); Some(buf) }
        Err(_) => None,
    }
}

fn decompress_parse_streaming(
    path: &str, counter: &Arc<AtomicUsize>, max_parse_threads: usize, t0: &Instant,
) -> Result<(RawData, Vec<ChunkState>), String> {
    let compressed = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;

    let gz_isize = if compressed.len() >= 4 {
        u32::from_le_bytes(compressed[compressed.len()-4..].try_into().unwrap()) as usize
    } else { 0 };
    let estimated = if gz_isize > compressed.len() {
        gz_isize + gz_isize / 10
    } else {
        compressed.len() * 25
    };

    if let Some(buf) = try_libdeflate(&compressed, estimated) {
        eprintln!("  read: {:.2}s ({}MB)", t0.elapsed().as_secs_f64(), buf.len() / 1024 / 1024);
        let raw = RawData::Vec(buf);
        let te = find_key(&raw, b"traceEvents").ok_or("no traceEvents found")?;
        let mut pos = te + "\"traceEvents\"".len();
        pos = skip_ws(&raw, pos);
        if pos < raw.len() && raw[pos] == b':' { pos += 1; }
        pos = skip_ws(&raw, pos);
        if pos >= raw.len() || raw[pos] != b'[' {
            return Err("malformed traceEvents".into());
        }
        let array_start = pos + 1;
        let n_threads = calc_n_threads(raw.len(), max_parse_threads);
        let split_points = find_split_points(&raw, array_start, n_threads);
        let n_chunks = split_points.len() - 1;
        let chunks: Vec<ChunkState> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..n_chunks).map(|i| {
                let start = split_points[i];
                let end = split_points[i + 1];
                let raw_ref = &raw;
                let ctr = &*counter;
                s.spawn(move || {
                    let mut state = ChunkState::new();
                    parse_chunk(raw_ref, start, end, &mut state, ctr);
                    state
                })
            }).collect();
            collect_chunks(handles)
        });
        eprintln!("  scan: {:.2}s ({} events, {}x parallel)",
            t0.elapsed().as_secs_f64(), chunks.iter().map(|c| c.total_events).sum::<usize>(), n_chunks);
        return Ok((raw, chunks));
    }

    let n_threads = calc_n_threads(estimated, max_parse_threads);

    let mut backing: Vec<u8> = Vec::with_capacity(estimated);
    let backing_addr = backing.as_mut_ptr() as usize;
    let backing_cap = backing.capacity();
    let write_pos = AtomicUsize::new(0);
    let decomp_done = AtomicBool::new(false);
    let overflow = AtomicBool::new(false);

    let result: Result<Vec<ChunkState>, String> = std::thread::scope(|s| {
        let wp = &write_pos;
        let dd = &decomp_done;
        let ov = &overflow;

        let decomp_addr = backing_addr;
        s.spawn(move || {
            let ptr = decomp_addr as *mut u8;
            let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
            let mut tmp = vec![0u8; 2 * 1024 * 1024];
            let mut pos = 0usize;
            loop {
                match decoder.read(&mut tmp) {
                    Ok(0) => break,
                    Ok(n) => {
                        if pos + n > backing_cap {
                            ov.store(true, Ordering::Release);
                            break;
                        }
                        unsafe {
                            std::ptr::copy_nonoverlapping(tmp.as_ptr(), ptr.add(pos), n);
                        }
                        pos += n;
                        wp.store(pos, Ordering::Release);
                    }
                    Err(e) => {
                        if pos == 0 { ov.store(true, Ordering::Release); }
                        else { eprintln!("  truncated gz: {e}"); }
                        break;
                    }
                }
            }
            dd.store(true, Ordering::Release);
        });

        let wait_for = |target: usize| -> usize {
            loop {
                let avail = wp.load(Ordering::Acquire);
                if avail >= target { return avail; }
                if dd.load(Ordering::Acquire) || ov.load(Ordering::Acquire) { return avail; }
                std::hint::spin_loop();
            }
        };

        let avail = wait_for(64 * 1024);
        if ov.load(Ordering::Acquire) { return Err("overflow".into()); }

        let read_ptr = backing_addr as *const u8;
        let slice = |len: usize| unsafe { std::slice::from_raw_parts(read_ptr, len) };

        let te = find_key(slice(avail), b"traceEvents")
            .ok_or_else(|| "no traceEvents found".to_string())?;
        let mut hpos = te + "\"traceEvents\"".len();
        let hdr = slice(avail);
        hpos = skip_ws(hdr, hpos);
        if hpos < hdr.len() && hdr[hpos] == b':' { hpos += 1; }
        hpos = skip_ws(hdr, hpos);
        if hpos >= hdr.len() || hdr[hpos] != b'[' {
            return Err("malformed traceEvents".into());
        }
        let array_start = hpos + 1;

        let chunk_size = (estimated - array_start) / n_threads;
        let mut handles: Vec<std::thread::ScopedJoinHandle<'_, ChunkState>> = Vec::new();
        let mut prev_start = array_start;

        for i in 1..n_threads {
            let target = array_start + i * chunk_size;
            let search_end = target + 256 * 1024;
            let avail = wait_for(search_end);
            if ov.load(Ordering::Acquire) { return Err("overflow".into()); }
            if let Some(bp) = find_event_start(slice(avail), target, avail) {
                let start = prev_start;
                let end = bp;
                let ctr = &*counter;
                let addr = backing_addr;
                handles.push(s.spawn(move || {
                    let sub = unsafe { std::slice::from_raw_parts((addr as *const u8).add(start), end - start) };
                    let mut state = ChunkState::new();
                    parse_chunk(sub, 0, sub.len(), &mut state, ctr);
                    for (_, _, ev) in &mut state.events {
                        if ev.args_off > 0 { ev.args_off += start as u32; }
                    }
                    state
                }));
                prev_start = bp;
            }
        }

        while !dd.load(Ordering::Acquire) { std::hint::spin_loop(); }
        if ov.load(Ordering::Acquire) { return Err("overflow".into()); }
        let final_len = wp.load(Ordering::Acquire);
        {
            let start = prev_start;
            let end = final_len;
            let ctr = &*counter;
            let addr = backing_addr;
            handles.push(s.spawn(move || {
                let sub = unsafe { std::slice::from_raw_parts((addr as *const u8).add(start), end - start) };
                let mut state = ChunkState::new();
                parse_chunk(sub, 0, sub.len(), &mut state, ctr);
                for (_, _, ev) in &mut state.events {
                    if ev.args_off > 0 { ev.args_off += start as u32; }
                }
                state
            }));
        }

        eprintln!("  stream: {:.2}s ({}MB, {} chunks)",
            t0.elapsed().as_secs_f64(), final_len / 1024 / 1024, handles.len());

        Ok(collect_chunks(handles))
    });

    let chunks = result?;
    let final_len = write_pos.load(Ordering::Acquire);
    unsafe { backing.set_len(final_len); }
    eprintln!("  scan: {:.2}s ({} events, streaming)",
        t0.elapsed().as_secs_f64(), chunks.iter().map(|c| c.total_events).sum::<usize>());
    Ok((RawData::Vec(backing), chunks))
}

fn build_trace(raw: RawData, chunks: Vec<ChunkState>, n_chunks: usize, t0: &Instant) -> Result<Trace, String> {
    let mut names: Vec<String> = vec![String::new()];
    let mut name_idx: FnvMap<u32> = FnvMap::default();
    name_idx.insert(0, 0);
    let mut cats: Vec<String> = vec![String::new()];
    let mut cat_idx: FnvMap<u32> = FnvMap::default();
    cat_idx.insert(0, 0);

    let mut track_map: HashMap<(u64, u64), Vec<Event>> = HashMap::new();
    let mut thread_names: HashMap<(u64, u64), String> = HashMap::new();
    let mut min_ts = f64::MAX;
    let mut max_ts = f64::MIN;
    let mut total_events: usize = 0;
    let mut scan_count: u64 = 0;

    for chunk in chunks {
        let name_remap = merge_intern_table(&mut names, &mut name_idx, &chunk.names, &chunk.name_idx);
        let cat_remap = merge_intern_table(&mut cats, &mut cat_idx, &chunk.cats, &chunk.cat_idx);

        for (pid, tid, mut ev) in chunk.events {
            ev.name = name_remap[ev.name as usize];
            ev.cat = cat_remap[ev.cat as usize];
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
    drop(name_idx);
    drop(cat_idx);

    let raw_buf: Arc<Vec<u8>> = Arc::new(raw.into_vec());

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
                        let c = &cat_ref[e.cat as usize];
                        c == "kernel" || c.starts_with("gpu_")
                    }).count();
                    let gpu = gpu_count > evs.len() / 2;
                    let label = tn_ref.get(&(pid, tid)).cloned().unwrap_or_else(|| {
                        if gpu { format!("GPU {tid}") } else { format!("Thread {tid}") }
                    });
                    evs.shrink_to_fit();
                    Track { label, gpu, events: evs, max_depth, prefix_max_dur, raw_buf_idx: 0 }
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

    eprintln!("  lanes: {:.2}s ({} tracks)", t2.elapsed().as_secs_f64(), tracks.len());
    Ok(Trace { tracks, names, cats, raw_bufs: vec![raw_buf], stats, max_ts, min_ts, total_events, device })
}

fn decompress_parse(path: &str, counter: &Arc<AtomicUsize>, max_parse_threads: usize, t0: &Instant) -> Result<(RawData, Vec<ChunkState>, usize), String> {
    let use_streaming = path.ends_with(".json.gz") && max_parse_threads != 1;
    if use_streaming {
        match decompress_parse_streaming(path, counter, max_parse_threads, t0) {
            Ok((r, c)) => { let n = c.len(); Ok((r, c, n)) }
            Err(_) => decompress_parse_seq(path, counter, max_parse_threads, t0)
        }
    } else {
        decompress_parse_seq(path, counter, max_parse_threads, t0)
    }
}

pub fn load_trace(path: &str, counter: &Arc<AtomicUsize>, max_parse_threads: usize) -> Result<Trace, String> {
    let t0 = Instant::now();
    let (raw, chunks, n_chunks) = decompress_parse(path, counter, max_parse_threads, &t0)?;
    let mut trace = build_trace(raw, chunks, n_chunks, &t0)?;
    compact_args(&mut trace);

    let event_bytes: usize = trace.tracks.iter().map(|t| t.events.len() * std::mem::size_of::<Event>()).sum();
    let str_bytes: usize = trace.names.iter().chain(trace.cats.iter()).map(|s| s.len() + std::mem::size_of::<String>()).sum();
    let args_bytes: usize = trace.raw_bufs.iter().map(|b| b.len()).sum();
    eprintln!("  memory: events={}MB strings={}MB args={}MB total={}MB",
        event_bytes / 1024 / 1024, str_bytes / 1024 / 1024, args_bytes / 1024 / 1024,
        (event_bytes + str_bytes + args_bytes) / 1024 / 1024);
    eprintln!("  total: {:.2}s", t0.elapsed().as_secs_f64());
    Ok(trace)
}

fn clone_trace(t: &Trace) -> Trace {
    Trace {
        tracks: t.tracks.clone(),
        names: t.names.clone(),
        cats: t.cats.clone(),
        raw_bufs: t.raw_bufs.clone(),
        stats: t.stats.clone(),
        max_ts: t.max_ts,
        min_ts: t.min_ts,
        total_events: t.total_events,
        device: t.device.clone(),
    }
}

fn send_progressive(trace: Trace, tx: &std::sync::mpsc::Sender<Result<Trace, String>>, t0: &Instant) {
    eprintln!("  ready: {:.2}s ({} tracks)", t0.elapsed().as_secs_f64(), trace.tracks.len());
    let mut compact_trace = clone_trace(&trace);
    let _ = tx.send(Ok(trace));

    compact_args(&mut compact_trace);
    let args_bytes: usize = compact_trace.raw_bufs.iter().map(|b| b.len()).sum();
    eprintln!("  compact: {:.2}s (args={}MB)", t0.elapsed().as_secs_f64(), args_bytes / 1024 / 1024);
    let _ = tx.send(Ok(compact_trace));
}

pub fn load_trace_progressive(
    path: &str, counter: &Arc<AtomicUsize>, max_parse_threads: usize,
    tx: &std::sync::mpsc::Sender<Result<Trace, String>>,
) {
    let t0 = Instant::now();
    let (raw, chunks, n_chunks) = match decompress_parse(path, counter, max_parse_threads, &t0) {
        Ok(v) => v,
        Err(e) => { let _ = tx.send(Err(e)); return; }
    };
    match build_trace(raw, chunks, n_chunks, &t0) {
        Ok(trace) => send_progressive(trace, tx, &t0),
        Err(e) => { let _ = tx.send(Err(e)); }
    }
}

pub(crate) fn is_trace_file(name: &str) -> bool {
    name.ends_with(".json") || name.ends_with(".json.gz")
        || name.ends_with(".tar.gz") || name.ends_with(".tgz")
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
                    if is_trace_file(name) {
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

fn merge_intern_direct(
    global: &mut Vec<String>,
    global_idx: &mut FnvMap<u32>,
    local: &[String],
) -> Vec<u32> {
    let mut remap = vec![0u32; local.len()];
    for (i, s) in local.iter().enumerate() {
        let hash = crate::parse::fnv1a(s.as_bytes());
        let gi = *global_idx.entry(hash).or_insert_with(|| {
            let id = global.len() as u32;
            global.push(s.clone());
            id
        });
        remap[i] = gi;
    }
    remap
}

pub fn merge_traces(traces: Vec<(usize, Trace)>) -> Trace {
    let global_min = traces.iter().map(|(_, t)| t.min_ts).fold(f64::MAX, f64::min);

    let mut names: Vec<String> = vec![String::new()];
    let mut name_idx: FnvMap<u32> = FnvMap::with_capacity_and_hasher(
        traces.iter().map(|(_, t)| t.names.len()).max().unwrap_or(0) * 2,
        Default::default(),
    );
    name_idx.insert(0, 0);
    let mut cats: Vec<String> = vec![String::new()];
    let mut cat_idx: FnvMap<u32> = FnvMap::with_capacity_and_hasher(
        traces.iter().map(|(_, t)| t.cats.len()).max().unwrap_or(0) * 2,
        Default::default(),
    );
    cat_idx.insert(0, 0);
    let mut device = String::new();

    let mut remap_info: Vec<(Vec<u32>, Vec<u32>, f64)> = Vec::with_capacity(traces.len());
    let mut all_raw_bufs: Vec<Arc<Vec<u8>>> = Vec::new();

    for (_, trace) in &traces {
        let time_offset = trace.min_ts - global_min;

        let name_remap = merge_intern_direct(&mut names, &mut name_idx, &trace.names);
        let cat_remap = merge_intern_direct(&mut cats, &mut cat_idx, &trace.cats);

        if device.is_empty() && !trace.device.is_empty() {
            device = trace.device.clone();
        }

        remap_info.push((name_remap, cat_remap, time_offset));
    }

    let remapped: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = traces.into_iter().zip(remap_info).enumerate().map(|(trace_idx, ((rank, mut trace), (name_remap, cat_remap, time_offset)))| {
            s.spawn(move || {
                let raw_buf_idx = trace_idx as u8;
                let mut max_ts: f64 = 0.0;
                let mut total_events = 0usize;
                let mut local_dur: HashMap<u32, Vec<f64>> = HashMap::new();
                for track in &mut trace.tracks {
                    track.label = format!("[rank {}] {}", rank, track.label);
                    track.raw_buf_idx = raw_buf_idx;
                    for ev in &mut track.events {
                        ev.ts += time_offset;
                        ev.name = name_remap[ev.name as usize];
                        ev.cat = cat_remap[ev.cat as usize];
                        max_ts = max_ts.max(ev.ts + ev.dur);
                        local_dur.entry(ev.name).or_default().push(ev.dur);
                    }
                    total_events += track.events.len();
                }

                (trace.tracks, trace.raw_bufs, max_ts, total_events, local_dur)
            })
        }).collect();
        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });

    let mut all_tracks: Vec<Track> = Vec::new();
    let mut total_events = 0;
    let mut max_ts: f64 = 0.0;
    let mut dur_map: HashMap<u32, Vec<f64>> = HashMap::new();

    for (tracks, raw_bufs, mt, te, local_dur) in remapped {
        all_tracks.extend(tracks);
        all_raw_bufs.extend(raw_bufs);
        total_events += te;
        max_ts = max_ts.max(mt);
        for (name, durs) in local_dur {
            dur_map.entry(name).or_default().extend(durs);
        }
    }

    all_tracks.sort_by(|a, b| a.label.cmp(&b.label));

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

    let mut trace = Trace {
        tracks: all_tracks, names, cats, raw_bufs: all_raw_bufs, stats,
        max_ts, min_ts: global_min, total_events, device,
    };
    compact_args(&mut trace);
    trace
}

fn load_trace_raw(path: &str, counter: &Arc<AtomicUsize>, max_parse_threads: usize) -> Result<Trace, String> {
    let t0 = Instant::now();
    let (raw, chunks, n_chunks) = decompress_parse(path, counter, max_parse_threads, &t0)?;
    build_trace(raw, chunks, n_chunks, &t0)
}

fn merge_traces_raw(traces: Vec<(usize, Trace)>) -> Trace {
    let global_min = traces.iter().map(|(_, t)| t.min_ts).fold(f64::MAX, f64::min);

    let mut names: Vec<String> = vec![String::new()];
    let mut name_idx: FnvMap<u32> = FnvMap::with_capacity_and_hasher(
        traces.iter().map(|(_, t)| t.names.len()).max().unwrap_or(0) * 2,
        Default::default(),
    );
    name_idx.insert(0, 0);
    let mut cats: Vec<String> = vec![String::new()];
    let mut cat_idx: FnvMap<u32> = FnvMap::with_capacity_and_hasher(
        traces.iter().map(|(_, t)| t.cats.len()).max().unwrap_or(0) * 2,
        Default::default(),
    );
    cat_idx.insert(0, 0);
    let mut device = String::new();

    let mut remap_info: Vec<(Vec<u32>, Vec<u32>, f64)> = Vec::with_capacity(traces.len());
    let mut all_raw_bufs: Vec<Arc<Vec<u8>>> = Vec::new();

    for (_, trace) in &traces {
        let time_offset = trace.min_ts - global_min;
        let name_remap = merge_intern_direct(&mut names, &mut name_idx, &trace.names);
        let cat_remap = merge_intern_direct(&mut cats, &mut cat_idx, &trace.cats);
        if device.is_empty() && !trace.device.is_empty() {
            device = trace.device.clone();
        }
        remap_info.push((name_remap, cat_remap, time_offset));
    }

    let remapped: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = traces.into_iter().zip(remap_info).enumerate().map(|(trace_idx, ((rank, mut trace), (name_remap, cat_remap, time_offset)))| {
            s.spawn(move || {
                let raw_buf_idx = trace_idx as u8;
                let mut max_ts: f64 = 0.0;
                let mut total_events = 0usize;
                let mut local_dur: HashMap<u32, Vec<f64>> = HashMap::new();
                for track in &mut trace.tracks {
                    track.label = format!("[rank {}] {}", rank, track.label);
                    track.raw_buf_idx = raw_buf_idx;
                    for ev in &mut track.events {
                        ev.ts += time_offset;
                        ev.name = name_remap[ev.name as usize];
                        ev.cat = cat_remap[ev.cat as usize];
                        max_ts = max_ts.max(ev.ts + ev.dur);
                        local_dur.entry(ev.name).or_default().push(ev.dur);
                    }
                    total_events += track.events.len();
                }
                (trace.tracks, trace.raw_bufs, max_ts, total_events, local_dur)
            })
        }).collect();
        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });

    let mut all_tracks: Vec<Track> = Vec::new();
    let mut total_events = 0;
    let mut max_ts: f64 = 0.0;
    let mut dur_map: HashMap<u32, Vec<f64>> = HashMap::new();

    for (tracks, raw_bufs, mt, te, local_dur) in remapped {
        all_tracks.extend(tracks);
        all_raw_bufs.extend(raw_bufs);
        total_events += te;
        max_ts = max_ts.max(mt);
        for (name, durs) in local_dur {
            dur_map.entry(name).or_default().extend(durs);
        }
    }

    all_tracks.sort_by(|a, b| a.label.cmp(&b.label));

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

    Trace {
        tracks: all_tracks, names, cats, raw_bufs: all_raw_bufs, stats,
        max_ts, min_ts: global_min, total_events, device,
    }
}

pub fn load_multi_progressive(
    rank_paths: Vec<(usize, String)>, counter: &Arc<AtomicUsize>, tpf: usize,
    tx: &std::sync::mpsc::Sender<Result<Trace, String>>,
) {
    let t0 = Instant::now();
    let results: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = rank_paths.iter().map(|(rank, path)| {
            let r = *rank;
            let ctr = counter.clone();
            s.spawn(move || (r, load_trace_raw(path, &ctr, tpf)))
        }).collect();
        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    let mut traces = Vec::new();
    for (rank, result) in results {
        match result {
            Ok(t) => traces.push((rank, t)),
            Err(e) => { eprintln!("  rank {rank}: {e}"); }
        }
    }
    if traces.is_empty() {
        let _ = tx.send(Err("all ranks failed to load".into()));
        return;
    }
    let trace = merge_traces_raw(traces);
    send_progressive(trace, tx, &t0);
}

fn compact_args(trace: &mut Trace) {
    if trace.raw_bufs.is_empty() { return; }
    let mut compact = vec![0u8];
    let raw_bufs: Vec<_> = trace.raw_bufs.iter().cloned().collect();
    for track in &mut trace.tracks {
        let raw = &raw_bufs[track.raw_buf_idx as usize];
        for ev in &mut track.events {
            if ev.args_off > 0 {
                let off = ev.args_off as usize;
                let len = if ev.args_len > 0 {
                    ev.args_len as usize
                } else {
                    skip_value(raw, off) - off
                };
                let new_off = compact.len();
                compact.extend_from_slice(&raw[off..off + len]);
                ev.args_off = new_off as u32;
                ev.args_len = len.min(u16::MAX as usize) as u16;
            }
        }
        track.raw_buf_idx = 0;
    }
    compact.shrink_to_fit();
    trace.raw_bufs = vec![Arc::new(compact)];
}

fn read_gz_tolerant<R: std::io::Read>(reader: R) -> Result<Vec<u8>, String> {
    let mut decoder = flate2::read::GzDecoder::new(reader);
    let mut buf = Vec::new();
    let mut tmp = vec![0u8; 1024 * 1024];
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
