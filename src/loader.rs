use crate::parse::*;
use crate::types::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{BufReader, Read, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use crate::time::Instant;

pub enum RawData {
    Vec(Vec<u8>),
    #[cfg(not(target_arch = "wasm32"))]
    Mmap(memmap2::Mmap),
}

impl std::ops::Deref for RawData {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            RawData::Vec(v) => v,
            #[cfg(not(target_arch = "wasm32"))]
            RawData::Mmap(m) => m,
        }
    }
}

impl RawData {
    fn into_vec(self) -> Vec<u8> {
        match self {
            RawData::Vec(v) => v,
            #[cfg(not(target_arch = "wasm32"))]
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
    flows: Vec<(u64, u64, u64, f64, bool)>,
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
            flows: Vec::new(),
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
        let mut flow_id: u64 = 0;

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
                            if ph != b'X' && ph != b'M' && ph != b's' && ph != b'f' {
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
                    [b'i', b'd'] => {
                        if pos < raw.len() && raw[pos] >= b'0' && raw[pos] <= b'9' {
                            let s = pos;
                            pos = skip_number(raw, pos);
                            flow_id = parse_f64(&raw[s..pos]) as u64;
                        } else { pos = skip_value(raw, pos); }
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
        } else if ph == b's' || ph == b'f' {
            state.flows.push((flow_id, pid, tid, ts, ph == b's'));
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

// Used only by decompress_parse_streaming's manual overlapped-decompress path
// below, which spawns real OS threads (native-only — see the comment there
// for why that path can't move to rayon).
#[cfg(not(target_arch = "wasm32"))]
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

// Runs `f(0)..f(n_chunks)` across rayon's pool (real OS threads natively,
// transparent sequential fallback on wasm32 — see module-level notes). Mirrors
// `collect_chunks`'s panic handling: a chunk that panics is dropped with a
// message instead of taking the whole parse down, matching the previous
// std::thread::scope + ScopedJoinHandle::join() behavior exactly.
fn parse_chunks_parallel<F>(n_chunks: usize, f: F) -> Vec<ChunkState>
where
    F: Fn(usize) -> ChunkState + Sync,
{
    (0..n_chunks)
        .into_par_iter()
        .filter_map(|i| match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(i))) {
            Ok(state) => Some(state),
            Err(e) => {
                let msg = e.downcast_ref::<&str>().map(|s| s.to_string())
                    .or_else(|| e.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_string());
                eprintln!("  parse thread panicked: {msg}");
                None
            }
        })
        .collect()
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

    let chunks: Vec<ChunkState> = parse_chunks_parallel(n_chunks, |i| {
        let start = split_points[i];
        let end = split_points[i + 1];
        let mut state = ChunkState::new();
        parse_chunk(&raw, start, end, &mut state, &counter);
        state
    });

    Ok((raw, chunks, n_chunks))
}

#[cfg(not(target_arch = "wasm32"))]
fn try_libdeflate(compressed: &[u8], estimated: usize) -> Option<Vec<u8>> {
    let mut decompressor = libdeflater::Decompressor::new();
    let mut buf = vec![0u8; estimated];
    match decompressor.gzip_decompress(compressed, &mut buf) {
        Ok(actual) => { buf.truncate(actual); Some(buf) }
        Err(_) => None,
    }
}

// wasm has no libdeflate binding (it's a C library); fall back to the
// flate2::GzDecoder path already used elsewhere in this file.
#[cfg(target_arch = "wasm32")]
fn try_libdeflate(_compressed: &[u8], _estimated: usize) -> Option<Vec<u8>> { None }

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
        let chunks: Vec<ChunkState> = parse_chunks_parallel(n_chunks, |i| {
            let start = split_points[i];
            let end = split_points[i + 1];
            let mut state = ChunkState::new();
            parse_chunk(&raw, start, end, &mut state, &counter);
            state
        });
        eprintln!("  scan: {:.2}s ({} events, {}x parallel)",
            t0.elapsed().as_secs_f64(), chunks.iter().map(|c| c.total_events).sum::<usize>(), n_chunks);
        return Ok((raw, chunks));
    }

    // The rest of this function overlaps background decompression with
    // progressive parsing of the bytes decompressed so far, using a spawned
    // thread plus a busy spin-wait on shared atomics (`wait_for` below) —
    // not the "split N independent chunks, run them, collect" shape the rest
    // of this file's parallelism was migrated to rayon for. That distinction
    // matters here: rayon's `scope`/`spawn` are only deadlock-safe under
    // nesting or its no-real-threads wasm fallback because the *waiting* side
    // (`join`/the end of `scope`) is cooperative — an idle worker steals other
    // pending work while it waits. A raw `spin_loop()` never calls back into
    // the scheduler, so it can't be helped along: on wasm's sequential
    // fallback the spawned decompressor would never run before this spins
    // forever (the "spawned" job is only picked up once the scope closure
    // returns), and even natively, if every rayon worker ends up spinning
    // here at once (e.g. loading several ranks in parallel on a
    // fully-subscribed pool), there's no worker left to run the decompressor
    // job either. Real `std::thread::spawn` sidesteps both failure modes by
    // guaranteeing a dedicated OS thread outside rayon's pool — so this stays
    // on `std::thread` and is native-only; `decompress_parse` already falls
    // back to `decompress_parse_seq` (fully migrated, safe) whenever this
    // function returns `Err`, so wasm just takes that path directly instead.
    #[cfg(target_arch = "wasm32")]
    {
        return Err("streaming gz decompression is native-only".into());
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
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
    let mut flow_events: Vec<(u64, u64, u64, f64, bool)> = Vec::new();
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

        flow_events.extend(chunk.flows.iter().copied());

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

    // vLLM traces embed a top-level `vllm_version` string (e.g.
    // "0.26.1rc1.dev528+gf8d03e774"). Plenty of traces lack it — leave it empty.
    let mut vllm_version = String::new();
    if let Some(vp) = find_key(&raw, b"vllm_version") {
        let mut q = vp + "\"vllm_version\"".len();
        q = skip_ws(&raw, q);
        if q < raw.len() && raw[q] == b':' { q += 1; }
        q = skip_ws(&raw, q);
        if q < raw.len() && raw[q] == b'"' {
            let end = skip_string(&raw, q);
            vllm_version = json_unescape(std::str::from_utf8(&raw[q + 1..end - 1]).unwrap_or(""));
        }
    }

    // `distributedInfo.{rank,world_size}` identify this rank within the job.
    // `"rank"` (singular) only appears at the top of the distributedInfo object;
    // the pg_config arrays use `"ranks"` (plural), which find_key won't match.
    let (mut dist_rank, mut dist_world) = (-1i32, 0i32);
    if let Some(di) = find_key(&raw, b"distributedInfo") {
        if let Some(r) = find_int_field(&raw, di, b"rank") { dist_rank = r as i32; }
        if let Some(w) = find_int_field(&raw, di, b"world_size") { dist_world = w as i32; }
    }

    eprintln!("  scan: {:.2}s ({} objects, {} events, {} names, {}x parallel)",
        t0.elapsed().as_secs_f64(), scan_count, total_events, names.len(), n_chunks);
    drop(name_idx);
    drop(cat_idx);

    let raw_buf: Arc<ArgsBuf> = Arc::new(ArgsBuf::Heap(raw.into_vec()));

    if min_ts == f64::MAX {
        return Err("no duration events found".into());
    }

    let t2 = Instant::now();

    let mut keyed_tracks: Vec<((u64, u64), Track)> = {
        let cat_ref = &cats;
        let tn_ref = &thread_names;
        track_map
            .into_par_iter()
            .map(|((pid, tid), mut evs)| {
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
                let track = Track { label, gpu, events: evs, max_depth, prefix_max_dur, raw_buf_idx: 0 };
                ((pid, tid), track)
            })
            .collect()
    };

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

    eprintln!("  lanes: {:.2}s ({} tracks, {} flow_pairs)", t2.elapsed().as_secs_f64(), tracks.len(), flow_pairs.len());
    Ok(Trace { tracks, names, cats, raw_bufs: vec![raw_buf], stats, max_ts, min_ts, total_events, device, vllm_version, dist_rank, dist_world, flow_pairs })
}

const CACHE_MAGIC: &[u8; 4] = b"TRV2";
// Bumped only when the header layout or an existing field's encoding changes.
// New *trailing* fields don't need a bump: the reader bounds-checks each one and
// defaults it when missing, and loads accept any version <= this (older caches
// simply lack the newer trailing fields). Only a newer-than-known cache is
// rejected, since its layout may have diverged.
const CACHE_VERSION: u32 = 3;

fn cache_path(source: &str, cache_dir: Option<&str>) -> String {
    if let Some(dir) = cache_dir {
        let fname = std::path::Path::new(source)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or(source);
        format!("{dir}/{fname}.tvcache")
    } else {
        format!("{source}.tvcache")
    }
}

pub fn cache_dir_for_folder(dir: &str) -> String {
    let cd = format!("{dir}.tvcache");
    std::fs::create_dir_all(&cd).ok();
    cd
}

fn source_meta(path: &str) -> Option<(u64, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?
        .duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
    Some((meta.len(), mtime))
}

fn write_u32(w: &mut impl std::io::Write, v: u32) { w.write_all(&v.to_le_bytes()).ok(); }
fn write_u64(w: &mut impl std::io::Write, v: u64) { w.write_all(&v.to_le_bytes()).ok(); }
fn write_f64(w: &mut impl std::io::Write, v: f64) { w.write_all(&v.to_le_bytes()).ok(); }

fn write_strings(w: &mut impl std::io::Write, strings: &[String]) {
    for s in strings {
        write_u32(w, s.len() as u32);
        w.write_all(s.as_bytes()).ok();
    }
}

fn pad_to_8(w: &mut impl std::io::Write, written: usize) {
    let rem = written % 8;
    if rem != 0 {
        let pad = [0u8; 8];
        w.write_all(&pad[..8 - rem]).ok();
    }
}

/// Real zstd's 4-byte frame magic — used to tell a compressed `.tvcache`
/// (every cache this app writes going forward) apart from an
/// already-on-disk uncompressed one written before this existed (still
/// valid — just starts with `CACHE_MAGIC` directly instead), so existing
/// caches don't need to be invalidated for this change to take effect.
/// Shared across platforms: native detects it in `read_and_decompress_cache`
/// (real `zstd` crate), wasm in `load_trace_from_bytes_progressive` (the
/// pure-Rust `ruzstd` decoder — verified directly against real output from
/// the native encoder byte-for-byte before wiring this in).
const ZSTD_FRAME_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// Wraps the main `.tvcache`/`_merged.tvcache` cache format in zstd —
/// unlike `export_gpu_only`'s xz (an occasional manual export), this needs
/// to stay fast enough on *every* trace load/save to not be felt. Measured
/// directly on a real 234MB (post synth-args-encoding) cache: level 3
/// compresses in ~0.1s and decompresses in ~0.1s for a 9.4x further
/// reduction (234MB -> 25MB) — see Cargo.toml's comment on this dependency
/// for the xz/gzip comparison that ruled those out here specifically.
/// `save_cache`/`save_merged_cache` aren't `cfg`-gated (they no-op via a
/// failed `std::fs::write` on wasm, same as before this existed), so this
/// needs a wasm-side passthrough purely so the crate reference compiles —
/// it's never actually reached at runtime there.
#[cfg(not(target_arch = "wasm32"))]
fn maybe_compress_cache(bytes: Vec<u8>) -> Vec<u8> {
    zstd::stream::encode_all(&bytes[..], 3).unwrap_or(bytes)
}
#[cfg(target_arch = "wasm32")]
fn maybe_compress_cache(bytes: Vec<u8>) -> Vec<u8> { bytes }

/// Reads a cache file and transparently zstd-decompresses it if it's in
/// the new compressed format (see `maybe_compress_cache`), or returns its
/// bytes as-is for an older uncompressed one. Every native cache reader
/// (`load_cache`, `load_cache_direct`, `load_merged_cache`) goes through
/// this before handing bytes to `load_cache_from_bytes`, so that function
/// itself never needs to know compression is involved at all.
#[cfg(not(target_arch = "wasm32"))]
fn read_and_decompress_cache(path: &str) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() >= 4 && bytes[0..4] == ZSTD_FRAME_MAGIC {
        zstd::stream::decode_all(&bytes[..]).ok()
    } else {
        Some(bytes)
    }
}

/// High bit of `Event::args_off`, used only within a `.tvcache` file's
/// on-disk bytes (never on a live in-memory `Trace` — `expand_synth_python_args`
/// always resolves it away before `load_cache_from_mmap`/`load_cache_from_bytes`
/// return) to mean "this event's args are an `append_synth_python_args`
/// template substitution, not a literal byte offset; the low 31 bits index
/// the trailing synth-args record array." Real args offsets never reach
/// 2^31 in practice, so this can't collide with a genuine offset.
const SYNTH_ARGS_FLAG: u32 = 0x8000_0000;

/// Finds `"<key>":` in `bytes` and parses the number (or `null`) right
/// after it, returning the parsed value plus the exact `[start, end)` byte
/// range of the number/`null` token itself — used to both extract the
/// value and slice out the surrounding template text.
fn find_num(bytes: &[u8], key: &[u8]) -> Option<(Option<i64>, usize, usize)> {
    let pos = bytes.windows(key.len()).position(|w| w == key)? + key.len();
    let mut i = pos;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b':') { i += 1; }
    let start = i;
    if bytes[i..].starts_with(b"null") {
        return Some((None, start, start + 4));
    }
    while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
    if i == start { return None; }
    let v = std::str::from_utf8(&bytes[start..i]).ok()?.parse::<i64>().ok()?;
    Some((Some(v), start, i))
}

/// Checks whether `bytes` is a `{"Python parent id": P, "Python id": I,
/// "Ev Idx": E}`-shaped blob (the dominant category — often >90% of a
/// CPU-heavy trace's raw size — in traces produced by this vLLM/torch
/// profiler instrumentation; see the module-level measurement in the
/// commit that introduced this) that can be regenerated byte-for-byte from
/// just those 3 integers plus a fixed template. The template is captured
/// from the *first* match in the whole trace and reused for every
/// subsequent one (rather than re-derived per event): if some event's
/// surrounding text doesn't byte-for-byte match what that template
/// produces, this returns `None` — a real formatting difference, however
/// rare, always falls back to storing the event's raw bytes untouched,
/// never a corrupted reconstruction.
fn try_match_python_schema(bytes: &[u8], template: &mut Option<[Vec<u8>; 4]>) -> Option<(Option<i64>, i64, i64)> {
    let (parent_opt, p0, p1) = find_num(bytes, b"\"Python parent id\"")?;
    let (Some(id), i0, i1) = find_num(bytes, b"\"Python id\"")? else { return None };
    let (Some(ev), e0, e1) = find_num(bytes, b"\"Ev Idx\"")? else { return None };
    if p1 > i0 || i1 > e0 || e1 > bytes.len() { return None; }

    if template.is_none() {
        *template = Some([
            bytes[..p0].to_vec(),
            bytes[p1..i0].to_vec(),
            bytes[i1..e0].to_vec(),
            bytes[e1..].to_vec(),
        ]);
    }
    let mut rendered = Vec::with_capacity(bytes.len());
    append_synth_python_args(&mut rendered, template.as_ref().unwrap(), id, parent_opt, ev);
    if rendered != bytes { return None; }
    Some((parent_opt, id, ev))
}

/// Writes `v`'s decimal digits straight into `out` — avoids the heap
/// allocation `i64::to_string()` would do for each call. Matters here
/// because `render_synth_python_args` runs once per synthesized event
/// (millions of times) on every load of a cache containing this scheme;
/// measured directly, switching from `.to_string()` cut a real 406MB
/// trace's reload time from ~340ms back down to native-cache-like speed.
fn write_i64(out: &mut Vec<u8>, v: i64) {
    use std::io::Write;
    write!(out, "{v}").ok();
}

fn write_i64_or_null(out: &mut Vec<u8>, v: Option<i64>) {
    match v {
        None => out.extend_from_slice(b"null"),
        Some(v) => write_i64(out, v),
    }
}

/// Inverse of `try_match_python_schema`'s reconstruction: renders the exact
/// original bytes from the template pieces and the 3 resolved values,
/// appending straight onto an existing buffer rather than allocating a
/// fresh one — used both by `try_match_python_schema`'s own verification
/// and by `expand_synth_python_args`'s hot loop (millions of calls per
/// load of a trace containing this scheme), where a per-event heap
/// allocation would otherwise show up directly in load time.
fn append_synth_python_args(out: &mut Vec<u8>, template: &[Vec<u8>; 4], id: i64, parent: Option<i64>, ev: i64) {
    out.extend_from_slice(&template[0]);
    write_i64_or_null(out, parent);
    out.extend_from_slice(&template[1]);
    write_i64(out, id);
    out.extend_from_slice(&template[2]);
    write_i64(out, ev);
    out.extend_from_slice(&template[3]);
}

/// Writes every track's `Event` array to `w` (same bytes `save_cache`/
/// `save_merged_cache` always wrote), except events matching
/// `try_match_python_schema` get their `args_off` replaced with a
/// `SYNTH_ARGS_FLAG`-tagged index instead of their real byte offset.
/// Returns the captured template (`None` if nothing ever matched) and the
/// per-synth-event `[id_delta, parent_delta, ev_delta]` records, in the
/// same track-then-event order the flagged indices refer to — the caller
/// writes these as a new trailing section. `id_delta`/`ev_delta` are
/// residuals against cheap per-track predictors (previous matched event's
/// id + 1, and this event's own id) rather than raw values: measured on a
/// real 2.6M-event trace, "id predicted exactly" holds 99.9975% of the
/// time and "ev - id" is a single constant for the *entire* trace, so the
/// residual stream is almost all zeroes — trivially small even completely
/// uncompressed, which this on-disk format always is (no xz here, unlike
/// the `export_gpu_only` side of this codebase — see `save_cache`'s doc
/// comment on why the main cache stays a flat mmap-able blob).
/// Builds the args buffer that actually gets written to disk (`new_args`)
/// and per-track sentinel-flagged `Event` copies to write instead of
/// `tracks[..].events` directly. Unlike a naive "add a trailing synth
/// section" approach, this *excludes* every synthesized event's raw bytes
/// from `new_args` entirely (that's the whole point — writing the trailing
/// section on top of the untouched original args blob would only grow the
/// file); every other event's bytes get copied over at a new offset since
/// removing the synthesized ones shifts everything after them. Must run
/// before the cache header is written, since the header's args-length
/// field needs `new_args`'s (smaller) size, not the original's.
fn build_synth_and_compact_args(
    tracks: &[Track], args_buf: &[u8],
) -> (Vec<u8>, Vec<Vec<Event>>, Option<[Vec<u8>; 4]>, Vec<[i32; 3]>) {
    let mut template: Option<[Vec<u8>; 4]> = None;
    let mut records: Vec<[i32; 3]> = Vec::new();
    let mut new_args: Vec<u8> = vec![0u8]; // offset 0 reserved, matching compact_args' convention
    let mut modified: Vec<Vec<Event>> = Vec::with_capacity(tracks.len());

    for t in tracks {
        let mut predicted_id: i64 = 0;
        let mut first = true;
        let mut track_events = Vec::with_capacity(t.events.len());
        for e in &t.events {
            let mut ev = *e;
            if e.args_off != 0 && e.args_len != 0 {
                let off = e.args_off as usize;
                let len = e.args_len as usize;
                if off + len <= args_buf.len() {
                    let bytes = &args_buf[off..off + len];
                    if let Some((parent_opt, id, ev_val)) = try_match_python_schema(bytes, &mut template) {
                        let id_delta = if first { id } else { id - predicted_id };
                        first = false;
                        predicted_id = id + 1;
                        let ev_delta = ev_val - id;
                        let parent_delta = match parent_opt {
                            None => i32::MIN,
                            Some(p) => (id - p) as i32,
                        };
                        records.push([id_delta as i32, parent_delta, ev_delta as i32]);
                        ev.args_off = SYNTH_ARGS_FLAG | (records.len() as u32 - 1);
                        ev.args_len = 0;
                    } else {
                        let new_off = new_args.len();
                        new_args.extend_from_slice(bytes);
                        ev.args_off = new_off as u32;
                        ev.args_len = len.min(u16::MAX as usize) as u16;
                    }
                } else {
                    ev.args_off = 0;
                    ev.args_len = 0;
                }
            }
            track_events.push(ev);
        }
        modified.push(track_events);
    }
    (new_args, modified, template, records)
}

/// Writes the trailing section `expand_synth_python_args` reads: a count,
/// then (only if nonzero) the 4 template pieces and the flat record array.
/// Always safe to call even when nothing matched (writes just a zero u32),
/// consistent with this format's other optional trailing fields.
fn write_synth_python_trailer(w: &mut impl std::io::Write, template: &Option<[Vec<u8>; 4]>, records: &[[i32; 3]]) {
    write_u32(w, records.len() as u32);
    if records.is_empty() { return; }
    let t = template.as_ref().expect("records implies a template was captured");
    for piece in t {
        write_u32(w, piece.len() as u32);
        w.write_all(piece).ok();
    }
    for r in records {
        w.write_all(&r[0].to_le_bytes()).ok();
        w.write_all(&r[1].to_le_bytes()).ok();
        w.write_all(&r[2].to_le_bytes()).ok();
    }
}

/// Reverses `build_synth_and_compact_args`/`write_synth_python_trailer`: reads
/// the trailing synth section (if present — absent/zero-count is the
/// overwhelmingly common case for traces without this exact schema, and is
/// always valid, just a no-op) and, for every `SYNTH_ARGS_FLAG`-tagged
/// event in `tracks`, regenerates its literal JSON bytes and appends them
/// to a copy of `base_args`, rewriting that event's `args_off`/`args_len`
/// to point at the appended copy. Returns `None` when there's nothing to
/// expand, in which case the caller keeps using `base_args` unchanged —
/// this only costs anything for traces that actually contain the schema.
///
/// By fully resolving the sentinel scheme here, the `Trace` this hands
/// back is byte-for-byte indistinguishable from one a fresh JSON parse
/// would produce: no other code (`merge_traces`, `compact_args`, the args
/// detail panel, ...) needs any awareness that this compaction exists.
fn expand_synth_python_args(d: &[u8], mut fpos: usize, base_args: &[u8], tracks: &mut [Track]) -> Option<Vec<u8>> {
    if fpos + 4 > d.len() { return None; }
    let n_synth = u32::from_le_bytes(d[fpos..fpos + 4].try_into().ok()?) as usize;
    fpos += 4;
    if n_synth == 0 { return None; }

    let read_lp = |fpos: &mut usize| -> Option<Vec<u8>> {
        if *fpos + 4 > d.len() { return None; }
        let len = u32::from_le_bytes(d[*fpos..*fpos + 4].try_into().ok()?) as usize;
        *fpos += 4;
        if *fpos + len > d.len() { return None; }
        let v = d[*fpos..*fpos + len].to_vec();
        *fpos += len;
        Some(v)
    };
    let template = [read_lp(&mut fpos)?, read_lp(&mut fpos)?, read_lp(&mut fpos)?, read_lp(&mut fpos)?];

    let rec_size = 12usize;
    if fpos + n_synth * rec_size > d.len() { return None; }
    let records = &d[fpos..fpos + n_synth * rec_size];

    let mut buf = base_args.to_vec();
    let mut idx = 0usize;
    for t in tracks.iter_mut() {
        let mut predicted_id: i64 = 0;
        let mut first = true;
        for e in t.events.iter_mut() {
            if e.args_off & SYNTH_ARGS_FLAG == 0 { continue; }
            if idx >= n_synth { return None; }
            let off = idx * rec_size;
            let id_delta = i32::from_le_bytes(records[off..off + 4].try_into().ok()?) as i64;
            let parent_delta = i32::from_le_bytes(records[off + 4..off + 8].try_into().ok()?);
            let ev_delta = i32::from_le_bytes(records[off + 8..off + 12].try_into().ok()?) as i64;
            idx += 1;

            let id = if first { id_delta } else { predicted_id + id_delta };
            first = false;
            predicted_id = id + 1;
            let ev = id + ev_delta;
            let parent = if parent_delta == i32::MIN { None } else { Some(id - parent_delta as i64) };

            let new_off = buf.len();
            append_synth_python_args(&mut buf, &template, id, parent, ev);
            e.args_off = new_off as u32;
            e.args_len = (buf.len() - new_off).min(u16::MAX as usize) as u16;
        }
    }
    if idx != n_synth { return None; }
    Some(buf)
}

/// Writes a GPU-only, xz-compressed ".tvcache.xz" containing just the GPU
/// tracks' timing skeleton (ts/dur/name/cat/depth) with every event's args
/// stripped (args_off/args_len zeroed) — the args blob is typically the
/// bulk of a trace's size (kernel launch params etc.), and isn't "timing,"
/// which is the whole point of this export. Also drops kernel stats and
/// flow pairs (CPU<->GPU launch correlations, meaningless without the CPU
/// side).
///
/// Compressed with xz specifically, not gzip: measured directly against a
/// real 87MB args blob from a production trace, xz got ~265x vs gzip's
/// ~65x — the repeated-schema, low-cardinality-field JSON these traces
/// produce is exactly what LZMA's larger dictionary window is good at. A
/// smarter kernel-aware template/dedup scheme was also measured (same
/// data) and lost badly to plain xz (4.1x vs 272x on the same kernel's
/// launches) — xz's entropy coding already captures both the fully-
/// constant fields and the few-distinct-values-repeated-thousands-of-times
/// fields far better than a naive text-based template split can.
///
/// The whole binary blob is built in memory first, then compressed as one
/// unit — deliberately NOT the streaming mmap-based approach the main
/// cache (`save_cache`/`load_cache_from_mmap`) uses for near-instant
/// reloads: compressed bytes can't be randomly accessed/zero-copy-mapped,
/// so reading this back (`load_cache_xz`) always fully decompresses into
/// a heap buffer first. Fine for an occasional manual export/re-import,
/// wrong for the automatic reload-on-every-open path, which is why only
/// this export uses it and the main cache format is untouched.
///
/// Deliberately a standalone writer rather than sharing `save_cache`'s
/// body: that function silently swallows I/O errors (`.ok()`) since it's a
/// best-effort background cache, whereas this is a user-initiated export
/// that should surface a real error if it fails.
#[cfg(not(target_arch = "wasm32"))]
pub fn export_gpu_only(trace: &Trace, dest_path: &str) -> Result<(), String> {
    let gpu_tracks: Vec<Track> = trace.tracks.iter()
        .filter(|t| t.gpu)
        .map(|t| Track {
            label: t.label.clone(),
            gpu: true,
            events: t.events.iter().map(|e| Event { args_off: 0, args_len: 0, ..*e }).collect(),
            max_depth: t.max_depth,
            prefix_max_dur: t.prefix_max_dur.clone(),
            raw_buf_idx: 0,
        })
        .collect();
    if gpu_tracks.is_empty() {
        return Err("this trace has no GPU tracks".to_string());
    }

    let total_events: u64 = gpu_tracks.iter().map(|t| t.events.len() as u64).sum();
    let (mut min_ts, mut max_ts) = (f64::MAX, f64::MIN);
    for t in &gpu_tracks {
        for e in &t.events {
            min_ts = min_ts.min(e.ts);
            max_ts = max_ts.max(e.ts + e.dur);
        }
    }

    let dest = std::path::Path::new(dest_path);
    let dir = dest.parent().ok_or_else(|| "invalid destination path".to_string())?;
    std::fs::create_dir_all(dir).map_err(|e| format!("create dir: {e}"))?;
    let io_err = |e: std::io::Error| e.to_string();
    let mut w: Vec<u8> = Vec::new();

    w.write_all(CACHE_MAGIC).map_err(io_err)?;
    write_u32(&mut w, CACHE_VERSION);
    write_u64(&mut w, 0); // no live source file to validate freshness against
    write_u64(&mut w, 0);
    write_f64(&mut w, max_ts);
    write_f64(&mut w, min_ts);
    write_u64(&mut w, total_events);
    write_u32(&mut w, gpu_tracks.len() as u32);
    write_u32(&mut w, trace.names.len() as u32);
    write_u32(&mut w, trace.cats.len() as u32);
    write_u32(&mut w, 0); // no kernel stats in a timing-only export
    write_u32(&mut w, trace.device.len() as u32);
    write_u64(&mut w, 0); // no args buffer
    write_u32(&mut w, 0); // padding to 80

    let mut written = 80usize;
    write_strings(&mut w, &trace.names);
    written += trace.names.iter().map(|s| 4 + s.len()).sum::<usize>();
    write_strings(&mut w, &trace.cats);
    written += trace.cats.iter().map(|s| 4 + s.len()).sum::<usize>();
    write_u32(&mut w, trace.device.len() as u32);
    w.write_all(trace.device.as_bytes()).map_err(io_err)?;
    written += 4 + trace.device.len();

    for t in &gpu_tracks {
        let label = t.label.as_bytes();
        let mut hdr = [0u8; 13];
        hdr[0..2].copy_from_slice(&(label.len() as u16).to_le_bytes());
        hdr[2] = 1; // gpu
        hdr[3..5].copy_from_slice(&t.max_depth.to_le_bytes());
        hdr[5..13].copy_from_slice(&(t.events.len() as u64).to_le_bytes());
        w.write_all(&hdr).map_err(io_err)?;
        w.write_all(label).map_err(io_err)?;
        written += 13 + label.len();
    }

    pad_to_8(&mut w, written);

    // Slim columnar (SoA) event encoding instead of the interleaved 32-byte
    // AoS struct: each field gets its own contiguous run (ts delta-encoded
    // per track, from 0), and args_off/args_len are dropped from disk
    // entirely rather than merely zeroed. Measured on a real export: ~28%
    // smaller after xz than writing standard AoS `Event` bytes directly —
    // grouping same-typed, row-correlated values together (especially the
    // per-track-monotonic timestamp deltas) gives LZMA far more exploitable
    // redundancy than the interleaved layout does. `expand_gpu_export` (in
    // `load_cache_xz`) reverses this back into standard AoS bytes before
    // handing them to `load_cache_from_bytes`, so no other reader needs to
    // understand this encoding.
    //
    // Timestamps and durations are additionally stored at their real
    // precision instead of full f64: measured directly against a real
    // 114K-event export, every delta-ts was an exact multiple of 1/2048ms
    // (the trace's actual clock tick) and every duration an exact multiple
    // of 1us. Storing them as u32 instead of f64 cuts these two fields —
    // ~94% of the compressed export — by a further ~13%.
    //
    // ts uses a plain u32 tick count: 2048 is a power of two, so
    // `ticks as f64 / 2048.0` reconstructs the *exact* original f64 bits
    // (binary division by a power of two is exact), with zero precision
    // loss. dur uses integer microseconds byte-plane-split (all low bytes,
    // then all 2nd bytes, ...) instead of interleaved u32: most durations
    // are under 65536us, so the top two planes are almost entirely zero
    // and compress away for nearly free (measured ~26% smaller than plain
    // interleaved u32; plane-splitting ts made it *worse* since tick values
    // span a much wider range with no reliably-zero byte). Unlike ts, dur's
    // reconstruction (`us as f64 / 1000.0`) is not a bit-exact round trip —
    // division by a non-power-of-two can differ from the original stored
    // double by 1 ULP (~1e-9 relative) — inconsequential at microsecond
    // magnitude but real, so a re-imported dur may not be `==` the source.
    //
    // ts and dur are additionally stored in kernel-grouped order rather
    // than chronological per-track order: same kernel + same problem size
    // means near-identical (often exactly identical) durations and similar
    // inter-launch gaps, so grouping same-`name` events adjacent lets LZMA
    // find far longer matching runs (measured on a real export: ~14%
    // smaller combined). This costs nothing extra to store — the grouping
    // permutation is a stable sort by `name`, and `name` is itself stored
    // below in original chronological order, so `expand_gpu_export`
    // re-derives the identical permutation from it and scatters ts/dur
    // back to their original positions. `name`/`cat`/`depth` stay in
    // chronological order (already near-free to compress; permuting them
    // too would only cost decode-time complexity for no size benefit).
    let ts_ticks: Vec<u32> = gpu_tracks.iter().flat_map(|t| {
        let mut prev = 0.0f64;
        t.events.iter().map(move |e| {
            let ticks = ((e.ts - prev) * 2048.0).round() as u32;
            prev = e.ts;
            ticks
        })
    }).collect();
    let dur_us: Vec<u32> = gpu_tracks.iter()
        .flat_map(|t| t.events.iter().map(|e| (e.dur * 1000.0).round() as u32))
        .collect();
    let names_flat: Vec<u32> = gpu_tracks.iter()
        .flat_map(|t| t.events.iter().map(|e| e.name))
        .collect();

    let mut perm: Vec<u32> = (0..total_events as u32).collect();
    perm.sort_by_key(|&i| names_flat[i as usize]);

    for &i in &perm {
        write_u32(&mut w, ts_ticks[i as usize]);
    }
    let dur_grouped: Vec<u32> = perm.iter().map(|&i| dur_us[i as usize]).collect();
    for shift in [0u32, 8, 16, 24] {
        for v in &dur_grouped {
            w.write_all(&[((v >> shift) & 0xFF) as u8]).map_err(io_err)?;
        }
    }
    for &name in &names_flat {
        write_u32(&mut w, name);
    }
    for t in &gpu_tracks {
        for e in &t.events { write_u32(&mut w, e.cat); }
    }
    for t in &gpu_tracks {
        for e in &t.events { w.write_all(&e.depth.to_le_bytes()).map_err(io_err)?; }
    }
    for t in &gpu_tracks {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                t.prefix_max_dur.as_ptr() as *const u8,
                t.prefix_max_dur.len() * 8,
            )
        };
        w.write_all(bytes).map_err(io_err)?;
    }
    // No stats bytes (count written as 0 above) and no args buffer (len 0).
    write_u32(&mut w, 0); // no flow pairs — CPU<->GPU correlations don't apply here

    write_u32(&mut w, trace.vllm_version.len() as u32);
    w.write_all(trace.vllm_version.as_bytes()).map_err(io_err)?;
    write_u32(&mut w, trace.dist_rank as u32);
    write_u32(&mut w, trace.dist_world as u32);

    let mut encoder = xz2::write::XzEncoder::new(Vec::new(), 9);
    encoder.write_all(&w).map_err(io_err)?;
    let compressed = encoder.finish().map_err(io_err)?;

    let tmp = format!("{dest_path}.tmp");
    std::fs::write(&tmp, &compressed).map_err(|e| format!("write: {e}"))?;
    std::fs::rename(&tmp, dest).map_err(|e| format!("rename: {e}"))?;
    Ok(())
}

/// Same GPU-only, args-stripped filtering as `export_gpu_only`, but written
/// in the plain interleaved-AoS layout `load_cache_from_bytes` already
/// understands natively, then gzip- (not xz-) compressed, as `.tvcache.gz`.
/// Exists solely for sharing: the wasm build's `?src=<url>` loader has no
/// LZMA decoder and can't reverse `export_gpu_only`'s columnar/kernel-
/// grouped encoding, so a `.tvcache.xz` export can't be opened over the
/// web. gzip (`flate2`) and this plain layout both already work unchanged
/// on both platforms, at the cost of a several-times-larger file than
/// `export_gpu_only` produces (no columnar/precision/grouping tricks, and
/// gzip's smaller window loses to xz) — the right tradeoff for a one-off
/// shared link over a local re-import.
#[cfg(not(target_arch = "wasm32"))]
pub fn export_gpu_only_web(trace: &Trace, dest_path: &str) -> Result<(), String> {
    let gpu_tracks: Vec<Track> = trace.tracks.iter()
        .filter(|t| t.gpu)
        .map(|t| Track {
            label: t.label.clone(),
            gpu: true,
            events: t.events.iter().map(|e| Event { args_off: 0, args_len: 0, ..*e }).collect(),
            max_depth: t.max_depth,
            prefix_max_dur: t.prefix_max_dur.clone(),
            raw_buf_idx: 0,
        })
        .collect();
    if gpu_tracks.is_empty() {
        return Err("this trace has no GPU tracks".to_string());
    }

    let total_events: u64 = gpu_tracks.iter().map(|t| t.events.len() as u64).sum();
    let (mut min_ts, mut max_ts) = (f64::MAX, f64::MIN);
    for t in &gpu_tracks {
        for e in &t.events {
            min_ts = min_ts.min(e.ts);
            max_ts = max_ts.max(e.ts + e.dur);
        }
    }

    let dest = std::path::Path::new(dest_path);
    let dir = dest.parent().ok_or_else(|| "invalid destination path".to_string())?;
    std::fs::create_dir_all(dir).map_err(|e| format!("create dir: {e}"))?;
    let io_err = |e: std::io::Error| e.to_string();
    let mut w: Vec<u8> = Vec::new();

    w.write_all(CACHE_MAGIC).map_err(io_err)?;
    write_u32(&mut w, CACHE_VERSION);
    write_u64(&mut w, 0);
    write_u64(&mut w, 0);
    write_f64(&mut w, max_ts);
    write_f64(&mut w, min_ts);
    write_u64(&mut w, total_events);
    write_u32(&mut w, gpu_tracks.len() as u32);
    write_u32(&mut w, trace.names.len() as u32);
    write_u32(&mut w, trace.cats.len() as u32);
    write_u32(&mut w, 0);
    write_u32(&mut w, trace.device.len() as u32);
    write_u64(&mut w, 0);
    write_u32(&mut w, 0);

    let mut written = 80usize;
    write_strings(&mut w, &trace.names);
    written += trace.names.iter().map(|s| 4 + s.len()).sum::<usize>();
    write_strings(&mut w, &trace.cats);
    written += trace.cats.iter().map(|s| 4 + s.len()).sum::<usize>();
    write_u32(&mut w, trace.device.len() as u32);
    w.write_all(trace.device.as_bytes()).map_err(io_err)?;
    written += 4 + trace.device.len();

    for t in &gpu_tracks {
        let label = t.label.as_bytes();
        let mut hdr = [0u8; 13];
        hdr[0..2].copy_from_slice(&(label.len() as u16).to_le_bytes());
        hdr[2] = 1; // gpu
        hdr[3..5].copy_from_slice(&t.max_depth.to_le_bytes());
        hdr[5..13].copy_from_slice(&(t.events.len() as u64).to_le_bytes());
        w.write_all(&hdr).map_err(io_err)?;
        w.write_all(label).map_err(io_err)?;
        written += 13 + label.len();
    }

    pad_to_8(&mut w, written);

    for t in &gpu_tracks {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                t.events.as_ptr() as *const u8,
                t.events.len() * std::mem::size_of::<Event>(),
            )
        };
        w.write_all(bytes).map_err(io_err)?;
    }
    for t in &gpu_tracks {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                t.prefix_max_dur.as_ptr() as *const u8,
                t.prefix_max_dur.len() * 8,
            )
        };
        w.write_all(bytes).map_err(io_err)?;
    }
    write_u32(&mut w, 0); // no flow pairs

    write_u32(&mut w, trace.vllm_version.len() as u32);
    w.write_all(trace.vllm_version.as_bytes()).map_err(io_err)?;
    write_u32(&mut w, trace.dist_rank as u32);
    write_u32(&mut w, trace.dist_world as u32);

    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
    encoder.write_all(&w).map_err(io_err)?;
    let compressed = encoder.finish().map_err(io_err)?;

    let tmp = format!("{dest_path}.tmp");
    std::fs::write(&tmp, &compressed).map_err(|e| format!("write: {e}"))?;
    std::fs::rename(&tmp, dest).map_err(|e| format!("rename: {e}"))?;
    Ok(())
}

/// Exports the *entire* trace — every track (CPU and GPU) with args left
/// completely intact — as a wasm-loadable `.tvcache`, unlike
/// `export_gpu_only_web`'s deliberately stripped-down (GPU tracks only, no
/// args at all) sibling. "Args intact" doesn't mean skipping the
/// python_function synth-args compaction `save_cache` already uses for the
/// main cache: that encoding is fully lossless (verified byte-for-byte
/// elsewhere in this file) and dramatically smaller, so there's no reason
/// not to use it here too — it's not "scrubbing," every original arg value
/// is still recoverable exactly. Reuses the same on-disk layout `save_cache`
/// writes (so the ordinary `load_cache_from_bytes` path reads it back with
/// no special-casing) and the same zstd wrapping, which is why this needs
/// no new wasm-side support: `load_trace_from_bytes_progressive`'s
/// `.tvcache` branch already knows how to decompress it via `ruzstd`.
/// `source_size`/`source_mtime` are written as 0 (there's no live source
/// file to freshness-check an export against, same as `export_gpu_only`).
#[cfg(not(target_arch = "wasm32"))]
pub fn export_full_web(trace: &Trace, dest_path: &str) -> Result<(), String> {
    let dest = std::path::Path::new(dest_path);
    let dir = dest.parent().ok_or_else(|| "invalid destination path".to_string())?;
    std::fs::create_dir_all(dir).map_err(|e| format!("create dir: {e}"))?;
    let io_err = |e: std::io::Error| e.to_string();
    let mut w: Vec<u8> = Vec::new();

    let total_events: u64 = trace.tracks.iter().map(|t| t.events.len() as u64).sum();
    let orig_args_buf = trace.raw_bufs.first().map(|b| &b[..]).unwrap_or(&[]);
    let (args_buf, modified_events, synth_template, synth_records) =
        build_synth_and_compact_args(&trace.tracks, orig_args_buf);

    w.write_all(CACHE_MAGIC).map_err(io_err)?;
    write_u32(&mut w, CACHE_VERSION);
    write_u64(&mut w, 0);
    write_u64(&mut w, 0);
    write_f64(&mut w, trace.max_ts);
    write_f64(&mut w, trace.min_ts);
    write_u64(&mut w, total_events);
    write_u32(&mut w, trace.tracks.len() as u32);
    write_u32(&mut w, trace.names.len() as u32);
    write_u32(&mut w, trace.cats.len() as u32);
    write_u32(&mut w, trace.stats.len() as u32);
    write_u32(&mut w, trace.device.len() as u32);
    write_u64(&mut w, args_buf.len() as u64);
    write_u32(&mut w, 0);

    let mut written = 80usize;
    write_strings(&mut w, &trace.names);
    written += trace.names.iter().map(|s| 4 + s.len()).sum::<usize>();
    write_strings(&mut w, &trace.cats);
    written += trace.cats.iter().map(|s| 4 + s.len()).sum::<usize>();
    write_u32(&mut w, trace.device.len() as u32);
    w.write_all(trace.device.as_bytes()).map_err(io_err)?;
    written += 4 + trace.device.len();

    for t in &trace.tracks {
        let label = t.label.as_bytes();
        let mut hdr = [0u8; 13];
        hdr[0..2].copy_from_slice(&(label.len() as u16).to_le_bytes());
        hdr[2] = t.gpu as u8;
        hdr[3..5].copy_from_slice(&t.max_depth.to_le_bytes());
        hdr[5..13].copy_from_slice(&(t.events.len() as u64).to_le_bytes());
        w.write_all(&hdr).map_err(io_err)?;
        w.write_all(label).map_err(io_err)?;
        written += 13 + label.len();
    }

    pad_to_8(&mut w, written);

    for events in &modified_events {
        let bytes = unsafe {
            std::slice::from_raw_parts(events.as_ptr() as *const u8, events.len() * std::mem::size_of::<Event>())
        };
        w.write_all(bytes).map_err(io_err)?;
    }
    for t in &trace.tracks {
        let bytes = unsafe {
            std::slice::from_raw_parts(t.prefix_max_dur.as_ptr() as *const u8, t.prefix_max_dur.len() * 8)
        };
        w.write_all(bytes).map_err(io_err)?;
    }
    let stats_bytes = unsafe {
        std::slice::from_raw_parts(trace.stats.as_ptr() as *const u8, trace.stats.len() * std::mem::size_of::<KernelStats>())
    };
    w.write_all(stats_bytes).map_err(io_err)?;
    w.write_all(&args_buf).map_err(io_err)?;

    write_u32(&mut w, trace.flow_pairs.len() as u32);
    if !trace.flow_pairs.is_empty() {
        let flow_bytes = unsafe {
            std::slice::from_raw_parts(trace.flow_pairs.as_ptr() as *const u8, trace.flow_pairs.len() * std::mem::size_of::<FlowPair>())
        };
        w.write_all(flow_bytes).map_err(io_err)?;
    }

    write_u32(&mut w, trace.vllm_version.len() as u32);
    w.write_all(trace.vllm_version.as_bytes()).map_err(io_err)?;
    write_u32(&mut w, trace.dist_rank as u32);
    write_u32(&mut w, trace.dist_world as u32);
    write_synth_python_trailer(&mut w, &synth_template, &synth_records);

    let compressed = maybe_compress_cache(w);
    let tmp = format!("{dest_path}.tmp");
    std::fs::write(&tmp, &compressed).map_err(|e| format!("write: {e}"))?;
    std::fs::rename(&tmp, dest).map_err(|e| format!("rename: {e}"))?;
    Ok(())
}

/// Uploads `file_path` (an `export_gpu_only_web` or `export_full_web`
/// output) to a new secret GitHub gist via the locally authenticated `gh`
/// CLI, and returns a
/// `?gist=<id>` shareable link for this app's own live site. Deliberately
/// shells out rather than calling GitHub's API directly: `gh` already has
/// the user's credentials configured (`gh auth login`), so this needs no
/// browser OAuth flow or pasted personal access token — the credential
/// never passes through this app at all.
///
/// `gh gist create` refuses binary content outright (it checks for valid
/// UTF-8 client-side before ever making a request, and a gzip-compressed
/// file reliably isn't), so this creates an empty placeholder gist through
/// that command first — trivially valid as plain text — then clones it as
/// the ordinary git repository every gist actually is under the hood and
/// pushes the real binary bytes via `git`, which never cares about
/// encoding. Requires `gh auth setup-git` to have registered a git
/// credential helper for gist.github.com; this runs it defensively (a
/// harmless no-op if already done) before attempting the clone.
#[cfg(not(target_arch = "wasm32"))]
pub fn upload_gist(file_path: &str) -> Result<String, String> {
    use std::process::{Command, Stdio};

    fn run(cmd: &mut Command) -> Result<std::process::Output, String> {
        let program = cmd.get_program().to_string_lossy().into_owned();
        cmd.output().map_err(|e| format!("failed to run {program}: {e}"))
    }

    fn run_ok(cmd: &mut Command) -> Result<(), String> {
        let program = cmd.get_program().to_string_lossy().into_owned();
        let out = run(cmd)?;
        if !out.status.success() {
            return Err(format!("{program} failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
        }
        Ok(())
    }

    let file_name = std::path::Path::new(file_path).file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "invalid export file path".to_string())?
        .to_string();

    let mut create = Command::new("gh");
    create.args(["gist", "create", "--desc", "trace-viewer GPU export (shared)", "-f", "placeholder.txt", "-"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = create.spawn().map_err(|e| format!("failed to run gh (is it installed?): {e}"))?;
    {
        use std::io::Write;
        child.stdin.take().unwrap().write_all(b"placeholder\n").map_err(|e| format!("gh stdin: {e}"))?;
    }
    let create_out = child.wait_with_output().map_err(|e| format!("gh gist create: {e}"))?;
    if !create_out.status.success() {
        return Err(format!("gh gist create failed: {}", String::from_utf8_lossy(&create_out.stderr).trim()));
    }
    let gist_url = String::from_utf8_lossy(&create_out.stdout).trim().to_string();
    let gist_id = gist_url.rsplit('/').next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("couldn't parse gist id from gh output: {gist_url:?}"))?
        .to_string();

    run(&mut Command::new("gh").args(["auth", "setup-git"])).ok();

    let tmp_dir = std::env::temp_dir().join(format!("tv-gist-upload-{gist_id}"));
    std::fs::create_dir_all(&tmp_dir).map_err(|e| format!("create temp dir: {e}"))?;

    let result: Result<(), String> = (|| {
        run_ok(Command::new("git")
            .args(["clone", &format!("https://gist.github.com/{gist_id}.git"), "."])
            .current_dir(&tmp_dir))?;
        std::fs::copy(file_path, tmp_dir.join(&file_name)).map_err(|e| format!("copy export into gist checkout: {e}"))?;
        std::fs::remove_file(tmp_dir.join("placeholder.txt")).ok();
        run_ok(Command::new("git").args(["add", "-A"]).current_dir(&tmp_dir))?;
        run_ok(Command::new("git").args(["commit", "-q", "-m", "Add GPU export"]).current_dir(&tmp_dir))?;
        run_ok(Command::new("git").args(["push", "-q"]).current_dir(&tmp_dir))?;
        Ok(())
    })();
    std::fs::remove_dir_all(&tmp_dir).ok();
    result?;

    Ok(format!("https://elvircrn.github.io/tv/?gist={gist_id}"))
}

pub fn save_cache(trace: &Trace, source_path: &str, cache_dir: Option<&str>) {
    if source_path.is_empty() { return; }
    let cp = cache_path(source_path, cache_dir);
    let (src_size, src_mtime) = match source_meta(source_path) {
        Some(v) => v,
        None => return,
    };
    let mut w: Vec<u8> = Vec::new();

    let total_events: u64 = trace.tracks.iter().map(|t| t.events.len() as u64).sum();
    let orig_args_buf = trace.raw_bufs.first().map(|b| &b[..]).unwrap_or(&[]);
    let (args_buf, modified_events, synth_template, synth_records) =
        build_synth_and_compact_args(&trace.tracks, orig_args_buf);

    w.write_all(CACHE_MAGIC).ok();
    write_u32(&mut w, CACHE_VERSION);
    write_u64(&mut w, src_size);
    write_u64(&mut w, src_mtime);
    write_f64(&mut w, trace.max_ts);
    write_f64(&mut w, trace.min_ts);
    write_u64(&mut w, total_events);
    write_u32(&mut w, trace.tracks.len() as u32);
    write_u32(&mut w, trace.names.len() as u32);
    write_u32(&mut w, trace.cats.len() as u32);
    write_u32(&mut w, trace.stats.len() as u32);
    write_u32(&mut w, trace.device.len() as u32);
    write_u64(&mut w, args_buf.len() as u64);
    write_u32(&mut w, 0); // padding to 80

    let mut written = 80usize;

    write_strings(&mut w, &trace.names);
    written += trace.names.iter().map(|s| 4 + s.len()).sum::<usize>();
    write_strings(&mut w, &trace.cats);
    written += trace.cats.iter().map(|s| 4 + s.len()).sum::<usize>();
    write_u32(&mut w, trace.device.len() as u32);
    w.write_all(trace.device.as_bytes()).ok();
    written += 4 + trace.device.len();

    for t in &trace.tracks {
        let label = t.label.as_bytes();
        let mut hdr = [0u8; 13];
        hdr[0..2].copy_from_slice(&(label.len() as u16).to_le_bytes());
        hdr[2] = t.gpu as u8;
        hdr[3..5].copy_from_slice(&t.max_depth.to_le_bytes());
        hdr[5..13].copy_from_slice(&(t.events.len() as u64).to_le_bytes());
        w.write_all(&hdr).ok();
        w.write_all(label).ok();
        written += 13 + label.len();
    }

    pad_to_8(&mut w, written);
    let padded = if written % 8 != 0 { 8 - written % 8 } else { 0 };
    written += padded;
    let _ = written;

    for events in &modified_events {
        let bytes = unsafe {
            std::slice::from_raw_parts(events.as_ptr() as *const u8, events.len() * std::mem::size_of::<Event>())
        };
        w.write_all(bytes).ok();
    }

    for t in &trace.tracks {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                t.prefix_max_dur.as_ptr() as *const u8,
                t.prefix_max_dur.len() * 8,
            )
        };
        w.write_all(bytes).ok();
    }

    let stats_bytes = unsafe {
        std::slice::from_raw_parts(
            trace.stats.as_ptr() as *const u8,
            trace.stats.len() * std::mem::size_of::<KernelStats>(),
        )
    };
    w.write_all(stats_bytes).ok();

    w.write_all(&args_buf).ok();

    write_u32(&mut w, trace.flow_pairs.len() as u32);
    if !trace.flow_pairs.is_empty() {
        let flow_bytes = unsafe {
            std::slice::from_raw_parts(
                trace.flow_pairs.as_ptr() as *const u8,
                trace.flow_pairs.len() * std::mem::size_of::<FlowPair>(),
            )
        };
        w.write_all(flow_bytes).ok();
    }

    // Optional trailing fields (each independently bounds-checked on read, so
    // older caches that lack them still load): vLLM version, then rank/world,
    // then the synth-python-args section (see build_synth_and_compact_args).
    write_u32(&mut w, trace.vllm_version.len() as u32);
    w.write_all(trace.vllm_version.as_bytes()).ok();
    write_u32(&mut w, trace.dist_rank as u32);
    write_u32(&mut w, trace.dist_world as u32);
    write_synth_python_trailer(&mut w, &synth_template, &synth_records);

    let compressed = maybe_compress_cache(w);
    let tmp = format!("{cp}.tmp");
    if std::fs::write(&tmp, &compressed).is_ok() {
        std::fs::rename(&tmp, &cp).ok();
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_cache(source_path: &str, cache_dir: Option<&str>) -> Option<Trace> {
    let cp = cache_path(source_path, cache_dir);
    let (src_size, src_mtime) = source_meta(source_path)?;
    let d = read_and_decompress_cache(&cp)?;
    if d.len() < 80 || &d[0..4] != CACHE_MAGIC { return None; }
    let r32 = |off: usize| u32::from_le_bytes(d[off..off + 4].try_into().unwrap());
    let r64 = |off: usize| u64::from_le_bytes(d[off..off + 8].try_into().unwrap());
    if r32(4) > CACHE_VERSION { return None; }
    if r64(8) != src_size || r64(16) != src_mtime { return None; }
    load_cache_from_bytes(&d)
}

// No persistent disk cache on the web build (no filesystem) — every open
// re-parses from scratch there.
#[cfg(target_arch = "wasm32")]
pub fn load_cache(_source_path: &str, _cache_dir: Option<&str>) -> Option<Trace> { None }

// Loads a `.tvcache` binary blob already in memory instead of opening a
// path directly — the shared parser every cache reader (wasm's File API
// reads, `load_cache_xz`/`load_cache_gz`'s decompressed export bytes, and
// native's own `load_cache`/`load_cache_direct`/`load_merged_cache`, all
// via `read_and_decompress_cache`) ultimately hands bytes to. Reads from a
// plain byte slice and copies the args range into `ArgsBuf::Heap`.
// Sequential (not `std::thread::scope`) since real threads panic on wasm32
// without the atomics build from the real-threading phase — harmless on
// native too, this isn't a hot path relative to the parsing already done
// to build the Trace being cached in the first place.
pub fn load_cache_from_bytes(d: &[u8]) -> Option<Trace> {
    if d.len() < 80 || &d[0..4] != CACHE_MAGIC { return None; }

    let r32 = |off: usize| u32::from_le_bytes(d[off..off + 4].try_into().unwrap());
    let r64 = |off: usize| u64::from_le_bytes(d[off..off + 8].try_into().unwrap());
    let rf64 = |off: usize| f64::from_le_bytes(d[off..off + 8].try_into().unwrap());
    if r32(4) > CACHE_VERSION { return None; }

    let max_ts = rf64(24);
    let min_ts = rf64(32);
    let total_events = r64(40) as usize;
    let n_tracks = r32(48) as usize;
    let n_names = r32(52) as usize;
    let n_cats = r32(56) as usize;
    let n_stats = r32(60) as usize;
    let _device_len = r32(64) as usize;
    let args_len = r64(68) as usize;

    let mut pos = 80usize;

    let read_strings = |pos: &mut usize, count: usize| -> Option<Vec<String>> {
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            if *pos + 4 > d.len() { return None; }
            let len = u32::from_le_bytes(d[*pos..*pos + 4].try_into().unwrap()) as usize;
            *pos += 4;
            if *pos + len > d.len() { return None; }
            v.push(String::from_utf8_lossy(&d[*pos..*pos + len]).into_owned());
            *pos += len;
        }
        Some(v)
    };

    let names = read_strings(&mut pos, n_names)?;
    let cats = read_strings(&mut pos, n_cats)?;

    if pos + 4 > d.len() { return None; }
    let dev_len = u32::from_le_bytes(d[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    if pos + dev_len > d.len() { return None; }
    let device = String::from_utf8_lossy(&d[pos..pos + dev_len]).into_owned();
    pos += dev_len;

    struct TrackHdr { label: String, gpu: bool, max_depth: u16, event_count: usize }
    let mut track_hdrs: Vec<TrackHdr> = Vec::with_capacity(n_tracks);
    let mut total_check: usize = 0;
    for _ in 0..n_tracks {
        if pos + 13 > d.len() { return None; }
        let label_len = u16::from_le_bytes(d[pos..pos + 2].try_into().unwrap()) as usize;
        let gpu = d[pos + 2] != 0;
        let max_depth = u16::from_le_bytes(d[pos + 3..pos + 5].try_into().unwrap());
        let event_count = u64::from_le_bytes(d[pos + 5..pos + 13].try_into().unwrap()) as usize;
        pos += 13;
        if pos + label_len > d.len() { return None; }
        let label = String::from_utf8_lossy(&d[pos..pos + label_len]).into_owned();
        pos += label_len;
        total_check += event_count;
        track_hdrs.push(TrackHdr { label, gpu, max_depth, event_count });
    }
    if total_check != total_events { return None; }

    if pos % 8 != 0 { pos += 8 - pos % 8; }

    let ev_size = std::mem::size_of::<Event>();
    let events_bytes = total_events * ev_size;
    if pos + events_bytes > d.len() { return None; }
    let all_events: &[Event] = unsafe {
        std::slice::from_raw_parts(d[pos..].as_ptr() as *const Event, total_events)
    };
    pos += events_bytes;

    let pmd_bytes = total_events * 8;
    if pos + pmd_bytes > d.len() { return None; }
    let all_pmd: &[f64] = unsafe {
        std::slice::from_raw_parts(d[pos..].as_ptr() as *const f64, total_events)
    };
    pos += pmd_bytes;

    let stats_size = std::mem::size_of::<KernelStats>();
    let stats_bytes = n_stats * stats_size;
    if pos + stats_bytes > d.len() { return None; }
    let stats: Vec<KernelStats> = unsafe {
        std::slice::from_raw_parts(d[pos..].as_ptr() as *const KernelStats, n_stats)
    }.to_vec();
    pos += stats_bytes;

    if pos + args_len > d.len() { return None; }
    let args_offset = pos;
    let after_args = pos + args_len;

    let mut offsets = Vec::with_capacity(n_tracks);
    let mut ev_off = 0usize;
    for hdr in &track_hdrs {
        offsets.push(ev_off);
        ev_off += hdr.event_count;
    }

    let mut tracks: Vec<Track> = track_hdrs.into_iter().zip(offsets).map(|(hdr, off)| {
        let n = hdr.event_count;
        Track {
            label: hdr.label,
            gpu: hdr.gpu,
            events: all_events[off..off + n].to_vec(),
            max_depth: hdr.max_depth,
            prefix_max_dur: all_pmd[off..off + n].to_vec(),
            raw_buf_idx: 0,
        }
    }).collect();

    let fp_size = std::mem::size_of::<FlowPair>();
    let mut flow_pairs = Vec::new();
    let mut fpos = after_args;
    if fpos + 4 <= d.len() {
        let n_flows = u32::from_le_bytes(d[fpos..fpos + 4].try_into().unwrap()) as usize;
        fpos += 4;
        if fpos + n_flows * fp_size <= d.len() {
            flow_pairs.reserve(n_flows);
            for i in 0..n_flows {
                let off = fpos + i * fp_size;
                let src_track = u32::from_le_bytes(d[off..off+4].try_into().unwrap());
                let dst_track = u32::from_le_bytes(d[off+4..off+8].try_into().unwrap());
                let src_ts = f64::from_le_bytes(d[off+8..off+16].try_into().unwrap());
                let dst_ts = f64::from_le_bytes(d[off+16..off+24].try_into().unwrap());
                flow_pairs.push(FlowPair { src_track, dst_track, src_ts, dst_ts });
            }
            fpos += n_flows * fp_size;
        }
    }

    let mut vllm_version = String::new();
    if fpos + 4 <= d.len() {
        let vlen = u32::from_le_bytes(d[fpos..fpos + 4].try_into().unwrap()) as usize;
        fpos += 4;
        if fpos + vlen <= d.len() {
            vllm_version = String::from_utf8_lossy(&d[fpos..fpos + vlen]).into_owned();
            fpos += vlen;
        }
    }

    let (mut dist_rank, mut dist_world) = (-1i32, 0i32);
    if fpos + 8 <= d.len() {
        dist_rank = i32::from_le_bytes(d[fpos..fpos + 4].try_into().unwrap());
        dist_world = i32::from_le_bytes(d[fpos + 4..fpos + 8].try_into().unwrap());
        fpos += 8;
    }

    let base_args = d[args_offset..args_offset + args_len].to_vec();
    let synth_expanded = expand_synth_python_args(d, fpos, &base_args, &mut tracks);
    let raw_bufs = vec![Arc::new(ArgsBuf::Heap(synth_expanded.unwrap_or(base_args)))];

    Some(Trace {
        tracks, names, cats, raw_bufs,
        stats, max_ts, min_ts, total_events, device, vllm_version, dist_rank, dist_world, flow_pairs,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn load_cache_direct(cache_path: &str) -> Option<Trace> {
    let d = read_and_decompress_cache(cache_path)?;
    load_cache_from_bytes(&d)
}

#[cfg(target_arch = "wasm32")]
fn load_cache_direct(_cache_path: &str) -> Option<Trace> { None }

/// Reverses `export_gpu_only`'s slim-SoA + per-track delta-encoded-timestamp
/// event encoding back into the standard AoS `Event` layout that
/// `load_cache_from_bytes` expects. Re-derives track boundaries from the
/// (unchanged) header/track-header section, then rebuilds a full
/// 32-byte-per-event array (args_off/args_len reconstituted as 0, matching
/// what `export_gpu_only` zeroed before ever writing them). Returns `None`
/// on any malformed input, matching `load_cache_from_bytes`'s own
/// bounds-checked style.
#[cfg(not(target_arch = "wasm32"))]
fn expand_gpu_export(d: &[u8]) -> Option<Vec<u8>> {
    if d.len() < 80 || &d[0..4] != CACHE_MAGIC { return None; }
    let r32 = |off: usize| u32::from_le_bytes(d[off..off + 4].try_into().unwrap());
    let r64 = |off: usize| u64::from_le_bytes(d[off..off + 8].try_into().unwrap());
    if r32(4) > CACHE_VERSION { return None; }

    let total_events = r64(40) as usize;
    let n_tracks = r32(48) as usize;
    let n_names = r32(52) as usize;
    let n_cats = r32(56) as usize;

    let mut pos = 80usize;
    let skip_strings = |pos: &mut usize, count: usize| -> Option<()> {
        for _ in 0..count {
            if *pos + 4 > d.len() { return None; }
            let len = u32::from_le_bytes(d[*pos..*pos + 4].try_into().unwrap()) as usize;
            *pos += 4;
            if *pos + len > d.len() { return None; }
            *pos += len;
        }
        Some(())
    };
    skip_strings(&mut pos, n_names)?;
    skip_strings(&mut pos, n_cats)?;

    if pos + 4 > d.len() { return None; }
    let dev_len = r32(pos) as usize;
    pos += 4 + dev_len;
    if pos > d.len() { return None; }

    let mut track_event_counts = Vec::with_capacity(n_tracks);
    let mut total_check = 0usize;
    for _ in 0..n_tracks {
        if pos + 13 > d.len() { return None; }
        let label_len = u16::from_le_bytes(d[pos..pos + 2].try_into().unwrap()) as usize;
        let event_count = u64::from_le_bytes(d[pos + 5..pos + 13].try_into().unwrap()) as usize;
        pos += 13 + label_len;
        if pos > d.len() { return None; }
        total_check += event_count;
        track_event_counts.push(event_count);
    }
    if total_check != total_events { return None; }

    let header_end = pos;
    let mut events_start = header_end;
    if events_start % 8 != 0 { events_start += 8 - events_start % 8; }

    // Mirrors export_gpu_only's layout: ts as plain u32 ticks of 1/2048ms,
    // dur as u32 microseconds byte-plane-split (4 separate n-byte runs,
    // least-significant plane first) — both stored in kernel-grouped
    // (stable-sorted-by-name) order rather than chronological order. Read
    // `name` first (it's the one array kept in original chronological
    // order) and re-derive the identical stable-sort-by-name permutation
    // the writer used, then scatter ts/dur back into chronological order
    // through it — no permutation table is stored on disk.
    let n = total_events;
    let ts_off = events_start;
    let dur_off = ts_off + n * 4;
    let name_off = dur_off + n * 4;
    let cat_off = name_off + n * 4;
    let depth_off = cat_off + n * 4;
    let events_end = depth_off + n * 2;
    if events_end > d.len() { return None; }

    let names_flat: Vec<u32> = (0..n)
        .map(|i| u32::from_le_bytes(d[name_off + i * 4..name_off + i * 4 + 4].try_into().unwrap()))
        .collect();
    let mut perm: Vec<u32> = (0..n as u32).collect();
    perm.sort_by_key(|&i| names_flat[i as usize]);

    let mut ts_ticks = vec![0u32; n];
    let mut dur_us = vec![0u32; n];
    for (j, &orig_i) in perm.iter().enumerate() {
        let orig_i = orig_i as usize;
        ts_ticks[orig_i] = u32::from_le_bytes(d[ts_off + j * 4..ts_off + j * 4 + 4].try_into().unwrap());
        dur_us[orig_i] = (d[dur_off + j] as u32)
            | ((d[dur_off + n + j] as u32) << 8)
            | ((d[dur_off + 2 * n + j] as u32) << 16)
            | ((d[dur_off + 3 * n + j] as u32) << 24);
    }

    let ev_size = std::mem::size_of::<Event>();
    let mut aos = vec![0u8; n * ev_size];
    let mut idx = 0usize;
    for &ec in &track_event_counts {
        let mut prev = 0.0f64;
        for i in 0..ec {
            let e_i = idx + i;
            let ts = prev + (ts_ticks[e_i] as f64) / 2048.0;
            prev = ts;
            let dur = (dur_us[e_i] as f64) / 1000.0;
            let name = names_flat[e_i];
            let cat = u32::from_le_bytes(d[cat_off + e_i * 4..cat_off + e_i * 4 + 4].try_into().unwrap());
            let depth = u16::from_le_bytes(d[depth_off + e_i * 2..depth_off + e_i * 2 + 2].try_into().unwrap());
            let ev = Event { ts, dur, name, cat, args_off: 0, depth, args_len: 0 };
            let dst = &mut aos[e_i * ev_size..e_i * ev_size + ev_size];
            dst.copy_from_slice(unsafe {
                std::slice::from_raw_parts(&ev as *const Event as *const u8, ev_size)
            });
        }
        idx += ec;
    }

    let mut out = Vec::with_capacity(events_start + aos.len() + (d.len() - events_end));
    out.extend_from_slice(&d[..header_end]);
    while out.len() % 8 != 0 { out.push(0); }
    out.extend_from_slice(&aos);
    out.extend_from_slice(&d[events_end..]);
    Some(out)
}

/// Reads an xz-compressed export (see `export_gpu_only`) — always a full
/// decompress into memory first, unlike `load_cache_direct`'s mmap-based
/// zero-copy path, since compressed bytes can't be randomly accessed.
#[cfg(not(target_arch = "wasm32"))]
fn load_cache_xz(cache_path: &str) -> Option<Trace> {
    let file = std::fs::File::open(cache_path).ok()?;
    let mut decoder = xz2::read::XzDecoder::new(BufReader::new(file));
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf).ok()?;
    let expanded = expand_gpu_export(&buf)?;
    load_cache_from_bytes(&expanded)
}

/// Reads an `export_gpu_only_web` output (`.tvcache.gz`) — plain standard
/// AoS layout under gzip, so unlike `load_cache_xz` this needs no inverse
/// transform before `load_cache_from_bytes`.
#[cfg(not(target_arch = "wasm32"))]
fn load_cache_gz(cache_path: &str) -> Option<Trace> {
    let file = std::fs::File::open(cache_path).ok()?;
    let mut decoder = flate2::read::GzDecoder::new(BufReader::new(file));
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf).ok()?;
    load_cache_from_bytes(&buf)
}

fn merged_cache_hash(cache_dir: &str) -> u64 {
    let mut entries: Vec<(String, u64)> = Vec::new();
    if let Ok(dir) = std::fs::read_dir(cache_dir) {
        for entry in dir.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".tvcache") && name != "_merged.tvcache" {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                entries.push((name, size));
            }
        }
    }
    entries.sort();
    let mut h: u64 = 0xcbf29ce484222325;
    for (name, size) in &entries {
        for &b in name.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        for &b in &size.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn save_merged_cache(trace: &Trace, cache_dir: &str) {
    let cp = format!("{cache_dir}/_merged.tvcache");
    let hash = merged_cache_hash(cache_dir);
    let mut w: Vec<u8> = Vec::new();

    let total_events: u64 = trace.tracks.iter().map(|t| t.events.len() as u64).sum();
    let orig_args_buf = trace.raw_bufs.first().map(|b| &b[..]).unwrap_or(&[]);
    let (args_buf, modified_events, synth_template, synth_records) =
        build_synth_and_compact_args(&trace.tracks, orig_args_buf);

    w.write_all(CACHE_MAGIC).ok();
    write_u32(&mut w, CACHE_VERSION);
    write_u64(&mut w, hash);
    write_u64(&mut w, 0);
    write_f64(&mut w, trace.max_ts);
    write_f64(&mut w, trace.min_ts);
    write_u64(&mut w, total_events);
    write_u32(&mut w, trace.tracks.len() as u32);
    write_u32(&mut w, trace.names.len() as u32);
    write_u32(&mut w, trace.cats.len() as u32);
    write_u32(&mut w, trace.stats.len() as u32);
    write_u32(&mut w, trace.device.len() as u32);
    write_u64(&mut w, args_buf.len() as u64);
    write_u32(&mut w, 0);

    let mut written = 80usize;

    write_strings(&mut w, &trace.names);
    written += trace.names.iter().map(|s| 4 + s.len()).sum::<usize>();
    write_strings(&mut w, &trace.cats);
    written += trace.cats.iter().map(|s| 4 + s.len()).sum::<usize>();
    write_u32(&mut w, trace.device.len() as u32);
    w.write_all(trace.device.as_bytes()).ok();
    written += 4 + trace.device.len();

    for t in &trace.tracks {
        let label = t.label.as_bytes();
        let mut hdr = [0u8; 13];
        hdr[0..2].copy_from_slice(&(label.len() as u16).to_le_bytes());
        hdr[2] = t.gpu as u8;
        hdr[3..5].copy_from_slice(&t.max_depth.to_le_bytes());
        hdr[5..13].copy_from_slice(&(t.events.len() as u64).to_le_bytes());
        w.write_all(&hdr).ok();
        w.write_all(label).ok();
        written += 13 + label.len();
    }

    pad_to_8(&mut w, written);

    for events in &modified_events {
        let bytes = unsafe {
            std::slice::from_raw_parts(events.as_ptr() as *const u8, events.len() * std::mem::size_of::<Event>())
        };
        w.write_all(bytes).ok();
    }

    for t in &trace.tracks {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                t.prefix_max_dur.as_ptr() as *const u8,
                t.prefix_max_dur.len() * 8,
            )
        };
        w.write_all(bytes).ok();
    }

    let stats_bytes = unsafe {
        std::slice::from_raw_parts(
            trace.stats.as_ptr() as *const u8,
            trace.stats.len() * std::mem::size_of::<KernelStats>(),
        )
    };
    w.write_all(stats_bytes).ok();

    w.write_all(&args_buf).ok();

    write_u32(&mut w, trace.flow_pairs.len() as u32);
    if !trace.flow_pairs.is_empty() {
        let flow_bytes = unsafe {
            std::slice::from_raw_parts(
                trace.flow_pairs.as_ptr() as *const u8,
                trace.flow_pairs.len() * std::mem::size_of::<FlowPair>(),
            )
        };
        w.write_all(flow_bytes).ok();
    }

    // Optional trailing fields (each independently bounds-checked on read, so
    // older caches that lack them still load): vLLM version, then rank/world,
    // then the synth-python-args section (see build_synth_and_compact_args).
    write_u32(&mut w, trace.vllm_version.len() as u32);
    w.write_all(trace.vllm_version.as_bytes()).ok();
    write_u32(&mut w, trace.dist_rank as u32);
    write_u32(&mut w, trace.dist_world as u32);
    write_synth_python_trailer(&mut w, &synth_template, &synth_records);

    let compressed = maybe_compress_cache(w);
    let tmp = format!("{cp}.tmp");
    if std::fs::write(&tmp, &compressed).is_ok() {
        std::fs::rename(&tmp, &cp).ok();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn load_merged_cache(cache_dir: &str) -> Option<Trace> {
    let cp = format!("{cache_dir}/_merged.tvcache");
    let d = read_and_decompress_cache(&cp)?;
    if d.len() < 16 { return None; }
    let stored_hash = u64::from_le_bytes(d[8..16].try_into().unwrap());
    let current_hash = merged_cache_hash(cache_dir);
    if stored_hash != current_hash { return None; }
    load_cache_from_bytes(&d)
}

#[cfg(target_arch = "wasm32")]
fn load_merged_cache(_cache_dir: &str) -> Option<Trace> { None }

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

pub fn load_trace(path: &str, counter: &Arc<AtomicUsize>, max_parse_threads: usize, cache_dir: Option<&str>) -> Result<Trace, String> {
    let t0 = Instant::now();

    #[cfg(not(target_arch = "wasm32"))]
    if path.ends_with(".tvcache.xz") {
        return load_cache_xz(path)
            .ok_or_else(|| "invalid or corrupt .tvcache.xz file".to_string())
            .inspect(|t| eprintln!("  cache (xz): {:.2}s ({} events)", t0.elapsed().as_secs_f64(), t.total_events));
    }
    #[cfg(not(target_arch = "wasm32"))]
    if path.ends_with(".tvcache.gz") {
        return load_cache_gz(path)
            .ok_or_else(|| "invalid or corrupt .tvcache.gz file".to_string())
            .inspect(|t| eprintln!("  cache (gz): {:.2}s ({} events)", t0.elapsed().as_secs_f64(), t.total_events));
    }
    if path.ends_with(".tvcache") {
        return load_cache_direct(path)
            .ok_or_else(|| "invalid or corrupt .tvcache file".to_string())
            .inspect(|t| eprintln!("  cache: {:.2}s ({} events)", t0.elapsed().as_secs_f64(), t.total_events));
    }

    if let Some(trace) = load_cache(path, cache_dir) {
        eprintln!("  cache: {:.2}s ({} events)", t0.elapsed().as_secs_f64(), trace.total_events);
        return Ok(trace);
    }

    let (raw, chunks, n_chunks) = decompress_parse(path, counter, max_parse_threads, &t0)?;
    let mut trace = build_trace(raw, chunks, n_chunks, &t0)?;
    compact_args(&mut trace);

    save_cache(&trace, path, cache_dir);

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
        vllm_version: t.vllm_version.clone(),
        dist_rank: t.dist_rank,
        dist_world: t.dist_world,
        flow_pairs: t.flow_pairs.clone(),
    }
}

fn send_progressive(trace: Trace, tx: &std::sync::mpsc::Sender<Result<Trace, String>>, t0: &Instant, source_path: &str, cache_dir: Option<&str>) {
    eprintln!("  ready: {:.2}s ({} tracks)", t0.elapsed().as_secs_f64(), trace.tracks.len());
    let mut compact_trace = clone_trace(&trace);
    let _ = tx.send(Ok(trace));

    compact_args(&mut compact_trace);
    save_cache(&compact_trace, source_path, cache_dir);
    let args_bytes: usize = compact_trace.raw_bufs.iter().map(|b| b.len()).sum();
    eprintln!("  compact: {:.2}s (args={}MB)", t0.elapsed().as_secs_f64(), args_bytes / 1024 / 1024);
    let _ = tx.send(Ok(compact_trace));
}

pub fn load_trace_progressive(
    path: &str, counter: &Arc<AtomicUsize>, max_parse_threads: usize,
    tx: &std::sync::mpsc::Sender<Result<Trace, String>>,
    cache_dir: Option<&str>,
) {
    let t0 = Instant::now();

    #[cfg(not(target_arch = "wasm32"))]
    if path.ends_with(".tvcache.xz") {
        match load_cache_xz(path) {
            Some(trace) => {
                eprintln!("  cache (xz): {:.2}s ({} events)", t0.elapsed().as_secs_f64(), trace.total_events);
                let _ = tx.send(Ok(trace));
            }
            None => { let _ = tx.send(Err("invalid or corrupt .tvcache.xz file".into())); }
        }
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if path.ends_with(".tvcache.gz") {
        match load_cache_gz(path) {
            Some(trace) => {
                eprintln!("  cache (gz): {:.2}s ({} events)", t0.elapsed().as_secs_f64(), trace.total_events);
                let _ = tx.send(Ok(trace));
            }
            None => { let _ = tx.send(Err("invalid or corrupt .tvcache.gz file".into())); }
        }
        return;
    }
    if path.ends_with(".tvcache") {
        match load_cache_direct(path) {
            Some(trace) => {
                eprintln!("  cache: {:.2}s ({} events)", t0.elapsed().as_secs_f64(), trace.total_events);
                let _ = tx.send(Ok(trace));
            }
            None => { let _ = tx.send(Err("invalid or corrupt .tvcache file".into())); }
        }
        return;
    }

    if let Some(trace) = load_cache(path, cache_dir) {
        eprintln!("  cache: {:.2}s ({} events)", t0.elapsed().as_secs_f64(), trace.total_events);
        let _ = tx.send(Ok(trace));
        return;
    }

    let (raw, chunks, n_chunks) = match decompress_parse(path, counter, max_parse_threads, &t0) {
        Ok(v) => v,
        Err(e) => { let _ = tx.send(Err(e)); return; }
    };
    match build_trace(raw, chunks, n_chunks, &t0) {
        Ok(trace) => send_progressive(trace, tx, &t0, path, cache_dir),
        Err(e) => { let _ = tx.send(Err(e)); }
    }
}

pub(crate) fn is_trace_file(name: &str) -> bool {
    name.ends_with(".json") || name.ends_with(".json.gz")
        || name.ends_with(".tar.gz") || name.ends_with(".tgz")
        || name.ends_with(".tvcache") || name.ends_with(".tvcache.xz") || name.ends_with(".tvcache.gz")
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
                    let dirname = ep.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if dirname == ".tvcache" || dirname.ends_with(".tvcache") { continue; }
                    dirs.push(ep);
                } else if let Some(name) = ep.file_name().and_then(|n| n.to_str()) {
                    if !name.ends_with(".tvcache") && is_trace_file(name) {
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

/// Numeric rank parsed back out of a merged track's `"  Rank {N} ..."`
/// label prefix (see the labeling loop below) — used to sort tracks by
/// actual rank *value* instead of lexicographically by the whole label
/// string, which would put "Rank 10" before "Rank 2" (both start with the
/// character '1' < '2' before any digit count is considered). Anything
/// that doesn't match the expected prefix sorts last rather than panicking
/// (shouldn't happen — every track here was just labeled by this same
/// function — but this isn't a hot path worth an `unwrap`).
fn merged_track_rank(label: &str) -> u32 {
    label.strip_prefix("  Rank ")
        .and_then(|rest| rest.split(' ').next())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(u32::MAX)
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
    let mut vllm_version = String::new();
    // A merged trace spans many ranks, so there's no single rank id; keep the
    // shared world_size (from the first rank that reports one) for context.
    let mut dist_world = 0i32;

    let mut remap_info: Vec<(Vec<u32>, Vec<u32>, f64)> = Vec::with_capacity(traces.len());
    let mut all_raw_bufs: Vec<Arc<ArgsBuf>> = Vec::new();

    for (_, trace) in &traces {
        let time_offset = trace.min_ts - global_min;

        let name_remap = merge_intern_direct(&mut names, &mut name_idx, &trace.names);
        let cat_remap = merge_intern_direct(&mut cats, &mut cat_idx, &trace.cats);

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

    let remapped: Vec<_> = traces.into_iter().zip(remap_info).enumerate()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(trace_idx, ((rank, mut trace), (name_remap, cat_remap, time_offset)))| {
            let raw_buf_idx = trace_idx as u8;
            let mut max_ts: f64 = 0.0;
            let mut total_events = 0usize;
            let mut local_dur: HashMap<u32, Vec<f64>> = HashMap::new();
            for track in &mut trace.tracks {
                track.label = format!("  Rank {} {}", rank, track.label);
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

    // Stable sort by rank *number* only (not the whole label string — see
    // merged_track_rank): tracks already arrive in a sensible per-rank
    // order (GPU-first, busiest first — each individual trace's own sort
    // in build_trace), and a stable sort preserves that within each rank
    // while fixing the cross-rank order to be numeric instead of
    // lexicographic.
    let mut sort_perm: Vec<usize> = (0..all_tracks.len()).collect();
    sort_perm.sort_by_key(|&i| merged_track_rank(&all_tracks[i].label));
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
        tracks: sorted_tracks, names, cats, raw_bufs: all_raw_bufs, stats,
        max_ts, min_ts: global_min, total_events, device, vllm_version,
        dist_rank: -1, dist_world, flow_pairs: all_flow_pairs,
    };
    compact_args(&mut trace);
    trace
}

pub fn load_multi_progressive(
    rank_paths: Vec<(usize, String)>, counter: &Arc<AtomicUsize>, tpf: usize,
    tx: &std::sync::mpsc::Sender<Result<Trace, String>>,
    cache_dir: Option<&str>, bypass_cache: bool,
) {
    let t0 = Instant::now();

    // On a watch-triggered reload (bypass_cache), skip the merged cache: its hash
    // is keyed off the per-file .tvcache names+sizes, which don't reflect a
    // source-file edit yet, so it would keep serving stale data. The per-file
    // load_trace below still uses its mtime-validated caches, so unchanged ranks
    // stay fast while the edited one is re-read and the merge re-saved.
    if !bypass_cache {
        if let Some(cd) = cache_dir {
            if let Some(trace) = load_merged_cache(cd) {
                eprintln!("  merged cache: {:.2}s ({} events)", t0.elapsed().as_secs_f64(), trace.total_events);
                let _ = tx.send(Ok(trace));
                return;
            }
        }
    }

    let cd = cache_dir.map(|s| s.to_string());
    // Preserve the previous std::thread::scope + JoinHandle::join().ok()
    // behavior: a rank whose load panics is dropped (with a message) rather
    // than taking the whole multi-rank load down with it.
    let results: Vec<(usize, Result<Trace, String>)> = rank_paths.par_iter().filter_map(|(rank, path)| {
        let r = *rank;
        let ctr = counter.clone();
        let cd_ref = cd.as_deref();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load_trace(path, &ctr, tpf, cd_ref))) {
            Ok(result) => Some((r, result)),
            Err(e) => {
                let msg = e.downcast_ref::<&str>().map(|s| s.to_string())
                    .or_else(|| e.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_string());
                eprintln!("  rank {r} load thread panicked: {msg}");
                None
            }
        }
    }).collect();
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
    let trace = merge_traces(traces);
    eprintln!("  merged: {:.2}s ({} events, {} flow_pairs)", t0.elapsed().as_secs_f64(), trace.total_events, trace.flow_pairs.len());
    if let Some(cd) = cache_dir {
        save_merged_cache(&trace, cd);
    }
    let _ = tx.send(Ok(trace));
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
    trace.raw_bufs = vec![Arc::new(ArgsBuf::Heap(compact))];
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
        #[cfg(not(target_arch = "wasm32"))]
        {
            let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("mmap {path}: {e}"))?;
            mmap.advise(memmap2::Advice::Sequential).ok();
            Ok(RawData::Mmap(mmap))
        }
        #[cfg(target_arch = "wasm32")]
        {
            Ok(RawData::Vec(std::fs::read(path).map_err(|e| format!("{path}: {e}"))?))
        }
    }
}

/// `read_bytes`'s decompression dispatch, but for a file already read into
/// memory (e.g. via the browser's File API, which hands back bytes with no
/// filesystem path attached) instead of a filesystem path to open. Dispatches
/// on `name` (typically `File::name()`) the same way `read_bytes` dispatches
/// on the path's suffix.
#[cfg(target_arch = "wasm32")]
fn read_bytes_named(name: &str, raw: Vec<u8>) -> Result<RawData, String> {
    if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        let gz = flate2::read::GzDecoder::new(std::io::Cursor::new(raw));
        let mut archive = tar::Archive::new(gz);
        let entries = archive.entries().map_err(|e| format!("tar {name}: {e}"))?;
        for entry in entries {
            let mut entry = entry.map_err(|e| format!("tar entry: {e}"))?;
            let ename = entry.path().map_err(|e| format!("tar path: {e}"))?.to_string_lossy().to_string();
            if ename.ends_with(".json.gz") {
                let buf = read_gz_tolerant(&mut entry).map_err(|e| format!("{ename}: {e}"))?;
                eprintln!("  extracted: {ename}");
                return Ok(RawData::Vec(buf));
            } else if ename.ends_with(".json") {
                let mut buf = Vec::new();
                match entry.read_to_end(&mut buf) {
                    Ok(_) => {}
                    Err(e) if buf.is_empty() => return Err(format!("read {ename}: {e}")),
                    Err(e) => eprintln!("  truncated tar entry after {} bytes: {e}", buf.len()),
                }
                eprintln!("  extracted: {ename}");
                return Ok(RawData::Vec(buf));
            }
        }
        Err(format!("no .json or .json.gz file found in {name}"))
    } else if name.ends_with(".gz") {
        let buf = read_gz_tolerant(&raw[..]).map_err(|e| format!("{name}: {e}"))?;
        Ok(RawData::Vec(buf))
    } else {
        Ok(RawData::Vec(raw))
    }
}

/// wasm equivalent of `load_trace_progressive`: takes bytes already read into
/// memory (there's no filesystem path to open on wasm32) instead of a path.
/// No disk cache to check/write (no filesystem in the browser — see
/// `load_cache`'s wasm stub) and no "instant preview then compact" double
/// send (there's no real background thread on wasm yet — see
/// `state::spawn_load_job` — so the job already runs to completion before
/// `poll_loading` gets a chance to observe an intermediate state).
#[cfg(target_arch = "wasm32")]
pub fn load_trace_from_bytes_progressive(
    name: &str, raw_input: Vec<u8>, counter: &Arc<AtomicUsize>,
    tx: &std::sync::mpsc::Sender<Result<Trace, String>>,
) {
    let t0 = Instant::now();

    if name.ends_with(".tvcache.gz") {
        let mut decoder = flate2::read::GzDecoder::new(&raw_input[..]);
        let mut buf = Vec::new();
        match decoder.read_to_end(&mut buf).ok().and_then(|_| load_cache_from_bytes(&buf)) {
            Some(trace) => {
                eprintln!("  cache (gz): {:.2}s ({} events)", t0.elapsed().as_secs_f64(), trace.total_events);
                let _ = tx.send(Ok(trace));
            }
            None => { let _ = tx.send(Err(format!("{name}: invalid or corrupt .tvcache.gz file"))); }
        }
        return;
    }
    if name.ends_with(".tvcache") {
        // Same zstd-detection `read_and_decompress_cache` does natively,
        // via the pure-Rust `ruzstd` decoder instead of the C-backed `zstd`
        // crate (no C toolchain available for a wasm32-unknown-unknown
        // cross-compile) — an uncompressed legacy cache passes through
        // unchanged, same as on native.
        let decompressed = if raw_input.len() >= 4 && raw_input[0..4] == ZSTD_FRAME_MAGIC {
            let mut out = Vec::new();
            match ruzstd::StreamingDecoder::new(&raw_input[..])
                .ok()
                .and_then(|mut d| d.read_to_end(&mut out).ok())
            {
                Some(_) => out,
                None => { let _ = tx.send(Err(format!("{name}: corrupt zstd-compressed .tvcache file"))); return; }
            }
        } else {
            raw_input
        };
        match load_cache_from_bytes(&decompressed) {
            Some(trace) => {
                eprintln!("  cache: {:.2}s ({} events)", t0.elapsed().as_secs_f64(), trace.total_events);
                let _ = tx.send(Ok(trace));
            }
            None => { let _ = tx.send(Err(format!("{name}: invalid or corrupt .tvcache file"))); }
        }
        return;
    }

    let raw = match read_bytes_named(name, raw_input) {
        Ok(r) => r,
        Err(e) => { let _ = tx.send(Err(e)); return; }
    };
    eprintln!("  read: {:.2}s ({}MB)", t0.elapsed().as_secs_f64(), raw.len() / 1024 / 1024);

    let te = match find_key(&raw, b"traceEvents") {
        Some(v) => v,
        None => { let _ = tx.send(Err(format!("{name}: no traceEvents found"))); return; }
    };
    let mut pos = te + "\"traceEvents\"".len();
    pos = skip_ws(&raw, pos);
    if pos < raw.len() && raw[pos] == b':' { pos += 1; }
    pos = skip_ws(&raw, pos);
    if pos >= raw.len() || raw[pos] != b'[' {
        let _ = tx.send(Err(format!("{name}: malformed traceEvents")));
        return;
    }
    let array_start = pos + 1;
    let n_threads = calc_n_threads(raw.len(), 0);
    let split_points = find_split_points(&raw, array_start, n_threads);
    let n_chunks = split_points.len() - 1;
    let chunks = parse_chunks_parallel(n_chunks, |i| {
        let start = split_points[i];
        let end = split_points[i + 1];
        let mut state = ChunkState::new();
        parse_chunk(&raw, start, end, &mut state, counter);
        state
    });

    match build_trace(raw, chunks, n_chunks, &t0) {
        Ok(mut trace) => {
            compact_args(&mut trace);
            eprintln!("  ready: {:.2}s ({} tracks)", t0.elapsed().as_secs_f64(), trace.tracks.len());
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
/// already in memory instead of a path, and skips the merged-cache round
/// trip entirely (no persistent cache on wasm — see `load_cache`'s stub).
#[cfg(target_arch = "wasm32")]
pub fn load_multi_from_bytes_progressive(
    rank_named_bytes: Vec<(usize, String, Vec<u8>)>,
    counter: &Arc<AtomicUsize>,
    tx: &std::sync::mpsc::Sender<Result<Trace, String>>,
) {
    let t0 = Instant::now();

    let results: Vec<(usize, Result<Trace, String>)> = rank_named_bytes.into_par_iter()
        .map(|(rank, name, bytes)| {
            let (tx2, rx2) = std::sync::mpsc::channel();
            load_trace_from_bytes_progressive(&name, bytes, counter, &tx2);
            let result = rx2.try_recv().unwrap_or_else(|_| Err(format!("{name}: no result")));
            (rank, result)
        })
        .collect();

    let mut traces = Vec::new();
    for (rank, result) in results {
        match result {
            Ok(t) => traces.push((rank, t)),
            Err(e) => eprintln!("  rank {rank}: {e}"),
        }
    }
    if traces.is_empty() {
        let _ = tx.send(Err("all ranks failed to load".into()));
        return;
    }
    let trace = merge_traces(traces);
    eprintln!("  merged: {:.2}s ({} events, {} flow_pairs)", t0.elapsed().as_secs_f64(), trace.total_events, trace.flow_pairs.len());
    let _ = tx.send(Ok(trace));
}
