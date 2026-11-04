//! Direct Chrome/kineto trace JSON (`.json`, `.pt.trace.json.gz`) ->
//! `KtzTrace` chunks.
//!
//! This is the pre-ktz fast path, restored: decompress the gzip stream once
//! (libdeflate natively), split the `traceEvents` array at event boundaries,
//! parse the chunks in parallel, and keep every event's args as an offset
//! into the decompressed JSON instead of copying the text out. The output is
//! the exact shape `KtzDecoder` produces, so `TraceScan` -> `assemble_track`
//! -> `finish_trace` and everything downstream stays ktz-driven: tracks,
//! selection, search and the args viewer never know which container the
//! events came from. `KtzTrace.args` is left for the caller, which points it
//! at the whole decompressed JSON (the raw buffer every `args_off` indexes).
//!
//! Parsing mirrors the old `parse_chunk` byte for byte: only `ph == X` with
//! a duration becomes a track event, `M` events feed `thread_names`, and
//! `s`/`f` events feed flow pairing.

use crate::ktz::KtzTrace;
use crate::parse::*;
use rayon::prelude::*;
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A decompressed JSON trace, ready to be folded into the shared scan.
pub struct ParsedJson {
    /// One per parse chunk. Each holds its own intern tables and events;
    /// `TraceScan::merge_chunk` folds them together.
    pub chunks: Vec<KtzTrace>,
    /// The decompressed JSON itself. Every event's `args_off` indexes into
    /// this buffer, and it becomes the finished trace's args buffer.
    pub raw: Vec<u8>,
}

/// gzip magic (RFC 1952).
pub fn is_gzip(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b
}

/// Whether the bytes plausibly are a JSON object/array (after whitespace).
/// Used only to pick the decoder and to give a non-trace file a friendly
/// error instead of a "no traceEvents found" from deep in the parser.
pub fn looks_like_json(data: &[u8]) -> bool {
    let p = skip_ws(data, 0);
    p < data.len() && (data[p] == b'{' || data[p] == b'[')
}

/// Decompress a gzip stream. Native tries libdeflate first (the fast path
/// for `.pt.trace.json.gz`), then flate2; wasm uses flate2's pure-Rust
/// backend. Both backends tolerate a truncated tail the way the old loader
/// did — a partially written trace is still worth opening.
pub fn decompress_gz(compressed: &[u8]) -> Result<Vec<u8>, String> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(buf) = try_libdeflate(compressed) {
        return Ok(buf);
    }
    let mut out = Vec::with_capacity(gz_size_hint(compressed));
    let mut decoder = flate2::read::GzDecoder::new(compressed);
    match decoder.read_to_end(&mut out) {
        Ok(_) => Ok(out),
        Err(e) if !out.is_empty() => {
            tlog!("  truncated gz stream after {} bytes: {e}", out.len());
            Ok(out)
        }
        Err(e) => Err(format!("gzip: {e}")),
    }
}

/// Decompressed size estimate from the gzip ISIZE trailer (size mod 2^32).
/// Falls back to a 25x ratio when the trailer is absent or doesn't look like
/// a real size (ktz's ratio on this JSON runs well past 25x, so the fallback
/// is deliberately generous).
fn gz_size_hint(compressed: &[u8]) -> usize {
    if compressed.len() >= 4 {
        let isize = u32::from_le_bytes(
            compressed[compressed.len() - 4..].try_into().unwrap(),
        ) as usize;
        // gzip stores the size mod 2^32, so a corrupt trailer can claim up
        // to 4GiB. Real trace ratios stay well under 4096x; anything
        // wilder is not trusted for the pre-allocation (libdeflate then
        // fails on the small buffer and flate2 grows the Vec as needed).
        if isize > compressed.len() && isize <= compressed.len().saturating_mul(4096) {
            return isize + isize / 10;
        }
    }
    compressed.len().saturating_mul(25)
}

/// Decompressed size of a `.gz` file straight from its last four bytes,
/// without decompressing — the loader's multi-rank wave sizing uses this so
/// a batch of `.json.gz` ranks is bounded by the JSON they expand to, not
/// by the compressed file size times an arbitrary factor.
pub fn gz_uncompressed_size(path: &str) -> Option<u64> {
    use std::io::{Seek, SeekFrom};
    let mut f = std::fs::File::open(path).ok()?;
    let len = f.seek(SeekFrom::End(0)).ok()?;
    if len < 4 { return None; }
    f.seek(SeekFrom::End(-4)).ok()?;
    let mut b = [0u8; 4];
    f.read_exact(&mut b).ok()?;
    let isize = u32::from_le_bytes(b) as u64;
    (isize > len).then_some(isize)
}

#[cfg(not(target_arch = "wasm32"))]
fn try_libdeflate(compressed: &[u8]) -> Option<Vec<u8>> {
    let mut buf = vec![0u8; gz_size_hint(compressed)];
    let mut decompressor = libdeflater::Decompressor::new();
    match decompressor.gzip_decompress(compressed, &mut buf) {
        Ok(actual) => { buf.truncate(actual); Some(buf) }
        Err(_) => None,
    }
}

/// Parse a decompressed JSON trace into per-chunk `KtzTrace`s. `raw` is
/// taken by value because the caller keeps it as the args buffer; the chunks
/// borrow nothing from it (all args are byte offsets).
pub fn parse(raw: Vec<u8>, counter: &AtomicUsize) -> Result<ParsedJson, String> {
    let te = find_key(&raw, b"traceEvents").ok_or("no traceEvents found")?;
    let mut pos = te + "\"traceEvents\"".len();
    pos = skip_ws(&raw, pos);
    if pos < raw.len() && raw[pos] == b':' { pos += 1; }
    pos = skip_ws(&raw, pos);
    if pos >= raw.len() || raw[pos] != b'[' {
        return Err("malformed traceEvents".into());
    }
    let array_start = pos + 1;

    let n_threads = calc_n_threads(raw.len());
    let split_points = find_split_points(&raw, array_start, n_threads);
    let n_chunks = split_points.len() - 1;

    let chunks: Vec<KtzTrace> = (0..n_chunks)
        .into_par_iter()
        .filter_map(|i| {
            let start = split_points[i];
            let end = split_points[i + 1];
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut state = KtzTrace::with_events(0);
                parse_chunk(&raw, start, end, &mut state, counter);
                state
            })) {
                Ok(state) => Some(state),
                Err(e) => {
                    let msg = e.downcast_ref::<&str>().map(|s| s.to_string())
                        .or_else(|| e.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "unknown panic".to_string());
                    tlog!("  parse thread panicked: {msg}");
                    None
                }
            }
        })
        .collect();

    if chunks.is_empty() {
        return Err("no traceEvents found".into());
    }
    Ok(ParsedJson { chunks, raw })
}

fn calc_n_threads(data_len: usize) -> usize {
    if data_len < 10 * 1024 * 1024 { return 1; }
    let avail = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
    avail.clamp(2, 8)
}

/// Byte-for-byte port of the old `parse_chunk`: scan one slice of the
/// `traceEvents` array, appending X events, thread names and flows to
/// `state`. `raw` is the whole decompressed buffer so `args_off` is an
/// absolute offset usable by every consumer.
fn parse_chunk(raw: &[u8], start: usize, chunk_end: usize, state: &mut KtzTrace, counter: &AtomicUsize) {
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
                            cat = state.cats.intern(&raw[s..pos - 1]);
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
                            name = state.names.intern(&raw[s..pos - 1]);
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
            state.events.push((pid, tid, crate::types::Event {
                ts, dur, name, cat, args_off, args_len, depth: 0,
            }));
            state.total_events += 1;
            counter.fetch_add(1, Ordering::Relaxed);
        } else if ph == b'M' {
            let name_str = state.names.str(name);
            if name_str == "thread_name" && args_off > 0 {
                let end = skip_value(raw, args_off as usize);
                let mut tmp_strs = InternTable::new();
                let mut tmp_pairs = Vec::new();
                parse_args_flat(&raw[args_off as usize..end], &mut tmp_strs, &mut tmp_pairs);
                for &[k, v] in &tmp_pairs {
                    if (k as usize) < tmp_strs.len() && tmp_strs.str(k) == "name" {
                        if (v as usize) < tmp_strs.len() {
                            state.thread_names.insert((pid, tid), tmp_strs.str(v).to_string());
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
