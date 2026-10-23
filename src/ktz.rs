//! KTZ compressed-trace decoding — the container format written by
//! CLionProjects/compression (`ktz c`). `decode` reconstructs the original
//! JSON text byte-for-byte identically to `ktz d`, so the output drops
//! straight into the normal parse pipeline (find_key / find_split_points /
//! parse_chunk) with no special-casing.
//!
//! Containers (see ktz.cpp):
//! - KTZ4/KTZ3: `KTZ{3,4}` + varint nEvents + varint nFrames + per-frame
//!   {varint payloadLen, varint compLen} + zstd frames back-to-back. The
//!   concatenated frame payloads are the "sections". KTZ4 per-name streams
//!   carry a column schema + per-event flags + fallback blobs; KTZ3 (and
//!   KTZ2) streams carry only blens+blobs.
//! - KTZ2: the whole file is a single zstd frame holding `KTZ2` + varint
//!   nEvents + sections.
//!
//! Sections, in order: 7 length-prefixed per-event columns (ph, pid, tid,
//! name, cat, ts, dur — one varint per event each, symbol ids with 0 =
//! absent), raw varint nStreams, per stream {raw varint nameId + section
//! (v4: one schema section; v2/3: blens section + blobs section)}, symtab
//! section (raw spans, quoted strings keep their quotes, slot 0 reserved),
//! head/foot section. ts is a zigzag+1 delta chained per (pidId,tidId) key;
//! dur is absolute ns via zigzag+1; both render as µs with exactly three
//! decimals, matching ktz's `ns_str`.
//!
//! Two pieces of delta state reset every 8192 events (EVENTS_PER_RANGE in
//! ktz.cpp — the writer uses one worker per range): the blob token dmap
//! (key id -> last int, driven by TK_INT/TK_DINT tokens) and each column's
//! delta base. The resets key off the GLOBAL event index, so they are
//! mirrored here exactly, including the order: the name-stream flag/blen
//! varints are consumed before the dmap reset, and the reset happens before
//! any of the event's tokens are decoded.

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Sanity caps so a corrupt or hostile file fails with an error instead of
/// an allocation abort. Real ktz files target ~16MiB column frames plus one
/// blob-dominated tail frame; anything near these limits is far beyond a
/// trace this app could render anyway.
const MAX_FRAME_PAYLOAD: usize = 1 << 31; // 2 GiB per frame
#[cfg(not(target_arch = "wasm32"))]
const MAX_TOTAL_PAYLOAD: usize = 1 << 33; // 8 GiB across all frames
#[cfg(target_arch = "wasm32")]
const MAX_TOTAL_PAYLOAD: usize = 1 << 31; // 32-bit usize; a browser tab can't hold more anyway

/// Mirrors `ktz d` from CLionProjects/compression: takes a whole .ktz file
/// (any of KTZ2/KTZ3/KTZ4) and returns the reconstructed JSON text.
/// `counter` receives one bump per rendered event (loader progress).
pub fn decode(data: &[u8], counter: &AtomicUsize) -> Result<Vec<u8>, String> {
    if data.len() >= 4 && (&data[..4] == b"KTZ3" || &data[..4] == b"KTZ4") {
        let is_v4 = &data[..4] == b"KTZ4";
        let mut c = Cur { d: data, p: 4 };
        let n_events = c.varint()?;
        let n_frames = c.varint()?;
        if n_frames > data.len() as u64 {
            return Err("corrupt ktz: frame count exceeds file size".into());
        }
        let mut specs: Vec<(usize, usize)> = Vec::with_capacity(n_frames as usize);
        let mut total_pay = 0usize;
        let mut total_comp = 0usize;
        for _ in 0..n_frames {
            let pay = usize::try_from(c.varint()?)
                .map_err(|_| "corrupt ktz: frame payload length overflow".to_string())?;
            let comp = usize::try_from(c.varint()?)
                .map_err(|_| "corrupt ktz: frame length overflow".to_string())?;
            if pay > MAX_FRAME_PAYLOAD {
                return Err(format!("corrupt ktz: frame payload of {pay} bytes exceeds the {MAX_FRAME_PAYLOAD}-byte limit"));
            }
            total_pay = total_pay.checked_add(pay).ok_or("corrupt ktz: total payload overflow")?;
            total_comp = total_comp.checked_add(comp).ok_or("corrupt ktz: total frame size overflow")?;
            specs.push((pay, comp));
        }
        if total_comp > data.len() - c.p {
            return Err("corrupt ktz: frame table overruns the file".into());
        }
        if total_pay > MAX_TOTAL_PAYLOAD {
            return Err(format!("corrupt ktz: total payload of {total_pay} bytes exceeds the {MAX_TOTAL_PAYLOAD}-byte limit"));
        }
        // c.p is now the start of the back-to-back frames.
        let frames: Vec<&[u8]> = {
            let mut o = c.p;
            specs.iter().map(|&(_, comp)| {
                let f = &data[o..o + comp];
                o += comp;
                f
            }).collect()
        };
        let parts: Vec<Vec<u8>> = frames
            .into_par_iter()
            .zip(&specs)
            .map(|(comp, &(pay, _))| decompress_frame(comp, pay))
            .collect::<Result<Vec<_>, _>>()?;
        let mut payload = Vec::with_capacity(total_pay);
        for p in parts {
            payload.extend_from_slice(&p);
        }
        render(&payload, n_events, is_v4, counter)
    } else {
        // v2: the whole file is one zstd frame holding KTZ2 + nEvents + sections
        let payload = decompress_whole(data)?;
        if payload.len() < 4 || &payload[..4] != b"KTZ2" {
            return Err("not a ktz file (bad magic and not a zstd frame)".into());
        }
        let mut c = Cur { d: &payload, p: 4 };
        let n_events = c.varint()?;
        render(&payload[c.p..], n_events, false, counter)
    }
}

/// Decompress one ktz zstd frame whose payload must be exactly `pay` bytes
/// (the frame table's payloadLen is validated against the actual output,
/// mirroring ktz.cpp's `out.pos != payLen[i]` check).
#[cfg(not(target_arch = "wasm32"))]
fn decompress_frame(comp: &[u8], pay: usize) -> Result<Vec<u8>, String> {
    let out = zstd::bulk::decompress(comp, pay)
        .map_err(|e| format!("corrupt ktz: zstd frame: {e}"))?;
    if out.len() != pay {
        return Err(format!("corrupt ktz: frame declares {pay} payload bytes but holds {}", out.len()));
    }
    Ok(out)
}

#[cfg(target_arch = "wasm32")]
fn decompress_frame(comp: &[u8], pay: usize) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut out = Vec::with_capacity(pay);
    let mut d = ruzstd::decoding::StreamingDecoder::new(comp)
        .map_err(|e| format!("corrupt ktz: zstd frame: {e}"))?;
    d.read_to_end(&mut out)
        .map_err(|e| format!("corrupt ktz: zstd frame: {e}"))?;
    if out.len() != pay {
        return Err(format!("corrupt ktz: frame declares {pay} payload bytes but holds {}", out.len()));
    }
    Ok(out)
}

/// KTZ2: decompress the whole file (one zstd frame).
#[cfg(not(target_arch = "wasm32"))]
fn decompress_whole(data: &[u8]) -> Result<Vec<u8>, String> {
    zstd::stream::decode_all(data).map_err(|e| format!("corrupt ktz: {e}"))
}

#[cfg(target_arch = "wasm32")]
fn decompress_whole(data: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut out = Vec::new();
    let mut d = ruzstd::decoding::StreamingDecoder::new(data)
        .map_err(|e| format!("corrupt ktz: {e}"))?;
    d.read_to_end(&mut out)
        .map_err(|e| format!("corrupt ktz: {e}"))?;
    Ok(out)
}

// =============================================================== reader state

/// Bounds-checked cursor over a byte slice. Every ktz stream is walked with
/// one of these; running past the end is a corrupt-file error, never UB
/// (the C++ reader just reads garbage there).
#[derive(Clone, Copy)]
struct Cur<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn new(d: &'a [u8]) -> Self {
        Cur { d, p: 0 }
    }

    fn byte(&mut self) -> Result<u8, String> {
        let b = *self.d.get(self.p).ok_or("corrupt ktz: unexpected end of data")?;
        self.p += 1;
        Ok(b)
    }

    /// LEB128 varint; at most 10 bytes fit in a u64.
    fn varint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        let mut shift = 0u32;
        loop {
            let b = self.byte()?;
            if shift < 64 {
                v |= ((b & 0x7f) as u64) << shift;
            }
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift > 63 {
                return Err("corrupt ktz: overlong varint".into());
            }
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).filter(|&e| e <= self.d.len());
        let end = end.ok_or("corrupt ktz: section overruns data")?;
        let s = &self.d[self.p..end];
        self.p = end;
        Ok(s)
    }

    fn rest(&self) -> &'a [u8] {
        &self.d[self.p..]
    }
}

fn read_section<'a>(c: &mut Cur<'a>) -> Result<&'a [u8], String> {
    let len = usize::try_from(c.varint()?)
        .map_err(|_| "corrupt ktz: section length overflow".to_string())?;
    if len > c.d.len() {
        return Err("corrupt ktz: section length exceeds payload".into());
    }
    c.take(len)
}

fn section_cur<'a>(c: &mut Cur<'a>) -> Result<Cur<'a>, String> {
    Ok(Cur::new(read_section(c)?))
}

/// Per-name stream state. v4 streams carry columns + flags; v2/3 streams
/// only blens + blobs (flags/cols stay empty and are never read).
struct NameStream<'a> {
    key_ids: Vec<u64>,
    kinds: Vec<u8>, // 0 = int, 1 = str
    cols: Vec<Cur<'a>>,
    col_last: Vec<i64>, // per-column delta base; reset at range changes
    col_range: usize,   // range of col_last; usize::MAX = none decoded yet
    wrap: bool,         // columns live inside one "args" object
    flags: Cur<'a>,
    blens: Cur<'a>,
    blobs: Cur<'a>,
}

fn parse_v4_stream(sec: &[u8]) -> Result<NameStream<'_>, String> {
    let mut q = Cur::new(sec);
    let n_keys = usize::try_from(q.varint()?)
        .map_err(|_| "corrupt ktz: key count overflow".to_string())?;
    // Each key costs at least a varint + a kind byte.
    if n_keys > sec.len() {
        return Err("corrupt ktz: key count exceeds section".into());
    }
    let wrap = q.byte()? != 0;
    let mut key_ids = Vec::with_capacity(n_keys);
    let mut kinds = Vec::with_capacity(n_keys);
    for _ in 0..n_keys {
        key_ids.push(q.varint()?);
        kinds.push(q.byte()?);
    }
    let mut cols = Vec::with_capacity(n_keys);
    for _ in 0..n_keys {
        let cl = usize::try_from(q.varint()?)
            .map_err(|_| "corrupt ktz: column length overflow".to_string())?;
        cols.push(Cur::new(q.take(cl)?));
    }
    let fl = usize::try_from(q.varint()?)
        .map_err(|_| "corrupt ktz: flags length overflow".to_string())?;
    let flags = Cur::new(q.take(fl)?);
    let bl = usize::try_from(q.varint()?)
        .map_err(|_| "corrupt ktz: blens length overflow".to_string())?;
    let blens = Cur::new(q.take(bl)?);
    let blobs = Cur::new(q.rest());
    Ok(NameStream {
        key_ids,
        kinds,
        cols,
        col_last: vec![0; n_keys],
        col_range: usize::MAX,
        wrap,
        flags,
        blens,
        blobs,
    })
}

// =============================================================== token stream

const TK_OBJ: u8 = 1;
const TK_ARR: u8 = 2;
const TK_STR: u8 = 3;
const TK_INT: u8 = 4;
const TK_RAWNUM: u8 = 5;
const TK_TRUE: u8 = 6;
const TK_FALSE: u8 = 7;
const TK_NULL: u8 = 8;
const TK_DINT: u8 = 9; // delta vs last int seen at the same key id
const TK_INTABS: u8 = 10; // absolute int that never touches the dmap

const EVENTS_PER_RANGE: u64 = 8192;

fn unzigzag(u: u64) -> i64 {
    ((u >> 1) ^ (u & 1).wrapping_neg()) as i64
}

/// Inverse of ktz.cpp's encode_zz1 (zigzag with 0 reserved for "absent").
fn decode_zz1(v: u64) -> i64 {
    unzigzag(v.wrapping_sub(1))
}

fn sym<'a>(syms: &[&'a [u8]], id: u64) -> Result<&'a [u8], String> {
    syms.get(id as usize).copied()
        .ok_or_else(|| format!("corrupt ktz: symbol id {id} out of range"))
}

// ---- integer formatting (C snprintf equivalents; no allocation) ----

fn push_u64(out: &mut Vec<u8>, mut v: u64) {
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    out.extend_from_slice(&buf[i..]);
}

/// "%lld"
fn push_i64(out: &mut Vec<u8>, v: i64) {
    if v < 0 {
        out.push(b'-');
        push_u64(out, (v as u64).wrapping_neg()); // i64::MIN -> 2^63, like C's -(unsigned long long)ns
    } else {
        push_u64(out, v as u64);
    }
}

/// ktz.cpp's ns_str: "%s%llu.%03llu" — nanoseconds as µs with exactly three
/// decimals.
fn push_ns(out: &mut Vec<u8>, ns: i64) {
    let neg = ns < 0;
    let a = if neg { (ns as u64).wrapping_neg() } else { ns as u64 };
    if neg {
        out.push(b'-');
    }
    push_u64(out, a / 1000);
    out.push(b'.');
    let m = (a % 1000) as u32;
    out.push(b'0' + (m / 100) as u8);
    out.push(b'0' + ((m / 10) % 10) as u8);
    out.push(b'0' + (m % 10) as u8);
}

/// Decode one value from a blob's token stream, appending its JSON text to
/// `out`. Mirrors ktz.cpp's emit_tok: strings come raw from the symbol
/// table (quotes included), ints via plain zigzag, TK_DINT as a delta
/// against dmap[keyId] (always updating it; TK_INT only updates when
/// keyId != 0), raw numbers byte-for-byte.
fn emit_tok(
    q: &mut Cur,
    key_id: u64,
    syms: &[&[u8]],
    dmap: &mut HashMap<u64, i64>,
    out: &mut Vec<u8>,
    depth: u32,
) -> Result<(), String> {
    if depth > 512 {
        return Err("corrupt ktz: token nesting too deep".into());
    }
    let tk = q.byte()?;
    match tk {
        TK_STR => out.extend_from_slice(sym(syms, q.varint()?)?),
        TK_DINT => {
            let d = unzigzag(q.varint()?);
            let v = dmap.get(&key_id).copied().unwrap_or(0).wrapping_add(d);
            dmap.insert(key_id, v);
            push_i64(out, v);
        }
        TK_INT | TK_INTABS => {
            let v = decode_zz1(q.varint()?.wrapping_add(1));
            if tk == TK_INT && key_id != 0 {
                dmap.insert(key_id, v);
            }
            push_i64(out, v);
        }
        TK_RAWNUM => {
            let len = usize::try_from(q.varint()?)
                .map_err(|_| "corrupt ktz: raw number length overflow".to_string())?;
            out.extend_from_slice(q.take(len)?);
        }
        TK_TRUE => out.extend_from_slice(b"true"),
        TK_FALSE => out.extend_from_slice(b"false"),
        TK_NULL => out.extend_from_slice(b"null"),
        TK_OBJ => {
            let n = q.varint()?;
            out.push(b'{');
            for j in 0..n {
                if j > 0 {
                    out.push(b',');
                }
                let tag = q.byte()?;
                if tag != TK_STR {
                    return Err("corrupt ktz: object key is not a string token".into());
                }
                let kid = q.varint()?;
                out.extend_from_slice(sym(syms, kid)?);
                out.push(b':');
                emit_tok(q, kid, syms, dmap, out, depth + 1)?;
            }
            out.push(b'}');
        }
        TK_ARR => {
            let n = q.varint()?;
            out.push(b'[');
            for j in 0..n {
                if j > 0 {
                    out.push(b',');
                }
                emit_tok(q, 0, syms, dmap, out, depth + 1)?;
            }
            out.push(b']');
        }
        _ => return Err(format!("corrupt ktz: unknown token {tk}")),
    }
    Ok(())
}

// =============================================================== event render

/// The sections -> JSON text pass; output is byte-identical to `ktz d`.
fn render(payload: &[u8], n_events: u64, is_v4: bool, counter: &AtomicUsize) -> Result<Vec<u8>, String> {
    // Each event consumes at least one byte from each of the 7 columns.
    if n_events > payload.len() as u64 {
        return Err("corrupt ktz: event count exceeds payload".into());
    }
    let mut c = Cur::new(payload);
    let mut c_ph = section_cur(&mut c)?;
    let mut c_pid = section_cur(&mut c)?;
    let mut c_tid = section_cur(&mut c)?;
    let mut c_name = section_cur(&mut c)?;
    let mut c_cat = section_cur(&mut c)?;
    let mut c_ts = section_cur(&mut c)?;
    let mut c_dur = section_cur(&mut c)?;

    // Per-name streams. Each stream costs at least a varint + a section
    // header, so the count can never exceed the payload size.
    let n_streams = c.varint()?;
    if n_streams > payload.len() as u64 {
        return Err("corrupt ktz: stream count exceeds payload".into());
    }
    let mut names: HashMap<u64, NameStream> = HashMap::with_capacity(n_streams as usize);
    for _ in 0..n_streams {
        let nid = c.varint()?;
        if is_v4 {
            let sec = read_section(&mut c)?;
            names.insert(nid, parse_v4_stream(sec)?);
        } else {
            let blens = section_cur(&mut c)?;
            let blobs = section_cur(&mut c)?;
            names.insert(nid, NameStream {
                key_ids: Vec::new(),
                kinds: Vec::new(),
                cols: Vec::new(),
                col_last: Vec::new(),
                col_range: usize::MAX,
                wrap: false,
                flags: Cur::new(&[]),
                blens,
                blobs,
            });
        }
    }

    // Symbol table: raw JSON spans, slot 0 reserved for "absent".
    let sym_sec = read_section(&mut c)?;
    let mut q = Cur::new(sym_sec);
    let n_syms = usize::try_from(q.varint()?)
        .map_err(|_| "corrupt ktz: symbol count overflow".to_string())?;
    if n_syms > sym_sec.len() {
        return Err("corrupt ktz: symbol count exceeds section".into());
    }
    let mut syms: Vec<&[u8]> = Vec::with_capacity(n_syms);
    for _ in 0..n_syms {
        let len = usize::try_from(q.varint()?)
            .map_err(|_| "corrupt ktz: symbol length overflow".to_string())?;
        syms.push(q.take(len)?);
    }

    // Header + footer text (everything before/after the event lines).
    let hf_sec = read_section(&mut c)?;
    let mut q = Cur::new(hf_sec);
    let hl = usize::try_from(q.varint()?)
        .map_err(|_| "corrupt ktz: header length overflow".to_string())?;
    let head = q.take(hl)?;
    let fl = usize::try_from(q.varint()?)
        .map_err(|_| "corrupt ktz: footer length overflow".to_string())?;
    let foot = q.take(fl)?;

    let mut out = Vec::with_capacity(head.len() + foot.len() + payload.len().saturating_mul(4));
    out.extend_from_slice(head);

    let mut dmap: HashMap<u64, i64> = HashMap::with_capacity(256);
    let mut last_ts: HashMap<u64, i64> = HashMap::new();

    macro_rules! put_sym {
        ($first:expr, $k:expr, $id:expr) => {
            if !$first {
                out.push(b',');
            }
            $first = false;
            out.push(b'"');
            out.extend_from_slice($k);
            out.extend_from_slice(b"\":");
            out.extend_from_slice(sym(&syms, $id)?);
        };
    }

    for i in 0..n_events {
        let ph_id = c_ph.varint()?;
        let pid_id = c_pid.varint()?;
        let tid_id = c_tid.varint()?;
        let name_id = c_name.varint()?;
        let cat_id = c_cat.varint()?;
        let ts_enc = c_ts.varint()?;
        let dur_enc = c_dur.varint()?;

        // Name-stream flag/blen are consumed BEFORE the range reset
        // (mirrors the reader's read order in ktz.cpp).
        let mut blen = 0usize;
        let mut args_mode = 0u8; // 0 = none, 1 = columnized, 2 = blob
        if let Some(ns) = names.get_mut(&name_id) {
            if is_v4 {
                let flag = ns.flags.varint()?;
                if flag == 1 {
                    args_mode = 1;
                } else if flag == 2 {
                    args_mode = 2;
                    blen = usize::try_from(ns.blens.varint()?)
                        .map_err(|_| "corrupt ktz: blob length overflow".to_string())?;
                }
            } else {
                blen = usize::try_from(ns.blens.varint()?)
                    .map_err(|_| "corrupt ktz: blob length overflow".to_string())?;
                if blen != 0 {
                    args_mode = 2;
                }
            }
        }

        if i % EVENTS_PER_RANGE == 0 {
            dmap.clear(); // mirror the writer's per-range worker reset
        }

        out.extend_from_slice(b"  {");
        let mut first = true;
        if ph_id != 0 { put_sym!(first, b"ph", ph_id); }
        if pid_id != 0 { put_sym!(first, b"pid", pid_id); }
        if tid_id != 0 { put_sym!(first, b"tid", tid_id); }
        if name_id != 0 { put_sym!(first, b"name", name_id); }
        if cat_id != 0 { put_sym!(first, b"cat", cat_id); }
        if ts_enc != 0 {
            let d = decode_zz1(ts_enc);
            // (uint64_t)pidId << 32 | (uint32_t)tidId — tid truncated to 32 bits
            let key = (pid_id << 32) | (tid_id as u32 as u64);
            let base = last_ts.get(&key).copied().unwrap_or(0);
            let ns = base.wrapping_add(d);
            last_ts.insert(key, ns);
            if !first {
                out.push(b',');
            }
            first = false;
            out.extend_from_slice(b"\"ts\":");
            push_ns(&mut out, ns);
        }
        if dur_enc != 0 {
            if !first {
                out.push(b',');
            }
            first = false;
            out.extend_from_slice(b"\"dur\":");
            push_ns(&mut out, decode_zz1(dur_enc));
        }

        if args_mode == 1 {
            // Columnized values; the delta base resets when the global event
            // index crosses into a new 8192-event range.
            let ns = names.get_mut(&name_id)
                .ok_or("corrupt ktz: columnized event references a missing name stream")?;
            let rng = (i / EVENTS_PER_RANGE) as usize;
            if rng != ns.col_range {
                ns.col_last.fill(0);
                ns.col_range = rng;
            }
            if ns.wrap {
                if !first {
                    out.push(b',');
                }
                first = true; // inner fields restart the comma logic
                out.extend_from_slice(b"\"args\":{");
            }
            for j in 0..ns.key_ids.len() {
                if !first {
                    out.push(b',');
                }
                first = false;
                out.extend_from_slice(sym(&syms, ns.key_ids[j])?);
                out.push(b':');
                let u = ns.cols[j].varint()?;
                let v = unzigzag(u);
                if ns.kinds[j] == 1 {
                    // str: delta of the symbol id
                    let idx = ns.col_last[j].wrapping_add(v);
                    out.extend_from_slice(sym(&syms, idx as u64)?);
                    ns.col_last[j] = idx;
                } else {
                    // int: delta vs the last value in this column
                    ns.col_last[j] = ns.col_last[j].wrapping_add(v);
                    push_i64(&mut out, ns.col_last[j]);
                }
            }
            if ns.wrap {
                out.push(b'}');
            }
        } else if blen != 0 {
            // Blob fallback: key/value pairs, braces omitted. The blob's
            // bytes are bounded exactly like ktz.cpp's q0/q0e cursors.
            let ns = names.get_mut(&name_id)
                .ok_or("corrupt ktz: blob event references a missing name stream")?;
            let region = ns.blobs.d.get(ns.blobs.p..ns.blobs.p + blen)
                .ok_or("corrupt ktz: blob length overruns stream")?;
            let mut q = Cur::new(region);
            while q.p < region.len() {
                if !first {
                    out.push(b',');
                }
                first = false;
                let tag = q.byte()?;
                if tag != TK_STR {
                    return Err("corrupt ktz: blob key is not a string token".into());
                }
                let kid = q.varint()?;
                out.extend_from_slice(sym(&syms, kid)?);
                out.push(b':');
                emit_tok(&mut q, kid, &syms, &mut dmap, &mut out, 0)?;
            }
            ns.blobs.p += blen;
        }

        out.extend_from_slice(if i + 1 < n_events { b"},\n" } else { b"}\n" });
        counter.fetch_add(1, Ordering::Relaxed);
    }

    out.extend_from_slice(foot);
    Ok(out)
}
