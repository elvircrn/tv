//! KTZ compressed-trace decoding — the container format written by
//! CLionProjects/compression (`ktz c`). The entry point is `KtzDecoder`,
//! which turns the columnar sections straight into events + args text (see
//! `KtzTrace`) without ever materializing the JSON document.
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

use crate::parse::{fnv1a, parse_args_flat, parse_f64, InternTable};
use crate::types::Event;
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

/// True for anything `KtzDecoder::new` accepts: KTZ3/KTZ4 magic, or a bare
/// zstd frame (KTZ2 — which may also lack a .ktz name, e.g. ?src= links).
pub fn is_ktz(data: &[u8]) -> bool {
    if data.len() >= 4 && (&data[..4] == b"KTZ3" || &data[..4] == b"KTZ4") {
        return true;
    }
    data.len() >= 4 && data[..4] == [0x28, 0xB5, 0x2F, 0xFD]
}

/// KTZ3/KTZ4 container: magic + nEvents + frame table + back-to-back zstd
/// frames. Returns (sections payload, nEvents, is_v4).
fn framed_payload(data: &[u8]) -> Result<(Vec<u8>, u64, bool), String> {
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
    Ok((payload, n_events, is_v4))
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

/// nEvents from a KTZ3/KTZ4 header (the first varint after the magic),
/// without decoding anything — loader wave-sizing uses it to estimate the
/// load's heap footprint before committing. None for KTZ2 (nEvents lives
/// inside the zstd frame) or any non-ktz / unreadable file. Compiles on
/// wasm too (std::fs stubs error there), but only the native path-based
/// loader ever calls it.
pub fn event_count_hint(path: &str) -> Option<u64> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut head = [0u8; 24];
    f.read_exact(&mut head).ok()?;
    if &head[..4] != b"KTZ3" && &head[..4] != b"KTZ4" { return None; }
    let mut c = Cur { d: &head, p: 4 };
    c.varint().ok()
}

/// Total decompressed section payload of a KTZ3/KTZ4 file, summed straight
/// from the frame table (magic + nEvents + nFrames + per-frame
/// {payloadLen, compLen} varints) without decompressing anything — the
/// exact transient-heap figure for one rank's load, used by the multi-rank
/// wave sizing. None for KTZ2 (its payload lives inside the zstd frame) or
/// any file whose table can't be read/validated from the head. Mirrors
/// framed_payload's table walk and validation; never allocates.
pub fn payload_size_hint(path: &str) -> Option<u64> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut head = vec![0u8; 4096];
    let n = f.read(&mut head).ok()?;
    head.truncate(n);
    if head.len() < 4 || (&head[..4] != b"KTZ3" && &head[..4] != b"KTZ4") { return None; }
    let mut c = Cur { d: &head, p: 4 };
    c.varint().ok()?; // nEvents (skip)
    let n_frames = c.varint().ok()?;
    if n_frames > head.len() as u64 { return None; }
    let mut total_pay: u64 = 0;
    let mut total_comp: u64 = 0;
    for _ in 0..n_frames {
        let pay = c.varint().ok()?;
        let comp = c.varint().ok()?;
        if pay > MAX_FRAME_PAYLOAD as u64 { return None; }
        total_pay = total_pay.checked_add(pay)?;
        total_comp = total_comp.checked_add(comp)?;
    }
    // Same table-overrun check framed_payload applies against the file, but
    // against the head we managed to read — a table that walks past it (or
    // past the file) is either corrupt or a huge frame count we can't size.
    if total_comp > (head.len() - c.p) as u64 { return None; }
    Some(total_pay)
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
}

// ============================================================= section ranges

/// A section's extent as offsets into the decoder's payload — the
/// resumable decoder stores positions instead of borrows, so sections are
/// ranges rather than slices.
#[derive(Clone, Copy)]
struct Sec {
    start: usize,
    end: usize,
}

/// Length-prefixed section at `c` (which walks the whole payload), as an
/// absolute range.
fn read_sec(payload: &[u8], c: &mut Cur) -> Result<Sec, String> {
    let len = usize::try_from(c.varint()?)
        .map_err(|_| "corrupt ktz: section length overflow".to_string())?;
    let start = c.p;
    let end = start
        .checked_add(len)
        .filter(|&e| e <= payload.len())
        .ok_or("corrupt ktz: section overruns data")?;
    c.p = end;
    Ok(Sec { start, end })
}

fn sym<'a>(payload: &'a [u8], syms: &[Sec], id: u64) -> Result<&'a [u8], String> {
    let s = syms.get(id as usize).copied()
        .ok_or_else(|| format!("corrupt ktz: symbol id {id} out of range"))?;
    payload
        .get(s.start..s.end)
        .ok_or_else(|| "corrupt ktz: symbol range out of bounds".to_string())
}

/// Read one varint at `pos` within `sec` (position relative to the section).
fn varint_at(payload: &[u8], sec: &Sec, pos: &mut usize) -> Result<u64, String> {
    let mut c = Cur::new(&payload[sec.start..sec.end]);
    c.p = *pos;
    let v = c.varint()?;
    *pos = c.p;
    Ok(v)
}

/// Per-name stream schema (immutable; shared by every step).
struct StreamLayout {
    key_ids: Vec<u64>,
    kinds: Vec<u8>, // 0 = int, 1 = str
    cols: Vec<Sec>,
    wrap: bool, // columns live inside one "args" object
    flags: Sec,
    blens: Sec,
    blobs: Sec,
}

impl StreamLayout {
    fn blob_only(blens: Sec, blobs: Sec) -> Self {
        Self {
            key_ids: Vec::new(),
            kinds: Vec::new(),
            cols: Vec::new(),
            wrap: false,
            flags: Sec { start: 0, end: 0 },
            blens,
            blobs,
        }
    }
}

fn parse_v4_layout(payload: &[u8], sec: Sec) -> Result<StreamLayout, String> {
    let mut q = Cur::new(&payload[sec.start..sec.end]);
    let n_keys = usize::try_from(q.varint()?)
        .map_err(|_| "corrupt ktz: key count overflow".to_string())?;
    // Each key costs at least a varint + a kind byte.
    if n_keys > sec.end - sec.start {
        return Err("corrupt ktz: key count exceeds section".into());
    }
    let wrap = q.byte()? != 0;
    let mut key_ids = Vec::with_capacity(n_keys);
    let mut kinds = Vec::with_capacity(n_keys);
    for _ in 0..n_keys {
        key_ids.push(q.varint()?);
        kinds.push(q.byte()?);
    }
    let sec_at = |q: &Cur, len: usize| Sec {
        start: sec.start + q.p,
        end: sec.start + q.p + len,
    };
    let mut cols = Vec::with_capacity(n_keys);
    for _ in 0..n_keys {
        let cl = usize::try_from(q.varint()?)
            .map_err(|_| "corrupt ktz: column length overflow".to_string())?;
        let s = sec_at(&q, cl);
        q.take(cl)?;
        cols.push(s);
    }
    let fl = usize::try_from(q.varint()?)
        .map_err(|_| "corrupt ktz: flags length overflow".to_string())?;
    let flags = sec_at(&q, fl);
    q.take(fl)?;
    let bl = usize::try_from(q.varint()?)
        .map_err(|_| "corrupt ktz: blens length overflow".to_string())?;
    let blens = sec_at(&q, bl);
    q.take(bl)?;
    let blobs = Sec { start: sec.start + q.p, end: sec.end };
    Ok(StreamLayout { key_ids, kinds, cols, wrap, flags, blens, blobs })
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

/// ns -> µs f64, bit-identical to the old text-parse pipeline: epoch-scale
/// nanoseconds exceed 2^53, so `ns as f64 / 1000.0` double-rounds to a
/// different f64 than parsing the "x.yyy" text did. Events' ts/dur values
/// must keep matching what a .json load produced, so go through the same
/// text -> parse_f64 path.
fn ns_to_us(ns: i64, scratch: &mut Vec<u8>) -> f64 {
    scratch.clear();
    push_ns(scratch, ns);
    parse_f64(scratch)
}

/// Decode one value from a blob's token stream, appending its JSON text to
/// `out`. Mirrors ktz.cpp's emit_tok: strings come raw from the symbol
/// table (quotes included), ints via plain zigzag, TK_DINT as a delta
/// against dmap[keyId] (always updating it; TK_INT only updates when
/// keyId != 0), raw numbers byte-for-byte.
fn emit_tok(
    q: &mut Cur,
    key_id: u64,
    payload: &[u8],
    syms: &[Sec],
    dmap: &mut HashMap<u64, i64>,
    out: &mut Vec<u8>,
    depth: u32,
) -> Result<(), String> {
    if depth > 512 {
        return Err("corrupt ktz: token nesting too deep".into());
    }
    let tk = q.byte()?;
    match tk {
        TK_STR => out.extend_from_slice(sym(payload, syms, q.varint()?)?),
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
                out.extend_from_slice(sym(payload, syms, kid)?);
                out.push(b':');
                emit_tok(q, kid, payload, syms, dmap, out, depth + 1)?;
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
                emit_tok(q, 0, payload, syms, dmap, out, depth + 1)?;
            }
            out.push(b']');
        }
        _ => return Err(format!("corrupt ktz: unknown token {tk}")),
    }
    Ok(())
}

// =============================================================== direct decode
//
// The sections -> events path: decodes the columnar format straight into
// the loader's chunk shape without ever materializing the JSON document
// (the old render()-then-rescan path inflated a 3.7MB c16 file to ~1.5GB
// of text). Semantics mirror what the old .json load produced: same
// interning, same fnv1a/parse_f64 pid/tid keys, bit-identical ts/dur
// f64s, same args text, same drop rules (only ph X-with-dur events
// become tracks; M thread_name and s/f flows feed their side tables).

/// The decode output — everything `parse_chunk` used to collect from the
/// rendered JSON, plus the compact args buffer (byte 0 reserved so
/// args_off = 0 means "no args"; each event's args JSON text is appended
/// back-to-back, exactly like `compact_args`' output).
pub struct KtzTrace {
    pub events: Vec<(u64, u64, Event)>,
    pub names: InternTable,
    pub cats: InternTable,
    pub thread_names: HashMap<(u64, u64), String>,
    pub flows: Vec<(u64, u64, u64, f64, bool)>,
    pub min_ts: f64,
    pub max_ts: f64,
    pub total_events: usize,
    pub scan_count: u64,
    pub args: Vec<u8>,
}

impl KtzTrace {
    pub fn with_events(n_events: u64) -> Self {
        Self {
            // Cap the pre-reserve: a hostile header can claim a huge count
            // (bounded only by payload size); growth past the cap is fine.
            events: Vec::with_capacity(n_events.min(1 << 21) as usize),
            names: InternTable::new(),
            cats: InternTable::new(),
            thread_names: HashMap::new(),
            flows: Vec::new(),
            min_ts: f64::MAX,
            max_ts: f64::MIN,
            total_events: 0,
            scan_count: 0,
            args: vec![0u8],
        }
    }
}

/// Resumable sections -> `KtzTrace` decoder. All positions are offsets
/// into the owned payload (never borrows), so decoding can pause/restart
/// across frames — the wasm single-threaded loader steps this from the
/// browser event loop, and native loaders just run `step` to completion.
pub struct KtzDecoder {
    // immutable after new()
    payload: Vec<u8>,
    is_v4: bool,
    n_events: u64,
    syms: Vec<Sec>,
    cols: [Sec; 7], // ph, pid, tid, name, cat, ts, dur
    streams: HashMap<u64, StreamLayout>,
    /// head ++ foot text, for the metadata scans finish_trace does
    /// (deviceProperties / vllm_version / distributedInfo).
    head_foot: Vec<u8>,
    // cursor positions (relative to each section) + delta state
    col_pos: [usize; 7],
    st: HashMap<u64, StreamState>,
    next_event: u64,
    dmap: HashMap<u64, i64>,
    last_ts: HashMap<u64, i64>,
    /// ts/dur text formatting and blob value rendering scratch
    scratch: Vec<u8>,
}

/// Cursor positions + delta bases for one name stream.
struct StreamState {
    col_pos: Vec<usize>,
    col_last: Vec<i64>,
    col_range: usize, // usize::MAX = none decoded yet
    flags_pos: usize,
    blens_pos: usize,
    blobs_pos: usize,
}

impl StreamState {
    fn new(l: &StreamLayout) -> Self {
        Self {
            col_pos: vec![0; l.cols.len()],
            col_last: vec![0; l.key_ids.len()],
            col_range: usize::MAX,
            flags_pos: 0,
            blens_pos: 0,
            blobs_pos: 0,
        }
    }
}

/// pid/tid symbol: quoted -> fnv1a of the inner bytes (the string-tid
/// scheme), else the number span parsed as f64 then truncated — both
/// exactly as parse_chunk keyed tracks.
fn num_or_hash(payload: &[u8], syms: &[Sec], id: u64) -> Result<u64, String> {
    if id == 0 {
        return Ok(0);
    }
    let sp = sym(payload, syms, id)?;
    if sp.first() == Some(&b'"') && sp.len() >= 2 {
        Ok(fnv1a(&sp[1..sp.len() - 1]))
    } else {
        Ok(parse_f64(sp) as u64)
    }
}

/// name/cat: interned only when the symbol is a quoted string, mirroring
/// parse_chunk's `raw[pos] == b'"'` check (0 = "" otherwise).
fn intern_sym(table: &mut InternTable, payload: &[u8], syms: &[Sec], id: u64) -> Result<u32, String> {
    let sp = sym(payload, syms, id)?;
    if sp.first() == Some(&b'"') && sp.len() >= 2 {
        Ok(table.intern(&sp[1..sp.len() - 1]))
    } else {
        Ok(0)
    }
}

impl KtzDecoder {
    /// Parse the container (zstd frames + section table + symtab). Does not
    /// decode events yet — call `step` until it returns true.
    pub fn new(data: &[u8]) -> Result<Self, String> {
        let (payload, n_events, is_v4) = if data.len() >= 4
            && (&data[..4] == b"KTZ3" || &data[..4] == b"KTZ4")
        {
            framed_payload(data)?
        } else {
            // v2: the whole file is one zstd frame holding KTZ2 + nEvents + sections
            let mut payload = decompress_whole(data)?;
            if payload.len() < 4 || &payload[..4] != b"KTZ2" {
                return Err("not a ktz file (bad magic and not a zstd frame)".into());
            }
            let (n_events, base) = {
                let mut c = Cur { d: &payload, p: 4 };
                let n = c.varint()?;
                (n, c.p)
            };
            payload.drain(..base); // sections start at offset 0, like v3/v4
            (payload, n_events, false)
        };
        if n_events > payload.len() as u64 {
            return Err("corrupt ktz: event count exceeds payload".into());
        }

        let mut c = Cur::new(&payload);
        let mut cols = [Sec { start: 0, end: 0 }; 7];
        for i in 0..7 {
            cols[i] = read_sec(&payload, &mut c)?;
        }
        let n_streams = c.varint()?;
        if n_streams > payload.len() as u64 {
            return Err("corrupt ktz: stream count exceeds payload".into());
        }
        let mut streams = HashMap::with_capacity(n_streams as usize);
        for _ in 0..n_streams {
            let nid = c.varint()?;
            if is_v4 {
                let sec = read_sec(&payload, &mut c)?;
                streams.insert(nid, parse_v4_layout(&payload, sec)?);
            } else {
                let blens = read_sec(&payload, &mut c)?;
                let blobs = read_sec(&payload, &mut c)?;
                streams.insert(nid, StreamLayout::blob_only(blens, blobs));
            }
        }

        // Symbol table: raw JSON spans, slot 0 reserved for "absent".
        let sym_sec = read_sec(&payload, &mut c)?;
        let mut q = Cur::new(&payload[sym_sec.start..sym_sec.end]);
        let n_syms = usize::try_from(q.varint()?)
            .map_err(|_| "corrupt ktz: symbol count overflow".to_string())?;
        if n_syms > sym_sec.end - sym_sec.start {
            return Err("corrupt ktz: symbol count exceeds section".into());
        }
        let mut syms: Vec<Sec> = Vec::with_capacity(n_syms);
        for _ in 0..n_syms {
            let len = usize::try_from(q.varint()?)
                .map_err(|_| "corrupt ktz: symbol length overflow".to_string())?;
            let start = sym_sec.start + q.p;
            q.take(len)?;
            syms.push(Sec { start, end: start + len });
        }

        // Header + footer text (everything before/after the event lines).
        let hf_sec = read_sec(&payload, &mut c)?;
        let mut q = Cur::new(&payload[hf_sec.start..hf_sec.end]);
        let hl = usize::try_from(q.varint()?)
            .map_err(|_| "corrupt ktz: header length overflow".to_string())?;
        let head_start = hf_sec.start + q.p;
        q.take(hl)?;
        let fl = usize::try_from(q.varint()?)
            .map_err(|_| "corrupt ktz: footer length overflow".to_string())?;
        let foot_start = hf_sec.start + q.p;
        q.take(fl)?;
        let mut head_foot = Vec::with_capacity(hl + fl);
        head_foot.extend_from_slice(&payload[head_start..head_start + hl]);
        head_foot.extend_from_slice(&payload[foot_start..foot_start + fl]);

        Ok(Self {
            col_pos: [0; 7],
            st: HashMap::new(),
            next_event: 0,
            dmap: HashMap::with_capacity(256),
            last_ts: HashMap::new(),
            scratch: Vec::new(),
            payload,
            is_v4,
            n_events,
            syms,
            cols,
            streams,
            head_foot,
        })
    }

    pub fn n_events(&self) -> u64 {
        self.n_events
    }

    pub fn into_head_foot(self) -> Vec<u8> {
        self.head_foot
    }

    /// Decode up to `max_events` more events into `out`. Returns true when
    /// the whole file is decoded. `counter` receives one bump per event
    /// (loader progress).
    pub fn step(
        &mut self,
        out: &mut KtzTrace,
        max_events: usize,
        counter: &AtomicUsize,
    ) -> Result<bool, String> {
        let Self {
            payload,
            is_v4,
            n_events,
            syms,
            cols,
            streams,
            col_pos,
            st,
            next_event,
            dmap,
            last_ts,
            scratch,
            ..
        } = self;
        let payload: &[u8] = payload;
        let mut cc: Vec<Cur> = Vec::with_capacity(7);
        for i in 0..7 {
            let mut c = Cur::new(&payload[cols[i].start..cols[i].end]);
            c.p = col_pos[i];
            cc.push(c);
        }

        for _ in 0..max_events {
            if *next_event >= *n_events {
                break;
            }
            let i = *next_event;
            *next_event += 1;
            out.scan_count += 1;

            let ph_id = cc[0].varint()?;
            let pid_id = cc[1].varint()?;
            let tid_id = cc[2].varint()?;
            let name_id = cc[3].varint()?;
            let cat_id = cc[4].varint()?;
            let ts_enc = cc[5].varint()?;
            let dur_enc = cc[6].varint()?;

            // Name-stream flag/blen are consumed BEFORE the range reset
            // (mirrors the reader's read order in ktz.cpp).
            let mut blen = 0usize;
            let mut args_mode = 0u8; // 0 = none, 1 = columnized, 2 = blob
            if let Some(l) = streams.get(&name_id) {
                let s = st.entry(name_id).or_insert_with(|| StreamState::new(l));
                if *is_v4 {
                    let flag = varint_at(payload, &l.flags, &mut s.flags_pos)?;
                    if flag == 1 {
                        args_mode = 1;
                    } else if flag == 2 {
                        args_mode = 2;
                        blen = usize::try_from(varint_at(payload, &l.blens, &mut s.blens_pos)?)
                            .map_err(|_| "corrupt ktz: blob length overflow".to_string())?;
                    }
                } else {
                    blen = usize::try_from(varint_at(payload, &l.blens, &mut s.blens_pos)?)
                        .map_err(|_| "corrupt ktz: blob length overflow".to_string())?;
                    if blen != 0 {
                        args_mode = 2;
                    }
                }
            }

            if i % EVENTS_PER_RANGE == 0 {
                dmap.clear(); // mirror the writer's per-range worker reset
            }

            // Common fields, decoded for every event: the ts delta chain
            // must advance even for events that get dropped, and name/cat
            // interning is what the stream keying below reads.
            let ph = if ph_id != 0 {
                let sp = sym(payload, syms, ph_id)?;
                if sp.first() == Some(&b'"') && sp.len() >= 2 { sp[1] } else { 0 }
            } else {
                0
            };
            let pid = num_or_hash(payload, syms, pid_id)?;
            let tid = num_or_hash(payload, syms, tid_id)?;
            let name = if name_id != 0 { intern_sym(&mut out.names, payload, syms, name_id)? } else { 0 };
            let cat = if cat_id != 0 { intern_sym(&mut out.cats, payload, syms, cat_id)? } else { 0 };
            let ts = if ts_enc != 0 {
                let d = decode_zz1(ts_enc);
                // (uint64_t)pidId << 32 | (uint32_t)tidId — tid truncated to 32 bits
                let key = (pid_id << 32) | (tid_id as u32 as u64);
                let base = last_ts.get(&key).copied().unwrap_or(0);
                let ns = base.wrapping_add(d);
                last_ts.insert(key, ns);
                ns_to_us(ns, scratch)
            } else {
                0.0
            };
            let (dur, has_dur) = if dur_enc != 0 {
                (ns_to_us(decode_zz1(dur_enc), scratch), true)
            } else {
                (0.0, false)
            };

            // Args are only written for events that will use them (X keeps
            // them for the detail panel, M thread_name parses "name" out of
            // them) — but the streams' cursors/dmaps always advance, since
            // later events' deltas are relative to them.
            let keep_args = (ph == b'X' && has_dur)
                || (ph == b'M' && out.names.str(name) == "thread_name");
            let mut args_off = 0u32;
            let mut args_len = 0u16;
            let mut args_end = 0usize; // exact end, for the thread_name parse
            let mut flow_id = 0u64;

            if args_mode == 1 {
                let l = streams.get(&name_id)
                    .ok_or("corrupt ktz: columnized event references a missing name stream")?;
                let s = st.get_mut(&name_id)
                    .ok_or("corrupt ktz: columnized event references a missing name stream")?;
                // Column delta base resets when the global event index
                // crosses into a new 8192-event range.
                let rng = (i / EVENTS_PER_RANGE) as usize;
                if rng != s.col_range {
                    s.col_last.fill(0);
                    s.col_range = rng;
                }
                if l.wrap {
                    // Columns live inside one "args" object.
                    let start = out.args.len();
                    if keep_args {
                        out.args.push(b'{');
                    }
                    for j in 0..l.key_ids.len() {
                        let key = sym(payload, syms, l.key_ids[j])?;
                        let u = varint_at(payload, &l.cols[j], &mut s.col_pos[j])?;
                        let v = unzigzag(u);
                        let str_idx = if l.kinds[j] == 1 {
                            // str: delta of the symbol id
                            let idx = s.col_last[j].wrapping_add(v);
                            s.col_last[j] = idx;
                            Some(idx)
                        } else {
                            // int: delta vs the last value in this column
                            s.col_last[j] = s.col_last[j].wrapping_add(v);
                            None
                        };
                        if keep_args {
                            if j > 0 {
                                out.args.push(b',');
                            }
                            out.args.extend_from_slice(key);
                            out.args.push(b':');
                            match str_idx {
                                Some(idx) => out.args.extend_from_slice(sym(payload, syms, idx as u64)?),
                                None => push_i64(&mut out.args, s.col_last[j]),
                            }
                        }
                    }
                    if keep_args {
                        out.args.push(b'}');
                        let len = out.args.len() - start;
                        args_off = start as u32;
                        args_len = if len <= u16::MAX as usize { len as u16 } else { 0 };
                        args_end = out.args.len();
                    }
                } else {
                    // Top-level columns; parse_chunk only ever read "id"
                    // out of these (flow events).
                    for j in 0..l.key_ids.len() {
                        let key = sym(payload, syms, l.key_ids[j])?;
                        let u = varint_at(payload, &l.cols[j], &mut s.col_pos[j])?;
                        let v = unzigzag(u);
                        if l.kinds[j] == 1 {
                            let idx = s.col_last[j].wrapping_add(v);
                            s.col_last[j] = idx;
                        } else {
                            s.col_last[j] = s.col_last[j].wrapping_add(v);
                            if key == b"\"id\"" {
                                flow_id = s.col_last[j] as u64;
                            }
                        }
                    }
                }
            } else if blen != 0 {
                // Blob fallback: key/value pairs. Every token is walked
                // (the dmap updates feed later events' TK_DINT) but only
                // kept keys render into the args buffer.
                let l = streams.get(&name_id)
                    .ok_or("corrupt ktz: blob event references a missing name stream")?;
                let s = st.get_mut(&name_id)
                    .ok_or("corrupt ktz: blob event references a missing name stream")?;
                let bstart = s.blobs_pos;
                let bend = bstart
                    .checked_add(blen)
                    .filter(|&e| e <= l.blobs.end - l.blobs.start)
                    .ok_or("corrupt ktz: blob length overruns stream")?;
                let region = &payload[l.blobs.start + bstart..l.blobs.start + bend];
                let mut q = Cur::new(region);
                while q.p < region.len() {
                    let tag = q.byte()?;
                    if tag != TK_STR {
                        return Err("corrupt ktz: blob key is not a string token".into());
                    }
                    let kid = q.varint()?;
                    let key = sym(payload, syms, kid)?;
                    let is_args = key == b"\"args\"";
                    let is_id = key == b"\"id\"";
                    let vstart = scratch.len();
                    emit_tok(&mut q, kid, payload, syms, dmap, scratch, 0)?;
                    if is_args && keep_args {
                        let len = scratch.len() - vstart;
                        let off = out.args.len();
                        out.args.extend_from_slice(&scratch[vstart..]);
                        args_off = off as u32;
                        args_len = if len <= u16::MAX as usize { len as u16 } else { 0 };
                        args_end = out.args.len();
                    }
                    if is_id && scratch.get(vstart).is_some_and(|b| b.is_ascii_digit()) {
                        flow_id = parse_f64(&scratch[vstart..]) as u64;
                    }
                }
                s.blobs_pos += blen;
            }

            if ph == b'X' && has_dur {
                out.min_ts = out.min_ts.min(ts);
                out.max_ts = out.max_ts.max(ts + dur);
                out.events.push((pid, tid, Event {
                    ts, dur, name, cat, args_off, depth: 0, args_len,
                }));
                out.total_events += 1;
            } else if ph == b'M' {
                if args_off != 0 && out.names.str(name) == "thread_name" {
                    let mut tmp_strs = InternTable::new();
                    let mut tmp_pairs = Vec::new();
                    parse_args_flat(&out.args[args_off as usize..args_end], &mut tmp_strs, &mut tmp_pairs);
                    for &[k, v] in &tmp_pairs {
                        if (k as usize) < tmp_strs.len() && tmp_strs.str(k) == "name" {
                            if (v as usize) < tmp_strs.len() {
                                out.thread_names.insert((pid, tid), tmp_strs.str(v).to_string());
                            }
                            break;
                        }
                    }
                }
            } else if ph == b's' || ph == b'f' {
                out.flows.push((flow_id, pid, tid, ts, ph == b's'));
            }

            counter.fetch_add(1, Ordering::Relaxed);
        }

        for i in 0..7 {
            col_pos[i] = cc[i].p;
        }
        Ok(*next_event >= *n_events)
    }
}
