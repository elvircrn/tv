use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
pub struct IdentityHasher(u64);
impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, _bytes: &[u8]) { unreachable!() }
    fn write_u64(&mut self, i: u64) { self.0 = i; }
}
pub type FnvMap<V> = HashMap<u64, V, BuildHasherDefault<IdentityHasher>>;

pub fn find_key(raw: &[u8], key: &[u8]) -> Option<usize> {
    let mut needle = Vec::with_capacity(key.len() + 2);
    needle.push(b'"');
    needle.extend_from_slice(key);
    needle.push(b'"');
    let finder = memchr::memmem::Finder::new(&needle);
    finder.find(raw)
}

pub fn skip_string(raw: &[u8], pos: usize) -> usize {
    let mut i = pos + 1;
    loop {
        match memchr::memchr2(b'"', b'\\', &raw[i..]) {
            Some(off) => {
                i += off;
                if raw[i] == b'"' { return i + 1; }
                i += 2;
            }
            None => return raw.len(),
        }
    }
}

pub fn skip_value(raw: &[u8], pos: usize) -> usize {
    if pos >= raw.len() { return pos; }
    match raw[pos] {
        b'"' => skip_string(raw, pos),
        b'{' | b'[' => skip_container(raw, pos),
        _ => {
            let mut i = pos;
            while i < raw.len()
                && !matches!(raw[i], b',' | b'}' | b']' | b' ' | b'\n' | b'\r' | b'\t')
            { i += 1; }
            i
        }
    }
}

fn skip_container(raw: &[u8], pos: usize) -> usize {
    let mut depth = 1u32;
    let mut i = pos + 1;
    while depth > 0 && i < raw.len() {
        match memchr::memchr3(b'"', b'{', b'}', &raw[i..]) {
            Some(off) => {
                for j in i..i + off {
                    match raw[j] {
                        b'[' => depth += 1,
                        b']' => { depth -= 1; if depth == 0 { return j + 1; } }
                        _ => {}
                    }
                }
                i += off;
                match raw[i] {
                    b'"' => i = skip_string(raw, i),
                    b'{' => { depth += 1; i += 1; }
                    b'}' => { depth -= 1; i += 1; }
                    _ => unreachable!(),
                }
            }
            None => {
                for j in i..raw.len() {
                    match raw[j] {
                        b'[' => depth += 1,
                        b']' => { depth -= 1; if depth == 0 { return j + 1; } }
                        _ => {}
                    }
                }
                return raw.len();
            }
        }
    }
    i
}

pub fn skip_value_hashed(raw: &[u8], pos: usize) -> (usize, u64) {
    let end = skip_value(raw, pos);
    let hash = fnv1a(&raw[pos..end]);
    (end, hash)
}

pub fn skip_number(raw: &[u8], pos: usize) -> usize {
    let mut i = pos;
    while i < raw.len()
        && (raw[i].is_ascii_digit() || matches!(raw[i], b'.' | b'-' | b'+' | b'e' | b'E'))
    { i += 1; }
    i
}

pub fn skip_ws(raw: &[u8], mut pos: usize) -> usize {
    while pos < raw.len() && matches!(raw[pos], b' ' | b'\n' | b'\r' | b'\t') { pos += 1; }
    pos
}

pub fn skip_ws_comma(raw: &[u8], mut pos: usize) -> usize {
    while pos < raw.len() && matches!(raw[pos], b' ' | b'\n' | b'\r' | b'\t' | b',') { pos += 1; }
    pos
}

const POW10: [f64; 23] = {
    let mut t = [1.0; 23];
    let mut i = 1;
    while i < 23 {
        t[i] = t[i - 1] * 10.0;
        i += 1;
    }
    t
};

pub fn parse_f64(bytes: &[u8]) -> f64 {
    if bytes.is_empty() { return 0.0; }
    let mut i = 0;
    let neg = bytes[0] == b'-';
    if neg { i += 1; }
    if i >= bytes.len() || (!bytes[i].is_ascii_digit() && bytes[i] != b'.') {
        return unsafe { std::str::from_utf8_unchecked(bytes) }.parse().unwrap_or(0.0);
    }
    let mut int: u64 = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        int = int.wrapping_mul(10).wrapping_add((bytes[i] - b'0') as u64);
        i += 1;
    }
    let mut frac: u64 = 0;
    let mut frac_digits: usize = 0;
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            frac = frac.wrapping_mul(10).wrapping_add((bytes[i] - b'0') as u64);
            frac_digits += 1;
            i += 1;
        }
    }
    let mut val = int as f64;
    if frac_digits > 0 {
        val += frac as f64 / if frac_digits < POW10.len() { POW10[frac_digits] } else { 10f64.powi(frac_digits as i32) };
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        let eneg = i < bytes.len() && bytes[i] == b'-';
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') { i += 1; }
        let mut exp: i32 = 0;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            exp = exp * 10 + (bytes[i] - b'0') as i32;
            i += 1;
        }
        if eneg { exp = -exp; }
        let abs = exp.unsigned_abs() as usize;
        if abs < POW10.len() {
            val = if exp >= 0 { val * POW10[abs] } else { val / POW10[abs] };
        } else {
            val *= 10f64.powi(exp);
        }
    }
    if neg { -val } else { val }
}

pub fn parse_args_flat(
    blob: &[u8],
    strs: &mut Vec<String>,
    idx: &mut FnvMap<u32>,
    pairs: &mut Vec<[u32; 2]>,
) {
    if blob.is_empty() || blob[0] != b'{' { return; }
    let mut p = 1;
    loop {
        p = skip_ws(blob, p);
        if p >= blob.len() || blob[p] == b'}' { return; }
        if blob[p] == b',' { p += 1; continue; }
        if blob[p] != b'"' { return; }
        let ks = p + 1;
        p = skip_string(blob, p);
        let ke = p - 1;
        p = skip_ws(blob, p);
        if p < blob.len() && blob[p] == b':' { p += 1; }
        p = skip_ws(blob, p);
        let vs = p;
        p = skip_value(blob, p);
        let ki = intern(&blob[ks..ke], strs, idx);
        let val_bytes = if blob[vs] == b'"' { &blob[vs + 1..p - 1] } else { &blob[vs..p] };
        let vi = intern(val_bytes, strs, idx);
        pairs.push([ki, vi]);
    }
}

pub fn intern(bytes: &[u8], table: &mut Vec<String>, index: &mut FnvMap<u32>) -> u32 {
    let hash = fnv1a(bytes);
    if let Some(&idx) = index.get(&hash) { return idx; }
    let idx = table.len() as u32;
    table.push(String::from_utf8_lossy(bytes).into_owned());
    index.insert(hash, idx);
    idx
}

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn find_split_points(raw: &[u8], start: usize, n: usize) -> Vec<usize> {
    if n <= 1 { return vec![start, raw.len()]; }
    let remaining = raw.len().saturating_sub(start);
    let chunk_size = remaining / n;
    let mut points = vec![start];
    let ph = memchr::memmem::Finder::new(b"\"ph\"");

    for t in 1..n {
        let target = start + t * chunk_size;
        let limit = raw.len().min(target + chunk_size);
        let mut pos = target;
        while pos < limit {
            match memchr::memchr(b'\n', &raw[pos..limit]) {
                Some(off) => {
                    let mut p = pos + off + 1;
                    while p < raw.len() && matches!(raw[p], b' ' | b'\t' | b'\r' | b'\n') { p += 1; }
                    if p < raw.len() && raw[p] == b'{' {
                        let check_end = raw.len().min(p + 500);
                        if ph.find(&raw[p..check_end]).is_some() {
                            points.push(p);
                            break;
                        }
                    }
                    pos += off + 1;
                }
                None => break,
            }
        }
    }
    points.push(raw.len());
    points
}

pub fn json_unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some(other) => { out.push('\\'); out.push(other); }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}
