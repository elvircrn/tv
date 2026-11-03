/// Interning table for parse-time strings (event names, categories, arg
/// keys/values): byte string -> u32 id, where the id is an *index into
/// `strings* — no pointers anywhere.
///
/// Data-oriented replacement for the old `(Vec<String>, HashMap<u64, u32>)`
/// pair keyed by the string's fnv1a hash: the hash table is one flat
/// open-addressed `Vec<u64>` of slots instead of hashbrown's bucket array.
/// Each slot packs the id in the low 32 bits (stored as id+1, so a slot of
/// 0 means empty) and the HIGH 32 bits of the hash in the upper bits — the
/// tag rides the same cache line as the id and is checked before the second
/// access into `hashes` that confirms the full 64-bit hash (preserving the
/// exact hash-identity semantics of the map this replaced). Linear probing
/// plus a 75% load cap keeps lookups to a slot or two for the small tables
/// interning produces (real traces have hundreds to low thousands of
/// distinct names, not millions).
pub struct InternTable {
    strings: Vec<String>,
    hashes: Vec<u64>,
    slots: Vec<u64>,
    mask: usize,
}

const INTERN_MIN_SLOTS: usize = 64;

impl InternTable {
    /// An empty table with the empty string pre-interned at id 0 — the old
    /// tables' `insert(0, 0)` seed, so id 0 always means "".
    pub fn new() -> Self {
        let mut t = Self {
            strings: Vec::new(),
            hashes: Vec::new(),
            slots: vec![0; INTERN_MIN_SLOTS],
            mask: INTERN_MIN_SLOTS - 1,
        };
        t.insert_entry(0, String::new());
        t
    }

    pub fn len(&self) -> usize { self.strings.len() }

    pub fn str(&self, id: u32) -> &str { &self.strings[id as usize] }

    /// Hand off the interned strings (for building a `Trace`, whose name/
    /// cat columns are plain `Vec<String>` once loading is done).
    pub fn into_strings(self) -> Vec<String> { self.strings }

    /// id for `bytes`, interning it if new. Identity is the fnv1a hash,
    /// exactly as in the `HashMap<u64, u32>` this replaced.
    pub fn intern(&mut self, bytes: &[u8]) -> u32 {
        let hash = fnv1a(bytes);
        if let Some(id) = self.lookup(hash) { return id; }
        let s = match String::from_utf8(bytes.to_vec()) {
            Ok(s) => s,
            Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
        };
        self.insert_entry(hash, s)
    }

    /// id for `s`, interning it if new (merge paths, which hold `&str`s).
    pub fn intern_str(&mut self, s: &str) -> u32 {
        let hash = fnv1a(s.as_bytes());
        if let Some(id) = self.lookup(hash) { return id; }
        self.insert_entry(hash, s.to_string())
    }

    /// Remap a local table's ids onto this (global) table, interning any
    /// string the global side hasn't seen. Replaces `merge_intern_table` /
    /// `merge_intern_direct`: iterating the local side is a straight walk
    /// over its parallel `hashes`/`strings` arrays — no map entries to
    /// chase.
    pub fn merge_from(&mut self, local: &InternTable) -> Vec<u32> {
        let mut remap = vec![0u32; local.strings.len()];
        for (i, (&hash, s)) in local.hashes.iter().zip(&local.strings).enumerate() {
            remap[i] = self.lookup(hash).unwrap_or_else(|| self.insert_entry(hash, s.clone()));
        }
        remap
    }

    /// `merge_from` for local sides that are plain string lists (final
    /// `Trace`s in the diff/merge path, which no longer carry tables).
    pub fn merge_strings(&mut self, local: &[String]) -> Vec<u32> {
        local.iter().map(|s| self.intern_str(s)).collect()
    }

    fn insert_entry(&mut self, hash: u64, s: String) -> u32 {
        let id = self.strings.len() as u32;
        self.strings.push(s);
        self.hashes.push(hash);
        self.place(hash, id as u64);
        self.maybe_grow();
        id
    }

    fn maybe_grow(&mut self) {
        if self.strings.len() * 4 > self.slots.len() * 3 {
            self.slots = vec![0; self.slots.len() * 2];
            self.mask = self.slots.len() - 1;
            for id in 0..self.hashes.len() {
                let h = self.hashes[id];
                self.place(h, id as u64);
            }
        }
    }

    /// Drop `hash -> id` into the first free probe slot. Caller guarantees
    /// one exists (load factor is capped at 75%).
    fn place(&mut self, hash: u64, id: u64) {
        let mut slot = (hash as usize) & self.mask;
        while self.slots[slot] != 0 { slot = (slot + 1) & self.mask; }
        self.slots[slot] = (hash & 0xffff_ffff_0000_0000) | (id + 1);
    }

    fn lookup(&self, hash: u64) -> Option<u32> {
        let mut slot = (hash as usize) & self.mask;
        loop {
            let v = self.slots[slot];
            if v == 0 { return None; }
            let id = (v & 0xffff_ffff) as usize - 1;
            if v >> 32 == hash >> 32 && self.hashes[id] == hash {
                return Some(id as u32);
            }
            slot = (slot + 1) & self.mask;
        }
    }
}

impl Default for InternTable {
    fn default() -> Self { Self::new() }
}

/// Find `"key": <integer>` at/after `start` in a JSON blob (works through
/// nested objects, since `find_key` is a plain byte search). Returns None if
/// absent or non-numeric.
pub fn find_int_field(blob: &[u8], start: usize, key: &[u8]) -> Option<i64> {
    let kp = find_key(&blob[start..], key)? + start;
    let mut q = kp + key.len() + 2; // past `"key"`
    q = skip_ws(blob, q);
    if q < blob.len() && blob[q] == b':' { q += 1; }
    q = skip_ws(blob, q);
    let neg = q < blob.len() && blob[q] == b'-';
    if neg { q += 1; }
    let s = q;
    while q < blob.len() && blob[q].is_ascii_digit() { q += 1; }
    if q == s { return None; }
    let n: i64 = std::str::from_utf8(&blob[s..q]).ok()?.parse().ok()?;
    Some(if neg { -n } else { n })
}

/// Find `"key": "value"` anywhere in a small JSON blob and return the string
/// value — including inside a nested object, since `find_key` is a plain byte
/// search. Good enough for one-off lookups in per-event args (e.g. CUPTI's
/// `occupancy.limitingFactors`) where a full nested-object parser would be
/// overkill for a blob that's typically a few hundred bytes.
pub fn find_str_field<'a>(blob: &'a [u8], key: &[u8]) -> Option<&'a str> {
    let kp = find_key(blob, key)?;
    let mut q = kp + key.len() + 2; // past `"key"`
    q = skip_ws(blob, q);
    if q < blob.len() && blob[q] == b':' { q += 1; }
    q = skip_ws(blob, q);
    if q < blob.len() && blob[q] == b'"' {
        let end = skip_string(blob, q);
        return std::str::from_utf8(&blob[q + 1..end - 1]).ok();
    }
    None
}

pub fn find_key(raw: &[u8], key: &[u8]) -> Option<usize> {
    let mut buf = [0u8; 66];
    let len = key.len() + 2;
    buf[0] = b'"';
    buf[1..1 + key.len()].copy_from_slice(key);
    buf[1 + key.len()] = b'"';
    memchr::memmem::find(raw, &buf[..len])
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

pub fn skip_ws(raw: &[u8], mut pos: usize) -> usize {
    while pos < raw.len() && matches!(raw[pos], b' ' | b'\n' | b'\r' | b'\t') { pos += 1; }
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
    strs: &mut InternTable,
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
        let ki = strs.intern(&blob[ks..ke]);
        let val_bytes = if blob[vs] == b'"' { &blob[vs + 1..p - 1] } else { &blob[vs..p] };
        let vi = strs.intern(val_bytes);
        pairs.push([ki, vi]);
    }
}

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn json_unescape(s: &str) -> String {
    if !s.contains('\\') { return s.to_string(); }
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
