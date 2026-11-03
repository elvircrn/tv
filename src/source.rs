//! Source cross-referencing: given a path to a vLLM checkout, resolve trace
//! events to the source lines they came from and open them in the user's
//! editor (`code -g file:line` by default — tv resolves, the editor renders).
//! Three linkage mechanisms, in falling order of precision:
//!
//! 1. `CallFrom` args on python_function events — exact `file:line` relative
//!    to the repo root (kineto's with_stack output).
//! 2. Triton/Tilelang kernel names — the kernel IS the `def` name of the
//!    jitted python function, so a `def NAME(` index over the repo's .py
//!    files resolves them exactly.
//! 3. csrc CUDA kernels — the trace carries demangled names
//!    (`void vllm::ns::FusedThing<c10::BFloat16>(...)`); the base name is
//!    searched in the C-family sources by content.
//!
//! aten/CUB/CUTLASS kernels are not in the vLLM repo and resolve to nothing;
//! the UI says so rather than guessing.
//!
//! Everything fs-touching is native-only; the wasm build has no checkout to
//! read, so the tab is cfg'd out there.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ============================================================== hits + lookup

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Py,
    CFamily,
}

impl Lang {
    fn of_path(p: &Path) -> Option<Self> {
        match p.extension().and_then(|e| e.to_str())? {
            "py" | "pyi" => Some(Lang::Py),
            "cu" | "cuh" | "cc" | "cpp" | "cxx" | "c" | "h" | "hpp" => Some(Lang::CFamily),
            _ => None,
        }
    }
}

/// One resolved source location.
#[derive(Clone)]
pub struct SourceHit {
    pub path: PathBuf,
    /// 1-based line the trace points at.
    pub line: u32,
    /// Path for display, relative to the checkout root.
    pub rel: String,
}

/// The demangled kernel name's base identifier: what you'd grep for.
/// `void vllm::deepseek_v4_fused_ops::FusedThing<c10::BFloat16, 0>(args)`
/// -> `FusedThing`; `mhc_pre_big_fuse_tilelang_kernel` stays itself.
pub fn kernel_base_name(kernel: &str) -> String {
    let mut s = kernel.trim();
    if let Some(rest) = s.strip_prefix("void ") { s = rest; }
    // Templates and parameter lists are not part of the searchable name.
    let s = s.split('<').next().unwrap_or(s);
    let s = s.split('(').next().unwrap_or(s);
    // The function name is the last namespace segment.
    let s = s.rsplit("::").next().unwrap_or(s);
    s.trim().to_string()
}

/// `vllm/v1/executor/x.py:942` -> the part before the last ':' (strip the
/// line number). Returns None for strings without a colon.
fn split_callfrom(s: &str) -> Option<(&str, u32)> {
    let (file, line) = s.rsplit_once(':')?;
    let line = line.parse().ok()?;
    Some((file, line))
}

// ============================================================ the "index"

/// A configured vLLM checkout. Nothing is precomputed — the old design
/// walked the whole tree at startup (8k files, ~1s, every launch) to build
/// a def index that silently went stale the moment you checked out a
/// different branch. Instead, each kernel name is resolved by an on-demand
/// parallel grep of the tree (`def NAME(` in python, `NAME(` at a word
/// boundary in C-family files), memoized so the scan runs once per
/// distinct name. A click is user-driven and rare; paying ~200ms on the
/// first click of each new kernel beats paying a full-tree walk on every
/// launch, and it always reflects the tree as it is right now.
pub struct SourceIndex {
    root: PathBuf,
    /// Kernel base name -> hits, the scan memo. A kernel repeats on every
    /// one of its thousands of launches; only distinct names scan.
    memo: HashMap<String, Vec<SourceHit>>,
    /// `git describe --tags --always --dirty`, computed on first use (one
    /// process spawn) and cached for the session.
    git_desc: Option<String>,
}

/// Directory names never worth walking into.
fn skip_dir(name: &str) -> bool {
    matches!(name, ".git" | "build" | "node_modules" | ".venv" | "venv"
        | "__pycache__" | ".mypy_cache" | ".pytest_cache" | "target" | "cmake-build-debug")
}

impl SourceIndex {
    pub fn new(root: PathBuf) -> SourceIndex {
        SourceIndex { root, memo: HashMap::new(), git_desc: None }
    }

    /// `git describe --tags --always --dirty` in the checkout, for the
    /// banner and the drift warning. Cached after the first call. No
    /// filesystem on wasm: always unknown there.
    pub fn git_desc(&mut self) -> Option<String> {
        #[cfg(not(target_arch = "wasm32"))]
        if self.git_desc.is_none() { self.git_desc = self.run_git_describe(); }
        self.git_desc.clone()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn run_git_describe(&self) -> Option<String> {
        let out = std::process::Command::new("git")
            .arg("-C").arg(&self.root)
            .args(["describe", "--tags", "--always", "--dirty"])
            .output()
            .ok()?;
        if !out.status.success() { return None; }
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if s.is_empty() { None } else { Some(s) }
    }

    fn hit(&self, path: PathBuf, line: u32) -> SourceHit {
        let rel = path.strip_prefix(&self.root)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| path.to_string_lossy().into_owned());
        SourceHit { rel, path, line }
    }

    /// Exact `vllm/v1/executor/x.py:942` resolution against the root.
    /// Rejects absolute paths and `..` so a hostile CallFrom can't point
    /// outside the checkout.
    pub fn resolve_callfrom(&self, s: &str) -> Option<SourceHit> {
        let (file, line) = split_callfrom(s)?;
        if file.starts_with('/') || file.split('/').any(|seg| seg == "..") { return None; }
        let path = self.root.join(file);
        if !path.is_file() { return None; }
        Some(self.hit(path, line))
    }

    /// Kernel-name resolution: on-demand scan, python defs first
    /// (triton/tilelang) then C-family content hits. If the exact name
    /// misses, the trailing `_`-separated segment is dropped — but ONLY
    /// when it looks like triton's mangled dims suffix (`0d1d2`), never a
    /// real name part: dropping `kernel` from
    /// `vectorized_elementwise_kernel` once produced `vectorized`, which
    /// substring-matched a different function and sent the viewer to a
    /// confidently wrong line. Up to two drops.
    pub fn resolve_kernel(&mut self, kernel: &str) -> Vec<SourceHit> {
        let mut base = kernel_base_name(kernel);
        if base.is_empty() { return Vec::new(); }
        for attempt in 0..3 {
            if let Some(v) = self.memo.get(&base) { return v.clone(); }
            let hits = self.scan(&base);
            self.memo.insert(base.clone(), hits.clone());
            if !hits.is_empty() { return hits; }
            // Drop the trailing triton dims: `name_0d1d2` -> `name`.
            let next = base.rsplit_once('_')
                .filter(|(_, seg)| is_triton_dims(seg))
                .map(|(h, _)| h.to_string());
            match next {
                Some(n) if attempt < 2 => base = n,
                _ => break,
            }
        }
        Vec::new()
    }

    /// One grep of the tree for `name`: `def name(` in python files, `name(`
    /// at an identifier word boundary in C-family files (the boundary
    /// matters — without it `vectorized(` matches inside
    /// `scaled_fp8_conversion_vectorized(`). Walk + read are parallel
    /// across files; python hits sort ahead of C hits because triton and
    /// tilelang kernels ARE python defs. Capped at 16 so a generic name
    /// can't produce an unbounded list.
    #[cfg(not(target_arch = "wasm32"))]
    fn scan(&self, name: &str) -> Vec<SourceHit> {
        use rayon::prelude::*;
        let t0 = std::time::Instant::now();
        let py_needle: Vec<u8> = format!("def {name}(").into_bytes();
        let c_needle: Vec<u8> = format!("{name}(").into_bytes();
        let mut files = Vec::new();
        self.collect_files(&self.root, &mut files);
        let (py, c): (Vec<SourceHit>, Vec<SourceHit>) = files
            .par_iter()
            .filter_map(|path| {
                let Ok(text) = std::fs::read(path) else { return None };
                let needle = match Lang::of_path(path) {
                    Some(Lang::Py) => &py_needle,
                    _ => &c_needle,
                };
                let mut out = Vec::new();
                // Line-by-line: a definition never spans lines.
                for (i, line) in text.split(|&b| b == b'\n').enumerate() {
                    if find_def(line, needle) {
                        out.push((i + 1) as u32);
                    }
                }
                if out.is_empty() { None } else { Some((path, out)) }
            })
            .map(|(path, lines)| {
                let py = Lang::of_path(path) == Some(Lang::Py);
                let hits: Vec<SourceHit> = lines.into_iter()
                    .map(|l| self.hit(path.clone(), l)).collect();
                if py { (hits, Vec::new()) } else { (Vec::new(), hits) }
            })
            .reduce(|| (Vec::new(), Vec::new()), |(mut pa, mut ca), (pb, cb)| {
                pa.extend(pb); ca.extend(cb); (pa, ca)
            });
        let mut hits = py;
        hits.extend(c);
        hits.truncate(16);
        let elapsed = t0.elapsed().as_secs_f64();
        if elapsed > 0.25 {
            crate::tlog!("source scan({name}): {} hits, {} files, {:.2}s",
                hits.len(), files.len(), elapsed);
        }
        hits
    }

    /// wasm stub: no filesystem to search. (resolve_kernel is compiled on
    /// wasm because Pane state references its types, but the Source tab UI
    /// that drives it is native-only, so it's never reached.)
    #[cfg(target_arch = "wasm32")]
    fn scan(&self, _name: &str) -> Vec<SourceHit> { Vec::new() }

    /// Recursive collect of the python + C-family files under `dir`.
    /// Re-walked per scan (the walk is cheap; the reads dominate) so a
    /// branch switch between clicks is picked up, not served stale.
    #[cfg(not(target_arch = "wasm32"))]
    fn collect_files(&self, dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for entry in rd.flatten() {
            let Ok(ft) = entry.file_type() else { continue };
            let path = entry.path();
            if ft.is_dir() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if !skip_dir(&name) { self.collect_files(&path, out); }
            } else if ft.is_file() && Lang::of_path(&path).is_some() {
                out.push(path);
            }
        }
    }
}

/// Is this `_`-separated name segment a triton dims suffix (`0d1d2`,
/// `1d3d5`)? Digits and 'd's only, non-empty.
fn is_triton_dims(seg: &str) -> bool {
    !seg.is_empty() && seg.chars().all(|c| c.is_ascii_digit() || c == 'd')
}

/// `needle` occurs in `line` at an identifier-word boundary. The char after
/// is always '(' (part of the needle), so only the char before matters.
fn find_def(line: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || line.len() < needle.len() { return false; }
    for i in 0..=line.len() - needle.len() {
        if &line[i..i + needle.len()] == needle
            && (i == 0 || !(line[i - 1].is_ascii_alphanumeric() || line[i - 1] == b'_'))
        {
            return true;
        }
    }
    false
}

// ============================================================== editor jump

/// torch/CUB/CUTLASS-namespace names: their source lives in pytorch, not
/// the vLLM checkout, so resolving them would only ever produce junk
/// (worst case a call site that happens to textually match — `c10d::send`
/// once landed on an unrelated `def send(` in a test file).
pub fn is_torch_internal(name: &str) -> bool {
    let n = name.trim();
    n.starts_with("aten::") || n.starts_with("c10::") || n.starts_with("c10d::")
        || n.starts_with("void at::") || n.starts_with("void c10::")
        || n.starts_with("at::native::") || n.starts_with("(anonymous namespace)::at::")
        || n.starts_with("cub::") || n.starts_with("cutlass")
}

/// The build commit a trace was built from, per its `vllm_version`, when
/// the string actually carries one. That's setuptools-scm's `+g<hex>`
/// (`0.26.1rc1.dev528+g1a2b3c4d5e6f`) — and ONLY that: a local tag like
/// `+llmd.cfee12e...` is a pipeline stamp that matches no repo, so
/// treating its hex as a commit is how jumps end up on a drifted
/// ~/vllm with a bogus "not the build commit" warning. Stamped dev
/// versions are recovered by distance instead (see `BuildRef::Dev`).
pub fn trace_build_commit(vllm_version: &str) -> Option<&str> {
    let hex = vllm_version.rsplit('+').next()?.strip_prefix('g')?;
    (hex.len() >= 7 && hex.chars().all(|c| c.is_ascii_hexdigit())).then_some(hex)
}

/// `code` -> ["code", "/usr/local/bin/code", ...]. An editor given as a
/// path is used as-is.
fn editor_candidates(editor: &str) -> Vec<String> {
    if editor.contains('/') { return vec![editor.to_string()]; }
    let mut cands = vec![editor.to_string()];
    if editor == "code" {
        // A GUI-launched app doesn't inherit the shell's PATH, so a bare
        // `code` fails even when the CLI is installed; try the usual spots.
        cands.push("/usr/local/bin/code".into());
        cands.push("/opt/homebrew/bin/code".into());
        if let Ok(home) = std::env::var("HOME") {
            cands.push(format!("{home}/.vscode/bin/code"));
        }
        cands.push("/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code".into());
    }
    cands
}

/// Percent-encode a filesystem path for a `vscode://file/` URI: keep the
/// URI unreserved set plus `/`, escape everything else (spaces, the
/// unicode the default glyph rules wouldn't render, ...).
pub(crate) fn encode_uri_path(p: &str) -> String {
    let mut out = String::with_capacity(p.len());
    for b in p.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9'
                | b'-' | b'_' | b'.' | b'~' | b'/' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Open `path:line` in the user's editor. `editor` is the editor's CLI
/// command ("code", "cursor", ...); Zed takes `file:line` directly, the
/// rest take `-g file:line`. VS Code is special: `-g` positions the
/// cursor only on a fresh open — when the file is already open in a
/// window it just focuses the tab and leaves the cursor where it was —
/// so `code` jumps via the `vscode://file/<path>:<line>` protocol URL
/// instead, whose handler applies the selection to an open editor too.
#[cfg(not(target_arch = "wasm32"))]
pub fn open_in_editor(editor: &str, hit: &SourceHit) -> Result<(), String> {
    let loc = format!("{}:{}", hit.path.display(), hit.line);
    for exe in editor_candidates(editor) {
        let mut c = std::process::Command::new(&exe);
        if editor == "zed" {
            c.arg(&loc);
        } else if editor == "code" {
            c.args(["--open-url",
                &format!("vscode://file/{}:{}", encode_uri_path(&hit.path.display().to_string()), hit.line)]);
        } else {
            c.arg("-g").arg(&loc);
        }
        match c.spawn() {
            Ok(_) => return Ok(()),
            // Not found -> try the next install location; anything else
            // (permission denied, ...) is a real failure, don't retry.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("couldn't launch `{}`: {e}", exe)),
        }
    }
    Err(format!(
        "couldn't launch `{}` (tried PATH and the usual install dirs - is the editor's shell command installed?)",
        editor))
}

// ===================================================== exact-build cloning

/// What a trace's `vllm_version` pins the source to. Commit (`+g<hex>`),
/// tag (clean release), or — the interesting case — a dev build whose
/// version carries no commit at all (`0.33.0rc1.dev573+llmd.<stamp>`):
/// setuptools-scm wrote that as bump(last tag) + ".dev" + N where N is
/// the commit distance from the tag, and the build commit is the Nth
/// commit after it, recoverable from repo history. `None` when the
/// string pins nothing usable.
#[derive(Clone)]
pub enum BuildRef {
    Commit(String),
    Tag(String),
    /// Dev build at `dist` commits after the tag that `base` bumps.
    Dev { base: String, dist: usize },
}

impl BuildRef {
    /// The identity used for display and failure memoization. For a Dev
    /// ref that's the plain version (the commit isn't known until
    /// inference runs).
    pub fn id(&self) -> String {
        match self {
            BuildRef::Commit(c) => c.clone(),
            BuildRef::Tag(t) => t.clone(),
            BuildRef::Dev { base, dist } => format!("{base}.dev{dist}"),
        }
    }

    /// Does a checkout's `git describe` match this ref? Commits and tags
    /// appear verbatim; a Dev ref's exact tree describes as
    /// `v<tag>-<dist>-g<sha>`, so the `-<dist>-g` marker is the check.
    pub fn matches_desc(&self, desc: &str) -> bool {
        match self {
            BuildRef::Commit(c) => desc.contains(c.as_str()),
            BuildRef::Tag(t) => desc.contains(t.as_str()),
            BuildRef::Dev { dist, .. } => desc.contains(&format!("-{dist}-g")),
        }
    }
}

/// The build ref a trace's `vllm_version` names, or None: a `+g<hex>`
/// commit, else a `.devN` distance (stamped or not), else a clean
/// release version -> tag.
pub fn trace_build_ref(vllm_version: &str) -> Option<BuildRef> {
    if let Some(c) = trace_build_commit(vllm_version) {
        return Some(BuildRef::Commit(c.to_string()));
    }
    let plain = vllm_version.split('+').next().unwrap_or(vllm_version);
    if let Some((base, dist)) = parse_dev(plain) {
        return Some(BuildRef::Dev { base: base.to_string(), dist });
    }
    let v = vllm_version.trim();
    if !v.contains('+') && !v.contains(".dev")
        && v.starts_with(|c: char| c.is_ascii_digit()) && v.contains('.')
    {
        return Some(BuildRef::Tag(format!("v{v}")));
    }
    None
}

/// "0.33.0rc1.dev573" -> ("0.33.0rc1", 573): setuptools-scm's dev count
/// is the commit distance from the last tag at build time.
pub(crate) fn parse_dev(plain: &str) -> Option<(&str, usize)> {
    let (base, n) = plain.rsplit_once(".dev")?;
    if base.is_empty() { return None; }
    Some((base, n.parse().ok()?))
}

/// Tag names that could be the `git describe` anchor for a dev build of
/// `base`: setuptools-scm writes the version as bump(anchor) + ".dev" +
/// distance, so the anchor is usually `base` with its pre-release
/// bumped DOWN (rc1 was built after a rc0 tag); try `base` itself and
/// the family's final release too. Order is best-guess-first; every
/// candidate is validated by describe before use.
pub(crate) fn candidate_anchors(base: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut push = |t: String| {
        if !out.contains(&t) { out.push(t); }
    };
    if let Some((pre, rel)) = base.rsplit_once("rc") {
        if let Ok(n) = rel.parse::<u32>() {
            if n > 0 { push(format!("v{pre}rc{}", n - 1)); }
            else { push(format!("v{pre}")); }
        }
    } else if let Some((pre, patch)) = base.rsplit_once('.') {
        // final base "X.Y.Z" -> the previous patch release
        if let Ok(z) = patch.parse::<u32>() {
            if z > 0 { push(format!("v{pre}.{}", z - 1)); }
        }
    }
    push(format!("v{base}"));
    // family final: "0.33.0rc1" -> "v0.33.0"
    if let Some(end) = base.find(|c: char| !c.is_ascii_digit() && c != '.') {
        push(format!("v{}", &base[..end]));
    }
    out
}

/// Run git, map failure to a message with trimmed stderr.
fn git(dir: Option<&Path>, args: &[&str]) -> Result<String, String> {
    let mut c = std::process::Command::new("git");
    if let Some(d) = dir { c.arg("-C").arg(d); }
    c.args(args);
    let out = c.output().map_err(|e| format!("couldn't run git: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("git {}: {}", args.join(" "), err.trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Where exact builds come from, in order: the user's fork first (their
/// branches' commits live there, not upstream), then vllm-project.
const BUILD_REMOTES: [&str; 2] = [
    "https://github.com/elvircrn/vllm",
    "https://github.com/vllm-project/vllm",
];

/// Resolve an abbreviated commit sha to its full form plus the remote that
/// has it. GitHub's fetch protocol only accepts full shas, but the REST API
/// resolves prefixes — which also doubles as the existence check: a nightly
/// pipeline stamp (`+llmd.cfee12e...`) 404s in every repo, and that's how
/// tv tells "real commit" from "stamp that matches no repo".
fn resolve_full_sha(hex: &str) -> Result<(String, &'static str), String> {
    let mut not_found = false;
    for remote in BUILD_REMOTES {
        let repo = remote.trim_start_matches("https://github.com/");
        let url = format!("https://api.github.com/repos/{repo}/commits/{hex}");
        let out = std::process::Command::new("curl")
            .args(["-s", "-m", "15", &url])
            .output()
            .map_err(|e| format!("couldn't run curl: {e}"))?;
        let body = String::from_utf8_lossy(&out.stdout).into_owned();
        if let Some(full) = extract_json_string(&body, "sha") {
            if full.len() == 40 && full.starts_with(hex) {
                return Ok((full, remote));
            }
            continue;
        }
        if body.contains("No commit found") { not_found = true; continue; }
        if body.contains("rate limit") {
            return Err("github api rate limited - try again in a minute or set a checkout manually".into());
        }
        // Network error / garbage: try the next remote, remember nothing.
    }
    if not_found {
        return Err(format!(
            "{hex} is not a commit in elvircrn/vllm or vllm-project/vllm - it looks like a \
             pipeline build stamp, not a repo commit. Set a checkout manually (or paste the \
             real build commit in the source setup popup)"));
    }
    Err(format!("couldn't resolve {hex} via the github api (network?)"))
}

/// The first `"key": "value"` string in `body` (whitespace-tolerant — the
/// api pretty-prints). Crude on purpose: the response's first field is the
/// sha we want, and pulling in a json crate for one field isn't worth it.
pub(crate) fn extract_json_string(body: &str, key: &str) -> Option<String> {
    let key_at = body.find(&format!("\"{key}\""))?;
    let colon = body[key_at..].find(':')? + key_at;
    let open = body[colon..].find('"')? + colon;
    let rest = &body[open + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Recover the exact commit a stamped dev version was built from, given
/// the object cache. setuptools-scm wrote `0.33.0rc1.dev573` as
/// bump(tag) + ".dev" + N from the build commit's `git describe`
/// (`v0.33.0rc0-573-g<sha>`), so the commit is the Nth commit after the
/// anchor tag on main. Needs main's commit history in the cache — a
/// blobless deepen, incremental after the first time — plus the anchor
/// tag (depth 1). Each candidate anchor is validated by re-describing
/// the picked commit, so a wrong anchor tag is rejected, not guessed at.
/// Upstream main is the truth for distance (CI builds from it); the fork
/// is the fallback when upstream is unreachable.
fn infer_dev_commit(cache: &Path, base: &str, dist: usize) -> Result<String, String> {
    let marker = format!("-{dist}-g");
    let mut errs = String::new();
    // Upstream first for this one, reverse of BUILD_REMOTES' fork-first
    // order: the fork's main can lag or carry extra commits, which would
    // shift the walk even when the distance check still validates.
    for remote in BUILD_REMOTES.iter().rev() {
        set_origin(cache, remote)?;
        // --depth=100000: the cache is shallow from per-commit fetches,
        // and a depthless fetch into a shallow repo keeps the old
        // boundary. Blobless, so this is commits + trees only.
        if let Err(e) = git(Some(cache), &["fetch", "--filter=blob:none",
            "--depth=100000", remote, "+refs/heads/main:refs/tv/main"])
        {
            errs.push_str(&format!("{e}; "));
            continue;
        }
        for tag in candidate_anchors(base) {
            let have = git(Some(cache), &["rev-parse", "--verify", "-q",
                &format!("refs/tags/{tag}^{{commit}}")]).is_ok();
            if !have {
                let spec = format!("refs/tags/{tag}:refs/tags/{tag}");
                let _ = git(Some(cache), &["fetch", "--depth", "1",
                    "--filter=blob:none", remote, &spec]);
                if git(Some(cache), &["rev-parse", "--verify", "-q",
                    &format!("refs/tags/{tag}^{{commit}}")]).is_err()
                {
                    continue;
                }
            }
            let range = format!("refs/tags/{tag}..refs/tv/main");
            let Ok(count) = git(Some(cache), &["rev-list", "--count", &range])
                .and_then(|c| c.parse::<usize>().map_err(|e| e.to_string()))
            else { continue };
            if count < dist { continue; }
            let Ok(c) = git(Some(cache), &["rev-list", "--skip",
                &(count - dist).to_string(), "--max-count", "1", &range])
            else { continue };
            // Validate: the picked commit must describe as exactly `dist`
            // commits after a tag — a wrong anchor gives some other
            // distance and is discarded.
            if git(Some(cache), &["describe", "--tags", &c])
                .is_ok_and(|d| d.contains(&marker))
            {
                return Ok(c);
            }
        }
    }
    Err(format!(
        "couldn't infer the build commit for {base}.dev{dist} from repo \
         history ({errs}no anchor tag walked to a matching commit)"))
}

/// Materialize the exact tree a trace was built from, as cheaply as the
/// network allows: a blobless depth-1 fetch (commit + trees only, no file
/// contents) into a persistent object cache (`~/.cache/tv/vllm.git`), then
/// a sparse worktree at `/tmp/vllm-<short>` holding only `vllm/` and
/// `csrc/` — the two trees source resolution scans — so the lazy blob
/// fetch during checkout downloads just those, not docs/, tests/, or
/// examples/. Returns the worktree path plus its `git describe` (the full
/// sha when tags weren't fetched, which still contains the abbreviated
/// hex, so the drift check passes). Reuses an existing worktree for the
/// same ref. Network: one api call (commits only, and only when the cache
/// lacks the sha) plus the filtered fetch.
pub fn materialize_build(build: &BuildRef) -> Result<(PathBuf, String), String> {
    let home = std::env::var("HOME").map_err(|_| "no $HOME".to_string())?;
    let cache = PathBuf::from(&home).join(".cache/tv/vllm.git");
    if !cache.is_dir() {
        git(None, &["init", "--bare", &cache.to_string_lossy()])?;
    }
    // A dev-distance ref resolves to its commit first (history walk in
    // the cache); after that it's an ordinary commit, already fetched.
    let inferred;
    let build = match build {
        BuildRef::Dev { base, dist } => {
            inferred = BuildRef::Commit(infer_dev_commit(&cache, base, *dist)?);
            &inferred
        }
        b => b,
    };
    // What the worktree checks out, and the /tmp dir name. Fetches go
    // through `origin` — the promisor remote lazy blob fetches use —
    // pointed at wherever the ref actually lives.
    let (checkout_ref, short) = match build {
        BuildRef::Dev { .. } => unreachable!("resolved to a commit above"),
        BuildRef::Commit(hex) => {
            // Fetch needs the full sha; the cache may already have it.
            if git(Some(&cache), &["cat-file", "-e", &format!("{hex}^{{commit}}")]).is_err() {
                let (full, remote) = resolve_full_sha(hex)?;
                set_origin(&cache, remote)?;
                git(Some(&cache), &["fetch", "--depth", "1", "--filter=blob:none", "origin", &full])?;
            }
            (hex.clone(), hex[..hex.len().min(10)].to_string())
        }
        BuildRef::Tag(t) => {
            if git(Some(&cache), &["rev-parse", "--verify", "-q", &format!("refs/tags/{t}")]).is_err() {
                fetch_tag(&cache, t)?;
            }
            (t.clone(), t.trim_start_matches('v').to_string())
        }
    };
    let wt = std::path::PathBuf::from("/tmp").join(format!("vllm-{short}"));
    // Reuse a worktree that's already at this ref; wipe anything else
    // (a partial checkout from a crash) and re-add. Tags peel to a commit
    // sha for the comparison; commits compare by prefix.
    let head_target = match build {
        BuildRef::Dev { .. } => unreachable!("resolved to a commit above"),
        BuildRef::Commit(hex) => hex.clone(),
        BuildRef::Tag(t) => git(Some(&cache), &["rev-parse", &format!("refs/tags/{t}^{{commit}}")])?,
    };
    let at_ref = git(Some(&wt), &["rev-parse", "HEAD"])
        .map(|h| h == head_target || h.starts_with(&head_target))
        .unwrap_or(false);
    if !at_ref {
        let _ = std::fs::remove_dir_all(&wt);
        git(Some(&cache), &["worktree", "prune"])?;
        git(Some(&cache), &["worktree", "add", "--no-checkout",
            &wt.to_string_lossy(), &checkout_ref])?;
        // Sparse: only the trees resolution scans. If the local git is too
        // old for `sparse-checkout`, fall back to a full checkout (more
        // blobs, same correctness).
        if git(Some(&wt), &["sparse-checkout", "set", "vllm", "csrc"]).is_ok() {
            git(Some(&wt), &["reset", "--hard"])?;
        } else {
            git(Some(&wt), &["checkout", "--detach", &checkout_ref])?;
        }
    }
    // --abbrev=40 so a sha fallback is the full commit: the drift check
    // compares the version string's 12-18 char hex against this, and a
    // 7-char abbreviation can't contain it.
    let desc = git(Some(&wt), &["describe", "--tags", "--always", "--dirty", "--abbrev=40"])
        .unwrap_or_else(|_| checkout_ref.clone());
    Ok((wt, desc))
}

/// Point the cache's `origin` at `remote`. Must be the remote the ref came
/// from: it's the promisor lazy blob fetches go to.
fn set_origin(cache: &Path, remote: &str) -> Result<(), String> {
    match git(Some(cache), &["remote", "get-url", "origin"]) {
        Ok(_) => git(Some(cache), &["remote", "set-url", "origin", remote]).map(|_| ()),
        Err(_) => git(Some(cache), &["remote", "add", "origin", remote]).map(|_| ()),
    }
}

/// Fetch a tag shallow + blobless, trying each remote in order (the fork
/// may lag upstream's tags, or carry some upstream doesn't).
fn fetch_tag(cache: &Path, tag: &str) -> Result<(), String> {
    let spec = format!("refs/tags/{tag}:refs/tags/{tag}");
    let mut last = String::new();
    for remote in BUILD_REMOTES {
        set_origin(cache, remote)?;
        match git(Some(cache), &["fetch", "--depth", "1", "--filter=blob:none", "origin", &spec]) {
            Ok(_) => return Ok(()),
            Err(e) => last = e,
        }
    }
    Err(format!("tag {tag} not found in any remote ({last})"))
}

/// The machine's default vLLM checkout, if one exists: `~/vllm` that looks
/// like the repo (has the `vllm/` package dir and a `.git`). Cheap — an
/// index now holds nothing but a root until the first jump, so there is no
/// startup cost to configure the Source features on machines with a
/// checkout at the conventional place.
#[cfg(not(target_arch = "wasm32"))]
pub fn default_checkout() -> Option<(String, Box<SourceIndex>)> {
    let Ok(home) = std::env::var("HOME") else { return None };
    let def = Path::new(&home).join("vllm");
    if !def.join("vllm").is_dir() || !def.join(".git").exists() {
        return None;
    }
    let disp = def.to_string_lossy().into_owned();
    Some((disp, Box::new(SourceIndex::new(def))))
}
