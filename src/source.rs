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
        // Cute-dsl JIT kernels (cutlass Python DSL) reach the trace as
        // `kernel_cutlass_kernel_<flattened module path><Symbol>_object_at_...`
        // — the whole name is one flattened identifier, so the plain
        // base-name grep below can't see them. Resolve by path prefix.
        if let Some(rest) = kernel.strip_prefix("kernel_cutlass_kernel_") {
            let rest = rest.split("_object_at_").next().unwrap_or(rest);
            if !rest.is_empty() {
                if let Some(v) = self.memo.get(kernel) { return v.clone(); }
                let hits = self.scan_cute(rest);
                self.memo.insert(kernel.to_string(), hits.clone());
                return hits;
            }
        }
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
    /// tilelang kernels ARE python defs.
    ///
    /// A C-family match can be a call site (`sm100_mqa_logits(config, ...)`)
    /// or the definition (`void sm100_mqa_logits(const uint32_t ...)`), and
    /// the picker must offer the definition first: the whole point of the
    /// jump is the kernel's body, and the files are collected in parallel so
    /// without a rank the order is arbitrary. Definitions (a match at column
    /// 0 or after a declaration keyword) sort ahead of call sites, and CUDA
    /// sources (.cu/.cuh) ahead of headers, where the host-side launcher of
    /// a JIT kernel usually lives.
    ///
    /// Capped at 16 so a generic name can't produce an unbounded list.
    #[cfg(not(target_arch = "wasm32"))]
    fn scan(&self, name: &str) -> Vec<SourceHit> {
        use rayon::prelude::*;
        let t0 = std::time::Instant::now();
        let py_needle: Vec<u8> = format!("def {name}(").into_bytes();
        let c_needle: Vec<u8> = format!("{name}(").into_bytes();
        let mut files = Vec::new();
        self.collect_files(&self.root, &mut files);
        // (rank, line) before the path is known, (rank, SourceHit) after.
        let (py, c): (Vec<SourceHit>, Vec<(u8, SourceHit)>) = files
            .par_iter()
            .filter_map(|path| {
                let Ok(text) = std::fs::read(path) else { return None };
                let py = Lang::of_path(path) == Some(Lang::Py);
                let needle = if py { &py_needle } else { &c_needle };
                let cuda = path.extension().and_then(|e| e.to_str())
                    .is_some_and(|e| e == "cu" || e == "cuh");
                let mut out = Vec::new();
                // Line-by-line: a definition never spans lines.
                for (i, line) in text.split(|&b| b == b'\n').enumerate() {
                    if find_def(line, needle) {
                        let rank = if py { 0 }
                            else { (!is_c_decl(line) as u8) * 2 + (!cuda as u8) };
                        out.push((rank, (i + 1) as u32));
                    }
                }
                if out.is_empty() { None } else { Some((path, py, out)) }
            })
            .map(|(path, py, lines)| {
                let hits: Vec<(u8, SourceHit)> = lines.into_iter()
                    .map(|(r, l)| (r, self.hit(path.clone(), l))).collect();
                if py { (hits.into_iter().map(|(_, h)| h).collect(), Vec::new()) }
                else { (Vec::new(), hits) }
            })
            .reduce(|| (Vec::new(), Vec::new()), |(mut pa, mut ca), (pb, cb)| {
                pa.extend(pb); ca.extend(cb); (pa, ca)
            });
        let mut c = c;
        c.sort_by(|(ra, a), (rb, b)| {
            ra.cmp(rb).then_with(|| a.rel.cmp(&b.rel)).then(a.line.cmp(&b.line))
        });
        let mut hits = py;
        hits.extend(c.into_iter().map(|(_, h)| h));
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

    /// Cute-dsl resolution for `kernel_cutlass_kernel_<flat><Symbol>`:
    /// `<flat>` is the defining module's in-repo path with the extension
    /// and separators dropped (`flashinfer/gemm/kernels/x.py` ->
    /// `flashinfergemmkernelsx`), `<Symbol>` the kernel entry name.
    /// Match files by longest flattened-path prefix, then locate the
    /// symbol inside. No file is read until the file match is decided,
    /// so this costs one directory walk plus a couple of reads.
    /// Dep files match on their in-repo path: the on-disk
    /// `deps/flashinfer/flashinfer/gemm/...` matches `flashinfergemm...`.
    #[cfg(not(target_arch = "wasm32"))]
    fn scan_cute(&self, mangled: &str) -> Vec<SourceHit> {
        let mut files = Vec::new();
        self.collect_files(&self.root, &mut files);
        let mut best: Vec<(usize, PathBuf)> = Vec::new(); // (flat len, path)
        for path in files {
            let Ok(rel) = path.strip_prefix(&self.root) else { continue };
            // The flattened name carries the module path without the
            // extension: strip it before dropping the separators.
            let rel = rel.with_extension("");
            let rel = rel.to_string_lossy();
            // Dep checkouts sit under deps/<name>/; the mangled name
            // carries the repo-internal path, so drop that prefix.
            let in_repo = match rel.split_once('/') {
                Some(("deps", rest)) if rest.contains('/') => {
                    &rest[rest.find('/').unwrap() + 1..]
                }
                _ => &rel,
            };
            let flat = in_repo.replace('/', "");
            if !flat.is_empty() && mangled.starts_with(&flat) {
                let len = flat.len();
                match best.first() {
                    Some((l, _)) if *l > len => {}
                    Some((l, _)) if *l == len => best.push((len, path)),
                    _ => best = vec![(len, path)],
                }
            }
        }
        let sym = best.first().map(|(l, _)| &mangled[*l..]).unwrap_or("");
        best.iter()
            .filter_map(|(_, path)| {
                let Ok(text) = std::fs::read(path) else { return None };
                let needle: Vec<u8> = format!("{sym}(").into_bytes();
                let mut line = 0;
                for (i, l) in text.split(|&b| b == b'\n').enumerate() {
                    if find_def(l, &needle) { line = (i + 1) as u32; break; }
                }
                if line == 0 {
                    // Not a def (a class, or the entry is generated):
                    // first textual occurrence, else the file itself.
                    let hay = String::from_utf8_lossy(&text);
                    line = hay.lines().position(|l| l.contains(sym))
                        .map_or(1, |i| (i + 1) as u32);
                }
                Some(self.hit(path.clone(), line))
            })
            .collect()
    }

    /// wasm stub, same deal as scan's.
    #[cfg(target_arch = "wasm32")]
    fn scan_cute(&self, _mangled: &str) -> Vec<SourceHit> { Vec::new() }

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

/// Does this C-family line look like a declaration of the matched symbol
/// rather than a use of it? Definitions sit at column 0 (the repo style
/// outside nested scopes) or start with a declaration keyword; call sites
/// are indented and start with the call itself (`sm100_mqa_logits(config`,
/// `m.foo(`, ...). Only used to order the picker, never to drop a hit —
/// the heuristic can be wrong, but the definition sorting first is what
/// makes the first row the kernel's body.
fn is_c_decl(line: &[u8]) -> bool {
    let start = match line.iter().position(|c| !c.is_ascii_whitespace()) {
        Some(i) => i,
        None => return false,
    };
    if start == 0 { return true; }
    let rest = &line[start..];
    const KEYWORDS: [&[u8]; 14] = [
        b"void", b"static", b"inline", b"extern", b"auto", b"const",
        b"template", b"struct", b"class", b"__global__", b"__device__",
        b"__host__", b"__forceinline__", b"CUTLASS_DEVICE",
    ];
    KEYWORDS.iter().any(|kw| {
        rest.starts_with(kw)
            && rest.get(kw.len()).is_some_and(|c| !c.is_ascii_alphanumeric() && *c != b'_')
    })
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

/// Kernel families the checkout *launches* but never contains: the trace
/// symbol is the name of a prebuilt cubin registered by a dependency, so no
/// amount of scanning finds it. `Some(explanation to show)` when `name`
/// belongs to one; `None` otherwise.
///
/// The motivating case is the TRT-LLM trtllm-gen FMHA cubins
/// (`fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDense...`): flashinfer ships
/// the cubin (flashinfer-cubin) and the launcher that selects it
/// (`csrc/trtllm_fmha_kernel_launcher.cu`), while the kernel body is generated
/// upstream in NVIDIA TensorRT-LLM. Grepping the checkout can only ever miss,
/// so the miss should say why instead of looking like a resolver bug.
pub fn external_kernel_family(name: &str) -> Option<&'static str> {
    let n = name.trim();
    if n.starts_with("fmhaSm") && n.contains("Kernel_") {
        return Some(
            "a prebuilt trtllm-gen FMHA cubin - flashinfer ships the cubin \
             and its launcher (csrc/trtllm_fmha_kernel_launcher.cu), the kernel \
             source is in TensorRT-LLM");
    }
    None
}

/// CUPTI reports a kernel from a module it cannot symbolicate as just
/// `Kernel` — that is what CuTe DSL's JIT-loaded cubins look like in a
/// vLLM trace. Scanning that name is worse than useless (it word-matches
/// unrelated `Kernel(` symbols), so such events are routed to launch-site
/// resolution through the event's `correlation`.
pub fn is_generic_kernel_name(name: &str) -> bool {
    matches!(name.trim(), "Kernel" | "kernel")
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
    // A GUI-launched app doesn't inherit the shell's PATH, so a bare
    // command fails even when the CLI is installed; try the usual spots.
    match editor {
        "code" => {
            cands.push("/usr/local/bin/code".into());
            cands.push("/opt/homebrew/bin/code".into());
            if let Ok(home) = std::env::var("HOME") {
                cands.push(format!("{home}/.vscode/bin/code"));
            }
            cands.push("/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code".into());
        }
        "gram" => {
            cands.push("/opt/homebrew/bin/gram".into());
            cands.push("/usr/local/bin/gram".into());
            cands.push("/Applications/Gram.app/Contents/MacOS/cli".into());
        }
        "clion" => {
            cands.push("/usr/local/bin/clion".into());
            cands.push("/opt/homebrew/bin/clion".into());
            if let Ok(home) = std::env::var("HOME") {
                cands.push(format!(
                    "{home}/Library/Application Support/JetBrains/Toolbox/scripts/clion"));
            }
            cands.push("/Applications/CLion.app/Contents/MacOS/clion".into());
        }
        _ => {}
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
/// command ("gram", "code", "cursor", ...); Gram and Zed take
/// `file:line` directly, CLion takes `--line <n> <path>`, and the rest
/// take `-g file:line`. VS Code is special: `-g` positions the cursor only
/// on a fresh open — when the file is already open in a window it just
/// focuses the tab and leaves the cursor where it was — so `code` jumps via
/// the `vscode://file/<path>:<line>` protocol URL instead, whose handler
/// applies the selection to an open editor too.
#[cfg(not(target_arch = "wasm32"))]
pub fn open_in_editor(editor: &str, hit: &SourceHit) -> Result<(), String> {
    let loc = format!("{}:{}", hit.path.display(), hit.line);
    for exe in editor_candidates(editor) {
        let mut c = std::process::Command::new(&exe);
        if editor == "zed" || editor == "gram" {
            c.arg(&loc);
        } else if editor == "code" {
            c.args(["--open-url",
                &format!("vscode://file/{}:{}", encode_uri_path(&hit.path.display().to_string()), hit.line)]);
        } else if editor == "clion" {
            // JetBrains launchers take the line as a separate flag, then the path.
            c.arg("--line").arg(hit.line.to_string()).arg(&hit.path);
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

/// Resolve a commit prefix against one repo via the GitHub API: full sha
/// on success. Also doubles as the existence check — a nightly pipeline
/// stamp (`+llmd.cfee12e...`) 404s in every repo, and that's how tv tells
/// "real commit" from "stamp that matches no repo".
fn api_resolve_sha(repo_slug: &str, hex: &str) -> Result<String, String> {
    let url = format!("https://api.github.com/repos/{repo_slug}/commits/{hex}");
    let out = std::process::Command::new("curl")
        .args(["-s", "-m", "15", &url])
        .output()
        .map_err(|e| format!("couldn't run curl: {e}"))?;
    let body = String::from_utf8_lossy(&out.stdout).into_owned();
    if let Some(full) = extract_json_string(&body, "sha") {
        if full.len() == 40 && full.starts_with(hex) {
            return Ok(full);
        }
        return Err(format!("{hex} not found in {repo_slug}"));
    }
    if body.contains("No commit found") {
        return Err(format!("{hex} is not a commit in {repo_slug}"));
    }
    if body.contains("rate limit") {
        return Err("github api rate limited - try again in a minute".into());
    }
    Err(format!("couldn't resolve {hex} in {repo_slug} via the github api (network?)"))
}

/// Resolve an abbreviated commit sha to its full form plus the remote that
/// has it. GitHub's fetch protocol only accepts full shas, but the REST API
/// resolves prefixes.
fn resolve_full_sha(hex: &str) -> Result<(String, &'static str), String> {
    let mut not_found = false;
    for remote in BUILD_REMOTES {
        let repo = remote.trim_start_matches("https://github.com/");
        match api_resolve_sha(repo, hex) {
            Ok(full) => return Ok((full, remote)),
            Err(e) if e.contains("is not a commit") => { not_found = true; }
            Err(_) => {} // Network error / garbage: try the next remote.
        }
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
/// a sparse worktree at `/tmp/vllm-<short>` holding only `vllm/`, `csrc/`
/// and `cmake/` — the trees source resolution scans, plus where the
/// dependency pins live — so the lazy blob fetch during checkout downloads
/// just those, not docs/, tests/, or examples/. The CMake-pinned deps
/// (deep_gemm and friends) are fetched into `<worktree>/deps/` alongside.
/// Returns the worktree path plus its `git describe` (the full
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
        // Sparse: only the trees resolution scans (plus cmake/, where the
        // dependency pins live). If the local git is too old for
        // `sparse-checkout`, fall back to a full checkout (more blobs,
        // same correctness).
        if git(Some(&wt), &["sparse-checkout", "set", "vllm", "csrc", "cmake"]).is_ok() {
            git(Some(&wt), &["reset", "--hard"])?;
        } else {
            git(Some(&wt), &["checkout", "--detach", &checkout_ref])?;
        }
    }
    // The CMake-pinned deps (deep_gemm, DeepSelect, FlashMLA,
    // flash-attention, ...) land in <wt>/deps/, where the scanner walks
    // them like any other source: kernels outside the vllm tree resolve
    // against the exact pinned source the binary was built with.
    // Best-effort — a failed dep never fails the build.
    materialize_deps(&wt);
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

// ================================================== cmake-pinned dependencies

/// A dependency vLLM pins via CMake FetchContent, parsed out of
/// `cmake/external_projects/*.cmake`: deep_gemm, DeepSelect, FlashMLA,
/// flash-attention, ... Their kernels show up in traces under their own
/// namespaces (`deep_gemm::sm100_fp8_fp4_mega_moe_impl<...>`) and resolve
/// nowhere in the vllm tree, because vllm pins them at build time rather
/// than vendoring them.
#[derive(Debug)]
pub struct DepPin {
    /// Directory name under `<worktree>/deps/` (the cmake file's stem,
    /// or the short name from the pip table).
    pub name: String,
    /// Clone URL, always github so far.
    pub url: String,
    /// Pinned ref: a full sha in every pin seen, but tags work too.
    pub tag: String,
}

/// The value token after a cmake keyword: `"quoted"` (possibly
/// `"${VAR}"`) or bare up to whitespace/`)`.
fn cmake_token(s: &str) -> Option<&str> {
    let s = s.trim_start();
    if let Some(rest) = s.strip_prefix('"') {
        let end = rest.find('"')?;
        Some(&rest[..end])
    } else {
        let end = s.find(|c: char| c.is_whitespace() || c == ')').unwrap_or(s.len());
        Some(&s[..end])
    }
}

/// Parse the FetchContent pins out of `cmake/external_projects/*.cmake`.
/// Two conventions in the wild, sometimes both in one file: direct
/// `GIT_REPOSITORY <url>` / `GIT_TAG <ref>` arguments, and
/// `set(_X_UPSTREAM_REPO "...")` variables referenced from them as
/// `"${_X_UPSTREAM_REPO}"`. Files with no git pin (tarball downloads,
/// e.g. triton_kernels) contribute nothing. Repo is paired with the first
/// GIT_TAG following it.
pub(crate) fn parse_dep_pins(dir: &Path) -> Vec<DepPin> {
    let mut out: Vec<DepPin> = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    let mut files: Vec<PathBuf> = rd.flatten().map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "cmake"))
        .collect();
    files.sort();
    for f in files {
        let Ok(text) = std::fs::read_to_string(&f) else { continue };
        // set(VAR "value") lines, for ${VAR} references.
        let mut vars = HashMap::new();
        for line in text.lines() {
            let t = line.trim();
            let Some(rest) = t.strip_prefix("set(") else { continue };
            let Some((k, v)) = rest.split_once(|c: char| c.is_whitespace() || c == '(') else { continue };
            if let Some(v) = cmake_token(v) {
                vars.insert(k.trim().to_string(), v.to_string());
            }
        }
        let resolve = |tok: &str| -> String {
            tok.strip_prefix("${").and_then(|r| r.strip_suffix('}'))
                .and_then(|name| vars.get(name).cloned())
                .unwrap_or_else(|| tok.to_string())
        };
        // (byte position, value) so each repo pairs with the next tag.
        let mut repos: Vec<(usize, String)> = Vec::new();
        let mut tags: Vec<(usize, String)> = Vec::new();
        for (pos, _) in text.match_indices("GIT_REPOSITORY") {
            if let Some(v) = cmake_token(&text[pos + "GIT_REPOSITORY".len()..]) {
                repos.push((pos, resolve(v)));
            }
        }
        for (pos, _) in text.match_indices("GIT_TAG") {
            if let Some(v) = cmake_token(&text[pos + "GIT_TAG".len()..]) {
                tags.push((pos, resolve(v)));
            }
        }
        let Some(name) = f.file_stem().map(|s| s.to_string_lossy().into_owned()) else { continue };
        for (rpos, url) in repos {
            if !url.starts_with("https://github.com/") { continue; }
            let Some(tag) = tags.iter().find(|(tpos, _)| *tpos > rpos).map(|(_, t)| t.clone()) else { continue };
            if tag.is_empty() || out.iter().any(|p| p.url == url && p.tag == tag) { continue; }
            out.push(DepPin { name: name.clone(), url, tag });
        }
    }
    out
}

/// pip packages we know a github repo for: package name -> (dep dir
/// name, clone URL). Their kernels appear in traces under the repo's
/// own path namespace, and vllm pins the exact version in
/// requirements/cuda.txt, so the checkout is recoverable like the CMake
/// pins. Add rows as traces show kernels from other pinned pips.
fn pip_github_repo(pkg: &str) -> Option<(&'static str, &'static str)> {
    match pkg {
        // The flashinfer cute-dsl kernels (`kernel_cutlass_kernel_
        // flashinfergemmkernels...`) are the biggest dep kernel family
        // in EP/MoE traces.
        "flashinfer-python" =>
            Some(("flashinfer", "https://github.com/flashinfer-ai/flashinfer")),
        _ => None,
    }
}

/// Exact `pkg==version` pins from a requirements file, for packages in
/// the table above. Version ranges (`>=`), URLs (` @ `) and environment
/// markers pin nothing exact, so those lines contribute nothing.
pub(crate) fn parse_pip_pins(file: &Path) -> Vec<DepPin> {
    let Ok(text) = std::fs::read_to_string(file) else { return Vec::new() };
    let mut out = Vec::new();
    for line in text.lines() {
        // Strip comment, then environment marker; both are `#`/`;`-out.
        let line = line.split('#').next().unwrap_or(line);
        let line = line.split(';').next().unwrap_or(line).trim();
        let Some((pkg, ver)) = line.split_once("==") else { continue };
        let (pkg, ver) = (pkg.trim(), ver.trim());
        if ver.is_empty() { continue; }
        if let Some((name, url)) = pip_github_repo(pkg) {
            out.push(DepPin {
                name: name.to_string(), url: url.to_string(), tag: format!("v{ver}"),
            });
        }
    }
    out
}

/// Fetch the CMake-pinned dependencies into `<worktree>/deps/<name>`,
/// where the source scanner walks them like any other tree — kernels
/// that live outside vllm (`deep_gemm::...`, flash-attention, ...)
/// resolve to the exact pinned source the binary was built against.
/// Each dep is its own shallow blobless repo at the pin; dep failures
/// are logged and skipped, since a missing dep costs a few unresolvable
/// kernels, not the whole build. Parallel: they're independent network
/// fetches. Returns the pin count (fetched or already present).
fn materialize_deps(wt: &Path) -> usize {
    // The pins live under cmake/ and requirements/, which the sparse
    // checkout must include (a no-op re-set when it already does).
    let _ = git(Some(wt), &["sparse-checkout", "set", "vllm", "csrc", "cmake", "requirements"]);
    let mut pins = parse_dep_pins(&wt.join("cmake/external_projects"));
    pins.extend(parse_pip_pins(&wt.join("requirements/cuda.txt")));
    let deps = wt.join("deps");
    if std::fs::create_dir_all(&deps).is_err() { return 0; }
    std::thread::scope(|s| {
        for pin in &pins {
            s.spawn(|| {
                let dir = deps.join(&pin.name);
                // Already at the pin? Shas compare directly (prefix so
                // abbreviated pins reuse too); tags through their local
                // ref, which fetch_dep writes for exactly this check.
                let at_pin = git(Some(&dir), &["rev-parse", "HEAD"])
                    .is_ok_and(|h| {
                        h == pin.tag || h.starts_with(&pin.tag)
                            || git(Some(&dir), &["rev-parse",
                                    &format!("refs/tags/{}^{{}}", pin.tag)])
                                .is_ok_and(|t| t == h)
                    });
                if at_pin { return; }
                let _ = std::fs::remove_dir_all(&dir);
                if let Err(e) = fetch_dep(&dir, pin) {
                    crate::tlog!("dep {}: {e}", pin.name);
                }
            });
        }
    });
    pins.len()
}

/// One dep: fresh shallow repo at the pin. Sha pins (every vllm CMake
/// pin so far) fetch directly, abbreviated shas go through the API, tag
/// names (pip pins like flashinfer's `v0.7.2.post1`) through the tag
/// refspec — which also leaves the local ref behind for the at-pin
/// reuse check in materialize_deps.
fn fetch_dep(dir: &Path, pin: &DepPin) -> Result<(), String> {
    let dir_s = dir.to_string_lossy().into_owned();
    git(None, &["init", "-q", &dir_s])?;
    git(Some(dir), &["remote", "add", "origin", &pin.url])?;
    let fetch = |r: &str| git(Some(dir), &["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", r]);
    let hexish = pin.tag.len() >= 7 && pin.tag.chars().all(|c| c.is_ascii_hexdigit());
    if hexish {
        if fetch(&pin.tag).is_err() {
            let slug = pin.url.trim_start_matches("https://github.com/").trim_end_matches(".git");
            fetch(&api_resolve_sha(slug, &pin.tag)?)?;
        }
    } else {
        let spec = format!("refs/tags/{0}:refs/tags/{0}", pin.tag);
        git(Some(dir), &["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", &spec])?;
    }
    // checkout triggers the promisor's lazy blob fetch for the files.
    git(Some(dir), &["checkout", "-q", "--detach", "FETCH_HEAD"]).map(|_| ())
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
