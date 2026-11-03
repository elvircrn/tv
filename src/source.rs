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

/// The build commit a trace was built from, per its `vllm_version`. Two
/// conventions in the wild: the llmd line appends `.<hex>` after a local
/// version tag (`0.33.0rc1.dev573+llmd.cfee12e985dc6f4f74fd`), and
/// setuptools-scm's is `+g<hex>` (`0.26.1rc1.dev528+g1a2b3c4d5e6f`).
/// None when the version string carries no commit (or a non-vllm trace).
pub fn trace_build_commit(vllm_version: &str) -> Option<&str> {
    let hexish = |s: &str| s.len() >= 7 && s.chars().all(|c| c.is_ascii_hexdigit());
    // llmd style: hex is the last '.'-separated segment.
    if let Some(seg) = vllm_version.rsplit('.').next() {
        if hexish(seg) { return Some(seg); }
    }
    // setuptools-scm style: hex follows the last '+', after a 'g'.
    let tail = vllm_version.rsplit('+').next()?;
    let hex = tail.strip_prefix('g').unwrap_or(tail);
    hexish(hex).then_some(hex)
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

/// Open `path:line` in the user's editor. `editor` is the editor's CLI
/// command ("code", "cursor", ...); VS Code and Cursor both take `-g
/// file:line`, Zed takes `file:line` directly.
#[cfg(not(target_arch = "wasm32"))]
pub fn open_in_editor(editor: &str, hit: &SourceHit) -> Result<(), String> {
    let loc = format!("{}:{}", hit.path.display(), hit.line);
    for exe in editor_candidates(editor) {
        let mut c = std::process::Command::new(&exe);
        if editor == "zed" { c.arg(&loc); } else { c.arg("-g").arg(&loc); }
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
