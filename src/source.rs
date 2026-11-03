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

// ================================================================ the index

/// A walked vLLM checkout: python def/class index for kernel-name lookups,
/// the C-family file list for content searches, and the checkout's git
/// description for the version-mismatch banner (a checkout at a different
/// commit than the traced build means line numbers can drift).
pub struct SourceIndex {
    root: PathBuf,
    /// `def NAME(` / `class NAME` -> (file, line). Triton/Tilelang kernels
    /// resolve through this exactly.
    py_defs: HashMap<String, Vec<(PathBuf, u32)>>,
    /// C-family files under the root, searched by content on demand.
    c_files: Vec<PathBuf>,
    /// Memoized content-search results (kernel base name -> hits).
    c_memo: HashMap<String, Vec<SourceHit>>,
    pub git_desc: Option<String>,
    pub n_files: usize,
}

/// Directory names never worth walking into.
fn skip_dir(name: &str) -> bool {
    matches!(name, ".git" | "build" | "node_modules" | ".venv" | "venv"
        | "__pycache__" | ".mypy_cache" | ".pytest_cache" | "target" | "cmake-build-debug")
}

impl SourceIndex {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn build(root: &Path) -> SourceIndex {
        let t0 = std::time::Instant::now();
        let mut idx = SourceIndex {
            root: root.to_path_buf(),
            py_defs: HashMap::new(),
            c_files: Vec::new(),
            c_memo: HashMap::new(),
            git_desc: None,
            n_files: 0,
        };
        idx.walk(root);
        idx.git_desc = idx.git_describe();
        crate::tlog!("source index: {} files ({} py defs, {} C files) in {:.2}s",
            idx.n_files, idx.py_defs.len(), idx.c_files.len(), t0.elapsed().as_secs_f64());
        idx
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn walk(&mut self, dir: &Path) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for entry in rd.flatten() {
            let Ok(ft) = entry.file_type() else { continue };
            let path = entry.path();
            if ft.is_dir() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if !skip_dir(&name) { self.walk(&path); }
            } else if ft.is_file() {
                self.n_files += 1;
                match Lang::of_path(&path) {
                    Some(Lang::Py) => self.index_py(&path),
                    Some(Lang::CFamily) => self.c_files.push(path),
                    None => {}
                }
            }
        }
    }

    /// Record every top-level `def NAME(` / `class NAME` at its line. A line
    /// scan, not a parse — good enough for name lookups.
    #[cfg(not(target_arch = "wasm32"))]
    fn index_py(&mut self, path: &Path) {
        let Ok(text) = std::fs::read_to_string(path) else { return };
        for (i, line) in text.lines().enumerate() {
            let t = line.trim_start();
            let rest = if let Some(r) = t.strip_prefix("def ") {
                r
            } else if let Some(r) = t.strip_prefix("class ") {
                r
            } else if let Some(r) = t.strip_prefix("async def ") {
                r
            } else {
                continue;
            };
            // Name runs to the first non-identifier char (a def with no
            // parens yet, or a class header).
            let name: String = rest
                .chars()
                .take_while(|&c| c.is_alphanumeric() || c == '_')
                .collect();
            if name.is_empty() { continue; }
            self.py_defs.entry(name).or_default().push((path.to_path_buf(), (i + 1) as u32));
        }
    }

    /// `git describe --tags --always --dirty` in the checkout, for the banner.
    #[cfg(not(target_arch = "wasm32"))]
    fn git_describe(&self) -> Option<String> {
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

    /// Kernel-name resolution: python defs first (triton/tilelang), then a
    /// content search of the C-family files. If the exact name misses, the
    /// trailing `_`-separated segment is dropped — but ONLY when it looks
    /// like triton's mangled dims suffix (`0d1d2`), never a real name part:
    /// dropping `kernel` from `vectorized_elementwise_kernel` once produced
    /// `vectorized`, which substring-matched a different function and sent
    /// the viewer to a confidently wrong line. Up to two drops.
    pub fn resolve_kernel(&mut self, kernel: &str) -> Vec<SourceHit> {
        let mut base = kernel_base_name(kernel);
        if base.is_empty() { return Vec::new(); }
        for attempt in 0..3 {
            let mut hits = self.py_lookup(&base);
            if hits.is_empty() { hits = self.c_lookup(&base); }
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

    fn py_lookup(&self, name: &str) -> Vec<SourceHit> {
        self.py_defs.get(name).map(|v| v.iter().map(|&(ref p, l)| self.hit(p.clone(), l)).collect())
            .unwrap_or_default()
    }

    /// Content search for `name(` in the C-family files, at a word boundary
    /// (the char before the match must not be an identifier char — otherwise
    /// `vectorized(` matches inside `scaled_fp8_conversion_vectorized(`).
    /// Memoized — a kernel name repeats on every one of its thousands of
    /// launches, and the first search is a full read of every .cu/.cc/.h
    /// in the repo.
    #[cfg(not(target_arch = "wasm32"))]
    fn c_lookup(&mut self, name: &str) -> Vec<SourceHit> {
        if let Some(v) = self.c_memo.get(name) { return v.clone(); }
        let t0 = std::time::Instant::now();
        let needle: Vec<u8> = format!("{name}(").into_bytes();
        // (path, line) pairs collected first, mapped to hits after the loop:
        // the iteration borrows self.c_files, and hit() borrows all of self.
        let mut found: Vec<(PathBuf, u32)> = Vec::new();
        'files: for path in &self.c_files {
            let Ok(text) = std::fs::read(path) else { continue };
            // Line-by-line: a definition never spans lines.
            for (i, line) in text.split(|&b| b == b'\n').enumerate() {
                if find_def(line, &needle) {
                    found.push((path.clone(), (i + 1) as u32));
                    if found.len() >= 8 { break 'files; }
                }
            }
        }
        let hits: Vec<SourceHit> = found.into_iter().map(|(p, l)| self.hit(p, l)).collect();
        let elapsed = t0.elapsed().as_secs_f64();
        if elapsed > 0.25 {
            crate::tlog!("c_lookup({name}): {} hits in {:.2}s", hits.len(), elapsed);
        }
        self.c_memo.insert(name.to_string(), hits.clone());
        hits
    }

    /// wasm stub: no filesystem to search. (resolve_kernel is compiled on
    /// wasm because Pane state references its types, but the Source tab UI
    /// that drives it is native-only, so it's never reached.)
    #[cfg(target_arch = "wasm32")]
    fn c_lookup(&mut self, _name: &str) -> Vec<SourceHit> { Vec::new() }

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
/// like the repo (has the `vllm/` package dir and a `.git`). Returns the
/// display path plus an already-running background index build, so the
/// Source features work with zero setup on a machine that has a checkout
/// at the conventional place — the walk costs the main thread nothing.
#[cfg(not(target_arch = "wasm32"))]
pub fn default_checkout_start() -> (String, Option<std::thread::JoinHandle<Box<SourceIndex>>>) {
    let Ok(home) = std::env::var("HOME") else { return (String::new(), None) };
    let def = Path::new(&home).join("vllm");
    if !def.join("vllm").is_dir() || !def.join(".git").exists() {
        return (String::new(), None);
    }
    let disp = def.to_string_lossy().into_owned();
    let handle = std::thread::spawn(move || Box::new(SourceIndex::build(&def)));
    (disp, Some(handle))
}
