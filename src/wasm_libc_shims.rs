//! Minimal libc shims for imgui-sys's C++ sources, needed only when compiling
//! for `wasm32-unknown-unknown`.
//!
//! `imgui-sys`'s `build.rs` compiles Dear ImGui's `.cpp` files with `cc`
//! regardless of target (there's no wasm-specific branch), so the resulting
//! object code references a handful of libc functions. Native builds get
//! those for free by linking the system libc; wasm32-unknown-unknown has no
//! libc at all, so `wasm-ld` was leaving them as unresolved imports from a
//! module literally named `"env"` — which no JS glue (wasm-bindgen's
//! `--target=web` output included) knows how to satisfy, so the module
//! failed to instantiate in the browser at all ("Failed to resolve module
//! specifier \"env\"").
//!
//! Two more of the original 16 unresolved symbols are eliminated by other
//! means, not shimmed here — see the comment block in scripts/wasm-env.sh:
//!   - `vsnprintf` — imgui is compiled with `-DIMGUI_USE_STB_SPRINTF`, which
//!     routes all string formatting through the bundled, libc-free
//!     `third-party/stb_sprintf.h` instead.
//!   - `__assert_fail` — imgui is compiled with `-DNDEBUG`, turning every
//!     `assert()` (which is what `IM_ASSERT` and stb_rectpack/stb_truetype's
//!     internal asserts expand to) into a no-op, per the standard C
//!     convention.
//!
//! The remaining functions split into two groups:
//!   - Pure computation, reimplemented for real below: `strncpy`, `memchr`,
//!     `strcmp`, `strstr`, `qsort`, `sscanf`. `sscanf` in particular is
//!     reachable at runtime (ini-settings parsing, and would also back
//!     numeric/hex text-input widgets if this app used `InputScalar`-family
//!     widgets — it currently only uses `InputText`, but the ini-parsing
//!     path in imgui.cpp/imgui_tables.cpp always runs at startup).
//!   - File I/O and debug-only logging, stubbed as always-fail/no-op:
//!     `fopen`, `fclose`, `ftell`, `fseek`, `fread`, `fwrite`, `fflush`,
//!     `printf`. This app never sets `io.ini_filename` to `None` (imgui-rs's
//!     `Context` only tracks that as a Rust-side cache; the underlying
//!     C `ImGuiIO::IniFilename` still defaults to `"imgui.ini"`), so these
//!     *are* reached at startup/shutdown — but there's no filesystem in the
//!     browser, and `fopen` returning NULL is an ordinary, already-handled
//!     Dear ImGui code path (equivalent to "no ini file yet"). `printf` and
//!     `qsort` are additionally dead in practice: the only call sites are
//!     `IMGUI_DEBUG_PRINTF` (gated behind a debug-log flag this app never
//!     turns on) and `imgui_demo.cpp`'s sorting demo (this app never shows
//!     the demo window) respectively — stubbed/reimplemented anyway for
//!     defensive correctness, in case that ever changes.
//!
//! Every signature below was checked against the actual wasm import types
//! declared by the compiled module (`wasm-objdump -x target/wasm32-unknown-unknown/debug/tv.wasm
//! | grep '<- env\.'`, cross-referenced against the `Type[]` section) rather
//! than assumed from the C prototypes. That check is what revealed that
//! `sscanf`/`printf` show up as plain (non-variadic) 3-/2-argument wasm
//! functions: clang's wasm32 C ABI packs a variadic call site's arguments
//! into one caller-allocated buffer in linear memory and passes a single
//! pointer to it (there's no stable Rust syntax for *defining* a genuine
//! C-variadic function — `extern "C" fn f(x: i32, ...)` — as of this
//! toolchain; see rust-lang/rust#44930). Since every `sscanf` conversion in
//! C is required by the language to take a pointer argument, that packed
//! buffer is, for `sscanf` specifically, always a plain sequence of 4-byte
//! (wasm32 pointer-sized) entries — no need to worry about mixed
//! alignment/size the way a general `va_list` reader would.

#![allow(non_snake_case)]

use std::os::raw::{c_char, c_int, c_void};

// ---------------------------------------------------------------------
// Pure computation
// ---------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn memchr(s: *const c_void, c: c_int, n: usize) -> *mut c_void {
    let s = s as *const u8;
    let byte = c as u8;
    for i in 0..n {
        if *s.add(i) == byte {
            return s.add(i) as *mut c_void;
        }
    }
    std::ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn strncpy(dst: *mut c_char, src: *const c_char, n: usize) -> *mut c_char {
    let mut ended = false;
    for i in 0..n {
        if !ended {
            let c = *src.add(i);
            *dst.add(i) = c;
            if c == 0 {
                ended = true;
            }
        } else {
            *dst.add(i) = 0;
        }
    }
    dst
}

#[no_mangle]
pub unsafe extern "C" fn strncmp(a: *const c_char, b: *const c_char, n: usize) -> c_int {
    for i in 0..n {
        let ca = *a.add(i) as u8;
        let cb = *b.add(i) as u8;
        if ca != cb || ca == 0 {
            return ca as c_int - cb as c_int;
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn strcmp(a: *const c_char, b: *const c_char) -> c_int {
    let mut i: isize = 0;
    loop {
        let ca = *a.offset(i) as u8;
        let cb = *b.offset(i) as u8;
        if ca != cb || ca == 0 {
            return ca as c_int - cb as c_int;
        }
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn strstr(haystack: *const c_char, needle: *const c_char) -> *mut c_char {
    let mut nlen = 0usize;
    while *needle.add(nlen) != 0 {
        nlen += 1;
    }
    if nlen == 0 {
        return haystack as *mut c_char;
    }
    let mut i = 0usize;
    loop {
        let mut j = 0usize;
        loop {
            let hc = *haystack.add(i + j);
            if hc == 0 {
                return std::ptr::null_mut();
            }
            let nc = *needle.add(j);
            if hc != nc {
                break;
            }
            j += 1;
            if j == nlen {
                return haystack.add(i) as *mut c_char;
            }
        }
        i += 1;
    }
}

// `malloc`/`free` back Dear ImGui's default allocator (`MallocWrapper`/
// `FreeWrapper` in imgui.cpp, installed unless the app calls
// `ImGui::SetAllocatorFunctions` — this app doesn't). These two, plus
// `strncmp` above (used by every `InputText` widget's buffer-recycling
// check in imgui_widgets.cpp), only showed up as unresolved *after* fixing
// the original 16 documented at the top of this file — `vsnprintf`'s
// `stbsp_vsnprintf` replacement apparently pulled the linker's attention to
// a different part of the call graph than before. Confirmed real (not an
// artifact) by grepping imgui's sources directly for `malloc(`/`free(`/
// `strncmp(` call sites.
//
// A malloc/free pair needs *some* way to recover the allocation size at
// `free` time (Rust's `dealloc` requires the original `Layout`), so a small
// fixed-size header carrying the total allocation size is stashed just
// before the pointer handed back to the caller.
const MALLOC_ALIGN: usize = 16;
const MALLOC_HEADER: usize = MALLOC_ALIGN;

#[no_mangle]
pub unsafe extern "C" fn malloc(size: usize) -> *mut c_void {
    if size == 0 {
        return std::ptr::null_mut();
    }
    let Some(total) = size.checked_add(MALLOC_HEADER) else {
        return std::ptr::null_mut();
    };
    let Ok(layout) = std::alloc::Layout::from_size_align(total, MALLOC_ALIGN) else {
        return std::ptr::null_mut();
    };
    let raw = std::alloc::alloc(layout);
    if raw.is_null() {
        return std::ptr::null_mut();
    }
    (raw as *mut usize).write(total);
    raw.add(MALLOC_HEADER) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    let raw = (ptr as *mut u8).sub(MALLOC_HEADER);
    let total = (raw as *mut usize).read();
    let layout = std::alloc::Layout::from_size_align_unchecked(total, MALLOC_ALIGN);
    std::alloc::dealloc(raw, layout);
}

type CompareFn = unsafe extern "C" fn(*const c_void, *const c_void) -> c_int;

/// Insertion sort (O(n^2), but every real call site — `imgui_demo.cpp`'s
/// sorting-table demo — operates on at most a few dozen elements, and this
/// app never shows the demo window, so this is never actually reached in
/// practice; correctness, not speed, is what matters here).
#[no_mangle]
pub unsafe extern "C" fn qsort(
    base: *mut c_void,
    nmemb: usize,
    size: usize,
    compar: Option<CompareFn>,
) {
    let Some(compar) = compar else { return };
    if nmemb < 2 || size == 0 {
        return;
    }
    let base = base as *mut u8;
    let mut tmp = vec![0u8; size];
    for i in 1..nmemb {
        let mut j = i;
        while j > 0 {
            let a = base.add((j - 1) * size) as *const c_void;
            let b = base.add(j * size) as *const c_void;
            if compar(a, b) <= 0 {
                break;
            }
            std::ptr::copy_nonoverlapping(base.add(j * size), tmp.as_mut_ptr(), size);
            std::ptr::copy(base.add((j - 1) * size), base.add(j * size), size);
            std::ptr::copy_nonoverlapping(tmp.as_ptr(), base.add((j - 1) * size), size);
            j -= 1;
        }
    }
}

// ---------------------------------------------------------------------
// sscanf — minimal but real implementation.
//
// Covers every directive actually used anywhere in imgui.cpp /
// imgui_widgets.cpp / imgui_tables.cpp (checked exhaustively by grepping
// all `sscanf(` call sites): literal characters, whitespace, `%d`, `%i`,
// `%u`, `%x`/`%X`, `%f`, `%c`, `%n`, `%%`, field widths (e.g. `%02X`), the
// `l`/`ll`/`h`/`hh`/`j`/`z`/`t` length modifiers (needed for `%lld`/`%llu`,
// used by DataTypeApplyFromText for 64-bit scalar types), and `*`
// assignment-suppression. `%i`'s base auto-detection is simplified to just
// "0x"/"0X" prefix -> hex, else decimal (no octal-via-leading-zero support —
// no call site in imgui ever produces or expects octal here).
// ---------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Len {
    Hh,
    H,
    Default,
    L,
    Ll,
}

unsafe fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// Reads the next packed vararg pointer out of the caller-allocated argument
/// buffer (see the module doc comment: on wasm32, every `sscanf` vararg is a
/// plain 4-byte pointer, packed sequentially with no padding).
unsafe fn next_arg_ptr(args_buf: *const u8, arg_off: &mut usize) -> *mut u8 {
    let p = args_buf.add(*arg_off);
    let addr = u32::from_le_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]);
    *arg_off += 4;
    addr as *mut u8
}

unsafe fn store_int(dst: *mut u8, val: i64, len: Len) {
    match len {
        Len::Hh => std::ptr::write_unaligned(dst as *mut i8, val as i8),
        Len::H => std::ptr::write_unaligned(dst as *mut i16, val as i16),
        Len::Ll => std::ptr::write_unaligned(dst as *mut i64, val),
        Len::Default | Len::L => std::ptr::write_unaligned(dst as *mut i32, val as i32),
    }
}

unsafe fn store_float(dst: *mut u8, val: f64, len: Len) {
    match len {
        Len::L | Len::Ll => std::ptr::write_unaligned(dst as *mut f64, val),
        _ => std::ptr::write_unaligned(dst as *mut f32, val as f32),
    }
}

/// Parses an integer in the given base (0 meaning "auto-detect via an
/// optional 0x/0X prefix, else decimal") starting at `*s`, honoring `width`
/// as a maximum character count. Returns `None` (and rewinds `*s`) if no
/// digits were found.
unsafe fn scan_int(s: &mut *const u8, width: usize, base: u32) -> Option<i64> {
    while is_space(**s) {
        *s = s.add(1);
    }
    let start = *s;
    let mut consumed = 0usize;
    let mut neg = false;
    if consumed < width && (**s == b'+' || **s == b'-') {
        neg = **s == b'-';
        *s = s.add(1);
        consumed += 1;
    }
    let mut effective_base = base;
    if base == 0 || base == 16 {
        if consumed + 1 < width
            && **s == b'0'
            && (*s.add(1) == b'x' || *s.add(1) == b'X')
        {
            *s = s.add(2);
            consumed += 2;
            effective_base = 16;
        } else if base == 0 {
            effective_base = 10;
        }
    }
    let mut val: i64 = 0;
    let mut any_digit = false;
    while consumed < width {
        let c = **s;
        let d = match c {
            b'0'..=b'9' => (c - b'0') as u32,
            b'a'..=b'f' => (c - b'a' + 10) as u32,
            b'A'..=b'F' => (c - b'A' + 10) as u32,
            _ => break,
        };
        if d >= effective_base {
            break;
        }
        val = val * effective_base as i64 + d as i64;
        any_digit = true;
        *s = s.add(1);
        consumed += 1;
    }
    if !any_digit {
        *s = start;
        return None;
    }
    Some(if neg { -val } else { val })
}

/// Parses a (simple, non-hex, no inf/nan) floating point literal, honoring
/// `width`. Returns `None` (and rewinds `*s`) if nothing could be parsed.
unsafe fn scan_float(s: &mut *const u8, width: usize) -> Option<f64> {
    while is_space(**s) {
        *s = s.add(1);
    }
    let start = *s;
    let mut consumed = 0usize;
    let mut neg = false;
    if consumed < width && (**s == b'+' || **s == b'-') {
        neg = **s == b'-';
        *s = s.add(1);
        consumed += 1;
    }
    let mut int_part = 0f64;
    let mut any_digit = false;
    while consumed < width && (**s).is_ascii_digit() {
        int_part = int_part * 10.0 + (**s - b'0') as f64;
        *s = s.add(1);
        consumed += 1;
        any_digit = true;
    }
    let mut frac_part = 0f64;
    if consumed < width && **s == b'.' {
        let dot_s = *s;
        let dot_consumed = consumed;
        *s = s.add(1);
        consumed += 1;
        let mut scale = 0.1;
        let mut frac_digits = false;
        while consumed < width && (**s).is_ascii_digit() {
            frac_part += (**s - b'0') as f64 * scale;
            scale *= 0.1;
            *s = s.add(1);
            consumed += 1;
            frac_digits = true;
        }
        if !frac_digits && !any_digit {
            // Bare "." with no digits on either side: not a number at all.
            *s = dot_s;
            consumed = dot_consumed;
        } else {
            any_digit = any_digit || frac_digits;
        }
    }
    if !any_digit {
        *s = start;
        return None;
    }
    let mut val = int_part + frac_part;
    if consumed < width && (**s == b'e' || **s == b'E') {
        let mark_s = *s;
        let mark_consumed = consumed;
        *s = s.add(1);
        consumed += 1;
        let mut exp_neg = false;
        if consumed < width && (**s == b'+' || **s == b'-') {
            exp_neg = **s == b'-';
            *s = s.add(1);
            consumed += 1;
        }
        let mut exp_val: i32 = 0;
        let mut exp_any = false;
        while consumed < width && (**s).is_ascii_digit() {
            exp_val = exp_val * 10 + (**s - b'0') as i32;
            *s = s.add(1);
            consumed += 1;
            exp_any = true;
        }
        if exp_any {
            let exp = if exp_neg { -exp_val } else { exp_val };
            val *= 10f64.powi(exp);
        } else {
            // No exponent digits after 'e'/'E': not part of the number.
            *s = mark_s;
            let _ = mark_consumed;
        }
    }
    Some(if neg { -val } else { val })
}

#[no_mangle]
pub unsafe extern "C" fn sscanf(str_: *const c_char, fmt: *const c_char, args: *mut u8) -> c_int {
    let mut s = str_ as *const u8;
    let s_start = s;
    let mut f = fmt as *const u8;
    let args_buf = args as *const u8;
    let mut arg_off: usize = 0;
    let mut matched: c_int = 0;

    loop {
        let fc = *f;
        if fc == 0 {
            break;
        }
        if is_space(fc) {
            while is_space(*f) {
                f = f.add(1);
            }
            while is_space(*s) {
                s = s.add(1);
            }
            continue;
        }
        if fc != b'%' {
            if *s != fc {
                return matched;
            }
            s = s.add(1);
            f = f.add(1);
            continue;
        }

        // Conversion directive.
        f = f.add(1);
        if *f == b'%' {
            if *s != b'%' {
                return matched;
            }
            s = s.add(1);
            f = f.add(1);
            continue;
        }

        let suppress = if *f == b'*' {
            f = f.add(1);
            true
        } else {
            false
        };

        let mut width = usize::MAX;
        if (*f).is_ascii_digit() {
            width = 0;
            while (*f).is_ascii_digit() {
                width = width * 10 + (*f - b'0') as usize;
                f = f.add(1);
            }
        }

        let mut len = Len::Default;
        loop {
            match *f {
                b'h' => {
                    f = f.add(1);
                    if *f == b'h' {
                        f = f.add(1);
                        len = Len::Hh;
                    } else {
                        len = Len::H;
                    }
                }
                b'l' => {
                    f = f.add(1);
                    if *f == b'l' {
                        f = f.add(1);
                        len = Len::Ll;
                    } else {
                        len = Len::L;
                    }
                }
                b'L' => {
                    f = f.add(1);
                    len = Len::Ll;
                }
                b'j' | b'z' | b't' | b'q' => {
                    f = f.add(1);
                    len = Len::L;
                }
                _ => break,
            }
        }

        let conv = *f;
        if conv == 0 {
            break;
        }
        f = f.add(1);

        match conv {
            b'd' | b'i' | b'u' => {
                let base = if conv == b'i' { 0 } else { 10 };
                match scan_int(&mut s, width, base) {
                    Some(v) => {
                        if !suppress {
                            let dst = next_arg_ptr(args_buf, &mut arg_off);
                            store_int(dst, v, len);
                            matched += 1;
                        }
                    }
                    None => return matched,
                }
            }
            b'x' | b'X' => match scan_int(&mut s, width, 16) {
                Some(v) => {
                    if !suppress {
                        let dst = next_arg_ptr(args_buf, &mut arg_off);
                        store_int(dst, v, len);
                        matched += 1;
                    }
                }
                None => return matched,
            },
            b'f' | b'e' | b'g' | b'E' | b'G' => match scan_float(&mut s, width) {
                Some(v) => {
                    if !suppress {
                        let dst = next_arg_ptr(args_buf, &mut arg_off);
                        store_float(dst, v, len);
                        matched += 1;
                    }
                }
                None => return matched,
            },
            b'c' => {
                let n = if width == usize::MAX { 1 } else { width };
                let dst = if suppress {
                    std::ptr::null_mut()
                } else {
                    next_arg_ptr(args_buf, &mut arg_off)
                };
                let mut k = 0usize;
                while k < n {
                    if *s == 0 {
                        break;
                    }
                    if !dst.is_null() {
                        *dst.add(k) = *s;
                    }
                    s = s.add(1);
                    k += 1;
                }
                if k < n {
                    return matched;
                }
                if !suppress {
                    matched += 1;
                }
            }
            b'n' => {
                if !suppress {
                    let dst = next_arg_ptr(args_buf, &mut arg_off);
                    let count = (s as usize).wrapping_sub(s_start as usize) as i32;
                    std::ptr::write_unaligned(dst as *mut i32, count);
                }
                // %n does not count towards the return value.
            }
            _ => {
                // Unhandled conversion: no call site in imgui uses anything
                // else (checked exhaustively). Stop rather than risk
                // misinterpreting the vararg buffer layout.
                return matched;
            }
        }
    }
    matched
}

// ---------------------------------------------------------------------
// File I/O and debug logging — no filesystem/console in the browser, so
// these are stubbed as always-fail/no-op. This is a normal, already-handled
// path in Dear ImGui (e.g. `fopen` returning NULL just means "no ini file
// yet, skip loading" — indistinguishable from a fresh install on any
// platform).
// ---------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn fopen(_filename: *const c_char, _mode: *const c_char) -> *mut c_void {
    std::ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn fclose(_stream: *mut c_void) -> c_int {
    0
}

#[no_mangle]
pub unsafe extern "C" fn ftell(_stream: *mut c_void) -> c_int {
    -1
}

#[no_mangle]
pub unsafe extern "C" fn fseek(_stream: *mut c_void, _offset: c_int, _whence: c_int) -> c_int {
    -1
}

#[no_mangle]
pub unsafe extern "C" fn fread(
    _ptr: *mut c_void,
    _size: usize,
    _nmemb: usize,
    _stream: *mut c_void,
) -> usize {
    0
}

#[no_mangle]
pub unsafe extern "C" fn fwrite(
    _ptr: *const c_void,
    _size: usize,
    _nmemb: usize,
    _stream: *mut c_void,
) -> usize {
    0
}

#[no_mangle]
pub unsafe extern "C" fn fflush(_stream: *mut c_void) -> c_int {
    0
}

/// `IMGUI_DEBUG_PRINTF` (imgui_internal.h) expands to a bare `printf`, used
/// only by `ShowDebugLogWindow`'s "output to TTY" toggle — this app never
/// exposes that window, so this is never actually invoked. Stubbed as a
/// no-op matching the wasm import's actual (non-variadic — see module doc
/// comment) 2-argument signature.
#[no_mangle]
pub unsafe extern "C" fn printf(_fmt: *const c_char, _args: *const u8) -> c_int {
    0
}
