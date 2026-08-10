# Sourced by build-wasm.sh / serve-wasm.sh. See build-wasm.sh for why each
# of these is needed (Apple clang has no wasm32 backend, wasi-libc supplies
# borrowed libc headers, cc-rs's default -lstdc++ doesn't exist here).
LLVM_PREFIX="$(brew --prefix llvm)"
WASI_SYSROOT="$(brew --prefix wasi-libc)/share/wasi-sysroot"
# Repo root. `git rev-parse` works regardless of the caller's cwd (as long as
# it's somewhere inside the repo) and, unlike `${BASH_SOURCE[0]}`-based
# resolution, doesn't depend on this file being sourced from an actual bash
# (as opposed to zsh, dash, ...) — this file is meant to be `source`d from
# any interactive shell, not just from build-wasm.sh/serve-wasm.sh.
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

export CXX_wasm32_unknown_unknown="$LLVM_PREFIX/bin/clang++"
export AR_wasm32_unknown_unknown="$LLVM_PREFIX/bin/llvm-ar"
# -mno-reference-types/-mno-multivalue: these are ABI-affecting wasm features
# that clang enables by default. imgui-sys's C++ doesn't use either, but if
# left on, the linked module's target_features section disagrees with what
# wasm-bindgen expects from Rust's own (matching, ABI-consistent) defaults,
# and wasm-bindgen-cli fails with "failed to find the
# `__wbindgen_externref_table_alloc` function".
# https://github.com/wasm-bindgen/wasm-bindgen/issues/4654
#
# -DNDEBUG: wasm32-unknown-unknown has no libc, so imgui's IM_ASSERT (a plain
# `assert()`) drags in an unresolvable `__assert_fail` import. NDEBUG makes
# assert() (and stb_rectpack/stb_truetype's asserts, which route through the
# same macro) a no-op, per the standard C convention, eliminating the need
# for `__assert_fail` entirely rather than having to implement it.
#
# -DIMGUI_USE_STB_SPRINTF + IMGUI_STB_SPRINTF_FILENAME: routes every
# ImFormatString/ImFormatStringV call (i.e. essentially all widget
# label/ID/tooltip formatting) through stb_sprintf.h's bundled,
# libc-independent sprintf instead of calling libc's vsnprintf, which
# doesn't exist on this target. See third-party/stb_sprintf.h (vendored
# from https://github.com/nothings/stb, public domain).
export CXXFLAGS_wasm32_unknown_unknown="-isystem $WASI_SYSROOT/include/wasm32-wasi -mno-reference-types -mno-multivalue -DNDEBUG -DIMGUI_USE_STB_SPRINTF -DIMGUI_STB_SPRINTF_FILENAME=\"$REPO_ROOT/third-party/stb_sprintf.h\""
export CXXSTDLIB_wasm32_unknown_unknown=""
