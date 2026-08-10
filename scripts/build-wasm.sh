#!/usr/bin/env bash
# Builds the wasm32-unknown-unknown target. Kept as a plain script (not
# .cargo/config.toml) so none of this ever leaks into the native
# `cargo build --profile dev-release` / `cargo test` loop, which must stay
# on plain stable with no special toolchain requirements.
#
# imgui-sys compiles Dear ImGui's C++ source via cc-rs/clang++ for whatever
# target you're building. Two things Apple's Xcode clang can't do that this
# needs:
#   1. wasm32 codegen at all (Apple's clang ships without the wasm backend)
#      -> use Homebrew LLVM's clang++ instead (`brew install llvm`).
#   2. libc headers (string.h, stdlib.h, ...) for that target
#      -> borrow wasi-libc's headers (`brew install wasi-libc`); none of
#         imgui's C++ core does file I/O, so pulling in wasi-libc's headers
#         (without ever touching actual WASI syscalls) is safe here.
# cc-rs also defaults to linking `-lstdc++`, which doesn't exist for this
# target; disabling it works because imgui-sys doesn't rely on the C++
# runtime for anything beyond what's already inlined.
set -euo pipefail
cd "$(dirname "$0")/.."

LLVM_PREFIX="$(brew --prefix llvm)"
WASI_SYSROOT="$(brew --prefix wasi-libc)/share/wasi-sysroot"

export CXX_wasm32_unknown_unknown="$LLVM_PREFIX/bin/clang++"
export AR_wasm32_unknown_unknown="$LLVM_PREFIX/bin/llvm-ar"
export CXXFLAGS_wasm32_unknown_unknown="-isystem $WASI_SYSROOT/include/wasm32-wasi"
export CXXSTDLIB_wasm32_unknown_unknown=""

cargo build --target wasm32-unknown-unknown --bin tv "$@"
