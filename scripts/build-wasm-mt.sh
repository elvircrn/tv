#!/usr/bin/env bash
# Builds the wasm32-unknown-unknown target with real Web Worker/rayon
# parallelism (the `mt` Cargo feature — see Cargo.toml). Separate from
# build-wasm.sh (the plain single-threaded, stable-toolchain build) rather
# than a flag on it, so that script keeps working with zero extra toolchain
# requirements as a robustness fallback. See wasm-env-mt.sh for why each env
# var this pulls in is needed (nightly toolchain, atomics/bulk-memory
# RUSTFLAGS, matching C++ CXXFLAGS).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/wasm-env-mt.sh

cargo build --target wasm32-unknown-unknown --bin tv --features mt "$@"
