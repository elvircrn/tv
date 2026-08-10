#!/usr/bin/env bash
# Runs the app in a browser via trunk's local dev server, with the
# Cross-Origin-Isolation headers the real-threading phase needs (see
# Trunk.toml). See wasm-env.sh for the imgui-sys C++ cross-compile workaround.
#
# --release: a debug wasm build is unusably slow for anything beyond "does
# it compile" — no LLVM optimization at all makes the same CPU-bound work
# (search, timeline layout on pan/zoom) an order of magnitude slower than
# native, which reads as broken rather than "just debug." For a fast
# compile-error-checking loop instead of actually using the app in a
# browser, use `scripts/build-wasm.sh` (plain debug, no trunk/serving).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/wasm-env.sh

trunk serve --release "$@"
