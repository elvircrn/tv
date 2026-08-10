#!/usr/bin/env bash
# Runs the real-multithreading (Web Worker + rayon) build in a browser via
# Trunk's local dev server. Separate entry point (index-mt.html) and dist dir
# from serve-wasm.sh so the two builds never collide or get confused for one
# another; same COOP/COEP headers apply (Trunk.toml's [serve.headers] isn't
# per-target, so both builds get them, which is harmless for the
# single-threaded one — see Trunk.toml's own comment). See wasm-env-mt.sh for
# the nightly toolchain / atomics+bulk-memory flags this pulls in, and
# index-mt.html for why it needs its own HTML entry point at all.
# --release: see serve-wasm.sh's comment — a debug wasm build is unusably
# slow regardless of how many worker threads it's spread across.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/wasm-env-mt.sh

trunk serve --release --dist dist-mt --port 8081 index-mt.html "$@"
