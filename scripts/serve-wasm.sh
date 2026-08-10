#!/usr/bin/env bash
# Runs the app in a browser via trunk's local dev server, with the
# Cross-Origin-Isolation headers the real-threading phase needs (see
# Trunk.toml). See wasm-env.sh for the imgui-sys C++ cross-compile workaround.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/wasm-env.sh

trunk serve "$@"
