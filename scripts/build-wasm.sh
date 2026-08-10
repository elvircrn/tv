#!/usr/bin/env bash
# Builds the wasm32-unknown-unknown target. Kept as a plain script (not
# .cargo/config.toml) so none of this ever leaks into the native
# `cargo build --profile dev-release` / `cargo test` loop, which must stay
# on plain stable with no special toolchain requirements. See wasm-env.sh
# for why each env var below is needed.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/wasm-env.sh

cargo build --target wasm32-unknown-unknown --bin tv "$@"
