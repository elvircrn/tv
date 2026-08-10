# Sourced by build-wasm-mt.sh / serve-wasm-mt.sh — the real-multithreading
# (Web Worker + rayon, via wasm-bindgen-rayon) wasm build. Layers
# atomics/bulk-memory support on top of everything wasm-env.sh already sets
# up for the single-threaded stable build (LLVM cross-compiler, wasi-libc
# headers, stb_sprintf shim, etc.) — none of that changes; only the
# target-feature baseline does.
#
# Relative to cwd (repo root), matching wasm-env.sh's own convention, not
# `${BASH_SOURCE[0]}` — the callers (build-wasm-mt.sh / serve-wasm-mt.sh)
# already `cd` to the repo root before sourcing this, and `BASH_SOURCE` isn't
# reliably populated when this runs under a non-bash interactive shell (zsh)
# sourcing it directly for manual debugging, unlike `$0`-based resolution.
source scripts/wasm-env.sh

# Forces the `cargo`/`rustc` binaries the rustup shims on PATH resolve to
# (which is what Trunk shells out to — it has no `+nightly`-equivalent flag
# of its own) onto the nightly toolchain. `rustup component add rust-src
# --toolchain nightly` and `rustup target add wasm32-unknown-unknown
# --toolchain nightly` must already be done (one-time, see task notes) for
# `-Z build-std` below to have a `std` source tree to rebuild from.
export RUSTUP_TOOLCHAIN=nightly

# wasm32's prebuilt `std` is compiled without atomics for maximum
# portability, so getting a thread-safe `std::sync`/`std::thread` requires
# rebuilding it from source with the same feature flags as the rest of the
# crate graph — that's what `-Z build-std` in .cargo/config.toml's
# `[unstable]` table (nightly-only; inert on the stable channel every other
# build in this repo uses) is for. The actual feature/link flags have to live
# here instead of that same file, though: `[target.wasm32-unknown-unknown]
# .rustflags` in .cargo/config.toml would apply unconditionally to *every*
# wasm32 build regardless of toolchain, including the plain stable
# single-threaded one in build-wasm.sh/serve-wasm.sh — which would then try
# to link atomics-enabled, shared-memory-importing code against the
# prebuilt (non-atomics) stable std, corrupting that build instead of fixing
# this one. RUSTFLAGS as a script-scoped env var only ever reaches the
# invocation that sources this file.
#
# Flag rundown (see https://github.com/RReverser/wasm-bindgen-rayon for the
# canonical recipe this mirrors):
#   target-feature=+atomics,+bulk-memory  - the actual wasm threading/shared-
#     memory proposals; `+mutable-globals` is Rust's default for this target
#     already (unlike the other two), so it's not re-listed here.
#   link-arg=--shared-memory    - makes the linked module import (not own) a
#     growable shared `WebAssembly.Memory`, backed by `SharedArrayBuffer` —
#     the whole point, and why Trunk.toml's COOP/COEP headers exist.
#   link-arg=--max-memory=4294967296 - reserves up to the full 4GiB a wasm32
#     address space can address; shared memory's max must be fixed at link
#     time (unlike a normal growable memory, workers need to agree up front).
#   link-arg=--import-memory    - required alongside --shared-memory: every
#     Worker instantiates the *same* module against the *same* memory object,
#     which only works if the module imports it rather than allocating its
#     own on each instantiation.
#   link-arg=--export=__wasm_init_tls / __tls_size / __tls_align / __tls_base
#     - each Worker has its own thread-local storage block carved out of the
#     shared linear memory; wasm-bindgen-rayon's JS glue
#     (workerHelpers.no-bundler.js) calls these exports to set up a fresh TLS
#     block on every new Worker before running any Rust code on it.
export RUSTFLAGS="-C target-feature=+atomics,+bulk-memory \
-C link-arg=--shared-memory \
-C link-arg=--max-memory=4294967296 \
-C link-arg=--import-memory \
-C link-arg=--export=__wasm_init_tls \
-C link-arg=--export=__tls_size \
-C link-arg=--export=__tls_align \
-C link-arg=--export=__tls_base"

# wasm-ld enforces that a shared-memory module's target_features section
# agree across *every* linked object, Rust or not — the same class of bug
# `-mno-reference-types`/`-mno-multivalue` above (in wasm-env.sh) works
# around, just in the opposite direction this time: there, Clang's *default*
# features had to be turned off to match Rust's non-atomics default; here,
# Clang has to explicitly opt *in* to atomics/bulk-memory to match Rust's
# (now non-default, RUSTFLAGS-enabled) opt-in. Without this, linking fails
# with wasm-ld's "--shared-memory is disallowed" class of error because
# imgui-sys's C++ objects don't declare the same feature set as the Rust
# objects being linked alongside them.
export CXXFLAGS_wasm32_unknown_unknown="$CXXFLAGS_wasm32_unknown_unknown -matomics -mbulk-memory"

# `ImGui::LogToTTY()` references the libc global `stdout` (a real `FILE*`
# object, not a function — wasm_libc_shims.rs only shims *functions*, and
# there's nothing to sensibly shim a TTY stream to in a browser anyway).
# It's dead code in the plain stable build (never called, so wasm-ld's
# function-level `--gc-sections` drops it) but the shared-memory linker mode
# `--shared-memory`/`--import-memory` require keeps every exported symbol's
# transitive callees live for correctness of multi-instantiation, which pulls
# LogToTTY (and its `stdout` reference) in as an undefined-symbol link
# failure. Disabling the whole TTY logging feature (also legitimately
# meaningless in a browser) removes the reference instead of needing a fake
# `stdout` shim.
export CXXFLAGS_wasm32_unknown_unknown="$CXXFLAGS_wasm32_unknown_unknown -DIMGUI_DISABLE_TTY_FUNCTIONS"
