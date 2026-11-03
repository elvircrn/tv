# tv

Vibe-coded native trace viewer for PyTorch/Chrome trace format. macOS only (Metal renderer).

Handles large traces (1M+ events) with GPU-accelerated rendering via imgui + Metal.

Traces load from `.ktz` files — the compressed columnar container written by
the `ktz` tool (`ktz c trace.json trace.json.ktz`). The viewer decodes the
columns directly; the JSON text is never materialized.

## Build

Requires Rust. Install via [rustup](https://rustup.rs/) if you don't have it.

```
cargo build --release
```

Binary at `target/release/tv`.

For development:

```
cargo build --profile dev-release
```

## Test

```
cargo test --profile dev-release
```

Tests run under `dev-release` (not the default `dev` profile) so they reuse the
dev-release dependency tree instead of materializing a second multi-GB
`target/debug` tree.

Both build and tests should pass before committing:

```
cargo build --profile dev-release && cargo test --profile dev-release
```

## Usage

```
tv trace.json.ktz
```

Multiple files open in tabs:

```
tv left.ktz right.ktz
```

Multi-rank distributed traces are detected automatically by filename (`*-rank-N.*`) and merged into a single pane with a shared time axis:

```
tv rank-0.pt.trace.json.ktz rank-1.pt.trace.json.ktz rank-2.pt.trace.json.ktz rank-3.pt.trace.json.ktz
```

You can also drag and drop files (or a directory of them) onto the window.

## Controls

- Scroll to zoom, drag to pan
- WASD to navigate (W/S zoom, A/D pan)
- Click an event to inspect it
- Drag-select a region to see aggregated stats
- `/` to search by kernel name
- Tab to toggle CPU tracks
