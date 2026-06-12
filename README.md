# tv

Native trace viewer for PyTorch/Chrome trace format. macOS only (Metal renderer).

Handles large traces (1M+ events) with GPU-accelerated rendering via imgui + Metal.

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
cargo test
```

Both build and tests should pass before committing:

```
cargo build --profile dev-release && cargo test
```

## Usage

```
tv trace.json
tv trace.json.gz
tv trace.tar.gz
```

Multiple files open in split panes:

```
tv left.json right.json
```

Multi-rank distributed traces are detected automatically by filename (`*-rank-N.*`) and merged into a single pane with a shared time axis:

```
tv rank-0.pt.trace.json.gz rank-1.pt.trace.json.gz rank-2.pt.trace.json.gz rank-3.pt.trace.json.gz
```

You can also drag and drop files onto the window.

## Controls

- Scroll to zoom, drag to pan
- WASD to navigate (W/S zoom, A/D pan)
- Click an event to inspect it
- Drag-select a region to see aggregated stats
- `/` to search by kernel name
- Tab to toggle CPU tracks
