# tv

Trace viewer for Chrome trace format (JSON). macOS only (Metal renderer).

## Build

```
cargo build --release
```

Binary lands at `target/release/tv`.

For faster iteration during development:

```
cargo build --profile dev-release
```

## Usage

```
tv trace.json
tv trace.json.gz
tv left.json right.json   # split-pane diff
```

## Tests

```
cargo test
```
