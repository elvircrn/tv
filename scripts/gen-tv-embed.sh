#!/usr/bin/env bash
# Build a fully self-contained Trace Viewer page (dist/tv-embed.html) from
# the trunk dist/ output. The wasm-bindgen glue and the wasm binary itself
# are inlined (glue minus its trailing export line, wasm as base64 that is
# decoded and handed to __wbg_init as raw bytes), so the page performs no
# fetch() and works from file:// — which lets other tools embed it via
# srcdoc (see the hard-tools campaign report generator).
#
# Run scripts/build-wasm.sh first; this script verifies the dist artifacts
# are newer than the sources so it never embeds a stale build.
set -euo pipefail
DIST="$(cd "$(dirname "$0")/../dist" && pwd)"
OUT="$DIST/tv-embed.html"

GLUE=$(ls "$DIST"/tv-*.js)
WASM=$(ls "$DIST"/tv-*_bg.wasm)

# Refuse to run on a stale build: every src/ file must predate the wasm.
if [ -n "$(find "$(dirname "$0")/../src" -name '*.rs' -newer "$WASM" -print -quit)" ]; then
  echo "error: src/ is newer than $WASM — run scripts/build-wasm.sh first" >&2
  exit 1
fi

# Strip the trailing `export {...}` so the glue can live in an inline module.
sed 's/^export { initSync, __wbg_init as default };$//' "$GLUE" > /tmp/tv-glue.js
base64 -i "$WASM" | tr -d '\n' > /tmp/tv-wasm.b64

cat > "$OUT" <<HDR
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Trace Viewer</title>
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: #161616; }
  canvas { display: block; outline: none; touch-action: none; }
</style>
</head>
<body>
<!-- Self-contained build of the trace-viewer wasm app: the wasm-bindgen
     glue and the wasm binary itself are inlined below, so this file needs
     no network and no sibling assets, and works when opened from file://.
     Trace loading happens via drag-and-drop onto the canvas (the ?src=
     remote-fetch path does not work from file://). -->
<script type="module">
HDR
cat /tmp/tv-glue.js >> "$OUT"
cat >> "$OUT" <<'FTR'

const B64 = `
FTR
cat /tmp/tv-wasm.b64 >> "$OUT"
cat >> "$OUT" <<'FTR'
`;
const bytes = Uint8Array.from(atob(B64), c => c.charCodeAt(0));
await __wbg_init({ module_or_path: bytes });
FTR
echo "</script>" >> "$OUT"
echo "</body>" >> "$OUT"
echo "</html>" >> "$OUT"
ls -lh "$OUT"
