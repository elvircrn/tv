#!/usr/bin/env bash
# Assemble TraceViewer.app — the macOS way to give the binary a real Dock/
# Finder icon (a bare Mach-O can't carry one). macOS only; elsewhere this
# script refuses to run. The icon is the vLLM mark rasterized from the same
# vector geometry the app draws (see scripts/gen-app-icon.py).
#
# Local use: the ad-hoc codesign at the end satisfies Gatekeeper on this
# machine. Distributing to others requires a Developer ID signature +
# notarization, which is out of scope here.
set -euo pipefail
[[ "$(uname)" == "Darwin" ]] || { echo "error: TraceViewer.app is macOS-only" >&2; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
cargo build --release

APP="$ROOT/TraceViewer.app"
ICONSET="$ROOT/target/AppIcon.iconset"
rm -rf "$APP" "$ICONSET"

mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp target/release/tv "$APP/Contents/MacOS/tv"

VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -1)"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleName</key><string>Trace Viewer</string>
	<key>CFBundleDisplayName</key><string>Trace Viewer</string>
	<key>CFBundleIdentifier</key><string>dev.trace-viewer.tv</string>
	<key>CFBundleVersion</key><string>$VERSION</string>
	<key>CFBundleShortVersionString</key><string>$VERSION</string>
	<key>CFBundleExecutable</key><string>tv</string>
	<key>CFBundleIconFile</key><string>AppIcon</string>
	<key>CFBundlePackageType</key><string>APPL</string>
	<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
	<key>LSMinimumSystemVersion</key><string>11.0</string>
	<key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

python3 scripts/gen-app-icon.py "$ICONSET"
iconutil -c icns -o "$APP/Contents/Resources/AppIcon.icns" "$ICONSET"
codesign --force --sign - "$APP"

echo "built $APP"
