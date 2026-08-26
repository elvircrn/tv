#!/usr/bin/env bash
# Builds the wasm app and deploys it to GitHub Pages (elvircrn.github.io/tv/)
# by pushing dist/ to the gh-pages branch. Deliberately NOT rebuilt in CI —
# see the commit that introduced this script for why: CI's from-scratch
# `cargo build` can never carry CLAUDE.md's 32-bit ImDrawIdx patch (applied
# by hand-editing files in ~/.cargo/registry, not expressible in Cargo.lock),
# which large (1M+ event) traces need to not overflow imgui's default 16-bit
# vertex-index limit. Deploying exactly what's built and verified here avoids
# that gap entirely.
#
# Run this after verifying changes locally (scripts/serve-wasm.sh), from a
# clean working tree on the commit you actually want live.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/wasm-env.sh
rm -rf target/wasm32-unknown-unknown target/wasm-bindgen target/wasm-opt dist
trunk build --release --public-url /tv/

# Trunk unconditionally emits a `<link rel="preload" ... as="fetch">` for the
# wasm binary regardless of whether a data-initializer is configured (see
# web/initializer.mjs). That preload races the browser's own fetcher ahead
# of — and gets coalesced with — the initializer's progress-tracked fetch()
# for the same URL, so by the time our script runs the file is often already
# served from cache with no visible download left to show progress for
# (trunk-rs/trunk has no attribute to opt a single link out of this). Strip
# just that one tag; the JS/initializer modulepreloads are untouched and
# still load eagerly.
python3 -c "
import re
path = 'dist/index.html'
html = open(path).read()
new_html = re.sub(r'<link rel=\"preload\"[^>]*type=\"application/wasm\"[^>]*>', '', html)
if new_html == html:
    raise SystemExit('expected to strip a wasm preload link but found none — did trunk change its output format?')
open(path, 'w').write(new_html)
"

commit=$(git rev-parse --short HEAD)
worktree_dir=$(mktemp -d)
trap 'git worktree remove --force "$worktree_dir" 2>/dev/null || true' EXIT

if git show-ref --verify --quiet refs/remotes/origin/gh-pages; then
  git worktree add "$worktree_dir" gh-pages
else
  git worktree add --orphan -b gh-pages "$worktree_dir"
fi

# Clear out the previous deploy's files (old-hash JS/wasm would otherwise
# accumulate forever — trunk names every build's assets with a fresh content
# hash) before copying the new build in.
find "$worktree_dir" -mindepth 1 -maxdepth 1 -not -name '.git' -exec rm -rf {} +
cp -r dist/* "$worktree_dir"/

git -C "$worktree_dir" add -A
if git -C "$worktree_dir" diff --cached --quiet; then
  echo "Nothing changed — dist/ output is identical to what's already deployed."
else
  git -C "$worktree_dir" commit -m "Deploy: $commit"
  git -C "$worktree_dir" push origin gh-pages
  echo "Pushed. https://elvircrn.github.io/tv/ should update within a minute or two."
fi
