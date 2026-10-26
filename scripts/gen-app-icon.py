#!/usr/bin/env python3
"""Rasterize the vLLM mark into a macOS .iconset for TraceViewer.app.

Uses the exact vector geometry from `draw_vllm_logo` (src/ui.rs) — the
official mark from vllm-project/media-kit, yellow triangle #fdb515 and blue
quadrilateral #30a2ff on a 32-unit grid — rendered with supersampled
coverage (anti-aliasing) over a white rounded-rect background. Straight-alpha RGBA
PNGs, stdlib only.

Usage: gen-app-icon.py <output.iconset>
"""
import struct
import sys
import zlib
from pathlib import Path

# Coordinates copied from draw_vllm_logo (src/ui.rs), y-down, 32-unit grid.
YELLOW = (253, 181, 21, 255)
BLUE = (48, 162, 255, 255)
BACKGROUND = (255, 255, 255, 255)
YELLOW_PTS = [(13.417, 7.835), (13.417, 30.286), (2.191, 7.835)]
BLUE_PTS = [(13.416, 30.286), (22.237, 30.286), (29.809, 1.714), (19.427, 7.179)]
GRID = 32.0
BOX_FRACTION = 0.68  # the 32-unit box occupies this share of the canvas
SS = 4  # supersampling factor per axis (16 samples/pixel)

SIZES = {  # iconutil's required iconset members
    "icon_16x16.png": 16, "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32, "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128, "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256, "icon_256x256@2x.png": 512,
    "icon_512x512.png": 512, "icon_512x512@2x.png": 1024,
}


def inside_convex(pts, x, y):
    """Cross-product sign test; pts must be in consistent winding order."""
    sign = 0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        if cross != 0:
            s = 1 if cross > 0 else -1
            if sign and s != sign:
                return False
            sign = s
    return True


def render(size):
    half = size / 2.0
    radius = size * 0.2237  # macOS icon corner radius (~Big Sur squircle)
    box = size * BOX_FRACTION
    scale = box / GRID
    origin = (size - box) / 2.0
    step = 1.0 / SS
    rgba = bytearray(size * size * 4)
    for py in range(size):
        for px in range(size):
            rect_cov = 0
            mark_cov = 0
            mark_color = None
            for sy in range(SS):
                for sx in range(SS):
                    x = px + (sx + 0.5) * step
                    y = py + (sy + 0.5) * step
                    dx = max(abs(x - half) - (half - radius), 0.0)
                    dy = max(abs(y - half) - (half - radius), 0.0)
                    if dx * dx + dy * dy <= radius * radius:
                        rect_cov += 1
                    # Mark coordinates: the 32-unit grid maps onto the box.
                    gx = (x - origin) / scale
                    gy = (y - origin) / scale
                    if inside_convex(BLUE_PTS, gx, gy):  # blue draws over yellow
                        mark_cov += 1
                        mark_color = BLUE
                    elif inside_convex(YELLOW_PTS, gx, gy):
                        mark_cov += 1
                        mark_color = YELLOW
            a_rect = rect_cov / (SS * SS)
            m = mark_cov / (SS * SS)
            # Source-over the mark onto the (possibly transparent) background.
            out_a = m + a_rect * (1 - m)
            if out_a == 0:
                rgb = (0, 0, 0)
            else:
                src = mark_color[:3] if m > 0 else BACKGROUND[:3]
                bg = BACKGROUND[:3]
                rgb = tuple(
                    round((s * m + b * a_rect * (1 - m)) / out_a)
                    for s, b in zip(src, bg))
            i = (py * size + px) * 4
            rgba[i:i + 4] = bytes(rgb) + bytes([round(out_a * 255)])
    return rgba


def write_png(path, size, rgba):
    def chunk(tag, data):
        out = struct.pack(">I", len(data)) + tag + data
        return out + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(rgba[y * size * 4:(y + 1) * size * 4])
                   for y in range(size))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b""))


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, size in SIZES.items():
        write_png(out_dir / name, size, render(size))
        print(f"{name}: {size}x{size}")


if __name__ == "__main__":
    main()
