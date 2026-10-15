#!/usr/bin/env python3
"""Render cover_align results as real image overlays.

  python src/render_overlay.py primary.png result.json out.png

Warps each chosen icon (matrices are source px -> primary px) onto a white canvas at
full opacity with its original colours, larger icons first so smaller ones sit on top;
the primary itself is never drawn, as in the final product. Outputs: <out> per-round strip, <out>_final.png the composite, and
<out>_icons.png the primary next to the composite for comparison.
"""
import json, sys
import cv2
import numpy as np


def load_rgba(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 3:
        im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
    return im


def warp_icon(icon_rgba, row, shape):
    """Icon in primary pixels: affine matrix, or piecewise-affine mesh for warped icons."""
    h, w = shape
    if "mesh_src" not in row:
        return cv2.warpAffine(icon_rgba, np.array(row["matrix_3x3"])[:2], (w, h), flags=cv2.INTER_AREA,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    from skimage.transform import PiecewiseAffineTransform
    src, dst = np.array(row["mesh_dst"]), np.array(row["mesh_src"])  # output px -> icon px
    if hasattr(PiecewiseAffineTransform, "from_estimate"):
        t = PiecewiseAffineTransform.from_estimate(src, dst)
    else:
        t = PiecewiseAffineTransform()
        t.estimate(src, dst)
    # evaluate the mesh map on a coarse grid, upsample it, then remap (skimage's per-pixel
    # warp over the full canvas is minutes per icon)
    step = 4
    ys, xs = np.mgrid[0:h + step:step, 0:w + step:step].astype(np.float32)
    m = t(np.column_stack([xs.ravel(), ys.ravel()])).astype(np.float32).reshape(*xs.shape, 2)
    # outside the mesh, widened by a cell so upsampling never blends real and invalid coordinates
    bad = cv2.dilate((m < 0).any(-1).astype(np.uint8), np.ones((3, 3), np.uint8))
    size = (xs.shape[1] * step, xs.shape[0] * step)
    mx = cv2.resize(m[..., 0], size, interpolation=cv2.INTER_LINEAR)[:h, :w]
    my = cv2.resize(m[..., 1], size, interpolation=cv2.INTER_LINEAR)[:h, :w]
    bad = cv2.resize(bad, size, interpolation=cv2.INTER_NEAREST)[:h, :w] > 0
    mx[bad], my[bad] = -1, -1
    return cv2.remap(icon_rgba, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def over(base, warped, opacity=1.0):
    a = (warped[..., 3:4] / 255.0) * opacity
    return (warped[..., :3] * a + base * (1 - a)).astype(np.uint8)


def composite(warped, shape):
    """Icons on white at full opacity, largest footprint first so smaller icons sit on top."""
    cur = np.full((*shape, 3), 255, np.uint8)
    for wp in sorted(warped, key=lambda x: -(x[..., 3] > 127).sum()):
        cur = over(cur, wp)
    return cur


def main(primary, result, out):
    prim = load_rgba(primary)
    rows = json.load(open(result))
    h, w = prim.shape[:2]
    if len(rows) > 200:  # too many for a per-round sheet: composite only, one icon in memory at a time
        cur = composite_many(rows, (h, w))
        cv2.imwrite(out, cur)
        write_final(prim, cur, out)
        return
    warped, panels = [warp_icon(load_rgba(r["image"]), r, (h, w)) for r in rows], []
    for i, r in enumerate(rows, 1):
        cur = composite(warped[:i], (h, w))
        tile = cv2.resize(cur, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA)
        cap = (f"round {i}: {r['image'].split('/')[-1][:34]}  -> {r['cumulative_coverage']:.0%}"
               + ("  (warped)" if "mesh_src" in r else ""))
        bar = np.full((34, tile.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, cap, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(np.vstack([bar, tile]))
    if not panels:
        sys.exit("no placements in result")
    cols = min(4, len(panels))
    while len(panels) % cols:
        panels.append(np.full_like(panels[0], 255))
    sheet = np.vstack([np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)])
    cv2.imwrite(out, sheet)
    write_final(prim, cur, out)


def composite_many(rows, shape):
    """composite() for many icons: each is warped, cropped to its footprint and kept small."""
    crops = []
    for r in rows:
        wp = warp_icon(load_rgba(r["image"]), r, shape)
        ys, xs = np.nonzero(wp[..., 3])
        if len(ys):
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crops.append(((wp[..., 3] > 127).sum(), y0, x0, wp[y0:y1, x0:x1].copy()))
    cur = np.full((*shape, 3), 255, np.uint8)
    for _, y0, x0, c in sorted(crops, key=lambda x: -x[0]):
        cur[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = over(cur[y0:y0 + c.shape[0], x0:x0 + c.shape[1]], c)
    return cur


def write_final(prim, cur, out):
    h = prim.shape[0]
    cv2.imwrite(out.replace(".png", "_final.png"), cur)
    pa = prim[..., 3:4] / 255.0
    ref = (prim[..., :3] * pa + 255 * (1 - pa)).astype(np.uint8)
    gap = np.full((h, 20, 3), 255, np.uint8)
    cv2.imwrite(out.replace(".png", "_icons.png"), np.hstack([ref, gap, cur]))
    print(f"wrote {out}, {out.replace('.png', '_final.png')} and {out.replace('.png', '_icons.png')}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
