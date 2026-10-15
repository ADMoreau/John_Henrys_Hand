#!/usr/bin/env python3
"""Per-image style features for separating icon/cartoon art from photos and textures.

  python src/style_filter.py LIBRARY_DIR runs/style.npz

On a 256 px thumbnail of each image's visible content (cropped to its bounding box):
  flat   fraction of pixels whose 5x5 colour neighbourhood is nearly constant
         (flat fills in cartoons and icons; photos are textured everywhere)
  ncol   number of 4-bit-per-channel colours needed to cover 90% of the pixels
  soft   fraction of the visible area that is semi-transparent (flares, smoke, clouds)
  fill   opaque fraction of the thumbnail
  frag   fraction of the visible area in specks (components < 0.5% of it): sprays, dust
  cover  visible content's bounding box as a fraction of the canvas
"""
import os, sys
import multiprocessing as mp
import cv2
import numpy as np


def features(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 3:
        im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
    ys, xs = np.nonzero(im[..., 3] > 16)
    if len(ys) == 0:
        return np.array([0, 999, 1, 0, 1, 0], np.float32)
    cover = (np.ptp(ys) + 1) * (np.ptp(xs) + 1) / (im.shape[0] * im.shape[1])
    im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    f = 256 / max(im.shape[:2])
    size = (max(1, round(im.shape[1] * f)), max(1, round(im.shape[0] * f)))
    # area-averaged mask keeps thin lines connected (for specks); nearest sampling keeps each
    # pixel's true alpha and colour, so thin opaque lines don't read as semi-transparent
    va = cv2.resize(im[..., 3], size, interpolation=cv2.INTER_AREA) > 16
    _, _, st, _ = cv2.connectedComponentsWithStats(va.astype(np.uint8), connectivity=8)
    small = st[1:, cv2.CC_STAT_AREA] < 0.005 * va.sum()
    frag = float(st[1:, cv2.CC_STAT_AREA][small].sum() / max(va.sum(), 1))
    im = cv2.resize(im, size, interpolation=cv2.INTER_NEAREST)
    a = im[..., 3]
    vis, opq = a > 16, a > 240
    if opq.sum() < 50:
        return np.array([0, 999, 1, opq.mean(), frag, cover], np.float32)
    lab = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2LAB).astype(np.float32)
    mu = cv2.blur(lab, (5, 5))
    var = cv2.blur(lab * lab, (5, 5)) - mu * mu
    sd = np.sqrt(np.clip(var, 0, None).sum(-1))
    core = cv2.erode(opq.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if core.sum() < 20:
        core = opq
    flat = float((sd[core] < 4.0).mean())
    q = (im[..., :3][opq] >> 4).astype(np.int32)
    _, cnt = np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2], return_counts=True)
    cnt = np.sort(cnt)[::-1]
    ncol = int(np.searchsorted(np.cumsum(cnt), 0.9 * cnt.sum()) + 1)
    soft = float((vis & ~opq).sum() / max(vis.sum(), 1))
    return np.array([flat, ncol, soft, opq.mean(), frag, cover], np.float32)


def main(root, out):
    paths = sorted(os.path.join(d, f) for d, _, fs in os.walk(root, followlinks=True)
                   for f in fs if f.lower().endswith(".png"))
    with mp.Pool(14) as pool:
        feats = pool.map(features, paths, chunksize=64)
    ok = [i for i, x in enumerate(feats) if x is not None]
    np.savez(out, paths=np.array([paths[i] for i in ok]), feats=np.stack([feats[i] for i in ok]),
             names=np.array(["flat", "ncol", "soft", "fill", "frag", "cover"]))
    print(f"{len(ok)} of {len(paths)} images -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
