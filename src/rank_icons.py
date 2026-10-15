#!/usr/bin/env python3
"""Rank library icons by whole-shape similarity to a primary.

  python src/rank_icons.py data/hand_silhouette.png library_dir/ --top 200 --out ranking.csv

Each image is reduced to a filled silhouette (alpha channel, else Otsu). The icon
is tried at 24 rotations x mirror; each variant is cropped to its bounding box and
resized to a square (so any stretch is free, matching the affine search), then
compared with the primary's square by soft IoU. Score = best IoU over variants.

This judges the icon as a whole against the whole primary. It is a cheap
pre-filter: ranked icons still go through cover_align.py for the exact fit.
"""
import argparse, csv, glob, os
import cv2
import numpy as np

N = 64          # canonical square size
ROTS = 24
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def silhouette(path, max_side=160):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError("unreadable")
    if im.ndim == 3 and im.shape[2] == 4:
        m = im[..., 3] > 127
    else:
        g = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border = np.concatenate([b[0], b[-1], b[:, 0], b[:, -1]])
        m = (b < 128) if border.mean() > 127 else (b > 127)
    m = m.astype(np.uint8)
    s = max_side / max(m.shape)
    if s < 1:
        m = cv2.resize(m.astype(np.float32), None, fx=s, fy=s, interpolation=cv2.INTER_AREA) > 0.5
        m = m.astype(np.uint8)
    return m


def canon(mask, angle=0.0, mirror=False):
    """Rotate/mirror, crop to bounding box, resize to N x N (soft mask in [0,1])."""
    m = mask[:, ::-1] if mirror else mask
    if angle:
        d = int(np.ceil(np.hypot(*m.shape))) + 2
        M = cv2.getRotationMatrix2D((m.shape[1] / 2, m.shape[0] / 2), angle, 1.0)
        M[:, 2] += (d - m.shape[1]) / 2, (d - m.shape[0]) / 2
        m = cv2.warpAffine(m.astype(np.float32), M, (d, d), flags=cv2.INTER_LINEAR)
        m = (m > 0.5).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(ys) < 20:
        return None
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    return cv2.resize(m, (N, N), interpolation=cv2.INTER_AREA)


def soft_iou(a, b):
    return float(np.minimum(a, b).sum() / max(np.maximum(a, b).sum(), 1e-9))


def variants(mask):
    out = []
    for mirror in (False, True):
        for a in np.linspace(0, 360, ROTS, endpoint=False):
            c = canon(mask, a, mirror)
            if c is not None:
                out.append(c)
    return out


_TARGET = None


def _score(path):
    try:
        vs = variants(silhouette(path))
        return path, (max(soft_iou(v, _TARGET) for v in vs) if vs else None)
    except Exception:
        return path, None


def rank(primary, library, jobs=None):
    global _TARGET
    _TARGET = canon(silhouette(primary))
    paths = sorted(p for p in glob.glob(os.path.join(library, "**", "*"), recursive=True)
                   if p.lower().endswith(EXTS))
    import multiprocessing as mp
    with mp.get_context("fork").Pool(jobs or os.cpu_count()) as pool:
        res = pool.map(_score, paths, chunksize=32)
    return sorted(((p, s) for p, s in res if s is not None), key=lambda x: -x[1])


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("primary")
    ap.add_argument("library")
    ap.add_argument("--top", type=int, default=30, help="how many to print")
    ap.add_argument("--out", default="ranking.csv")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    ranked = rank(a.primary, a.library, a.jobs)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "iou", "path"])
        for i, (p, s) in enumerate(ranked, 1):
            w.writerow([i, f"{s:.4f}", p])
    print(f"ranked {len(ranked)} images -> {a.out}")
    for i, (p, s) in enumerate(ranked[:a.top], 1):
        print(f"{i:3d}  {s:.3f}  {os.path.basename(p)}")
