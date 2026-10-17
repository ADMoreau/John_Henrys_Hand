"""Occlusion-aware coverage: count only the icon outline that stays visible in the render.

The render draws icons largest footprint first, so smaller ones sit on top and hide whatever
part of a larger icon's outline falls under them. A Canvas keeps, in primary pixels:
  top   : per pixel, the footprint (px) of the topmost icon there (the smallest; inf = empty)
  records: for every primary point an icon covers, the pixel of the icon point that covers it
           and that icon's footprint; a record is visible while no smaller icon lies on its pixel.
A primary point is covered while at least one of its records is visible. Placing an icon both
adds its own (visible) records and can hide records of larger icons under its footprint, so its
gain is what it adds minus what it hides (Canvas.score).
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree
import cover_align as ca


class Canvas:
    def __init__(self, prim, hw):
        self.h, self.w = hw
        self.Pinv = np.linalg.inv(prim.T)
        self.n = len(prim.pts)
        self.top = np.full((self.h, self.w), np.inf, np.float32)
        self.rec_p = np.zeros(0, np.int64)   # primary point index
        self.rec_q = np.zeros(0, np.int64)   # flat pixel of the covering icon point
        self.rec_a = np.zeros(0, np.float32)  # footprint of the covering icon

    # ---- per-placement geometry (worker side) ----
    def footprint(self, sh, path, M, warp):
        """Opaque pixels (alpha > 127) of the placed icon exactly as render_overlay draws it, so
        the draw order (footprint size) matches the render. Returns (y0, x0, mask, area)."""
        import render_overlay as ro
        rgba = ro.load_rgba(path)
        row = ca.placement_row(dict(image=path, M=M, warp=warp), sh, self.Pinv, rgba.shape[:2])
        m = ro.warp_icon(rgba, row, (self.h, self.w))[..., 3] > 127
        ys, xs = np.nonzero(m)
        if not len(ys):
            return 0, 0, np.zeros((1, 1), bool), 0.0
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return int(y0), int(x0), m[y0:y1, x0:x1].copy(), float(len(ys))

    def rec_pixels(self, sh, M, warp, mask, prim, fp, snap=3):
        """Flat pixel of the icon point nearest to each primary point in `mask`, moved to the
        nearest pixel the icon is drawn on (footprint fp) within `snap` px: outline points straddle
        the alpha edge, so they can fall just outside it. Points with no drawn pixel that close
        are dropped (that outline is not on screen). Returns (primary indices, flat pixels)."""
        from scipy import ndimage
        idx = np.flatnonzero(mask)
        y0, x0, m, _ = fp
        if not len(idx) or not m.any():
            return idx[:0], idx[:0]
        Q = ca.apply(self.Pinv, ca.place(sh, M, warp)[0])
        _, nn = cKDTree(Q).query(ca.apply(self.Pinv, prim.pts[idx]))
        q = np.round(Q[nn]).astype(np.int64)
        pad = np.pad(~m, snap + 1, constant_values=True)
        d, (iy, ix) = ndimage.distance_transform_edt(pad, return_indices=True)
        y, x = q[:, 1] - y0 + snap + 1, q[:, 0] - x0 + snap + 1
        ok = (y >= 0) & (y < pad.shape[0]) & (x >= 0) & (x < pad.shape[1])
        ok[ok] = d[y[ok], x[ok]] <= snap
        py = iy[y[ok], x[ok]] + y0 - snap - 1
        px = ix[y[ok], x[ok]] + x0 - snap - 1
        return idx[ok], py * self.w + px

    def attach(self, c, sh, prim):
        """Add the occlusion geometry to a candidate dict from best_for_remainder."""
        c["fp"] = self.footprint(sh, c["image"], c["M"], c.get("warp"))
        c["rp"], c["rq"] = self.rec_pixels(sh, c["M"], c.get("warp"), c["mask"], prim, c["fp"])
        return c

    # ---- state ----
    def _in_fp(self, fp, q):
        y0, x0, m, _ = fp
        y, x = q // self.w - y0, q % self.w - x0
        ok = (y >= 0) & (y < m.shape[0]) & (x >= 0) & (x < m.shape[1])
        out = np.zeros(len(q), bool)
        out[ok] = m[y[ok], x[ok]]
        return out

    def visible(self):
        return self.top.ravel()[self.rec_q] >= self.rec_a

    def covered(self):
        cov = np.zeros(self.n, bool)
        cov[self.rec_p[self.visible()]] = True
        return cov

    def score(self, c, prim, cfg, covered=None, vis=None):
        """Occlusion-aware score of candidate c against the current canvas, with the masks:
        own = primary points c would visibly cover, lost = covered points it would hide."""
        covered = self.covered() if covered is None else covered
        vis = self.visible() if vis is None else vis
        a = c["fp"][3]
        own = np.zeros(self.n, bool)
        own[c["rp"][self.top.ravel()[c["rq"]] >= a]] = True
        hid = vis & (self.rec_a > a)
        hid[hid] = self._in_fp(c["fp"], self.rec_q[hid])
        nvis = np.bincount(self.rec_p[vis], minlength=self.n)
        nhid = np.bincount(self.rec_p[hid], minlength=self.n)
        lost = (nvis > 0) & (nhid == nvis) & ~own
        w = prim.w
        new, overlap = float(w[own & ~covered].sum()), float(w[own & covered].sum())
        return new - cfg.beta * overlap - c["pen"] - float(w[lost].sum()), own, lost

    def add(self, c):
        y0, x0, m, a = c["fp"]
        self.rec_p = np.r_[self.rec_p, c["rp"]]
        self.rec_q = np.r_[self.rec_q, c["rq"]]
        self.rec_a = np.r_[self.rec_a, np.full(len(c["rp"]), a, np.float32)]
        ys, xs = np.nonzero(m)
        ys, xs = ys + y0, xs + x0
        ok = (ys >= 0) & (ys < self.h) & (xs >= 0) & (xs < self.w)
        t = self.top[ys[ok], xs[ok]]
        self.top[ys[ok], xs[ok]] = np.minimum(t, a)
