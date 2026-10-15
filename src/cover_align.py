#!/usr/bin/env python3
"""
cover_align v2 - cover a primary outline with transformed library icons.

What changed from v1
  * Edge point sets: silhouette + internal lines (eyes, hair, etc.), not just
    the outer contour. Transparent PNGs use the alpha channel as the mask.
  * Every point carries a tangent direction; ICP matches and coverage both
    require tangents to roughly agree (no credit for perpendicular crossings).
  * Sampling by spacing, not by count: a dense set for scoring, a coarse
    random subset for ICP speed.
  * Affine by default with a non-degeneracy guard: singular values are clipped
    so a shape can stretch up to --max-aniso:1 but never flattens into a line.
  * Placement score = new coverage - beta*overlap - alpha*stray length.
  * Global search: ~1M affine hypotheses (incl. stretch) scored by distance-map
    lookup, then ICP refines the best 40. Local ICP alone can't find
    strongly stretched partial matches from undistorted starting guesses.
  * Residual-driven selection: each round re-aligns every remaining icon to
    the still-uncovered part of the primary, then takes the best placement.
  * --exact re-picks the final set by integer programming (scipy milp) over
    every placement seen: the true optimum for that candidate pool.
  * --selftest builds a ground-truth puzzle and checks the pieces are found.

Usage
  python src/cover_align.py primary.png library_dir/ --out result.png [--exact]
  python src/cover_align.py --selftest [--exact]
Inputs: images (png/jpg/bmp/webp) or .npy closed polylines (Nx2).
Deps: numpy, scipy>=1.9, opencv-python (images only), matplotlib (plot only)
"""
import argparse, glob, json, os, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.spatial import cKDTree


# ---------------- geometry ----------------
def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def apply(M, pts):
    return pts @ M[:2, :2].T + M[:2, 2]


def apply_dir(M, tan):
    v = tan @ M[:2, :2].T
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def normalizer(pts):
    """3x3 matrix mapping pts to zero mean, unit RMS radius."""
    c = pts.mean(0)
    s = np.sqrt(((pts - c) ** 2).sum(1).mean())
    return np.array([[1 / s, 0, -c[0] / s], [0, 1 / s, -c[1] / s], [0, 0, 1.0]])


def fps_order(pts, n):
    """Farthest-point sampling: n indices spread evenly over pts. Deterministic (starts
    nearest the centroid), and every prefix is itself a well-spread subset."""
    n = min(n, len(pts))
    idx = [int(np.argmin(((pts - pts.mean(0)) ** 2).sum(1)))]
    d = ((pts - pts[idx[0]]) ** 2).sum(1)
    for _ in range(n - 1):
        idx.append(int(d.argmax()))
        d = np.minimum(d, ((pts - pts[idx[-1]]) ** 2).sum(1))
    return np.array(idx)


def grid_subsample(pts, tan, w, spacing):
    """One point per spacing-sized cell -> roughly uniform density (first point in a cell wins)."""
    keys = np.floor(pts / spacing).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return pts[idx], tan[idx], w[idx]


# ---------------- loading ----------------
def polyline_points(poly, step):
    loop = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1], step)
    pts = np.column_stack([np.interp(t, s, loop[:, 0]), np.interp(t, s, loop[:, 1])])
    tan = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
    return pts, tan / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12)


def simplified_regions(img, mask, cfg, min_region=None):
    """Low-detail edge map for icons: flatten texture, reduce to a few colour regions,
    drop small regions (text, pattern cells), merge neighbours that differ mostly in
    brightness (shading), and keep only the region boundaries + silhouette."""
    import cv2
    from scipy import ndimage as ndi
    from skimage import graph
    bgr = np.ascontiguousarray(img[:, :, :3]) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(cv2.pyrMeanShiftFiltering(bgr, 10, 28), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 0.4  # brightness differences (lighting) count less than colour differences
    inside = mask.ravel()
    K = min(cfg.icon_colors, int(inside.sum()))
    _, lbl, _ = cv2.kmeans(lab.reshape(-1, 3)[inside], K, None,
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    L = np.full(mask.size, -1)
    L[inside] = lbl.ravel()
    L = L.reshape(mask.shape)
    reg, nxt = np.zeros(mask.shape, np.int64), 1
    for c in range(K):
        r, n = ndi.label(L == c)
        reg[r > 0] = r[r > 0] + nxt - 1
        nxt += n
    small = np.bincount(reg.ravel()) < (min_region or cfg.icon_min_region) * mask.sum()
    small[0] = False
    _, (iy, ix) = ndi.distance_transform_edt(small[reg] & mask | ~mask, return_indices=True)
    reg = reg[iy, ix]
    reg[~mask] = 0
    rag = graph.rag_mean_color(lab, reg, mode="distance")
    if 0 in rag:
        rag.remove_node(0)
    reg = graph.cut_threshold(reg, rag, cfg.icon_merge, in_place=False)
    reg[~mask] = -1
    b = np.zeros(mask.shape, bool)
    b[:, 1:] |= reg[:, 1:] != reg[:, :-1]
    b[1:, :] |= reg[1:, :] != reg[:-1, :]
    return b


def clean_silhouette(mask, open_r=2, keep_frac=0.05, smooth=1.5, fill=True):
    """Icon silhouette cleanup: open away thin streaks/wisps, drop specks much smaller than
    the main body (comparable parts, e.g. a pair of mittens, are kept), fill holes, smooth."""
    import cv2
    from scipy import ndimage as ndi
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1, 2 * open_r + 1))
    m = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_OPEN, k)
    lab, n = ndi.label(m)
    if n:
        a = ndi.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.flatnonzero(a >= keep_frac * a.max()))
    if fill:
        m = ndi.binary_fill_holes(m)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), smooth) > 0.5
    return np.where(m, 255, 0).astype(np.uint8)


def centerlines(gray, inside, min_px=8):
    """One-pixel centre lines of the drawn strokes inside the silhouette, so each line is one
    run of points (Canny traces both sides of every stroke, so a line counted up to twice).
    Strokes are the minority tone: light lines on a dark woodcut, or dark lines on paper."""
    import cv2
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    v = gray[inside]
    t, _ = cv2.threshold(v.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = ((gray > t) if (v > t).mean() < 0.5 else (gray <= t)) & inside
    lab, _ = ndi.label(ink, np.ones((3, 3)))
    keep = np.bincount(lab.ravel()) >= min_px  # drop specks
    keep[0] = False
    return skeletonize(keep[lab])


def image_edges(path, cfg=None, icon_edges="full"):
    """Edge points of an image. icon_edges: "full" = silhouette + all Canny edges inside
    (the primary's default), "centerline" = silhouette + the centre line of each drawn
    stroke (the primary with --prim-edges centerline), "simplified" = silhouette + boundaries between flattened
    colour regions, "outline" = silhouette boundary only, "detail" = like simplified but
    keeps small regions, letters and holes (light cleanup), so an icon's inner lines must
    land on the primary's lines too."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("unreadable")
    f = 1.0
    if icon_edges not in ("full", "centerline") and max(img.shape[:2]) > 256:  # detail is removed anyway; work small
        f = 256 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    if img.ndim == 3 and img.shape[2] == 4:  # transparent PNG: alpha is the mask
        mask = np.where(img[:, :, 3] > 127, 255, 0).astype(np.uint8)
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
        if border.mean() > 127:
            mask = 255 - mask
    if icon_edges == "detail":
        mask = clean_silhouette(mask, open_r=0, keep_frac=0.003, smooth=0.7, fill=False)
        if mask.sum() == 0:
            raise ValueError("empty silhouette after cleanup")
    elif icon_edges not in ("full", "centerline"):
        mask = clean_silhouette(mask)
        if mask.sum() == 0:
            raise ValueError("empty silhouette after cleanup")
    k = np.ones((3, 3), np.uint8)
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k) > 0
    if icon_edges == "outline":
        inner = np.zeros_like(boundary)
    elif icon_edges in ("simplified", "detail"):
        inner = simplified_regions(img, mask > 0, cfg, 0.002 if icon_edges == "detail" else None) \
            & (cv2.erode(mask, k, iterations=1 if icon_edges == "detail" else 2) > 0)
    elif icon_edges == "centerline":
        inner = centerlines(gray, cv2.erode(mask, k, iterations=2) > 0)
    else:
        inner = (cv2.Canny(gray, 50, 150) > 0) & (cv2.erode(mask, k, iterations=2) > 0)
    E = (boundary | inner).astype(np.float32)
    if E.sum() < 10:
        raise ValueError("no edges found")
    # tangent = 90 deg from the dominant gradient direction (structure tensor)
    B = cv2.GaussianBlur(E, (0, 0), 1.5)
    gx, gy = cv2.Sobel(B, cv2.CV_32F, 1, 0), cv2.Sobel(B, cv2.CV_32F, 0, 1)
    jxx, jxy, jyy = (cv2.GaussianBlur(g, (0, 0), 2) for g in (gx * gx, gx * gy, gy * gy))
    ys, xs = np.nonzero(E)
    th = 0.5 * np.arctan2(2 * jxy[ys, xs], jxx[ys, xs] - jyy[ys, xs]) + np.pi / 2
    # points are reported in the original image's pixels, so output matrices stay valid
    return np.column_stack([xs, ys]) / f, np.column_stack([np.cos(th), np.sin(th)]), boundary[ys, xs]


def load_raw(path, cfg=None, icon_edges="full"):
    if path.lower().endswith(".npy"):
        poly = np.load(path).astype(float)
        per = np.linalg.norm(np.diff(np.vstack([poly, poly[:1]]), axis=0), axis=1).sum()
        pts, tan = polyline_points(poly, per / 5000)
        return pts, tan, np.ones(len(pts), bool)
    return image_edges(path, cfg, icon_edges)


def silhouette_area(path):
    """Silhouette area in the image's own pixels (alpha > 127, or Otsu foreground)."""
    if path.lower().endswith(".npy"):
        x, y = np.load(path).astype(float).T
        return 0.5 * abs(x @ np.roll(y, 1) - y @ np.roll(x, 1))
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3 and img.shape[2] == 4:
        return float((img[:, :, 3] > 127).sum())
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = (mask > 127).mean()
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return float(mask.size * ((1 - fg) if border.mean() > 127 else fg))


class Shape:
    def __init__(self, path, cfg, rng, icon=False):
        pts, tan, bnd = load_raw(path, cfg, cfg.icon_edges if icon else cfg.prim_edges)
        self.T = normalizer(pts)  # pixels -> unit RMS radius (tangents unaffected)
        self.area = silhouette_area(path) * abs(np.linalg.det(self.T[:2, :2]))  # in normalised units
        self.px = np.sqrt(abs(np.linalg.det(self.T[:2, :2])))  # normalised units per source pixel
        order = np.argsort(~bnd, kind="stable")  # outline first, so it wins shared cells
        w = np.where(bnd, 1.0, cfg.interior_weight)[order]
        self.spacing = cfg.spacing if icon else cfg.prim_spacing  # point spacing, own units
        self.pts, self.tan, wb = grid_subsample(apply(self.T, pts)[order], tan[order],
                                                np.column_stack([w, bnd[order]]), self.spacing)
        self.w, self.bnd = wb[:, 0], wb[:, 1] > 0
        sel = fps_order(self.pts, cfg.n_coarse)  # coarse set for ICP; its prefix seeds the global search
        self.cpts, self.ctan = self.pts[sel], self.tan[sel]
        self.tree = cKDTree(self.pts)
        if icon:
            build_chains(self, cfg)  # icon-side runs for the strict on-a-line test

    def __setstate__(self, d):
        d.setdefault("spacing", 0.02)  # pickled before per-shape spacing: the default tol/2
        self.__dict__.update(d)


# ---------------- alignment ----------------
def fit(S, D, cfg, reflect):
    """Least-squares transform S -> D, then clip singular values (no collapse)."""
    ms, md = S.mean(0), D.mean(0)
    Sc, Dc = S - ms, D - md
    if cfg.mode == "similarity":
        var = (Sc ** 2).sum() / len(S)
        if var < 1e-12:
            return None
        U, sv, Vt = np.linalg.svd(Dc.T @ Sc / len(S))
        d = (-1.0 if reflect else 1.0) * np.sign(np.linalg.det(U @ Vt))
        A = (sv[0] + d * sv[1]) / var * (U @ np.diag([1.0, d]) @ Vt)
    else:
        X, *_ = np.linalg.lstsq(Sc, Dc, rcond=None)
        A = X.T
    U, sv, Vt = np.linalg.svd(A)
    sv = np.clip(sv, *cfg.scale_range)
    sv[1] = max(sv[1], sv[0] / cfg.max_aniso)  # the anti-flattening guard
    A = U @ np.diag(sv) @ Vt
    M = np.eye(3)
    M[:2, :2], M[:2, 2] = A, md - A @ ms
    return M


def icp(S, St, TP, TT, tree, M, cfg, iters):
    """Trimmed, tangent-gated ICP of points S (tangents St) onto target TP/TT."""
    n_keep = max(int(cfg.trim * len(S)), 6)
    for _ in range(iters):
        d, idx = tree.query(apply(M, S))
        ok = np.abs((apply_dir(M, St) * TT[idx]).sum(1)) >= cfg.cos_tol
        if ok.sum() < 6:
            return None
        k = min(n_keep, int(ok.sum()))
        keep = np.argpartition(np.where(ok, d, np.inf), k - 1)[:k]
        Mn = fit(S[keep], TP[idx[keep]], cfg, reflect=np.linalg.det(M[:2, :2]) < 0)
        if Mn is None:
            return None
        done = np.abs(Mn - M).max() < 1e-4
        M = Mn
        if done:
            break
    return M


def matched(A, At, treeB, Bt, r, cos_tol, k=6):
    """For each point of A: is a point of B within r with a similar tangent?"""
    d, idx = treeB.query(A, k=k, distance_upper_bound=r)
    valid = np.isfinite(d)
    idx = np.where(valid, idx, 0)
    agree = np.abs(np.einsum("nkd,nd->nk", Bt[idx], At)) >= cos_tol
    return (valid & agree).any(1)


def chance_at(prim, Q, f=1.0):
    """Expected credit a randomly oriented point at Q would earn by luck (raster lookup), at
    the match radius of fine factor f (the nearest precomputed level at or above it)."""
    chance = prim.r_chance[min(l for l in prim.r_chance if l >= f - 1e-9)]
    ij = np.floor((Q - prim.r_lo) / prim.r_h).astype(np.int64)
    H, W = chance.shape
    inb = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
    out = np.zeros(len(Q))
    out[inb] = chance[ij[inb, 1], ij[inb, 0]]
    return out


def fine_factor(sc, cfg):
    """--fine-ref: an icon placed smaller than fine_ref (in primary RMS radii) is held to a
    proportionally tighter match tolerance and shorter minimum runs (never below fine_min of
    the usual), so a small icon has to follow fine detail closely instead of collecting loose
    credit over it. 1 = the usual tolerances."""
    return 1.0 if cfg.fine_ref <= 0 else float(np.clip(sc / cfg.fine_ref, cfg.fine_min, 1.0))


def tol_factor(sc, cfg):
    """Match tolerance as a fraction of --tol: the fine factor, capped by --max-tol so big icons
    must also trace a line closely (minimum runs etc. still follow fine_factor)."""
    f = fine_factor(sc, cfg)
    return min(f, cfg.max_tol / cfg.tol) if cfg.max_tol > 0 else f


# ---------------- non-linear warp ----------------
class Warp:
    """Smooth displacement field applied after the affine placement (primary frame):
    x -> x + sum_k G(x, c_k) w_k with Gaussian G of width s (as in Coherent Point Drift)."""

    def __init__(self, C, W, s):
        self.C, self.W, self.s = C, W, s

    def kernel(self, X):
        d2 = ((X[:, None, :] - self.C[None]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * self.s ** 2))

    def __call__(self, X):
        return X + self.kernel(X) @ self.W

    def jac(self, X):
        """2x2 Jacobian of the warp at each point."""
        G = self.kernel(X)
        dG = -G[..., None] * (X[:, None, :] - self.C[None]) / self.s ** 2  # (N, K, 2)
        return np.eye(2)[None] + np.einsum("kd,nke->nde", self.W, dG)


class WarpChain:
    """Warps applied in sequence (coarse to fine). Each stage is a smooth Warp defined on the
    output of the previous one; the Jacobian is the product of the stage Jacobians."""

    def __init__(self, stages):
        self.stages = stages

    @property
    def W(self):
        return np.concatenate([w.W for w in self.stages]) if self.stages else np.zeros((0, 2))

    def __call__(self, X):
        for w in self.stages:
            X = w(X)
        return X

    def jac(self, X):
        J = np.broadcast_to(np.eye(2), (len(X), 2, 2))
        for w in self.stages:
            J = w.jac(X) @ J
            X = w(X)
        return J


def place(sh, M, warp=None, pts=None, tan=None):
    """Icon points/tangents in the primary frame under affine M and optional warp."""
    pts = sh.pts if pts is None else pts
    tan = sh.tan if tan is None else tan
    Q, Qt = apply(M, pts), apply_dir(M, tan)
    if warp is not None:
        Qt = np.einsum("nde,ne->nd", warp.jac(Q), Qt)
        Qt /= np.linalg.norm(Qt, axis=1, keepdims=True) + 1e-12
        Q = warp(Q)
    return Q, Qt


def warp_ok(warp, M, X, cfg):
    """No folding, no local collapse, no stretch beyond the cap anywhere on the icon."""
    J = warp.jac(X) @ M[:2, :2]
    sv = np.linalg.svd(J, compute_uv=False)
    det0 = np.linalg.det(M[:2, :2])
    return (np.linalg.det(J) / det0 > cfg.warp_min_area).all() and (sv[:, 0] / sv[:, 1] <= cfg.warp_max_aniso).all()


def fit_warp(sh, M, TP, TT, tree, cfg, rng):
    """Non-rigid ICP, coarse to fine: bend the affinely placed icon toward target edges TP/TT.

    Stages use Gaussian widths cfg.warp_scales (relative to icon size), widest first, each
    composed on top of the previous ones, so large smooth bends (a whole limb, the whole
    shape curving) come first and fine local bends after; one fine field asked to move far
    folds instead. Within a stage, each step matches warped icon points to the nearest
    target point with an agreeing tangent within a radius that shrinks from
    warp_reach*icon size (scaled by the stage width) to warp_radius*tol, keeps the closest
    warp_trim, and solves
      min sum ||x0 + G w - t||^2 + lambda * n * tr(W' G_cc W)
    for the stage field, then backs off (halves W) until the whole chain passes warp_ok."""
    n = min(len(sh.pts), 400)
    sel = rng.choice(len(sh.pts), n, replace=False)
    P0, T0 = place(sh, M, pts=sh.pts[sel], tan=sh.tan[sel])
    sc = np.sqrt(abs(np.linalg.det(M[:2, :2])))
    chain = WarpChain([])
    finest = min(cfg.warp_scales)
    for width in cfg.warp_scales:
        X0 = chain(P0)  # this stage's field is defined on the output of the earlier stages
        # control points: farthest-point sampling over the placed icon (fewer for wide stages)
        k = int(np.clip(round(cfg.warp_ctrl * (finest / width) ** 2), 9, min(cfg.warp_ctrl, n)))
        idx = [0]
        d = np.linalg.norm(X0 - X0[0], axis=1)
        for _ in range(k - 1):
            idx.append(int(d.argmax()))
            d = np.minimum(d, np.linalg.norm(X0 - X0[idx[-1]], axis=1))
        C = X0[idx]
        stage = Warp(C, np.zeros((len(C), 2)), width * sc)
        G0, Gcc = stage.kernel(X0), stage.kernel(C)
        reach = cfg.warp_reach * sc * np.sqrt(width / finest)  # wider stage, longer reach
        r_end = cfg.warp_radius * cfg.tol * tol_factor(sc, cfg)
        radii = np.geomspace(max(reach, r_end), r_end, cfg.warp_iters)
        for radius in radii:
            cur = WarpChain(chain.stages + [stage])
            X, Xt = place(sh, M, cur, pts=sh.pts[sel], tan=sh.tan[sel])
            dist, j = tree.query(X, k=8, distance_upper_bound=radius)
            # nearest of the 8 whose tangent agrees
            valid = np.isfinite(dist)
            jj = np.where(valid, j, 0)
            agree = valid & (np.abs(np.einsum("nkd,nd->nk", TT[jj], Xt)) >= cfg.cos_tol)
            first = np.argmax(agree, 1)
            ok = agree[np.arange(len(X)), first]
            j = jj[np.arange(len(X)), first]
            dist = np.where(ok, dist[np.arange(len(X)), first], np.inf)
            if ok.sum() < 10:
                break
            keep = np.flatnonzero(ok)
            keep = keep[np.argsort(dist[keep])[:max(10, int(cfg.warp_trim * len(keep)))]]
            Gm = G0[keep]
            A = Gm.T @ Gm + cfg.warp_lambda * len(keep) * Gcc + 1e-9 * np.eye(len(C))
            Wn = np.linalg.solve(A, Gm.T @ (TP[j[keep]] - X0[keep]))
            for _ in range(8):
                if warp_ok(WarpChain(chain.stages + [Warp(C, Wn, stage.s)]), M, P0, cfg):
                    break
                Wn = 0.5 * (Wn + stage.W)  # back off toward the last valid field
            else:
                break
            stage = Warp(C, Wn, stage.s)
        if np.abs(stage.W).max() > 0:
            chain = WarpChain(chain.stages + [stage])
    return chain if chain.stages else None


def evaluate(sh, M, prim, covered, cfg, warp=None):
    """score = new - beta*overlap - pen, with pen = alpha*stray + gamma*chance.

    new/overlap are weighted primary points (outline 1, interior --interior-weight).
    stray: icon outline that lands on nothing. chance: credit the icon's points would
    collect by luck where they land, so busy textures over dense detail earn ~0 net.
    Both are in primary-point units and fixed per placement (keeps the ILP linear)."""
    sc = np.sqrt(abs(np.linalg.det(M[:2, :2])))
    if cfg.max_area and sh.area * sc * sc > cfg.max_area * prim.area:  # too big a footprint
        return -np.inf, np.zeros(len(prim.pts), bool), 0.0, 0.0
    if cfg.max_upscale and sc * sh.px / prim.px > cfg.max_upscale:  # blown up past its resolution
        return -np.inf, np.zeros(len(prim.pts), bool), 0.0, 0.0
    if cfg.max_scale and sc > cfg.max_scale:  # larger than the current size cap
        return -np.inf, np.zeros(len(prim.pts), bool), 0.0, 0.0
    Q, Qt = place(sh, M, warp)
    f, ft = fine_factor(sc, cfg), tol_factor(sc, cfg)
    # scaled-up pieces have sparser points, so widen the radius slightly
    mask = matched(prim.pts, prim.tan, cKDTree(Q), Qt, cfg.tol * ft + 0.5 * sh.spacing * sc, cfg.cos_tol)
    mask = keep_runs(prim, mask, cfg, cfg.min_run * f)
    on = matched(Q, Qt, prim.tree, prim.tan, cfg.tol * ft + 0.5 * prim.spacing, cfg.cos_tol)
    on = keep_runs(sh, on, cfg, cfg.icon_min_run * f, sc)
    stray = float((~on).sum() * sc)
    pen = cfg.alpha * stray + cfg.gamma * float(chance_at(prim, Q, ft).sum() * sc)
    new, overlap = float(prim.w[mask & ~covered].sum()), float(prim.w[mask & covered].sum())
    return new - cfg.beta * overlap - pen, mask, stray, pen


def mkey(name, M):
    return (name,) + tuple(np.round(M[:2].ravel() * 10).astype(int))


def build_raster(prim, cfg, chance=True):
    """Distance map of the primary + index of the nearest primary point per cell, and (with
    chance) the chance maps evaluate() charges for, one per fine-factor level."""
    from scipy.ndimage import distance_transform_edt
    h = prim.spacing / 2
    lo = prim.pts.min(0) - 1.0
    nx, ny = (np.ceil((prim.pts.max(0) + 1.0 - lo) / h).astype(int) + 1)
    grid = np.full((ny, nx), -1, np.int64)
    ij = np.floor((prim.pts - lo) / h).astype(int)
    grid[ij[:, 1], ij[:, 0]] = np.arange(len(prim.pts))
    dist, (iy, ix) = distance_transform_edt(grid < 0, return_indices=True)
    prim.r_lo, prim.r_h, prim.r_dist, prim.r_idx = lo, h, dist * h, grid[iy, ix]
    # chance map: for a point at this cell with a uniformly random tangent, the expected
    # weighted credit from primary points within the match radius (0 far from any edge)
    prim.r_chance = {}
    for level in (cfg.chance_levels if chance else ()):
        r = cfg.tol * level + 0.5 * prim.spacing
        cy, cx = np.nonzero(prim.r_dist < r)
        C = lo + (np.column_stack([cx, cy]) + 0.5) * h
        d, idx = prim.tree.query(C, k=12, distance_upper_bound=r)
        valid = np.isfinite(d)
        idx = np.where(valid, idx, 0)
        Tn, Wn = prim.tan[idx], np.where(valid, prim.w[idx], 0.0)
        acc = np.zeros(len(C))
        angles = np.linspace(0, np.pi, 18, endpoint=False)
        for a in angles:
            agree = np.abs(Tn[..., 0] * np.cos(a) + Tn[..., 1] * np.sin(a)) >= cfg.cos_tol
            acc += np.where(agree, Wn, 0.0).max(1)
        prim.r_chance[level] = np.zeros(prim.r_dist.shape)
        prim.r_chance[level][cy, cx] = acc / len(angles)
    build_chains(prim, cfg)


def build_chains(prim, cfg):
    """Adjacency along a shape's lines: neighbouring points whose tangents agree, within 2
    spacings (or 1.6x the typical neighbour distance, for icons upscaled from few pixels).
    Connected points along this graph form the runs that keep_runs() credits."""
    from scipy.sparse import coo_matrix
    nn = np.median(prim.tree.query(prim.pts, k=2)[0][:, 1]) if len(prim.pts) > 1 else 0.0
    i, j = prim.tree.query_pairs(max(2 * prim.spacing, 1.6 * nn), output_type="ndarray").T
    ok = np.abs((prim.tan[i] * prim.tan[j]).sum(1)) >= cfg.cos_tol
    i, j, n = i[ok], j[ok], len(prim.pts)
    prim.chain = coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()


def keep_runs(shape, mask, cfg, min_run=None, scale=1.0):
    """Keep only points of `mask` that belong to a connected run of masked points at least
    min_run long along the shape's own lines (default --min-run; scale converts the shape's
    point spacing to primary units). Primary side: scattered hits on texture earn nothing.
    Icon side: an icon point is on a line only if a stretch of the icon's line follows it."""
    from scipy.sparse.csgraph import connected_components
    min_run = cfg.min_run if min_run is None else min_run
    idx = np.flatnonzero(mask)
    if min_run <= 0 or len(idx) == 0:
        return mask
    _, lab = connected_components(shape.chain[idx][:, idx], directed=False)
    size = np.bincount(lab)
    out = np.zeros_like(mask)
    out[idx[size[lab] * shape.spacing * scale >= min_run]] = True
    return out


def global_search(sh, prim, cfg, rng):
    """Score ~1M affine hypotheses by distance-map lookup; return the best distinct ones.

    With --fine-ref, small scales (--small-scales) are searched too, centred on the target's
    own points (a translation grid fine enough for small icons over the whole primary would
    be far too many), with their own --small-hyps quota, since their tighter match radius
    makes their raw counts lower than large placements'."""
    sel = np.arange(min(cfg.n_glob, len(sh.cpts)))  # evenly spread prefix of the FPS order
    mu = sh.cpts[sel].mean(0)
    S, St = sh.cpts[sel] - mu, sh.ctan[sel]
    c0, half = prim.pts.mean(0), (prim.pts.max(0) - prim.pts.min(0)) / 2
    g = np.linspace(-1, 1, cfg.t_grid)
    T = np.stack(np.meshgrid(c0[0] + g * half[0], c0[1] + g * half[1]), -1).reshape(-1, 2)
    cap = (lambda xs: tuple(x for x in xs if x <= 1.25 * cfg.max_scale)) if cfg.max_scale else tuple
    hyps = _search(S, St, mu, prim, cfg, cap(cfg.init_scales), T, cfg.n_hyp) if cap(cfg.init_scales) else []
    if cfg.fine_ref > 0 and cap(cfg.small_scales) and len(prim.pts) >= 6:
        Tc = prim.pts[fps_order(prim.pts, cfg.small_centers)]
        hyps += _search(S, St, mu, prim, cfg, cap(cfg.small_scales), Tc, cfg.small_hyps)
    return hyps


def _search(S, St, mu, prim, cfg, scales, T, n_hyp):
    """global_search over the given scales and translations T: count icon points landing
    near a target line with an agreeing tangent (within coarse_r, tightened by the fine
    factor of the scale), keep the best n_hyp after non-maximum suppression."""
    H, W = prim.r_dist.shape
    linears, fs = [], []
    for mirror in (1.0, -1.0):
        for th in np.linspace(0, 2 * np.pi, cfg.n_rot, endpoint=False):
            for s in scales:
                for a in [x for x in cfg.stretches if x <= cfg.max_aniso]:
                    for phi in ([0.0] if a == 1 else np.linspace(0, np.pi, 6, endpoint=False)):
                        stretch = rot(phi) @ np.diag([np.sqrt(a), 1 / np.sqrt(a)]) @ rot(-phi)
                        linears.append(s * rot(th) @ stretch @ np.diag([mirror, 1.0]))
                        fs.append(fine_factor(s, cfg))
    scores = np.empty((len(linears), len(T)), np.int32)
    for li, L in enumerate(linears):
        X = (S @ L.T)[None] + T[:, None]
        Lt = St @ L.T
        Lt /= np.linalg.norm(Lt, axis=1, keepdims=True) + 1e-12
        ij = np.floor((X - prim.r_lo) / prim.r_h).astype(np.int64)
        inb = (ij[..., 0] >= 0) & (ij[..., 0] < W) & (ij[..., 1] >= 0) & (ij[..., 1] < H)
        ix, iy = np.clip(ij[..., 0], 0, W - 1), np.clip(ij[..., 1], 0, H - 1)
        agree = np.abs((prim.tan[prim.r_idx[iy, ix]] * Lt[None]).sum(-1)) >= cfg.cos_tol
        scores[li] = (inb & (prim.r_dist[iy, ix] < cfg.coarse_r * fs[li]) & agree).sum(1)
    n_top = min(n_hyp * 20, scores.size)
    top = np.argpartition(scores, -n_top, axis=None)[-n_top:]
    top = top[np.argsort(scores.ravel()[top])[::-1]]
    hyps, seen = [], []
    for f in top:  # non-maximum suppression: skip near-duplicate hypotheses
        li, ti = divmod(int(f), len(T))
        L = linears[li]
        X = S @ L.T + T[ti]
        if all(np.linalg.norm(X - Y, axis=1).mean() > 0.15 * fs[li] for Y in seen):
            M = np.eye(3)
            M[:2, :2], M[:2, 2] = L, T[ti] - L @ mu
            hyps.append(M)
            seen.append(X)
            if len(hyps) >= n_hyp:
                break
    return hyps


def initial_placements(name, sh, prim, cfg, rng):
    """Global hypotheses -> ICP refinement -> keep a few diverse good placements."""
    Ms = {}
    for M0 in global_search(sh, prim, cfg, rng):
        M = icp(sh.cpts, sh.ctan, prim.pts, prim.tan, prim.tree, M0, cfg, cfg.iters)
        if M is not None:
            Ms.setdefault(mkey(name, M), M)
    none = np.zeros(len(prim.pts), bool)
    scored = sorted(((evaluate(sh, M, prim, none, cfg)[:2], M) for M in Ms.values()),
                    key=lambda x: -x[0][0])
    keep = []
    for (s, mask), M in scored:
        if s <= 0 or len(keep) >= cfg.keep_top:
            break
        if all((mask & m).sum() / max((mask | m).sum(), 1) < 0.8 for m, _ in keep):
            keep.append((mask, M))
    return [M for _, M in keep]


def sub_prim(prim, keep, cfg):
    """The primary restricted to points `keep` (e.g. the uncovered ones), with its own raster,
    so the global search and ICP look only at what is left to cover."""
    from types import SimpleNamespace
    r = SimpleNamespace(pts=prim.pts[keep], tan=prim.tan[keep], w=prim.w[keep], bnd=prim.bnd[keep],
                        spacing=prim.spacing)
    r.tree = cKDTree(r.pts)
    build_raster(r, cfg, chance=False)  # evaluate() charges chance on the full primary
    return r


def best_for_remainder(name, sh, prim, rem, covered, cfg, rng, keep=0):
    """Best placement of one icon against the remainder `rem`: global search + ICP onto the
    remainder, scored by evaluate() on the full primary (stray, chance, overlap with
    `covered`), then warps of the best --warp-cands distinct affine placements (a placement
    that fits poorly as-is may fit best once bent). Returns a candidate dict or None; with
    keep=K its "keep" entry holds up to K distinct affine placements for refine_placements."""
    return refine_placements(name, sh, global_search(sh, rem, cfg, rng), prim, rem, covered, cfg, rng, keep)


def refine_placements(name, sh, M0s, prim, rem, covered, cfg, rng, keep=0):
    """best_for_remainder from given starting placements instead of a global search: ICP onto
    `rem`, evaluate, warp the best distinct ones. Re-fitting an icon's kept placements to a
    later round's remainder costs ~1 s against ~30 s for a fresh search."""
    Ms = {}
    for M0 in M0s:
        M = icp(sh.cpts, sh.ctan, rem.pts, rem.tan, rem.tree, M0, cfg, cfg.iters)
        if M is not None:
            Ms.setdefault(mkey(name, M), M)
    scored = sorted((dict(image=name, M=M, warp=None, **dict(zip(("score", "mask", "stray", "pen"),
                                                                    evaluate(sh, M, prim, covered, cfg))))
                     for M in Ms.values()), key=lambda c: -c["score"])
    if not scored:
        return None
    best = scored[0]
    seeds = []
    for c in scored:  # distinct placements: masks overlapping < 80%
        if len(seeds) >= max(cfg.warp_cands if cfg.warp else 0, keep) or not np.isfinite(c["score"]):
            break
        if all((c["mask"] & d["mask"]).sum() / max((c["mask"] | d["mask"]).sum(), 1) < 0.8 for d in seeds):
            seeds.append(c)
    if cfg.warp:
        for c in seeds[:cfg.warp_cands]:
            warp = fit_warp(sh, c["M"], rem.pts, rem.tan, rem.tree, cfg, rng)
            if warp is not None:
                s, mask, stray, pen = evaluate(sh, c["M"], prim, covered, cfg, warp)
                if s > best["score"]:
                    best = dict(image=name, M=c["M"], warp=warp, mask=mask, score=s, stray=stray, pen=pen)
    if keep:
        best["keep"] = [c["M"] for c in seeds[:keep]]
    return best


# ---------------- selection ----------------
def wcov(prim, covered):
    """Weighted coverage: outline points count 1, interior points --interior-weight."""
    return float(prim.w[covered].sum() / prim.w.sum())


def residual_greedy(shapes, init, prim, cfg):
    n = len(prim.pts)
    covered = np.zeros(n, bool)
    chosen, used, pool = [], set(), {}
    while len(chosen) < cfg.max_images and wcov(prim, covered) < cfg.target:
        unc = ~covered
        if unc.sum() < 6:
            break
        UP, UT = prim.pts[unc], prim.tan[unc]
        utree, cands = cKDTree(UP), []

        def consider(name, M, warp=None):
            s, mask, stray, pen = evaluate(shapes[name], M, prim, covered, cfg, warp)
            if cfg.exact:
                key = mkey(name, M) + (("warp", round(float(np.abs(warp.W).sum()), 4)) if warp else ())
                pool.setdefault(key, dict(image=name, M=M, warp=warp, idx=np.flatnonzero(mask),
                                          stray=stray, pen=pen))
            cands.append(dict(image=name, M=M, warp=warp, mask=mask, score=s, stray=stray, pen=pen))

        for name, sh in shapes.items():
            if name in used:
                continue
            for M0 in init[name]:
                # re-align toward what is still uncovered
                Mr = icp(sh.cpts, sh.ctan, UP, UT, utree, M0, cfg, cfg.refine_iters)
                for M in (M0, Mr):
                    if M is not None:
                        consider(name, M)
        if cfg.warp and cands:
            # bend the best few affine candidates (one per image) toward the uncovered edges
            seen, top = set(), []
            for c in sorted(cands, key=lambda c: -c["score"]):
                if c["image"] not in seen:
                    seen.add(c["image"])
                    top.append(c)
                if len(top) >= cfg.warp_top:
                    break
            wrng = np.random.default_rng([cfg.seed, len(chosen)])
            for c in top:
                warp = fit_warp(shapes[c["image"]], c["M"], UP, UT, utree, cfg, wrng)
                if warp is not None:
                    consider(c["image"], c["M"], warp)
        best = max(cands, key=lambda c: c["score"]) if cands else None
        if best is None or best["score"] < cfg.min_gain * prim.w.sum():
            break
        best["gain"] = float(prim.w[best["mask"] & ~covered].sum())
        covered |= best["mask"]
        best["coverage"] = wcov(prim, covered)
        used.add(best["image"])
        chosen.append(best)
        if cfg.verbose:
            print(f"  round {len(chosen)}: {os.path.basename(best['image'])} -> {best['coverage']:.1%}"
                  + (" (warped)" if best["warp"] is not None else ""))
    return chosen, covered, list(pool.values())


def exact_select(pool, prim, cfg):
    """max  sum(w*y) - beta*overlap - pen  s.t. y_i <= #placements covering i,
    <= 1 placement per image, <= max_images placements.  Linear, so solvable exactly."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix, hstack, identity
    n, J = len(prim.pts), len(pool)
    if J == 0:
        return None
    rows = np.concatenate([p["idx"] for p in pool])
    cols = np.concatenate([np.full(len(p["idx"]), j) for j, p in enumerate(pool)])
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, J))
    w = prim.w
    cov = np.asarray(A.T @ w).ravel()  # weighted points each placement covers
    pen = np.array([p["pen"] for p in pool])
    names = sorted({p["image"] for p in pool})
    g = np.array([names.index(p["image"]) for p in pool])
    G = csr_matrix((np.ones(J), (g, np.arange(J))), shape=(len(names), J))
    cons = [LinearConstraint(hstack([-A, identity(n)]), -np.inf, 0),
            LinearConstraint(hstack([G, csr_matrix((len(names), n))]), -np.inf, 1),
            LinearConstraint(hstack([csr_matrix(np.ones((1, J))), csr_matrix((1, n))]), -np.inf, cfg.max_images)]
    c = np.concatenate([cfg.beta * cov + pen, -(1 + cfg.beta) * w])
    res = milp(c, constraints=cons, integrality=np.r_[np.ones(J), np.zeros(n)],
               bounds=Bounds(0, 1), options={"time_limit": cfg.time_limit})
    if res.x is None:
        return None
    remaining = [pool[j] for j in np.flatnonzero(res.x[:J] > 0.5)]
    covered, chosen = np.zeros(n, bool), []
    while remaining:  # list in order of contribution
        p = remaining.pop(int(np.argmax([w[q["idx"]][~covered[q["idx"]]].sum() for q in remaining])))
        mask = np.zeros(n, bool)
        mask[p["idx"]] = True
        gain = float(w[mask & ~covered].sum())
        covered |= mask
        chosen.append(dict(image=p["image"], M=p["M"], warp=p["warp"], mask=mask, stray=p["stray"], pen=p["pen"],
                           gain=gain, coverage=wcov(prim, covered)))
    return chosen, covered, res.message


# ---------------- driver ----------------
def objective(placements, w, cfg):
    """Total score of a set of (mask, pen) placements; equals the sum of greedy scores."""
    if not placements:
        return 0.0, 0.0
    union = np.logical_or.reduce([m for m, _ in placements])
    total = sum(float(w[m].sum()) for m, _ in placements)
    u = float(w[union].sum())
    return u - cfg.beta * (total - u) - sum(p for _, p in placements), u / float(w.sum())


def placement_row(c, sh, Pinv, hw=None):
    """A placement as render_overlay draws it: source px -> primary px matrix, plus for warped
    icons a mesh over the icon's whole canvas (so soft or thin parts away from the edge points
    aren't clipped), rendered piecewise-affine."""
    row = dict(matrix_3x3=(Pinv @ c["M"] @ sh.T).tolist())
    if c.get("warp") is not None:
        import cv2
        ih, iw = hw or cv2.imread(c["image"], cv2.IMREAD_UNCHANGED).shape[:2]
        px, py = 0.04 * iw, 0.04 * ih  # a little past the canvas, so renders never trim its edge
        gx, gy = np.meshgrid(np.linspace(-px, iw + px, 33), np.linspace(-py, ih + py, 33))
        src = np.column_stack([gx.ravel(), gy.ravel()])
        dst = apply(Pinv, c["warp"](apply(c["M"] @ sh.T, src)))
        row.update(mesh_src=src.tolist(), mesh_dst=dst.tolist())
    return row


def report(chosen, covered, prim, shapes, out, method):
    out_pts = prim.bnd
    print(f"\n{method}: {wcov(prim, covered):.1%} weighted coverage with {len(chosen)} images "
          f"(outline {covered[out_pts].mean():.1%}"
          + (f", interior {covered[~out_pts].mean():.1%})" if (~out_pts).any() else ")"))
    Pinv, rows = np.linalg.inv(prim.T), []
    for c in chosen:
        Mpix = Pinv @ c["M"] @ shapes[c["image"]].T  # source pixels -> primary pixels
        sv = np.linalg.svd(c["M"][:2, :2], compute_uv=False)
        print(f"  {os.path.basename(c['image'])}: +{c['gain']:.0f} -> {c['coverage']:.1%}, "
              f"stretch {sv[0] / sv[1]:.1f}:1, stray {c['stray']:.0f}, penalty {c['pen']:.0f}")
        row = dict(image=c["image"], gain=c["gain"], cumulative_coverage=float(c["coverage"]),
                   stretch=float(sv[0] / sv[1]), **placement_row(c, shapes[c["image"]], Pinv))
        if c.get("warp") is not None:
            dst, src = np.array(row["mesh_dst"]), np.array(row["mesh_src"])
            print(f"    warped: mean bend {np.abs(dst - apply(Mpix, src)).mean():.1f}px")
        rows.append(row)
    base = os.path.splitext(out)[0]
    with open(base + ".json", "w") as f:
        json.dump(rows, f, indent=2)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    Pp = apply(Pinv, prim.pts)
    ax.scatter(*Pp.T, s=4, c="k", label="primary")
    ax.scatter(*Pp[~covered].T, s=12, c="red", label="uncovered", zorder=5)
    for c, col in zip(chosen, plt.cm.tab10.colors):
        Q = apply(Pinv, place(shapes[c["image"]], c["M"], c.get("warp"))[0])
        ax.scatter(*Q.T, s=2, color=col, label=os.path.basename(c["image"]))
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(fontsize=7, markerscale=3)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out} and {base}.json")


_PRIM = _CFG = None


def _align_one(args):
    """Worker: load one library image and run its global search (independent per image)."""
    i, p = args
    rng = np.random.default_rng([_CFG.seed, i])
    try:
        sh = Shape(p, _CFG, rng, icon=True)
    except Exception as e:
        return p, None, None, str(e)
    return p, sh, initial_placements(p, sh, _PRIM, _CFG, rng), None


def solve(primary, lib_paths, cfg, out):
    global _PRIM, _CFG
    rng = np.random.default_rng(cfg.seed)
    prim = Shape(primary, cfg, rng)
    build_raster(prim, cfg)
    _PRIM, _CFG = prim, cfg  # inherited by forked workers
    jobs = max(1, min(cfg.jobs, len(lib_paths)))
    shapes, init = {}, {}
    work = list(enumerate(lib_paths))
    if jobs > 1:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(jobs)
        results = pool.imap(_align_one, work, chunksize=1)
    else:
        results = map(_align_one, work)
    for n, (p, sh, placements, err) in enumerate(results, 1):
        if sh is None:
            print(f"skip {p}: {err}")
            continue
        shapes[p], init[p] = sh, placements
        if cfg.verbose:
            print(f"[{n}/{len(lib_paths)}] {os.path.basename(p)}: {len(placements)} placements", flush=True)
    if jobs > 1:
        pool.close()
        pool.join()
    chosen, covered, pool_ = residual_greedy(shapes, init, prim, cfg)
    method = "Greedy (residual-driven)"
    obj, _ = objective([(c["mask"], c["pen"]) for c in chosen], prim.w, cfg)
    if cfg.exact:
        ex = exact_select(pool_, prim, cfg)
        if ex is not None:
            ex_obj, _ = objective([(c["mask"], c["pen"]) for c in ex[0]], prim.w, cfg)
            if ex_obj >= obj:  # a timed-out ILP can return something worse (even empty)
                chosen, covered, msg = ex
                obj = ex_obj
                method = f"Exact ILP over {len(pool_)} placements [{msg}]"
            else:
                print(f"Exact ILP ({ex[2]}) scored {ex_obj:.1f} < greedy {obj:.1f}; keeping greedy")
    report(chosen, covered, prim, shapes, out, method)
    print(f"Objective: {obj:.1f}")
    return dict(chosen=chosen, covered=covered, prim=prim, shapes=shapes, init=init, objective=obj)


def selftest(cfg):
    """Cut a known primary into 4 pieces, close and warp each, add 8 distractors.

    Each true piece carries an artificial closing curve that lands on nothing, so the
    test is scored with the stray weight it was designed for (alpha 0.15); at a heavy
    stray penalty the true arrangement itself scores negative."""
    cfg.alpha = 0.15
    import tempfile
    rng = np.random.default_rng(7)
    d = tempfile.mkdtemp(prefix="cover_selftest_")
    lib = os.path.join(d, "lib")
    os.makedirs(lib)
    th = np.linspace(0, 2 * np.pi, 800, endpoint=False)
    r = 1 + 0.3 * np.cos(3 * th) + 0.15 * np.sin(5 * th)
    P = 200 * np.column_stack([r * np.cos(th), r * np.sin(th)]) + 300
    np.save(os.path.join(d, "primary.npy"), P)
    c, n = P.mean(0), len(P)

    def warp(pts, max_stretch, mirror=False):
        a, s = rng.uniform(1, max_stretch), rng.uniform(0.6, 1.4)
        A = rot(rng.uniform(0, 6.28)) @ np.diag([s * a, s / a]) @ rot(rng.uniform(0, 6.28))
        if mirror:
            A = A @ np.diag([-1.0, 1.0])
        W = np.eye(3)
        W[:2, :2], W[:2, 2] = A, rng.uniform(-100, 100, 2)
        return apply(W, pts), W

    truth = {}
    for k in range(4):  # quarter of the primary + an inner closing curve
        arc = P[np.arange(int((k / 4 - 0.03) * n), int(((k + 1) / 4 + 0.03) * n)) % n]
        ctrl = c + 0.2 * (arc[len(arc) // 2] - c)
        t = np.linspace(0, 1, 60)[1:-1, None]
        back = (1 - t) ** 2 * arc[-1] + 2 * (1 - t) * t * ctrl + t ** 2 * arc[0]
        name = f"piece{k}.npy"
        pts, W = warp(np.vstack([arc, back]), 1.8, mirror=(k == 1))
        np.save(os.path.join(lib, name), pts)
        truth[os.path.join(lib, name)] = np.linalg.inv(W)  # piece pixels -> primary pixels
    u = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    for k in range(8):  # ellipses, stars, squircles, random blobs
        rr = [np.ones_like(u),
              1 + 0.35 * np.cos(rng.integers(4, 8) * u),
              (np.abs(np.cos(u)) ** 4 + np.abs(np.sin(u)) ** 4) ** -0.25,
              1 + sum(rng.uniform(-.2, .2) * np.cos(h * u + rng.uniform(0, 6)) for h in range(2, 6))][k % 4]
        shp = rng.uniform(40, 160) * np.column_stack([rr * np.cos(u), rr * np.sin(u)])
        np.save(os.path.join(lib, f"distractor{k}.npy"), warp(shp, 2.5)[0])
    t0 = time.time()
    res = solve(os.path.join(d, "primary.npy"), sorted(glob.glob(os.path.join(lib, "*.npy"))),
                cfg, os.path.join(d, "selftest.png"))
    prim, shapes, n = res["prim"], res["shapes"], len(res["prim"].pts)
    none = np.zeros(n, bool)
    recall, gt = 0, []
    for path, Winv in truth.items():
        sh = shapes[path]
        Mt = prim.T @ Winv @ np.linalg.inv(sh.T)  # true transform in normalized coords
        _, m, _, p = evaluate(sh, Mt, prim, none, cfg)
        gt.append((m, p))
        # affine lets an arc slide along a curve with equal fit, so judge placements by
        # score, not by distance to the true transform
        true_score = evaluate(sh, Mt, prim, none, cfg)[0]
        found = max((evaluate(sh, M, prim, none, cfg)[0] for M in res["init"][path]), default=-np.inf)
        recall += found >= 0.9 * true_score
    gt_obj, gt_cov = objective(gt, prim.w, cfg)
    ok = recall >= 3 and res["objective"] >= 0.95 * gt_obj
    print(f"\nSelftest ({time.time() - t0:.0f}s, files in {d})")
    print(f"  alignment recall: {recall}/4 true pieces placed at >=90% of their true-placement score")
    print(f"  ground truth: objective {gt_obj:.1f}, coverage {gt_cov:.1%}")
    print(f"  program:      objective {res['objective']:.1f}, coverage {res['covered'].mean():.1%}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def parse_cfg(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("primary", nargs="?")
    ap.add_argument("library", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=["affine", "similarity"], default="affine")
    ap.add_argument("--max-aniso", type=float, default=4.0, help="max stretch ratio (anti-flattening guard)")
    ap.add_argument("--tol", type=float, default=0.04, help="coverage distance, in primary RMS radii")
    ap.add_argument("--angle-tol", type=float, default=30, help="max tangent disagreement, degrees")
    ap.add_argument("--alpha", type=float, default=1.0, help="penalty per unit of stray outline")
    ap.add_argument("--beta", type=float, default=0.3, help="penalty per unit of re-covered outline")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="weight of the chance correction (credit an icon would get by luck)")
    ap.add_argument("--icon-edges", choices=["outline", "simplified", "detail", "full"], default="outline",
                    help="which icon edges are matched against the primary (the primary uses --prim-edges)")
    ap.add_argument("--prim-edges", choices=["full", "centerline"], default="full",
                    help="primary edges: full = Canny (both sides of every stroke), centerline = one "
                         "line per drawn stroke")
    ap.add_argument("--icon-merge", type=float, default=22,
                    help="merge icon colour regions closer than this (higher = less detail)")
    ap.add_argument("--icon-min-region", type=float, default=0.015,
                    help="icon regions smaller than this fraction of the icon are absorbed")
    ap.add_argument("--icon-colors", type=int, default=8, help="colour clusters per icon")
    ap.add_argument("--warp", action="store_true", help="non-linear refinement of top candidates")
    ap.add_argument("--warp-lambda", type=float, default=0.005, help="bend penalty (higher = stiffer)")
    ap.add_argument("--warp-width", type=float, default=0.4,
                    help="control-point influence width, relative to the icon's size (finest stage)")
    ap.add_argument("--warp-coarse", type=str, default="",
                    help="wider stages solved first, comma-separated widths relative to icon size, "
                         "e.g. 1.0,0.5 (large smooth bends before fine ones; empty = single stage)")
    ap.add_argument("--warp-ctrl", type=int, default=25, help="number of warp control points")
    ap.add_argument("--warp-cands", type=int, default=1,
                    help="distinct affine placements per icon that get warped (best after warping wins)")
    ap.add_argument("--warp-top", type=int, default=8, help="candidates warped per round")
    ap.add_argument("--warp-reach", type=float, default=0.3,
                    help="how far warped parts may search for a line at first, relative to icon size")
    ap.add_argument("--warp-min-area", type=float, default=0.3,
                    help="smallest local area ratio a warp may shrink to (anti-collapse guard)")
    ap.add_argument("--warp-max-aniso", type=float, default=6.0, help="max local stretch after warping")
    ap.add_argument("--interior-weight", type=float, default=1.0,
                    help="credit for interior edge points relative to outline points")
    ap.add_argument("--min-run", type=float, default=0.25,
                    help="covered points count only in connected runs at least this long along the "
                         "primary's lines, in primary RMS radii (0 = off)")
    ap.add_argument("--icon-min-run", type=float, default=0.0,
                    help="icon points count as on a line only in connected runs of the icon's lines "
                         "at least this long once placed, in primary RMS radii (0 = off)")
    ap.add_argument("--max-area", type=float, default=0.0,
                    help="largest footprint of one placed icon, as a fraction of the primary's "
                         "silhouette area (0 = no cap); keeps a few icons from covering everything")
    ap.add_argument("--max-upscale", type=float, default=0.0,
                    help="largest enlargement of an icon over its source pixels (0 = no cap); thin-line "
                         "images blown up 15-50x blur into a grey haze over the whole canvas")
    ap.add_argument("--prim-spacing", type=float, default=0.0,
                    help="primary point spacing in primary RMS radii (0 = --tol/2); finer resolves "
                         "detail finer than the tolerance, e.g. fingerprint lines")
    ap.add_argument("--fine-ref", type=float, default=0.0,
                    help="icons placed smaller than this (primary RMS radii) get proportionally "
                         "tighter tolerance and minimum runs, and --small-scales are searched (0 = off)")
    ap.add_argument("--fine-min", type=float, default=0.25, help="smallest fine factor")
    ap.add_argument("--max-scale", type=float, default=0.0,
                    help="largest placement, as RMS radius in primary RMS radii (0 = no cap)")
    ap.add_argument("--max-tol", type=float, default=0.0,
                    help="cap on the match tolerance (primary RMS radii) for icons of any size, so big "
                         "icons must trace lines as closely as this (0 = off: --tol, shrunk by --fine-ref)")
    ap.add_argument("--small-scales", type=str, default="0.08,0.13,0.2",
                    help="extra global-search scales with --fine-ref, comma-separated")
    ap.add_argument("--small-hyps", type=int, default=20, help="hypotheses kept from the small scales")
    ap.add_argument("--small-centers", type=int, default=256,
                    help="target points the small scales are centred on")
    ap.add_argument("--trim", type=float, default=0.5, help="fraction of icon points ICP fits per step")
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--min-gain", type=float, default=0.01)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--exact", action="store_true", help="final pick by integer programming")
    ap.add_argument("--time-limit", type=float, default=60, help="ILP time limit, seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker processes for the per-image search")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default="result.png")
    cfg = ap.parse_args(argv)
    cfg.spacing, cfg.cos_tol = cfg.tol / 2, np.cos(np.radians(cfg.angle_tol))
    cfg.scale_range, cfg.coarse_r = (0.15, 3.0), 2.5 * cfg.tol
    cfg.n_coarse, cfg.n_rot, cfg.t_grid, cfg.n_hyp = 200, 24, 21, 40
    cfg.n_glob = 128
    cfg.init_scales, cfg.stretches = (0.3, 0.45, 0.65, 0.95), (1.0, 1.6, 2.5)
    cfg.iters, cfg.refine_iters, cfg.keep_top = 20, 15, 6
    cfg.warp_iters, cfg.warp_radius, cfg.warp_trim = 20, 3.0, 0.9
    cfg.warp_scales = sorted({float(x) for x in cfg.warp_coarse.split(",") if x.strip()}
                             | {cfg.warp_width}, reverse=True)
    cfg.prim_spacing = cfg.prim_spacing or cfg.spacing
    cfg.small_scales = tuple(float(x) for x in cfg.small_scales.split(",") if x.strip())
    cfg.chance_levels = (1.0,)
    if cfg.fine_ref > 0:
        cfg.scale_range = (min(cfg.small_scales + (0.15,)) * 0.6, cfg.scale_range[1])
        cfg.chance_levels = tuple(sorted({1.0, 0.75, 0.5, 0.35, cfg.fine_min}, reverse=True))
    if cfg.max_tol > 0:
        cfg.chance_levels = tuple(sorted(set(cfg.chance_levels) | {min(1.0, cfg.max_tol / cfg.tol)}, reverse=True))
    cfg.verbose = not cfg.quiet
    return cfg, ap


def main():
    cfg, ap = parse_cfg()
    if cfg.selftest:
        raise SystemExit(0 if selftest(cfg) else 1)
    if not (cfg.primary and cfg.library):
        ap.error("give primary and library, or --selftest")
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".npy")
    paths = sorted(p for p in glob.glob(os.path.join(cfg.library, "*")) if p.lower().endswith(exts))
    solve(cfg.primary, paths, cfg, cfg.out)


if __name__ == "__main__":
    main()
