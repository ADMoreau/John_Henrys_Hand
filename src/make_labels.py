#!/usr/bin/env python3
"""Training labels for the next-icon scorer: the aligner's true gain for (icon, remainder) pairs.

  python src/make_labels.py LIBRARY_DIR --out runs/labels --states 125 --icons 24 [cover_align flags]

A "state" is a primary plus a covered mask (what earlier rounds already took). Primaries are
hand_cutout.png (--hand-frac of states) and random library images with full edges, so the
scorer sees more than one primary. Covered masks are unions of 0-3 random disks. For each
state, --icons random library icons are aligned to the uncovered remainder with
best_for_remainder() and the score (cover_align's objective for that placement) is stored.

One .npz per state, written as soon as it finishes; rerunning skips finished states.
"""
import argparse, glob, os, sys, time
import numpy as np
import cover_align as ca

_CFG = _ARGS = _LIB = None


def make_state(i):
    """Deterministic primary path + covered mask for state i."""
    rng = np.random.default_rng([_ARGS.seed, i])
    path = _ARGS.hand if rng.random() < _ARGS.hand_frac else _LIB[rng.integers(len(_LIB))]
    prim = ca.Shape(path, _CFG, rng)
    ca.build_raster(prim, _CFG)
    covered = np.zeros(len(prim.pts), bool)
    if rng.random() > 0.25:  # 25% of states are round 1 (nothing covered)
        for _ in range(rng.integers(1, 4)):
            c = prim.pts[rng.integers(len(prim.pts))]
            covered |= np.linalg.norm(prim.pts - c, axis=1) < rng.uniform(0.25, 0.8)
    if (~covered).mean() < 0.2:  # keep at least a fifth of the primary to cover
        covered[:] = False
    return path, prim, covered, rng


def label_state(i):
    out = os.path.join(_ARGS.out, f"state_{i:05d}.npz")
    if os.path.exists(out):
        return i, "exists", 0.0
    t0 = time.time()
    try:
        path, prim, covered, rng = make_state(i)
    except Exception as e:
        return i, f"primary failed: {e}", 0.0
    rem = ca.sub_prim(prim, ~covered, _CFG)
    icons, scores, gains, ipts, itan, n_icon = [], [], [], [], [], []
    while len(icons) < _ARGS.icons:
        p = _LIB[rng.integers(len(_LIB))]
        if p == path or p in icons:
            continue
        try:
            sh = ca.Shape(p, _CFG, rng, icon=True)
        except Exception:
            continue
        best = ca.best_for_remainder(p, sh, prim, rem, covered, _CFG, rng)
        icons.append(p)
        scores.append(best["score"] if best else 0.0)
        gains.append(float(prim.w[best["mask"] & ~covered].sum()) if best else 0.0)
        ipts.append(sh.pts.astype(np.float32))
        itan.append(sh.tan.astype(np.float32))
        n_icon.append(len(sh.pts))
    np.savez_compressed(
        out + ".tmp.npz", primary=path, prim_pts=prim.pts.astype(np.float32),
        prim_tan=prim.tan.astype(np.float32), prim_w=prim.w.astype(np.float32), covered=covered,
        icons=np.array(icons), scores=np.array(scores), gains=np.array(gains),
        icon_pts=np.concatenate(ipts), icon_tan=np.concatenate(itan), icon_n=np.array(n_icon))
    os.replace(out + ".tmp.npz", out)
    return i, f"{os.path.basename(path)} unc {(~covered).mean():.0%} best {max(scores) / prim.w.sum():.3f}", \
        time.time() - t0


def main():
    global _CFG, _ARGS, _LIB
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("library")
    ap.add_argument("--out", default="runs/labels")
    ap.add_argument("--hand", default="data/hand_cutout.png")
    ap.add_argument("--hand-frac", type=float, default=0.4)
    ap.add_argument("--states", type=int, default=125)
    ap.add_argument("--icons", type=int, default=24, help="icons labelled per state")
    ap.add_argument("--start", type=int, default=0, help="first state index (to extend a run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--eligible", default="", help="only use library paths listed here (eligible.py)")
    _ARGS, rest = ap.parse_known_args()
    _CFG, _ = ca.parse_cfg(["x", "y"] + rest)
    _CFG.verbose = False
    _LIB = sorted(glob.glob(os.path.join(_ARGS.library, "**", "*.png"), recursive=True))
    if _ARGS.eligible:
        ok = set(open(_ARGS.eligible).read().split())
        _LIB = [p for p in _LIB if p in ok]
    os.makedirs(_ARGS.out, exist_ok=True)
    print(f"library {len(_LIB)} icons, states {_ARGS.start}..{_ARGS.start + _ARGS.states - 1}, "
          f"{_ARGS.icons} icons each, cfg: {' '.join(rest)}", flush=True)
    import multiprocessing as mp
    t0, done = time.time(), 0
    with mp.get_context("fork").Pool(_ARGS.jobs) as pool:
        for i, msg, dt in pool.imap_unordered(label_state, range(_ARGS.start, _ARGS.start + _ARGS.states)):
            done += 1
            print(f"[{done}/{_ARGS.states}] state {i}: {msg} ({dt:.0f}s, elapsed {time.time() - t0:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
