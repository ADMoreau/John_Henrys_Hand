#!/usr/bin/env python3
"""Re-score existing label states under new cover_align flags, without re-drawing them.

  python src/rescore.py runs/mined runs/mined_v4 --jobs 14 [cover_align flags]

Same states (primary, covered mask) and same icons; each icon is re-aligned with
best_for_remainder under the given flags and its score/gain replaced. Pairs whose old score is
at or below --skip-below (pp of the primary) are copied unchanged: flags that only forbid
placements (stretch, upscale caps) can only lower a score, and the scorer floors targets there
anyway (scorer.py train --clip-low). Resumable: finished states are skipped.
"""
import argparse, glob, os, time
import multiprocessing as mp
import numpy as np
import cover_align as ca

_ST = {}


def _job(a):
    f, j, p = a
    st = _ST[f]
    rng = np.random.default_rng([_ST["cfg"].seed, 7, j])
    try:
        sh = ca.Shape(p, _ST["cfg"], rng, icon=True)
        b = ca.best_for_remainder(p, sh, st["prim"], st["rem"], st["covered"], _ST["cfg"], rng)
    except Exception:
        return f, j, None
    if b is None:
        return f, j, (0.0, 0.0)
    return f, j, (b["score"], float(st["prim"].w[b["mask"] & ~st["covered"]].sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--skip-below", type=float, default=-5.0)
    args, rest = ap.parse_known_args()
    os.makedirs(args.out, exist_ok=True)
    todo = [f for f in sorted(glob.glob(os.path.join(args.src, "state_*.npz")))
            if not os.path.exists(os.path.join(args.out, os.path.basename(f)))]
    t0 = time.time()
    for n, f in enumerate(todo, 1):
        d = dict(np.load(f))
        cfg, _ = ca.parse_cfg([str(d["primary"]), "x"] + rest)
        prim = ca.Shape(str(d["primary"]), cfg, np.random.default_rng(cfg.seed))
        ca.build_raster(prim, cfg)
        if len(prim.pts) != len(d["covered"]):
            raise SystemExit(f"{f}: primary has {len(prim.pts)} points, state has {len(d['covered'])}; "
                             "the edge flags must match the ones the state was made with")
        covered = d["covered"]
        _ST.clear()
        _ST.update(cfg=cfg)
        _ST[f] = dict(prim=prim, rem=ca.sub_prim(prim, ~covered, cfg), covered=covered)
        y = 100 * d["scores"] / d["prim_w"].sum()
        jobs = [(f, j, str(p)) for j, p in enumerate(d["icons"]) if y[j] > args.skip_below]
        with mp.get_context("fork").Pool(cfg.jobs) as pool:
            res = pool.map(_job, jobs, chunksize=1)
        old = y.copy()
        for _, j, r in res:
            if r is not None:
                d["scores"][j], d["gains"][j] = r
        new = 100 * d["scores"] / d["prim_w"].sum()
        tmp = os.path.join(args.out, os.path.basename(f) + ".tmp.npz")
        np.savez_compressed(tmp, **d)
        os.replace(tmp, os.path.join(args.out, os.path.basename(f)))
        print(f"{n}/{len(todo)} {os.path.basename(f)}: {len(jobs)} re-aligned, best {old.max():.2f} -> "
              f"{np.nanmax(new[np.isfinite(new)]) if np.isfinite(new).any() else float('nan'):.2f}pp, "
              f"lowered {(new < old - 0.01).sum()}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
