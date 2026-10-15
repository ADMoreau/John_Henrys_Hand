#!/usr/bin/env python3
"""Ground-truth labels: each icon's best single-placement score from cover_align.

  python src/labels.py primary.png library_dir/ --out runs/labels.json [cover_align flags]

For every icon: global search + ICP (as in cover_align), then the best placement's
score on an empty primary. This is exactly "how well can this icon alone cover the
primary", used to check whether a cheap ranking predicts it.
"""
import glob, json, os, sys
import numpy as np
import cover_align as ca

_PRIM = _CFG = None


def _label(args):
    i, p = args
    rng = np.random.default_rng([_CFG.seed, i])
    try:
        sh = ca.Shape(p, _CFG, rng, icon=True)
        none = np.zeros(len(_PRIM.pts), bool)
        best = max((ca.evaluate(sh, M, _PRIM, none, _CFG)[0]
                    for M in ca.initial_placements(p, sh, _PRIM, _CFG, rng)), default=0.0)
        return p, float(best)
    except Exception:
        return p, None


if __name__ == "__main__":
    argv = sys.argv[1:]
    out = "runs/labels.json"
    if "--out" in argv:
        k = argv.index("--out")
        out = argv[k + 1]
        del argv[k:k + 2]
    cfg, _ = ca.parse_cfg(argv)
    paths = sorted(p for p in glob.glob(os.path.join(cfg.library, "**", "*"), recursive=True)
                   if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")))
    rng = np.random.default_rng(cfg.seed)
    prim = ca.Shape(cfg.primary, cfg, rng)
    ca.build_raster(prim, cfg)
    _PRIM, _CFG = prim, cfg
    import multiprocessing as mp
    with mp.get_context("fork").Pool(cfg.jobs) as pool:
        res = {}
        for n, (p, s) in enumerate(pool.imap(_label, list(enumerate(paths)), chunksize=1), 1):
            res[p] = s
            if n % 10 == 0:
                print(f"{n}/{len(paths)}", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}")
