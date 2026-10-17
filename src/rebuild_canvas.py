#!/usr/bin/env python3
"""Re-attach a checkpoint's placements to a fresh occlusion canvas, then optionally prune.

  python src/rebuild_canvas.py IN.ckpt OUT.ckpt OUT_prefix --flags configs/flags_vN.txt [--prune]

Footprints and credited pixels are recomputed with the current occlusion.py / render_overlay.py
(a checkpoint keeps the ones it was searched with), and the checkpoint's coverage is replaced by
the canvas's. --prune then removes, one at a time, placements whose removal raises coverage:
icons that hide more visible outline than they add. Writes OUT.ckpt (a cover_ranked checkpoint
to resume with --occlusion; pruned images may be aligned again) and OUT_prefix.json/.png.
"""
import argparse, os, pickle, sys, time
from multiprocessing import Pool
import cv2
import numpy as np
import cover_align as ca
import occlusion

_ST = {}


def _attach(i):
    c, st = dict(_ST["chosen"][i]), _ST
    st["canvas"].attach(c, st["shapes"][c["image"]], st["prim"])
    return c["fp"], c["rp"], c["rq"]


def build(prim, hw, chosen):
    canvas = occlusion.Canvas(prim, hw)
    for c in chosen:
        canvas.add(c)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_in")
    ap.add_argument("ckpt_out")
    ap.add_argument("out")
    ap.add_argument("--flags", required=True, help="the run's search flags (configs/flags_vN.txt)")
    ap.add_argument("--primary", default="data/hand_cutout.png")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--jobs", type=int, default=14)
    args = ap.parse_args()
    cfg, _ = ca.parse_cfg([args.primary, args.out] + open(args.flags).read().split())
    prim = ca.Shape(args.primary, cfg, np.random.default_rng(cfg.seed))
    hw = cv2.imread(args.primary, cv2.IMREAD_UNCHANGED).shape[:2]
    ck = pickle.load(open(args.ckpt_in, "rb"))
    chosen, shapes = ck["chosen"], ck["shapes"]
    print(f"{args.ckpt_in}: {len(chosen)} icons, checkpoint coverage {ca.wcov(prim, ck['covered']):.1%}",
          flush=True)

    t0 = time.time()
    _ST.update(chosen=chosen, shapes=shapes, prim=prim, canvas=occlusion.Canvas(prim, hw))
    with Pool(args.jobs) as pool:
        geo = pool.map(_attach, range(len(chosen)), chunksize=4)
    for c, (fp, rp, rq) in zip(chosen, geo):
        c["fp"], c["rp"], c["rq"] = fp, rp, rq
    covered = build(prim, hw, chosen).covered()
    print(f"re-attached in {time.time() - t0:.0f}s: {ca.wcov(prim, covered):.1%} covered", flush=True)

    removed = []
    while args.prune:
        base = ca.wcov(prim, covered)
        gains = []
        for i in range(len(chosen)):  # coverage with each icon left out
            gains.append(ca.wcov(prim, build(prim, hw, chosen[:i] + chosen[i + 1:]).covered()) - base)
        order = [i for i in np.argsort(gains)[::-1] if gains[i] > 0]
        if not order:
            break
        drop = set()
        for i in order:  # most harmful first; accept each only if it still helps with the others gone
            keep = [c for k, c in enumerate(chosen) if k not in drop | {i}]
            cov = build(prim, hw, keep).covered()
            if ca.wcov(prim, cov) > ca.wcov(prim, covered):
                drop.add(i)
                covered = cov
        removed += [chosen[i]["image"] for i in sorted(drop)]
        chosen = [c for k, c in enumerate(chosen) if k not in drop]
        print(f"pruned {len(drop)}: {len(chosen)} icons, {ca.wcov(prim, covered):.1%} covered", flush=True)

    used = set(ck["used"]) - set(removed)
    ck.update(chosen=chosen, covered=covered, used=used,
              shapes={p: s for p, s in shapes.items() if p in used | set(ck["kept"])})
    with open(args.ckpt_out + ".tmp", "wb") as f:
        pickle.dump(ck, f)
    os.replace(args.ckpt_out + ".tmp", args.ckpt_out)
    print(f"wrote {args.ckpt_out}" + (f"; removed {len(removed)} icons" if removed else ""), flush=True)
    ca.report(chosen, covered, prim, shapes, args.out, f"Rebuilt canvas ({args.ckpt_in})")


if __name__ == "__main__":
    main()
