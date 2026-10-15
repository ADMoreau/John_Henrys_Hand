#!/usr/bin/env python3
"""List the library images eligible for the cover search, one path per line.

  python src/eligible.py LIBRARY runs/style_labels.npz runs/eligible.txt --exclude data/excluded.txt

Dropped:
  * not transparent: no alpha channel or under 5% see-through pixels (opaque rectangles such
    as photos with their background, which would hide everything under them)
  * glow images (lens flares, light, smoke, fire): P(glow) >= --max-glow in the zero-shot
    style labels from style_zeroshot.py
  * text logos only with --no-text
  * images listed in --exclude (one path per line, # comments; relative paths are taken from
    the library root): ones rejected by eye; a line naming a directory drops everything under
    it (e.g. letter and number folders)
"""
import argparse, os
import multiprocessing as mp
import cv2
import numpy as np


def transparent(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim == 2 or im.shape[2] != 4:
        return False
    return bool((im[..., 3] < 16).mean() >= 0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("library")
    ap.add_argument("style")
    ap.add_argument("out")
    ap.add_argument("--max-glow", type=float, default=0.3)
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--exclude", default="", help="file of paths to drop (data/excluded.txt)")
    args = ap.parse_args()
    paths = sorted(os.path.join(d, f) for d, _, fs in os.walk(args.library, followlinks=True)
                   for f in fs if f.lower().endswith(".png"))
    with mp.Pool(14) as pool:
        tr = pool.map(transparent, paths, chunksize=64)
    z = np.load(args.style)
    C = list(z["classes"])
    P = dict(zip(z["paths"], z["prob"]))
    excl = set()
    if args.exclude:
        excl = {os.path.join(args.library, l) for l in (l.split("#")[0].strip() for l in open(args.exclude)) if l}
    excl_dirs = tuple(d.rstrip("/") + "/" for d in excl if os.path.isdir(d))
    keep, n_glow, n_text = [], 0, 0
    for p, t in zip(paths, tr):
        pr = P.get(p)
        if not t or pr is None or p in excl or any(p.startswith(d) for d in excl_dirs):
            continue
        if pr[C.index("glow")] >= args.max_glow:
            n_glow += 1
            continue
        if args.no_text and pr[C.index("text")] >= 0.5:
            n_text += 1
            continue
        keep.append(p)
    with open(args.out, "w") as f:
        f.write("\n".join(keep) + "\n")
    print(f"{len(keep)} of {len(paths)} eligible: {len(paths) - sum(tr)} not transparent, "
          f"{n_glow} glow" + (f", {n_text} text" if args.no_text else "") +
          (f", {sum(p in excl or p.startswith(excl_dirs) for p in paths) if excl else 0} excluded"
           if excl else "") + f" -> {args.out}")


if __name__ == "__main__":
    main()
