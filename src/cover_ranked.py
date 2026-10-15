#!/usr/bin/env python3
"""Residual greedy over the full library, with the learned scorer choosing what to align.

  python src/cover_ranked.py data/hand_cutout.png --emb runs/library_emb.npz --model runs/scorer.pt \
      --out runs/ranked.png --shortlist 64 [cover_align flags, e.g. --warp --warp-lambda 0.001]

Each round: embed the current state (uncovered/covered primary points), score every library
icon, align the top --shortlist to the uncovered remainder with cover_align's full search
(best_for_remainder), place the best. Stops on --max-images, --target or --min-gain like
cover_align. Prints, per round, how the scorer's ranking compared with the aligner's.

With --reuse K, every aligned icon keeps up to K distinct placements; in later rounds they are
re-fitted to the new remainder (ICP + warp, ~1 s) instead of being thrown away, so each round
competes the new shortlist against everything aligned before. Icons whose refit stops paying
(score <= 0) leave the pool.

With --occlusion, coverage counts only outline left visible in the render (occlusion.py): an
icon's score is what it visibly adds minus the covered outline of larger icons it would hide.

With --stages "a1:m1,a2:m2,...", running out of icons that pay (--exhaust-rounds rounds in a row
with nothing above --min-gain) moves on to the next stage instead of stopping: stray penalty
--alpha a_k and size cap --max-scale m_k, so the smaller icons placed later pay less for stray.

With --checkpoint PATH the run saves its state after every finished round and, when PATH
exists at start, resumes from it, so it can be stopped (freeing its memory) and restarted;
rounds are seeded by index, so a resumed run matches an uninterrupted one.
"""
import argparse, os, pickle, time
import numpy as np
import cover_align as ca
import scorer

_ST = {}


def _align(args):
    i, p = args
    st = _ST
    rng = np.random.default_rng([st["cfg"].seed, st["round"], i])
    try:
        sh = ca.Shape(p, st["cfg"], rng, icon=True)
    except Exception:
        return p, None, None
    return p, sh, ca.best_for_remainder(p, sh, st["prim"], st["rem"], st["covered"], st["cfg"], rng,
                                        keep=st["reuse"])


def _refit(p):
    st = _ST
    rng = np.random.default_rng([st["cfg"].seed, st["round"], 1 << 20, st["index"][p]])
    return p, ca.refine_placements(p, st["shapes"][p], st["kept"][p], st["prim"], st["rem"], st["covered"],
                                   st["cfg"], rng, keep=st["reuse"])


def _occlude(sh, best):
    """--occlusion: re-score a placement for what it visibly adds and what it hides."""
    st = _ST
    if st["canvas"] is None or best is None or not best["score"] > 0:
        return best
    st["canvas"].attach(best, sh, st["prim"])
    best["score_raw"] = best["score"]
    best["score"] = st["canvas"].score(best, st["prim"], st["cfg"], st["covered"], st["vis"])[0]
    return best


def _task(t):
    if t[0] == "new":
        p, sh, best = _align(t[1])
        return "new", p, sh, _occlude(sh, best)
    p, best = _refit(t[1])
    return "refit", p, _occlude(_ST["shapes"][p], best)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("primary")
    ap.add_argument("--emb", default="runs/library_emb.npz")
    ap.add_argument("--model", default="runs/scorer.pt")
    ap.add_argument("--shortlist", type=int, default=64, help="icons aligned per round")
    ap.add_argument("--save-labels", default="", help="save each round's shortlist with its true "
                    "scores here, in make_labels.py format, to retrain the scorer on what it gets wrong")
    ap.add_argument("--checkpoint", default="", help="save state here after every round; resume "
                    "from it if it exists")
    ap.add_argument("--reuse", type=int, default=0, help="keep this many placements per aligned icon "
                    "and re-fit them in later rounds (0 = align only the new shortlist)")
    ap.add_argument("--picks", type=int, default=1, help="most icons placed per round: after the "
                    "best, the next best of the same round's candidates (scores updated for overlap)")
    ap.add_argument("--pick-frac", type=float, default=0.5, help="extra picks in a round must score at "
                    "least this fraction of the round's first pick")
    ap.add_argument("--occlusion", action="store_true", help="count only outline that stays visible "
                    "in the render (smaller icons drawn on top hide larger icons' outline)")
    ap.add_argument("--stages", default="", help="alpha:max_scale pairs to step through, in order, "
                    "each time the current settings run out of icons that pay")
    ap.add_argument("--exhaust-rounds", type=int, default=2, help="rounds in a row with nothing paying "
                    "that count as running out (then the next --stages entry, or stop); 0 = a full pass: "
                    "run out only once every eligible icon has been tried at this stage and nothing pays")
    ap.add_argument("--eligible", default="", help="list of eligible library paths (eligible.py); "
                    "others are never placed")
    args, rest = ap.parse_known_args()
    cfg, _ = ca.parse_cfg([args.primary, "x"] + rest)
    import torch
    torch.set_num_threads(4)
    model = scorer.load_model(args.model)
    lib = np.load(args.emb)
    paths, E = list(lib["paths"]), torch.from_numpy(lib["emb"])
    banned = np.zeros(len(paths), bool)
    if args.eligible:
        ok = set(open(args.eligible).read().split())
        banned = np.array([p not in ok for p in paths])
        print(f"eligible: {(~banned).sum()} of {len(paths)} images", flush=True)

    prim = ca.Shape(args.primary, cfg, np.random.default_rng(cfg.seed))
    ca.build_raster(prim, cfg)
    covered = np.zeros(len(prim.pts), bool)
    chosen, used, shapes, n_round = [], set(), {}, 0
    kept, first_round = {}, {}  # --reuse: icon -> placements to re-fit, round it was first aligned
    index = {p: k for k, p in enumerate(paths)}
    if args.checkpoint and os.path.exists(args.checkpoint):
        ck = pickle.load(open(args.checkpoint, "rb"))
        assert len(ck["covered"]) == len(covered), "checkpoint is for a different primary/flags"
        chosen, used, kept, first_round, shapes, covered = (ck[k] for k in (
            "chosen", "used", "kept", "first_round", "shapes", "covered"))
        n_round = ck.get("rounds", len(chosen))
    stages = [tuple(float(x) for x in st.split(":")) for st in args.stages.split(",") if st.strip()]
    resumed = args.checkpoint and os.path.exists(args.checkpoint)
    stage, dry = (ck.get("stage", 0), ck.get("dry", 0)) if resumed else (0, 0)
    tried = set(ck.get("tried", ())) if resumed else set()  # aligned at this stage without paying

    def set_stage(k):
        if k > 0:
            cfg.alpha, cfg.max_scale = stages[k - 1]
            print(f"stage {k}: alpha {cfg.alpha:g}, max scale {cfg.max_scale:g} "
                  f"({cfg.max_scale * 368.8:.0f}px rms on the hand)", flush=True)
    set_stage(stage)
    canvas = None
    if args.occlusion:
        import cv2
        import occlusion
        canvas = occlusion.Canvas(prim, cv2.imread(args.primary, cv2.IMREAD_UNCHANGED).shape[:2])
        for c in chosen:
            canvas.add(c)
        if chosen:
            assert np.array_equal(canvas.covered(), covered), "checkpoint covered != replayed canvas"
    if chosen:
        print(f"resumed from {args.checkpoint}: {len(chosen)} icons placed, "
              f"{ca.wcov(prim, covered):.1%} covered, {len(kept)} kept", flush=True)
    def save():
        need = used | set(kept)
        tmp = args.checkpoint + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(dict(chosen=chosen, rounds=n_round, used=used, kept=kept, first_round=first_round,
                             covered=covered, shapes={p: shapes[p] for p in need}, stage=stage, dry=dry,
                             tried=tried), f)
        os.replace(tmp, args.checkpoint)

    import multiprocessing as mp
    while len(chosen) < cfg.max_images and ca.wcov(prim, covered) < cfg.target:
        if (~covered).sum() < 6:
            break
        t0 = time.time()
        n_round += 1
        with torch.no_grad():
            es = model.embed_state(torch.from_numpy(scorer.state_raster(prim.pts, prim.tan, covered)[None]))
            pred = model.score(E, es).numpy()
        pred[[k for k, p in enumerate(paths) if p in used or p in kept or p in tried]] = -np.inf
        pred[banned] = -np.inf
        short = np.argsort(-pred)[:args.shortlist]
        short = short[np.isfinite(pred[short])]  # fewer left untried than the shortlist
        _ST.update(cfg=cfg, prim=prim, rem=ca.sub_prim(prim, ~covered, cfg), covered=covered,
                   round=n_round, reuse=args.reuse, shapes=shapes, kept=kept, index=index, canvas=canvas,
                   vis=canvas.visible() if canvas is not None else None)
        refits = sorted(kept)
        tasks = [("new", (int(k), paths[k])) for k in short] + [("refit", p) for p in refits]
        with mp.get_context("fork").Pool(cfg.jobs) as pool:  # slow fresh searches first
            out = pool.map(_task, tasks, chunksize=1)
        res = [r[1:] for r in out if r[0] == "new"]
        cands, true = [], np.full(len(short), -np.inf)
        for j, (p, sh, best) in enumerate(res):
            if best is not None:
                shapes[p] = sh
                best["pred"], best["pred_rank"] = float(pred[short[j]]), j + 1
                cands.append(best)
                true[j] = best["score"]
        tried.update(p for p, sh, b in res if b is None or not b["score"] > 0)
        n_refit = 0
        for _, p, best in (r for r in out if r[0] == "refit"):
            if best is not None and best["score"] > 0:
                best["pred"], best["pred_rank"] = np.nan, None
                cands.append(best)
                n_refit += 1
            elif p in kept:  # stopped paying: tried at this stage (a full pass must not search it again)
                kept.pop(p)
                tried.add(p)
        if args.reuse:
            for c in cands:
                if c.get("keep") and c["score"] > 0:
                    kept[c["image"]] = c["keep"]
                    first_round.setdefault(c["image"], n_round)
        best = max(cands, key=lambda c: c["score"]) if cands else None
        if args.save_labels:
            ok = [j for j, (p, sh, b) in enumerate(res) if sh is not None]
            os.makedirs(args.save_labels, exist_ok=True)
            tag = os.path.splitext(os.path.basename(cfg.out))[0]
            np.savez_compressed(
                os.path.join(args.save_labels, f"state_{tag}_r{len(chosen):02d}.npz"), primary=args.primary,
                prim_pts=prim.pts.astype(np.float32), prim_tan=prim.tan.astype(np.float32),
                prim_w=prim.w.astype(np.float32), covered=covered,
                icons=np.array([res[j][0] for j in ok]),
                scores=np.array([res[j][2]["score"] if res[j][2] else 0.0 for j in ok]),
                gains=np.array([float(prim.w[res[j][2]["mask"] & ~covered].sum()) if res[j][2] else 0.0 for j in ok]),
                icon_pts=np.concatenate([res[j][1].pts for j in ok]).astype(np.float32),
                icon_tan=np.concatenate([res[j][1].tan for j in ok]).astype(np.float32),
                icon_n=np.array([len(res[j][1].pts) for j in ok]))
        W = prim.w.sum()
        if cfg.verbose and cands:
            from scipy.stats import spearmanr
            ok = np.isfinite(true)
            rho = spearmanr(pred[short][ok], true[ok])[0] if ok.sum() > 2 else np.nan
            src = (f"scorer rank {best['pred_rank']}, pred {best['pred']:.2f}% vs true "
                   f"{100 * best['score'] / W:.2f}%" if best["pred_rank"] else
                   f"re-fit, first aligned in round {first_round[best['image']]}, {100 * best['score'] / W:.2f}%")
            print(f"  round {n_round}" + (f" (icon {len(chosen) + 1})" if n_round != len(chosen) + 1 else "")
                  + f": aligned top {len(short)}" + (f" + re-fit {len(refits)} kept "
                  f"({n_refit} still pay)" if refits else "") + f" in {time.time() - t0:.0f}s; "
                  f"spearman(pred, true) {rho:.2f}; best: {src}", flush=True)
        if best is None or best["score"] < cfg.min_gain * W:
            dry += 1
            if (len(short) > 0) if args.exhaust_rounds == 0 else (dry < args.exhaust_rounds):
                pass
            elif stage < len(stages):
                stage, dry = stage + 1, 0
                tried.clear()
                set_stage(stage)
            else:
                break
            if args.checkpoint:
                save()
            continue
        dry = 0
        # --picks: after the best, keep taking the next best of this round's candidates, their
        # scores updated for what was just covered (exact: a placement's mask and penalty don't
        # depend on the covered set; newly covered points move from new to overlap)
        floor = max(cfg.min_gain * W, args.pick_frac * best["score"])
        for _ in range(args.picks):
            if canvas is not None:
                before = covered
                canvas.add(best)
                covered = canvas.covered()
                newly = covered & ~before
                best["gain"] = float(prim.w[covered].sum() - prim.w[before].sum())
            else:
                newly = best["mask"] & ~covered
                best["gain"] = float(prim.w[newly].sum())
                covered |= best["mask"]
            best["coverage"] = ca.wcov(prim, covered)
            used.add(best["image"])
            kept.pop(best["image"], None)
            chosen.append(best)
            if cfg.verbose:
                print(f"    -> {os.path.relpath(best['image'], os.path.dirname(paths[0]) + '/..')} "
                      f"-> {best['coverage']:.1%}" + (" (warped)" if best["warp"] is not None else ""), flush=True)
            cands = [c for c in cands if c["image"] != best["image"]]
            if canvas is not None:  # exact re-score against the new canvas (hiding can go both ways)
                vis = canvas.visible()
                for c in cands:
                    if c["score"] > 0:
                        c["score"] = canvas.score(c, prim, cfg, covered, vis)[0]
            else:
                for c in cands:
                    c["score"] -= (1 + cfg.beta) * float(prim.w[c["mask"] & newly].sum())
            best = max(cands, key=lambda c: c["score"]) if cands else None
            if best is None or best["score"] < floor or len(chosen) >= cfg.max_images:
                break
        if args.checkpoint:
            save()
    ca.report(chosen, covered, prim, shapes, cfg.out, f"Scorer-ranked greedy (shortlist {args.shortlist})")


if __name__ == "__main__":
    main()
