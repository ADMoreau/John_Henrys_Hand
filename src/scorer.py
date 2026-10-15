#!/usr/bin/env python3
"""Next-icon scorer: predicts cover_align's gain for (icon outline, uncovered remainder).

  python src/scorer.py train runs/labels --out runs/scorer.pt
  python src/scorer.py embed LIBRARY_DIR --model runs/scorer.pt --out runs/library_emb.npz [cover_align flags]

Two small CNNs: one embeds an icon's outline (--icon-res, default 64, rotation/mirror/stretch-
augmented, since the aligner searches those), one embeds the primary state (--state-res,
default 128: uncovered points, their
orientation, covered points). A head scores every (icon, state) pair from the two embeddings,
so icon embeddings are computed once for the whole library and each greedy round only
re-embeds the state. Trained listwise per state (rank the icons as the aligner would) plus a
small regression term so predictions stay in gain units (% of the primary).
"""
import argparse, glob, os, time
import numpy as np

N_ICON, N_STATE, R_ICON, R_STATE = 64, 128, 2.5, 2.5
AUG_STRETCH = 2.5  # icon augmentation; match cover_align --max-aniso


def set_res(n_icon, n_state):
    """Raster sizes (powers of two); load_model sets them from the checkpoint."""
    global N_ICON, N_STATE
    N_ICON, N_STATE = n_icon, n_state


# ---------------- rasterization (numpy) ----------------
def splat(pts, tan, n, R, w=None):
    """Occupancy + doubled-angle orientation channels (sign-free tangent) on an n x n grid."""
    ij = np.floor((pts + R) / (2 * R) * n).astype(np.int64)
    ok = (ij >= 0).all(1) & (ij < n).all(1)
    ij, tan = ij[ok], tan[ok]
    w = np.ones(len(ij)) if w is None else w[ok]
    th2 = 2 * np.arctan2(tan[:, 1], tan[:, 0])
    out = np.zeros((3, n, n), np.float32)
    for c, v in enumerate((w, w * np.cos(th2), w * np.sin(th2))):
        np.add.at(out[c], (ij[:, 1], ij[:, 0]), v)
    occ = np.maximum(out[0], 1e-6)
    out[1:] /= occ  # mean orientation per cell
    out[0] = np.minimum(out[0], 1.0)
    return out


def rand_linear(rng, stretch=2.5):
    """Random rotation x mirror x stretch: transforms the aligner's search makes free."""
    th, phi = rng.uniform(0, 2 * np.pi, 2)
    a = np.exp(rng.uniform(0, np.log(stretch)))
    rot = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    A = rot(th) @ rot(phi) @ np.diag([np.sqrt(a), 1 / np.sqrt(a)]) @ rot(-phi)
    return A @ np.diag([rng.choice([-1.0, 1.0]), 1.0])


def icon_raster(pts, tan, rng=None):
    pts = pts - pts.mean(0)
    if rng is not None:
        A = rand_linear(rng, AUG_STRETCH)
        pts, tan = pts @ A.T, tan @ A.T
        tan = tan / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12)
    pts = pts / np.sqrt((pts ** 2).sum(1).mean())  # unit RMS radius, as cover_align normalizes
    return splat(pts, tan, N_ICON, R_ICON)


def state_raster(pts, tan, covered, rng=None):
    """Primary points are already in cover_align's normalized frame (unit RMS radius)."""
    if rng is not None:  # rotation/mirror only: the state's scale is meaningful to the icon fit
        A = rand_linear(rng, stretch=1.0)
        pts, tan = pts @ A.T, tan @ A.T
    unc = splat(pts[~covered], tan[~covered], N_STATE, R_STATE)
    cov = splat(pts[covered], tan[covered], N_STATE, R_STATE)[:1] if covered.any() \
        else np.zeros((1, N_STATE, N_STATE), np.float32)
    return np.concatenate([unc, cov])


# ---------------- model ----------------
def _torch():
    import torch
    return torch


def build_model(dim=128):
    torch = _torch()
    nn = torch.nn

    def widths(n, base):
        extra = int(round(np.log2(n / 4))) - len(base)
        return (16,) * extra + base if extra >= 0 else base[-extra:]

    def enc(cin, widths):
        layers, c = [], cin
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(),
                       nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2)]
            c = w
        return nn.Sequential(*layers), c

    class Scorer(nn.Module):
        def __init__(self):
            super().__init__()
            # pool down to 4x4: 64 -> (32, 64, 96, 128), each doubling adds a stage in front
            self.icon, ci = enc(3, widths(N_ICON, (32, 64, 96, 128)))
            self.state, cs = enc(4, widths(N_STATE, (32, 48, 64, 96, 128)))
            self.icon_fc = nn.Sequential(nn.Linear(2 * ci, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.state_fc = nn.Sequential(nn.Linear(2 * cs, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.head = nn.Sequential(nn.Linear(3 * dim, 256), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

        @staticmethod
        def _pool(f):
            return torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)

        def embed_icon(self, x):
            return self.icon_fc(self._pool(self.icon(x)))

        def embed_state(self, x):
            return self.state_fc(self._pool(self.state(x)))

        def score(self, ei, es):
            """ei (n, d) icons; es (n, d) or (1, d) state -> (n,) predicted gain in % of primary."""
            es = es.expand_as(ei)
            return self.head(torch.cat([ei, es, ei * es], 1)).squeeze(1)

    return Scorer()


def load_model(path):
    """Model from a checkpoint, with the raster sizes it was trained at."""
    torch = _torch()
    ck = torch.load(path)
    set_res(ck.get("n_icon", 64), ck.get("n_state", 128))
    model = build_model()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


# ---------------- data ----------------
def load_labels(label_dirs, clip_low=None):
    """States from one or more label dirs (comma-separated): make_labels.py output and/or
    rounds saved by cover_ranked.py --save-labels. clip_low floors the targets (pp) so very
    bad icons don't dominate what the model spends its capacity on."""
    states = []
    files = sorted(f for d in label_dirs.split(",") for f in glob.glob(os.path.join(d, "state_*.npz")))
    for f in files:
        d = np.load(f)
        W = float(d["prim_w"].sum())
        offs = np.concatenate([[0], np.cumsum(d["icon_n"])])
        states.append(dict(
            file=f, primary=str(d["primary"]), pts=d["prim_pts"], tan=d["prim_tan"], covered=d["covered"],
            icons=list(d["icons"]),  # objective gain, % of the primary
            y=np.maximum(100 * d["scores"] / W, -np.inf if clip_low is None else clip_low),
            icon_pts=[d["icon_pts"][a:b] for a, b in zip(offs[:-1], offs[1:])],
            icon_tan=[d["icon_tan"][a:b] for a, b in zip(offs[:-1], offs[1:])]))
    return states


def subsample(s, k, rng):
    """At most k of a state's icons (always its best), so a step's activations stay bounded:
    a 256-icon shortlist state at 128px is gigabytes per conv layer."""
    if len(s["y"]) <= k:
        return s
    keep = np.concatenate([[s["y"].argmax()], rng.choice(np.delete(np.arange(len(s["y"])), s["y"].argmax()),
                                                         k - 1, replace=False)])
    return dict(s, y=s["y"][keep], icon_pts=[s["icon_pts"][j] for j in keep],
                icon_tan=[s["icon_tan"][j] for j in keep])


def batch_tensors(states, rng, augment):
    torch = _torch()
    S = np.stack([state_raster(s["pts"], s["tan"], s["covered"], rng if augment else None) for s in states])
    I = np.stack([icon_raster(p, t, rng if augment else None)
                  for s in states for p, t in zip(s["icon_pts"], s["icon_tan"])])
    group = np.concatenate([np.full(len(s["y"]), k) for k, s in enumerate(states)])
    y = np.concatenate([s["y"] for s in states]).astype(np.float32)
    return torch.from_numpy(S), torch.from_numpy(I), torch.from_numpy(group), torch.from_numpy(y)


def predict_states(model, states):
    torch = _torch()
    model.eval()
    out = []
    with torch.no_grad():
        for s in states:
            S, I, g, y = batch_tensors([s], None, False)
            out.append(model.score(model.embed_icon(I), model.embed_state(S)).numpy())
    return out


def rank_metrics(preds, states):
    """Per held-out state: is the aligner's best icon the scorer's pick / in its top 3?
    regret = best true gain - true gain of the scorer's pick (pp of the primary)."""
    from scipy.stats import spearmanr
    top1, top3, regret, rho, rnd_regret = [], [], [], [], []
    for p, s in zip(preds, states):
        y = s["y"]
        best = y.argmax()
        order = np.argsort(-p)
        top1.append(order[0] == best)
        top3.append(best in order[:3])
        regret.append(y[best] - y[order[0]])
        rnd_regret.append(y[best] - y.mean())
        rho.append(spearmanr(p, y)[0])
    n = len(states[0]["y"])
    return dict(top1=np.mean(top1), top3=np.mean(top3), regret=np.mean(regret), spearman=np.nanmean(rho),
                random_top1=1 / n, random_top3=3 / n, random_regret=np.mean(rnd_regret))


def train(args):
    torch = _torch()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    global AUG_STRETCH
    set_res(args.icon_res, args.state_res)
    AUG_STRETCH = args.aug_stretch
    states = load_labels(args.labels, args.clip_low)
    if args.val_from:  # validate on deployment-like shortlists, e.g. one cover_ranked run
        val = [s for s in states if args.val_from in os.path.basename(s["file"])]
        tr = [s for s in states if args.val_from not in os.path.basename(s["file"])]
    else:
        perm = rng.permutation(len(states))
        n_val = max(1, int(round(args.val_frac * len(states))))
        val, tr = [states[i] for i in perm[:n_val]], [states[i] for i in perm[n_val:]]
    print(f"{len(states)} states ({sum(len(s['y']) for s in states)} pairs): {len(tr)} train, {len(val)} val")
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_ep = None, -1
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for b in range(0, len(tr), args.batch_states):
            chunk = [subsample(s, args.max_icons, rng) for s in tr[b:b + args.batch_states]]
            S, I, g, y = batch_tensors(chunk, rng, True)
            p = model.score(model.embed_icon(I), model.embed_state(S)[g])
            loss = 0.0
            for k in range(len(chunk)):  # listwise: softmax over each state's icons
                m = g == k
                target = torch.softmax(y[m] / args.tau, 0)
                loss = loss - (target * torch.log_softmax(p[m] / args.tau, 0)).sum()
            loss = loss / len(chunk) + args.mse * torch.nn.functional.smooth_l1_loss(p, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(chunk)
        sched.step()
        rng.shuffle(tr)
        m = rank_metrics(predict_states(model, val), val)
        print(f"ep {ep:3d} loss {tot / len(tr):.4f} | val top1 {m['top1']:.2f} top3 {m['top3']:.2f} "
              f"regret {m['regret']:.2f}pp spearman {m['spearman']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if best is None or m["regret"] < best["regret"]:
            best, best_ep = m, ep
            torch.save(dict(state_dict=model.state_dict(), val_files=[s["file"] for s in val],
                            n_icon=N_ICON, n_state=N_STATE), args.out)
    print(f"best epoch {best_ep}: " + ", ".join(f"{k} {v:.3f}" for k, v in best.items()))


# ---------------- library embedding ----------------
_CFG = None


def _icon_raster_job(p):
    import cover_align as ca
    try:
        sh = ca.Shape(p, _CFG, np.random.default_rng(0), icon=True)
    except Exception:
        return p, None
    return p, icon_raster(sh.pts, sh.tan)


def embed(args):
    global _CFG
    import cover_align as ca
    torch = _torch()
    torch.set_num_threads(args.threads)
    _CFG, _ = ca.parse_cfg(["x", "y"] + args.rest)  # same icon-edge flags as the labels
    paths = sorted(glob.glob(os.path.join(args.library, "**", "*.png"), recursive=True))
    model = load_model(args.model)
    t0 = time.time()
    if args.cache and os.path.exists(args.cache + ".paths.txt"):
        # icon rasters depend only on the library and the icon-edge flags, not on the model
        keep = open(args.cache + ".paths.txt").read().split("\n")
        R = np.load(args.cache + ".npy", mmap_mode="r")
        if R.shape[-1] != N_ICON:
            raise SystemExit(f"{args.cache}.npy holds {R.shape[-1]}px rasters, the model wants {N_ICON}px")
    else:
        import multiprocessing as mp
        keep, rs = [], []
        with mp.get_context("fork").Pool(args.jobs) as pool:
            for n, (p, r) in enumerate(pool.imap(_icon_raster_job, paths, chunksize=32), 1):
                if r is not None:
                    keep.append(p)
                    rs.append(r.astype(np.float16))
                if n % 5000 == 0:
                    print(f"{n}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
        R = np.stack(rs)
        if args.cache:
            np.save(args.cache + ".npy", R)
            open(args.cache + ".paths.txt", "w").write("\n".join(keep))
    embs = []
    with torch.no_grad():
        for b in range(0, len(keep), 512):
            embs.append(model.embed_icon(torch.from_numpy(np.asarray(R[b:b + 512], np.float32))).numpy())
    np.savez(args.out, paths=np.array(keep), emb=np.concatenate(embs))
    print(f"embedded {len(keep)}/{len(paths)} icons -> {args.out} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("labels")
    t.add_argument("--out", default="runs/scorer.pt")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch-states", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=1e-3)
    t.add_argument("--tau", type=float, default=0.5, help="listwise temperature, in pp of the primary")
    t.add_argument("--mse", type=float, default=0.1, help="weight of the regression term")
    t.add_argument("--val-frac", type=float, default=0.2)
    t.add_argument("--clip-low", type=float, default=None, help="floor for targets, pp (e.g. -5)")
    t.add_argument("--threads", type=int, default=8)
    t.add_argument("--icon-res", type=int, default=64)
    t.add_argument("--max-icons", type=int, default=64, help="icons per state per step (memory bound)")
    t.add_argument("--state-res", type=int, default=128)
    t.add_argument("--aug-stretch", type=float, default=2.5, help="icon stretch augmentation; match --max-aniso")
    t.add_argument("--val-from", default="", help="validate on states whose file name contains this "
                   "(e.g. al_it3: one cover_ranked run's shortlists) instead of a random split")
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("embed")
    e.add_argument("library")
    e.add_argument("--model", default="runs/scorer.pt")
    e.add_argument("--out", default="runs/library_emb.npz")
    e.add_argument("--jobs", type=int, default=14)
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--cache", default="", help="icon raster cache prefix (reused if present; must "
                   "match the library and icon-edge flags)")
    args, rest = ap.parse_known_args()
    if rest and args.cmd == "train":
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    args.rest = rest  # embed: cover_align flags, e.g. --icon-edges detail
    train(args) if args.cmd == "train" else embed(args)


if __name__ == "__main__":
    main()
