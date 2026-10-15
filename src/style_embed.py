#!/usr/bin/env python3
"""SigLIP image embeddings for the library, for zero-shot style filtering (icon art vs photos,
textures, flares, text logos).

  python src/style_embed.py LIBRARY_DIR runs/siglip.npz [--limit N]

Each image is cropped to its visible content and composited on white, the way it will appear
in the final product. Embeddings are L2-normalised; resumable via the partial file.
"""
import argparse, os, time
import multiprocessing as mp
import cv2
import numpy as np

MODEL = "google/siglip-base-patch16-224"


def load(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 3:
        im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
    ys, xs = np.nonzero(im[..., 3] > 16)
    if len(ys):
        im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = max(im.shape[:2])
    f = 224 / s
    im = cv2.resize(im, (max(1, round(im.shape[1] * f)), max(1, round(im.shape[0] * f))),
                    interpolation=cv2.INTER_AREA)
    a = im[..., 3:4] / 255.0
    rgb = (im[..., 2::-1] * a + 255 * (1 - a)).astype(np.uint8)
    out = np.full((224, 224, 3), 255, np.uint8)  # pad to square, centred
    y0, x0 = (224 - rgb.shape[0]) // 2, (224 - rgb.shape[1]) // 2
    out[y0:y0 + rgb.shape[0], x0:x0 + rgb.shape[1]] = rgb
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    import torch
    from transformers import AutoModel
    torch.set_num_threads(12)
    model = AutoModel.from_pretrained(MODEL).eval()
    paths = sorted(os.path.join(d, f) for d, _, fs in os.walk(args.root, followlinks=True)
                   for f in fs if f.lower().endswith(".png"))
    if args.limit:
        paths = paths[::max(1, len(paths) // args.limit)][:args.limit]
    part = args.out + ".partial.npz"
    done_p, done_e = [], []
    if os.path.exists(part):
        z = np.load(part)
        done_p, done_e = list(z["paths"]), list(z["emb"])
    todo = paths[len(done_p):] if done_p == paths[:len(done_p)] else paths
    if todo is paths:
        done_p, done_e = [], []
    mean, std = np.float32(0.5), np.float32(0.5)
    t0 = time.time()
    with mp.Pool(4) as pool:
        for b in range(0, len(todo), args.batch):
            chunk = todo[b:b + args.batch]
            ims = pool.map(load, chunk)
            keep = [i for i, x in enumerate(ims) if x is not None]
            x = torch.from_numpy((np.stack([ims[i] for i in keep]).astype(np.float32) / 255 - mean) / std)
            with torch.no_grad():
                e = model.get_image_features(pixel_values=x.permute(0, 3, 1, 2))
                if not torch.is_tensor(e):
                    e = e.pooler_output
                e = torch.nn.functional.normalize(e, dim=-1).numpy().astype(np.float16)
            done_p += [chunk[i] for i in keep]
            done_e += list(e)
            n = b + len(chunk)
            if (b // args.batch) % 20 == 19 or n == len(todo):
                np.savez(part, paths=np.array(done_p), emb=np.stack(done_e))
                rate = n / (time.time() - t0)
                print(f"{len(done_p)}/{len(paths)}  {rate:.1f} img/s  eta {(len(todo) - n) / rate / 60:.0f} min",
                      flush=True)
    np.savez(args.out, paths=np.array(done_p), emb=np.stack(done_e))
    os.remove(part)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
