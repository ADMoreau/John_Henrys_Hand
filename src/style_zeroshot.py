#!/usr/bin/env python3
"""Zero-shot style labels from SigLIP embeddings (see style_embed.py).

  python src/style_zeroshot.py runs/siglip.npz runs/style_labels.npz

Each image gets a softmax over the prompt classes below (groups of prompts, scores averaged
over the group). cover_ranked.py --style drops 'glow' images (flares, light, smoke, fire)
and text logos; everything else is eligible.
"""
import sys
import numpy as np

CLASSES = {
    "icon": ["a cartoon illustration", "a clip art drawing", "an icon", "a cartoon character",
             "a line drawing", "a flat vector illustration", "an engraving illustration",
             "a flat icon", "a logo symbol", "a badge emblem", "a 3D cartoon character render",
             "a silhouette", "an emoji", "a symbol", "a toy figure"],
    "photo": ["a photograph", "a photo of a real object", "a cut-out photo", "a realistic photo"],
    "texture": ["water splash", "a texture", "particles", "an abstract pattern", "clouds",
                "dust", "a spray of droplets"],
    "glow": ["a lens flare", "a light glow effect", "smoke", "fire flames", "a light beam",
             "sparkles", "a blurry shape"],
    "text": ["a text logo", "a word written in a font", "a company wordmark", "text"],
}
MODEL = "google/siglip-base-patch16-224"


def text_emb():
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(MODEL), AutoModel.from_pretrained(MODEL).eval()
    out = {}
    for k, ps in CLASSES.items():
        t = tok(ps, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            e = model.get_text_features(**t)
            if not torch.is_tensor(e):
                e = e.pooler_output
        out[k] = torch.nn.functional.normalize(e, dim=-1).numpy()
    return out


def classify(E):
    T = text_emb()
    E = E.astype(np.float32)
    logit = np.stack([(E @ T[k].T).mean(1) for k in CLASSES], 1) * 100
    p = np.exp(logit - logit.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


if __name__ == "__main__":
    z = np.load(sys.argv[1])
    P = classify(z["emb"])
    np.savez(sys.argv[2], paths=z["paths"], prob=P, classes=np.array(list(CLASSES)))
    lab = P.argmax(1)
    for i, k in enumerate(CLASSES):
        print(f"{k:8s} {np.mean(lab == i):.1%}")
