#!/bin/bash
# The v11 run: v10 with the stray penalty eased to --alpha 3 (v9 2, v10 4) and a lower floor
# (0.01% gain, up to 300 icons) for more icons. Resumable from runs/v11.ckpt.
cd "$(dirname "$0")/../.."
F=$(cat configs/flags_v11.txt)
.venv/bin/python -u src/cover_ranked.py data/hand_cutout.png --emb runs/library_emb_v4.npz --model runs/scorer_v4.pt \
    --eligible runs/eligible.txt --shortlist 256 --reuse 8 --picks 10 --max-images 300 --jobs 15 $F \
    --min-gain 0.0001 --checkpoint runs/v11.ckpt --out runs/v11.png >> runs/v11.log 2>&1
grep -q '^Saved runs/v11.png' runs/v11.log && .venv/bin/python src/render_overlay.py data/hand_cutout.png runs/v11.json runs/v11_render.png >> runs/v11.log 2>&1
