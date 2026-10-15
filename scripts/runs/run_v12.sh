#!/bin/bash
# The v12 run: v11 flags plus occlusion-aware coverage (--occlusion: only outline left visible
# in the render counts, and an icon pays for the covered outline it hides), big icons held to an
# 8 px match tolerance (--max-tol 0.0217), and no practical icon cap (0.002% floor, 1500 icons,
# up to 20 picks per round). Resumable from runs/v12.ckpt.
cd "$(dirname "$0")/../.."
F=$(cat configs/flags_v12.txt)
.venv/bin/python -u src/cover_ranked.py data/hand_cutout.png --emb runs/library_emb_v4.npz --model runs/scorer_v4.pt \
    --eligible runs/eligible.txt --shortlist 256 --reuse 8 --picks 20 --pick-frac 0.3 --max-images 1500 --jobs 15 $F \
    --occlusion --min-gain 0.00002 --checkpoint runs/v12.ckpt --out runs/v12.png >> runs/v12.log 2>&1
grep -q '^Saved runs/v12.png' runs/v12.log && .venv/bin/python src/render_overlay.py data/hand_cutout.png runs/v12.json runs/v12_render.png >> runs/v12.log 2>&1
