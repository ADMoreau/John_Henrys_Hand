#!/bin/bash
# The v13 run: continues v12 (occlusion-aware, 8px cap) and, each time icons stop paying (2 rounds
# in a row with nothing above the floor, never re-trying an icon within a stage), steps down to a
# lower stray penalty with a smaller size cap: alpha:max_scale 2:0.13, 1.5:0.1, 1:0.08, 0.5:0.05.
# Resumable from runs/v13.ckpt (seeded from runs/v12.ckpt).
cd "$(dirname "$0")/../.."
F=$(cat configs/flags_v13.txt)
.venv/bin/python -u src/cover_ranked.py data/hand_cutout.png --emb runs/library_emb_v4.npz --model runs/scorer_v4.pt \
    --eligible runs/eligible.txt --shortlist 256 --reuse 8 --picks 20 --pick-frac 0.3 --max-images 3000 --jobs 15 $F \
    --occlusion --min-gain 0.00002 --stages 2:0.13,1.5:0.1,1:0.08,0.5:0.05 --exhaust-rounds 2 \
    --checkpoint runs/v13.ckpt --out runs/v13.png >> runs/v13.log 2>&1
grep -q '^Saved runs/v13.png' runs/v13.log && .venv/bin/python src/render_overlay.py data/hand_cutout.png runs/v13.json runs/v13_render.png >> runs/v13.log 2>&1
