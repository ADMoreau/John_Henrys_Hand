#!/bin/bash
# The v7 run: v6 with non-rigid warping off (same library, shortlist, reuse, min-gain), to measure
# what warping is worth end to end. Resumable: starting it again continues from runs/v7.ckpt.
# No --save-labels: these no-warp scores would not match the scorer's warped training labels.
cd "$(dirname "$0")/../.."
F=$(cat configs/flags_v7_nowarp.txt)
.venv/bin/python -u src/cover_ranked.py data/hand_cutout.png --emb runs/library_emb_v4.npz --model runs/scorer_v4.pt \
    --eligible runs/eligible.txt --shortlist 256 --reuse 8 --max-images 30 --jobs 15 $F --min-gain 0.002 \
    --checkpoint runs/v7.ckpt --out runs/v7.png >> runs/v7.log 2>&1
grep -q '^Saved runs/v7.png' runs/v7.log && .venv/bin/python src/render_overlay.py data/hand_cutout.png runs/v7.json runs/v7_render.png >> runs/v7.log 2>&1
