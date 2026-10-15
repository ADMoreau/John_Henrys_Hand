#!/bin/bash
# The v14 run: continues v13 with full passes (--exhaust-rounds 0: a stage ends only when every
# eligible icon has been tried at it and nothing pays): first at v13's last stage (alpha 0.5,
# 18px rms), then alpha 0.25 at 15px and alpha 0 at 11px (0.03 added to the small search scales).
# Resumable from runs/v14.ckpt (seeded from runs/v13.ckpt).
cd "$(dirname "$0")/../.."
F=$(cat configs/flags_v14.txt)
.venv/bin/python -u src/cover_ranked.py data/hand_cutout.png --emb runs/library_emb_v4.npz --model runs/scorer_v4.pt \
    --eligible runs/eligible.txt --shortlist 256 --reuse 8 --picks 20 --pick-frac 0.3 --max-images 5000 --jobs 15 $F \
    --occlusion --min-gain 0.00002 --target 1.0 --stages 2:0.13,1.5:0.1,1:0.08,0.5:0.05,0.25:0.04,0:0.03 \
    --exhaust-rounds 0 --checkpoint runs/v14.ckpt --out runs/v14.png >> runs/v14.log 2>&1
grep -q '^Saved runs/v14.png' runs/v14.log && .venv/bin/python src/render_overlay.py data/hand_cutout.png runs/v14.json runs/v14_render.png >> runs/v14.log 2>&1
