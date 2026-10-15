#!/bin/bash
# The v6 run (30 icons, no letters/numbers), resumable: stopping it loses at most the round in
# progress; starting it again continues from runs/v6.ckpt.
cd "$(dirname "$0")/../.."
F=$(cat configs/flags_v4.txt)
.venv/bin/python -u src/cover_ranked.py data/hand_cutout.png --emb runs/library_emb_v4.npz --model runs/scorer_v4.pt \
    --eligible runs/eligible.txt --shortlist 256 --reuse 8 --max-images 30 --jobs 15 $F --min-gain 0.002 \
    --save-labels runs/mined_v4 --checkpoint runs/v6.ckpt --out runs/v6.png >> runs/v6.log 2>&1
grep -q '^Saved runs/v6.png' runs/v6.log && .venv/bin/python src/render_overlay.py data/hand_cutout.png runs/v6.json runs/v6_render.png >> runs/v6.log 2>&1
