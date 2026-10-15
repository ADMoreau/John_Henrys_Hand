#!/bin/bash
# The v8 run: thinned primary (one line per stroke, 3px point spacing) and small icons
# (--fine-ref: icons below 0.3 primary radii are searched and held to a tighter tolerance),
# up to 10 icons per round, down to 0.05% gain, up to 100 icons. Resumable from runs/v8.ckpt.
cd "$(dirname "$0")/../.."
F=$(cat configs/flags_v8.txt)
.venv/bin/python -u src/cover_ranked.py data/hand_cutout.png --emb runs/library_emb_v4.npz --model runs/scorer_v4.pt \
    --eligible runs/eligible.txt --shortlist 256 --reuse 8 --picks 10 --max-images 100 --jobs 15 $F \
    --min-gain 0.0005 --checkpoint runs/v8.ckpt --out runs/v8.png >> runs/v8.log 2>&1
grep -q '^Saved runs/v8.png' runs/v8.log && .venv/bin/python src/render_overlay.py data/hand_cutout.png runs/v8.json runs/v8_render.png >> runs/v8.log 2>&1
