#!/bin/bash
# The v10 run: v9 plus smaller icons (search down to 0.05 primary radii, fits to 0.03), a lower
# floor (0.02% gain, up to 200 icons), stray icon outline penalised twice as hard (--alpha 4), and
# thin outline art excluded (runs/excluded.txt). Resumable from runs/v10.ckpt.
cd "$(dirname "$0")/../.."
F=$(cat configs/flags_v10.txt)
.venv/bin/python -u src/cover_ranked.py data/hand_cutout.png --emb runs/library_emb_v4.npz --model runs/scorer_v4.pt \
    --eligible runs/eligible.txt --shortlist 256 --reuse 8 --picks 10 --max-images 200 --jobs 15 $F \
    --min-gain 0.0002 --checkpoint runs/v10.ckpt --out runs/v10.png >> runs/v10.log 2>&1
grep -q '^Saved runs/v10.png' runs/v10.log && .venv/bin/python src/render_overlay.py data/hand_cutout.png runs/v10.json runs/v10_render.png >> runs/v10.log 2>&1
