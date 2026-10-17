#!/bin/bash
# The v15 run: continues v14 after its canvas was rebuilt with the fixed renderer footprints and
# drawn-only credit (src/rebuild_canvas.py runs/v14.ckpt runs/v14p.ckpt ...), with a much lower
# gain floor: a full pass at alpha 0 / 11px, then one at 7px rms (0.02 added to the small search
# scales). Renders at 1x and 3x. Resumable from runs/v15.ckpt (seeded from runs/v14p.ckpt).
cd "$(dirname "$0")/../.."
F=$(cat configs/flags_v15.txt)
.venv/bin/python -u src/cover_ranked.py data/hand_cutout.png --emb runs/library_emb_v4.npz --model runs/scorer_v4.pt \
    --eligible runs/eligible.txt --shortlist 256 --reuse 8 --picks 20 --pick-frac 0.3 --max-images 5000 --jobs 15 $F \
    --occlusion --min-gain 0.000001 --target 1.0 --stages 0:0.03,0:0.02 \
    --exhaust-rounds 0 --checkpoint runs/v15.ckpt --out runs/v15.png >> runs/v15.log 2>&1
grep -q '^Saved runs/v15.png' runs/v15.log && .venv/bin/python src/render_overlay.py data/hand_cutout.png runs/v15.json runs/v15_render.png >> runs/v15.log 2>&1 \
    && .venv/bin/python src/render_overlay.py data/hand_cutout.png runs/v15.json runs/v15_render_x3.png --scale 3 >> runs/v15.log 2>&1
