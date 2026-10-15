#!/bin/bash
# After rescore.py (labels_v4r, mined_v4) and the 128px raster cache: train the 128/256 scorer,
# embed the library, run the 12-icon search under configs/flags_v4.txt, render.
set -e
cd "$(dirname "$0")/../.."
F=$(cat configs/flags_v4.txt)
L=${LIBRARY:?set LIBRARY to the icon library directory}
while pgrep -f '^.venv/bin/python -u src/rescore[.]py' > /dev/null || ! grep -q '^embedded' runs/cache128.log; do sleep 60; done
[ -f runs/scorer_v4.log ] && grep -q '^best epoch' runs/scorer_v4.log || \
  .venv/bin/python -u src/scorer.py train runs/labels_v4r,runs/mined_v4 --clip-low -5 --epochs 30 \
      --icon-res 128 --state-res 256 --aug-stretch 2 --val-from al_it3 --threads 14 --batch-states 4 --max-icons 64 \
      --out runs/scorer_v4.pt > runs/scorer_v4.log 2>&1
[ -f runs/library_emb_v4.npz ] || .venv/bin/python -u src/scorer.py embed $L --model runs/scorer_v4.pt \
    --out runs/library_emb_v4.npz --cache runs/icon_rasters_detail128 --icon-edges detail > runs/library_emb_v4.log 2>&1
# a scorer that still can't order real shortlists gets a wider net
rho=$(grep -o 'spearman [-0-9.]*' runs/scorer_v4.log | tail -1 | awk '{print $2}')
sl=$(awk -v r="$rho" 'BEGIN{print (r < 0.3) ? 256 : 64}')
echo "val spearman $rho -> shortlist $sl"
grep -q '^Saved runs/v4.png' runs/v4.log 2>/dev/null || \
  .venv/bin/python -u src/cover_ranked.py data/hand_cutout.png --emb runs/library_emb_v4.npz --model runs/scorer_v4.pt \
      --eligible runs/eligible.txt --shortlist $sl --max-images 12 --jobs 15 $F --min-gain 0.002 \
      --save-labels runs/mined_v4 --out runs/v4.png > runs/v4.log 2>&1 || true
grep -q '^Saved runs/v4.png' runs/v4.log
.venv/bin/python src/render_overlay.py data/hand_cutout.png runs/v4.json runs/v4_render.png
grep 'weighted coverage' runs/v4.log
