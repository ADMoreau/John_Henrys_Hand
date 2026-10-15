#!/bin/bash
# Scorer self-training loop: run the greedy with the current scorer, keep every round's
# shortlist with its true scores (runs/mined), retrain on labels_v3 + mined, re-embed, repeat.
# Resumable: each step is skipped when its output already exists.
set -e
cd "$(dirname "$0")/../.."
F="--warp --warp-lambda 0.00003 --warp-width 0.15 --warp-coarse 1.0,0.5,0.3 --warp-cands 6 --warp-ctrl 64 --warp-reach 0.6 --warp-max-aniso 12 --warp-min-area 0.15 --icon-edges detail --icon-min-run 0.25 --alpha 2"
L=${LIBRARY:?set LIBRARY to the icon library directory}

# cover_ranked sometimes aborts in interpreter teardown (free(): invalid pointer) after it has
# saved everything, so success means "the log says Saved", not exit code 0.
ranked() {  # ranked <tag>
  grep -q '^Saved runs/'$1'.png' runs/$1.log 2>/dev/null && return 0
  .venv/bin/python -u src/cover_ranked.py data/hand_cutout.png --emb $emb --model $model --eligible runs/eligible.txt \
      --shortlist 64 --max-images 12 --jobs 14 $F --min-gain 0.002 --save-labels runs/mined \
      --out runs/$1.png > runs/$1.log 2>&1 || true
  grep -q '^Saved runs/'$1'.png' runs/$1.log
  .venv/bin/python src/render_overlay.py data/hand_cutout.png runs/$1.json runs/$1_render.png
}

model=runs/scorer_v3.pt; emb=runs/library_emb_v3.npz
for it in 1 2 3; do
  ranked al_it$it
  [ -f runs/scorer_m$it.pt ] || .venv/bin/python -u src/scorer.py train runs/labels_v3,runs/mined --clip-low -5 \
      --epochs 30 --out runs/scorer_m$it.pt > runs/scorer_m$it.log 2>&1
  [ -f runs/library_emb_m$it.npz ] || .venv/bin/python -u src/scorer.py embed $L --model runs/scorer_m$it.pt \
      --out runs/library_emb_m$it.npz --cache runs/icon_rasters_detail --icon-edges detail > runs/library_emb_m$it.log 2>&1
  model=runs/scorer_m$it.pt; emb=runs/library_emb_m$it.npz
  echo "== iteration $it: $(grep -c -- '->' runs/al_it$it.log) icons, $(grep 'weighted coverage' runs/al_it$it.log)"
done
ranked al_final
echo "== final: $(grep 'weighted coverage' runs/al_final.log)"
for f in runs/al_it*.log runs/al_final.log; do echo "$f: $(grep -o 'scorer rank [0-9]*' $f | tr '\n' ' ')"; done
