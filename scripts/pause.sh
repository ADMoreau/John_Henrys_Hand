#!/bin/bash
# Stop a run started by scripts/resume.sh and free its memory; resume.sh continues it from the
# last finished round (at most the round in progress is lost).  scripts/pause.sh v14
run=${1:?usage: scripts/pause.sh RUN   (e.g. v14)}
root=$(cd "$(dirname "$0")/.." && pwd)
systemctl --user stop "image-fitter-$run.service" 2>/dev/null
systemctl --user reset-failed "image-fitter-$run.service" 2>/dev/null
echo "stopped; $(grep -c -- '    -> ' "$root/runs/$run.log" 2>/dev/null) icons logged, checkpoint: $(ls -l --time-style=+%H:%M "$root/runs/$run.ckpt" 2>/dev/null | awk '{print $6}')"
