#!/bin/bash
# Start (or continue) a run in the background as a systemd user service, so it survives the
# terminal closing (not a full logout):  scripts/resume.sh v14  -> scripts/runs/run_v14.sh
# Runs with --checkpoint pick up from their last finished round.
set -e
run=${1:?usage: scripts/resume.sh RUN   (e.g. v14)}
root=$(cd "$(dirname "$0")/.." && pwd)
[ -x "$root/scripts/runs/run_$run.sh" ] || { echo "no scripts/runs/run_$run.sh"; exit 1; }
systemctl --user reset-failed "image-fitter-$run.service" 2>/dev/null || true
systemd-run --user --unit="image-fitter-$run" --working-directory="$root" "$root/scripts/runs/run_$run.sh"
