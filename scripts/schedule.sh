#!/bin/bash
# Start a run at a set time (systemd timer):  scripts/schedule.sh v14 '2026-10-15 01:00:00'
# Cancel with: systemctl --user stop image-fitter-RUN-start.timer
set -e
run=${1:?usage: scripts/schedule.sh RUN 'YYYY-MM-DD HH:MM:SS'}
when=${2:?usage: scripts/schedule.sh RUN 'YYYY-MM-DD HH:MM:SS'}
root=$(cd "$(dirname "$0")/.." && pwd)
systemd-run --user --unit="image-fitter-$run-start" --on-calendar="$when" "$root/scripts/resume.sh" "$run"
