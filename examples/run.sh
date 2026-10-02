#!/usr/bin/env bash
# Render one example case (examples/README.md): check its inputs, run its clients in lock-step, decode them and tile
# them into one grid video.
#
#   bash examples/run.sh <case> [tools/run_session.py options, e.g. --gpus 4,5,6]
#
# WEIGHTS: the paths file of tools/download_weights.py (default weights/paths.yaml). OUT: the output directory
# (default runs/examples/<case>); it must not exist yet, since every session starts from a fresh pool.
set -euo pipefail
cd "$(dirname "$0")/.."

CASE=${1:?usage: bash examples/run.sh <case> [run_session.py options]}
shift
DATA=examples/data/$CASE
WEIGHTS=${WEIGHTS:-weights/paths.yaml}
OUT=${OUT:-runs/examples/$CASE}
PYTHON=${PYTHON:-python}

[ -f "$DATA/config.yaml" ] || { echo "unknown case '$CASE' (examples/README.md lists them)" >&2; exit 1; }
[ -f "$WEIGHTS" ] || { echo "$WEIGHTS not found: python tools/download_weights.py --out-dir weights" >&2; exit 1; }
[ ! -e "$OUT" ] || { echo "$OUT exists: remove it or set OUT" >&2; exit 1; }

"$PYTHON" tools/check_examples.py "$CASE"
"$PYTHON" tools/run_session.py --config configs/infer/worldcast_4step.yaml --config "$WEIGHTS" \
    --config "$DATA/config.yaml" --all-groups --out-dir "$OUT" "$@"
"$PYTHON" tools/decode.py --config "$WEIGHTS" --latents "$OUT"/*/*/latents.npy
PYTHON="$PYTHON" bash examples/grid.sh "$OUT/grid.mp4" "$OUT"/*/*/video.mp4
