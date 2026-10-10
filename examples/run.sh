#!/usr/bin/env bash
# Render one example case (examples/README.md): check its inputs, run its clients in lockstep, decode them and tile
# them into one grid video, then compare the latents with the reference run's (GT player states).
#
#   bash examples/run.sh <case> [tools/run_session.py options, e.g. --gpus 4,5,6]
#
# WEIGHTS: the paths file of tools/download_weights.py (default weights/paths.yaml). DATA: the --out-dir of
# tools/download_examples.py (default examples). OUT: the output directory (default runs/examples/<case>); it must
# not exist yet: every session starts on a fresh world state. PYTHON: the interpreter (default python).
# SHOWCASE=1 also writes OUT/showcase.mp4 (tools/make_showcase.py): the views with every other player's GT box and
# a map of the ten players. The summary gives the frames each of the two decodes to out of the session's, 1 + 4 (N - 1)
# for the N latents of its shortest client. A video that does not decode without a decoder error to every frame encoded
# is not written and fails the run, and so does one short of the session's frames (e.g. a client's video.mp4 that is
# short). A difference from the reference run is reported, not fatal. With --set player_state.source=predicted (the
# closed loop) the latents are not compared: the reference runs are on the GT player states.
set -euo pipefail
cd "$(dirname "$0")/.."

CASE=${1:?usage: bash examples/run.sh <case> [run_session.py options]}
shift
DATA=${DATA:-examples}
CONFIG=$DATA/data/$CASE/config.yaml
WEIGHTS=${WEIGHTS:-weights/paths.yaml}
OUT=${OUT:-runs/examples/$CASE}
PYTHON=${PYTHON:-python}

[ -f "$CONFIG" ] || { echo "$CONFIG not found: python tools/download_examples.py --out-dir $DATA $CASE" >&2; exit 1; }
[ -f "$WEIGHTS" ] || { echo "$WEIGHTS not found: python tools/download_weights.py --out-dir weights" >&2; exit 1; }
[ ! -e "$OUT" ] || { echo "$OUT exists: remove it or set OUT" >&2; exit 1; }
if [ "${SHOWCASE:-}" = 1 ] && ! "$PYTHON" -c "import PIL" 2>/dev/null; then
  echo "SHOWCASE=1: the showcase is drawn with Pillow: pip install -e \".[showcase]\"" >&2
  exit 1
fi

"$PYTHON" tools/check_examples.py "$CONFIG"
# the lockstep wait polls the world state every 50 ms instead of 2 s: latency only, the latents are the same
"$PYTHON" tools/run_session.py --config "$WEIGHTS" --config "$CONFIG" \
    --all-rounds --out-dir "$OUT" --set world_state.poll_s=0.05 "$@"

# Decode on the session's GPUs, the GPUs at once: each client on its GPU, of --gpus as run_session.py reads it, else
# of the visible GPUs in turn as run_session.py places the clients (without one, on the CPU: no GPU, one decode); on
# each GPU tools/decode.py loads the VAE once and decodes its clients one after the other
GPUS= DEVICE= SOURCE=gt
args=("$@")
for ((i = 0; i < $#; i++)); do
  case ${args[i]} in
    --gpus) GPUS=${args[i + 1]:-} ;;
    --gpus=*) GPUS=${args[i]#--gpus=} ;;
    --device) DEVICE=${args[i + 1]:-} ;;
    --device=*) DEVICE=${args[i]#--device=} ;;
    --set) case ${args[i + 1]:-} in player_state.source=*) SOURCE=${args[i + 1]#player_state.source=} ;; esac ;;
    --set=player_state.source=*) SOURCE=${args[i]#--set=player_state.source=} ;;
  esac
done
GPUS=${GPUS// /}
[ -n "$GPUS" ] || GPUS=$("$PYTHON" -c \
    'from worldcast.engine.inference.session import visible_gpus; print(",".join(visible_gpus()))')
IFS=, read -r -a gpus <<< "$GPUS"
latents=("$OUT"/*/*/latents.npy)
placed=()
for ((k = 0; k < ${#latents[@]}; k++)); do
  if [ ${#gpus[@]} -gt 0 ]; then placed+=("${gpus[k % ${#gpus[@]}]}"); else placed+=(""); fi
done
pids=() started=,
for ((k = 0; k < ${#latents[@]}; k++)); do
  gpu=${placed[k]}
  case $started in *,"$gpu",*) continue ;; esac
  started=$started$gpu,
  on_gpu=()
  for ((j = k; j < ${#latents[@]}; j++)); do
    [ "${placed[j]}" != "$gpu" ] || on_gpu+=("${latents[j]}")
  done
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" tools/decode.py --config "$WEIGHTS" \
      ${DEVICE:+--device "$DEVICE"} --latents "${on_gpu[@]}" &
  pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo "a client's decode failed: see above" >&2; exit 1; }

# The session's frames: 1 + 4 (N - 1) for the N latents of its shortest client (the grid ends with its video)
frames=$("$PYTHON" -c 'import sys
import numpy as np
from worldcast.data.latents import video_frame_count
print(video_frame_count(min(len(np.load(path, mmap_mode="r")) for path in sys.argv[1:])))' "${latents[@]}")
# grid.sh's and make_showcase.py's line, "VIDEO decodes to F frames: ...", with the session's frames; F short of them
# fails the run (each tool fails on its own when its video does not decode to every frame it encoded)
of_session() {
  local video=${1%% decodes to *} decoded=${1#* decodes to }
  decoded=${decoded%% frames: *}
  if [ "$decoded" != "$frames" ]; then
    echo "run.sh: $video decodes to $decoded of the session's $frames frames (e.g. a client's video.mp4 is short)" >&2
    return 1
  fi
  echo "$video decodes to $decoded of the session's $frames frames: ${1#* frames: }"
}
line=$(PYTHON="$PYTHON" bash examples/grid.sh "$OUT/grid.mp4" "$OUT"/*/*/video.mp4)
grid=$(of_session "$line")
showcase=
if [ "${SHOWCASE:-}" = 1 ]; then
  line=$("$PYTHON" tools/make_showcase.py --config "$CONFIG" --session "$OUT" --out "$OUT/showcase.mp4")
  showcase=$(of_session "$line")
fi

echo
echo "$CASE: each client's latents.npy, video.mp4 and client.json in $OUT/<round>/<media_id>/"
echo "$grid"
[ -z "$showcase" ] || echo "$showcase"
status=0
if [ "$SOURCE" = predicted ]; then
  echo "player_state.source=predicted: the latents are not compared with the reference runs' (GT player states)"
else
  "$PYTHON" tools/verify_reference.py check "$OUT" || status=$?
fi
case $status in
  0) ;;
  1) echo "(the reference stack: docs/reproduce.md, \"Bit-exact reproduction\")" ;;
  *) exit "$status" ;;
esac
