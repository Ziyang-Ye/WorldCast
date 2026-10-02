#!/usr/bin/env bash
# Tile client videos into one grid video, in the order given: up to five per row (four as 2 x 2).
#
#   bash examples/grid.sh OUT.mp4 client1.mp4 client2.mp4 ...
#
# Every tile keeps the clients' native 672x384; the grid ends with the shortest input. FFMPEG names the binary
# (default: the one of imageio-ffmpeg, a package dependency and a build with libx264; else ffmpeg on PATH, which
# must have libx264: LGPL builds, e.g. conda's or some distributions', stop with "Unrecognized option 'preset'").
# CRF sets the x264 quality (default 23).
set -euo pipefail

OUT=${1:?usage: bash examples/grid.sh OUT.mp4 client.mp4 ...}
shift
N=$#
[ "$N" -ge 1 ] || { echo "grid.sh: no input videos" >&2; exit 1; }
if [ -z "${FFMPEG:-}" ]; then
  FFMPEG=$("${PYTHON:-python}" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null \
           || command -v ffmpeg) || { echo "grid.sh: no ffmpeg (pip install imageio-ffmpeg, or set FFMPEG)" >&2; exit 1; }
fi

if [ "$N" -eq 4 ]; then COLS=2; elif [ "$N" -le 5 ]; then COLS=$N; else COLS=5; fi
ROWS=$(( (N + COLS - 1) / COLS ))

inputs=()
for v in "$@"; do inputs+=(-i "$v"); done
filter=""
for ((r = 0; r < ROWS; r++)); do
  row=""
  for ((c = 0; c < COLS; c++)); do
    i=$(( r * COLS + c ))
    if [ "$i" -lt "$N" ]; then
      filter+="[$i:v]setpts=PTS-STARTPTS,scale=672:384[t$i];"
    else  # an empty tile at the end of the last row
      filter+="color=c=black:s=672x384:r=16[t$i];"
    fi
    row+="[t$i]"
  done
  if [ "$COLS" -gt 1 ]; then filter+="${row}hstack=inputs=$COLS:shortest=1[r$r];"; else filter+="${row}null[r$r];"; fi
done
if [ "$ROWS" -gt 1 ]; then
  for ((r = 0; r < ROWS; r++)); do filter+="[r$r]"; done
  filter+="vstack=inputs=$ROWS:shortest=1[out]"
else
  filter+="[r0]null[out]"
fi

"$FFMPEG" -loglevel error -y "${inputs[@]}" -filter_complex "$filter" -map "[out]" \
  -c:v libx264 -preset medium -crf "${CRF:-23}" -pix_fmt yuv420p -r 16 -movflags +faststart "$OUT"
echo "grid: $OUT ($N clients, ${COLS}x${ROWS})"
