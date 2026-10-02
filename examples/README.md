# Examples

Six recorded Counter-Strike 2 rounds to render with the released weights, at least one per map. Every client renders
one player's view from that player's recorded controls and the recorded positions of all ten players (GT player
states, as in the paper's Table 3). The clients of a round run in lock-step, one GPU each, and share their generated
blocks.

| case | map | match, round | clients | length | GPUs | preview |
|---|---|---|---|---|---|---|
| `dust2_r09` | Dust2 | 2392774, r09 | 5 (T: p05-p09) | 20 s | 5 | <img src="assets/dust2_r09.jpg" width="360"> |
| `dust2_r16` | Dust2 | 2392809, r16 | 3 (CT: p06-p08) | 30 s | 3 | <img src="assets/dust2_r16.jpg" width="360"> |
| `mirage_r16` | Mirage | 2393226, r16 | 3 (T: p00-p02) | 30 s | 3 | <img src="assets/mirage_r16.jpg" width="360"> |
| `nuke_r07` | Nuke | 2393223, r07 | 4 (T: p05-p08) | 20 s | 4 | <img src="assets/nuke_r07.jpg" width="360"> |
| `ancient_r18` | Ancient | 2393400, r18 | 4 (T: p00-p02, p04) | 20 s | 4 | <img src="assets/ancient_r18.jpg" width="360"> |
| `ancient_r02` | Ancient | 2392968, r02 | 10 (all) | 60 s | 10 | <img src="assets/ancient_r02.jpg" width="360"> |

A client takes about 15 GiB of GPU memory on an NVIDIA H20 (the paper's GPU), so one GPU can run two clients: the
GPUs column is one GPU per client, but `bash examples/run.sh ancient_r02 --gpus 0,1,2,3,4,5,6,7,0,1` runs the ten
clients of `ancient_r02` on eight GPUs with the same latents.

## Get the data

After [Install and Download weights](../README.md#install):

```bash
python tools/download_examples.py     # all six cases, 200 MB; or name some: ... mirage_r16 nuke_r07
python tools/check_examples.py        # CPU only: every client's inputs load
```

The files come from the WorldCast repository on Hugging Face (`weights.hf_repo_id`, as for the weights). A case
holds the files of [docs/data.md](../docs/data.md) under `examples/data/<case>/`: `round_index.jsonl` (one row per
client), `media_index.jsonl` (all ten players), `opencs2/` (their tick tables), `first_latents/`, `vislabels/`,
`obslabels/`, and `config.yaml` with these paths and the length. The recorded videos of the rounds are in OpenCS2
(`video_path` in `media_index.jsonl`).

## Render

```bash
bash examples/run.sh dust2_r09
bash examples/run.sh dust2_r16
bash examples/run.sh mirage_r16
bash examples/run.sh nuke_r07
bash examples/run.sh ancient_r18
bash examples/run.sh ancient_r02
```

`run.sh` checks the inputs, runs the clients with `tools/run_session.py` on GPUs 0, 1, ..., decodes them with
`tools/decode.py` and tiles them with ffmpeg into `runs/examples/<case>/grid.mp4` (clients in slot order, left to
right and top to bottom). Each client's `latents.npy` and `video.mp4` are in `runs/examples/<case>/<round>/<media_id>/`.
Arguments after the case go to `run_session.py`, e.g. `--gpus 4,5,6` or `--attention sdpa`; `WEIGHTS` and `OUT` set
the weights paths file and the output directory. A session needs a fresh pool: remove the output directory before
running a case again.

Ten clients on two machines that share a file system: start five clients on each with `tools/run_client.py` and
one pool directory, then decode and tile on either machine.

```bash
P=runs/examples/ancient_r02                     # on the shared file system
for i in 0 1 2 3 4; do                          # on the second machine: 5 6 7 8 9
  CUDA_VISIBLE_DEVICES=$((i % 5)) python tools/run_client.py --config configs/infer/worldcast_4step.yaml \
      --config weights/paths.yaml --config examples/data/ancient_r02/config.yaml \
      --index-row $i --live-pool-dir $P/pool --out-dir $P/p$i &
done; wait

python tools/decode.py --config weights/paths.yaml --latents $P/p?/latents.npy
bash examples/grid.sh $P/grid.mp4 $P/p?/video.mp4
python tools/check_examples.py ancient_r02 --run-dir $P
```

## Expected output

`examples/expected/<case>.mp4`, downloaded with the data, is the reference render of the case, made with the paper's
weights and settings, in the layout of `grid.mp4`. On the paper's stack (NVIDIA H20, torch 2.9.1+cu128 and
flash-attn 2.8.3) the latents equal the reference run's; this compares them with its recorded fingerprints (the
plain prefix of every client, and all latents where the reference run had the case's length):

```bash
python tools/check_examples.py dust2_r09 --run-dir runs/examples/dust2_r09
```

On other GPUs or attention kernels the frames drift apart over the round but should look alike.
