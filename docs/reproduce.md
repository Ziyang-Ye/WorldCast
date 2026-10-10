# Reproducing Table 3

The WorldCast row of Table 3 renders 32 rounds of 3 clients each, with recorded player states and GT visibility labels.
The clients of a round run in lockstep through a shared directory, one GPU each. The defaults are the settings of the
paper's Table-3 runs; `--round-length` gives every round its recorded length (at most 120 s, `--max-seconds`), as
Table 3 ran it.

## Run the rounds

```bash
mkdir -p data && cp examples/data_paths.yaml data/paths.yaml     # then point it at the data
python tools/run_session.py --config weights/paths.yaml --config data/paths.yaml \
    --all-rounds --round-length --gpus 0,1,2 --out-dir runs/table3
python tools/decode.py --config weights/paths.yaml --latents runs/table3/*/*/latents.npy
```

`data/paths.yaml` names the round index (96 rows, one per client), the media index, the tick tables, the first latents
and the two label sets. The files are listed in [docs/inference.md](inference.md#data), their formats are in
[docs/data.md](data.md).

`tools/run_session.py` runs the rounds one after the other, each round's three clients on GPUs 0, 1 and 2. It writes,
per round, `runs/table3/<round>/<media_id>/{latents.npy,client.json,client.log}` for every client and the shared world
state in `runs/table3/<round>/world_state/`. `tools/decode.py` writes a `video.mp4` next to each `latents.npy`.

- `--round-of INDEX_ROW` runs the round of one index row.
- `tools/run_client.py` runs one client, so the clients of a round can run on several machines that share a file
  system. Their environment: [examples/README.md](../examples/README.md#render).

## Bit-exact reproduction

The reference runs ran on this stack:

| | reference stack |
|---|---|
| GPU | NVIDIA H20 |
| torch | 2.9.1+cu128 |
| flash-attn | 2.8.3 (flash-attention 2): `model.attention: flash`, the default |
| CPU | x86-64 (the entry noise is drawn on the CPU) |

On it the latents equal those of the paper's runs bit for bit, for the clients whose runs recorded fingerprints: nine
Table-3 clients, the rows 57-59, 75-77 and 90-92 of the paper's 96-row round index
(`examples/table3_fingerprints.json`), and the clients of the example cases (`examples/manifest.json`). A fingerprint
is the first 32 hex digits of the sha256 of a float32 tensor: of the first frame (latent 0 in bf16), of latents 0-24
and of all latents; the Table-3 runs also recorded the entry noise. The fingerprints are of the latents, so compare
`latents.npy`, not the mp4. What the latents depend on, point by point: [docs/inference.md](inference.md#numerics).

`tools/verify_reference.py` compares a run with them:

```bash
python tools/verify_reference.py check runs/table3                  # every client under runs/table3 with a reference run
python tools/verify_reference.py client --config weights/paths.yaml --config data/paths.yaml \
    --index-row 57 --out-dir runs/verify/client                     # one GPU: one client through latent 24
```

`check` prints one line per fingerprint and how many clients match; `client` runs the client of one index row through
latent 24, compares its first frame, its latents 0-24 and its entry noise, and writes `latents_0_24.npy` and
`report.json` to `--out-dir`. Both exit with status 1 on any difference.
