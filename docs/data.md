# Data

A client renders one player of one recorded Counter-Strike 2 round from the [OpenCS2 dataset](https://huggingface.co/datasets/blanchon/opencs2_dataset)
(CC BY 4.0). It never reads the recorded video: it reads the recorded engine state and controls of all ten players
(the oracle player states of the paper's Table 3), the player's GT visibility labels, two observer-signal curves and
the first latent of the round. Every path comes from the config (`paths.*`); a missing file is an error.

## What a client reads

| artefact | config key | file | used for |
|---|---|---|---|
| round index | `paths.round_index` (+ `run.index_row`, counted from 0) | JSONL, one row per client | which player, which start frame, the round's other clients (lock-step peers) |
| media index | `paths.media_index` | JSONL, one row per player-round | media id -> tick file, player slots of a round, frame count |
| tick tables | `paths.dataset_root` | `rounds/match_id=<m>/map_name=<map>/round=<rr>/player=<pp>/ticks.parquet` | states, controls, substeps, team of all ten players |
| first latents | `paths.latent_cache_root` | `<media_id>.npz`, member `win_<start:06d>` | latent 0 = the window's sink |
| visibility labels | `paths.visibility_label_root` | `<media_id>.npz` | GT visibility of the other players (gates the player state field) |
| observer signals | `paths.obs_signal_label_root` | `flashlabels/<media_id>.npz`, `scopelabels/<media_id>.npz` | flash and scope inputs of the generator; scoped field of view |

Example `data/paths.yaml` (a copy of `examples/data_paths.yaml`; pass it as an extra `--config`):

```yaml
paths:
  round_index: data/wholeround_index.jsonl
  media_index: data/media_index.jsonl
  dataset_root: data/opencs2           # holds rounds/...
  latent_cache_root: data/first_latents
  visibility_label_root: data/vislabels
  obs_signal_label_root: data/obslabels  # holds flashlabels/ and scopelabels/
```

## Formats

**Round index** (`wholeround_index.jsonl`, 96 rows for the paper: 32 rounds x 3 clients). Fields read:

```json
{"media_id": "2392812-de_dust2-r09-p09", "start_frame": 0, "match_id": 2392812, "round": 9,
 "map_name": "de_dust2", "latent_key": "win_000000", "player_slot": 9,
 "group_media": ["2392812-de_dust2-r09-p05", "2392812-de_dust2-r09-p08", "2392812-de_dust2-r09-p09"],
 "group_slots": [5, 8, 9]}
```

`media_id` = `<match>-<map>-r<round>-p<slot>`. `start_frame` is a source frame at 32 fps; `latent_key` must equal
`win_<start_frame:06d>`. `group_media` lists the round's rendered clients (this one included): its peers are the
others. A row whose `group_media` is only itself runs alone (no lock-step). Rows are numbered from 0 in file order
(`run.index_row`, `--index-row`, `--group-of`). `tools/run_session.py --round-length` also reads `round_seconds`
(the round's recorded length in seconds) and requests `1 + 4 * 10 * floor(min(round_seconds, 120) / 10)` latents
(the cap is `--max-seconds`). Other fields are ignored.

**Media index**: `media_id, match_id, map_name, round, player_slot, fps (32), video_frames, ticks_path, ticks_rows,
ticks_file_size, ticks_sha256` (optional), `capture_start_tick` (optional; equal within a round), `video_path`,
`player_side`. A round slot without media loads as an absent player (dead, silent, team 0).

**Tick tables** (64 Hz, one per player, all starting at the round's capture start so `t = 0` is the same instant
for every player): `tick, t` (s), `x, y, z` (feet, engine units, 1 u = 0.0254 m, +z up), `yaw, pitch` (degrees,
pitch > 0 looks down), `is_alive`, `active` (held buttons), `delta_pitch, delta_yaw` (degrees per tick),
`input_weapon`, `team_num` (2 T, 3 CT). File size and, when the index carries it, sha256 must match the media row;
the digest also keys the jump-button recall, so a re-exported parquet with other bytes is refused.

**First latents**: `<media_id>.npz`, member `win_<start:06d>` float16 `[1, 41, 48, 24, 42]` (Wan2.2 VAE latents of
161 frames at 384x672). Only latent 0 is read (widened to fp32, cast to bf16 by the client). The builder of the
paper's cache is lost and a re-encode is not bit-exact, so the release ships latent 0 of each client.

**Visibility labels**: `<media_id>.npz` with `_binary_visible`, `_binary_eval_valid`, `_in_frustum`,
`_binary_offscreen`, each `[10, video_frames]` bool at 32 fps (engine line-of-sight tests). Visible means
confirmed visible; occluded and unknown both read as not visible. A latent counts a player as visible if any of its
four pixel frames does.

**Observer signals**: `flashlabels/<media_id>.npz` (`lum` float16 per video frame, `stride`, `hot_threshold`
0.85) and `scopelabels/<media_id>.npz` (`scoped_vis`, `level` 0/1/2, ... per video frame). Per latent: flash = any
of its frames brighter than the threshold; scope on/level at its last frame; `valid = 0` where a curve cannot be
sampled. They were computed from the recorded video by scripts of the paper's research code, which are not part of
this release. A missing file is an error (the research code silently read it as "unknown").

## How a window is built (`worldcast/data`)

- Latents `N` = `run.latents` (441 = 110 s), clipped to the observer's tick coverage and to `run.max_blocks`, then
  rounded down to `1 + 4k`; at least 29 (the 25-latent plain prefix plus one block).
- Pixel frame `k` = source frame `start + 2k` (16 fps); latent `f` covers pixel frames `4f-3 .. 4f` (latent 0:
  frame 0); `T = 1 + 4(N - 1)` pixel frames.
- States: sample-and-hold of the last tick at or before each pixel frame; `alive` is 0 past the coverage.
- Controls: 11 buttons OR'd per frame, camera deltas summed and mu-law quantised with the `noclip` encoding
  (`data.camera_encoding`; the paper's, passed explicitly), weapon id (52-way) forward-filled; four ordered substeps
  per frame for the field.
- Cameras: camera-to-world from the recorded position (eye 64 u above the feet), yaw and pitch.

## For the paper's Table-3 rounds

These files are not published yet; this section lists what the release will provide. It ships the artefacts rather
than their producers (several producers are not part of this release):
the 96-row round index, the media rows and tick tables of all ten players of the 32 rounds (320 files, byte-identical
to OpenCS2), latent 0 of the 96 clients, their 96 visibility-label files and 192 observer-signal files.
