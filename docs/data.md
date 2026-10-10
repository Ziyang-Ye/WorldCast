# Data

Training and evaluation read recordings of Counter-Strike 2 rounds and files derived from them. This page lists the
files, the config key that names each and what each holds; `examples/train_paths.yaml` names every path. The released
weights ([docs/inference.md](inference.md#weights)) and the six example rounds ([examples/README.md](../examples/README.md))
are on the [Hub](https://huggingface.co/ZiyangYe/WorldCast); inference reads the subset of the files that
[docs/inference.md](inference.md#data) describes.

## OpenCS2

The recordings are those of [OpenCS2](https://huggingface.co/datasets/blanchon/opencs2_dataset) (Julien Blanchon, 2026):
the first-person views of the ten players of Counter-Strike 2 rounds, rendered from demos published on HLTV, at
1280 x 720 and 32 fps, with every player's controls and state at each 64 Hz tick. One player's view of one round is a
*recording*; its `media_id` is `<match>-<map>-r<round>-p<slot>`. `data.dataset_root` is the folder that holds
`rounds/`:

```
index/pov_rounds.parquet                                                   one row per recording
rounds/match_id=<m>/map_name=<map>/round=<rr>/player=<pp>/video.mp4        the video
rounds/match_id=<m>/map_name=<map>/round=<rr>/player=<pp>/ticks.parquet    the tick table
```

**Frames.** The 32 fps frames of a recording are its *source frames*, two ticks apart (the media index's
`video_frames` counts them). The generator works on every second source frame, its *video frames* (16 fps): video frame
`k` is source frame `start + 2 k` in a window that starts at source frame `start`, and latent frame `f` holds the video
frames `4 f - 3 .. 4 f` (latent frame 0 holds video frame 0), so a block of 4 latent frames is 16 video frames, one
second.

The paper uses four maps, `de_dust2`, `de_mirage`, `de_nuke` and `de_ancient` (118,390 recordings, 343 matches):

```bash
hf download blanchon/opencs2_dataset --repo-type dataset --revision <revision> --local-dir data/opencs2 \
  --include "index/pov_rounds.parquet" "rounds/*/map_name=de_dust2/*" "rounds/*/map_name=de_mirage/*" \
            "rounds/*/map_name=de_nuke/*" "rounds/*/map_name=de_ancient/*" --exclude "*.preview.mp4"
```

Pin `--revision` to the commit hash of the dataset, not to a branch name, so that the files stay the same: a tick file
is checked against the media index by its size and sha256.

**Terms.** The dataset card states: "`.dem` source data is mirrored from HLTV; downstream use is bound by the original
tournament terms. Renders and metadata are released as **CC-BY-4.0**." (read on the card, October 7, 2026). The
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode) (Section 3(a)) requires anyone who shares
OpenCS2, or a file derived from it (a media index, labels, latents, a bucket index), to credit the creator, to name
the license and link it, to link the dataset and to indicate if the material was modified. For example: "Derived from
the OpenCS2 dataset (Julien Blanchon, 2026, https://huggingface.co/datasets/blanchon/opencs2_dataset), CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Modified: <what the file is>." The tournaments' terms are theirs; you
are responsible for complying with them.

**Citation** (the dataset card's):

```bibtex
@misc{blanchon2026opencs2,
  author       = {Julien Blanchon},
  title        = {OpenCS2 Dataset},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://github.com/julien-blanchon/opencs2-dataset}}
}
```

## The files

| file | config key | used by |
|---|---|---|
| [media index](#media-index) | `data.media_index` | stages 2-4 |
| [raw-video manifest](#raw-video-manifest) | `data.train_manifest` | stages 1, 1_long |
| [bucket index](#bucket-index-evaluation-reserve-and-validation-index) | `data.bucket_dir` | stages 2-4 |
| [evaluation reserve](#bucket-index-evaluation-reserve-and-validation-index) | `data.exclude_media_manifest` | stages 2-4 |
| [validation index](#bucket-index-evaluation-reserve-and-validation-index) | `validation.index` | evaluation |
| [latent cache](#latent-cache) | `data.latent_cache_root` | stages 2-4, evaluation |
| [GT visibility labels](#gt-visibility-labels) | `data.visibility_label_root` | stages 2-4 |
| [flash and scope labels](#flash-and-scope-labels) | `data.observer_signal_label_root` | stages 2-4 |
| [collision meshes](#collision-meshes) | `data.collision_meshes` | stages 2s, 3, 4 |
| umT5 embedding of the fixed prompt | `data.prompt_embedding` | all ([weights](inference.md#weights)) |
| Wan2.2-TI2V-5B snapshot | `model.wan22_root` | all ([weights](inference.md#weights)) |

## Formats

### Media index

JSON lines, one object per recording, keys sorted. It tells the training stages and the inference client which files
hold a recording and what they must contain. The fields are listed in [docs/inference.md](inference.md#data):

```json
{"capture_start_tick":118864,"fps":32.0,"map_name":"de_mirage","match_id":2393226,"media_id":"2393226-de_mirage-r16-p00","player_slot":0,"round":16,"ticks_file_size":148542,"ticks_path":"rounds/match_id=2393226/map_name=de_mirage/round=16/player=00/ticks.parquet","ticks_rows":4406,"ticks_sha256":"a15a74733503b418420beaa70bffe988fd32205769bef49b7af7d2c205549176","video_frames":2204,"video_path":"rounds/match_id=2393226/map_name=de_mirage/round=16/player=00/video.mp4"}
```

`ticks_path` and `video_path` are relative to `data.dataset_root`. A tick table is checked against its row by size and
sha256.

### Raw-video manifest

Stages 1 and 1_long (`data.train_manifest`): JSONL, one record per recording: `media_id, match_id, map_name, round,
player_slot, video_path, ticks_path` (absolute, or relative to `data.dataset_root`), `video_frames` (source frames),
`video_file_size, ticks_file_size, ticks_sha256, ticks_rows, fps, pixel_frames` (the video frames of a window: 81 for
stage 1, 161 for 1_long), `stride, max_tick_gap_seconds` (1.5 / 64), `start_frames` (sorted source frames, multiples of
`stride`) and `n_windows`. The reader also takes `skip_frame` (source frames per video frame, 2) and `sample_weight`
(per record) or `window_sample_weights` (per window). A window is served when every video frame owns a tick, the last
tick of a frame is at most 1.5 ticks (1.5 / 64 s) before it and no two ticks of the window are further apart; a record
of another sampling, or with start frames off the stride or past the video, is refused
(`worldcast.data.training.RawVideoWindows`).

### Bucket index, evaluation reserve and validation index

- **Bucket index** (`data.bucket_dir`): five files, `train_q00-01.jsonl` ... `train_q07-10.jsonl`, each the windows of
  one map (the paper's: `train_q00-01.jsonl` de_dust2, `train_q01-03.jsonl` de_mirage, `train_q03-05.jsonl` de_nuke,
  `train_q05-07.jsonl` de_ancient, `train_q07-10.jsonl` empty; `data.bucket_weights` weighs the files). One JSON
  object per 41-latent window of one player's view, ordered by `media_id` and `start_frame`:
  `{"latent_key":"win_001288","map_name":"de_dust2","match_id":2392812,"media_id":"2392812-de_dust2-r09-p09","round":9,"start_frame":1288,"vis_available":true,"vis_pair_count":3}`.
  `start_frame` is a source frame (the window covers the source frames `start .. start + 320`, 161 video frames);
  `match_id` and `round` are checked against the media index; `latent_key` is optional and must be `win_<start:06d>`;
  `vis_available` and `vis_pair_count` (visible pairs of that player and another player over the window) weight the
  sampling. A `(media_id, start_frame)` pair occurs once across the files.
- **Evaluation reserve** (`data.exclude_media_manifest`): text, one media id per line, ascending; every id occurs in
  the bucket index.
- **Validation index** (`validation.index`): the rows of the bucket index (the same fields), ordered by `map_name`,
  `media_id` and `start_frame`. The paper's holds 64 windows, 16 per map; evaluation and the in-training validation
  check the file by sha256 against the paper's index (`validation.index_sha256` sets another). A row may also carry
  `num_visible_players`: the evaluation groups its per-stratum scores by it (`vis<n>`; a row without it is grouped by
  its `quality` field if it has one, else as `vis?`).

### Latent cache

`data.latent_cache_root`: `<media_id>.npz`, member `win_<start:06d>` per window, float16 `[1, 41, 48, 24, 42]`: Wan2.2
VAE latents (with its 48-channel normalization) of the 161 video frames at the source frames `start, start + 2, ...,
start + 320` (41 latent frames), resized bilinearly to 384 x 672 and mapped to `(x / 255 - 0.5) * 2`. The cache is
encoded by the VAE in bf16 on CUDA (weights and pixels). In training, a teammate's members are the candidates of a
client's memory frames. The inference client reads only latent 0 of its window from this layout
([docs/inference.md](inference.md#data)); the example cases ship it.

### GT visibility labels

`data.visibility_label_root`: `<media_id>.npz` with arrays of `[10, T]`, `T` the recording's `video_frames` (source
frames): row `s` is player slot `s` as seen by the recording's player (its own row is empty), and column `k` is source
frame `k`, engine tick `capture_start_tick + 2 k`. A slot the round has no recording of is a row of zeros.

| array | meaning |
|---|---|
| `_binary_visible` | the player sees player `s`: line-of-sight tests from the eye to the body of `s` are clear of the map's opaque triangles and of the other players. True only where `_binary_eval_valid` |
| `_binary_eval_valid` | the label is known: `s` is visible or occluded. The recording's player and `s` are alive, the round is on, the view is not scoped, `s` is in the frustum and the state of every other player is known |
| `_in_frustum` | at least one test point of `s` falls in the camera's field of view (106.26 x 73.74 degrees), whether the label is known or not |
| `_binary_offscreen` | `s` was tested and no test point falls in the field of view |

Visible, occluded and unknown are three values: visible is `_binary_visible`, occluded is `_binary_eval_valid` and not
visible, and false outside `_binary_eval_valid` is unknown, not occluded (`worldcast.data.labels.visibility_rows`
returns visible within valid). A round of fewer than ten recordings has no known
label: another player without a tick table could stand in any line of sight, so `_binary_eval_valid` is False
throughout. The files hold further arrays (the diagnostics of each test and a record of the inputs) that nothing
reads.

### Flash and scope labels

`data.observer_signal_label_root`, one sample per source frame (32 fps) of the recorded video, `n` the recording's
source frames.

**Flash file** `flashlabels/<media_id>.npz`:

| array | dtype, shape | meaning |
|---|---|---|
| `lum` | float16 `[n]` | the screen's mean luminance of each frame, in [0, 1] (the 64 x 36 gray frame / 255, averaged) |
| `stride` | int16 | 1: one sample per source frame |
| `hot_threshold` | float32 | 0.85: a frame brighter than it is flash-white |

**Scope file** `scopelabels/<media_id>.npz`:

| array | dtype, shape | meaning |
|---|---|---|
| `corner_max` | float16 `[n]` | the brightest of the four 8 x 8 corner patches (mean gray in [0, 1]) |
| `center` | float16 `[n]` | the 8 x 8 center patch (rows 14-21, columns 28-35) |
| `scoped_vis` | uint8 `[n]` | 1 where the scope's overlay is on screen: `corner_max < 0.04` and `center > 0.08` (a dark vignette around a lit center) |
| `attack2` | uint8 `[n]` | 1 where the scope button is held in any tick of the frame (the ticks in `(t - 1/32 s, t]`) |
| `weapon_id` | int16 `[n]` | the weapon the frame's last tick names (the 52-way ids of `OPENCS2_WEAPONS`; `<none>` if it names none); -1 beyond `ncov` |
| `level` | int8 `[n]` | the zoom level, 0 unscoped, 1 and 2: a press of the scope button steps the level of an awp, ssg08, scar20 or g3sg1 (0, 1, 2, 0) or of an aug or sg556 (0, 1, 0), any other weapon is unscoped, and a level counts only while `scoped_vis` is 1 |
| `ncov` | int32 | the source frames the tick table covers (its span in frames, less 2): the frames beyond have no button and weapon -1 |
| `thresholds` | float32 `[2]` | 0.04, 0.08: the corner and center thresholds above |

The reader (`worldcast.data.labels`) samples a window's source frames from them: a latent frame is flash if any of its
video frames is above `hot_threshold`, and takes the scope's overlay and `level` at its last video frame. It takes a
scope file for the recording's frames when the arrays hold exactly its `video_frames`. In training, a recording
without a flash or a scope file reads as unknown (no latent frame of it has a valid signal); a client requires both
files ([docs/inference.md](inference.md#data)).

### Collision meshes

`data.collision_meshes`: `{map_name: path}` to a glTF binary of the map's world physics, engine units, +z up like the
tick tables; the paper's maps are `de_dust2`, `de_mirage`, `de_nuke` and `de_ancient`. The nodes of an export carry one
transform to glTF meters (an axis permutation and 0.0254), which the loader (`load_collision_mesh`) does not apply.
The map files of the game are Valve's: this repository and the Hub carry no map file and no export of one.
