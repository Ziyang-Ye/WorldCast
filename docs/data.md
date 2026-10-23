# Data

Training and evaluation read recordings of Counter-Strike 2 rounds and files derived from them. This page says where
the data comes from, what each file is for, the command that builds it and how to check the result against the paper's
files; the formats are in [docs/training.md](training.md#data-formats) and [docs/inference.md](inference.md#data). The
release publishes the weights ([docs/inference.md](inference.md#weights)) and the six example rounds
([examples/README.md](../examples/README.md)) on the [Hub](https://huggingface.co/ZiyangYe/WorldCast); the evaluation
sets' data is the dataset [ZiyangYe/WorldCast-data](https://huggingface.co/datasets/ZiyangYe/WorldCast-data)
(`tools/download_eval_data.py`), and the paper's lists and evaluation sets ship in `configs/data/` and `configs/eval/`
([The paper's files](#the-papers-files)). Every training file is built from the sources below with
`tools/prepare_data.py`, one step per file.

## Where the data comes from

| source | what WorldCast takes from it | terms | how to get it |
|---|---|---|---|
| [OpenCS2](https://huggingface.co/datasets/blanchon/opencs2_dataset) | the recorded videos and tick tables of Counter-Strike 2 rounds | CC BY 4.0 ([below](#opencs2)) | `hf download` ([below](#opencs2)) |
| [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | the VAE that encodes the latent cache; for stage 1 the backbone | Apache-2.0, the Alibaba Wan Team ([NOTICE](../NOTICE)) | `tools/download_weights.py` ([below](#wan22)) |
| the game's map files (Counter-Strike 2) | the collision meshes and the visual meshes | Valve's ([below](#the-games-map-files)) | your own copy of the game: WorldCast carries no map file |

### OpenCS2

[OpenCS2](https://huggingface.co/datasets/blanchon/opencs2_dataset) (Julien Blanchon, 2026) holds the first-person
views of the ten players of Counter-Strike 2 rounds, rendered from demos published on HLTV, at 1280 x 720 and 32 fps,
with every player's controls and state at each 64 Hz tick. One player's view of one round is a *recording*; its
`media_id` is `<match>-<map>-r<round>-p<slot>`. The dataset's layout; `data.dataset_root` is the folder that holds
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

The four maps of the paper, without the previews that training does not read (the recordings of these maps are 5.96 TB
of video and 17.8 GB of tick tables):

```bash
hf download blanchon/opencs2_dataset --repo-type dataset --revision <revision> --local-dir data/opencs2 \
  --include "index/pov_rounds.parquet" "rounds/*/map_name=de_dust2/*" "rounds/*/map_name=de_mirage/*" \
            "rounds/*/map_name=de_nuke/*" "rounds/*/map_name=de_ancient/*" --exclude "*.preview.mp4"
```

Pin `--revision` (the `revision` argument of the Hub's Python download functions) to the commit hash of the dataset you
download, not to a branch name, so that the files stay the same, and keep the hash with the files you build from them:
a tick file is checked against the media index by its size and sha256 (`verify_ticks_file`). The paper's data has
these counts, to check a download against:

| map | recordings | matches | rounds |
|---|---:|---:|---:|
| `de_dust2` | 33,139 | 164 | 3,327 |
| `de_mirage` | 30,538 | 145 | 3,054 |
| `de_nuke` | 26,057 | 125 | 2,606 |
| `de_ancient` | 28,656 | 135 | 2,866 |
| total | 118,390 | 343 | 11,853 |

343 matches and 569 match-maps (a match played on several maps counts once per map) over the four maps, split by match
into 303 training matches (10,480 rounds) and 40 held-out matches (1,373 rounds).

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

### Wan2.2

The latent cache is encoded with the VAE of Wan2.2-TI2V-5B, and stage 1 trains its backbone (Apache-2.0).
`tools/download_weights.py --out-dir weights` fetches `config.json`, the VAE and the tokenizer into
`weights/Wan2.2-TI2V-5B/` (`model.wan22_root`) at the revision the weights were trained from
(`921dbaf3f1674a56f47e83fb80a34bac8a8f203e`,
[docs/inference.md](inference.md#weights)); stage 1 also needs the backbone files from the same revision
([docs/training.md](training.md#install-and-data)).

## How the pieces fit

Each step reads what the steps above it in this table write, and the files of the game and of OpenCS2:

| step | reads | writes |
|---|---|---|
| `media-index` | the OpenCS2 download | the media index |
| `manifest` | the media index, the videos and tick tables | the raw-video manifest |
| `observer-labels` | the media index, the videos and tick tables | the flash and scope labels |
| `meshes` | the game's `<map>.vpk` | the collision meshes (`--kind collision`), the visual meshes (`--kind visual`) |
| `visibility` | the media index, the tick tables (and videos), the visual meshes | the GT visibility labels |
| `buckets` | the media index, the GT visibility labels, the held-out matches and the excluded media (`configs/data/`) | the bucket index, the evaluation reserve, the validation index |
| `latent-cache` | the media index, the videos, the Wan2.2 VAE, the windows of the bucket index | the latent cache |

| file | config key | used by | built by |
|---|---|---|---|
| media index | `data.media_index` | stages 2-4 | `media-index` |
| raw-video manifest | `data.train_manifest` | stages 1, 1_long | `manifest` |
| flash and scope labels | `data.observer_signal_label_root` | stages 2-4 | `observer-labels` |
| collision meshes | `data.collision_meshes` | stages 2s, 3, 4 | `meshes` |
| GT visibility labels | `data.visibility_label_root` | stages 2-4 | `visibility` |
| bucket index | `data.bucket_dir` | stages 2-4 | `buckets` |
| evaluation reserve | `data.exclude_media_manifest` | stages 2-4 | `buckets` |
| validation index | `validation.index` | evaluation | `buckets` |
| latent cache | `data.latent_cache_root` | stages 2-4, evaluation | `latent-cache` |
| umT5 embedding of the fixed prompt | `data.prompt_embedding` | all | released ([weights](inference.md#weights)) |
| Wan2.2-TI2V-5B snapshot | `model.wan22_root` | all | `tools/download_weights.py` |

`examples/train_paths.yaml` names every path. Each step has `--help` after its name, and a step that cannot run on what
it is given ends with its usage, `prepare_data.py <step>: error: <message>` and exit status 2.

## Building the files

### Media index

`python tools/prepare_data.py media-index` writes the media index of an OpenCS2 download
([format](inference.md#data)): one row per player-round of the maps you choose. The index tells stages 2-4, the manifest
below and the inference client which files hold a recording and what they must contain.

**Reads** `<dataset-root>/index/pov_rounds.parquet`, OpenCS2's own index, and for every player-round of the chosen
maps its `rounds/match_id=<m>/map_name=<map>/round=<rr>/player=<pp>/video.mp4` and `ticks.parquet`. It needs pyarrow
and the video reader of the package (OpenCV, or decord when installed).

**Writes** the media index: JSON lines, one object per player-round, keys sorted, no spaces, ordered by map, match,
round and player slot (so the file is the same for any `--workers`):

```json
{"capture_start_tick":118864,"fps":32.0,"map_name":"de_mirage","match_id":2393226,"media_id":"2393226-de_mirage-r16-p00","player_slot":0,"round":16,"ticks_file_size":148542,"ticks_path":"rounds/match_id=2393226/map_name=de_mirage/round=16/player=00/ticks.parquet","ticks_rows":4406,"ticks_sha256":"a15a74733503b418420beaa70bffe988fd32205769bef49b7af7d2c205549176","video_frames":2204,"video_path":"rounds/match_id=2393226/map_name=de_mirage/round=16/player=00/video.mp4"}
```

| field | comes from |
|---|---|
| `media_id, match_id, map_name, round, player_slot, video_frames, capture_start_tick` | OpenCS2's index |
| `video_path, ticks_path` | OpenCS2's index, relative to the download |
| `fps` | 32.0, the recordings' frame rate |
| `ticks_rows, ticks_file_size, ticks_sha256` | the tick table, read once |

The paths are relative to `data.dataset_root` (`paths.dataset_root` for inference), the directory that holds `rounds/`.

```bash
python tools/prepare_data.py media-index --dataset-root data/opencs2 --out data/media_index.jsonl
python tools/prepare_data.py media-index --dataset-root data/opencs2 --out data/media_index.jsonl \
    --maps de_dust2 de_nuke --workers 16
```

| option | default | meaning |
|---|---|---|
| `--dataset-root DIR` | required | the download: `index/pov_rounds.parquet` and `rounds/` |
| `--out FILE` | required | the media index (its directory is created) |
| `--maps MAP [MAP ...]` | `de_dust2 de_mirage de_nuke de_ancient` (the paper's) | the maps to index; a map the OpenCS2 index does not have ends the step |
| `--workers N` | `min(8, CPUs)` | threads that check recordings in parallel |

**Checks.** A player-round gets a row only when its video exists and has the byte size of the OpenCS2 index
(`video_missing`, `video_size`), the video holds the index's frame count and its first frame decodes (`video_frames`;
read with `worldcast.data.video.load_video_window`), and its tick table exists, can be read, has rows and starts at the
index's `capture_start_tick` (`ticks_missing`, `ticks_unreadable`, `ticks_empty`, `ticks_start`). A player-round that
fails a check is left out and counted by reason on stderr, with the first message of each reason, so a partial download
gives the index of what it holds:

```
media-index: 4120 rows -> data/media_index.jsonl
media-index: left out 3 player-rounds:
  video_size: 3 (first: video of media 2392062-de_dust2-r08-p06 has 3299204 bytes, the OpenCS2 index says 3299205: ...)
```

A missing `index/pov_rounds.parquet`, an index that cannot be read, a map the index does not have and a download in
which no player-round passes end the step with a usage error; nothing is written.

**Check against the paper's media index** ([its digest](#the-papers-files)). The paper's media index has six more fields
per row (`capture_end_tick, media_schema_version, player_side, ticks_mtime_ns, video_file_size, video_mtime_ns`); the
release reads the thirteen above. On the rows both files have, those thirteen are equal:

```python
import json

FIELDS = ["media_id", "match_id", "map_name", "round", "player_slot", "fps", "video_frames", "video_path",
          "ticks_path", "ticks_rows", "ticks_file_size", "ticks_sha256", "capture_start_tick"]
rows = lambda path: {r["media_id"]: r for r in map(json.loads, open(path))}
mine, paper = rows("data/media_index.jsonl"), rows("paper_media_index.jsonl")
shared = sorted(mine.keys() & paper.keys())
print(len(shared), [(m, f) for m in shared for f in FIELDS if mine[m][f] != paper[m][f]])  # (n, [])
```

`ticks_sha256` agrees only if the tick tables are byte-identical to the paper's: it keys the jump recall of the controls
(`worldcast.data.recordings.verify_ticks_file`).

### Raw-video manifest

`python tools/prepare_data.py manifest` writes the manifest that stages 1 and 1_long train on (`data.train_manifest`):
for every recording the start frames of its windows.

**Reads** the media index of the previous section (`--media-index`), and for each recording in it the tick table
(`ticks.parquet`: `t, active, delta_pitch, delta_yaw, input_weapon`, and `is_alive` for stage 1_long) and the video
file, under `--dataset-root`. It checks the tick table against the media index (size, sha256, rows).

**Writes** JSON lines, one record per recording that has a window, keys sorted, ordered by map, match, round and
player slot ([format](training.md#data-formats)); paths relative to `data.dataset_root`:

```json
{"fps":32.0,"map_name":"de_dust2","match_id":2392774,"max_tick_gap_seconds":0.0234375,"media_id":"2392774-de_dust2-r02-p01","n_windows":421,"pixel_frames":81,"player_slot":1,"round":2,"skip_frame":2,"start_frames":[0,8,16,...,3360],"stride":8,"ticks_file_size":260321,"ticks_path":"rounds/match_id=2392774/map_name=de_dust2/round=02/player=01/ticks.parquet","ticks_rows":7052,"ticks_sha256":"969905b0...","video_file_size":...,"video_frames":3527,"video_path":"rounds/match_id=2392774/map_name=de_dust2/round=02/player=01/video.mp4"}
```

`video_frames` counts source frames and `pixel_frames` video frames ([Frames](#opencs2); `skip_frame` 2). A window starting at source frame `s` covers the source frames `s, s + 2, ..., s + 2 (pixel_frames -
1)`, so a recording of `V` frames has the start frames `0, stride, 2 stride, ...` up to `V - (2 (pixel_frames - 1) +
1)`, less the windows that a rule below leaves out.

```bash
# stage 1: windows of 81 video frames (5 s), start frames every 8 source frames
python tools/prepare_data.py manifest --dataset-root data/opencs2 --media-index data/media_index.jsonl \
    --out data/stage1/train.jsonl
# stage 1_long: windows of 161 video frames (10 s), the player alive throughout
python tools/prepare_data.py manifest --dataset-root data/opencs2 --media-index data/media_index.jsonl \
    --stage 1_long --out data/stage1_long/train.jsonl
# without the evaluation reserve, with a decode check of every video
python tools/prepare_data.py manifest --dataset-root data/opencs2 --media-index data/media_index.jsonl \
    --out data/stage1/train.jsonl --exclude-media configs/data/eval_reserve.txt --decode-scan \
    --excluded-out data/excluded.txt
```

| option | default | meaning |
|---|---|---|
| `--dataset-root DIR` | required | the download; the media index's paths are relative to it |
| `--media-index FILE` | required | the media index (`media-index`) |
| `--out FILE` | required | the manifest (its directory is created) |
| `--stage {1,1_long}` | `1` | the stage that reads it: 81 video frames (stage 1) or 161 (stage 1_long, which also keeps only the windows in which the player is alive at every tick); the window of the stage's config in `configs/train/` |
| `--stride N` | `8` | source frames between window starts: one latent frame |
| `--maps MAP [MAP ...]` | every map of the media index | keep only these maps |
| `--exclude-media FILE` | none | media ids to leave out, one per line: the evaluation reserve (`data.exclude_media_manifest`), or the media of held-out matches |
| `--decode-scan` | off | decode every video end to end with ffmpeg (the one bundled with `imageio-ffmpeg`) and leave out each that fails or does not decode to the media index's frame count |
| `--excluded-out FILE` | none | write the ids of the recordings that a check left out (not those of `--exclude-media`), one per line, in the format of `--exclude-media` |
| `--workers N` | `min(8, CPUs)` | threads that check recordings in parallel |

**Window rules.** A window is kept when the reader of the stage (`worldcast.data.training.RawVideoWindows`) can serve
it: every video frame owns a tick, the last tick of a frame is at most 1.5 ticks (1.5 / 64 s) before it, and no two
ticks of the window are further apart. A window is left out too when its ticks hold a button that OpenCS2 does not list
(the keyboard turns `turn_left` and `turn_right` are among them) and, for stage 1_long, when the player is dead at any
tick of it.

**Left out and counted.** A recording that has no window is left out and counted by reason on stderr (`ticks_missing`,
`ticks_mismatch` (size or sha256 differ from the media index), `ticks_rows`, `ticks_unreadable`, `ticks_time` (the times
are not strictly increasing), `ticks_turn`, `video_missing`, `short_video` (fewer source frames than a window spans),
`no_window`; with `--decode-scan` also `decode_error`, `decode_frames`, `decode_timeout`), and so are the windows of the
other recordings that a rule left out:

```
manifest: stage 1, 4098 recordings, 902114 windows -> data/stage1/train.jsonl
manifest: left out 22 recordings:
  short_video: 22 (first: video of media 2391808-de_ancient-r01-p03 has 120 frames, a window needs 161)
manifest: left out 3411 windows by ticks
manifest: left out 2057 windows by buttons
```

An unreadable media index, a record without `video_path`, an option the step does not take (`--stage 2`), an
exclusion file that repeats an id and a result without any recording end the step with a usage error and write nothing.

**Check.** `RawVideoWindows` reads the manifest and refuses a record of another sampling (`pixel_frames`, `skip_frame`,
`max_tick_gap_seconds`) or with start frames off the stride or past the video; every window of the file can be served:

```python
from worldcast.config.training import STAGES
from worldcast.data.controls import TICK_GATE_SECONDS
from worldcast.data.training import RawVideoWindows
from worldcast.data.window import WindowSpec

stage = STAGES["1"]
spec = WindowSpec(stage.latent_frames, max_tick_gap_seconds=TICK_GATE_SECONDS,
                  camera_encoding=stage.camera_encoding)
windows = RawVideoWindows("data/stage1/train.jsonl", spec=spec, dataset_root="data/opencs2")
item = windows[len(windows) // 2]   # frames [3, 81, 384, 672], buttons [81, 11], view_deltas [81, 2], weapon [81]
```

### Flash and scope labels

`python tools/prepare_data.py observer-labels` computes the flash and scope labels from the recorded video and writes
the files that training and the client read (`data.observer_signal_label_root`; format below and in
[Data](inference.md#data)). It runs on the CPU.

**Reads**

| what | where |
|---|---|
| media index | `data.media_index` |
| the recordings' videos (`video_path` of each media row) | under `data.dataset_root` |
| the recordings' tick tables (`ticks_path` of each media row; need pyarrow) | under `data.dataset_root` |

The paths come from `train_paths.yaml` (`--config train_paths.yaml`; no stage config is needed) or `--set`. The videos
are decoded by the ffmpeg of `imageio-ffmpeg`, which the package installs (`IMAGEIO_FFMPEG_EXE` names another binary).

**Writes** `flashlabels/<media_id>.npz` and `scopelabels/<media_id>.npz` into `--out-dir` (default
`data.observer_signal_label_root`), one sample per source frame (32 fps) of the recorded video. A frame of the video is
scaled to 64 x 36 and turned to gray; the labels are read off these frames, every one of them.

| option | default | meaning |
|---|---|---|
| `--config YAML`, `--set KEY=VALUE` | | the paths, as in every tool |
| `--out-dir DIR` | `data.observer_signal_label_root` | the label root to write; `flashlabels/` and `scopelabels/` go under it |
| `--media-id ID [ID ...]` | all | only these recordings |
| `--workers N` | 1 | N processes, each labeling one recording at a time |
| `--videos DIR` | | in place of the media index: the `<media_id>.mp4` files of a directory, 32 fps; writes the flash labels only (the scope labels need the tick tables) |
| `--compare DIR` | | compute the labels and compare them with the files under the label root `DIR` instead of writing (below) |

**Resuming.** A recording whose files exist and match (the frame count of the media row, the thresholds below) is
skipped, so the same command continues a stopped run. A file is written under a temporary name and renamed, so a run
that is killed leaves no half file. The labels do not depend on `--workers`.

**Checks.** A video that runs at another fps than its media row, or that decodes to another number of frames than the
row's `video_frames`, is reported (`FAILED <media_id>: ...`) and gets no file: the reader takes a scope file for a
recording's frames only when their counts agree. The flash label stands on the video alone, the scope label on the
video and the tick table: a tick table that cannot serve the video (a source frame without a tick, a button outside
the OpenCS2 enum, a size or digest that differs from the media row's) fails the scope label of that recording and not
its flash label. The run goes on after a failed recording and the exit status is 1; a refused config or option exits
with status 2.

```bash
python tools/prepare_data.py observer-labels --config train_paths.yaml --workers 16
# one recording, or only the flash labels of a directory of videos
python tools/prepare_data.py observer-labels --config train_paths.yaml --media-id 2393226-de_mirage-r16-p00
python tools/prepare_data.py observer-labels --videos clips --out-dir data/obslabels
```

**Flash file** `flashlabels/<media_id>.npz`, `n` the recording's source frames:

| array | dtype, shape | meaning |
|---|---|---|
| `lum` | float16 `[n]` | the screen's mean luminance of each frame, in [0, 1] (gray / 255, averaged over the 64 x 36 pixels) |
| `stride` | int16 | 1: one sample per source frame |
| `hot_threshold` | float32 | 0.85: a frame brighter than it is flash-white |
| `n_hot` | int32 | the frames above the threshold |

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

The reader ([Data](inference.md#data), `worldcast.data.labels`) samples a window's source frames from them: a latent
frame is flash if any of its video frames is above `hot_threshold`, and takes the scope's overlay and `level` at its
last video frame. It takes a scope file for the recording's frames when the arrays hold exactly its `video_frames`. In
training, a recording without a flash or a scope file reads as unknown (no latent frame of it has a valid signal); a
client requires both files ([docs/inference.md](inference.md#data)).

**Check against other label files.** `--compare DIR` computes the labels of the chosen recordings and prints, for every
array, whether the files under `DIR` hold the same, so a rebuild can be checked against the paper's label files:

```bash
python tools/prepare_data.py observer-labels --config train_paths.yaml --compare paper/obslabels \
    --media-id 2393226-de_mirage-r16-p00 2393226-de_mirage-r16-p01
```

```
flash/lum: equal in all of 2 recordings
flash/stride: equal in all of 2 recordings
flash/hot_threshold: equal in all of 2 recordings
flash/n_hot: equal in all of 2 recordings
scope/corner_max: equal in all of 2 recordings
scope/center: equal in all of 2 recordings
scope/scoped_vis: equal in all of 2 recordings
scope/attack2: equal in all of 2 recordings
scope/weapon_id: equal in all of 2 recordings
scope/level: equal in all of 2 recordings
scope/ncov: equal in all of 2 recordings
scope/thresholds: equal in all of 2 recordings
2 compared
```

Each line is `equal in all` or `differs in K of N recordings, largest absolute difference D`; the exit status is 1 if an
array differs or a file is missing. The paper's flash and scope label directories hold 118,390 and 118,304 files.

### Meshes

The collision and the visual meshes are made from the game's map files, which are Valve's: read
[the game's map files](#the-games-map-files) first. `python tools/prepare_data.py meshes` exports a map's glTF binary
from your copy of the game with the command-line tool of
[ValveResourceFormat](https://github.com/ValveResourceFormat/ValveResourceFormat) (Source 2 Viewer; MIT license,
Copyright (c) 2015 ValveResourceFormat Contributors; the paper's meshes were exported with its release 19.2), checks the
file as its reader loads it and writes it to `--out-dir`:

```bash
python tools/prepare_data.py meshes --vpk-dir "<game>/game/csgo/maps" --cli <the tool> --map de_dust2 --out-dir data/meshes
python tools/prepare_data.py meshes --vpk-dir "<game>/game/csgo/maps" --cli <the tool> --map de_dust2 --kind visual \
    --out-dir data/meshes
python tools/prepare_data.py meshes --glb exported.glb --map de_dust2 --out-dir data/meshes   # a file exported already
```

| option | default | meaning |
|---|---|---|
| `--vpk-dir DIR` or `--glb FILE` | one of the two | the maps folder of your copy of the game, where `<map>.vpk` lies, or a file exported already |
| `--cli TOOL` | with `--vpk-dir` | the tool: a path, or a name your shell finds |
| `--map NAME` | required | the map, e.g. `de_dust2` |
| `--kind collision\|visual` | `collision` | which mesh of the map |
| `--out-dir DIR` | required | receives the mesh |

The step prints and runs the export command of the paper's Dust2 meshes, with the map's name; `-f` names the resource
of the kind:

```
Source2Viewer-CLI -i <maps folder>/de_dust2.vpk -o <work folder> -d -f maps/de_dust2/world_physics.vmdl_c
    --gltf_export_format glb --gltf_export_materials --gltf_textures_adapt --gltf_export_extras
```

| `--kind` | `-f` | written as | read as |
|---|---|---|---|
| `collision` | `maps/<map>/world_physics.vmdl_c`, the world physics | `<map>_world_collision_complete.glb`, the names of `examples/train_paths.yaml` | `data.collision_meshes` of stages 2s, 3 and 4 |
| `visual` | `maps/<map>.vmap_c`, the render geometry | `<map>_visual_complete.glb` | `--visual-meshes` of `visibility` |

It prints the mesh's size and the sha256 digests of the file and of its geometry, and with `--vpk-dir` of `<map>.vpk`,
next to the paper's. A mesh that differs from the paper's is written all the same. The *geometry* digest is the check;
it does not depend on how the file is packed, which the paper's files changed (their textures are embedded, and an
export refers to them as separate PNG files), so the file digest of a fresh export differs.

#### Collision meshes

Training reads one glTF binary of each map's world physics ([format](training.md#data-formats)). The vertices of the
file are in engine units, +z up, like the tick tables; the nodes of an export carry one transform to glTF meters (an
axis permutation and 0.0254), which the loader (`load_collision_mesh`) does not apply and the step requires (a file with
another transform is refused). The geometry digest is the sha256 of the loaded vertices (float64) and faces (int64):

```
data/meshes/de_mirage_world_collision_complete.glb: 133,133 triangles, 399,399 vertices, bounds (u) [-4096.0, -3864.0, -448.0] to [2048.0, 2048.0, 1024.0]
file sha256 111059fa0f0cabfe99ee1a7f381704cd80d8ee61de9979a4fdcf433201d495cf: matches the paper's
geometry sha256 2b0dd691deebb85704f9b8841e482a38b39432b1f7f9b6787615a9af8edfded5: matches the paper's
```

The paper's collision meshes, and the packages (`<map>.vpk`) they were exported from:

| map | `<map>.vpk` sha256 | mesh file sha256 | geometry sha256 |
|---|---|---|---|
| `de_dust2` | `a3a424bb8156539845fd453348e6b01b5b1114470d00677ff6250aff0a550467` | `88fd9726db274f03079ee715e489b562ebce6e3ccd87c2ca6aa438c875a77509` | `311c359c39a273ecc7999f57b891cc5b6cf9cd3c14934dfee2e4937ef6edf390` |
| `de_mirage` | `ec578cb2d63c235dc720ed4d56353917de6895a147180952cf1f566be053df41` | `111059fa0f0cabfe99ee1a7f381704cd80d8ee61de9979a4fdcf433201d495cf` | `2b0dd691deebb85704f9b8841e482a38b39432b1f7f9b6787615a9af8edfded5` |
| `de_nuke` | `f0f970155c00db9c16d888576e063fb0f845d7f9a0b5eebb1e78cab0c4c0aa92` | `e1663e1cfcd6d5224ebb5707831941be1a8a4395a50a94f5202d02b8938cc9c6` | `014bd86e71a1684deb1b8787e6e9cbfe5e1c7ed7916604bc65dee6b71d563fe3` |
| `de_ancient` | `5dffb06306d5351985fc2b2c06849753c3f528474e9e93dba670b9c290463a18` | `94c5d88b68a6455b75c81744989e7753e8b4bd278cf3396bb0880ea5d46c085e` | `d56f8d86b46021f4399c67bb0bab2502f41eb0b92f107b1017c3ed4f477647d7` |

#### Visual meshes

The GT visibility labels are tested against each map's *visual* mesh, the render geometry with its materials, not against
the collision mesh, which has the physics surfaces (clips, skybox, glass as one solid) and lacks the props and the
materials that say which triangles block. `--kind visual` exports it from the same `<map>.vpk`
(`--gltf_export_materials` and `--gltf_export_extras` are what the visibility step reads: the materials' alpha modes and
names, and the shader name in their extras). The step loads the file as `visibility` does: it keeps each node's
transform and takes the vertices from glTF meters, +y up, to engine units, +z up; it reads no texture, so a file that
refers to its textures as PNG files is as good as one with them embedded. A visual mesh is large (the paper's: 342 MB
for de_mirage, 579 MB for de_nuke, 1.98 GB for de_ancient), and `visibility --workers` loads it once per process. The
geometry digest is the sha256 of the triangles it keeps (`Occluders.summary`), which the visibility label files of this
step record as `_meta.mesh.sha256`:

| map | file sha256 (as the paper's label files record it) | geometry sha256 | opaque / possible triangles |
|---|---|---|---|
| `de_dust2` | `7bf712db0feeac3952225702af919d85e9d982010495021ddfc8cfdbee05f9a7` | `0e67055cb233452fd6b4ab191d583ca929d35b57998430b00ab055914aadee7d` | 4,148,529 / 373,391 |
| `de_mirage` | `5093262ba6a5080f3902c4e346fbb1b6bc8bcb9a0544b7960764c727063a72e1` | `e03efaf149ef174bd76a184ec0457033ad6c22d30a0c3b0bdee7418c487dac4c` | 1,058,806 / 229,478 |
| `de_nuke` | `0916c1e9d620827c1cd855d51227dbc0604d7410af4aad9998ce2c3f8e1a8111` | `1a3e02b477dedc02731a49d66c08df9a4e3d1bf521378e256625f3404629c5dc` | 3,570,745 / 18,972 |
| `de_ancient` | `4ad522fc22ce2545ed8df9b39cb9be0cebcf20f9c40acda1384761ee7aba7cb0` | `5432736ca4a92c499bacd458be9d0b18c88ac6c01309843301cbe706d4f34b34` | 8,277,692 / 918,997 |

For `de_dust2` the geometry and the triangle counts are those of the export in the Dust2 server pack (file sha256
`65388512501b3bc271c39dee14a691f6e8a0471e322ab17897727d400db89fc3`), which reproduces the labels exactly.

### GT visibility labels

`python tools/prepare_data.py visibility` makes the GT visibility labels of stages 2-4 (`data.visibility_label_root`):
for every recording, which other players its player sees, frame by frame. It tests each of the nine other players of
the round against the player's view on the map's mesh, with the recorded positions, views and stances of all ten
players. It runs on the CPU.

**Reads** the media index (`--media-index`); the tick table of every recording of each round it labels, under
`--dataset-root` (needs pyarrow); the map's visual mesh (`--visual-meshes`, from [Meshes](#meshes) with `--kind
visual`); the videos of the recordings under `--dataset-root` (`video_path` of the media index; needs OpenCV), unless
`--scope none`; optionally OpenCS2's `index/pov_rounds.parquet` (`--pov-index`).

**Writes** `<out-dir>/<media_id>.npz` (`data.visibility_label_root`), one per labeled recording. A rerun keeps a file
that is already there when its inputs and settings are the same, so an interrupted run resumes, and the files do not
depend on `--workers`.

```bash
python tools/prepare_data.py visibility --media-index data/media_index.jsonl --dataset-root data/opencs2 \
    --out-dir data/vislabels --visual-meshes de_nuke=data/meshes/de_nuke_visual_complete.glb --workers 8
python tools/prepare_data.py visibility --media-index data/media_index.jsonl --dataset-root data/opencs2 \
    --out-dir data/vislabels --pov-index data/opencs2/index/pov_rounds.parquet --capsule-unknown \
    --visual-meshes de_dust2=data/meshes/de_dust2_visual_complete.glb --media-id 2392774-de_dust2-r09-p05
```

| option | default | meaning |
|---|---|---|
| `--media-index FILE` | required | the media index; every recording in it is labeled, or those of `--media-id` |
| `--dataset-root DIR` | required | holds `rounds/match_id=<m>/map_name=<map>/round=<rr>/player=<pp>/` (tick tables, videos) |
| `--out-dir DIR` | required | receives `<media_id>.npz` (created) |
| `--visual-meshes MAP=GLB` | | a map's visual mesh; repeat for each map of the labeled rounds |
| `--meshes MAP=GLB` | | a map's collision mesh (`data.collision_meshes`, [Collision meshes](#collision-meshes)), for a map without `--visual-meshes`: every triangle of it blocks, so its labels differ from the visual mesh's |
| `--pov-index FILE` | derived | OpenCS2's `index/pov_rounds.parquet`: each player's `death_tick` and each round's `round_end_tick`. Without it the step reads them off the tick tables: a player is dead from its first row that is not alive, or from the tick after the last row of a table that ends before the round does; the round ends 32 ticks before the last recorded tick (the recordings run half a second past a round's end) |
| `--scope video\|none` | `video` | `video`: the frames seen through a sniper scope are found in the recorded videos and not evaluated; `none`: every frame is taken as unscoped (no video is read) |
| `--capsule-unknown` | off | a player that only another player hides, while the map leaves a part of its body clear, is unknown, not occluded; the paper's de_dust2 labels were made with it, those of the other three maps without |
| `--media-id ID [ID ...]` | every one | label only these recordings |
| `--workers N` | 1 | worker processes, each loading the meshes of the maps of the rounds it takes (a run of rounds ordered by map) |
| `--overwrite` | off | redo the files that are already there; without it a file made from other inputs or settings ends the step |

A file that is already there is kept only if its media index row, its tick tables (sha256), its round events, the mesh
(file sha256) and the settings are the same. A map without a mesh, a media id that is not in the index, a missing
`video_path` with `--scope video`, a file made from other inputs and a `--pov-index` without a row of a recording end
the step with a usage error.

#### The labels

A file holds arrays of `[10, T]`, `T` the recording's `video_frames` (source frames): row `s` is player slot `s` as seen
by the recording's player (its own row is empty), and column `k` is source frame `k`, engine tick
`capture_start_tick + 2 k`. A slot the round has no recording of is a row of zeros. These are the arrays
[docs/inference.md](inference.md#data) names:

| array | meaning |
|---|---|
| `_binary_visible` | the player sees player `s`: some head probe, or at least two of the eleven body probes, are clear of the map's opaque triangles and of the other players. True only where `_binary_eval_valid` |
| `_binary_eval_valid` | the label is known: `s` is visible or occluded. The recording's player and `s` are alive, the round is on, the view is not scoped, `s` is in the frustum, the state of every other player is known and the probes decide (none clear, a head probe clear, or two clear) |
| `_in_frustum` | at least one probe of `s` falls in the camera's field of view (106.26 x 73.74 degrees), whether the label is known or not |
| `_binary_offscreen` | `s` was tested and no probe falls in the field of view |

Visible, occluded and unknown are three values: visible is `_binary_visible`, occluded is `_binary_eval_valid` and not
visible (`_binary_invisible`), and false outside `_binary_eval_valid` is unknown, not occluded
(`worldcast.data.labels.visibility_rows` returns visible within valid; `frustum_rows` returns `_in_frustum`).

*How a pair is tested.* The eye is 64 u above the player's feet (46 u crouched). A target has eleven probes (three at the
head, three at the shoulders, three at the waist, two at the legs, spread across the line of sight) on a body 72 u high
(54 u crouched) and 32 u wide. A probe is clear if the segment from the eye to it meets no opaque triangle of the map's
visual mesh and no other player: another player is a vertical capsule of radius 16 u on its feet (a dead player blocks
nothing). Triangles of mask, blend and effect materials do not block; they make a probe ambiguous in the diagnostic
arrays. Exactly one clear probe below the head decides nothing. Frames in which the recording's player is seen through a
sniper scope (the field of view is narrower and a black surround hides the corners), and the frame on each side, are
not evaluated.

The other arrays say how a label was reached; bit `k` of a mask is probe `k` of the eleven, head probes first:

| array | meaning |
|---|---|
| `_binary_invisible`, `_binary_train_valid` | `_binary_eval_valid` and not visible; `_binary_eval_valid` again |
| `slot_<s>` | `[T]` per recorded slot, `_binary_visible` of row `s` |
| `_projected_mask`, `_clear_mask_upper`, `_clear_mask_lower`, `_alpha_ambiguous_mask` (uint16) | probes inside the field of view; clear of the opaque triangles; clear of them and of the possible blockers; clear of the first and blocked by the second |
| `_binary_player_blocked_mask`, `_binary_clear_mask` (uint16) | probes only another player blocks; probes clear of the map and of the players |
| `_static_los_upper`, `_static_los_lower`, `_los_upper`, `_los_lower`, `_map_label_valid`, `_geometry_label_valid`, `_label_valid` | the map's bounds of visibility without and with the possible blockers and the players, and when they agree |
| `_tested_count`, `_clear_count_upper`, `_clear_count_lower` (uint8) | probes projected, clear of the opaque triangles, clear of every blocker |
| `_player_proxy_blocker_slots`, `_unknown_blocker_slots`, `_blocker_state_complete`, `_player_proxy_ambiguous_mask`, `_unknown_blocker_ambiguous_mask`, `_player_possible_ambiguous_mask`, `_possible_ambiguous_mask` | which players block, whose state is unknown, and the probes each decides |
| `_target_state_valid`, `_target_alive`, `_target_known_dead`, `_target_duck` | the target's state at the frame; dead is known, a missing tick is not |
| `_observer_state_valid`, `_observer_alive`, `_observer_duck`, `_round_active`, `_scoped`, `_view_geometry_valid`, `_view_texture_usable`, `_frame_tick_calibration_valid`, `_frame_tick` (int32) | the recording's player at the frame; `_view_texture_usable` and `_frame_tick_calibration_valid` are True throughout |
| `_slot_expected`, `_slot_present`, `_slot_team_num` (int8), `_same_team` | `[10]` |
| `_meta` | JSON bytes (`worldcast.data.preparation._visibility_labels.decode_meta`): the media row's ids, `capture_start_tick`, `round_end_tick`, the settings, the inputs' digests, the mesh (triangle counts of the opaque and the possible layer and the sha256 of their geometry), the scoped frames and the cells that `--capsule-unknown` moved |

A round of fewer than ten recordings has no known label: another player without a tick table could stand in any line of
sight, so `_binary_eval_valid` is False throughout (`_in_frustum` and `_binary_offscreen` are still set).

#### Check against the paper's labels

The example cases ship the labels of their clients (`examples/data/<case>/vislabels/`,
[Examples](../examples/README.md)) with the tick tables of all ten players. Label the same recordings with the visual
mesh of the table above and compare the arrays. The examples carry no videos, so use `--scope none` (the rounds below
have no scoped frame in the files named). With the paper's visual meshes, every array of every file of `nuke_r07`,
`mirage_r16`, `ancient_r18` and `ancient_r02` is equal (the `_meta.mesh.sha256` of the table is the mesh's):

```bash
python tools/download_examples.py --out-dir examples nuke_r07
python tools/prepare_data.py visibility --media-index examples/data/nuke_r07/media_index.jsonl \
    --dataset-root examples/data/nuke_r07/opencs2 --out-dir data/check/nuke_r07 --scope none \
    --visual-meshes de_nuke=data/meshes/de_nuke_visual_complete.glb --media-id 2393223-de_nuke-r07-p05
python - <<'EOF'
import numpy as np
paper = np.load("examples/data/nuke_r07/vislabels/2393223-de_nuke-r07-p05.npz")
ours = np.load("data/check/nuke_r07/2393223-de_nuke-r07-p05.npz")
skip = {"_meta", "_map_meta"}   # _map_meta (the mesh's alignment) is in the files of the three maps besides de_dust2
print([k for k in paper.files if k not in skip and not np.array_equal(paper[k], ours[k])] or "equal")
EOF
```

`dust2_r09` and `dust2_r16` are equal too, made with `--capsule-unknown`; `dust2_r16`'s player 6 is scoped in about
half its frames, which `--scope none` does not find (compare its players 7 and 8). If the arrays differ, the cells that
differ show where your mesh differs from the paper's: compare `_meta.mesh.sha256` with the table above first.
`--pov-index` makes the round events those of OpenCS2's index; the tick tables give the same events for these rounds,
except that the last frames of a recording that ends with its round can differ.

### Bucket index, evaluation reserve and validation index

`python tools/prepare_data.py buckets` builds the files that stages 2-4 train on from the media index and the GT
visibility labels: the bucket index (`data.bucket_dir`), the evaluation reserve (`data.exclude_media_manifest`) and the
validation index (`validation.index`). It runs on the CPU. Every choice is a hashed or seeded order, so a rerun writes
the same files. Five subcommands:

| subcommand | what it does | writes |
|---|---|---|
| `split` | draws the held-out matches: whole matches, in which each map holds its share of all windows | a text file of match ids |
| `windows` | the windows of the matches that are not held out, with their visible pairs | the five bucket files |
| `reserve` | picks the evaluation reserve out of the bucket index: whole rounds | a text file of media ids |
| `validation` | picks the validation windows out of the windows of the held-out matches | a JSONL index |
| `weights` | solves the sampling weight of each bucket file | prints it |

Each subcommand prints a JSON summary (counts, and the sha256 of every file it wrote) and has `--help`; `weights` and
`validation` also print the config lines that use the files, `data.bucket_weights` and `validation.index_sha256` (unset,
both settings are the paper's). A refused option or input ends with a usage message and exit status 2.

**Reads**

| what | where |
|---|---|
| media index | `--media-index` (`data.media_index`) |
| GT visibility labels, `<media_id>.npz` ([format](inference.md#data)) | `--visibility-label-root` (`data.visibility_label_root`) |
| the recordings' tick tables (need pyarrow; only with `--dataset-root`) | `--dataset-root` (`data.dataset_root`) |
| the held-out matches | `--held-out`: a text file, one match id per line |
| the excluded media | `--exclude-media`: a text file, one media id per line |
| the bucket index (`reserve`, `weights`) | `--bucket-dir` (`data.bucket_dir`) |

**The paper's two lists** are in `configs/data/`: `held_out_matches.txt` (40 match ids, `sha256`
`09baa1611a25e1ff1dff60d0f032860a1f081180fa018d1dc12307b1b3764cce`), the input `--held-out` of `windows` and `validation`, and `excluded_media.txt` (227
media ids, `sha256` `c6f4f8471abd7e048784988073f6e735af505e06f198b10935ce3e90b039dc52`), the input `--exclude-media` of both: the matches the paper's validation windows
come from, and the recordings left out of the paper's files (for example videos that do not decode). OpenCS2 alone does
not give them.

#### Formats

- **Held-out matches** (`held_out_matches.txt`): text, one match id per line, ascending.
- **Excluded media** (`--exclude-media`): text, one media id per line; recordings that are left out of every file (for
  example videos that do not decode). Each id must be in the media index.
- **Bucket index**: five files, `train_q00-01.jsonl` ... `train_q07-10.jsonl`; file `i` holds the windows of map `i` of
  `--maps` (default the paper's four: `de_dust2`, `de_mirage`, `de_nuke`, `de_ancient`), and the files after the last
  map are empty, so each file is a slot of one map, not a range of a score. One JSON object per window, keys sorted,
  compact separators, one line each:
  `{"latent_key":"win_001288","map_name":"de_dust2","match_id":2392812,"media_id":"2392812-de_dust2-r09-p09","round":9,"start_frame":1288,"vis_available":true,"vis_pair_count":3}`.
  The rows are ordered by `media_id` and `start_frame`. A `(media_id, start_frame)` pair occurs once. The fields are
  those of [Data formats](training.md#data-formats) and `map_name`.
- **Evaluation reserve**: text, one media id per line, ascending; every id occurs in the bucket index.
- **Validation index**: the rows of the bucket index (the same fields), ordered by `map_name`, `media_id` and
  `start_frame`.

#### Splitting the matches

```bash
python tools/prepare_data.py buckets split --media-index data/media_index.jsonl --seed 1 \
    --out data/held_out_matches.txt
```

| option | default | meaning |
|---|---|---|
| `--media-index PATH` | required | the media index |
| `--out PATH` | required | the held-out matches |
| `--seed STR` | required | orders the draw |
| `--fraction F` | 0.1 | the share of all windows that is held out |
| `--exclude-matches PATH` | | matches that are never drawn, one id per line: for example the matches of the raw-video manifests, so that the held-out matches are unseen by stage 1 |
| `--keep-matches PATH` | | matches that stay held out, one id per line |

A match is held out whole, on every map it was played on. The windows of a match are counted on each map as `windows`
writes them without `--dataset-root`; every map has to hold `fraction / (number of maps)` of all windows. The matches
of `--keep-matches` are held out; the others, but those of `--exclude-matches`, are taken in the order of
`sha256(seed NUL match_id)`, a match if some map it was played on is still short of its share, until no map is short.
The summary lists the matches, match-maps (a match on one map), rounds and recordings of each side.

The paper's 40 held-out matches are in `configs/data/held_out_matches.txt` (303 training matches with 10,480 rounds and
40 held-out matches with 1,373 rounds, of 569 match-maps and 11,853 rounds). Give them to `windows` and `validation`
with `--held-out`; the summary of either prints these counts for the media index.

#### The bucket index

```bash
python tools/prepare_data.py buckets windows --media-index data/media_index.jsonl \
    --held-out configs/data/held_out_matches.txt --visibility-label-root data/vislabels \
    --exclude-media configs/data/excluded_media.txt --out-dir data/buckets
```

The paper's four maps, with the tick check on de_dust2 as the paper's file has it:

```bash
python tools/prepare_data.py buckets windows --media-index data/media_index.jsonl \
    --held-out configs/data/held_out_matches.txt --visibility-label-root data/vislabels \
    --exclude-media configs/data/excluded_media.txt --dataset-root data/opencs2 \
    --tick-gate-maps de_dust2 --out-dir data/buckets
```

| option | default | meaning |
|---|---|---|
| `--media-index PATH` | required | the media index |
| `--held-out PATH` | required | the held-out matches; their windows are not written |
| `--visibility-label-root DIR` | required | the GT visibility labels |
| `--out-dir DIR` | required | receives the five bucket files |
| `--maps MAP ...` | the paper's four | the maps, in the order of the bucket files (at most five) |
| `--exclude-media PATH` | | media ids to leave out |
| `--dataset-root DIR` | | the OpenCS2 recordings: with it a window must pass the tick tables (below) |
| `--tick-gate-maps MAP ...` | every map of `--maps` | the maps whose windows are checked against their tick tables (needs `--dataset-root`) |

**Windows.** A window is 41 latent frames: the 161 video frames at the source frames `start, start + 2, ..., start +
320` of a recording, so it needs `start + 321` source frames in the video (`video_frames` of the media row). The grid
windows start at the multiples of 161 source frames, the midpoints 80 after them (`161 k + 80`): neighbors share half
their frames. Without `--dataset-root`, every grid window and every midpoint that fits in the video is written (a
recording shorter than 321 frames has none).

With `--dataset-root`, a grid window is kept only if the recording's tick table serves it: no frame of it without a
tick, none whose last tick is more than 1.5 / 64 s before it, no gap of more than 1.5 / 64 s between ticks inside it
(the check of `worldcast.data.controls.frame_tick_ranges`), and the player alive at every tick of its span. A midpoint
is then written between each two neighboring kept grid windows only (its ticks lie within theirs). The paper's de_dust2
file was built so, the files of the other three maps without the check.

**Visible pairs.** `vis_pair_count` is the number of (player, video frame) pairs of the window in which another player is
visible: the labels `_binary_visible` where `_binary_eval_valid` ([Data](inference.md#data)), the recording player's own
row left out. A recording without a label file adds no window (the summary counts it under `left_out_media`, with the
excluded recordings and those without a window); every written row has `vis_available` true.

**The sampling weights.** The summary ends with `slot_weights`, the weight of each file with which the sampler (a window
weighs its file's weight times `(1 + vis_pair_count) ** 0.5`, the windows without pairs floored to 15 % of the total)
draws every map equally often, the largest weight 1, and `draw_share_with_loader_weights`, the share of the draws each
file gets under `worldcast.data.training.BUCKET_WEIGHTS`; `matches_loader_weights` says whether the two are equal.
`buckets weights --bucket-dir data/buckets` prints the same for existing files, with the `data.bucket_weights` line to
put in `train_paths.yaml`. For the paper's files it prints `[0.687417402, 0.86527334, 1.0, 0.92999783, 1.0]`, which are
`BUCKET_WEIGHTS`, and a quarter of the draws for each map.

#### The evaluation reserve

```bash
python tools/prepare_data.py buckets reserve --bucket-dir data/buckets --maps de_dust2 \
    --target-capacity 2806 --out data/eval_reserve.txt
```

| option | default | meaning |
|---|---|---|
| `--bucket-dir DIR` | required | the bucket index |
| `--out PATH` | required | the reserved media ids |
| `--target-capacity N` | required | the capacity (below) to reach |
| `--maps MAP ...` | all | reserve from the windows of these maps only |
| `--seed STR` | the paper's | orders the draw |

The reserve is whole rounds: every recording of a reserved round is reserved, so that no recording of a reserved round
is trained on. A recording's capacity is the number of its grid windows (a start that is a multiple of 161) with at
least 100 visible pairs, no two sharing a frame (taken from the first, each 322 or more source frames after the last); a
round's is the sum over its recordings, and a round without capacity is never taken. The matches, in the order of
`sha256(seed NUL "match" NUL match_id)`, give a round each (the first of its rounds in the order of
`sha256(seed NUL "round" NUL match_id NUL round)`), then a second, up to four rounds of a match, until the capacity
reaches `--target-capacity`; the command refuses a target out of reach. Set `data.exclude_media_manifest` to the file:
the loader drops these recordings and checks that each occurs in the bucket index.

For the paper's files, `--maps de_dust2 --target-capacity 2806` writes the paper's `configs/data/eval_reserve.txt`:
1,591 recordings of 161 rounds of 146 matches.

#### The validation index

```bash
python tools/prepare_data.py buckets validation --media-index data/media_index.jsonl \
    --held-out configs/data/held_out_matches.txt --visibility-label-root data/vislabels \
    --exclude-media configs/data/excluded_media.txt --dataset-root data/opencs2 \
    --tick-gate-maps de_dust2 --out data/validation_index.jsonl
```

It takes the options of `windows` (`--out` is the index file, not a directory), and:

| option | default | meaning |
|---|---|---|
| `--per-map N` | 16 | windows per map |
| `--seed STR` | the paper's | orders the draw |

The windows are those `windows` writes for the held-out matches. Per map, the candidates are the windows with at least
the 0.75 quantile (linear interpolation) of the map's `vis_pair_count`; they are taken in the order of
`sha256(seed NUL map_name NUL media_id NUL start_frame)`, one window per recording, the first `--per-map`. The command
refuses a map with fewer recordings among its candidates. The paper's 64 windows (16 per map, four maps) are such a
draw. `tools/evaluate.py` and the in-training validation read the file as `validation.index`; they check it by sha256
against the paper's index (`validation.index_sha256` sets another).

#### Check against the paper's files

The bucket files of the paper carry more fields than the format above (`player_slot, window_source, vis_valid_count,
vis_visible_frames, vis_train_valid_count, quality_available`; de_dust2's also the discriminator's quality and the
recordings' paths), and de_dust2's is ordered by match and quality bucket. From the
paper's inputs (its media index, `configs/data/held_out_matches.txt`, `configs/data/excluded_media.txt` and its
visibility labels, and for de_dust2 the OpenCS2 recordings) `windows` writes the paper's rows on the fields above; the
summary prints the sha256 of each file, and these are the sha256 of the paper's files cut to these fields:

| file | map | rows | sha256 |
|---|---|---:|---|
| `train_q00-01.jsonl` | `de_dust2` | 651,666 | `c4e80f0dc62b89eacba37b5e3ba10e34106964de3726576f1d943f64679a8b74` |
| `train_q01-03.jsonl` | `de_mirage` | 579,532 | `f079d10ca7beae14047bd52d13e12d727ab7992fa94c50eab5564457638c7e30` |
| `train_q03-05.jsonl` | `de_nuke` | 488,562 | `79cb2fa905be93f7bd854064bd08d74c30a287bcce3ebc2098a818beab065058` |
| `train_q05-07.jsonl` | `de_ancient` | 504,945 | `f340cc4c6fd1edc8fd455fb054d10e22c13da5649afa04e3f129078b00b7bfe1` |
| `train_q07-10.jsonl` | empty | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The paper's evaluation reserve, `configs/data/eval_reserve.txt`, is written byte for byte: 1,591 ids,
`6f8311e7efa8091ea77584033dc44a4de8849d23d9894cf724e004e2aab88346`. The rows of the paper's validation index cut to the fields above hash to `79d4d99dd6410ced7f25e639d6f37550a5ec578c93c2e8bb641605ef67f2d924`;
`buckets weights` on the paper's files prints `BUCKET_WEIGHTS`.

### Latent cache

`python tools/prepare_data.py latent-cache` encodes windows of the recordings with the Wan2.2 VAE and writes the latent
cache that stages 2-4 and the evaluation read (`data.latent_cache_root`; format:
[Data formats](training.md#data-formats)). In training, a teammate's windows are the candidates of a client's memory
frames. The inference client reads only latent 0 of its window from this layout
([docs/inference.md](inference.md#data)); the example cases ship it.

**Reads**

| what | where |
|---|---|
| media index | `data.media_index` |
| the recordings' videos (`video_path` of each media row) | under `data.dataset_root` |
| Wan2.2 snapshot (`Wan2.2_VAE.pth`) | `model.wan22_root` |
| the windows | the bucket files of `data.bucket_dir`; or `--windows`; or `--stride` |

The paths come from `train_paths.yaml` (`--config train_paths.yaml`; no stage config is needed) or `--set`.

**Writes** `<media_id>.npz` into `--out-dir` (default `data.latent_cache_root`): a member `win_<start:06d>` per window,
float16 `[1, 41, 48, 24, 42]`. A window is the VAE's latents of the 161 video frames at the source frames
`start, start + 2, ..., start + 320` (41 latent frames), resized bilinearly to 384 x 672 and mapped to
`(x / 255 - 0.5) * 2`, with the VAE's own latent normalization. The VAE runs in bf16 on CUDA (weights and pixels), as the paper's cache.

| option | default | meaning |
|---|---|---|
| `--config YAML`, `--set KEY=VALUE` | | the paths, as in every tool |
| `--windows JSONL ...` | the bucket files | JSONL files with a `media_id` and a `start_frame` per line (the bucket files and the validation index are such files; other fields are ignored) |
| `--stride N` | | instead of `--windows`: windows from every N-th source frame of every video, as far as a whole window fits |
| `--media-id ID [ID ...]` | all | only the windows of these recordings |
| `--out-dir DIR` | `data.latent_cache_root` | where the archives go |
| `--shard I/N` | `0/1` | this process encodes recordings I, I + N, ... of the sorted list (I from 0) |
| `--batch-size N` | 1 | windows per VAE call; each window of a batch holds 0.5 GB of float32 pixels in host memory |
| `--device` | `cuda` | `cpu` encodes in float32, which is not what the paper's cache used |

**Resuming.** A run skips the windows an archive already holds, so the same command continues a stopped run, and windows
added to the list are added to their archives (the members already there are kept). An archive is written to
`<media_id>.npz.part` and renamed, so a run that is killed leaves no half archive. A recording that cannot be decoded is
reported (stderr, and the exit status is 1) and the run goes on; the others are written.

**One GPU**

```bash
python tools/prepare_data.py latent-cache --config train_paths.yaml
```

**N GPUs.** One process per GPU, each its part of the recordings (`--shard` of the same N for every process, and the
same windows):

```bash
N=8
for i in $(seq 0 $((N - 1))); do
    CUDA_VISIBLE_DEVICES=$i python tools/prepare_data.py latent-cache --config train_paths.yaml \
        --shard $i/$N > latent_cache_$i.log 2>&1 &
done
wait
```

`N` need not equal the number of GPUs (several processes can share one). Run the same loop again to finish what a failed
or killed process left.

Other window lists:

```bash
# the windows of a validation index, or any JSONL of media_id and start_frame
python tools/prepare_data.py latent-cache --config train_paths.yaml --windows configs/data/validation_index.jsonl

# one recording tiled by a window every 161 source frames (5 s)
python tools/prepare_data.py latent-cache --config train_paths.yaml --stride 161 --media-id 2391547-de_dust2-r01-p00 \
    --out-dir runs/latent_cache_check
```

**Size.** A window takes 3,967,488 bytes of values, 3,967,616 with the `.npy` header: 3.97 MB. A recording's archive is
its window count times that (a 110 s recording tiled with `--stride 161` has 20 windows, 79 MB); the total is the number
of windows times 3.97 MB. The paper's bucket files list 2,224,705 windows of 103,979 recordings (8.8 TB; at
most 59 windows, 234 MB, per recording); with the validation windows the paper's cache has
2,493,852 windows of 117,595 recordings (9.9 TB, 9.0 TiB).

**Relation to the paper's cache.** The windows, the frame selection, the resize, the scaling, the VAE's normalization
and its bf16 precision are those of the paper's cache; a re-encode is not bit-equal with it.

## The paper's files

These digests identify the files of the paper's data, so that a file you build can be compared with them. The data of
the evaluation sets (their windows' latents, labels and tick tables) is the dataset
[ZiyangYe/WorldCast-data](https://huggingface.co/datasets/ZiyangYe/WorldCast-data), fetched by
`tools/download_eval_data.py`. The lists and the evaluation sets ship with the repository, verbatim: `configs/data/` holds what training reads (the held-out
matches, the excluded media, the evaluation reserve, the validation index) and `configs/eval/` the paper's evaluation
sets. Every window and round of the evaluation sets and of the validation index lies in the 40 held-out matches. To
train on exactly the paper's training data, leave out the held-out matches, the excluded media and the evaluation
reserve: the reserve is 1,591 `de_dust2` recordings of training matches that training does not read either, none of
them from a held-out match.

| file | what | size | sha256 |
|---|---|---:|---|
| media index | 118,390 rows, one per recording of the four maps | 75,576,789 B | `3215a9c84f810436acad1c7ea59d2322b6b4b6b2079bc4c302940217e5c6de8b` |
| bucket index, `train_q00-01.jsonl` | `de_dust2`, 651,666 windows | 634,879,426 B | `332b519499ca22059bee75ecde0c35cc5c447e9d2b1743ae8b50305268d2eda0` |
| bucket index, `train_q01-03.jsonl` | `de_mirage`, 579,532 windows | 186,730,549 B | `0c3cd8c3c6cad50d99c4c6b15f419e25a9546dc2ffdd6ad22c7f9526639f1a7d` |
| bucket index, `train_q03-05.jsonl` | `de_nuke`, 488,562 windows | 155,485,978 B | `25ee4921e1b6118a69c366ff857838688db1e5bd566baf8e4f6d06cea5e6ab4b` |
| bucket index, `train_q05-07.jsonl` | `de_ancient`, 504,945 windows | 163,727,246 B | `a3e4fdda5013d488415a95fe5206d3ef61dc5c0af81b0da41e6dcf35ee63b246` |
| bucket index, `train_q07-10.jsonl` | empty | 0 B | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| evaluation reserve, `configs/data/eval_reserve.txt` | 1,591 media ids, all `de_dust2` | 39,775 B | `6f8311e7efa8091ea77584033dc44a4de8849d23d9894cf724e004e2aab88346` |
| validation index, `configs/data/validation_index.jsonl` | 64 windows, 16 per map | 31,453 B | `a15580a409d02b580e7220073e1ff5f037724c49e749b86905e335fe80030c49` |
| ten-second windows of Tables 1 and 2, `configs/eval/windows.jsonl` | 256 windows, 64 per map, two views each | 114,554 B | `ed151280cd6879c23babb609735a5ebad67e8ded59669bdea9878de1217dd1a5` |
| their views, `configs/eval/window_views.jsonl` | 512 rows, one per view, what the metrics are computed on | 198,198 B | `72709db3c3553f7e910f0ec1c8feea2519eef85dc1cd81a5c585047331e63b74` |
| round index of Table 3, `configs/eval/round_index.jsonl` | 96 rows: 32 rounds, 3 clients each | 49,531 B | `073cf4b851950032f037d9e266da78e666a45895a3b737ea8df2611c4e8bf10e` |
| held-out matches, `configs/data/held_out_matches.txt` | 40 match ids | 320 B | `09baa1611a25e1ff1dff60d0f032860a1f081180fa018d1dc12307b1b3764cce` |
| excluded media, `configs/data/excluded_media.txt` | 227 media ids | 5,870 B | `c6f4f8471abd7e048784988073f6e735af505e06f198b10935ce3e90b039dc52` |

In the paper's bucket directory the five files hold one map each, not a range of a quality score, and the map's
sampling weight is the entry of `BUCKET_WEIGHTS` at its position (`BUCKET_NAMES`), solved so that each map is drawn 25%
of the time. The meshes' digests are in [Meshes](#meshes). The round index of Table 3 is read by `tools/run_session.py`
(`paths.round_index`, [docs/inference.md](inference.md#data)). The two evaluation sets are drawn from the held-out
matches by the rules of the paper's Appendix (evaluation sets); no step of `tools/prepare_data.py` builds them, so use
the files. The validation index keeps the paper's file byte for byte, so its sha256 matches `validation.index_sha256`;
its `source_*` fields name where the paper's recordings were stored and are not read.

## The game's map files

The collision and the visual meshes are exported from Counter-Strike 2's map files, which are Valve's. This repository
and the Hub carry no map file and no export of one: each user exports the meshes from their own copy of the game with
the open tool ValveResourceFormat (`tools/prepare_data.py meshes`) and checks the result against the digests of the
paper's meshes ([collision](#collision-meshes), [visual](#visual-meshes)). Share the digests, not the meshes.

The terms that apply to the game's files are Valve's, in the [Steam Subscriber
Agreement](https://store.steampowered.com/subscriber_agreement/) (the page says it was last updated on September 10,
2026; quoted as read on October 7, 2026), among them:

> **2.A** Valve hereby grants, and you accept, a non-exclusive license and right, to use the Content and Services for
> your personal, non-commercial use (except where commercial use is expressly allowed herein or in the applicable
> Subscription Terms).

> **2.G** Except as otherwise permitted under this Agreement (including any Subscription Terms or Rules of Use), or
> under applicable law notwithstanding these restrictions, you may not, in whole or in part, copy, photocopy,
> reproduce, publish, distribute, translate, reverse engineer, derive source code from, modify, disassemble,
> decompile, create derivative works based on, or remove any proprietary notices or labels from the Content and
> Services or any software accessed via Steam without the prior consent, in writing, of Valve.

> **2.G** You are entitled to use the Content and Services for your own personal use, but you are not entitled to: (i)
> sell, grant a security interest in or transfer reproductions of the Content and Services to other parties in any way,
> nor to rent, lease or license the Content and Services to others without the prior written consent of Valve, except
> to the extent expressly permitted elsewhere in this Agreement (including any Subscription Terms or Rules of Use);

> **2.D** Valve appreciates the community of Subscribers that creates fan art, fan fiction, and audio-visual works that
> reference Valve games ("Fan Art"). You may incorporate content from Valve games into your Fan Art. Except as
> otherwise set forth in this Section or in any Subscription Terms, you may use, reproduce, publish, perform, display
> and distribute Fan Art that incorporates content from Valve games however you wish, but solely on a non-commercial
> basis.

These are Valve's terms, not WorldCast's, and WorldCast does not interpret them: read the agreement, and ask Valve
where it is unclear. You are responsible for complying with them in what you export, keep and share. WorldCast's license
(Apache-2.0) covers its code, not Valve's files. Counter-Strike and Counter-Strike 2 are trademarks of Valve
Corporation; WorldCast is not affiliated with or endorsed by Valve.
