"""Build a demo library (``library.json``, previews, mock clips) from a manifest.

Per round start the manifest names the map, the video frame the room starts at and, per seat, the
rendered video the mock client plays (one clip per seat: a client's own render of that round). The
recorded round (``match_id``, ``round``) is what the WorldCast client plays; with the data paths
(``media_index``, ``dataset_root``; docs/inference.md, "Data") each seat's team, spawn pose and
loadout come from its tick table, else the manifest gives ``team`` and ``spawn`` per seat.

Live play shows only the players a human plays, and every window of a block attends the round's
first frame, so a first frame that shows people would show ghosts. A seat of a recorded round is
therefore offered only if no other player may be seen in its first frame
(:func:`~worldcast.demo.library.players_in_first_frame`), by the dataset's GT visibility labels at
the round start (``visibility_label_root``, the data paths' ``paths.visibility_label_root``;
docs/inference.md, "Data"). Live play does not track the scope either, and the generator draws a
black scope view from the first frame when the scope is unknown for the weapon held: a seat whose
loadout holds a weapon that zooms (:data:`~worldcast.data.camera.SCOPE_ZOOM_FOV`) is left out
too, whether the loadout comes from its tick table or from the manifest. Each round start logs how
many of its seats it offers and why a seat was left out; one none of whose seats qualifies is left
out, a ``cover`` that was left out gives way to the first seat that remains, and a library with no
round start is refused. A seat without a recording has no labels and is offered as it is, unless
its loadout zooms.

Manifest (paths relative to the manifest)::

    clip_seconds: 10
    media_index: data/media_index.jsonl
    dataset_root: data/opencs2
    visibility_label_root: data/vislabels
    round_starts:
      - id: ancient-r02
        map: de_ancient
        label: Round 2
        note: Ten players                  # optional: a line under the title
        match_id: 2392968
        round: 2
        video_start: 178                   # first video frame of the clips (16 fps)
        cover: 8                           # optional: the slot whose first frame shows the round
        seats:
          - {slot: 0, video: renders/p00.mp4}
          - {slot: 5, video: renders/p05.mp4, team: T, spawn: [x, y, z, yaw, pitch],
             loadout: [ak47, glock, knife, hegrenade]}

A seat's ``team``, ``spawn`` and ``loadout`` replace the tick table's. Needs ``imageio-ffmpeg``
(video), ``pyarrow`` (tick tables) and Pillow.
"""

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from worldcast.data import MediaIndex, TickTable, normalize_weapon_name, read_player_ticks
from worldcast.data.latents import FPS, FRAME_SIZE, SOURCE_FRAMES_PER_VIDEO_FRAME

from .library import (
    DEFAULT_LOADOUT,
    Library,
    RoundStart,
    Seat,
    players_in_first_frame,
    zooming_weapons,
)

__all__ = ["build_library"]

log = logging.getLogger(__name__)

#: JPEG quality of the clip frames and of the previews.
CLIP_QUALITY, PREVIEW_QUALITY = 86, 82
#: Engine team ids.
TEAMS = {2: "T", 3: "CT"}
#: Weapon classes, to pick a seat's loadout from the weapons it held.
PISTOLS = {
    "glock",
    "usp_silencer",
    "hkp2000",
    "p250",
    "deagle",
    "fiveseven",
    "tec9",
    "elite",
    "cz75a",
    "revolver",
}
GRENADES = {"hegrenade", "smokegrenade", "flashbang", "molotov", "incgrenade", "decoy"}
NOT_PRIMARY = PISTOLS | GRENADES | {"<none>", "<unk>", "c4", "planted_c4", "taser", "inferno"}


def extract_clip(video: Path, out_dir: Path, start: int, frames: int) -> int:
    """Write the frames of a video from one frame on as ``000000.jpg``, ``000001.jpg``, ...

    Args:
        video (Path): the rendered video, 16 fps.
        out_dir (Path): the clip's directory.
        start (int): the first video frame.
        frames (int): how many frames at most.

    Returns:
        int: the frames written.

    Raises:
        ValueError: the video has no frame from ``start`` on.
    """
    import imageio_ffmpeg

    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = FRAME_SIZE
    reader = imageio_ffmpeg.read_frames(
        str(video),
        input_params=["-ss", f"{start / FPS:.4f}"],
        output_params=["-vf", f"scale={width}:{height}"],
    )
    size = next(reader)["size"]
    written = 0
    for raw in reader:
        if written == frames:
            break
        Image.frombytes("RGB", size, raw).save(out_dir / f"{written:06d}.jpg", quality=CLIP_QUALITY)
        written += 1
    reader.close()
    if not written:
        raise ValueError(f"{video} has no frame from frame {start} on")
    return written


def seat_from_ticks(ticks: TickTable, source_fps: float, start_frame: int) -> dict:
    """A player's ``team``, its ``spawn`` pose at source frame ``start_frame`` and its ``loadout``:
    the primary weapon, the pistol and the grenade it held longest in the round."""
    i = int(np.argmin(np.abs(ticks.t - start_frame / source_fps)))
    team = TEAMS.get(int(ticks.team_num[i]), "T")
    counts = Counter(normalize_weapon_name(weapon) for weapon in ticks.input_weapon)
    held = [weapon for weapon, _ in counts.most_common()]
    primaries = [w for w in held if w not in NOT_PRIMARY and not w.startswith(("knife", "bayonet"))]
    primary, pistol, knife, grenade = DEFAULT_LOADOUT[team]
    loadout = [
        next(iter(primaries), primary),
        next((w for w in held if w in PISTOLS), pistol),
        knife,
        next((w for w in held if w in GRENADES), grenade),
    ]
    pose = (ticks.x, ticks.y, ticks.z, ticks.yaw, ticks.pitch)
    return {"team": team, "spawn": [round(float(v[i]), 2) for v in pose], "loadout": loadout}


def build_library(manifest_path: str | Path, out: str | Path) -> Path:
    """Build the library of a manifest (the module docstring).

    Args:
        manifest_path (str | Path): the manifest YAML.
        out (str | Path): the output directory: ``library.json``, clips, previews.

    Returns:
        Path: the library's ``library.json``.

    Raises:
        ValueError: a seat of a recorded round has no ``visibility_label_root`` to be checked
            against, a seat lacks its team and spawn, or no round start offers a seat.
    """
    manifest_path, out = Path(manifest_path).resolve(), Path(out).resolve()
    manifest = yaml.safe_load(manifest_path.read_text())
    here = manifest_path.parent
    frames = int(float(manifest.get("clip_seconds", 10)) * FPS)
    media_index = None
    if "media_index" in manifest:
        media_index = MediaIndex.load(here / manifest["media_index"])
    labels = None
    if "visibility_label_root" in manifest:
        labels = here / manifest["visibility_label_root"]
    round_starts = {}
    for r in manifest["round_starts"]:
        start_frame = SOURCE_FRAMES_PER_VIDEO_FRAME * int(r["video_start"])
        recorded = {}
        if media_index is not None and "match_id" in r:
            recorded = media_index.slots((int(r["match_id"]), r["map"], int(r["round"])))
        seats = []
        for s in r["seats"]:
            slot = int(s["slot"])
            seat = {}
            if slot in recorded:
                if labels is None:
                    raise ValueError(
                        f"{r['id']}, seat {slot}: a seat of a recorded round is offered only if no"
                        " other player may be seen in its first frame: give visibility_label_root"
                        " (paths.visibility_label_root) in the manifest"
                    )
                seen = players_in_first_frame(labels, recorded[slot].media_id, slot, start_frame)
                if seen:
                    log.info(
                        "%s, seat %d left out: players %s may be seen in its first frame",
                        r["id"],
                        slot,
                        ", ".join(map(str, seen)),
                    )
                    continue
                ticks = read_player_ticks(recorded[slot], here / manifest["dataset_root"])
                seat = seat_from_ticks(ticks, recorded[slot].fps, start_frame)
            seat.update({key: s[key] for key in ("team", "spawn", "loadout") if key in s})
            if "spawn" not in seat or "team" not in seat:
                raise ValueError(
                    f"{r['id']}, seat {slot}: give its team and spawn, or the recorded round"
                    " (match_id, round) with media_index and dataset_root"
                )
            if zooming := zooming_weapons(seat.get("loadout", ())):
                log.info(
                    "%s, seat %d left out: its loadout holds %s, which zooms (live play does not"
                    " track the scope)",
                    r["id"],
                    slot,
                    ", ".join(zooming),
                )
                continue
            clip, preview = f"{r['id']}/p{slot:02d}", f"{r['id']}/p{slot:02d}.jpg"
            extract_clip(here / s["video"], out / clip, int(r["video_start"]), frames)
            Image.open(out / clip / "000000.jpg").save(out / preview, quality=PREVIEW_QUALITY)
            seats.append(Seat.from_json({"slot": slot, **seat, "preview": preview, "clip": clip}))
        if not seats:
            log.info("%s left out: none of its %d seats qualifies", r["id"], len(r["seats"]))
            continue
        log.info("%s offers %d of %d seats", r["id"], len(seats), len(r["seats"]))
        cover = int(r.get("cover", seats[0].slot))
        if cover not in {seat.slot for seat in seats}:
            log.info(
                "%s: seat %d is not offered; seat %d is its cover", r["id"], cover, seats[0].slot
            )
            cover = seats[0].slot
        round_starts[r["id"]] = RoundStart(
            id=r["id"],
            map=r["map"],
            label=r["label"],
            start_frame=start_frame,
            seats=tuple(seats),
            note=r.get("note", ""),
            cover=cover,
            match_id=r.get("match_id"),
            round=r.get("round"),
        )
    if not round_starts:
        raise ValueError(
            f"{manifest_path}: no round start offers a seat whose first frame shows nobody and"
            " whose loadout holds no weapon that zooms (the log says why each was left out);"
            " nothing to build"
        )
    return Library(round_starts, out).save()
