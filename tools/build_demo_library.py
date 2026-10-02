#!/usr/bin/env python3
"""Build a demo library (``library.json``, seat previews, mock clips, radar images) from a small YAML manifest.

Per round start the manifest names the map, the source frame the room starts at and, per seat, the rendered video the
mock engine plays (one clip per seat: a client's own render of that round). Optional: the round's OpenCS2 tick
tables, for each seat's spawn pose, team and primary weapon at the start; a nav mesh, for the minimap's radar image.

Manifest (paths relative to the manifest)::

    clip_seconds: 10
    rounds:
      - id: ancient-r02
        map: de_ancient
        label: Round 2
        video_start: 178                  # first video frame of the clips (16 fps); start_frame = 2 x this
        cover: 8                          # the seat whose first frame shows the round in the lobby
        ticks: ticks/rounds/match_id=2392968/map_name=de_ancient/round=02   # holds player=NN/ticks.parquet
        nav: meshes/de_ancient_nav.npz    # V [n, 3], F [m, 3] in engine units (or a .glb, needs trimesh)
        seats:
          - {seat: 0, video: renders/2392968-de_ancient-r02-p00.mp4}
          - {seat: 5, video: renders/2392968-de_ancient-r02-p05.mp4, team: T, spawn: [x, y, z, yaw, pitch]}

Needs ``imageio-ffmpeg`` (video), ``pyarrow`` (ticks) and Pillow. Example::

    python tools/build_demo_library.py --manifest library.yaml --out runs/demo-library
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo.library import DEFAULT_LOADOUT  # noqa: E402
from worldcast.data.actions import normalize_weapon_name  # noqa: E402

FPS = 16
SOURCE_FPS = 32
FRAME_SIZE = (672, 384)
RADAR_PX_PER_UNIT = 0.4
RADAR_MARGIN_PX = 24
TEAMS = {2: "T", 3: "CT"}
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
OTHER = {"<none>", "<unk>", "c4", "planted_c4", "taser", "inferno"}


def extract_clip(video: Path, out_dir: Path, start: int, frames: int, quality: int) -> int:
    """Write ``frames`` frames from video frame ``start`` on as ``000000.jpg ...``; returns how many were written."""
    import imageio_ffmpeg

    out_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio_ffmpeg.read_frames(
        str(video),
        input_params=["-ss", f"{start / FPS:.4f}"],
        output_params=["-vf", f"scale={FRAME_SIZE[0]}:{FRAME_SIZE[1]}"],
    )
    meta = next(reader)
    w, h = meta["size"]
    written = 0
    for raw in reader:
        if written == frames:
            break
        image = Image.frombytes("RGB", (w, h), raw)
        image.save(out_dir / f"{written:06d}.jpg", quality=quality)
        written += 1
    reader.close()
    return written


def seat_from_ticks(ticks_dir: Path, seat: int, start_frame: int) -> dict:
    """Spawn pose, team and loadout of one player at source frame ``start_frame`` (32 fps)."""
    import pyarrow.parquet as pq

    table = pq.read_table(
        ticks_dir / f"player={seat:02d}" / "ticks.parquet",
        columns=["t", "x", "y", "z", "yaw", "pitch", "team_num", "input_weapon"],
    ).to_pydict()
    i = int(np.argmin(np.abs(np.asarray(table["t"]) - start_frame / SOURCE_FPS)))
    team = TEAMS.get(int(table["team_num"][i]), "T")
    held = Counter(normalize_weapon_name(w) for w in table["input_weapon"])
    loadout = list(DEFAULT_LOADOUT[team])
    primaries = [
        w
        for w, _ in held.most_common()
        if w not in PISTOLS | GRENADES | OTHER and not w.startswith("knife") and w != "bayonet"
    ]
    pistols = [w for w, _ in held.most_common() if w in PISTOLS]
    grenades = [w for w, _ in held.most_common() if w in GRENADES]
    loadout[0] = primaries[0] if primaries else loadout[0]
    loadout[1] = pistols[0] if pistols else loadout[1]
    loadout[3] = grenades[0] if grenades else loadout[3]
    spawn = [round(float(table[k][i]), 2) for k in ("x", "y", "z", "yaw", "pitch")]
    return {"team": team, "spawn": spawn, "loadout": loadout}


def load_mesh(path: Path):
    if path.suffix == ".npz":
        data = np.load(path)
        return data["V"], data["F"]
    import trimesh

    scene = trimesh.load(path, force="scene", skip_materials=True)
    to_engine = np.linalg.inv(
        np.array([[0, 0.0254, 0], [0, 0, 0.0254], [0.0254, 0, 0]])
    )  # glTF metres -> units
    vertices, faces, offset = [], [], 0
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        vertices.append(trimesh.transform_points(mesh.vertices, transform) @ to_engine.T)
        faces.append(mesh.faces + offset)
        offset += len(mesh.vertices)
    return np.concatenate(vertices), np.concatenate(faces)


def draw_radar(nav: Path, out: Path) -> dict:
    """A top-down radar from a nav mesh: walkable floor shaded by height, a light outline; returns its transform."""
    vertices, faces = load_mesh(nav)
    s, m = RADAR_PX_PER_UNIT, RADAR_MARGIN_PX
    x0, y0 = float(vertices[:, 0].min()), float(vertices[:, 1].max())
    width = int((vertices[:, 0].max() - x0) * s) + 2 * m
    height = int((y0 - vertices[:, 1].min()) * s) + 2 * m
    zs = vertices[faces][:, :, 2].mean(1)
    lo, hi = np.percentile(zs, [2, 98])
    floor = Image.new("L", (width, height), 0)
    mask = Image.new("L", (width, height), 0)
    paint, cover = ImageDraw.Draw(floor), ImageDraw.Draw(mask)
    for tri, z in sorted(zip(vertices[faces], zs), key=lambda tz: tz[1]):
        points = [(m + s * (x - x0), m + s * (y0 - y)) for x, y, _ in tri]
        shade = int(44 + 40 * np.clip((z - lo) / max(hi - lo, 1.0), 0, 1))
        paint.polygon(points, fill=shade)
        cover.polygon(points, fill=255)
    mask = mask.filter(ImageFilter.MaxFilter(3))
    edge = ImageChops.subtract(mask, mask.filter(ImageFilter.MinFilter(3)))
    rgb = Image.merge("RGB", [floor.point(lambda v: int(v * k)) for k in (0.92, 0.97, 1.1)])
    outline = Image.new("RGB", (width, height), (168, 190, 220))
    rgb = Image.composite(outline, rgb, edge.point(lambda v: int(v * 0.55)))
    rgba = rgb.convert("RGBA")
    rgba.putalpha(mask.point(lambda v: 235 if v else 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(out, lossless=False, quality=88)
    return {"x0": round(x0, 2), "y0": round(y0, 2), "cx": m, "cy": m, "scale": s}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--out", required=True, help="output directory (library.json and its files)"
    )
    parser.add_argument("--quality", type=int, default=86, help="JPEG quality of the clip frames")
    args = parser.parse_args(argv)
    import yaml

    manifest_path = Path(args.manifest).resolve()
    manifest = yaml.safe_load(manifest_path.read_text())
    here, out = manifest_path.parent, Path(args.out).resolve()
    frames = int(float(manifest.get("clip_seconds", 10)) * FPS)
    rounds = []
    for r in manifest["rounds"]:
        start_frame = 2 * int(r["video_start"])
        radar = None
        if r.get("nav"):
            radar = {
                "image": f"maps/{r['map']}.webp",
                **draw_radar(here / r["nav"], out / "maps" / f"{r['map']}.webp"),
            }
        seats = []
        for s in r["seats"]:
            seat = int(s["seat"])
            info = seat_from_ticks(here / r["ticks"], seat, start_frame) if r.get("ticks") else {}
            info.update({k: s[k] for k in ("team", "spawn", "loadout") if k in s})
            clip = f"{r['id']}/p{seat:02d}"
            n = extract_clip(
                here / s["video"], out / clip, int(r["video_start"]), frames, args.quality
            )
            preview = f"{r['id']}/p{seat:02d}.jpg"
            Image.open(out / clip / "000000.jpg").save(out / preview, quality=82)
            media_id = s.get("media_id") or Path(s["video"]).name.split("__")[0]
            seats.append(
                {"seat": seat, "media_id": media_id, **info, "preview": preview, "clip": clip}
            )
            print(
                f"  {r['id']} seat {seat}: {n} frames, {info.get('team')}, spawn"
                f" {info.get('spawn')}",
                flush=True,
            )
        rounds.append(
            {
                "id": r["id"],
                "map": r["map"],
                "label": r["label"],
                "start_frame": start_frame,
                "note": r.get("note", ""),
                "cover": int(r.get("cover", seats[0]["seat"])),
                "radar": radar,
                "seats": seats,
            }
        )
    (out / "library.json").write_text(json.dumps({"rounds": rounds}, indent=1))
    print(f"{out / 'library.json'}: {len(rounds)} round start(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
