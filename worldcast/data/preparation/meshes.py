"""The meshes of the maps: exported from the game's map files.

``--vpk-dir`` runs the command-line tool of ValveResourceFormat (Source 2 Viewer) on ``<map>.vpk``,
the map's package in the game's files, and exports a glTF binary of the map; ``--glb`` takes one
that is exported already. ``--kind collision`` (the default) is the map's world physics, which
``data.collision_meshes`` names; ``--kind visual`` is its render geometry, which the visibility
labels are tested against. The file is checked as its reader loads it and written to
``<out-dir>/<map>_world_collision_complete.glb`` or ``<out-dir>/<map>_visual_complete.glb``; its
sha256, its geometry's and the package's are printed beside the paper's. The game's files are
Valve's and are not part of this repository (docs/data.md).

Examples::

    python tools/prepare_data.py meshes --vpk-dir <maps folder> --cli <tool> --map de_dust2 \\
        --out-dir data/meshes
    python tools/prepare_data.py meshes --vpk-dir <maps folder> --cli <tool> --map de_dust2 \\
        --kind visual --out-dir data/meshes
    python tools/prepare_data.py meshes --glb <exported.glb> --map de_dust2 --out-dir data/meshes
"""

import argparse
import hashlib
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from worldcast.utils.files import sha256_file

from ..map_mesh import load_collision_mesh
from ._visibility_geometry import Occluders, load_visual_occluders

__all__ = [
    "KINDS",
    "PAPER_MESH_FILE_SHA256",
    "PAPER_MESH_GEOMETRY_SHA256",
    "PAPER_VISUAL_FILE_SHA256",
    "PAPER_VISUAL_GEOMETRY_SHA256",
    "PAPER_VPK_SHA256",
    "U_TO_GLTF",
    "add_arguments",
    "check_export",
    "check_visual",
    "comparison",
    "export_command",
    "geometry_sha256",
    "run",
]

#: The kinds of mesh: the world physics (``data.collision_meshes``) and the render geometry (the
#: visibility labels' occluders).
KINDS = ("collision", "visual")
#: What the tool exports of ``<map>.vpk`` for each kind.
RESOURCES = {"collision": "maps/{map}/world_physics.vmdl_c", "visual": "maps/{map}.vmap_c"}
#: The name each kind is written under.
RESULTS = {
    "collision": "{map}_world_collision_complete.glb",
    "visual": "{map}_visual_complete.glb",
}

#: The paper's collision mesh of each map: the sha256 of its file (a glTF binary).
PAPER_MESH_FILE_SHA256 = {
    "de_dust2": "88fd9726db274f03079ee715e489b562ebce6e3ccd87c2ca6aa438c875a77509",
    "de_mirage": "111059fa0f0cabfe99ee1a7f381704cd80d8ee61de9979a4fdcf433201d495cf",
    "de_nuke": "e1663e1cfcd6d5224ebb5707831941be1a8a4395a50a94f5202d02b8938cc9c6",
    "de_ancient": "94c5d88b68a6455b75c81744989e7753e8b4bd278cf3396bb0880ea5d46c085e",
}
#: The sha256 of the geometry of the same meshes (:func:`geometry_sha256`), which the file's packing
#: (its textures, its metadata) does not change.
PAPER_MESH_GEOMETRY_SHA256 = {
    "de_dust2": "311c359c39a273ecc7999f57b891cc5b6cf9cd3c14934dfee2e4937ef6edf390",
    "de_mirage": "2b0dd691deebb85704f9b8841e482a38b39432b1f7f9b6787615a9af8edfded5",
    "de_nuke": "014bd86e71a1684deb1b8787e6e9cbfe5e1c7ed7916604bc65dee6b71d563fe3",
    "de_ancient": "d56f8d86b46021f4399c67bb0bab2502f41eb0b92f107b1017c3ed4f477647d7",
}
#: The sha256 of the map package (``<map>.vpk``) each of them was exported from.
PAPER_VPK_SHA256 = {
    "de_dust2": "a3a424bb8156539845fd453348e6b01b5b1114470d00677ff6250aff0a550467",
    "de_mirage": "ec578cb2d63c235dc720ed4d56353917de6895a147180952cf1f566be053df41",
    "de_nuke": "f0f970155c00db9c16d888576e063fb0f845d7f9a0b5eebb1e78cab0c4c0aa92",
    "de_ancient": "5dffb06306d5351985fc2b2c06849753c3f528474e9e93dba670b9c290463a18",
}
#: The sha256 of the file of each map's visual mesh, as the paper's label files record it.
PAPER_VISUAL_FILE_SHA256 = {
    "de_dust2": "7bf712db0feeac3952225702af919d85e9d982010495021ddfc8cfdbee05f9a7",
    "de_mirage": "5093262ba6a5080f3902c4e346fbb1b6bc8bcb9a0544b7960764c727063a72e1",
    "de_nuke": "0916c1e9d620827c1cd855d51227dbc0604d7410af4aad9998ce2c3f8e1a8111",
    "de_ancient": "4ad522fc22ce2545ed8df9b39cb9be0cebcf20f9c40acda1384761ee7aba7cb0",
}
#: The sha256 of the triangles of the same meshes (``Occluders.summary["sha256"]``), which the
#: file's packing does not change: the digest the label files record as ``_meta.mesh.sha256``.
PAPER_VISUAL_GEOMETRY_SHA256 = {
    "de_dust2": "0e67055cb233452fd6b4ab191d583ca929d35b57998430b00ab055914aadee7d",
    "de_mirage": "e03efaf149ef174bd76a184ec0457033ad6c22d30a0c3b0bdee7418c487dac4c",
    "de_nuke": "1a3e02b477dedc02731a49d66c08df9a4e3d1bf521378e256625f3404629c5dc",
    "de_ancient": "5432736ca4a92c499bacd458be9d0b18c88ac6c01309843301cbe706d4f34b34",
}

#: The one transform every node of a collision export shares: engine units (u) to glTF metres, an
#: axis permutation times 0.0254, ``(x, y, z) -> (y, z, x) * 0.0254``. The vertices stay in u.
U_TO_GLTF = 0.0254 * np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add this step's options to its subcommand's parser."""
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--vpk-dir",
        help="the game's maps folder (game/csgo/maps), which holds <map>.vpk; exports with --cli",
    )
    source.add_argument("--glb", help="a glTF binary of the map, exported already")
    parser.add_argument("--cli", help="ValveResourceFormat's command-line tool: a path or a name")
    parser.add_argument("--map", required=True, help="the map's name, e.g. de_dust2")
    parser.add_argument(
        "--kind",
        choices=KINDS,
        default="collision",
        help="collision: the world physics (data.collision_meshes); visual: the render geometry "
        "(visibility --visual-meshes)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="receives <map>_world_collision_complete.glb or <map>_visual_complete.glb",
    )


def export_command(
    cli: str, vpk: str | Path, map_name: str, out_dir: str | Path, kind: str = "collision"
) -> list[str]:
    """The command-line tool's call that exports a mesh of ``map_name``.

    Args:
        cli (str): the tool.
        vpk (str | Path): the map's package, ``<map>.vpk``.
        map_name (str): the map's name, e.g. ``de_dust2``.
        out_dir (str | Path): where the tool writes.
        kind (str): ``collision`` (the world physics) or ``visual`` (the render geometry).

    Returns:
        list[str]: the program and its arguments.
    """
    return [
        str(cli),
        *("-i", str(vpk)),
        *("-o", str(out_dir)),
        "-d",
        *("-f", RESOURCES[kind].format(map=map_name)),
        *("--gltf_export_format", "glb"),
        "--gltf_export_materials",
        "--gltf_textures_adapt",
        "--gltf_export_extras",
    ]


def geometry_sha256(mesh: Any) -> str:
    """The sha256 of a mesh's vertices (float64 ``[V, 3]``) and faces (int64 ``[F, 3]``), row-major
    and little-endian, in this order."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype="<i8").tobytes())
    return digest.hexdigest()


def check_export(path: str | Path) -> Any:
    """The collision mesh in a glTF binary, as ``data.collision_meshes`` reads it.

    The loader takes each geometry's vertices as they are (in u, +z up) and ignores the node
    transform, which an export shares between its nodes as :data:`U_TO_GLTF`; a file with another
    transform is refused, as its vertices would not be in u.

    Args:
        path (str | Path): the glTF binary.

    Returns:
        trimesh.Trimesh: the mesh, in u.
    """
    import trimesh

    scene = trimesh.load(str(path), process=False)
    nodes = list(scene.graph.nodes_geometry) if isinstance(scene, trimesh.Scene) else []
    if not nodes:
        raise ValueError(f"{path}: not a glTF scene with geometry")
    transform = scene.graph[nodes[0]][0]
    # the export writes the matrix in float32: 0.0254 reads 0.025399996, its zeros up to 3e-9
    if not np.allclose(transform[:3, :3], U_TO_GLTF, rtol=1e-6, atol=1e-8):
        raise ValueError(
            f"{path}: the nodes' transform is not the export's (u to metres, (x, y, z) -> (y, z, x)"
            " * 0.0254), so its vertices are not known to be in u"
        )
    return load_collision_mesh(path)


def check_visual(path: str | Path) -> Occluders:
    """The visual mesh in a glTF binary as the visibility labels read it: the occluders, whose
    ``summary`` holds the triangle counts and the sha256 of the triangles.

    Raises:
        ValueError: not a self-contained glTF binary, or without an opaque triangle.
    """
    return load_visual_occluders(path)


def comparison(map_name: str, digests: Mapping[str, str], kind: str = "collision") -> list[str]:
    """One line per digest: its value, and whether it matches the paper's for ``map_name``.

    Args:
        map_name (str): the map's name.
        digests (Mapping[str, str]): sha256 by kind of digest: ``file``, ``geometry`` and ``vpk``.
        kind (str): ``collision`` or ``visual``, the mesh the digests are of.

    Returns:
        list[str]: the lines, in that order, of the digests ``digests`` has.
    """
    tables = {
        "collision": {"file": PAPER_MESH_FILE_SHA256, "geometry": PAPER_MESH_GEOMETRY_SHA256},
        "visual": {"file": PAPER_VISUAL_FILE_SHA256, "geometry": PAPER_VISUAL_GEOMETRY_SHA256},
    }[kind]
    tables = {**tables, "vpk": PAPER_VPK_SHA256}
    lines = []
    for name, table in tables.items():
        if name not in digests:
            continue
        paper = table.get(map_name)
        if paper is None:
            verdict = f"the paper has no {name} for {map_name}"
        elif digests[name] == paper:
            verdict = "matches the paper's"
        else:
            verdict = f"differs from the paper's ({paper})"
        lines.append(f"{name} sha256 {digests[name]}: {verdict}")
    return lines


def _resolve_cli(cli: str | None) -> str:
    if cli is None:
        raise ValueError(
            "--vpk-dir needs --cli, the command-line tool of ValveResourceFormat "
            "(https://github.com/ValveResourceFormat/ValveResourceFormat)"
        )
    found = shutil.which(cli)
    if found is None:
        raise ValueError(f"--cli {cli}: no such executable")
    return found


def _find_vpk(vpk_dir: str, map_name: str) -> Path:
    vpk = Path(vpk_dir) / f"{map_name}.vpk"
    if not vpk.is_file():
        raise ValueError(
            f"{vpk}: no such file; --vpk-dir is the game's maps folder (game/csgo/maps)"
        )
    return vpk


def _export(cli: str, vpk: Path, map_name: str, kind: str, work: Path) -> Path:
    command = export_command(cli, vpk, map_name, work, kind)
    print(shlex.join(command), flush=True)
    status = subprocess.run(command, check=False).returncode
    if status != 0:
        raise ValueError(f"{cli} exited with status {status}")
    written = sorted(work.rglob("*.glb"))
    if len(written) != 1:
        raise ValueError(
            f"{cli} wrote {len(written)} .glb files, expected one for "
            f"{RESOURCES[kind].format(map=map_name)} in {vpk}"
        )
    return written[0]


def _check(kind: str, path: Path) -> tuple[str, str]:
    """The sha256 of the geometry of the mesh in ``path``, and a line that describes it."""
    if kind == "visual":
        summary = check_visual(path).summary
        return summary["sha256"], (
            f"{summary['opaque_triangles']:,} opaque and {summary['possible_triangles']:,} "
            "possible triangles"
        )
    mesh = check_export(path)
    low, high = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    return geometry_sha256(mesh), (
        f"{len(mesh.faces):,} triangles, {len(mesh.vertices):,} vertices, "
        f"bounds (u) {low.round(1).tolist()} to {high.round(1).tolist()}"
    )


def run(args: argparse.Namespace) -> int:
    """Export the map's mesh (or take ``--glb``), check it, write it and compare it with the
    paper's; returns the exit status (a mesh that differs from the paper's is not an error)."""
    vpk = None
    if args.glb is None:
        cli = _resolve_cli(args.cli)
        vpk = _find_vpk(args.vpk_dir, args.map)
    elif not Path(args.glb).is_file():
        raise ValueError(f"--glb {args.glb}: no such file")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = out / RESULTS[args.kind].format(map=args.map)
    with tempfile.TemporaryDirectory(dir=out) as work:
        if vpk is None:
            exported = Path(args.glb)
        else:
            exported = _export(cli, vpk, args.map, args.kind, Path(work))
        geometry, description = _check(args.kind, exported)
        if not (result.exists() and result.samefile(exported)):
            shutil.copyfile(exported, result)
    print(f"{result}: {description}")
    digests = {"file": sha256_file(result), "geometry": geometry}
    if vpk is not None:
        digests["vpk"] = sha256_file(vpk)
    print("\n".join(comparison(args.map, digests, args.kind)))
    return 0
