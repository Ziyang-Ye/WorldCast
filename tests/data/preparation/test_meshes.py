"""The collision meshes step: the export command, the check of an export, the comparison with the
paper's meshes. The command-line tool is a stand-in script."""

import argparse
import hashlib
import json
import os
import shlex
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import yaml

from worldcast.config.loader import usage_errors
from worldcast.data.map_mesh import load_collision_mesh
from worldcast.data.preparation import meshes
from worldcast.utils.files import sha256_file

REPO = Path(__file__).resolve().parents[3]
MAP = "de_test"
#: The two boxes of the synthetic export: extents, and where the second one stands, u.
BOXES = ((400.0, 400.0, 400.0), (100.0, 100.0, 100.0))
SECOND_BOX_AT = 1000.0
needs_posix = pytest.mark.skipif(os.name == "nt", reason="the stand-in tool is a script")


def write_export(
    path: Path,
    *,
    matrix: Callable[[int, np.ndarray], np.ndarray] | None = None,
    names: str = "group",
) -> Path:
    """A glTF binary laid out as the tool writes it: the vertices in u, one transform of the nodes
    (``matrix``: the transform of box ``index`` given the shared one)."""
    trimesh = pytest.importorskip("trimesh")
    shared = np.eye(4)
    shared[:3, :3] = meshes.U_TO_GLTF
    scene = trimesh.Scene()
    for index, extents in enumerate(BOXES):
        box = trimesh.creation.box(extents=extents)
        box.apply_translation([index * SECOND_BOX_AT, 0.0, 0.0])
        node = f"{names}_{index}"
        transform = shared if matrix is None else matrix(index, shared)
        scene.add_geometry(box, node_name=node, geom_name=node, transform=transform)
    path.write_bytes(scene.export(file_type="glb"))
    return path


def stand_in_tool(tmp_path: Path, export: Path, *, files: int = 1, status: int = 0) -> Path:
    """A script that records its arguments in ``tool.args`` and writes ``files`` copies of
    ``export`` under the output folder it is given."""
    log = tmp_path / "tool.args"
    script = tmp_path / "tool"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, shutil, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        f"Path({str(log)!r}).write_text(json.dumps(args))\n"
        f"folder = Path(args[args.index('-o') + 1]) / 'maps' / {MAP!r}\n"
        "folder.mkdir(parents=True)\n"
        f"for index in range({files}):\n"
        "    name = 'world_physics.glb' if index == 0 else f'other_{index}.glb'\n"
        f"    shutil.copyfile({str(export)!r}, folder / name)\n"
        f"sys.exit({status})\n"
    )
    script.chmod(0o755)
    return script


def arguments(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    meshes.add_arguments(parser)
    return parser.parse_args(list(argv))


@pytest.fixture
def export(tmp_path) -> Path:
    return write_export(tmp_path / "exported.glb")


@pytest.fixture
def maps_folder(tmp_path) -> Path:
    folder = tmp_path / "maps"
    folder.mkdir()
    (folder / f"{MAP}.vpk").write_bytes(b"a map package")
    return folder


def test_the_export_command_is_the_recorded_one():
    command = meshes.export_command("tool", "/game/maps/de_dust2.vpk", "de_dust2", "/work")
    assert command == [
        *("tool", "-i", "/game/maps/de_dust2.vpk", "-o", "/work", "-d"),
        *("-f", "maps/de_dust2/world_physics.vmdl_c"),
        *("--gltf_export_format", "glb", "--gltf_export_materials"),
        *("--gltf_textures_adapt", "--gltf_export_extras"),
    ]


def test_the_visual_export_takes_the_map_in_place_of_its_world_physics():
    visual = meshes.export_command("tool", "/g/de_dust2.vpk", "de_dust2", "/work", "visual")
    collision = meshes.export_command("tool", "/g/de_dust2.vpk", "de_dust2", "/work")
    assert visual[visual.index("-f") + 1] == "maps/de_dust2.vmap_c"
    visual[visual.index("-f") + 1] = collision[collision.index("-f") + 1]
    assert visual == collision  # the other arguments are the same


def test_a_checked_visual_export_is_its_occluders(export):
    summary = meshes.check_visual(export).summary
    assert (summary["opaque_triangles"], summary["possible_triangles"]) == (24, 0)
    assert len(summary["sha256"]) == 64


def test_a_checked_export_is_the_mesh_in_u(export):
    mesh = meshes.check_export(export)
    assert (len(mesh.vertices), len(mesh.faces)) == (16, 24)
    np.testing.assert_array_equal(mesh.vertices.min(axis=0), [-200.0, -200.0, -200.0])
    np.testing.assert_array_equal(mesh.vertices.max(axis=0), [SECOND_BOX_AT + 50.0, 200.0, 200.0])
    np.testing.assert_array_equal(mesh.vertices, load_collision_mesh(export).vertices)


def test_the_geometry_digest_does_not_depend_on_the_files_packing(tmp_path, export):
    other = write_export(tmp_path / "renamed.glb", names="surface")
    assert sha256_file(export) != sha256_file(other)
    mesh = meshes.check_export(export)
    by_hand = hashlib.sha256(
        np.asarray(mesh.vertices, "<f8").tobytes() + np.asarray(mesh.faces, "<i8").tobytes()
    ).hexdigest()
    assert (
        meshes.geometry_sha256(mesh)
        == by_hand
        == meshes.geometry_sha256(meshes.check_export(other))
    )


def test_an_export_in_another_convention_is_refused(tmp_path):
    metres = write_export(tmp_path / "metres.glb", matrix=lambda index, shared: np.eye(4))
    with pytest.raises(ValueError, match="not the export's"):
        meshes.check_export(metres)

    def shifted(index, shared):
        moved = shared.copy()
        moved[:3, 3] = [0.0, 0.0, 5.0 * index]
        return moved

    with pytest.raises(ValueError, match="one shared, translation-free transform"):
        meshes.check_export(write_export(tmp_path / "shifted.glb", matrix=shifted))


def test_the_comparison_says_which_digests_are_the_papers(monkeypatch):
    monkeypatch.setitem(meshes.PAPER_MESH_FILE_SHA256, MAP, "f" * 64)
    monkeypatch.setitem(meshes.PAPER_MESH_GEOMETRY_SHA256, MAP, "a" * 64)
    digests = {"file": "f" * 64, "geometry": "b" * 64}
    assert meshes.comparison(MAP, digests) == [
        f"file sha256 {'f' * 64}: matches the paper's",
        f"geometry sha256 {'b' * 64}: differs from the paper's ({'a' * 64})",
    ]
    assert meshes.comparison("de_other", digests) == [
        f"file sha256 {'f' * 64}: the paper has no file for de_other",
        f"geometry sha256 {'b' * 64}: the paper has no geometry for de_other",
    ]
    assert meshes.comparison(MAP, {"vpk": "c" * 64}) == [
        f"vpk sha256 {'c' * 64}: the paper has no vpk for {MAP}"
    ]


def test_the_comparison_of_a_visual_mesh_reads_the_visual_tables(monkeypatch):
    monkeypatch.setitem(meshes.PAPER_VISUAL_GEOMETRY_SHA256, MAP, "a" * 64)
    digests = {"file": "f" * 64, "geometry": "a" * 64}
    assert meshes.comparison(MAP, digests, "visual") == [
        f"file sha256 {'f' * 64}: the paper has no file for {MAP}",
        f"geometry sha256 {'a' * 64}: matches the paper's",
    ]


def test_an_exported_file_is_written_checked_and_compared(tmp_path, export, capsys, monkeypatch):
    geometry = meshes.geometry_sha256(meshes.check_export(export))
    monkeypatch.setitem(meshes.PAPER_MESH_GEOMETRY_SHA256, MAP, geometry)
    monkeypatch.setitem(meshes.PAPER_MESH_FILE_SHA256, MAP, "0" * 64)
    out = tmp_path / "meshes"
    assert meshes.run(arguments("--glb", str(export), "--map", MAP, "--out-dir", str(out))) == 0
    result = out / f"{MAP}_world_collision_complete.glb"
    assert result.read_bytes() == export.read_bytes()
    assert list(out.iterdir()) == [result]
    printed = capsys.readouterr().out
    assert "16 vertices" in printed and "24 triangles" in printed
    assert f"geometry sha256 {geometry}: matches the paper's\n" in printed
    assert f"file sha256 {sha256_file(export)}: differs from the paper's ({'0' * 64})\n" in printed
    assert "vpk" not in printed
    # the written file is a valid input in its place
    args = arguments("--glb", str(result), "--map", MAP, "--out-dir", str(out))
    assert meshes.run(args) == 0 and result.read_bytes() == export.read_bytes()


@needs_posix
def test_the_tool_exports_the_maps_world_physics(tmp_path, export, maps_folder, capsys):
    tool = stand_in_tool(tmp_path, export)
    out = tmp_path / "meshes"
    args = arguments(
        *("--vpk-dir", str(maps_folder), "--cli", str(tool), "--map", MAP, "--out-dir", str(out))
    )
    assert meshes.run(args) == 0
    given = json.loads((tmp_path / "tool.args").read_text())
    work = given[given.index("-o") + 1]
    assert Path(work).parent == out
    vpk = maps_folder / f"{MAP}.vpk"
    assert given == meshes.export_command(tool, vpk, MAP, work)[1:]
    result = out / f"{MAP}_world_collision_complete.glb"
    assert result.read_bytes() == export.read_bytes()
    assert list(out.iterdir()) == [result]  # the tool's folder is gone
    printed = capsys.readouterr().out
    assert printed.startswith(shlex.join(meshes.export_command(tool, vpk, MAP, work)))
    assert f"vpk sha256 {sha256_file(vpk)}: the paper has no vpk for {MAP}" in printed


def test_a_visual_file_is_written_checked_and_compared(tmp_path, export, capsys, monkeypatch):
    geometry = meshes.check_visual(export).summary["sha256"]
    monkeypatch.setitem(meshes.PAPER_VISUAL_GEOMETRY_SHA256, MAP, geometry)
    out = tmp_path / "meshes"
    args = arguments(
        *("--glb", str(export), "--map", MAP, "--kind", "visual", "--out-dir", str(out))
    )
    assert meshes.run(args) == 0
    result = out / f"{MAP}_visual_complete.glb"
    assert result.read_bytes() == export.read_bytes() and list(out.iterdir()) == [result]
    printed = capsys.readouterr().out
    assert "24 opaque and 0 possible triangles" in printed
    assert f"geometry sha256 {geometry}: matches the paper's\n" in printed


@needs_posix
def test_the_tool_exports_the_maps_render_geometry(tmp_path, export, maps_folder, capsys):
    tool = stand_in_tool(tmp_path, export)
    out = tmp_path / "meshes"
    args = arguments(
        *("--vpk-dir", str(maps_folder), "--cli", str(tool), "--map", MAP, "--out-dir", str(out)),
        *("--kind", "visual"),
    )
    assert meshes.run(args) == 0
    given = json.loads((tmp_path / "tool.args").read_text())
    work = given[given.index("-o") + 1]
    vpk = maps_folder / f"{MAP}.vpk"
    assert given == meshes.export_command(tool, vpk, MAP, work, "visual")[1:]
    assert given[given.index("-f") + 1] == f"maps/{MAP}.vmap_c"
    assert list(out.iterdir()) == [out / f"{MAP}_visual_complete.glb"]


@needs_posix
def test_the_tool_is_found_by_name(tmp_path, export, maps_folder, monkeypatch):
    stand_in_tool(tmp_path, export)
    monkeypatch.setenv("PATH", str(tmp_path))
    out = tmp_path / "meshes"
    args = arguments(
        *("--vpk-dir", str(maps_folder), "--cli", "tool", "--map", MAP, "--out-dir", str(out))
    )
    assert meshes.run(args) == 0
    assert json.loads((tmp_path / "tool.args").read_text())[0] == "-i"


def test_a_missing_tool_or_file_is_a_usage_error(tmp_path, export, maps_folder):
    out = str(tmp_path / "meshes")
    folder = str(maps_folder)
    with pytest.raises(ValueError, match="--vpk-dir needs --cli"):
        meshes.run(arguments("--vpk-dir", folder, "--map", MAP, "--out-dir", out))
    with pytest.raises(ValueError, match="--cli no-such-tool: no such executable"):
        meshes.run(
            arguments(
                *("--vpk-dir", folder, "--cli", "no-such-tool", "--map", MAP, "--out-dir", out)
            )
        )
    with pytest.raises(ValueError, match="no such file; --vpk-dir is the game's maps folder"):
        meshes.run(
            arguments(
                *(
                    "--vpk-dir",
                    str(tmp_path),
                    "--cli",
                    sys.executable,
                    "--map",
                    MAP,
                    "--out-dir",
                    out,
                )
            )
        )
    with pytest.raises(ValueError, match="--glb .*nothing.glb: no such file"):
        meshes.run(
            arguments("--glb", str(tmp_path / "nothing.glb"), "--map", MAP, "--out-dir", out)
        )
    assert not Path(out).exists()


@needs_posix
@pytest.mark.parametrize(
    "files, status, message",
    [
        (1, 3, "exited with status 3"),
        (0, 0, "wrote 0 .glb files, expected one for maps/de_test/world_physics.vmdl_c"),
        (2, 0, "wrote 2 .glb files"),
    ],
)
def test_a_failed_export_is_a_usage_error(tmp_path, export, maps_folder, files, status, message):
    tool = stand_in_tool(tmp_path, export, files=files, status=status)
    out = tmp_path / "meshes"
    args = arguments(
        *("--vpk-dir", str(maps_folder), "--cli", str(tool), "--map", MAP, "--out-dir", str(out))
    )
    with pytest.raises(ValueError, match=message):
        meshes.run(args)
    assert list(out.iterdir()) == []  # no result, and the tool's folder is gone


def test_a_usage_error_ends_the_tool_with_its_message(tmp_path, capsys):
    parser = argparse.ArgumentParser(prog="prepare_data.py meshes")
    meshes.add_arguments(parser)
    args = parser.parse_args(
        ["--vpk-dir", str(tmp_path), "--map", MAP, "--out-dir", str(tmp_path / "meshes")]
    )
    with pytest.raises(SystemExit) as stop, usage_errors(parser):
        meshes.run(args)
    assert stop.value.code == 2
    assert "--vpk-dir needs --cli" in capsys.readouterr().err


def test_one_source_is_required(tmp_path, export):
    with pytest.raises(SystemExit):
        arguments("--map", MAP, "--out-dir", str(tmp_path))
    with pytest.raises(SystemExit):
        arguments(
            *("--glb", str(export), "--vpk-dir", str(tmp_path), "--map", MAP, "--out-dir", "o")
        )


def test_the_step_writes_the_names_the_training_paths_give():
    paths = yaml.safe_load((REPO / "examples" / "train_paths.yaml").read_text())
    named = paths["data"]["collision_meshes"]
    assert (
        set(named) == set(meshes.PAPER_MESH_FILE_SHA256) == set(meshes.PAPER_MESH_GEOMETRY_SHA256)
    )
    assert set(named) == set(meshes.PAPER_VPK_SHA256)
    assert set(named) == set(meshes.PAPER_VISUAL_FILE_SHA256)
    assert set(named) == set(meshes.PAPER_VISUAL_GEOMETRY_SHA256)
    for map_name, path in named.items():
        assert Path(path).name == meshes.RESULTS["collision"].format(map=map_name)
        assert meshes.RESULTS["visual"].format(map=map_name) == f"{map_name}_visual_complete.glb"


def test_docs_data_lists_the_papers_digests():
    text = (REPO / "docs" / "data.md").read_text()
    for table in (
        meshes.PAPER_MESH_FILE_SHA256,
        meshes.PAPER_MESH_GEOMETRY_SHA256,
        meshes.PAPER_VPK_SHA256,
        meshes.PAPER_VISUAL_FILE_SHA256,
        meshes.PAPER_VISUAL_GEOMETRY_SHA256,
    ):
        for map_name, digest in table.items():
            row = next(
                line for line in text.splitlines() if f"`{map_name}`" in line and digest in line
            )
            assert row.startswith("|")
