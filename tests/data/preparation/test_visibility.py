"""The GT visibility labels: a pair's line of sight, a recording's labels and the tool's files.

The scenes are a few boxes on a flat map and players standing still, in Source units: the observer
at the origin looking along +x, its target 500 u ahead.
"""

import argparse
import json
import struct
from dataclasses import replace

import numpy as np
import pytest

from worldcast.data.labels import frustum_rows, visibility_rows
from worldcast.data.preparation import _visibility_labels as labels
from worldcast.data.preparation import visibility
from worldcast.data.preparation._visibility_geometry import (
    BODY_PROBES,
    load_collision_occluders,
    load_visual_occluders,
)
from worldcast.data.preparation._visibility_scope import SCOPE_CORNER_RATIO, scoped_frames

START = 1000  #: the first tick of every table
FRAMES = 24  #: video frames of a recording (32 fps, two ticks each)
TICKS = 2 * FRAMES + 16  #: rows of a table
OBSERVER, TARGET = 0, 1
FAR_AWAY = (-4000.0, 300.0)  #: the other players stand behind the observer, far away
BEYOND = START + 10_000  #: a round end no frame reaches


# ------------------------------------------------------------------------------------ the maps
def _box(center, half):
    """The 8 corners and 12 triangles of an axis-aligned box (Source units)."""
    corners = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64
    )
    vertices = np.asarray(center) + corners * np.asarray(half)
    faces = []
    for a, b, c, d in (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ):
        faces += [(a, b, c), (a, c, d)]
    return vertices, np.array(faces, dtype=np.uint32)


def write_glb(path, parts):
    """A GLB of boxes. Each part: ``boxes`` ``[(center, half)]`` in Source units, an optional
    ``material`` (a glTF material dict), ``name``, ``translation`` of its node and
    ``parent_translation`` of a node above it (glTF metres)."""
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "meshes": [],
        "materials": [],
        "accessors": [],
        "bufferViews": [],
        "buffers": [],
    }
    blob = bytearray()

    def add(array, target, kind):
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": array.nbytes}
        blob.extend(array.tobytes())
        document["bufferViews"].append(view)
        count = len(array) if kind != "SCALAR" else array.size
        code = 5126 if array.dtype == np.float32 else 5125
        document["accessors"].append(
            {
                "bufferView": len(document["bufferViews"]) - 1,
                "componentType": code,
                "count": count,
                "type": kind,
            }
        )
        return len(document["accessors"]) - 1

    for part in parts:
        boxes = [_box(c, h) for c, h in part["boxes"]]
        offsets = np.cumsum([0] + [len(v) for v, _ in boxes[:-1]])
        source = np.concatenate([v for v, _ in boxes])
        faces = np.concatenate([f + o for (_, f), o in zip(boxes, offsets)]).astype(np.uint32)
        gltf = (source[:, [1, 2, 0]] * 0.0254).astype(np.float32)
        primitive = {
            "attributes": {"POSITION": add(gltf, 34962, "VEC3")},
            "indices": add(faces.reshape(-1), 34963, "SCALAR"),
        }
        if part.get("material") is not None:
            document["materials"].append(part["material"])
            primitive["material"] = len(document["materials"]) - 1
        document["meshes"].append({"primitives": [primitive]})
        node = {"mesh": len(document["meshes"]) - 1, "name": part.get("name", "part")}
        if "translation" in part:
            node["translation"] = list(part["translation"])
        document["nodes"].append(node)
        index = len(document["nodes"]) - 1
        if "parent_translation" in part:
            document["nodes"].append(
                {"translation": list(part["parent_translation"]), "children": [index]}
            )
            index += 1
        document["scenes"][0]["nodes"].append(index)
    document["buffers"].append({"byteLength": len(blob)})
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    blob.extend(b"\0" * (-len(blob) % 4))
    body = struct.pack("<II", len(text), 0x4E4F534A) + text
    body += struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)


#: A wall 20 u thick, 400 u wide and 300 u high at x = 250: between the observer and its target.
WALL = {"boxes": [((250.0, 0.0, 150.0), (10.0, 200.0, 150.0))], "name": "wall"}
#: The same, 60 u high: over it a standing player sees, a crouched one does not.
LOW_WALL = {"boxes": [((250.0, 0.0, 30.0), (10.0, 200.0, 30.0))], "name": "low_wall"}


@pytest.fixture(scope="module")
def embree():
    pytest.importorskip("trimesh")
    pytest.importorskip("trimesh.ray.ray_pyembree")


@pytest.fixture(scope="module")
def walled(tmp_path_factory, embree):
    path = tmp_path_factory.mktemp("maps") / "walled.glb"
    write_glb(path, [WALL])
    return load_visual_occluders(path)


@pytest.fixture(scope="module")
def low_walled(tmp_path_factory, embree):
    path = tmp_path_factory.mktemp("maps") / "low_walled.glb"
    write_glb(path, [LOW_WALL])
    return load_visual_occluders(path)


# ----------------------------------------------------------------------------------- the players
def player(x, y, *, z=0.0, yaw=0.0, team=2, duck=False, dies=None, rows=TICKS):
    """A player standing still; dead from row ``dies`` on (its table then runs 8 more rows)."""
    states = np.tile([x, y, z, yaw, 0.0], (rows, 1))
    alive = np.ones(rows, dtype=bool)
    death = None
    if dies is not None:
        rows = dies + 8
        states, alive = states[:rows], alive[:rows]
        alive[dies:] = False
        death = START + dies
    return labels.PlayerTicks(
        tick=START + np.arange(rows, dtype=np.int64),
        states=states.astype(np.float64),
        alive=alive,
        crouched=np.full(rows, duck),
        team=team,
        death_tick=death,
    )


def round_of(target=(500.0, 0.0), *, observer=None, rows=TICKS, **players):
    """Ten players: the observer (slot 0), the target (slot 1) and eight far behind the observer;
    ``players`` replace or remove (None) a slot by name ``p<slot>``."""
    roster = {
        OBSERVER: observer or player(0.0, 0.0, rows=rows),
        TARGET: player(*target, team=3, rows=rows),
    }
    for slot in range(2, 10):
        roster[slot] = player(FAR_AWAY[0], FAR_AWAY[1] * slot, team=3, rows=rows)
    for name, value in players.items():
        if value is None:
            del roster[int(name[1:])]
        else:
            roster[int(name[1:])] = value
    return roster


def label(roster, occluders, *, scoped=None, capsule_unknown=False, round_end=BEYOND):
    arrays, moved = labels.observer_labels(
        OBSERVER,
        roster,
        fps=32.0,
        frames=FRAMES,
        round_end_tick=round_end,
        scoped=np.zeros(FRAMES, bool) if scoped is None else scoped,
        occluders=occluders,
        capsule_unknown=capsule_unknown,
    )
    return arrays, moved


def row(arrays, name, slot=TARGET):
    return arrays[name][slot]


# --------------------------------------------------------------------------- the line of sight
def test_a_target_in_the_open_is_visible(walled):
    arrays, _ = label(round_of(target=(500.0, 450.0)), walled)
    assert row(arrays, "_in_frustum").all() and row(arrays, "_binary_eval_valid").all()
    assert row(arrays, "_binary_visible").all() and not row(arrays, "_binary_invisible").any()
    assert int(row(arrays, "_tested_count")[0]) == len(BODY_PROBES)
    assert row(arrays, "_clear_count_upper")[0] == len(BODY_PROBES)
    # the observer's own row, and a slot without anyone, say nothing
    assert not arrays["_binary_eval_valid"][OBSERVER].any()
    assert arrays["_binary_eval_valid"][2:].sum() == 0  # the others are behind the player


def test_a_target_behind_a_wall_is_occluded_not_unknown(walled):
    arrays, _ = label(round_of(target=(500.0, 0.0)), walled)
    assert row(arrays, "_in_frustum").all()
    assert row(arrays, "_binary_eval_valid").all()  # known ...
    assert not row(arrays, "_binary_visible").any()  # ... and not visible
    assert row(arrays, "_binary_invisible").all()
    assert row(arrays, "_clear_mask_upper").sum() == 0 and row(arrays, "_projected_mask")[0] == 2047


def test_a_target_outside_the_frustum_is_offscreen(walled):
    arrays, _ = label(round_of(target=(-500.0, 0.0)), walled)  # behind the observer
    assert not row(arrays, "_in_frustum").any() and row(arrays, "_binary_offscreen").all()
    assert not row(arrays, "_binary_eval_valid").any() and not row(arrays, "_binary_visible").any()
    side, _ = label(round_of(target=(100.0, 900.0)), walled)  # 83 degrees to the side
    assert not row(side, "_in_frustum").any()


def test_crouching_changes_the_eye_and_the_body(low_walled):
    standing, _ = label(round_of(), low_walled)
    assert row(standing, "_binary_visible").all()  # the head clears the 60 u wall
    crouched_eye, _ = label(round_of(observer=player(0.0, 0.0, duck=True)), low_walled)
    assert (
        row(crouched_eye, "_binary_invisible").all() and row(crouched_eye, "_observer_duck").all()
    )
    crouched_target, _ = label(round_of(p1=player(500.0, 0.0, team=3, duck=True)), low_walled)
    assert (
        row(crouched_target, "_binary_invisible").all()
        and row(crouched_target, "_target_duck").all()
    )


def test_a_scoped_view_is_not_evaluated(walled):
    roster = round_of(target=(500.0, 450.0))
    plain, _ = label(roster, walled)
    scoped = np.zeros(FRAMES, bool)
    scoped[5:9] = True
    arrays, _ = label(roster, walled, scoped=scoped)
    assert arrays["_scoped"].tolist() == scoped.tolist()
    assert not arrays["_view_geometry_valid"][scoped].any()
    for name in ("_in_frustum", "_binary_visible", "_binary_eval_valid", "_tested_count"):
        assert not row(arrays, name)[scoped].any()
        np.testing.assert_array_equal(row(arrays, name)[~scoped], row(plain, name)[~scoped])
    assert row(arrays, "_target_alive").all()  # what is not a view stays as it was
    assert not row(arrays, "_binary_offscreen")[scoped].any()


# --------------------------------------------------------------------------------- the players
def test_dead_players_and_a_ended_round_are_not_evaluated(walled):
    roster = round_of(target=(500.0, 450.0), p1=player(500.0, 450.0, team=3, dies=20))
    arrays, _ = label(roster, walled)
    assert row(arrays, "_binary_visible")[:10].all()
    # the target dies at tick START + 20, frame 10; its table ends at row 28, frame 14
    assert not row(arrays, "_target_alive")[10:].any()
    assert row(arrays, "_target_known_dead")[10:].all()  # known dead past the end of its table
    assert not row(arrays, "_target_state_valid")[15:].any()
    assert not row(arrays, "_binary_eval_valid")[10:].any()
    assert row(arrays, "_label_valid")[10:].all()  # a known death is a known label

    observer = round_of(target=(500.0, 450.0), observer=player(0.0, 0.0, dies=16))
    arrays, _ = label(observer, walled)
    assert arrays["_observer_alive"].tolist() == [True] * 8 + [False] * (FRAMES - 8)
    assert not arrays["_binary_eval_valid"][:, 8:].any()

    arrays, _ = label(round_of(target=(500.0, 450.0)), walled, round_end=START + 20)
    assert arrays["_round_active"].tolist() == [True] * 11 + [False] * (FRAMES - 11)
    assert not row(arrays, "_binary_eval_valid")[11:].any()


def test_another_players_capsule_hides_a_target(walled):
    roster = round_of(target=(500.0, 450.0), p2=player(250.0, 225.0, team=3))
    arrays, moved = label(roster, walled)
    assert row(arrays, "_binary_player_blocked_mask").all() and moved == 0
    assert row(arrays, "_binary_eval_valid").all() and row(arrays, "_binary_invisible").all()
    assert row(arrays, "_static_los_upper").all()  # the map alone would show the target
    assert row(arrays, "_player_proxy_blocker_slots").tolist() == [1 << 2] * FRAMES
    again, moved = label(roster, walled, capsule_unknown=True)
    assert moved == FRAMES
    assert not row(again, "_binary_eval_valid").any() and not row(again, "_binary_invisible").any()
    assert not row(again, "_binary_train_valid").any()
    assert row(again, "_in_frustum").all() and not row(again, "_binary_visible").any()
    assert not labels.hidden_by_players(again).any()  # nothing is left to move
    dead_blocker = round_of(target=(500.0, 450.0), p2=player(250.0, 225.0, team=3, dies=0))
    free, _ = label(dead_blocker, walled)
    assert row(free, "_binary_visible").all()  # a dead player blocks nothing


def test_a_player_with_no_recording_leaves_the_round_unknown(walled):
    arrays, _ = label(round_of(target=(500.0, 450.0), p9=None), walled)
    assert not arrays["_blocker_state_complete"][TARGET].any()
    assert row(arrays, "_unknown_blocker_slots").tolist() == [1 << 9] * FRAMES
    assert row(arrays, "_in_frustum").all()  # where the target is stays known
    assert not row(arrays, "_binary_eval_valid").any() and not row(arrays, "_binary_visible").any()
    assert not arrays["_slot_present"][9] and arrays["_slot_expected"].all()


def test_a_label_file_has_the_slots_teams_and_frames_of_its_recording(walled):
    arrays, _ = label(round_of(target=(500.0, 450.0), p4=None), walled)
    assert arrays["_frame_tick"].tolist() == list(range(START, START + 2 * FRAMES, 2))
    assert arrays["_slot_present"].tolist() == [i != 4 for i in range(10)]
    assert arrays["_slot_team_num"].tolist() == [2, 3, 3, 3, -1, 3, 3, 3, 3, 3]
    assert arrays["_same_team"].tolist() == [True] + [False] * 9
    assert {k for k in arrays if k.startswith("slot_")} == {
        f"slot_{i}" for i in (1, 2, 3, 5, 6, 7, 8, 9)
    }
    for name in ("_binary_visible", "_binary_eval_valid", "_in_frustum", "_binary_offscreen"):
        assert arrays[name].shape == (10, FRAMES) and arrays[name].dtype == np.bool_


def test_a_frame_takes_the_last_tick_at_or_before_it():
    table = player(1.0, 2.0)
    state = labels.frame_state(
        table, np.array([START - 1, START, START + 1, START + TICKS - 1, START + TICKS])
    )
    assert state.valid.tolist() == [False, True, True, True, True]
    gap = labels.PlayerTicks(
        tick=np.array([0, 1, 2, 4]),  # tick 3 is missing
        states=np.zeros((4, 5)),
        alive=np.ones(4, bool),
        crouched=np.zeros(4, bool),
        team=2,
        death_tick=None,
    )
    assert labels.frame_state(gap, np.array([2, 3, 4, 5, 6])).valid.tolist() == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert labels.frame_ticks(START, 32.0, 3).tolist() == [START, START + 2, START + 4]
    with pytest.raises(ValueError, match="multiple of the frame rate"):
        labels.frame_ticks(START, 30.0, 3)


def test_the_round_events_are_derived_from_the_tables():
    def table(last, dead_from=None):
        tick = np.arange(last - 99, last + 1)
        alive = np.ones(100, bool)
        if dead_from is not None:
            alive[dead_from:] = False
        return tick, alive

    # slots 0 and 1 live to the end (their tables run 32 ticks past it); 2 died on a row of its
    # table; 3 died with its table ending the tick before; 4 lives, its table ending with the round
    tables = {0: table(1032), 1: table(1032), 2: table(700, 93), 3: table(800), 4: table(1000)}
    end, death = visibility.derive_round_events(
        {slot: t for slot, (t, _) in tables.items()}, {slot: a for slot, (_, a) in tables.items()}
    )
    assert end == 1000
    assert death == {0: None, 1: None, 2: 694, 3: 801, 4: None}


# ---------------------------------------------------------------------------------- the meshes
@pytest.fixture(scope="module")
def materials(tmp_path_factory, embree):
    """Five boxes at x = 100 in lanes of 100 u in y, one per kind of material."""
    kinds = [
        ("opaque", {"name": "walls/brick", "alphaMode": "OPAQUE"}),
        ("glass", {"name": "glass/window", "alphaMode": "BLEND"}),
        ("fence", {"name": "props/fence", "alphaMode": "MASK"}),
        ("effect", {"name": "fx/steam", "extras": {"vmat": {"ShaderName": "csgo_effects.vfx"}}}),
        ("tool", {"name": "materials/tools/toolsclip"}),
        ("bare", None),
    ]
    parts = [
        {"boxes": [((100.0, 100.0 * lane, 100.0), (5.0, 40.0, 40.0))], "material": m, "name": n}
        for lane, (n, m) in enumerate(kinds)
    ]
    path = tmp_path_factory.mktemp("maps") / "materials.glb"
    write_glb(path, parts)
    return [name for name, _ in kinds], load_visual_occluders(path)


def test_the_material_of_a_triangle_decides_what_it_blocks(materials):
    names, occluders = materials
    lanes = np.arange(len(names)) * 100.0
    origins = np.stack([np.zeros(len(names)), lanes, np.full(len(names), 100.0)], axis=1)
    endpoints = origins + [200.0, 0.0, 0.0]
    assert occluders.summary["opaque_triangles"] == 24  # a brick box and a box of no material
    assert (
        occluders.summary["possible_triangles"] == 36
    )  # glass, fence, effect; the tool box is out
    opaque = occluders.opaque.blocked(origins, endpoints)
    possible = occluders.possible.blocked(origins, endpoints)
    assert dict(zip(names, opaque)) == {
        "opaque": True,
        "glass": False,
        "fence": False,
        "effect": False,
        "tool": False,
        "bare": True,
    }
    assert dict(zip(names, possible)) == {
        "opaque": False,
        "glass": True,
        "fence": True,
        "effect": True,
        "tool": False,
        "bare": False,
    }


def test_a_possible_blocker_makes_the_lower_bound_unknown(tmp_path, embree):
    write_glb(
        tmp_path / "glass.glb",
        [
            {"boxes": [((900.0, 5000.0, 0.0), (5.0, 5.0, 5.0))], "name": "far"},
            {
                "boxes": [((250.0, 0.0, 150.0), (10.0, 200.0, 150.0))],
                "material": {"name": "glass", "alphaMode": "BLEND"},
                "name": "glass",
            },
        ],
    )
    arrays, _ = label(round_of(), load_visual_occluders(tmp_path / "glass.glb"))
    assert row(arrays, "_static_los_upper").all() and not row(arrays, "_static_los_lower").any()
    assert row(arrays, "_alpha_ambiguous_mask").tolist() == [2047] * FRAMES
    assert not row(arrays, "_map_label_valid").any()  # the interval's ends disagree
    assert row(arrays, "_binary_visible").all()  # the binary label takes the opaque ones only


def test_a_node_transform_places_the_triangles(tmp_path, embree):
    box = {"boxes": [((0.0, 0.0, 100.0), (5.0, 40.0, 40.0))], "name": "box"}
    # 50 u along Source +y (glTF +x) at the node and again at its parent: the box moves to y = 100
    move = 0.0254 * 50
    write_glb(
        tmp_path / "moved.glb",
        [dict(box, translation=(move, 0, 0), parent_translation=(move, 0, 0))],
    )
    occluders = load_visual_occluders(tmp_path / "moved.glb")
    origins = np.array([[-50.0, 100.0, 100.0], [-50.0, 0.0, 100.0]])
    blocked = occluders.opaque.blocked(origins, origins + [100.0, 0.0, 0.0])
    assert blocked.tolist() == [True, False]  # the box now stands at y = 100


def test_a_glb_needs_opaque_triangles_and_a_glb(tmp_path, embree):
    write_glb(tmp_path / "tool.glb", [dict(WALL, material={"name": "materials/tools/clip"})])
    with pytest.raises(ValueError, match="no opaque triangle"):
        load_visual_occluders(tmp_path / "tool.glb")
    (tmp_path / "not.glb").write_bytes(b"x" * 64)
    with pytest.raises(ValueError, match="not a GLB"):
        load_visual_occluders(tmp_path / "not.glb")


def test_a_collision_mesh_is_an_opaque_occluder(tmp_path, embree):
    import trimesh

    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(
            extents=(20.0, 400.0, 300.0),
            transform=trimesh.transformations.translation_matrix((250.0, 0.0, 150.0)),
        )
    )
    scene.export(tmp_path / "collision.glb")
    occluders = load_collision_occluders(tmp_path / "collision.glb")
    assert occluders.possible is None
    arrays, _ = label(round_of(), occluders)
    assert row(arrays, "_binary_invisible").all()
    assert (row(arrays, "_clear_mask_lower") == row(arrays, "_clear_mask_upper")).all()


# -------------------------------------------------------------------------------- the video
@pytest.fixture(scope="module")
def scope_video(tmp_path_factory):
    pytest.importorskip("cv2")
    pytest.importorskip("imageio_ffmpeg")
    import imageio.v2 as imageio

    path = tmp_path_factory.mktemp("video") / "scope.mp4"
    writer = imageio.get_writer(str(path), fps=32, codec="libx264", quality=9, macro_block_size=1)
    for index in range(14):
        frame = np.full((96, 168, 3), 120, np.uint8)
        if index in (5, 6, 11):  # the scope's black surround
            for rows, columns in (
                (slice(0, 16), slice(0, 21)),
                (slice(0, 16), slice(-21, None)),
                (slice(-16, None), slice(0, 21)),
                (slice(-16, None), slice(-21, None)),
            ):
                frame[rows, columns] = 0
        writer.append_data(frame)
    writer.close()
    return path


def test_scoped_frames_are_found_from_the_corners(scope_video):
    mask = scoped_frames(scope_video, 14)
    # frames 5, 6 and 11 and the frame on each side
    expected = np.zeros(14, bool)
    expected[[4, 5, 6, 7, 10, 11, 12]] = True
    assert mask.tolist() == expected.tolist()
    assert SCOPE_CORNER_RATIO == 0.25
    with pytest.raises(RuntimeError, match="ends at frame 14"):
        scoped_frames(scope_video, 15)
    with pytest.raises(RuntimeError, match="more than 13"):
        scoped_frames(scope_video, 13)
    with pytest.raises(RuntimeError, match="cannot open"):
        scoped_frames(scope_video.parent / "missing.mp4", 14)


# ------------------------------------------------------------------------------------ the tool
def columns(table: labels.PlayerTicks) -> dict[str, list]:
    n = len(table.tick)
    return {
        "tick": table.tick.tolist(),
        "t": ((table.tick - table.tick[0]) / 64.0).tolist(),
        "x": table.states[:, 0].tolist(),
        "y": table.states[:, 1].tolist(),
        "z": table.states[:, 2].tolist(),
        "yaw": table.states[:, 3].tolist(),
        "pitch": table.states[:, 4].tolist(),
        "is_alive": table.alive.tolist(),
        "active": [["duck"] if d else [] for d in table.crouched],
        "team_num": [table.team] * n,
    }


def media_rows(roster):
    return [
        {
            "media_id": f"1-de_test-r01-p{slot:02d}",
            "match_id": 1,
            "map_name": "de_test",
            "round": 1,
            "player_slot": slot,
            "fps": 32.0,
            "video_frames": FRAMES,
            "ticks_path": f"p{slot}/ticks.parquet",
            "ticks_rows": len(table.tick),
            "ticks_file_size": 1,
            "capture_start_tick": START,
            "video_path": f"p{slot}/video.mp4",
        }
        for slot, table in roster.items()
    ]


@pytest.fixture()
def world(tmp_path, monkeypatch, walled):
    """A round on ``de_test`` (a wall; target hidden behind it) served to the tool from memory:
    the media index and one placeholder file per tick table."""
    roster = round_of(rows=TICKS + 32)
    rows = media_rows(roster)
    (tmp_path / "recordings").mkdir()
    for slot in roster:
        path = tmp_path / "recordings" / f"p{slot}" / "ticks.parquet"
        path.parent.mkdir()
        path.write_bytes(b"x")
    index = tmp_path / "media_index.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in rows))
    glb = tmp_path / "walled.glb"
    write_glb(glb, [WALL])
    by_path = {r["ticks_path"]: columns(roster[r["player_slot"]]) for r in rows}
    monkeypatch.setattr(
        visibility,
        "read_tick_columns",
        lambda path, names, rows: by_path[str(path)[len(str(tmp_path / "recordings")) + 1 :]],
    )
    return {
        "tmp": tmp_path,
        "roster": roster,
        "index": index,
        "glb": glb,
        "ids": [r["media_id"] for r in rows],
    }


def run_tool(world, *options, out="labels"):
    parser = argparse.ArgumentParser()
    visibility.add_arguments(parser)
    args = parser.parse_args(
        [
            "--media-index", str(world["index"]),
            "--dataset-root", str(world["tmp"] / "recordings"),
            "--out-dir", str(world["tmp"] / out),
            "--visual-meshes", f"de_test={world['glb']}",
            "--scope", "none",
            *options,
        ]
    )  # fmt: skip
    return visibility.run(args)


def test_the_tool_writes_the_files_the_loader_reads(world, capsys):
    assert run_tool(world, "--media-id", world["ids"][0]) == 0
    assert "1 written, 0 kept" in capsys.readouterr().out
    out = world["tmp"] / "labels"
    assert [p.name for p in out.iterdir()] == [f"{world['ids'][0]}.npz"]  # no temporary file left
    frames = np.arange(FRAMES)
    visible, valid = visibility_rows(out, world["ids"][0], frames)
    assert visible.shape == valid.shape == (10, FRAMES)
    assert valid[TARGET].all() and not visible[TARGET].any()  # behind the wall: occluded
    assert frustum_rows(out, world["ids"][0], frames)[TARGET].all()
    with np.load(out / f"{world['ids'][0]}.npz") as archive:
        meta = labels.decode_meta(archive["_meta"])
    assert meta["schema"] == labels.SCHEMA and meta["observer_slot"] == 0
    assert meta["settings"]["mesh"] == "visual" and meta["settings"]["round_events"] == "ticks"
    assert meta["mesh"]["opaque_triangles"] == 12 and meta["mesh"]["possible_triangles"] == 0
    assert len(meta["mesh"]["sha256"]) == 64
    assert meta["capture_start_tick"] == START and meta["round_end_tick"] == START + TICKS - 1
    assert set(meta["inputs"]["ticks_sha256"]) == set(world["ids"])


def test_an_interrupted_run_resumes_and_a_changed_one_asks_to_overwrite(world, capsys):
    run_tool(world, "--media-id", world["ids"][0])
    path = world["tmp"] / "labels" / f"{world['ids'][0]}.npz"
    stamp = path.stat().st_mtime_ns
    run_tool(world, "--media-id", world["ids"][0], world["ids"][2])
    assert "1 written, 1 kept" in capsys.readouterr().out.splitlines()[-1]
    assert path.stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match="--overwrite"):
        run_tool(world, "--media-id", world["ids"][0], "--capsule-unknown")
    run_tool(world, "--media-id", world["ids"][0], "--capsule-unknown", "--overwrite")
    with np.load(path) as archive:
        assert labels.decode_meta(archive["_meta"])["settings"]["capsule_unknown"] is True
    # the same inputs under another directory give the same arrays
    run_tool(world, "--media-id", world["ids"][0], out="again")
    with np.load(world["tmp"] / "again" / path.name) as again, np.load(path) as first:
        assert set(again.files) == set(first.files)
        assert all(np.array_equal(again[k], first[k]) for k in first.files if k != "_meta")


def test_the_tool_labels_every_recording_by_default(world, capsys):
    run_tool(world)
    assert "10 written, 0 kept" in capsys.readouterr().out
    assert len(list((world["tmp"] / "labels").iterdir())) == 10


def test_a_bad_request_is_refused_with_what_to_do(world):
    with pytest.raises(ValueError, match="not in the media index"):
        run_tool(world, "--media-id", "no-such-media")
    with pytest.raises(ValueError, match="at least 1"):
        run_tool(world, "--workers", "0")
    parser = argparse.ArgumentParser()
    visibility.add_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--media-index", "a", "--dataset-root", "b", "--out-dir", "c", "--meshes", "nope"]
        )
    args = parser.parse_args(
        [
            "--media-index",
            str(world["index"]),
            "--dataset-root",
            "b",
            "--out-dir",
            str(world["tmp"] / "x"),
        ]
    )
    with pytest.raises(ValueError, match="--visual-meshes de_test=<GLB>"):
        visibility.run(args)
    args = parser.parse_args(
        [
            "--media-index",
            str(world["index"]),
            "--dataset-root",
            str(world["tmp"] / "recordings"),
            "--out-dir",
            str(world["tmp"] / "x"),
            "--meshes",
            f"de_test={world['glb']}",
            "--scope",
            "video",
        ]
    )
    rows = [json.loads(line) for line in world["index"].read_text().splitlines()]
    for r in rows:
        r.pop("video_path")
    world["index"].write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="--scope none"):
        visibility.run(args)


def test_a_tick_table_that_disagrees_with_its_row_is_refused(world, monkeypatch):
    rows = [json.loads(line) for line in world["index"].read_text().splitlines()]
    for r in rows:
        r["capture_start_tick"] = START + 1
    world["index"].write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(RuntimeError, match="capture_start_tick"):
        run_tool(world, "--media-id", world["ids"][0])


@pytest.fixture()
def parquet_world(tmp_path, embree):
    """The same round as real parquet files."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    roster = round_of(rows=TICKS + 32)
    rows = media_rows(roster)
    for r in rows:
        path = tmp_path / "recordings" / r["ticks_path"]
        path.parent.mkdir(parents=True)
        pq.write_table(pa.table(columns(roster[r["player_slot"]])), path)
        r["ticks_file_size"] = path.stat().st_size
    index = tmp_path / "media_index.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in rows))
    glb = tmp_path / "walled.glb"
    write_glb(glb, [WALL])
    return {
        "tmp": tmp_path,
        "roster": roster,
        "index": index,
        "glb": glb,
        "ids": [r["media_id"] for r in rows],
        "pa": pa,
        "pq": pq,
    }


def test_workers_do_not_change_the_files(parquet_world):
    run_tool(parquet_world, out="one")
    run_tool(parquet_world, "--workers", "2", out="two")
    one, two = parquet_world["tmp"] / "one", parquet_world["tmp"] / "two"
    assert sorted(p.name for p in one.iterdir()) == sorted(p.name for p in two.iterdir())
    for path in one.iterdir():
        with np.load(path) as a, np.load(two / path.name) as b:
            assert all(np.array_equal(a[k], b[k]) for k in a.files)


def test_the_pov_index_gives_the_deaths_and_the_round_end(parquet_world):
    pa, pq = parquet_world["pa"], parquet_world["pq"]
    ids = parquet_world["ids"]
    path = parquet_world["tmp"] / "pov_rounds.parquet"
    pq.write_table(
        pa.table(
            {
                "media_id": ids + ["1-de_test-r09-p00"],
                "death_tick": pa.array([None, START + 30] + [None] * 8 + [None], pa.int64()),
                "round_end_tick": [START + 40] * 10 + [START + 99],
            }
        ),
        path,
    )
    run_tool(parquet_world, "--pov-index", str(path), "--media-id", ids[0])
    with np.load(parquet_world["tmp"] / "labels" / f"{ids[0]}.npz") as archive:
        arrays = {k: archive[k] for k in archive.files}
    meta = labels.decode_meta(arrays["_meta"])
    assert meta["settings"]["round_events"] == "index" and meta["round_end_tick"] == START + 40
    assert arrays["_round_active"].tolist() == [True] * 21 + [False] * (FRAMES - 21)
    assert arrays["_target_known_dead"][TARGET].tolist() == [False] * 15 + [True] * (FRAMES - 15)
    missing = pa.table(
        {
            "media_id": ids[:3],
            "death_tick": pa.array([None] * 3, pa.int64()),
            "round_end_tick": [1] * 3,
        }
    )
    pq.write_table(missing, parquet_world["tmp"] / "short.parquet")
    with pytest.raises(ValueError, match="no row for"):
        run_tool(
            parquet_world,
            "--pov-index",
            str(parquet_world["tmp"] / "short.parquet"),
            "--overwrite",
            out="other",
        )


def test_settings_are_part_of_a_files_identity(world):
    job = visibility.VisibilityJob("i", "d", "o")
    assert visibility._settings(job, {"mesh": "visual", "mesh_sha256": "x"}) == {
        "mesh": "visual",
        "mesh_sha256": "x",
        "scope": "video",
        "capsule_unknown": False,
        "round_events": "ticks",
    }
    assert replace(job, pov_index="p").pov_index == "p"
