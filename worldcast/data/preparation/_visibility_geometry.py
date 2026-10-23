"""The geometry of the visibility labels: a map's occluders, and the line of sight of a player pair.

A target is tested through eleven probes on its body, each with a segment from the observer's eye
to the probe against the map's visual mesh (its opaque triangles block; mask, blend and effect
triangles only make a probe ambiguous) and against the other players, standing capsules of radius
16 u. The arithmetic is numpy float64, as the labels were made (:mod:`worldcast.data.camera` is
float32 torch).
"""

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from worldcast.data.camera import NEAR_U
from worldcast.data.game import (
    BODY_HEIGHT,
    BODY_HEIGHT_CROUCHED,
    CROUCHED_EYE_HEIGHT,
    EYE_HEIGHT,
    HFOV_DEGREES,
    PLAYER_WIDTH,
    VFOV_DEGREES,
)
from worldcast.data.map_mesh import MapMesh

__all__ = [
    "BODY_PROBES",
    "Occluders",
    "LineOfSight",
    "LineOfSightWithPlayers",
    "PlayerCapsules",
    "apply_player_capsules",
    "load_collision_occluders",
    "load_visual_occluders",
    "player_capsule_blocks",
    "line_of_sight",
]

#: Metres per engine unit: the glTF files of a map are in metres, +y up.
METERS_PER_UNIT = 0.0254
#: Radius of the capsule that stands for another player, u.
CAPSULE_RADIUS = 16.0
#: The capsule's axis starts this far above the player's feet and ends this far below its top, u.
CAPSULE_AXIS_INSET = 16.0
#: A ray against a capsule is shortened by this at both ends, u.
CAPSULE_RAY_TRIM = 0.5
#: A segment ignores a surface this close to its eye, and one this close to its target, u.
SEGMENT_EPS = 0.05
#: ``(height fraction, lateral fraction, head)`` of the eleven probes of a body: three at the head,
#: three at the shoulders, three at the waist, two at the legs.
BODY_PROBES = (
    (0.92, -0.35, True),
    (0.92, 0.00, True),
    (0.92, 0.35, True),
    (0.72, -0.80, False),
    (0.72, 0.00, False),
    (0.72, 0.80, False),
    (0.48, -0.60, False),
    (0.48, 0.00, False),
    (0.48, 0.60, False),
    (0.24, -0.45, False),
    (0.24, 0.45, False),
)
_HEIGHT_FRACTION = np.array([p[0] for p in BODY_PROBES])
_LATERAL_FRACTION = np.array([p[1] for p in BODY_PROBES])
_HEAD = np.array([p[2] for p in BODY_PROBES])
_PROBE_BITS = (np.uint16(1) << np.arange(len(BODY_PROBES), dtype=np.uint16))[None]


# ------------------------------------------------------------------------------------- the map
class _SegmentBlocker:
    """Embree segment test against one triangle soup: is anything between an eye and a point."""

    def __init__(self, mesh: Any) -> None:
        self.intersector = MapMesh(mesh).intersector

    def blocked(self, origins: np.ndarray, endpoints: np.ndarray) -> np.ndarray:
        delta = endpoints - origins
        lengths = np.linalg.norm(delta, axis=1)
        usable = lengths > 2.0 * SEGMENT_EPS
        directions = np.zeros_like(delta)
        directions[usable] = delta[usable] / lengths[usable, None]
        shifted = origins + directions * SEGMENT_EPS
        blocked = np.zeros(len(origins), dtype=bool)
        if not usable.any():
            return blocked
        locations, ray_ids, _ = self.intersector.intersects_location(
            shifted[usable], directions[usable], multiple_hits=False
        )
        if len(ray_ids):
            hit_distance = np.linalg.norm(locations - shifted[usable][ray_ids], axis=1)
            ids = np.flatnonzero(usable)[ray_ids]
            blocked[ids] = hit_distance < lengths[ids] - 2.0 * SEGMENT_EPS
        return blocked


@dataclass
class Occluders:
    """What blocks the line of sight on a map.

    Attributes:
        opaque (_SegmentBlocker): the triangles that block for certain.
        possible (_SegmentBlocker | None): the triangles that may block (mask, blend, effects).
        summary (dict): the triangle counts of the two layers and the sha256 of their vertices and
            faces, which does not depend on the file's container (``_meta.mesh`` of a label file).
    """

    opaque: _SegmentBlocker
    possible: _SegmentBlocker | None
    summary: dict


def load_collision_occluders(path: str | Path) -> Occluders:
    """A map's collision mesh (``data.collision_meshes``) as the opaque occluder."""
    from worldcast.data.map_mesh import load_collision_mesh

    mesh = load_collision_mesh(path)
    digest = hashlib.sha256(mesh.vertices.tobytes() + mesh.faces.tobytes())
    summary = {"opaque_triangles": len(mesh.faces), "possible_triangles": 0}
    return Occluders(_SegmentBlocker(mesh), None, {**summary, "sha256": digest.hexdigest()})


_GLTF_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_GLTF_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_GLB_MAGIC = 0x46546C67
_GLB_JSON, _GLB_BIN = 0x4E4F534A, 0x004E4942


def _read_glb(path: Path) -> tuple[dict, np.ndarray]:
    """A GLB's JSON document and its binary chunk (memory-mapped)."""
    with open(path, "rb") as handle:
        magic, _, total = struct.unpack("<III", handle.read(12))
        if magic != _GLB_MAGIC:
            raise ValueError(f"{path} is not a GLB file")
        document, binary, offset = None, None, 12
        while offset < total:
            length, kind = struct.unpack("<II", handle.read(8))
            if kind == _GLB_JSON:
                document = json.loads(handle.read(length))
            elif kind == _GLB_BIN:
                binary = np.memmap(path, dtype=np.uint8, mode="r", offset=offset + 8, shape=length)
                handle.seek(length, 1)
            else:
                handle.seek(length, 1)
            offset += 8 + length
    if document is None or binary is None:
        raise ValueError(f"{path} is not a self-contained GLB (no JSON or binary chunk)")
    return document, binary


def _accessor(document: dict, binary: np.ndarray, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = _GLTF_DTYPES[accessor["componentType"]]
    components = _GLTF_COMPONENTS[accessor["type"]]
    count = accessor["count"]
    packed = dtype.itemsize * components
    stride = view.get("byteStride") or packed
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    if count == 0:
        return np.empty((0, components), dtype=dtype)
    view_array = np.ndarray(
        shape=(count, components),
        dtype=dtype,
        buffer=binary,
        offset=offset,
        strides=(stride, dtype.itemsize),
    )
    return np.array(view_array, copy=True)


def _node_matrix(node: dict) -> np.ndarray:
    if node.get("matrix"):
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    translation = np.asarray(node.get("translation") or [0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.asarray(node.get("scale") or [1.0, 1.0, 1.0], dtype=np.float64)
    x, y, z, w = np.asarray(node.get("rotation") or [0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def _material_class(document: dict, index: int | None, node_name: str) -> str:
    """``OPAQUE``, ``TOOL`` (not geometry), or what may block: ``MASK``, ``BLEND``, ``EFFECT``,
    ``UNKNOWN``."""
    if index is None:
        return "OPAQUE"
    material = document["materials"][index]
    name = str(material.get("name") or "").lower()
    mode = str(material.get("alphaMode") or "OPAQUE").upper()
    extras = material.get("extras")
    vmat = extras.get("vmat", {}) if isinstance(extras, dict) else {}
    shader = str(vmat.get("ShaderName", "")) if isinstance(vmat, dict) else ""
    if (
        name.startswith("materials/tools/")
        or "toolssolidblocklight" in name
        or "toolssolidblocklight" in node_name.lower()
    ):
        return "TOOL"
    if shader == "csgo_effects.vfx":
        return "EFFECT"
    if mode in ("OPAQUE", "MASK", "BLEND"):
        return mode
    return "UNKNOWN"


def load_visual_occluders(path: str | Path) -> Occluders:
    """A map's visual GLB (the game's render geometry) as occluders.

    Triangles of an opaque material block; those of a mask, blend, effect or unknown material may
    block; tool materials and degenerate triangles are dropped. Each node's transform is applied,
    then the vertices go from glTF metres (+y up) to engine units (+z up).

    Raises:
        ValueError: not a self-contained GLB, or no opaque triangle in it.
    """
    import trimesh

    document, binary = _read_glb(Path(path))
    layers = {name: ([], [], [0]) for name in ("opaque", "possible")}  # vertices, faces, offset

    def add_primitive(primitive: dict, transform: np.ndarray, node_name: str) -> None:
        if primitive.get("mode", 4) != 4:
            return
        positions = _accessor(document, binary, primitive["attributes"]["POSITION"])
        if "indices" in primitive:
            flat = _accessor(document, binary, primitive["indices"]).reshape(-1).astype(np.int64)
        else:
            flat = np.arange(len(positions), dtype=np.int64)
        material = _material_class(document, primitive.get("material"), node_name)
        if material == "TOOL" or not len(flat):
            return
        used, inverse = np.unique(flat, return_inverse=True)
        vertices = trimesh.transform_points(positions[used], transform)
        vertices = vertices[..., [2, 0, 1]] / METERS_PER_UNIT
        vertices = vertices.astype(np.float32, copy=False)
        faces = inverse.reshape(-1, 3).astype(np.int32, copy=False)
        triangles = vertices[faces]
        twice_area = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        faces = faces[np.isfinite(triangles).all(axis=(1, 2)) & (twice_area > 1e-7)]
        if not len(faces):
            return
        layer_vertices, layer_faces, offset = layers[
            "opaque" if material == "OPAQUE" else "possible"
        ]
        layer_vertices.append(vertices)
        layer_faces.append(faces + offset[0])
        offset[0] += len(vertices)

    def visit(index: int, parent: np.ndarray) -> None:
        node = document["nodes"][index]
        transform = parent @ _node_matrix(node)
        if node.get("mesh") is not None:
            for primitive in document["meshes"][node["mesh"]]["primitives"]:
                add_primitive(primitive, transform, str(node.get("name") or f"node_{index}"))
        for child in node.get("children") or []:
            visit(child, transform)

    scenes = document.get("scenes")
    if scenes:
        roots = scenes[document.get("scene", 0)].get("nodes") or []
    else:
        children = {child for node in document["nodes"] for child in node.get("children") or []}
        roots = [i for i in range(len(document["nodes"])) if i not in children]
    for root in roots:
        visit(root, np.eye(4))

    blockers: dict[str, _SegmentBlocker | None] = {}
    summary: dict[str, Any] = {}
    digest = hashlib.sha256()
    for name, (vertices, faces, _) in layers.items():
        summary[f"{name}_triangles"] = sum(len(f) for f in faces)
        if not faces:
            blockers[name] = None
            continue
        for chunk in (*vertices, *faces):
            digest.update(np.ascontiguousarray(chunk).tobytes())
        mesh = trimesh.Trimesh(
            vertices=np.concatenate(vertices), faces=np.concatenate(faces), process=False
        )
        blockers[name] = _SegmentBlocker(mesh)
    if blockers["opaque"] is None:
        raise ValueError(f"{path} has no opaque triangle")
    return Occluders(
        blockers["opaque"], blockers["possible"], {**summary, "sha256": digest.hexdigest()}
    )


# ------------------------------------------------------------------------------------ the pairs
def camera_axes(yaw_degrees: np.ndarray, pitch_degrees: np.ndarray) -> tuple[np.ndarray, ...]:
    """``(forward, right, up)`` of the engine's cameras, each ``[N, 3]`` float64."""
    yaw, pitch = np.radians(yaw_degrees), np.radians(pitch_degrees)
    cosine = np.cos(pitch)
    forward = np.stack([np.cos(yaw) * cosine, np.sin(yaw) * cosine, -np.sin(pitch)], axis=-1)
    right = np.stack([forward[:, 1], -forward[:, 0], np.zeros(len(forward))], axis=-1)
    right /= np.linalg.norm(right, axis=-1, keepdims=True) + 1e-12
    return forward, right, np.cross(right, forward)


def body_probes(
    observer_xyz: np.ndarray, target_xyz: np.ndarray, target_crouched: np.ndarray
) -> np.ndarray:
    """``[N, 11, 3]`` probe points on the targets' bodies, spread across the line from the
    observer."""
    horizontal = target_xyz[:, :2] - observer_xyz[:, :2]
    norm = np.linalg.norm(horizontal, axis=1)
    lateral = (
        np.stack([-horizontal[:, 1], horizontal[:, 0]], axis=1)
        / np.where(norm < 1e-6, 1.0, norm)[:, None]
    )
    lateral[norm < 1e-6] = (0.0, 1.0)
    height = np.where(target_crouched, BODY_HEIGHT_CROUCHED, BODY_HEIGHT)
    points = np.repeat(target_xyz[:, None, :3], len(BODY_PROBES), axis=1)
    points[:, :, :2] += (
        lateral[:, None, :] * (_LATERAL_FRACTION * (PLAYER_WIDTH * 0.5))[None, :, None]
    )
    points[:, :, 2] += _HEIGHT_FRACTION[None, :] * height[:, None]
    return points


def project_probes(
    eyes: np.ndarray, yaw: np.ndarray, pitch: np.ndarray, probes: np.ndarray
) -> np.ndarray:
    """``[N, 11]`` bool: the probes that fall inside the camera's frustum (no screen margin)."""
    forward, right, up = camera_axes(yaw, pitch)
    delta = probes - eyes[:, None, :]
    depth = np.einsum("nkc,nc->nk", delta, forward)
    safe = np.maximum(depth, NEAR_U)
    tan_h = math.tan(math.radians(HFOV_DEGREES / 2.0))
    tan_v = math.tan(math.radians(VFOV_DEGREES / 2.0))
    u = (np.einsum("nkc,nc->nk", delta, right) / safe) / tan_h * 0.5 + 0.5
    v = 0.5 - (np.einsum("nkc,nc->nk", delta, up) / safe) / tan_v * 0.5
    return (depth > NEAR_U) & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)


def _pack(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint16) * _PROBE_BITS).sum(axis=1, dtype=np.uint16)


def _unpack(masks: np.ndarray) -> np.ndarray:
    return ((masks[:, None] >> np.arange(len(BODY_PROBES), dtype=np.uint16)[None]) & 1).astype(bool)


def _decide(clear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(visible, decisive)`` of ``[N, 11]`` clear probes: visible if a head probe or two probes
    are clear; a lone clear probe of the body is undecided."""
    count = clear.sum(axis=1)
    visible = (clear & _HEAD[None]).any(axis=1) | (count >= 2)
    return visible, (count == 0) | visible


@dataclass
class LineOfSight:
    """The map's line of sight of ``N`` observer-target frames.

    Attributes:
        valid (np.ndarray): ``[N]`` bool, the frames that were tested.
        probes (np.ndarray): ``[N, 11, 3]`` float64, the target's probe points.
        projected (np.ndarray): ``[N]`` uint16, bit ``k`` set: probe ``k`` is inside the frustum.
        clear_upper (np.ndarray): ``[N]`` uint16, the projected probes no opaque triangle blocks.
        clear_lower (np.ndarray): ``[N]`` uint16, those no opaque or possible triangle blocks.
        ambiguous (np.ndarray): ``[N]`` uint16, those only a possible triangle blocks.
        tested_count (np.ndarray): ``[N]`` uint8, projected probes.
        clear_count_upper (np.ndarray): ``[N]`` uint8, probes in ``clear_upper``.
        los_upper (np.ndarray): ``[N]`` bool, visible with the opaque triangles alone.
        los_lower (np.ndarray): ``[N]`` bool, visible with the possible triangles blocking too.
        label_valid (np.ndarray): ``[N]`` bool, both are decisive and agree.
    """

    valid: np.ndarray
    probes: np.ndarray
    projected: np.ndarray
    clear_upper: np.ndarray
    clear_lower: np.ndarray
    ambiguous: np.ndarray
    tested_count: np.ndarray
    clear_count_upper: np.ndarray
    los_upper: np.ndarray
    los_lower: np.ndarray
    label_valid: np.ndarray

    @property
    def in_frustum(self) -> np.ndarray:
        return self.valid & (self.tested_count > 0)


def observer_eyes(states: np.ndarray, crouched: np.ndarray) -> np.ndarray:
    """``[N, 3]`` camera positions of observers at ``states`` ``[N, >=3]`` (feet)."""
    eyes = states[:, :3].copy()
    eyes[:, 2] += np.where(crouched, CROUCHED_EYE_HEIGHT, EYE_HEIGHT)
    return eyes


def line_of_sight(
    observer: np.ndarray,
    target: np.ndarray,
    *,
    observer_crouched: np.ndarray,
    target_crouched: np.ndarray,
    valid: np.ndarray,
    occluders: Occluders,
) -> LineOfSight:
    """The map's line of sight from each observer state to the target's body at the same frame.

    Args:
        observer (np.ndarray): ``[N, 5]`` float64 ``x, y, z, yaw, pitch`` (feet; u, degrees).
        target (np.ndarray): ``[N, 5]`` float64, the target's state.
        observer_crouched (np.ndarray): ``[N]`` bool.
        target_crouched (np.ndarray): ``[N]`` bool.
        valid (np.ndarray): ``[N]`` bool, the frames to test; the others stay empty.
        occluders (Occluders): the map.
    """
    n = len(observer)
    probes = np.zeros((n, len(BODY_PROBES), 3))
    projected = np.zeros((n, len(BODY_PROBES)), dtype=bool)
    eyes = np.zeros((n, 3))
    ids = np.flatnonzero(valid)
    if len(ids):
        eyes[ids] = observer_eyes(observer[ids], observer_crouched[ids])
        probes[ids] = body_probes(observer[ids], target[ids], target_crouched[ids])
        projected[ids] = project_probes(eyes[ids], observer[ids, 3], observer[ids, 4], probes[ids])
    frame_ids, probe_ids = np.nonzero(projected)
    upper = np.zeros_like(projected)
    lower = np.zeros_like(projected)
    ambiguous = np.zeros_like(projected)
    if len(frame_ids):
        origins, endpoints = eyes[frame_ids], probes[frame_ids, probe_ids]
        blocked_opaque = occluders.opaque.blocked(origins, endpoints)
        upper[frame_ids, probe_ids] = ~blocked_opaque
        if occluders.possible is None:
            lower[:] = upper
        else:
            blocked_possible = occluders.possible.blocked(origins, endpoints)
            ambiguous[frame_ids, probe_ids] = ~blocked_opaque & blocked_possible
            lower[frame_ids, probe_ids] = ~blocked_opaque & ~blocked_possible
    los_upper, decisive_upper = _decide(upper)
    los_lower, decisive_lower = _decide(lower)
    return LineOfSight(
        valid=valid,
        probes=probes,
        projected=_pack(projected),
        clear_upper=_pack(upper),
        clear_lower=_pack(lower),
        ambiguous=_pack(ambiguous),
        tested_count=projected.sum(axis=1).astype(np.uint8),
        clear_count_upper=upper.sum(axis=1).astype(np.uint8),
        los_upper=valid & los_upper,
        los_lower=valid & los_lower,
        label_valid=valid & decisive_upper & decisive_lower & (los_upper == los_lower),
    )


@dataclass
class PlayerCapsules:
    """Which probes of a pair test other players may block.

    Attributes:
        blocked (np.ndarray): ``[N, 11]`` bool, probes a capsule intersects.
        unknown (np.ndarray): ``[N, 11]`` bool, projected probes while some other player's state
            is unknown.
        blocker_slots (np.ndarray): ``[N, 11]`` uint16, bit ``s``: the capsule of slot ``s`` blocks.
        unknown_slots (np.ndarray): ``[N]`` uint16, bit ``s``: slot ``s``'s state is unknown.
        complete (np.ndarray): ``[N]`` bool, every other player's state is known.
    """

    blocked: np.ndarray
    unknown: np.ndarray
    blocker_slots: np.ndarray
    unknown_slots: np.ndarray
    complete: np.ndarray


def player_capsule_blocks(
    observer: np.ndarray,
    observer_crouched: np.ndarray,
    probes: np.ndarray,
    projected: np.ndarray,
    others: np.ndarray,
    alive: np.ndarray,
    crouched: np.ndarray,
    known: np.ndarray,
    slots: np.ndarray,
) -> PlayerCapsules:
    """The probes the other players' capsules block, frame by frame.

    A player is a vertical capsule on its feet (radius :data:`CAPSULE_RADIUS`, axis inset
    :data:`CAPSULE_AXIS_INSET`); the ray from the eye to a probe is shortened by
    :data:`CAPSULE_RAY_TRIM` at both ends. A dead player or one of unknown state blocks nothing; a
    projected probe is unknown while any other player's state is.

    Args:
        observer (np.ndarray): ``[N, >=3]`` the observer's states.
        observer_crouched (np.ndarray): ``[N]`` bool.
        probes (np.ndarray): ``[N, 11, 3]`` the target's probe points (:class:`LineOfSight`).
        projected (np.ndarray): ``[N]`` uint16 projected probes of the pair test.
        others (np.ndarray): ``[N, P, >=3]`` the other players' states (feet).
        alive (np.ndarray): ``[N, P]`` bool.
        crouched (np.ndarray): ``[N, P]`` bool.
        known (np.ndarray): ``[N, P]`` bool, the player's state (or death) is known.
        slots (np.ndarray): ``[P]`` int, their slots (unique, below 16).
    """
    n, count = len(observer), probes.shape[1]
    unknown_state = ~known
    complete = ~unknown_state.any(axis=1)
    slot_bits = np.left_shift(np.uint16(1), slots.astype(np.uint16))[None, :]
    unknown_slots = np.bitwise_or.reduce(
        np.where(unknown_state, slot_bits, np.uint16(0)), axis=1, initial=np.uint16(0)
    )
    inside = _unpack(projected)
    blocked = np.zeros((n, count), dtype=bool)
    blocker_slots = np.zeros((n, count), dtype=np.uint16)
    unknown = inside & ~complete[:, None]
    frame_ids, probe_ids = np.nonzero(inside)
    if len(frame_ids) and len(slots):
        origins = observer_eyes(observer, observer_crouched)[frame_ids]
        endpoints = probes[frame_ids, probe_ids]
        delta = endpoints - origins
        lengths = np.linalg.norm(delta, axis=1)
        usable = lengths > 2.0 * CAPSULE_RAY_TRIM
        direction = delta / np.where(usable, lengths, 1.0)[:, None]
        start = origins + direction * CAPSULE_RAY_TRIM
        span = (endpoints - direction * CAPSULE_RAY_TRIM) - start
        span_squared = np.sum(span * span, axis=1)

        centers = others[frame_ids, :, :3]
        valid = alive[frame_ids] & known[frame_ids] & np.isfinite(centers).all(axis=2)
        valid &= usable[:, None]
        body = np.where(crouched[frame_ids], BODY_HEIGHT_CROUCHED, BODY_HEIGHT)
        bottom = centers[:, :, 2] + CAPSULE_AXIS_INSET
        top = centers[:, :, 2] + body - CAPSULE_AXIS_INSET

        # Exact distance of a ray segment to a vertical capsule axis: the side candidate keeps the
        # ray's z on the axis and minimises the xy distance; the two end points cover the caps.
        ray_z, span_z = start[:, None, 2], span[:, None, 2]
        moving = np.abs(span_z) > 1e-12
        safe_span_z = np.where(moving, span_z, 1.0)
        t_a, t_b = (bottom - ray_z) / safe_span_z, (top - ray_z) / safe_span_z
        enter = np.maximum(np.minimum(t_a, t_b), 0.0)
        leave = np.minimum(np.maximum(t_a, t_b), 1.0)
        resting = ~moving & (ray_z >= bottom) & (ray_z <= top)
        enter, leave = np.where(resting, 0.0, enter), np.where(resting, 1.0, leave)
        side_valid = (moving | resting) & (enter <= leave)

        relative = start[:, None, :2] - centers[:, :, :2]
        span_xy = span[:, None, :2]
        a = np.sum(span_xy * span_xy, axis=2)
        b = np.sum(relative * span_xy, axis=2)
        parameter = np.where(a > 1e-12, -b / np.maximum(a, 1e-12), 0.0)
        parameter = np.minimum(np.maximum(parameter, enter), leave)
        closest = relative + parameter[:, :, None] * span_xy
        distance = np.where(side_valid, np.sum(closest * closest, axis=2), np.inf)

        norm = np.maximum(span_squared[:, None], 1e-12)

        def cap_distance(z: np.ndarray) -> np.ndarray:
            point = centers.copy()
            point[:, :, 2] = z
            offset = point - start[:, None, :]
            t = np.clip(np.sum(offset * span[:, None, :], axis=2) / norm, 0.0, 1.0)
            near = start[:, None, :] + t[:, :, None] * span[:, None, :]
            return np.sum((near - point) ** 2, axis=2)

        distance = np.minimum(distance, np.minimum(cap_distance(bottom), cap_distance(top)))
        hit = valid & (distance <= CAPSULE_RADIUS**2 + 1e-9 * max(1.0, CAPSULE_RADIUS**2))
        blocked[frame_ids, probe_ids] = hit.any(axis=1)
        blocker_slots[frame_ids, probe_ids] = np.bitwise_or.reduce(
            np.where(hit, slot_bits, np.uint16(0)), axis=1, initial=np.uint16(0)
        )
    return PlayerCapsules(blocked, unknown, blocker_slots, unknown_slots, complete)


@dataclass
class LineOfSightWithPlayers:
    """A pair test with the other players applied (:func:`apply_player_capsules`).

    Attributes:
        los_lower (np.ndarray): ``[N]`` bool, visible with possible triangles and players blocking.
        label_valid (np.ndarray): ``[N]`` bool.
        clear_lower (np.ndarray): ``[N]`` uint16 probes clear of every possible blocker.
        clear_count_lower (np.ndarray): ``[N]`` uint8.
        proxy_ambiguous (np.ndarray): ``[N]`` uint16, probes only a capsule blocks.
        unknown_ambiguous (np.ndarray): ``[N]`` uint16, probes open to the map that are unknown
            because a player's state is.
        possible_ambiguous (np.ndarray): ``[N]`` uint16, the union of the two.
    """

    los_lower: np.ndarray
    label_valid: np.ndarray
    clear_lower: np.ndarray
    clear_count_lower: np.ndarray
    proxy_ambiguous: np.ndarray
    unknown_ambiguous: np.ndarray
    possible_ambiguous: np.ndarray


def apply_player_capsules(test: LineOfSight, capsules: PlayerCapsules) -> LineOfSightWithPlayers:
    """The pair test's lower bound with the other players' capsules blocking."""
    inside = _unpack(test.projected)
    valid = test.valid[:, None]
    blocked = (capsules.blocked | capsules.unknown) & inside & valid
    proxy = capsules.blocked & inside & valid
    unknown = capsules.unknown & inside & valid
    upper = _unpack(test.clear_upper)
    lower = _unpack(test.clear_lower) & ~blocked
    los_lower, decisive_lower = _decide(lower)
    los_lower &= test.valid
    decisive_upper = (test.clear_count_upper == 0) | test.los_upper
    return LineOfSightWithPlayers(
        los_lower=los_lower,
        label_valid=test.valid & decisive_lower & decisive_upper & (los_lower == test.los_upper),
        clear_lower=_pack(lower),
        clear_count_lower=lower.sum(axis=1).astype(np.uint8),
        proxy_ambiguous=_pack(proxy & upper),
        unknown_ambiguous=_pack(unknown & upper),
        possible_ambiguous=_pack(blocked & upper),
    )
