"""The map geometry: a map's collision mesh, its rays and its occlusion test.

Optional dependency, imported when used: ``trimesh`` with its embree backend.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .camera import as_float64, grid_rays
from .latents import TOKEN_GRID

__all__ = ["EPS_U", "MapMesh", "MeshLibrary", "load_collision_mesh", "visible_from"]

#: Self-hit slack of the occlusion ray, u.
EPS_U = 4.0


def visible_from(intersector: Any, points: np.ndarray, eye: np.ndarray, eps: float) -> np.ndarray:
    """``[S]`` bool: no mesh lies between ``eye`` and the point (the first hit no nearer than the
    point minus ``eps``); points within ``eps`` of the eye are not visible."""
    out = np.zeros(len(points), bool)
    if len(points) == 0:
        return out
    d = points - eye
    distance = np.linalg.norm(d, axis=1)
    ok = distance > eps
    if not ok.any():
        return out
    directions = d[ok] / distance[ok][:, None]
    origins = np.repeat(eye[None, :], int(ok.sum()), 0)
    locations, rays, _ = intersector.intersects_location(origins, directions, multiple_hits=False)
    hit = np.full(int(ok.sum()), np.inf)
    if len(rays):
        np.minimum.at(hit, rays, np.linalg.norm(locations - origins[rays], axis=1))
    out[np.flatnonzero(ok)] = hit >= (distance[ok] - eps)
    return out


class MapMesh:
    """One map's collision mesh with an embree ray intersector (u, world +z up)."""

    def __init__(self, mesh: Any) -> None:
        try:
            from trimesh.ray import ray_pyembree
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "memory frames need trimesh with its embree backend (pyembree or embreex)"
            ) from exc
        self.intersector = ray_pyembree.RayMeshIntersector(mesh)
        # builds the BVH now (no effect on any result)
        self.intersector.intersects_location(
            np.zeros((1, 3)), np.array([[1.0, 0, 0]]), multiple_hits=False
        )

    def visible(self, points: np.ndarray, eye: np.ndarray) -> np.ndarray:
        """``[S]`` bool: the world points ``[S, 3]`` are visible from ``eye`` ``[3]``
        (:func:`visible_from`)."""
        points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
        eye = np.asarray(eye, dtype=np.float64).reshape(3)
        return visible_from(self.intersector, points, eye, EPS_U)

    def surface_tokens(
        self, c2w: np.ndarray, tans: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cast one ray per token of the 12 x 21 grid through camera ``c2w``.

        Args:
            c2w (np.ndarray): ``[4, 4]`` camera-to-world (an array or a tensor).
            tans (Sequence[float]): ``(tan_h, tan_v)`` of its field of view.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: the first hits ``[M, 3]`` float64, their
            token ids ``[M]`` int64 (row-major, ascending) and the hit faces' normals ``[M, 3]``;
            tokens whose ray misses are absent.
        """
        c = as_float64(c2w)
        rays = grid_rays(c, tans, TOKEN_GRID)
        origins = np.broadcast_to(c[:3, 3], rays.shape).copy()
        locations, ray_ids, faces = self.intersector.intersects_location(
            origins, rays, multiple_hits=False
        )
        order = np.argsort(ray_ids)
        return (
            np.asarray(locations)[order],
            np.asarray(ray_ids, dtype=np.int64)[order],
            np.asarray(self.intersector.mesh.face_normals)[np.asarray(faces)[order]],
        )


def load_collision_mesh(path: str | Path) -> Any:
    """A ``*_world_collision_complete.glb`` as one ``trimesh.Trimesh`` in u (+z up).

    Every geometry node of the file shares one translation-free transform (an axis permutation
    times 0.0254, u to glTF metres), so each geometry's raw vertices are already in u and are
    concatenated as they are.
    """
    import trimesh

    scene = trimesh.load(str(path), process=False)
    transforms = [scene.graph[n][0] for n in scene.graph.nodes_geometry]
    if max(np.abs(t - transforms[0]).max() for t in transforms) != 0.0 or not np.allclose(
        transforms[0][:3, 3], 0.0
    ):
        raise ValueError(f"{path}: node transforms are not one shared, translation-free transform")
    vertices, faces, offset = [], [], 0
    for node in scene.graph.nodes_geometry:
        geometry = scene.geometry[scene.graph[node][1]]
        faces.append(np.asarray(geometry.faces, dtype=np.int64) + offset)
        vertices.append(np.asarray(geometry.vertices, dtype=np.float64))
        offset += len(vertices[-1])
    return trimesh.Trimesh(vertices=np.vstack(vertices), faces=np.vstack(faces), process=False)


class MeshLibrary:
    """One :class:`MapMesh` per map, loaded on first use and kept (a loader worker reads windows of
    every map).

    Args:
        maps (Mapping[str, str | Path | MapMesh]): per map name, the path of its collision mesh
            (config ``data.collision_meshes``) or the mesh itself.
    """

    def __init__(self, maps: Mapping[str, str | Path | MapMesh]) -> None:
        self._maps = {str(name): source for name, source in maps.items()}

    def get(self, map_name: str) -> MapMesh:
        """The mesh of ``map_name``; ``KeyError`` for a map the library was not given."""
        if map_name not in self._maps:
            raise KeyError(f"no collision mesh for map {map_name!r} in data.collision_meshes")
        if not isinstance(self._maps[map_name], MapMesh):
            self._maps[map_name] = MapMesh(load_collision_mesh(self._maps[map_name]))
        return self._maps[map_name]
