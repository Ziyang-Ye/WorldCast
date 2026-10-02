"""A small synthetic world for the scene golden tests: a box room with a pillar and an open roof.

Depth maps are ray-cast from the geometry, so different cameras see one consistent surface
(coverage, eviction and retrieval then have something real to decide).  Computed independently of
the code under test.
"""

import numpy as np

H, W = 24, 42
FAR = 4096.0
TAN = (1.3333, 0.75)  # the paper run's fixed retrieval fov (tan_h, tan_v)

ROOM_LO = np.array([-600.0, -250.0, -600.0])  # x right, y down, z forward (world = identity camera)
ROOM_HI = np.array([600.0, 250.0, 600.0])
PILLAR_LO = np.array([-90.0, -250.0, 230.0])
PILLAR_HI = np.array([90.0, 250.0, 330.0])


def rot(yaw: float, pitch: float = 0.0) -> np.ndarray:
    """Rotation whose columns are right / down / forward: yaw about the down axis, then pitch about
    right."""
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return ry @ rx


def cam(eye, yaw: float, pitch: float = 0.0) -> np.ndarray:
    c = np.eye(4)
    c[:3, :3] = rot(yaw, pitch)
    c[:3, 3] = eye
    return c


def rays(c2w, tan=TAN) -> np.ndarray:
    xs = (np.arange(W) + 0.5) * 2 / W - 1
    ys = (np.arange(H) + 0.5) * 2 / H - 1
    xx, yy = np.meshgrid(xs, ys)
    d = np.stack([xx.ravel() * tan[0], yy.ravel() * tan[1], np.ones(H * W)], 1)
    return d @ np.asarray(c2w, dtype=np.float64)[:3, :3].T


def room_depth(c2w, tan=TAN) -> np.ndarray:
    """Axial depth ``[24, 42]`` of the room seen from ``c2w``; rays leaving through the roof (``y =
    -250``) miss."""
    c2w = np.asarray(c2w, dtype=np.float64)
    eye, r = c2w[:3, 3], rays(c2w, tan)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lo = (ROOM_LO[None] - eye[None]) / r
        t_hi = (ROOM_HI[None] - eye[None]) / r
        t_exit_axis = np.where(r > 0, t_hi, t_lo)
        t_exit_axis = np.where(r == 0, np.inf, t_exit_axis)
        axis = np.argmin(t_exit_axis, 1)
        t_exit = t_exit_axis[np.arange(len(r)), axis]
        roof = (axis == 1) & (r[:, 1] < 0)
        p_lo = (PILLAR_LO[None] - eye[None]) / r
        p_hi = (PILLAR_HI[None] - eye[None]) / r
        t_near = np.nanmax(np.minimum(p_lo, p_hi), 1)
        t_far = np.nanmin(np.maximum(p_lo, p_hi), 1)
    pillar = (t_near <= t_far) & (t_near > 0) & (t_near < t_exit)
    d = np.where(pillar, t_near, np.where(roof, FAR, t_exit))
    return d.reshape(H, W)


def trajectory(k: int, n: int) -> np.ndarray:
    """Client ``k``'s cameras for latents ``0..n-1``, ``[n, 4, 4]``: a circle around the room
    centre, turning."""
    out = np.zeros((n, 4, 4))
    for li in range(n):
        a = 0.09 * li + 2.1 * k
        eye = np.array([170.0 * np.cos(a), 30.0 * np.sin(0.3 * li + k), 170.0 * np.sin(a) - 60.0])
        out[li] = cam(eye, yaw=0.55 * k + 0.07 * li, pitch=0.05 * np.sin(0.2 * li))
    return out


# --------------------------------------------------------------- fake latents and a stub depth head
#: Fake latents are ``[C, H, W] = [2, 3, 3]``: channel 0 carries a code (1000 k + latent index) that
#: tells the stub depth head which camera the latent was "drawn" from; channel 1 is content, which
#: the fake generator derives from the retrieved memory slot, so every read feeds into later
#: latents, depths and decisions.
LAT_SHAPE = (2, 3, 3)


def base_latent(k: int, li: int):
    import torch

    x = torch.zeros(LAT_SHAPE, dtype=torch.float32)
    x[0] = float(1000 * k + li)
    x[1] = float(np.sin(0.37 * li + 1.3 * k)) + 0.01 * torch.arange(9, dtype=torch.float32).reshape(
        3, 3
    )
    return x


def gen_block(k: int, s: int, slot):
    """The fake generator: block ``s`` of client ``k`` ``[4, 2, 3, 3]`` float32, shifted by the
    slot's content."""
    import torch

    out = torch.stack([base_latent(k, s + j) for j in range(4)])
    if slot is not None:
        sig = torch.as_tensor(slot, dtype=torch.float32)[:, 1].mean()
        out[:, 1] = out[:, 1] + 0.5 * sig
    return out


class StubDepth:
    """``depth_grid(latents [T, 2, 3, 3]) -> log axial depth [T, 4, 24, 42]`` float32, from the
    latent's code (its camera, ray-cast in the room) and its content (a few-percent scale), like the
    real head's interface.
    """

    def __init__(self, trajs) -> None:
        self.trajs = trajs
        self.n_calls = 0
        self.n_latents = 0
        self._room: dict = {}

    def depth_grid(self, latents) -> np.ndarray:
        x = (
            latents.detach().cpu().float().numpy()
            if hasattr(latents, "detach")
            else np.asarray(latents, dtype=np.float32)
        )
        out = np.empty((x.shape[0], 4, H, W), np.float32)
        for t in range(x.shape[0]):
            k, li = divmod(int(round(float(x[t, 0, 0, 0]))), 1000)
            if (k, li) not in self._room:
                self._room[(k, li)] = room_depth(self.trajs[k][li])
            d = self._room[(k, li)]
            f = 1.0 + 0.03 * np.tanh(float(x[t, 1].mean()))
            dd = np.where(d >= 0.95 * FAR, FAR, d * f)
            for c in range(4):
                out[t, c] = np.log(dd * (1.0 + 0.002 * (c - 3)))
        self.n_calls += 1
        self.n_latents += int(x.shape[0])
        return out
