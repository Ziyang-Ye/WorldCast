"""FVD: the Fréchet distance against closed forms, what I3D is fed, and the clip sets (CPU,
stand-ins for decord, OpenCV and the detector)."""

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from worldcast.data.latents import WINDOW_VIDEO_FRAMES
from worldcast.engine.evaluation import fvd
from worldcast.engine.evaluation.fvd import (
    FEATURE_DIM,
    MIN_CLIPS,
    SEGMENT_FRAMES,
    SQRTM_EPS,
    frechet_distance,
    fvd_from_dir,
    i3d_features,
    load_i3d,
    read_window,
    round_features,
)


class RecordingI3D(torch.nn.Module):
    """A stand-in detector: records what it is fed; its feature is the per-channel mean and
    standard deviation of the clip, in float64."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs: list[tuple[torch.Tensor, bool, bool, bool]] = []

    def forward(
        self, x: torch.Tensor, rescale: bool, resize: bool, return_features: bool
    ) -> torch.Tensor:
        self.inputs.append((x.clone(), rescale, resize, return_features))
        moments = [x.mean(dim=(2, 3, 4)), x.std(dim=(2, 3, 4))]
        return torch.cat(moments, dim=1).double()


def frames_of(indices: list[int], height: int = 2, width: int = 3) -> np.ndarray:
    """``[T, H, W, 3]`` uint8 frames that carry their index: ``i % 256`` and ``i // 256``."""
    frames = np.zeros((len(indices), height, width, 3), dtype=np.uint8)
    frames[..., 0] = np.array(indices)[:, None, None] % 256
    frames[..., 1] = np.array(indices)[:, None, None] // 256
    return frames


def fake_decord(monkeypatch: pytest.MonkeyPatch, frames: int) -> None:
    """A stand-in decord whose videos hold ``frames`` frames that carry their index."""

    class VideoReader:
        def __init__(self, path: str, num_threads: int) -> None:
            self.path = path

        def __len__(self) -> int:
            return frames

        def get_batch(self, indices: list[int]) -> types.SimpleNamespace:
            assert max(indices) < frames
            batch = frames_of(indices)
            return types.SimpleNamespace(asnumpy=lambda: batch)

    monkeypatch.setitem(sys.modules, "decord", types.SimpleNamespace(VideoReader=VideoReader))


def indices_fed(x: torch.Tensor, low: int = 0, high: int = 1) -> list[int]:
    """The frame indices a ``[1, 3, T, H, W]`` detector input at 0-255 carries."""
    values = x[0, :, :, 0, 0].round().long()
    return (values[low] + 256 * values[high]).tolist()


def gaussian_set(n: int, d: int, mean: np.ndarray, std: np.ndarray, seed: int) -> np.ndarray:
    """``[n, d]`` features whose sample mean is ``mean`` and sample covariance ``diag(std^2)``."""
    z = np.random.default_rng(seed).standard_normal((n, d))
    q, _ = np.linalg.qr(z - z.mean(axis=0))
    return np.sqrt(n - 1) * q * std + mean


def test_the_distance_has_its_closed_forms():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    mu_r, mu_f = rng.standard_normal(16), rng.standard_normal(16)
    s, t = rng.uniform(0.2, 3.0, 16), rng.uniform(0.2, 3.0, 16)
    real, fake = gaussian_set(200, 16, mu_r, s, 1), gaussian_set(200, 16, mu_f, t, 2)
    # commuting covariances: |mu_r - mu_f|^2 + sum (s - t)^2
    expected = np.sum((mu_r - mu_f) ** 2) + np.sum((s - t) ** 2)
    assert frechet_distance(real, fake) == pytest.approx(expected, rel=1e-9)
    # a shift: the means alone
    shift = rng.standard_normal(16)
    assert frechet_distance(real, real + shift) == pytest.approx(shift @ shift, rel=1e-9)
    # a scale and a shift: |(1 - a) mu - b|^2 + (1 - a)^2 tr C
    a = 0.8
    gap = (1 - a) * real.mean(axis=0) - shift
    expected = gap @ gap + (1 - a) ** 2 * np.trace(np.cov(real, rowvar=False))
    assert frechet_distance(real, a * real + shift) == pytest.approx(expected, rel=1e-9)
    assert frechet_distance(real, real) == pytest.approx(0.0, abs=1e-9)


def test_the_distance_is_taken_in_float64_whatever_the_input():
    pytest.importorskip("scipy")
    g = torch.Generator().manual_seed(3)
    real, fake = torch.randn(80, 8, generator=g), torch.randn(80, 8, generator=g) + 0.5
    value = frechet_distance(real, fake)
    assert value == frechet_distance(real.numpy(), fake.numpy())
    assert value == frechet_distance(real.double(), fake.double())
    assert isinstance(value, float)


def test_a_small_imaginary_part_of_the_root_is_dropped_a_large_one_raises(monkeypatch):
    linalg = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(4)
    real, fake = rng.standard_normal((50, 6)), rng.standard_normal((50, 6)) * 2.0
    exact = frechet_distance(real, fake)
    root = linalg.sqrtm

    monkeypatch.setattr(linalg, "sqrtm", lambda a: root(a).real + 1e-4j)
    assert frechet_distance(real, fake) == pytest.approx(exact, rel=1e-12)
    monkeypatch.setattr(linalg, "sqrtm", lambda a: root(a).real + 1e-2j)
    with pytest.raises(FloatingPointError, match="imaginary"):
        frechet_distance(real, fake)


def test_a_root_that_is_not_finite_is_taken_again_with_eps_on_the_diagonals(monkeypatch):
    linalg = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(5)
    real, fake = rng.standard_normal((40, 5)), rng.standard_normal((40, 5))
    root, products = linalg.sqrtm, []

    def failing_once(a: np.ndarray) -> np.ndarray:
        products.append(a)
        return np.full_like(a, np.nan) if len(products) == 1 else root(a)

    monkeypatch.setattr(linalg, "sqrtm", failing_once)
    value = frechet_distance(real, fake)
    eye = np.eye(5)
    cov_r, cov_f = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    np.testing.assert_array_equal(
        products[1], (cov_r + SQRTM_EPS * eye) @ (cov_f + SQRTM_EPS * eye)
    )
    assert np.isfinite(value)


def test_the_distance_refuses_what_it_cannot_measure():
    pytest.importorskip("scipy")
    features = np.random.default_rng(6).standard_normal((10, 4))
    with pytest.raises(ValueError, match="two real"):
        frechet_distance(features[:1], features)
    with pytest.raises(ValueError, match="expected"):
        frechet_distance(features, features[:, :3])
    broken = features.copy()
    broken[2, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        frechet_distance(features, broken)


def test_i3d_sees_the_whole_clip_at_0_255_with_its_own_rescale_and_resize():
    model = RecordingI3D()
    clip = torch.rand(5, 3, 4, 6, generator=torch.Generator().manual_seed(7))
    clip[0] = 2.0  # out of range: clamped to 1
    feature = i3d_features(model, clip)
    x, rescale, resize, return_features = model.inputs[0]
    assert (rescale, resize, return_features) == (True, True, True)
    assert x.shape == (1, 3, 5, 4, 6)  # [1, 3, T, H, W]
    assert torch.equal(x[0, :, 0], torch.full((3, 4, 6), 255.0))  # frame 0 is fed
    torch.testing.assert_close(x[0, :, 1:], clip[1:].permute(1, 0, 2, 3) * 255.0)
    assert feature.shape == (6,) and feature.dtype == torch.float32
    with pytest.raises(ValueError, match="T >= 2"):
        i3d_features(model, clip[:1])


def test_a_window_is_all_its_161_frames_from_frame_0(monkeypatch):
    fake_decord(monkeypatch, frames=WINDOW_VIDEO_FRAMES + 9)
    clip = read_window("window.mp4")
    assert clip.shape == (WINDOW_VIDEO_FRAMES, 3, 2, 3) and clip.dtype == torch.float32
    index = (clip[:, 0, 0, 0] * 255).round() + 256 * (clip[:, 1, 0, 0] * 255).round()
    assert index.long().tolist() == list(range(WINDOW_VIDEO_FRAMES))


def make_eval_dir(root: Path, count: int) -> Path:
    for i in range(count):
        for kind in ("generation", "gt"):
            (root / kind).mkdir(parents=True, exist_ok=True)
            (root / kind / f"m{i:03d}__start{80 * i:06d}.mp4").write_bytes(f"{kind}{i}".encode())
    return root


def clip_of_file(path: str | Path) -> torch.Tensor:
    """A stand-in for :func:`read_window`: a clip drawn from the file's bytes."""
    seed = int(hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8], 16)
    return torch.rand(4, 3, 2, 2, generator=torch.Generator().manual_seed(seed))


def test_an_eval_dir_scores_every_generated_window_against_its_gt(monkeypatch, tmp_path):
    pytest.importorskip("scipy")
    eval_dir = make_eval_dir(tmp_path / "eval", MIN_CLIPS)
    monkeypatch.setattr(fvd, "read_window", clip_of_file)
    model = RecordingI3D()
    stems = sorted(p.name for p in (eval_dir / "generation").glob("*.mp4"))
    real = np.stack([i3d_features(model, clip_of_file(eval_dir / "gt" / s)) for s in stems])
    fake = np.stack([i3d_features(model, clip_of_file(eval_dir / "generation" / s)) for s in stems])
    expected = frechet_distance(real, fake)
    assert fvd_from_dir(eval_dir, model) == {"fvd": expected, "clips": MIN_CLIPS}

    cache = tmp_path / "real.npz"
    assert fvd_from_dir(eval_dir, model, real_cache=cache)["fvd"] == expected

    def generated_only(path: str | Path) -> torch.Tensor:
        assert Path(path).parent.name == "generation", "a cached gt clip was decoded again"
        return clip_of_file(path)

    monkeypatch.setattr(fvd, "read_window", generated_only)
    assert fvd_from_dir(eval_dir, model, real_cache=cache)["fvd"] == expected


def test_an_eval_dir_needs_64_windows_of_matching_shapes(monkeypatch, tmp_path):
    monkeypatch.setattr(fvd, "read_window", clip_of_file)
    small = make_eval_dir(tmp_path / "small", MIN_CLIPS - 1)
    with pytest.raises(ValueError, match="fewer than 64"):
        fvd_from_dir(small, RecordingI3D())

    def shorter_gt(path: str | Path) -> torch.Tensor:
        clip = clip_of_file(path)
        return clip[1:] if Path(path).parent.name == "gt" else clip

    monkeypatch.setattr(fvd, "read_window", shorter_gt)
    with pytest.raises(ValueError, match="gt"):
        fvd_from_dir(make_eval_dir(tmp_path / "full", MIN_CLIPS), RecordingI3D())


def fake_cv2(monkeypatch: pytest.MonkeyPatch, source_frames: int, calls: list) -> None:
    """A stand-in OpenCV whose recordings hold ``source_frames`` 4 x 6 BGR frames carrying their
    index in blue (``% 256``) and green (``// 256``); its resize halves a frame."""

    class VideoCapture:
        def __init__(self, path: str) -> None:
            self.next = 0

        def get(self, prop: int) -> float:
            return float(source_frames)

        def grab(self) -> bool:
            self.next += 1
            return self.next <= source_frames

        def retrieve(self) -> tuple[bool, np.ndarray]:
            return True, frames_of([self.next - 1], height=4, width=6)[0]

        def release(self) -> None:
            pass

    def resize(frame: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
        calls.append((size, interpolation))
        return frame[::2, ::2]

    module = types.SimpleNamespace(
        VideoCapture=VideoCapture,
        CAP_PROP_FRAME_COUNT=7,
        INTER_LINEAR=1,
        COLOR_BGR2RGB=4,
        resize=resize,
        cvtColor=lambda frame, code: frame[..., ::-1],
    )
    monkeypatch.setitem(sys.modules, "cv2", module)


def test_a_whole_round_is_scored_by_ten_second_segments_after_the_first_frame(monkeypatch):
    video_frames = 1 + 2 * SEGMENT_FRAMES + 7  # two complete segments
    source_frames = 2 * video_frames - 1
    calls: list = []
    fake_decord(monkeypatch, frames=video_frames)
    fake_cv2(monkeypatch, source_frames, calls)
    monkeypatch.setattr(fvd, "FRAME_SIZE", (2, 3))
    model = RecordingI3D()
    generated, real = round_features(model, "round.mp4", "rec.mp4", source_frames=source_frames)
    assert generated.shape == real.shape == (2, 6)
    assert set(calls) == {((3, 2), 1)}  # every frame resized to 384 x 672, bilinear
    fed = [x for x, *_ in model.inputs]
    for k in range(2):
        segment = list(range(1 + SEGMENT_FRAMES * k, 1 + SEGMENT_FRAMES * (k + 1)))
        assert indices_fed(fed[2 * k]) == segment
        # BGR -> RGB: the index is in blue (now channel 2) and green
        assert indices_fed(fed[2 * k + 1], low=2, high=1) == [2 * i for i in segment]
    with pytest.raises(RuntimeError, match="frames, expected"):
        round_features(model, "round.mp4", "rec.mp4", source_frames=source_frames + 1)


def test_a_round_shorter_than_one_segment_has_no_features(monkeypatch):
    fake_decord(monkeypatch, frames=SEGMENT_FRAMES)
    generated, real = round_features(RecordingI3D(), "round.mp4", "rec.mp4", source_frames=1)
    assert generated.shape == real.shape == (0, FEATURE_DIM)


def test_load_i3d_takes_only_the_papers_detector(tmp_path):
    path = tmp_path / "i3d_torchscript.pt"
    path.write_bytes(b"not the detector")
    with pytest.raises(ValueError, match="sha256"):
        load_i3d(path, "cpu")
