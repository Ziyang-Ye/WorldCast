"""The decoder: the latents layout, the streaming-decode arguments and the mp4 writer."""

import numpy as np
import pytest
import torch

from worldcast.engine.inference import decode as D
from worldcast.modeling.wan22 import vae as V


def _latents(n: int, h: int, w: int, seed: int) -> torch.Tensor:
    """Pipeline-layout latents ``[1, N, 48, h, w]`` float32, as ``latents.npy[None]``."""
    return torch.randn(1, n, 48, h, w, generator=torch.Generator().manual_seed(seed))


def test_stream_decode_pixels_rejects_bad_arguments(new_tiny_vae):
    new = new_tiny_vae
    dec = V.Wan22StreamingDecoder(new)
    scale = D.vae_decode_scale("cpu", torch.float32)
    with pytest.raises(ValueError, match="chunk"):
        list(D.stream_decode_pixels(dec, _latents(5, 4, 6, 1), chunk=0, scale=scale))
    with pytest.raises(ValueError, match="one video"):
        list(
            D.stream_decode_pixels(dec, _latents(5, 4, 6, 1).expand(2, -1, -1, -1, -1), scale=scale)
        )


def _frames(n: int) -> np.ndarray:
    """Smooth gradients plus noise, uint8 ``[n, 384, 672, 3]``."""
    g = np.random.default_rng(73)
    y, x = np.mgrid[0:384, 0:672]
    base = np.stack([x * 255 / 671, y * 255 / 383, (x + y) * 255 / 1054], -1)
    frames = [np.clip(base + 20 * i + g.normal(0, 8, base.shape), 0, 255) for i in range(n)]
    return np.stack(frames).astype(np.uint8)


def _reference_write(frames, path) -> int:
    """The paper writer, as in ``wholeround_decode.py:169-182``."""
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        path, fps=16.0, codec="libx264", quality=8, macro_block_size=1, ffmpeg_log_level="error"
    )
    written = 0
    try:
        for frame in frames:
            writer.append_data(frame)
            written += 1
    finally:
        writer.close()
    return written


def test_write_mp4_matches_the_paper_writer(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    frames = _frames(9)
    ref, new = tmp_path / "ref.mp4", tmp_path / "out" / "new.mp4"
    assert _reference_write(frames, str(ref)) == 9
    assert D.write_mp4(iter(frames), new) == 9
    assert not (new.parent / ".new.mp4").exists()
    assert new.read_bytes() == ref.read_bytes()
    pytest.importorskip("cv2")
    import cv2

    assert D.count_mp4_frames(new) == 9
    cap = cv2.VideoCapture(str(new))
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (
        672,
        384,
    )
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(16.0)
    cap.release()


def test_load_latents_checks_the_block_layout(tmp_path):
    np.save(tmp_path / "ok.npy", np.zeros((9, 48, 24, 42), np.float32))
    lat = D.load_latents(tmp_path / "ok.npy")
    assert lat.shape == (1, 9, 48, 24, 42) and lat.dtype == torch.float32
    np.save(tmp_path / "bad.npy", np.zeros((8, 48, 24, 42), np.float32))
    with pytest.raises(ValueError, match="1 \\+ 4 k"):
        D.load_latents(tmp_path / "bad.npy")
    np.save(tmp_path / "flat.npy", np.zeros((9, 48), np.float32))
    with pytest.raises(ValueError, match="expected latents"):
        D.load_latents(tmp_path / "flat.npy")
