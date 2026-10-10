"""The decoder of a client's latents: the whole-sequence and the streaming decode, the uint8 frames,
the latents file, and the mp4 writer with its check that the file decodes to every frame written."""

import re
import shutil
import subprocess

import numpy as np
import pytest
import torch

from worldcast.engine.inference import decode as D
from worldcast.modeling.wan22 import vae as V


def _latents(n: int, h: int, w: int, seed: int) -> torch.Tensor:
    """Latents ``[1, N, 48, h, w]`` float32, as ``latents.npy[None]``."""
    return torch.randn(1, n, 48, h, w, generator=torch.Generator().manual_seed(seed))


def test_the_streaming_decode_gives_the_frames_of_one_decode(tiny_vae):
    latents = _latents(5, 2, 3, seed=5)
    pixels = D.decode_pixels(tiny_vae, latents)
    assert pixels.shape == (17, 3, 32, 48) and 0 <= float(pixels.min()) <= float(pixels.max()) <= 1
    scale = V.latent_scale("cpu", torch.float32)
    with torch.no_grad():
        stream = V.Wan22StreamingDecoder(tiny_vae)
        chunks = list(D.stream_decode_pixels(stream, latents, chunk=2, scale=scale))
    assert [len(chunk) for chunk in chunks] == [5, 8, 4]  # 1 + 4, 4 + 4, 4 frames
    assert torch.allclose(pixels, torch.cat(chunks), atol=1e-6)


def test_a_frame_decoder_takes_the_latents_as_they_arrive(tiny_vae):
    latents = _latents(3, 2, 3, seed=6)
    decoder = D.WanFrameDecoder(tiny_vae)
    frames = [decoder.decode(latents[0, i : i + 1]) for i in range(3)]
    assert [len(f) for f in frames] == [1, 4, 4]  # latent frame 0 is one video frame
    assert torch.allclose(torch.cat(frames), D.decode_pixels(tiny_vae, latents), atol=1e-6)


def test_stream_decode_pixels_rejects_bad_arguments(tiny_vae):
    stream = V.Wan22StreamingDecoder(tiny_vae)
    scale = V.latent_scale("cpu", torch.float32)
    with pytest.raises(ValueError, match="decode chunk must be >= 1"):
        list(D.stream_decode_pixels(stream, _latents(5, 4, 6, 1), chunk=0, scale=scale))
    two_videos = _latents(5, 4, 6, 1).expand(2, -1, -1, -1, -1)
    with pytest.raises(ValueError, match="takes one video, got batch 2"):
        list(D.stream_decode_pixels(stream, two_videos, scale=scale))


def test_frames_are_rounded_half_to_even_into_uint8():
    values = [0.0, 0.5, 1.0, 1.2, -0.1, 0.25, 0.75, 0.3]
    pixels = torch.tensor(values).reshape(1, 1, 2, 4).repeat(1, 3, 1, 1)
    frames = D.pixels_to_uint8_frames(pixels)
    assert frames.shape == (1, 2, 4, 3) and frames.dtype == np.uint8 and frames.flags.c_contiguous
    # 127.5 rounds to 128 and 76.5 to 76 (half to even); values outside [0, 1] are clamped
    assert frames[0, :, :, 0].tolist() == [[0, 128, 255, 255], [0, 64, 191, 76]]


def test_decoded_frames_are_the_uint8_frames_of_the_pixels(tiny_vae):
    latents = _latents(3, 2, 3, seed=7)
    frames = list(D.decode_frames(tiny_vae, latents, chunk=2))
    assert len(frames) == 9 and frames[0].shape == (32, 48, 3) and frames[0].dtype == np.uint8
    whole = D.pixels_to_uint8_frames(D.decode_pixels(tiny_vae, latents))
    assert np.abs(np.stack(frames).astype(int) - whole.astype(int)).max() <= 1


def _frames(n: int) -> np.ndarray:
    """Smooth gradients plus noise, uint8 ``[n, 384, 672, 3]``."""
    g = np.random.default_rng(73)
    y, x = np.mgrid[0:384, 0:672]
    base = np.stack([x * 255 / 671, y * 255 / 383, (x + y) * 255 / 1054], -1)
    frames = [np.clip(base + 20 * i + g.normal(0, 8, base.shape), 0, 255) for i in range(n)]
    return np.stack(frames).astype(np.uint8)


def _video(path) -> tuple[int, int, int, float]:
    """The frame count, width, height and fps of an mp4."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = 0
    while capture.grab():
        count += 1
    capture.release()
    return count, *size, fps


def _ramps(n: int) -> list[np.ndarray]:
    """``n`` uint8 frames ``[64, 128, 3]`` of colour ramps that move from frame to frame; none
    takes 4 KiB of the mp4, so that a missing page of the file holds the start of a frame."""
    y, x = np.mgrid[0:64, 0:128]
    return [
        np.stack(
            [(x * 8 + 3 * i) % 256, (y * 16 + 5 * i) % 256, np.full_like(x, 7 * i % 256)], -1
        ).astype(np.uint8)
        for i in range(n)
    ]


def test_the_mp4_holds_every_frame_at_16_fps(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("cv2")
    path = tmp_path / "out" / "video.mp4"
    assert D.write_mp4(iter(_frames(9)), path) == 9
    assert [p.name for p in path.parent.iterdir()] == ["video.mp4"]  # no partial file is left
    assert _video(path) == (9, 672, 384, pytest.approx(16.0))
    assert D.mp4_decoded_frames(path) == (9, "")


def test_an_interrupted_mp4_leaves_no_file(tmp_path):
    pytest.importorskip("imageio_ffmpeg")

    def frames():
        yield from _ramps(2)
        raise KeyboardInterrupt

    path = tmp_path / "out" / "video.mp4"
    with pytest.raises(KeyboardInterrupt):
        D.write_mp4(frames(), path)
    assert list(path.parent.iterdir()) == []


@pytest.mark.parametrize("loss", ["frames", "tail", "page", "bytes"])
def test_an_mp4_that_lost_frames_is_not_written(tmp_path, monkeypatch, lossy_ffmpeg, loss):
    """Its index whole and its frames missing: all of them, those from the page at the middle on,
    that page alone (the decoder finds a frame of no size), or 256 bytes in a frame (the frame
    decodes, with errors). The decode stops at the first error, and the mp4 is removed."""
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", lossy_ffmpeg(loss))
    path = tmp_path / "out" / "video.mp4"
    with pytest.raises(RuntimeError) as lost:
        D.write_mp4(iter(_ramps(40)), path)
    said = re.fullmatch(
        rf"{re.escape(str(path))} not written: it decodes to (\d+) of its 40 frames; ffmpeg:"
        r" (\[h264 @ 0x[0-9a-f]+\] .+) \(the video file is incomplete\)",
        str(lost.value),
    )
    assert said, str(lost.value)
    decoded, error = int(said[1]), said[2]
    # the frames before the first error: none of lost frames, some before a loss at the middle,
    # any before the largest frame (which lost bytes)
    assert decoded in {"frames": [0], "bytes": range(40)}.get(loss, range(1, 40))
    assert (loss == "bytes") != bool(re.search(r"Invalid NAL unit size \(0 > \d+\)\.$", error))
    assert list(path.parent.iterdir()) == []


def test_the_check_decodes_the_frames_the_index_lists(tmp_path, monkeypatch, lossy_ffmpeg):
    """An mp4 that lost its frames from the middle on and whose index is whole: a copy of its
    stream (imageio-ffmpeg's frame count before 0.5) counts every frame, the decode stops at the
    first frame lost."""
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    import imageio.v2 as imageio

    path = str(tmp_path / "lost.mp4")
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", lossy_ffmpeg("tail"))
    writer = imageio.get_writer(path, **D.MP4_WRITER_KWARGS)
    for frame in _ramps(40):
        writer.append_data(frame)
    writer.close()
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE")
    copy = [imageio_ffmpeg.get_ffmpeg_exe(), "-i", path, "-map", "0:v:0", "-c", "copy"]
    counted = subprocess.run([*copy, "-f", "null", "-"], capture_output=True, text=True).stderr
    assert re.findall(r"frame=\s*(\d+)", counted)[-1] == "40"
    decoded, error = D.mp4_decoded_frames(path)
    assert 0 < decoded < 40
    assert re.fullmatch(r"\[h264 @ 0x[0-9a-f]+\] Invalid NAL unit size \(0 > \d+\)\.", error)
    # an ffmpeg that fails without a word
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", shutil.which("false"))
    assert D.mp4_decoded_frames(path) == (0, "exit status 1")


def test_latents_are_decoded_into_an_mp4(tiny_vae, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("cv2")
    path = tmp_path / "video.mp4"
    assert D.decode_to_mp4(tiny_vae, _latents(3, 2, 3, seed=8), path, chunk=2) == 9
    assert _video(path) == (9, 48, 32, pytest.approx(16.0))


def test_load_latents_checks_the_block_layout(tmp_path):
    np.save(tmp_path / "ok.npy", np.zeros((9, 48, 24, 42), np.float32))
    latents = D.load_latents(tmp_path / "ok.npy")
    assert latents.shape == (1, 9, 48, 24, 42) and latents.dtype == torch.float32
    np.save(tmp_path / "bad.npy", np.zeros((8, 48, 24, 42), np.float32))
    with pytest.raises(ValueError, match="8 latents is not 1 \\+ 4 k"):
        D.load_latents(tmp_path / "bad.npy")
    np.save(tmp_path / "flat.npy", np.zeros((9, 48), np.float32))
    with pytest.raises(ValueError, match="expected latents"):
        D.load_latents(tmp_path / "flat.npy")
