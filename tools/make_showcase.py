#!/usr/bin/env python3
"""Make the showcase video of a round with GT player states: its views, a box at every other
player's recorded position in each view, and a map of the ten players.

The views are those of a finished session (``--session``: each client's ``video.mp4`` beside its
``client.json``, as ``examples/run.sh`` leaves them) or of a case's reference run (``--expected``:
``examples/expected/<case>.mp4``, tiled as ``examples/grid.sh`` tiles a session). The recordings
that ``--config`` names (the case's config) give every player's state per video frame and each
client's GT visibility labels. In each view, labelled with its player, a box stands at every other
living player's recorded position in front of the camera and within the view's width, projected
into the client's recorded camera as the player state field projects players: its foot at the
player's feet, as tall as the body, bold where the field's gate passes it (the client's label has
the player in view). A view that shows the scope's mask (black around a lit disc) is drawn without
boxes and with the note "scoped: no boxes" beside its label. Beside the views, a map of the round
from above shows the ten players with their trails over the faint paths of the whole round, and
each client's field of view, between the bands of its time and its legend. Runs on the CPU; the
tick tables need pyarrow and the drawing needs Pillow (``pip install -e ".[showcase]"``).

Examples::

    python tools/make_showcase.py --config examples/data/mirage_r16/config.yaml \\
        --session runs/examples/mirage_r16 --out runs/examples/mirage_r16/showcase.mp4
    python tools/make_showcase.py --config examples/data/mirage_r16/config.yaml \\
        --expected examples/expected/mirage_r16.mp4 --out runs/showcase/mirage_r16.mp4
"""

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import islice
from pathlib import Path

import numpy as np

from worldcast.config.inference import InferenceConfig, load_config
from worldcast.config.loader import add_config_args, parse_overrides, usage_errors
from worldcast.data import read_round_index
from worldcast.data.game import HFOV_DEGREES
from worldcast.data.latents import FPS, FRAME_SIZE
from worldcast.data.recordings import ALIVE_INDEX, RoundIndexRow
from worldcast.engine.inference.decode import write_mp4
from worldcast.engine.inference.loading import load_window
from worldcast.engine.inference.session import round_name, rounds_of
from worldcast.player_state.projection import player_boxes
from worldcast.player_state.states import pack_substeps

#: Views per row of the grid, as ``examples/grid.sh`` lays them out (four as 2 x 2).
GRID_COLUMNS = 5
#: Outline of a box, px: where the field's gate passes the player (by its GT label), elsewhere.
BOX_WIDTH_IN_VIEW, BOX_WIDTH_HIDDEN = 3, 1
#: The players' colours, by player slot, each standing out on a dark scene (a darker P9 is lost in
#: a dark tunnel). A black edge (:data:`EDGE`) sets them off a bright one.
PLAYER_COLORS = (
    "#E67F27",
    "#3D94E8",
    "#3FB576",
    "#BF5AF2",
    "#FF375F",
    "#64D2FF",
    "#FFD60A",
    "#AC8E68",
    "#8C8AFF",
    "#30D158",
)
#: Engine team ids.
TEAMS = {2: "T", 3: "CT"}
#: The black edge of a box (a pixel each side of its outline) and of a name (``EDGE_WIDTH`` px).
EDGE, EDGE_WIDTH = (0, 0, 0), 2
#: Height of the text, px; a chip's height (a line of text on black) and its gap to the edge.
FONT_SIZE = 14
CHIP_HEIGHT, CHIP_GAP = FONT_SIZE + 8, 6
#: The map's legend; the note beside a view's label while it shows the scope (and no boxes), and
#: the colour of both.
LEGEND, SCOPED_NOTE, NOTE_COLOR = (
    "box: recorded position, bold: in view (GT label)",
    "scoped: no boxes",
    "#C8C8C8",
)
#: The scope's mask in a view: beyond ``SCOPE_RING`` px of its centre at most 1% of the pixels
#: brighter than ``SCOPE_BLACK`` (grey, 0-255), and its disc, within ``SCOPE_DISC`` px, brighter
#: than ``SCOPE_LIT`` on average.
SCOPE_RING, SCOPE_DISC = 190, 140
SCOPE_BLACK, SCOPE_LIT = 16, 25
#: x264 quality of the showcase, that of ``examples/grid.sh`` (the views were decoded at the
#: paper's).
SHOWCASE_CRF = 23
#: The showcase's encoding, as ``examples/grid.sh`` encodes its grid: x264 at :data:`SHOWCASE_CRF`,
#: 16 fps (the imageio writer's arguments, for ``write_mp4``). The index follows the frames (no
#: ``-movflags +faststart``), so ffmpeg writes the file in one pass and never reads it back.
SHOWCASE_WRITER = dict(
    fps=FPS,
    codec="libx264",
    quality=None,
    macro_block_size=1,
    ffmpeg_log_level="error",
    output_params=["-preset", "medium", "-crf", str(SHOWCASE_CRF)],
)
#: The map: the margin around the players' paths, u; the colours of its ground and of the whole
#: round's paths.
MAP_MARGIN_U = 150.0
MAP_GROUND, MAP_PATHS = (11, 11, 13), (44, 48, 56)
#: The map's marks, in shares of its side: a path's width, a player's radius, a client's radius
#: and the length of a client's field of view; the field of view's opacity (0-255).
PATH_WIDTH, PLAYER_RADIUS, CLIENT_RADIUS, VIEW_LENGTH = 0.008, 0.012, 0.018, 0.12
VIEW_OPACITY = 72
#: A player's trail on the map: its path over the last seconds.
TRAIL_SECONDS = 3


@dataclass(frozen=True)
class RecordedRound:
    """What the showcase draws of a round's recordings, at each of its ``T`` video frames.

    Attributes:
        media_ids (tuple[str, ...]): the clients, in the order of their views (player slots).
        clients (tuple[int, ...]): their player slots.
        states (np.ndarray): ``[P, T, 6]`` float32 every player's recorded state.
        team_ids (np.ndarray): ``[P]`` int64 every player's engine team (0 without a recording).
        boxes (np.ndarray): ``[C, T, P, 4]`` float32 every player's box in each client's view
            (:func:`~worldcast.player_state.projection.player_boxes`).
        shown (np.ndarray): ``[C, T, P]`` bool, the box can be drawn: another living player in
            front of the camera.
        in_view (np.ndarray): ``[C, T, P]`` bool, the box is bold: the field's gate passes the
            player, by the client's GT visibility label of the video frame.
    """

    media_ids: tuple[str, ...]
    clients: tuple[int, ...]
    states: np.ndarray
    team_ids: np.ndarray
    boxes: np.ndarray
    shown: np.ndarray
    in_view: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.states.shape[1])


def load_round(cfg: InferenceConfig, rows: Sequence[RoundIndexRow]) -> RecordedRound:
    """The recordings of the clients ``rows`` of one round, as long as every client's window."""
    windows = sorted((load_window(cfg, row) for row in rows), key=lambda w: w.media.player_slot)
    frames = min(window.spec.video_frames for window in windows)
    item = windows[0].item
    states = item.player_states[:, :frames]
    controls = pack_substeps(
        item.player_control_substeps[:, :frames], item.player_control_substep_valid[:, :frames]
    )
    boxes, shown, in_view = [], [], []
    for window in windows:
        labels = window.item.client_visibility[:, :frames] > 0.5
        visible = labels & window.item.client_visibility_valid[:, :frames]
        view = player_boxes(
            states.transpose(0, 1), controls.transpose(0, 1), visible.T, window.media.player_slot
        )
        boxes.append(view.boxes.numpy())
        shown.append(view.shown.numpy())
        in_view.append(view.in_view.numpy())
    return RecordedRound(
        media_ids=tuple(window.media.media_id for window in windows),
        clients=tuple(window.media.player_slot for window in windows),
        states=states.numpy(),
        team_ids=item.player_team_ids.numpy(),
        boxes=np.stack(boxes),
        shown=np.stack(shown),
        in_view=np.stack(in_view),
    )


def shows_scope(view: np.ndarray) -> bool:
    """Whether a view, ``[384, 672, 3]`` uint8, shows the scope's mask (:data:`SCOPE_RING`):
    black around a lit disc, read from the view's pixels and not from the client's scope label."""
    ring, disc = _scope_regions()
    grey = view.mean(axis=-1)
    return bool((grey[ring] > SCOPE_BLACK).mean() <= 0.01 and grey[disc].mean() > SCOPE_LIT)


@cache
def _scope_regions() -> tuple[np.ndarray, np.ndarray]:
    frame_h, frame_w = FRAME_SIZE
    y, x = np.mgrid[:frame_h, :frame_w]
    radius = np.hypot(y - (frame_h - 1) / 2.0, x - (frame_w - 1) / 2.0)
    return radius > SCOPE_RING, radius < SCOPE_DISC


def grid_layout(views: int) -> tuple[int, int]:
    """``(columns, rows)`` of ``views`` views as ``examples/grid.sh`` tiles them."""
    columns = 2 if views == 4 else min(views, GRID_COLUMNS)
    return columns, -(-views // columns)


class Showcase:
    """The showcase's frames of a recorded round: its views in the layout of ``examples/grid.sh``,
    the map beside them, a square as tall as the grid.

    Args:
        recorded (RecordedRound): the round.
    """

    def __init__(self, recorded: RecordedRound) -> None:
        from PIL import Image, ImageDraw, ImageFont

        self.recorded = recorded
        self.teams = [TEAMS.get(int(team), "") for team in recorded.team_ids]
        self.colors = list(PLAYER_COLORS)
        self.font = ImageFont.load_default(size=FONT_SIZE)
        self.columns, self.rows = grid_layout(len(recorded.clients))
        frame_h, frame_w = FRAME_SIZE
        self.side = self.rows * frame_h
        self.size = (self.columns * frame_w + self.side, self.rows * frame_h)
        self.positions = recorded.states[..., :2]
        self.alive = recorded.states[..., ALIVE_INDEX] > 0.5
        low = self.positions[self.alive].min(axis=0) - MAP_MARGIN_U
        high = self.positions[self.alive].max(axis=0) + MAP_MARGIN_U
        # the players between the bands of the time and the legend, a dot's radius off each edge
        rim = CLIENT_RADIUS * self.side
        top = CHIP_GAP + CHIP_HEIGHT + rim
        bottom = self.side - CHIP_GAP - CHIP_HEIGHT - rim
        self.centre = (low + high) / 2.0
        self.middle = (self.side / 2.0, (top + bottom) / 2.0)
        self.scale = min(
            (self.side - 2.0 * rim) / (high - low)[0], (bottom - top) / (high - low)[1]
        )
        self.map = Image.new("RGB", (self.side, self.side), MAP_GROUND)
        draw = ImageDraw.Draw(self.map)
        for slot in range(len(self.positions)):  # the whole round's paths outline the map
            self._path(draw, slot, 0, recorded.frame_count, MAP_PATHS)

    def on_map(self, x: float, y: float) -> tuple[float, float]:
        """The map's pixel of the engine position ``x, y`` (north up)."""
        return (
            self.middle[0] + (x - self.centre[0]) * self.scale,
            self.middle[1] - (y - self.centre[1]) * self.scale,
        )

    def frames(self, views: Iterator[Sequence[np.ndarray]]) -> Iterator[np.ndarray]:
        """The showcase's frames, ``[H, W, 3]`` uint8, one per frame of the views (each a
        ``[384, 672, 3]`` uint8 frame per client), at most ``T``."""
        from PIL import Image

        frame_h, frame_w = FRAME_SIZE
        for t, tiles in enumerate(islice(views, self.recorded.frame_count)):
            canvas = Image.new("RGB", self.size)
            for c, tile in enumerate(tiles):
                column, row = c % self.columns, c // self.columns
                canvas.paste(self.view(tile, c, t), (column * frame_w, row * frame_h))
            canvas.paste(self.map_at(t), (self.columns * frame_w, 0))
            yield np.asarray(canvas)

    def view(self, tile: np.ndarray, c: int, t: int):
        """Client ``c``'s view at frame ``t``, labelled, with the other players' boxes (none while
        it shows the scope: :func:`shows_scope`), each named above it where its name covers
        neither a label nor a nearer player's name or box."""
        from PIL import Image, ImageDraw

        scoped = shows_scope(tile)
        image = Image.fromarray(np.array(tile))
        draw = ImageDraw.Draw(image)
        frame_h, frame_w = FRAME_SIZE
        recorded = self.recorded
        slot = recorded.clients[c]
        label = f"P{slot + 1} {self.teams[slot]}".strip()
        chips = [self._chip_at(draw, (CHIP_GAP, CHIP_GAP), label)]
        if scoped:
            right = chips[0][0][2]
            chips.append(self._chip_at(draw, (right + CHIP_GAP, CHIP_GAP), SCOPED_NOTE))
        x0, y0, x1, y1 = recorded.boxes[c, t].astype(float).T
        # a box of a player within the view's width, not the side of one beside it
        drawn = recorded.shown[c, t] & (x0 + x1 >= 0) & (x0 + x1 <= 2 * frame_w)
        drawn &= (y1 >= 0) & (y0 <= frame_h)
        nearest = sorted(np.flatnonzero(drawn), key=lambda other: y0[other] - y1[other])  # tallest
        if scoped:
            nearest = []
        for other in reversed(nearest):
            width = BOX_WIDTH_IN_VIEW if recorded.in_view[c, t, other] else BOX_WIDTH_HIDDEN
            box = [x0[other], y0[other], x1[other], y1[other]]
            edge = [box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1]  # a pixel each side
            draw.rectangle(edge, outline=EDGE, width=width + 2)
            draw.rectangle(box, outline=self.colors[other], width=width)
        taken = [rectangle for rectangle, _ in chips]  # the labels, then the names and the boxes
        for other in nearest:
            name = f"P{other + 1}"
            text = draw.textlength(name, font=self.font)
            left = (x0[other] + x1[other] - text) / 2.0
            left = min(max(left, EDGE_WIDTH), frame_w - text - EDGE_WIDTH)
            spot = (left, max(y0[other] - FONT_SIZE - 4, 0.0))
            self._name(draw, (frame_w, frame_h), [spot], name, self.colors[other], taken)
            taken.append((x0[other], y0[other], x1[other], y1[other]))
        for (rectangle, text), color in zip(chips, [self.colors[slot], NOTE_COLOR]):
            self._chip(draw, rectangle, text, color)
        return image

    def map_at(self, t: int):
        """The map at frame ``t``: every living player and its trail (:data:`TRAIL_SECONDS`),
        each client's field of view."""
        from PIL import ImageColor, ImageDraw

        image = self.map.copy()
        draw = ImageDraw.Draw(image, "RGBA")
        states = self.recorded.states[:, t]
        living = np.flatnonzero(states[:, ALIVE_INDEX] > 0.5)
        for slot in living:
            self._path(draw, slot, t + 1 - round(TRAIL_SECONDS * FPS), t + 1, self.colors[slot])
        reach = VIEW_LENGTH * self.side
        for slot in self.recorded.clients:
            x, y, _, yaw = states[slot, :4]
            if slot in living:
                px, py = self.on_map(x, y)
                fill = ImageColor.getrgb(self.colors[slot]) + (VIEW_OPACITY,)
                start, end = -yaw - HFOV_DEGREES / 2.0, -yaw + HFOV_DEGREES / 2.0
                draw.pieslice([px - reach, py - reach, px + reach, py + reach], start, end, fill)
        radii = {
            slot: (CLIENT_RADIUS if slot in self.recorded.clients else PLAYER_RADIUS) * self.side
            for slot in living
        }
        chips = [
            self._chip_at(draw, (CHIP_GAP, CHIP_GAP), f"GT player states  {t / FPS:5.2f} s"),
            self._chip_at(draw, (CHIP_GAP, self.side - CHIP_GAP - CHIP_HEIGHT), LEGEND),
        ]
        taken = [rectangle for rectangle, _ in chips]  # the labels, the dots, then the names
        for slot, radius in radii.items():
            px, py = self.on_map(*states[slot, :2])
            outline = "white" if slot in self.recorded.clients else None
            taken.append((px - radius, py - radius, px + radius, py + radius))
            draw.ellipse(taken[-1], fill=self.colors[slot], outline=outline, width=2)
        for slot in sorted(radii, key=lambda slot: slot not in self.recorded.clients):
            px, py = self.on_map(*states[slot, :2])
            name, radius = f"P{slot + 1}", radii[slot]
            text = draw.textlength(name, font=self.font)
            top, left = py - FONT_SIZE / 2.0 - 1, px - text / 2.0
            spots = [  # right of the dot, left of it, above it, below it
                (px + radius + 3, top),
                (px - radius - 3 - text, top),
                (left, py - radius - FONT_SIZE - 4),
                (left, py + radius),
            ]
            self._name(draw, (self.side, self.side), spots, name, "white", taken)
        for (rectangle, text), color in zip(chips, ["white", NOTE_COLOR]):
            self._chip(draw, rectangle, text, color)
        return image

    def _path(self, draw, slot: int, start: int, end: int, color) -> None:
        """Player ``slot``'s path over the frames ``start .. end - 1`` where it lives."""
        start = max(start, 0)
        alive = self.alive[slot, start:end]
        path = [self.on_map(x, y) for x, y in self.positions[slot, start:end][alive]]
        if len(path) > 1:
            draw.line(path, fill=color, width=round(PATH_WIDTH * self.side), joint="curve")

    def _name(self, draw, size: tuple[int, int], spots, name: str, color, taken: list) -> None:
        """``name`` with a black edge at the first of ``spots`` (top left corners) where it is
        inside the image of ``size`` and clear of the boxes ``taken`` (which its box joins);
        nowhere if there is no such spot."""
        for spot in spots:
            x0, y0, x1, y1 = draw.textbbox(spot, name, font=self.font, stroke_width=EDGE_WIDTH)
            inside = x0 >= 0 and y0 >= 0 and x1 <= size[0] and y1 <= size[1]
            if inside and all(x1 <= o[0] or o[2] <= x0 or y1 <= o[1] or o[3] <= y0 for o in taken):
                taken.append((x0, y0, x1, y1))
                draw.text(spot, name, color, self.font, stroke_width=EDGE_WIDTH, stroke_fill=EDGE)
                return

    def _chip_at(self, draw, at: tuple[float, float], text: str) -> tuple[tuple, str]:
        """The rectangle of a chip of ``text`` whose top left is ``at``, :data:`CHIP_HEIGHT` tall,
        and the text."""
        x, y = at
        return (x, y, x + draw.textlength(text, font=self.font) + 12, y + CHIP_HEIGHT), text

    def _chip(self, draw, rectangle: tuple, text: str, color) -> None:
        """``text`` on the black chip ``rectangle`` (:meth:`_chip_at`)."""
        draw.rectangle(rectangle, fill=(0, 0, 0))
        draw.text((rectangle[0] + 6, rectangle[1] + 3), text, fill=color, font=self.font)


def video_frames(path: str | Path) -> Iterator[np.ndarray]:
    """The frames of an mp4, ``[H, W, 3]`` uint8."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        width, height = next(reader)["size"]
        for raw in reader:
            yield np.frombuffer(raw, np.uint8).reshape(height, width, 3)
    finally:
        reader.close()


def check_size(path: str | Path, size: tuple[int, int]) -> None:
    """Raise ``ValueError`` unless the mp4 ``path`` is ``size = (width, height)``; a missing file:
    ``FileNotFoundError``."""
    import imageio_ffmpeg

    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not found")
    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        found = tuple(next(reader)["size"])
    finally:
        reader.close()
    if found != size:
        raise ValueError(f"{path} is {found[0]} x {found[1]}, not {size[0]} x {size[1]}")


def session_clients(
    session: Path, rows: Sequence[RoundIndexRow]
) -> list[tuple[RoundIndexRow, Path]]:
    """The clients of a finished session of one round, by their ``client.json``: each one's row of
    the round index ``rows`` and its ``video.mp4``.

    Raises:
        ValueError: no client, a client of another round index, clients of several rounds, or a
            client whose latents are not decoded.
    """
    clients, rounds = [], {}
    for meta in sorted(session.rglob("client.json")):
        summary = json.loads(meta.read_text())
        index, media_id = int(summary["index_row"]), summary["media_id"]
        if not 0 <= index < len(rows) or rows[index].media_id != media_id:
            raise ValueError(f"{meta}: {media_id} is not row {index} of the config's round index")
        rounds.setdefault(round_name(rows[index]), meta.parent.parent)
        clients.append((rows[index], meta.with_name("video.mp4")))
    if not clients:
        raise ValueError(f"no client.json under {session}: give a session's output directory")
    if len(rounds) > 1:
        raise ValueError(
            f"{session} holds {len(rounds)} rounds: give one round's folder, e.g."
            f" {next(iter(rounds.values()))}"
        )
    for _, video in clients:
        if not video.is_file():
            raise ValueError(f"{video} not found: decode the session first (tools/decode.py)")
    return clients


def session_views(videos: Sequence[Path]) -> Iterator[list[np.ndarray]]:
    """The clients' frames, one per video at each step, until the shortest video ends."""
    readers = [video_frames(video) for video in videos]
    try:
        for tiles in zip(*readers):
            yield list(tiles)
    finally:
        for reader in readers:
            reader.close()


def grid_views(path: str | Path, views: int) -> Iterator[list[np.ndarray]]:
    """The views of a grid video (``examples/grid.sh``) of ``views`` views, frame by frame."""
    columns, _ = grid_layout(views)
    frame_h, frame_w = FRAME_SIZE
    for frame in video_frames(path):
        yield [
            frame[
                (c // columns) * frame_h : (c // columns + 1) * frame_h,
                (c % columns) * frame_w : (c % columns + 1) * frame_w,
            ]
            for c in range(views)
        ]


def main(argv: Sequence[str] | None = None) -> int:
    """Write the showcase of ``--session`` or ``--expected`` to ``--out``; returns the exit
    status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    views = parser.add_mutually_exclusive_group(required=True)
    views.add_argument("--session", help="a finished session's output directory")
    views.add_argument("--expected", help="a case's reference run, examples/expected/<case>.mp4")
    parser.add_argument("--out", required=True, help="the showcase's mp4")
    args = parser.parse_args(argv)
    try:
        import PIL  # noqa: F401
    except ModuleNotFoundError:
        parser.error('the showcase is drawn with Pillow: pip install -e ".[showcase]"')

    frame_h, frame_w = FRAME_SIZE
    with usage_errors(parser):
        cfg = load_config(args.config, parse_overrides(args.overrides))
        cfg.paths.require("round_index")
        rows = read_round_index(cfg.paths.round_index)
        if args.session:
            found = session_clients(Path(args.session), rows)
            clients = [row for row, _ in found]
            videos = {row.media_id: video for row, video in found}
        else:
            rounds = rounds_of(rows)
            if len(rounds) != 1:
                raise ValueError(f"{cfg.paths.round_index} is not one round: give a case's config")
            clients = [rows[i] for i in rounds[0]]
        try:
            recorded = load_round(cfg, clients)
        except ModuleNotFoundError as error:  # pyarrow, the tick tables' reader
            raise ValueError(
                f"{error}: install the package with its dependencies, pip install -e ."
            ) from error
        columns, grid_rows = grid_layout(len(clients))
        if args.session:
            paths = [videos[media_id] for media_id in recorded.media_ids]
            for path in paths:
                check_size(path, (frame_w, frame_h))
            frames = session_views(paths)
        else:
            check_size(args.expected, (columns * frame_w, grid_rows * frame_h))
            frames = grid_views(args.expected, len(clients))
    showcase = Showcase(recorded)
    written = write_mp4(showcase.frames(frames), args.out, writer=SHOWCASE_WRITER)
    print(
        f"{args.out} decodes to {written} frames: {len(clients)} views ({columns}x{grid_rows}) and"
        " the map"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
