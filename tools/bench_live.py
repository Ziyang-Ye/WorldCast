#!/usr/bin/env python3
"""One live client in a worker's session on a scripted player: input to photon with and without
refresh (``play.refresh``).

The client of ``--index-row`` plays live as the worker of the demo config ``--demo`` serves it
(its ``client.worldcast`` inference settings over ``--config``, its bound, its ``play`` settings):
the worker's pacing, its send queue and, with ``--refresh on``, the refreshes of the block on screen
(``worldcast.demo.worker``). A script of input samples (``--script``, 60 a second, drawn from
``--seed``) plays the player: ``walk`` (the movement keys change every 0.3-1.5 s), ``orbit`` (a
strafe held while the turn changes every 0.3-1.5 s), ``look`` (flicks of the mouse, 10-60 degrees of
yaw and 5-25 of pitch every 0.4-1.5 s, through the page's turn governor) or ``idle``. Each change
or flick is an input event; its input to photon runs from its sample at the worker to the light of
the first frame shown whose controls hold it, with 5 ms assumed for the network and 10 ms for the
display, the frames shown by the pacing's display at ``--fps``
(``worldcast.demo.latency.measure_live``):

    python tools/bench_live.py --config weights/paths.yaml \
        --config examples/data/mirage_r16/config.yaml --index-row 0 \
        --demo configs/demo/worldcast.yaml --script walk --seconds 60 --refresh on \
        --out-dir runs/bench-live/walk-on

Writes ``<out-dir>/bench.json`` (input to photon p50 / p90 / p99, over every event and over those
after the first two measured blocks, frames the send queue dropped, the frames shown per second,
stalls per minute, refreshes per block and what a refresh takes on the host and the GPU, the lead
the next block starts with, the seams: the change between consecutive frames shown at a refresh's
switch, at a block's start and elsewhere, and of each event that turns, how soon the view has
turned half and nine tenths of it: in the controls of the frames shown and in their pixels, and how
far the view turns for the turn the controls hold) and ``<out-dir>/displayed.mp4``, the frames as
shown. ``--mapping``, ``--threshold-deg``, ``--yaw-gain``, ``--pitch-gain``, ``--send-lead-ms``,
``--gather-ms``, ``--margin-frames``, ``--preview-step`` and ``--swap-sent`` (``--no-swap-sent``)
replace the demo config's ``play`` settings of those names (``input_mapping``,
``refresh_threshold_deg``, ...);
``--refresh-devices cuda:0,cuda:1`` gives the client a replica of the generator on each card after
its own (``refresh_devices``). bench.json also has the same measures of the frames the page's
playout shows (``page``: its margin and jitter buffer), the previews (``preview``), the frames sent
again (``swap``), each card's refreshes (``per_card``) and how busy each card was (``gpu_busy``),
and the warm-up's checks of the replicas (``replicas_equal``). ``--settings``, ``--taehv``,
``--decoder``, ``--decoder-device`` and ``--decoder-channels-last`` are those of
``tools/bench_realtime.py`` (over the demo config's serving); ``--no-warm-up`` skips the worker's
warm-up (on the CPU).
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from worldcast.config.loader import add_config_args, parse_overrides, usage_errors
from worldcast.data import RoundIndexRow, load_round_index_row
from worldcast.demo.config import INPUT_MAPPINGS
from worldcast.demo.config import load_config as load_demo_config
from worldcast.demo.latency import SCRIPTS, SETTINGS, measure_live
from worldcast.demo.library import Library, RoundStart, Seat
from worldcast.demo.worldcast_client import WorldCastClient, inference_config


def main(argv: Sequence[str] | None = None) -> int:
    """Measure the live client of ``--index-row`` on a script; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument(
        "--index-row", type=int, required=True, help="the client's row of the round index"
    )
    parser.add_argument(
        "--demo",
        metavar="CONFIG",
        required=True,
        help="the demo's config: the worker's inference, serving and play settings",
    )
    parser.add_argument("--refresh", choices=("on", "off"), default="on", help="play.refresh")
    parser.add_argument("--script", choices=SCRIPTS, default="walk")
    parser.add_argument("--seconds", type=float, default=60.0, help="the script's length")
    parser.add_argument("--seed", type=int, default=0, help="the script's draw")
    parser.add_argument("--fps", type=float, help="frames shown per second (play.fps)")
    parser.add_argument("--mapping", choices=INPUT_MAPPINGS, help="play.input_mapping")
    parser.add_argument("--threshold-deg", type=float, help="play.refresh_threshold_deg")
    parser.add_argument("--yaw-gain", type=float, help="play.yaw_gain")
    parser.add_argument("--pitch-gain", type=float, help="play.pitch_gain")
    parser.add_argument("--send-lead-ms", type=float, help="play.send_lead_ms")
    parser.add_argument("--gather-ms", type=float, help="play.gather_ms")
    parser.add_argument("--margin-frames", type=float, help="play.margin_frames")
    parser.add_argument("--preview-step", type=int, help="play.preview_step")
    parser.add_argument(
        "--swap-sent", action=argparse.BooleanOptionalAction, help="play.swap_sent (or not)"
    )
    parser.add_argument(
        "--refresh-devices",
        nargs="+",
        help="the cards of the refreshes, the client's first, comma- or space-separated"
        " (refresh_devices)",
    )
    parser.add_argument("--settings", choices=tuple(SETTINGS), default="fast")
    parser.add_argument("--taehv", help="taew2_2.pth, the tiny decoder (--decoder taehv)")
    parser.add_argument("--decoder", choices=("taehv", "wan"))
    parser.add_argument("--decoder-device", help="where the decoder runs, e.g. cuda:1")
    parser.add_argument(
        "--decoder-channels-last",
        action="store_true",
        help="the Wan2.2 VAE's decoding convolutions in the channels-last layout (--decoder wan)",
    )
    parser.add_argument(
        "--no-warm-up",
        action="store_true",
        help="skip the warm-up (its first blocks then compile and capture CUDA graphs)",
    )
    parser.add_argument("--out-dir", required=True, help="receives bench.json and displayed.mp4")
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds: positive")
    if args.refresh_devices and args.refresh == "off":
        parser.error("--refresh-devices: the cards of the refreshes, with --refresh on")

    with usage_errors(parser):
        demo = load_demo_config([args.demo])
        replaced = {"refresh": args.refresh == "on"}
        for key, value in (
            ("fps", args.fps),
            ("input_mapping", args.mapping),
            ("refresh_threshold_deg", args.threshold_deg),
            ("yaw_gain", args.yaw_gain),
            ("pitch_gain", args.pitch_gain),
            ("send_lead_ms", args.send_lead_ms),
            ("gather_ms", args.gather_ms),
            ("margin_frames", args.margin_frames),
            ("preview_step", args.preview_step),
            ("swap_sent", args.swap_sent),
        ):
            if value is not None:
                replaced[key] = value
        play = replace(demo.play, **replaced)
        worker = demo.client.worldcast
        attention, fields = SETTINGS[args.settings]
        serving = {**worker.serving, **fields}
        for key, value in (("decoder", args.decoder), ("decoder_device", args.decoder_device)):
            if value is not None:
                serving[key] = value
        if args.decoder_channels_last:
            serving["decoder_channels_last"] = True
        if args.refresh_devices:
            serving["refresh_devices"] = ",".join(args.refresh_devices).split(",")
        config = replace(
            worker,
            configs=list(args.config),
            overrides={
                **worker.overrides,
                "model.attention": attention,
                **parse_overrides(args.overrides),
            },
            taehv_path=args.taehv or worker.taehv_path,
            serving=serving,
        )
        config.serving_options()
        cfg = inference_config(config, demo.scene_state)
        cfg.paths.require("round_index", "media_index")
        row = load_round_index_row(cfg.paths.round_index, args.index_row)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = WorldCastClient(config, demo.scene_state, bench_library(row), refresh=play.refresh)
    if not args.no_warm_up:
        client.warm_up()
    measured = measure_live(
        client,
        row,
        play=play,
        script=args.script,
        seconds=args.seconds,
        seed=args.seed,
        out_dir=out,
    )
    result = dict(
        index_row=args.index_row,
        demo=args.demo,
        refresh=args.refresh,
        script=args.script,
        seconds=args.seconds,
        seed=args.seed,
        display_fps=play.fps,
        mapping=play.mapping,
        gains=play.gains,
        threshold_deg=play.refresh_threshold_deg,
        send_lead_ms=play.send_lead_ms,
        gather_ms=play.gather_ms,
        margin_frames=play.margin_frames,
        preview_step=play.preview_step,
        swap_sent=play.swap_sent,
        settings=args.settings,
        serving=serving,
        replicas_equal=client.client.replicas_equal,
        **measured,
    )
    (out / "bench.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(summary(result)), flush=True)
    return 0


def summary(result: dict) -> dict:
    """The line printed of a bench: input to photon (p50, p90), the control's and the camera's
    response (t50, t90 of their p50s), stalls, of the pacing's display and of the page's, the
    previews, each card's refreshes and busy share."""
    out = {key: result[key] for key in ("fps", "stalls_per_min", "refreshes_per_block")}
    for name, measured in (("", result), ("page_", result["page"])):
        if not measured:
            continue
        out[f"{name}i2p"] = [measured["i2p_s"]["p50"], measured["i2p_s"]["p90"]]
        for kind in ("control", "camera"):
            response = measured[f"{kind}_response"]
            out[f"{name}{kind}"] = [response[level]["p50"] for level in ("t50_s", "t90_s")]
    out["page_holds_per_min"] = result["page"].get("holds_per_min")
    out["view_gain"] = result["view_gain"]["turn"]["p50"]
    if result["preview"]:
        preview = result["preview"]
        out["preview"] = dict(
            screen_share=preview["screen_share"], psnr_db=preview["vs_final"]["median_psnr_db"]
        )
    out["per_card"] = {card: c["refreshes"] for card, c in result["per_card"].items()}
    out["gpu_busy"] = result["gpu_busy"]
    return out


def bench_library(row: RoundIndexRow) -> Library:
    """A library of the row's round start with the row's player as its one seat, which the
    warm-up replays."""
    seat = Seat(slot=row.player_slot, team="T", spawn=(0.0,) * 5)
    start = RoundStart(
        "bench",
        row.map_name,
        "bench",
        row.start_frame,
        (seat,),
        match_id=row.match_id,
        round=row.round,
    )
    return Library({start.id: start})


if __name__ == "__main__":
    sys.exit(main())
