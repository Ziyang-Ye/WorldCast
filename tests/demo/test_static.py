"""The page's constants come from the server: the button order of an input sample in the worker's
``welcome`` (``test_worker.py``), the players' colours, the name length, the generated frame's
pixel size and the camera's field of view and pitch range in ``GET /api/lobby``; the scripts keep
no copy, and bind keys only to buttons the model has."""

import asyncio
import math
import re

from tests.demo.support import demo
from worldcast.data.camera import half_angle_tangents
from worldcast.data.controls import CAMERA_MAX_DEGREES, CONTROL_BUTTONS
from worldcast.data.game import PITCH_MAX_DEGREES, PITCH_MIN_DEGREES
from worldcast.data.latents import FPS, FRAME_SIZE
from worldcast.demo.clients import LivePlayerState
from worldcast.demo.coordinator import PLAYER_NAME_LENGTH, STATIC
from worldcast.demo.library import PLAYER_COLORS

SCRIPTS = {path.name: path.read_text(encoding="utf-8") for path in sorted(STATIC.glob("*.js"))}
CSS = (STATIC / "app.css").read_text(encoding="utf-8")


async def lobby() -> dict:
    async with demo(1) as (local, http):
        return await (await http.get(f"{local.url}/api/lobby")).json()


def test_the_lobby_carries_the_pages_constants():
    answer = asyncio.run(lobby())
    tans = answer["page"].pop("camera_tans")
    assert answer["page"] == {
        "player_colors": list(PLAYER_COLORS),
        "player_name_length": PLAYER_NAME_LENGTH,
        "frame_pixels": [FRAME_SIZE[1], FRAME_SIZE[0]],
        "pitch_range_degrees": [PITCH_MIN_DEGREES, PITCH_MAX_DEGREES],
    }
    # the camera of the rays and of retrieval, the generator's (worldcast.data.camera)
    assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(tans, half_angle_tangents()))
    assert answer["page"]["frame_pixels"] == [672, 384]
    assert (len(PLAYER_COLORS), PLAYER_NAME_LENGTH) == (10, 24)
    assert [start["id"] for start in answer["round_starts"]] == ["arena"]


def test_the_scripts_keep_no_copy_of_the_constants():
    text = "\n".join(SCRIPTS.values())
    assert PLAYER_COLORS[:3] == ("#E67F27", "#3D94E8", "#3FB576")  # the project page's clients
    assert not [color for color in PLAYER_COLORS[:3] if color.lower() in text.lower()]
    assert 'maxlength="24"' not in text
    # the bit of a button comes from the welcome message's order, never from a list in the page
    assert "m.buttons.map((b, i) => [b, 1 << i])" in SCRIPTS["input.js"]
    assert "'forward', 'back'" not in text


async def script_headers() -> tuple[int, str | None]:
    async with demo(1) as (local, http):
        response = await http.get(f"{local.url}/static/stream.js")
        return response.status, response.headers.get("Cache-Control")


def test_the_scripts_are_revalidated_on_every_load():
    """A browser never runs a page older than the workers it talks to (``ack``, below)."""
    assert asyncio.run(script_headers()) == (200, "no-cache")


def test_the_page_reads_no_user_agent_or_platform():
    """The page is the same on every machine: it asks the browser for features, never for who it
    is (a gamepad's id names the device, not the browser, and ``pad.js`` reads its vendor and
    product)."""
    names = ("userAgent", "navigator.platform", "navigator.vendor")
    for name, text in SCRIPTS.items():
        assert not any(n in text for n in names), name


def test_the_page_acknowledges_each_frame_as_it_arrives():
    """The worker measures the browser's link by the acknowledgements and keeps the frames in
    flight within ``worker.max_inflight_ms`` of it (``test_worker.py``): the page acknowledges
    each one on arrival, before decoding it."""
    stream = SCRIPTS["stream.js"]
    arrival = stream.index("  binary(data) {")
    ack = stream.index("this.send({ t: 'ack', r: header.k });")
    assert arrival < ack < stream.index("createImageBitmap", arrival)


def test_the_page_says_when_it_is_hidden_and_counts_the_session_in_hours():
    """A hidden page still acknowledges every frame, so the worker is told when it is hidden and
    releases a seat whose page stays hidden (``test_worker.py``); a session has no end, so its
    clock reads hours, minutes and seconds."""
    play, common = SCRIPTS["play.js"], SCRIPTS["common.js"]
    assert "const sendVisibility = () => stream.send({ t: 'vis', h: document.hidden });" in play
    assert "document.addEventListener('visibilitychange', sendVisibility);" in play
    assert "document.removeEventListener('visibilitychange', sendVisibility);" in play
    welcome = play.index("if (m.t === 'welcome') {")
    assert welcome < play.index("      sendVisibility();", welcome) < play.index("m.t === 'ended'")
    assert 'id="clock">0:00:00</span>' in play
    assert (
        "return `${Math.floor(s / 3600)}:${two(Math.floor(s / 60) % 60)}:${two(s % 60)}`;" in common
    )


def test_the_playout_follows_the_rate_the_blocks_arrive_at():
    """The page shows each frame for a period chosen from the measured block period, within 0.8-1.25
    of its frame period, and holds a frame missing when due instead of rebuffering. Its due times
    run from frame to frame, so it catches up on any display; one late block moves the measured
    period little, a pause of the stream starts the measurement over, and the period never
    follows blocks faster than the worker paces them (docs/demo.md, "Latency")."""
    stream = SCRIPTS["stream.js"]
    playout = stream[stream.index("class Playout {") : stream.index("class FrameStream {")]
    assert "const STRETCH = [0.8, 1.25];" in stream
    # the margin is the worker's play.margin_frames, from its welcome (test_config.py)
    assert "const margin = Math.max(this.margin * base, 2 * this.jitter);" in playout
    measure = playout[playout.index("  measure(gap) {") : playout.index("  take(now) {")]
    assert "if (gap > PAUSE_BLOCKS * this.blockMs) {" in measure
    assert "const deviation = Math.min(Math.max(gap - this.blockMs, -bound), bound);" in measure
    assert "this.blockMs = Math.max(this.paced, this.blockMs + share * deviation);" in measure
    take = playout[playout.index("  take(now) {") : playout.index("  period(due) {")]
    assert "if (!this.holding) this.holds.push(now);" in take
    assert "this.playing = false" not in take  # playback never stops to refill a buffer
    assert "this.next = due + this.period(due);" in take  # not now + period: a frame clock
    assert "const frames = this.queue.length + i - this.newest;" in playout  # dropped frames
    assert "return measured ? Math.min(period, STRETCH[1] * base) : base;" in playout
    assert (
        "new Playout(1000 / message.fps, message.frames_per_block, buffer, message.backlog_frames,"
        "\n        message.margin_frames)" in stream
    )


def test_the_playout_drops_the_frames_before_a_block_beyond_the_backlog():
    """A display further behind than its period can make up (a main thread busy at the start, a
    page that was hidden, a block that came early) drops its oldest frames when a block's first
    frame (the header's ``f``) arrives, those before it beyond what it shows in the backlog's frame
    periods: the backlog's frames while it keeps up with the frame period, whatever the noise of
    its measured interval, fewer on a slower display. The page keeps no number of its own for the
    backlog, the worker's welcome gives it (``test_worker.py``), and the frames dropped give up
    their bitmaps. The interval is not measured across a pause of the page. This pins the text;
    what the page shows then runs in headless Chrome, ``test_playout.py``."""
    stream = SCRIPTS["stream.js"]
    playout = stream[stream.index("class Playout {") : stream.index("class FrameStream {")]
    # the numbers the rule rests on: a pause is two block periods; the display's interval starts at
    # the frame period (a display that keeps up) and follows each shown frame by a fifth
    assert "const PAUSE_BLOCKS = 2;" in stream
    assert "const SHOWN_FOLLOW = 0.2;" in stream
    assert "this.every = period;" in playout
    add = playout[playout.index("  add(frame) {") : playout.index("  measure(gap) {")]
    assert "if (!frame.header.f) return [];" in add
    assert (
        "const keeps = this.every < STRETCH[1] * period ? this.backlog"
        " : Math.floor((this.backlog * period) / this.every);" in add
    )
    assert "return this.queue.splice(0, Math.max(0, i - keeps));" in add
    take = playout[playout.index("  take(now) {") : playout.index("  period(due) {")]
    assert "gap <= PAUSE_BLOCKS * this.blockMs" in take
    assert "this.every += SHOWN_FOLLOW * (gap - this.every);" in take
    assert (
        "for (const dropped of this.playout.add({ header, bitmap, recv })) dropped.bitmap.close();"
        in stream
    )
    assert not re.search(r"backlog\s*[=:]\s*\d", "\n".join(SCRIPTS.values()))


def test_a_frame_received_again_takes_the_place_of_the_one_waiting_if_it_is_newer():
    """A newer version of a frame received (the worker's ``play.swap_sent``: the header's ``v``,
    ``pv``, and ``r`` for a frame sent again) takes the place of the frame waiting, whose bitmap is
    given up, or is given up itself: when it is not newer, when none waits (a frame sent again that
    the page never got), and any frame at or before the frame on screen; it measures no block
    period and drops no backlog. The frames go to the playout in the order they arrived, whatever
    order their decodes finish in (``InOrder``). What the page and the bench's model of it do with
    them is ``tests/demo/test_playout.py``'s."""
    stream = SCRIPTS["stream.js"]
    playout = stream[stream.index("class Playout {") : stream.index("class FrameStream {")]
    add = playout[playout.index("  add(frame) {") : playout.index("  measure(gap) {")]
    again = add[: add.index("const before = this.arrived.get(k - this.perBlock);")]
    assert "if (k <= this.shownK) return [frame];" in again
    assert "if (this.arrived.has(k) || frame.header.r) {" in again
    assert "if (i < 0 || rank(this.queue[i].header) >= rank(frame.header)) return [frame];" in again
    assert "this.shownK = frame.header.k;" in playout
    assert "this.decoded.push(decoding, (bitmap) => {" in stream


def test_a_preview_frame_goes_into_no_panorama():
    """A refresh's preview (``pv``), a coarse estimate of its frame, is shown but kept out of the
    instant turn's panorama, as a frame whose viewmodel may be anywhere is."""
    stream = SCRIPTS["stream.js"]
    pose = stream[stream.index("  pose(frame) {") : stream.index("  look(frame) {")]
    assert "part: pv ? 'none' : this.motion.part(frame.header)" in pose
    assert "if (pose.part === 'none') return;" in SCRIPTS["turn.js"]


def test_the_page_binds_keys_to_the_models_buttons():
    play = SCRIPTS["play.js"]
    bound = set()
    for name in ("KEYS", "MOUSE", "CAPS", "CAPS_ROW"):
        literal = re.search(rf"const {name} = (.*?);\n", play, flags=re.S).group(1)
        bound |= set(re.findall(r"'([a-z_0-9]+)'", literal))
    assert bound == set(CONTROL_BUTTONS) and len(CONTROL_BUTTONS) == 11


def test_the_page_shows_the_rows_the_field_read():
    """The HUD shows the room time of the own row (``s``); how the own row is known (``st``) is the
    latency breakdown's, not the bar's. ``M`` asks the worker for the field's markers (``m``,
    ``test_worker.py``) and draws them, dimmed where the rows of the header (``p``) say the field
    does not draw the player. The page reads a row at the places ``LivePlayerState.wire`` puts
    them, and has no map of the players: the rows reach the page for its markers and the clock,
    not to be drawn from above."""
    play = SCRIPTS["play.js"]
    row = LivePlayerState(slot=3, t=1.5, x=1, y=2, z=3, yaw=4, pitch=5, drawn=False).wire()
    assert (row[0], row[11]) == (3, 0) and len(row) == 12
    assert "const SLOT = 0;\nconst DRAWN = 11;\n" in play
    assert "$('#clock').textContent = clock(h.s[1]);" in play
    assert "state.position = h.st;" in play and 'id="status"' not in play
    assert "meter.panel(rtt, state.framesPerBlock, state.position)" in play
    assert "state.shown = { h, drawn: new Map(h.p.map((row) => [row[SLOT], row[DRAWN]])) };" in play
    assert "ctx.globalAlpha = drawn.get(slot) === 0 ? UNDRAWN_ALPHA : 1;" in play
    assert "if (down && !e.repeat && e.code === 'KeyM') toggleMarkers();" in play
    assert "stream.send({ t: 'markers', on: state.markers });" in play
    assert "for (const [slot, u, v, depth] of h.m || []) {" in play
    assert not (STATIC / "minimap.js").exists()
    page = "\n".join([*SCRIPTS.values(), (STATIC / "app.css").read_text(encoding="utf-8")])
    assert not re.search(r"minimap|radar|field_sources|Minimap", page)


def test_the_play_view_shows_the_frame_at_its_own_pixel_size():
    """The canvas holds the frame's own pixels, whatever the screen: a frame stretched into a
    larger store (the device pixel ratio, the window's size) is only blurrier, and the browser
    scales the canvas to its box. The stage is the frame's size from the server and never larger;
    it shrinks in the frame's aspect ratio where the window is smaller (the sizes it takes in a
    browser: ``test_play_layout.py``)."""
    stream, play = SCRIPTS["stream.js"], SCRIPTS["play.js"]
    constructor = stream[
        stream.index("  constructor(url, canvas, options = {}) {") : stream.index(
            "  send(message) {"
        )
    ]
    size = constructor.index("[canvas.width, canvas.height] = page.frame_pixels;")
    assert size < constructor.index(
        "this.turn = unsent && InstantTurn.on(canvas, () => this.unturn());"
    )
    assert size < constructor.index(
        "this.ctx = this.turn ? null : canvas.getContext('2d', { alpha: false });"
    )
    # the instant turn draws on the canvas's own pixels
    assert "this.size = [canvas.width, canvas.height];" in SCRIPTS["turn.js"]
    for script in (stream, SCRIPTS["turn.js"]):
        assert not re.search(r"devicePixelRatio|ResizeObserver|getBoundingClientRect", script)
    assert 'style="--acc:${color};--frame-w:${width}px;--frame-ratio:${width / height}"' in play
    assert "--stage-w: max(240px, min(var(--frame-w)," in CSS
    assert "calc((100vh - var(--reserve)) * var(--frame-ratio))" in CSS
    assert (
        ".stage { position: relative; width: var(--stage-w); aspect-ratio: var(--frame-ratio);"
        in CSS
    )
    assert ".stage.native #view { image-rendering: pixelated; }" in CSS  # not the markers


def test_the_stage_holds_the_view_and_the_panels_are_outside_it():
    """Over the frame while one plays: the crosshair and the field's markers, nothing else. The
    HUD, the toast and the panels are the page's, around the stage (their places in a browser:
    ``test_play_layout.py``)."""
    play = SCRIPTS["play.js"]
    markup = play[play.index("function playMarkup(") : play.index("// The pause overlay's markup")]
    stage = re.search(r'<div class="stage" id="stage">(.*?)</div>', markup, flags=re.S).group(1)
    assert re.findall(r"<(\w+)", stage.replace("${crosshair}", "<svg")) == [
        "canvas",
        "canvas",
        "svg",
    ]
    assert markup.index('id="overlay"') > markup.index("</footer>") > markup.index('id="stage"')
    assert "stage.append" not in play and "$('#toast')" in play
    assert ".overlay { position: fixed; inset: 0;" in CSS


def test_the_play_view_draws_each_frame_as_it_was_generated_or_turned_by_the_frames_shown():
    """Every frame is drawn on a canvas of the frame's own pixels as it was generated: on a 2D
    canvas, or by the instant turn (WebGL2) turned by the turn it does not hold yet as far as the
    frames shown reach, from their pixels only, so nothing mirrored, blurred, stretched or black
    shows at the view's edge (``test_instant_turn.py`` runs it in Chrome). The instant turn is a
    setting, on by default, and the play view's alone; a spectator's frames are drawn as they are.
    Only the motion feedback, a setting that is off by default, scales and shifts the frame
    (``test_motion_feedback.py`` runs it in Chrome)."""
    stream, play, feedback = SCRIPTS["stream.js"], SCRIPTS["play.js"], SCRIPTS["feedback.js"]
    turn, watch = SCRIPTS["turn.js"], SCRIPTS["watch.js"]
    assert not re.search(r"LookWarp|LookTurn|texture2D|turnOf", stream + turn)
    assert "export const INSTANT_TURN_DEFAULT = false;" in turn
    assert "instant: store('wc.instantturn') ?? INSTANT_TURN_DEFAULT," in play
    assert "unsent: () => input.unsent(), instant: settings.instant," in play
    assert "unsent" not in watch
    # a pixel of the turned view is the newest frame's, the panorama's or (the viewmodel) the newest
    # frame's own: none is made up, and the CPU's test of the view (Coverage) has every pixel known
    display = turn[turn.index("const DISPLAY = ") : turn.index("// The instant turn on a canvas")]
    assert "colour = vec4(mix(scene, still, m), 1.0);" in display
    assert "CLAMP_TO_EDGE" in turn and "REPEAT" not in turn and "blur" not in turn.lower()
    assert "export const FEEDBACK_DEFAULT = 'off';" in feedback
    assert "export const FEEDBACK_SHARE = { off: 0, light: 1, strong: 2 };" in feedback
    draw = stream[
        stream.index("  drawFlat(bitmap) {") : stream.index("  // Where the point (u, v)")
    ]
    assert (
        "if (zoom === 1 && shift === 0) {\n      ctx.drawImage(bitmap, 0, 0, width, height);"
        in draw
    )
    assert "if (this.ctx) this.drawFlat(frame.bitmap);" in stream
    show = stream[stream.index("  show(now) {") : stream.index("  stats() {")]
    assert "if (this.last && (this.turn || moving)) this.draw(this.last, false);" in show
    assert "const at = stream.place(u, v);\n      if (!at) continue;" in play


def test_the_latency_breakdown_is_docked_or_a_popover_by_the_width_of_the_window():
    """The breakdown opens from the stats at the right of the view in a window of 1260 px or more
    (landscape), and as a popover under them in any other. The style has the one query and the page
    asks the style which layout it is in (``--popover``), so the two cannot disagree
    (``test_play_layout.py`` shows what the mouse does in each)."""
    play = SCRIPTS["play.js"]
    query = "(max-width: 1259px), (max-aspect-ratio: 1/1)"
    assert "(max-width" not in play and "matchMedia(POPOVER)" not in play
    assert (
        "const popover = () => getComputedStyle(play).getPropertyValue('--popover').trim() === '1';"
        in play
    )
    assert (
        f"@media {query} {{\n  .play {{ --popover: 1; }}\n  .lat-panel {{ position: fixed;" in CSS
    )
    assert CSS.count(query) == 1
    assert "if (!touch && !(state.engaged && popover())) openPanel(panel.hidden);" in play
    assert "if (e.target === overlay && !panel.hidden && popover()) openPanel(false);" in play


def test_the_position_status_and_what_it_means_are_in_the_breakdown():
    latency = SCRIPTS["latency.js"]
    for status in ("guarded", "reacquiring", "degraded"):
        assert f"  {status}: " in latency
    assert "YOUR POSITION" in latency and "Object.hasOwn(POSITIONS, position)" in latency


def test_a_marker_is_an_outline_that_shows_at_any_distance():
    """The ring's horizontal radius is 4000 over the depth within 12 to 30 px: the figure's size
    close up, 60 px across at most, and never narrower than 24 px across; its vertical radius is
    0.45 of that and at least 7 px (14 px high). An outline of the player's colour on a dark one,
    with the label above it."""
    play = SCRIPTS["play.js"]
    assert "const MARKER_SCALE = 4000;\nconst MARKER_RADIUS = [12, 30];" in play
    assert "const MARKER_FLAT = 0.45;\nconst MARKER_MIN_HEIGHT = 7;" in play
    assert "const ry = Math.max(MARKER_MIN_HEIGHT * dpr, rx * MARKER_FLAT);" in play
    assert "ctx.strokeStyle = 'rgba(0, 0, 0, .7)';" in play and "ctx.strokeText(label" in play


def test_a_touch_device_plays_with_the_pad_and_is_not_told_it_cannot():
    """A coarse pointer plays with the on-screen pad, outside the stage (after the view, before
    the rows below it), and its pause panel, not a note, says how."""
    play = SCRIPTS["play.js"]
    assert (
        "matchMedia('(pointer: coarse)').matches && !matchMedia('(any-pointer: fine)').matches"
        in play
    )
    assert "mobile-note" not in play + CSS and "keyboard and a mouse" not in play
    markup = play[play.index("function playMarkup(") : play.index("// The pause overlay's markup")]
    assert markup.index('id="stage"') < markup.index("${touch ? padMarkup() : ''}")
    assert markup.index("${touch ? padMarkup() : ''}") < markup.index("<footer")
    assert "<h2>${touch ? 'Tap' : 'Click'} to play</h2>" in play


def test_the_arrow_keys_and_the_sticks_ask_for_turns_above_what_the_generator_follows():
    """A held arrow key asks for a yaw rate of 120 degrees a second (the Turn speed setting's
    default, 20 to 320) and for 0.8 of it in pitch; the right stick asks for nothing within 0.12
    of its travel, for 25 degrees a second just beyond it and for the maximum turn speed at full
    deflection; a direction of the left stick is held beyond 0.4. The yaw is followed from about
    20 degrees a second on (measured on an H200 for the yaw): the defaults and the stick's floor
    are above it, the setting's range goes below it. A stalled frame asks for a quarter of a
    second at most. The curve, the integration and the keys are checked in Chrome
    (test_play_input.py)."""
    inputs, play = SCRIPTS["input.js"], SCRIPTS["play.js"]
    for line in (
        "export const TURN_DEGREES_PER_SECOND = 120;",
        "export const PITCH_TURN_SHARE = 0.8;",
        "export const TURN_SPEED_RANGE = [20, 320];",
        "export const MAX_FRAME_STEP_S = 0.25;",
        "export const STICK_HOLD = 0.4;",
        "export const LOOK_DEAD_ZONE = 0.12;",
        "export const LOOK_FLOOR = 25;",
    ):
        assert line in inputs, line
    assert "LOOK_TOP" not in inputs + SCRIPTS["pad.js"]
    assert 0.8 * 120 > 20 and 25 > 20  # the default pitch and the stick's floor, above 20
    assert "MAX_FRAME_STEP_S" in play and "arrowLook" in play


def test_the_mouse_turns_as_far_as_it_moves_and_the_keys_and_sticks_at_a_rate():
    """As in the game: the mouse's turn goes in the next sample whole, with no cap, ramp or limit
    on it (the worker keeps each frame within what the generator was trained on), but for the
    first movement after the pointer lock took hold, the pointer's jump to it; the arrow keys and
    the sticks turn at a rate, at most the maximum turn speed (240 degrees a second by default, up
    to the 320 the generator was trained on: 20 degrees a frame at 16 frames a second), a stick's
    top speed. What the page sends is checked in Chrome (test_play_input.py)."""
    inputs, play, pad = SCRIPTS["input.js"], SCRIPTS["play.js"], SCRIPTS["pad.js"]
    for line in (
        "export const MAX_TURN_DEGREES_PER_SECOND = 240;",
        "export const MAX_TURN_RANGE = [30, 320];",
    ):
        assert line in inputs, line
    assert CAMERA_MAX_DEGREES * FPS == 320
    assert not re.search(r"governor|Governor|RAMP|PENDING_LIMIT|MOUSE_STEP", inputs + play + pad)
    turn = inputs[inputs.index("  turn(dx, dy, time) {") : inputs.index("  lockTaken() {")]
    assert "this.look(dy * this.sens, -dx * this.sens, time);" in turn
    assert "if (this.skipMove) {" in turn
    assert "if (document.pointerLockElement) input.lockTaken();" in play
    look = inputs[inputs.index("  look(dp, dy, time, began = time) {") : inputs.index("  capped(")]
    assert "this.dp += dp;" in look and "this.dy += dy;" in look
    assert "input.lookAtRate(dp, dy, step, now, pressed > previous ? pressed : now);" in play
    assert pad.count("this.input.lookAtRate(dp, dy, dt, now);") == 2
    assert ".look(" not in pad and "stickLook(" in pad
    assert pad.count("this.input.maxRate") == 2
    assert "export const SAMPLE_RATE = 60;" in inputs
    assert "input.maxRate = settings.maxTurn;" in play


def test_the_pad_buttons_are_the_models_and_the_settings_have_their_keys():
    """The gamepad and the pad hold buttons the model has; the settings the pause panel keeps in
    localStorage are the turn speed, the maximum turn speed, whether the mouse looks, the mouse's
    sensitivity and the motion feedback."""
    pad, play = SCRIPTS["pad.js"], SCRIPTS["play.js"]
    gamepad = re.search(r"const GAMEPAD_BUTTONS = (.*?);\n", pad, flags=re.S).group(1)
    assert set(re.findall(r": '([a-z_0-9]+)'", gamepad)) <= set(CONTROL_BUTTONS)
    assert len(re.findall(r": '([a-z_0-9]+)'", gamepad)) == 6
    assert set(re.findall(r"button\('([a-z]+)'", pad)) == {"jump", "duck", "attack", "next"}
    assert {"jump", "duck", "attack"} <= set(CONTROL_BUTTONS)
    for key in ("wc.arrowturn", "wc.turncap", "wc.mouselook", "wc.sens", "wc.motion", "wc.walk"):
        assert f"'{key}'" in play, key


def test_walk_by_default_holds_the_walk_button_with_the_movement_buttons_in_one_place():
    """The page holds the walk button while a movement button is held and the player does not
    hold it (the key that runs); the buttons are named once, in ``input.js``, which the samples
    are made from; ``Shift`` is still the key of the walk button. What it does with real keys runs
    in Chrome (``test_walk.py``)."""
    inputs, play = SCRIPTS["input.js"], SCRIPTS["play.js"]
    assert (
        "export const MOVEMENT_BUTTONS = ['forward', 'move_left', 'back', 'move_right'];" in inputs
    )
    assert "export const WALK_BUTTON = 'speed';" in inputs
    assert "ShiftLeft: 'speed'" in play and "const held = this.carried(pressed);" in inputs
    assert "input.walk = settings.walk;" in play and "store('wc.walk') !== false" in play
    assert (
        "input.pressed.has(cap.dataset.b)" in play
    )  # the caps show the keys, not the walk the page holds


def test_the_motion_feedback_goes_from_the_buttons_to_the_stream_and_is_off_for_none_of_them():
    """The buttons that go down and up reach the controller with the sample that carries them,
    every frame drawn tells it what it holds, and the stream asks it for the scale and the shift
    at each animation frame, which the 2D canvas draws the frame with. The controller and the
    pixels are checked in Chrome (``test_feedback.py``, ``test_motion_feedback.py``)."""
    inputs, play, stream = SCRIPTS["input.js"], SCRIPTS["play.js"], SCRIPTS["stream.js"]
    assert "this.edges(down, up, time, this.seq);" in inputs
    assert "(...edges) => feedback.buttons(...edges)" in play
    assert "feedback.arrived(h.q, t.shown);" in play
    assert "moved: (now) => feedback.at(now)" in play
    assert "const { zoom, shift } = this.move;" in stream
    assert "const move = this.moved ? this.moved(now) : NO_MOVE;" in stream
    assert "export class MotionFeedback" in SCRIPTS["feedback.js"]


def test_f_inspects_with_a_knife_only_and_says_so():
    """The inspect button holds only while the weapon held is one of the welcome's
    ``inspect_weapons`` (the knives; the worker clears it from the other frames too,
    ``test_inputs.py``); F's key cap and its line of the legend say so and are greyed with another
    weapon. What the page sends with real keys runs in Chrome (``test_play_input.py``)."""
    inputs, play = SCRIPTS["input.js"], SCRIPTS["play.js"]
    assert "export const INSPECT_BUTTON = 'look_at_weapon';" in inputs
    assert "this.inspectIds = new Set(m.inspect_weapons);" in inputs
    assert "if (!this.canInspect()) held.delete(INSPECT_BUTTON);" in inputs
    assert "const INSPECT_SAYS = 'F — inspect (knife)';" in play
    assert "[INSPECT_BUTTON, 'F', '', INSPECT_SAYS]" in play
    assert ">Inspect (knife)</dd>" in play and "Y inspect (knife)" in play
    assert ".cap.unavailable, .legend .unavailable { opacity: .38; }" in CSS
