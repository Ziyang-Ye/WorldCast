// A worker's frame stream on a canvas: WebSocket in, JPEG decode, a playout that follows the rate the blocks arrive at
// and never falls more than a few frames behind a block. Each frame is acknowledged as it arrives: the worker keeps
// only a few frames unacknowledged on the way, so a slow link gets fewer, smaller frames instead of an ever older
// backlog. Each frame is drawn as it was generated, pixel for pixel. The play view turns it at every animation frame by
// the turn it does not contain yet, as far as the pixels of the frames shown reach (the instant turn, turn.js; a
// setting, on by default), and the motion feedback (feedback.js), when the player turns it on, scales and shifts it by
// the move it does not contain yet.

import { median, page } from './common.js';
import { INSTANT_TURN_DEFAULT, InstantTurn, ViewmodelMotion } from './turn.js';

const PING_MS = 1000;
// The playout's period stays within these factors of the frame period at which the blocks arrive.
const STRETCH = [0.8, 1.25];
// The measured block period and its jitter move this share of the way to each block's.
const FOLLOW = 0.2;
// A block's measurement counts as at most this many frame periods, or four jitters, off the estimate: one late block
// moves the estimate little, a lasting change in a few blocks.
const OUTLIER_FRAMES = 2;
// A frame that arrives more than this many block periods after the frame a block before it ends a pause of the
// stream (the player away, a stalled GPU), not a block period: the measurement starts over. A frame shown more than
// this many block periods after the one before it ends a pause of the page (hidden): it is not the display's interval.
const PAUSE_BLOCKS = 2;
// The interval at which the display shows its frames moves this share of the way to each frame's.
const SHOWN_FOLLOW = 0.2;
// A frame due within this many ms is shown at this animation frame rather than the next.
const SLACK_MS = 2;
// The playout (time-stretching): which received frame to show, and when. Frames come a block at a time; frame k is
// expected a block period after frame k - perBlock arrived, the period measured from the arrivals. Each frame is shown
// for the shortest period that still has every coming frame arrive a margin before it is due (`margin` frame periods,
// the worker's play.margin_frames, or two jitters), within STRETCH of the
// blocks' frame period: the display slows down while the blocks come late and catches up when they come early,
// instead of freezing and bursting. A frame missing when due is a hold: the frame on screen stays, and the missing one
// shows as soon as it arrives. There is no rebuffer. The slowing and the catching up are bounded by STRETCH; a display
// further behind than they can make up (a main thread busy for the first seconds, a page that was hidden, a block that
// came early) drops its oldest frames instead: when a block's first frame arrives, the frames before it that the
// display would not show within `backlog` frame periods are dropped, so that frame waits for the frames kept in front
// of it and for the display's next frame time, whatever the backlog was.
// A version of a frame against another of it (header.v, the refresh that gave it; header.pv, its preview): a newer
// version ranks above an older one, a version's own frame above its preview.
const rank = (header) => 2 * (header.v ?? 0) + (header.pv ? 0 : 1);

// Hands the values of promises to `use` in the order they were pushed, whatever order they settle in; one that fails is
// skipped.
export class InOrder {
  constructor() {
    this.last = Promise.resolve();
  }

  push(pending, use) {
    const before = this.last;
    const settled = pending.then((value) => ({ value }), () => null);
    this.last = settled
      .then(async (result) => {
        await before;
        if (result) use(result.value);
      })
      .catch((error) => console.error(error));
    return this.last;
  }
}

export class Playout {
  // period: ms per frame at the worker's play.fps; perBlock: the frames of a block; buffer: the frames to hold before
  // the first one is shown; backlog: the frame periods that the frames of the blocks before may still wait when a
  // block's first frame arrives; margin: the frame periods every coming frame is to arrive before it is due.
  constructor(period, perBlock, buffer, backlog, margin) {
    Object.assign(this, { perBlock, buffer, backlog, margin });
    this.paced = period * perBlock;          // the block period the worker paces for: the blocks come no faster, ms
    this.blockMs = this.paced;               // the period at which the blocks arrive, ms
    this.jitter = 0;                         // the mean deviation of the arrivals from it, ms
    this.arrived = new Map();                // frame -> when it arrived (ms), over the newest two blocks
    this.newest = -1;                        // the newest frame received
    this.queue = [];                         // the frames received and not yet shown, by frame
    this.playing = false;
    this.next = 0;                           // when the next frame is due
    this.holding = false;
    this.holds = [];                         // when a frame on screen began to hold
    this.every = period;                     // the interval at which the display shows its frames, smoothed, ms
    this.shownAt = null;                     // when the frame on screen was shown
    this.shownK = -1;                        // the frame on screen
  }

  // A frame received: {header, recv}, its index header.k, whether it starts its block header.f, when it arrived recv
  // (ms). Returns the frames dropped: at a block's first frame, the oldest frames before it beyond what the display
  // shows in `backlog` frame periods (none unless the display fell behind). A display that keeps up with the frame
  // period (its interval within what the playout stretches to) keeps `backlog` frames: the measured interval is noisy
  // by a few percent, which would otherwise round the count down by one at random. A slower display keeps the frames
  // it shows in `backlog` frame periods, fewer the slower it is. The count steps once, from `backlog` to one fewer, at
  // an interval of STRETCH[1] frame periods: a display that sits right there keeps one frame more or fewer by the noise
  // of its interval, which is accepted. A frame at or before the frame on screen is dropped. A frame received again or
  // sent again (header.r: a newer version, the worker's play.swap_sent) takes the place of the one waiting if it ranks
  // above it (rank), which is dropped, and is dropped itself otherwise or with none waiting; it measures nothing.
  add(frame) {
    const k = frame.header.k;
    if (k <= this.shownK) return [frame];
    if (this.arrived.has(k) || frame.header.r) {
      const i = this.queue.findIndex((f) => f.header.k === k);
      if (i < 0 || rank(this.queue[i].header) >= rank(frame.header)) return [frame];
      return this.queue.splice(i, 1, frame);
    }
    const before = this.arrived.get(k - this.perBlock);
    if (before !== undefined) this.measure(frame.recv - before);
    this.arrived.set(k, frame.recv);
    this.newest = Math.max(this.newest, k);
    for (const old of this.arrived.keys()) if (old <= this.newest - 2 * this.perBlock) this.arrived.delete(old);
    let i = this.queue.length;               // decodes may finish out of order: keep the queue sorted by frame
    while (i > 0 && this.queue[i - 1].header.k > k) i -= 1;
    this.queue.splice(i, 0, frame);
    if (!frame.header.f) return [];
    const period = this.paced / this.perBlock;
    const keeps = this.every < STRETCH[1] * period ? this.backlog : Math.floor((this.backlog * period) / this.every);
    return this.queue.splice(0, Math.max(0, i - keeps));
  }

  // A frame arrived `gap` ms after the frame a block before it: a measurement of the block period.
  measure(gap) {
    if (gap > PAUSE_BLOCKS * this.blockMs) {
      this.blockMs = this.paced;
      this.jitter = 0;
      return;
    }
    const share = FOLLOW / this.perBlock;    // each of a block's frames measures the period once
    const bound = Math.max(OUTLIER_FRAMES * this.blockMs / this.perBlock, 4 * this.jitter);
    const deviation = Math.min(Math.max(gap - this.blockMs, -bound), bound);
    this.jitter += share * (Math.abs(deviation) - this.jitter);
    this.blockMs = Math.max(this.paced, this.blockMs + share * deviation);
  }

  // The frame to show at `now` (ms), or null: none is due, or the one due has not arrived (a hold).
  take(now) {
    if (!this.playing) {
      if (this.queue.length < this.buffer) return null;
      this.playing = true;
      this.next = now;
    }
    if (now < this.next - SLACK_MS) return null;
    const frame = this.queue.shift();
    if (!frame) {
      if (!this.holding) this.holds.push(now);
      this.holding = true;
      return null;
    }
    // The next frame is due a period after this one was due, not after the animation frame that shows it, up to one
    // later: the display keeps the period's rate. After a hold, or a frame period late (a hidden page), from now.
    const late = this.holding || now - this.next >= this.blockMs / this.perBlock;
    const gap = now - this.shownAt;
    if (!this.holding && this.shownAt !== null && gap <= PAUSE_BLOCKS * this.blockMs) {
      this.every += SHOWN_FOLLOW * (gap - this.every);
    }
    this.shownAt = now;
    this.holding = false;
    this.shownK = frame.header.k;
    const due = late ? now : this.next;
    this.next = due + this.period(due);
    return frame;
  }

  // How long the frame due at `due` stays on screen: the shortest period that has each coming frame i arrive a margin
  // before it is due, after this frame, the frames that wait and those still to come before i; within STRETCH of the
  // blocks' frame period, which it is until a block period has been measured.
  period(due) {
    const base = this.blockMs / this.perBlock;
    const margin = Math.max(this.margin * base, 2 * this.jitter);
    let period = STRETCH[0] * base;
    let measured = false;
    for (let i = this.newest + 1; i <= this.newest + this.perBlock; i += 1) {
      const before = this.arrived.get(i - this.perBlock);
      if (before === undefined) continue;
      measured = true;
      const frames = this.queue.length + i - this.newest;
      period = Math.max(period, (before + this.blockMs + margin - due) / frames);
    }
    return measured ? Math.min(period, STRETCH[1] * base) : base;
  }
}

// The frame as it is: neither scaled nor moved.
const NO_MOVE = { zoom: 1, shift: 0 };

export class FrameStream {
  // url: the worker's play or watch socket; canvas: where frames go. buffer: the frames to hold before the first one
  // is shown (null: the worker's play.jitter_buffer_frames, from its welcome). unsent() (the play view): the turn the
  // input has not sent yet, degrees [pitch, yaw]; with it the frames are drawn by the instant turn, while `instant` (the
  // setting) is on turned by the turn they do not hold yet, unless the browser cannot draw it (no WebGL2, or a lost
  // context that does not come back: the frames are then drawn as they are, on a canvas that replaces `canvas`).
  // moved(now) (the play view): the motion feedback at the animation frame `now`, {zoom, shift} (feedback.js), which
  // every frame is drawn with, on either canvas. onFrame(header, {shown, recv}) after a frame is drawn; onText(message)
  // for JSON messages.
  constructor(url, canvas, options = {}) {
    const {
      buffer = null, unsent = null, instant = INSTANT_TURN_DEFAULT, moved = null, onFrame = () => {}, onText = () => {},
      onClose = () => {},
    } = options;
    Object.assign(this, { canvas, buffer, unsent, instant, moved, onFrame, onText, onClose });
    this.move = NO_MOVE;                     // the motion feedback now
    this.playout = null;                     // made from the worker's welcome (its first message)
    this.decoded = new InOrder();            // the frames to the playout in the order they arrived
    this.shownAt = [];
    this.rtts = [];
    this.sent = [0, 0];                      // the turn of the input samples the session has: the worker's, then this page's
    [canvas.width, canvas.height] = page.frame_pixels;   // the frame's own pixels, whatever size the canvas is shown at
    this.turn = unsent && InstantTurn.on(canvas, () => this.unturn());
    this.ctx = this.turn ? null : canvas.getContext('2d', { alpha: false });
    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';
    this.ws.onmessage = (e) => (typeof e.data === 'string' ? this.text(JSON.parse(e.data)) : this.binary(e.data));
    this.ws.onclose = (e) => { this.closed = true; this.onClose(e); };
    this.pinger = setInterval(() => this.send({ t: 'ping', c: performance.now() }), PING_MS);
    this.raf = requestAnimationFrame((t) => this.show(t));
  }

  // Send a message; an input sample's turn counts in the turn the session has.
  send(message) {
    if (this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify(message));
    if (message.t === 'in') this.sent = [this.sent[0] + message.dp, this.sent[1] + message.dy];
  }

  // The instant turn is given up: a canvas that gave a WebGL context gives no 2D one, so a canvas of the same attributes
  // takes its place, and the frame on screen is drawn on it.
  unturn() {
    if (!this.turn) return;
    const { canvas } = this;
    this.canvas = canvas.cloneNode(false);
    canvas.replaceWith(this.canvas);
    this.turn = null;
    this.ctx = this.canvas.getContext('2d', { alpha: false });
    if (this.last) this.draw(this.last, false);
  }

  close() {
    clearInterval(this.pinger);
    clearTimeout(this.turn?.waiting);
    cancelAnimationFrame(this.raf);
    this.ws.onclose = null;
    this.ws.close();
    this.playout?.queue.forEach((f) => f.bitmap.close());
  }

  text(message) {
    if (message.t === 'welcome') {
      const buffer = this.buffer ?? Math.max(1, message.jitter_buffer_frames);
      this.playout = new Playout(1000 / message.fps, message.frames_per_block, buffer, message.backlog_frames,
        message.margin_frames);
      this.sent = [this.sent[0] + message.tn[0], this.sent[1] + message.tn[1]];
      this.motion = new ViewmodelMotion(message);
    }
    if (message.t === 'pong') {
      this.rtts.push(performance.now() - message.c);
      if (this.rtts.length > 5) this.rtts.shift();
    }
    this.onText(message);
  }

  // A frame received: acknowledged at once, decoded, and given to the playout in the order the frames arrived, whatever
  // order their decodes finish in (a frame and its newer version sent right after it).
  binary(data) {
    const recv = performance.now();
    const n = new DataView(data).getUint32(0);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 4, n)));
    this.send({ t: 'ack', r: header.k });
    const decoding = createImageBitmap(new Blob([new Uint8Array(data, 4 + n)], { type: 'image/jpeg' }));
    this.decoded.push(decoding, (bitmap) => {
      if (this.closed) return bitmap.close();
      for (const dropped of this.playout.add({ header, bitmap, recv })) dropped.bitmap.close();
    });
  }

  // Draw a received frame {header, bitmap} (a browser that gives no 2D context draws nothing); keep: it is the frame on
  // screen from now on (the instant turn's newest frame).
  draw(frame, keep = true) {
    if (this.turn) {
      const now = performance.now();
      if (keep) this.turn.shown(frame.bitmap, this.pose(frame), now);
      this.turn.draw(this.look(frame), this.move, now);
    } else if (this.ctx) this.drawFlat(frame.bitmap);
    if (keep) {
      if (this.last) this.last.bitmap.close();
      this.last = frame;
    }
  }

  // Where a frame was taken: its orientation, degrees (its header's yaw turned in the session and the camera's pitch),
  // and the player's own position, engine units; and the part of it the instant turn's panorama takes, by what its
  // viewmodel may be doing (ViewmodelMotion).
  pose(frame) {
    const { tn, cp, s, pv } = frame.header;
    // a refresh's preview (a coarse estimate of its frame) goes into no panorama
    return { yaw: tn[1], pitch: cp, at: s.slice(2, 5), part: pv ? 'none' : this.motion.part(frame.header) };
  }

  // Where the player looks now, degrees {yaw, pitch}: the frame's orientation turned by the turn it does not hold (the
  // turn the session has, sent or not, less the frame's tn), the pitch within the camera's range; the frame's own while
  // the instant turn is off.
  look(frame) {
    const { tn, cp } = frame.header;
    if (!this.instant) return { yaw: tn[1], pitch: cp };
    const [pitch, yaw] = this.unsent();
    const [low, high] = page.pitch_range_degrees;
    const down = cp + this.sent[0] + pitch - tn[0];
    return { yaw: this.sent[1] + yaw, pitch: Math.min(Math.max(down, low), high) };
  }

  // The frame on the canvas, pixel for pixel; under the motion feedback, scaled about the centre and moved as it says, and
  // where that leaves the canvas uncovered, the frame mirrored about its edges.
  drawFlat(bitmap) {
    const { ctx, canvas } = this;
    const { zoom, shift } = this.move;
    const [width, height] = [canvas.width, canvas.height];
    ctx.imageSmoothingQuality = 'high';
    if (zoom === 1 && shift === 0) {
      ctx.drawImage(bitmap, 0, 0, width, height);
      return;
    }
    const [w, h] = [width * zoom, height * zoom];
    const [x, y] = [(width - w) / 2 + shift * width, (height - h) / 2];
    const covered = x <= 0 && y <= 0 && x + w >= width && y + h >= height;
    for (const i of covered ? [0] : [0, -1, 1]) {
      for (const j of covered ? [0] : [0, -1, 1]) {
        ctx.save();
        ctx.translate(x + (i ? i + 1 : 0) * w, y + (j ? j + 1 : 0) * h);
        ctx.scale(i ? -1 : 1, j ? -1 : 1);
        ctx.drawImage(bitmap, 0, 0, w, h);
        ctx.restore();
      }
    }
  }

  // Where the point (u, v) of the frame on screen shows on the canvas, both as fractions of the width and height from
  // the top left: turned by the instant turn, if any, and moved by the motion feedback; null where it is behind the view.
  place(u, v) {
    const at = this.turn ? this.turn.place(u, v) : [u, v];
    const { zoom, shift } = this.move;
    if (!at || (zoom === 1 && shift === 0)) return at;
    return [0.5 + zoom * (at[0] - 0.5) + shift, 0.5 + zoom * (at[1] - 0.5)];
  }

  // Once per animation frame: draw the frame the playout says is due; under the instant turn or the motion feedback, the
  // frame on screen again as far as the player has turned or moved since.
  show(now) {
    this.raf = requestAnimationFrame((t) => this.show(t));
    const move = this.moved ? this.moved(now) : NO_MOVE;
    const moving = move.zoom !== this.move.zoom || move.shift !== this.move.shift;
    this.move = move;
    const frame = this.playout?.take(now);
    if (!frame) {
      if (this.last && (this.turn || moving)) this.draw(this.last, false);
      return;
    }
    this.draw(frame);
    this.shownAt.push(now);
    while (this.shownAt.length && now - this.shownAt[0] > 1000) this.shownAt.shift();
    this.onFrame(frame.header, { shown: now, recv: frame.recv });
  }

  // fps shown, round trip (ms), stalls (holds) in the last 10 s, and a 1-3 quality score.
  stats() {
    const now = performance.now();
    const { playout } = this;
    if (playout) playout.holds = playout.holds.filter((t) => now - t < 10000);
    const stalls = playout ? playout.holds.length : 0;
    const rtt = median(this.rtts);
    const byRtt = Number.isNaN(rtt) ? 2 : rtt < 60 ? 3 : rtt < 150 ? 2 : 1;
    const byStall = stalls === 0 ? 3 : stalls <= 3 ? 2 : 1;
    return { fps: this.shownAt.length, rtt, stalls, quality: Math.min(byRtt, byStall) };
  }
}
