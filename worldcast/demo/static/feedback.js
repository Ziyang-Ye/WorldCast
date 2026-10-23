// Motion feedback: the view's pre-response to a movement button. A button reaches the generator at the next block and shows
// in the frames some time after that, so from the press to the first drawn frame that folds it in (latency.js) the view
// answers at once with a small change of the frame on screen, the way a VR headset shifts the last frame to the head's
// newest pose: forward scales the view up about its centre, back scales it down, a strafe shifts it against the move. Each
// held button has one ramp, from its press to the cap; the frame that folds it in hands the move over to the generated
// frames, and the ramp falls to nothing over HANDOVER_S, as it does when the button is let go before then (a stop is not
// answered). The change is a function of the buttons' state alone, within the caps: it never builds up. It is off unless
// the player turns it on (the pause panel): a scaled or shifted frame is blurred, and where it leaves the view uncovered
// the view shows the frame mirrored about its edge; off, every frame is drawn as it was generated.

// The setting's values, and the share of the caps each takes. Off draws the frames as they are.
export const FEEDBACK_SHARE = { off: 0, light: 1, strong: 2 };
export const FEEDBACK_DEFAULT = 'off';
// The caps at share 1. Forward scales the view up by DOLLY_IN, back down by DOLLY_OUT (both about the view's centre); a
// strafe moves the view by STRAFE_SHIFT of its width, against the move (the world moves right as the player steps left).
export const DOLLY_IN = 0.04;
export const DOLLY_OUT = 0.03;
export const STRAFE_SHIFT = 0.025;
// A press reaches its cap after RAMP_S, along an ease-out; the move falls to nothing over HANDOVER_S from the frame that
// folds it in.
export const RAMP_S = 0.4;
export const HANDOVER_S = 0.2;

// The buttons that move the player, with what each does to the view at its cap: [scale, shift].
const MOVES = {
  forward: [DOLLY_IN, 0], back: [-DOLLY_OUT, 0], move_left: [0, STRAFE_SHIFT], move_right: [0, -STRAFE_SHIFT],
};
const easeOut = (x) => 1 - (1 - x) ** 2;
const smooth = (x) => x * x * (3 - 2 * x);
const clamp01 = (x) => Math.min(1, Math.max(0, x));

// One button's ramp: idle; waiting (since its press, for the frame that folds in input sample `seq`); or fading (from `from`
// since the frame, or the release), with the next press `next` held back until the fade is over.
class Ramp {
  constructor() {
    Object.assign(this, { phase: 'idle', since: 0, seq: 0, from: 0, next: null });
  }

  // The share of the cap at `now` (ms).
  level(now) {
    if (this.phase === 'wait') return easeOut(clamp01((now - this.since) / 1000 / RAMP_S));
    if (this.phase === 'fade') return this.from * (1 - smooth(clamp01((now - this.since) / 1000 / HANDOVER_S)));
    return 0;
  }

  // A fade that is over leaves the ramp idle; a press that waited for it starts its ramp where the fade ended.
  settle(now) {
    const end = this.since + HANDOVER_S * 1000;
    if (this.phase !== 'fade' || now < end) return;
    this.phase = 'idle';
    if (this.next) Object.assign(this, { phase: 'wait', since: end, seq: this.next.seq, next: null });
  }

  fade(now) {
    Object.assign(this, { from: this.level(now), phase: 'fade', since: now });
  }

  press(time, seq) {
    if (this.phase === 'idle') Object.assign(this, { phase: 'wait', since: time, seq });
    else if (this.phase === 'fade') this.next = { seq };
  }

  release(now) {
    this.next = null;
    if (this.phase === 'wait') this.fade(now);
  }

  // A frame drawn at `now` holds the input samples up to `q`.
  folded(q, now) {
    if (this.next && this.next.seq <= q) this.next = null;
    if (this.phase === 'wait' && this.seq <= q) this.fade(now);
  }
}

export class MotionFeedback {
  // setting: a key of FEEDBACK_SHARE.
  constructor(setting = FEEDBACK_DEFAULT) {
    this.ramps = Object.fromEntries(Object.keys(MOVES).map((b) => [b, new Ramp()]));
    this.set(setting);
  }

  // Take the setting `setting`; nothing is under way after it.
  set(setting) {
    this.strength = FEEDBACK_SHARE[setting] ?? FEEDBACK_SHARE[FEEDBACK_DEFAULT];
    for (const b of Object.keys(MOVES)) this.ramps[b] = new Ramp();
  }

  // The buttons `pressed` and `released` at `time`, the press carried by input sample `seq` (PlayerInput's `buttons`).
  buttons(pressed, released, time, seq) {
    if (!this.strength) return;
    for (const b of released) this.ramps[b]?.release(time);
    for (const b of pressed) this.ramps[b]?.press(time, seq);
  }

  // A frame drawn at `now` holds the input samples up to `q` (its header's q): the buttons they carry are in the frames
  // from here on.
  arrived(q, now) {
    for (const ramp of Object.values(this.ramps)) {
      ramp.settle(now);
      ramp.folded(q, now);
    }
  }

  // The view's change at `now` (ms): `zoom`, the scale of the frame about its centre (1: none), and `shift`, how far it
  // moves to the right as a share of its width.
  at(now) {
    let [zoom, shift] = [1, 0];
    for (const [b, [scale, move]] of Object.entries(MOVES)) {
      const ramp = this.ramps[b];
      ramp.settle(now);
      const level = this.strength * ramp.level(now);
      zoom += scale * level;
      shift += move * level;
    }
    return { zoom, shift };
  }
}
