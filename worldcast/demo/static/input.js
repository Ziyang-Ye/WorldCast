// The player's keyboard, mouse, gamepad and touch pad as input samples: the held buttons, the turn since the last sample,
// the weapon. The mouse turns the view as it moves, as in the game: its turn goes in the next sample whole, however far
// and fast (the worker does a turn in as many frames as keep each within what the generator was trained on). A held
// arrow key and a stick turn at a rate. A sample goes out at once when a button or the weapon changes and at the
// animation frames, at most SAMPLE_RATE a second.

const TURN_ONSET_GAP_MS = 150;     // a turn after this long a rest counts as a new input

// A held arrow key turns the yaw at this rate by default (the Turn speed setting), degrees a second, and the pitch at
// PITCH_TURN_SHARE of it: a little over half the game's own arrow-key turn (210 degrees a second).
export const TURN_DEGREES_PER_SECOND = 120;
export const PITCH_TURN_SHARE = 0.8;
export const TURN_SPEED_RANGE = [20, 320];      // the Turn speed setting's range, degrees a second of the yaw
export const TURN_SPEED_STEP = 5;
// The rate sources, the arrow keys and the sticks, turn the yaw at most this fast, degrees a second (the Max turn speed
// setting's default: a stick at full deflection), the pitch at PITCH_TURN_SHARE of it; the mouse turns as far as it
// moves. The setting goes up to 320, what the generator was trained on: 20 degrees a frame at 16 frames a second.
export const MAX_TURN_DEGREES_PER_SECOND = 240;
export const MAX_TURN_RANGE = [30, 320];        // the Max turn speed setting's range
export const MAX_TURN_STEP = 10;
// An animation frame that stalled asks for the turn of at most this long: a busy page does not ask for a spin afterwards.
export const MAX_FRAME_STEP_S = 0.25;
// The samples that go out at the animation frames are SAMPLE_RATE a second on average, on a display of any refresh rate; a
// button or the weapon changing is sent at once besides. A frame earns the credit of its time (a sample costs one), sends its
// sample when the credit holds SAMPLE_DUE of one, and the credit stays within [-1, SAMPLE_CREDIT_MAX]. The turn commanded in a
// frame that sends no sample goes with the next one. The frames of a display of 60 Hz or less each send theirs, though they
// differ a little in length.
export const SAMPLE_RATE = 60;
const SAMPLE_DUE = 0.5;
const SAMPLE_CREDIT_MAX = 1.5;
// A stick holds a movement direction beyond this share of its travel (eight directions, as the keys).
export const STICK_HOLD = 0.4;
// The buttons that move the player: while one is held and Walk by default is on, the page holds the walk button (`speed`).
export const MOVEMENT_BUTTONS = ['forward', 'move_left', 'back', 'move_right'];
export const WALK_BUTTON = 'speed';
// The inspect button: it holds only while the weapon held is one of the welcome's inspect_weapons (the knives, whose
// inspect the generator learned; a gun's it learned badly). The worker clears it from the other frames too.
export const INSPECT_BUTTON = 'look_at_weapon';
// A look stick turns nothing within LOOK_DEAD_ZONE of its travel; beyond it the turn rises from LOOK_FLOOR degrees a
// second to the maximum turn speed at full deflection, with the square of the deflection past the dead zone. The floor
// is above the 20 degrees a second from which the generator follows a sustained yaw (measured on an H200).
export const LOOK_DEAD_ZONE = 0.12;
export const LOOK_FLOOR = 25;

// An arrow key's turn: [pitch, yaw] per unit rate (pitch > 0 looks down, yaw > 0 turns left).
export const ARROWS = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, 1], ArrowRight: [0, -1] };

// The turn the held arrow keys `codes` ask for, [pitch, yaw] degrees a second, at a yaw rate of `speed`; opposite keys
// cancel.
export function arrowLook(codes, speed) {
  let [pitch, yaw] = [0, 0];
  for (const code of codes) {
    pitch += ARROWS[code][0];
    yaw += ARROWS[code][1];
  }
  return [pitch * speed * PITCH_TURN_SHARE, yaw * speed];
}

// The turn a look stick at (x, y) asks for, [pitch, yaw] degrees a second, up to `top` at full deflection; x runs right
// and y down, each -1 to 1 over the stick's travel. Pushing right turns right, pushing down looks down, as the mouse does.
export function stickLook(x, y, top) {
  const deflection = Math.hypot(x, y);
  if (deflection <= LOOK_DEAD_ZONE) return [0, 0];
  const past = Math.min(1, (deflection - LOOK_DEAD_ZONE) / (1 - LOOK_DEAD_ZONE));
  const rate = LOOK_FLOOR + (top - LOOK_FLOOR) * past * past;
  return [(rate * y) / deflection, (-rate * x) / deflection];
}

// The movement buttons a stick at (x, y) holds (x right, y down, each -1 to 1 over its travel): each direction beyond
// STICK_HOLD, so two of them in a diagonal.
export function stickHeld(x, y) {
  const held = new Set();
  if (y < -STICK_HOLD) held.add('forward');
  if (y > STICK_HOLD) held.add('back');
  if (x < -STICK_HOLD) held.add('move_left');
  if (x > STICK_HOLD) held.add('move_right');
  return held;
}

export class PlayerInput {
  // loadout: the weapons of keys 1-4. sens: degrees per pixel (null: the worker's). send(seq, onsets, sample): a sample
  // goes out; onsets: when the inputs it carries began. changed(): a button or the weapon changed. edges(pressed,
  // released, time, seq): the buttons that went down and up at `time`, in the sample `seq` that went out for them.
  constructor(loadout, sens, send, changed = () => {}, edges = () => {}) {
    Object.assign(this, { loadout, sens, send, changed, edges });
    this.holds = new Map();      // source (keys, mouse, pad, touch) -> the buttons it holds
    this.held = new Set();       // the buttons held, as the samples carry them: those of every source
    this.pressed = new Set();    // the buttons the player holds, whichever source: the walk button is the key that runs
    this.walk = false;           // Walk by default: the walk button is held while a movement button is, and Shift lets go of it
    this.dp = 0;
    this.dy = 0;
    this.seq = 0;
    this.weapon = 0;
    this.lastWeapon = 0;
    this.weaponIds = {};
    this.inspectIds = new Set();   // the weapon ids with which the inspect button holds
    this.bit = {};
    this.lastLook = 0;
    this.onset = null;       // when the turn the next sample carries began
    this.maxRate = MAX_TURN_DEGREES_PER_SECOND;   // the rate sources' cap on the yaw, degrees a second
    this.skipMove = false;   // the next mouse movement is the pointer's jump to a lock that took hold
    this.credit = 0;         // the samples the frames have earned and not spent
  }

  // The worker's welcome: the bit of each button (bit i of a sample is button i), the weapon ids, those with which the
  // inspect button holds, the loadout.
  welcome(m) {
    this.bit = Object.fromEntries(m.buttons.map((b, i) => [b, 1 << i]));
    this.weaponIds = m.weapon_ids;
    this.inspectIds = new Set(m.inspect_weapons);
    this.loadout = m.seat.loadout;
    this.sens = this.sens || m.mouse_degrees_per_pixel;
  }

  // Send a sample now. eventTime: when the key, button or weapon change behind it happened (none: a periodic sample).
  sample(eventTime) {
    this.seq += 1;
    const onsets = [eventTime, this.onset].filter((t) => t !== undefined && t !== null);
    this.onset = null;
    let mask = 0;
    for (const b of this.held) mask |= this.bit[b];
    const [dp, dy] = [+this.dp.toFixed(4), +this.dy.toFixed(4)];
    const w = this.weaponIds[this.loadout[this.weapon]] ?? 0;
    this.send(this.seq, onsets, { t: 'in', q: this.seq, b: mask, dp, dy, w });
    this.dp = 0;
    this.dy = 0;
    this.credit = Math.max(this.credit - 1, -1);
  }

  // At every animation frame (`dt` s since the last), after the frame's turn: its sample, if the frames have earned one.
  // Returns whether it went out.
  tick(dt) {
    this.credit = Math.min(this.credit + dt * SAMPLE_RATE, SAMPLE_CREDIT_MAX);
    if (this.credit < SAMPLE_DUE) return false;
    this.sample();
    return true;
  }

  // `source` holds exactly `buttons` from `time` on; a button is held while any source holds it. False when no button
  // of the sample changes.
  hold(source, buttons, time) {
    this.holds.set(source, new Set(buttons));
    return this.commit(time);
  }

  // A button goes down or up at `time`, for `source`; false when the buttons held do not change.
  press(button, down, time, source = 'keys') {
    const buttons = new Set(this.holds.get(source));
    if (down) buttons.add(button); else buttons.delete(button);
    return this.hold(source, buttons, time);
  }

  // Every source lets go of its buttons at `time`.
  releaseAll(time) {
    this.holds.clear();
    return this.commit(time);
  }

  // Whether the weapon held inspects: one of the welcome's inspect_weapons, a knife.
  canInspect() {
    return this.inspectIds.has(this.weaponIds[this.loadout[this.weapon]]);
  }

  // The buttons the samples carry for the buttons the player holds: the same but the inspect button with a weapon that
  // does not inspect, and with Walk by default on, the walk button while a movement button is held and the player does
  // not hold it (the key that runs) - Shift alone holds nothing.
  carried(pressed) {
    const held = new Set(pressed);
    if (!this.canInspect()) held.delete(INSPECT_BUTTON);
    if (!this.walk) return held;
    if (MOVEMENT_BUTTONS.some((b) => pressed.has(b)) && !pressed.has(WALK_BUTTON)) held.add(WALK_BUTTON);
    else held.delete(WALK_BUTTON);
    return held;
  }

  commit(time) {
    const pressed = new Set([...this.holds.values()].flatMap((buttons) => [...buttons]));
    const shown = pressed.size !== this.pressed.size || [...pressed].some((b) => !this.pressed.has(b));
    this.pressed = pressed;
    const held = this.carried(pressed);
    const down = [...held].filter((b) => !this.held.has(b));
    const up = [...this.held].filter((b) => !held.has(b));
    this.held = held;
    if (!down.length && !up.length) {
      if (shown) this.changed();
      return false;
    }
    this.sample(time);
    this.edges(down, up, time, this.seq);
    this.changed();
    return true;
  }

  // Take weapon i of the loadout at `time`; false when there is none or it is held. The inspect button held goes with
  // the weapon: it holds from a knife on and lets go at a gun.
  select(i, time) {
    if (i < 0 || i >= this.loadout.length || i === this.weapon) return false;
    this.lastWeapon = this.weapon;
    this.weapon = i;
    this.held = this.carried(this.pressed);
    this.sample(time);
    this.changed();
    return true;
  }

  // Take the weapon `step` places on in the loadout (a negative step: before), round.
  cycle(step, time) {
    const n = this.loadout.length;
    return this.select((((this.weapon + step) % n) + n) % n, time);
  }

  // The view turns by (dp, dy) degrees at `time` (pitch > 0 looks down, yaw > 0 turns left): the next sample carries it.
  // A turn after a rest began at `began` (the key press behind it, if the page knows it: by default `time`).
  look(dp, dy, time, began = time) {
    if (!dp && !dy) return;
    if (time - this.lastLook > TURN_ONSET_GAP_MS) this.onset ??= began;
    this.lastLook = time;
    this.dp += dp;
    this.dy += dy;
  }

  // The rates (pitch, yaw) kept within the rate sources' caps: the yaw's maxRate, the pitch's PITCH_TURN_SHARE of it.
  capped(pitchRate, yawRate) {
    const [pitch, yaw] = [PITCH_TURN_SHARE * this.maxRate, this.maxRate];
    return [Math.max(-pitch, Math.min(pitch, pitchRate)), Math.max(-yaw, Math.min(yaw, yawRate))];
  }

  // The view turns at (pitchRate, yawRate) degrees a second for `dt` s (a held key, a stick), within the caps.
  lookAtRate(pitchRate, yawRate, dt, time, began = time) {
    const [pitch, yaw] = this.capped(pitchRate, yawRate);
    this.look(pitch * dt, yaw * dt, time, began);
  }

  // The mouse moved by (dx, dy) pixels at `time`: the view turns by as much (mouse right turns right, yaw going down; down
  // looks down), but for the first movement after the pointer lock took hold (lockTaken), the pointer's jump to the lock.
  turn(dx, dy, time) {
    if (this.skipMove) {
      this.skipMove = false;
      return;
    }
    this.look(dy * this.sens, -dx * this.sens, time);
  }

  // The pointer lock took hold: the browser may report the pointer's jump to it as the next movement.
  lockTaken() {
    this.skipMove = true;
  }

  // The turn made and not sent yet, degrees [pitch, yaw]: the next sample's (the instant turn shows it at once,
  // turn.js).
  unsent() {
    return [this.dp, this.dy];
  }
}
