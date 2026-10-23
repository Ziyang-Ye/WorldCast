// Playing with a gamepad (the browser's standard mapping, or the layout of an Xbox, DualShock 4 or DualSense pad that
// a Linux driver gives no standard mapping) and, on a touch device, with the on-screen pad: a
// left stick that moves, a right stick that looks and four buttons. Each feeds the player's input at every animation frame
// (update) from the sticks' deflection, the same way: the left stick holds movement buttons, the right stick asks for a
// turn at the rate input.js's response curve gives, up to the maximum turn speed.

import { stickHeld, stickLook } from './input.js';

// A standard-mapping gamepad's buttons, by index: A jump, B crouch, X reload, Y inspect, the triggers fire (right) and
// alt-fire (left); the bumpers take the weapon before and after the one held, the D-pad's up, right, down and left
// weapons 1-4. Axes: the left stick's x and y, the right stick's.
const GAMEPAD_BUTTONS = { 0: 'jump', 1: 'duck', 2: 'reload', 3: 'look_at_weapon', 7: 'attack', 6: 'attack2' };
const BUMPERS = { 4: -1, 5: 1 };
const DPAD_WEAPONS = { 12: 0, 15: 1, 13: 2, 14: 3 };
const [LEFT_X, LEFT_Y, RIGHT_X, RIGHT_Y] = [0, 1, 2, 3];

// A pad that has no standard mapping (Linux drivers give many none: mapping "") is read by the layout of its vendor and
// product, which Chrome names in the pad's id as "(Vendor: 045e Product: 028e)" and Firefox as "045e-028e-". Each layout lists
// where the pad keeps what the standard mapping has: the raw index of its buttons 0-5 (the face buttons A, B, X, Y and the
// bumpers, and where they are buttons, the triggers 6 and 7), its sticks' axes, its triggers' axes (they rest at -1 and read 1
// pulled) and its D-pad, [x, y] with right and down positive. Any other pad is not used.
const TRIGGER_PULLED = 0.5;      // a trigger axis beyond this counts as pressed; some drivers rest it at 0 until it first moves
const DPAD_PUSHED = 0.5;
// The D-pad of a DualShock 4 as one axis (a hat) reads -1 for up and rises by 2/7 through the eight directions clockwise to 1
// for up and left, and 9/7 at rest.
const HAT = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];
const hat = (value) => HAT[Math.round((value + 1) * 3.5)] ?? [0, 0];
const dpadAxes = (pad) => [pad.axes[6] ?? 0, pad.axes[7] ?? 0];
const RAW_LAYOUTS = {
  // Microsoft's Xbox 360, One and Series pads through the xpad driver: A, B, X, Y, LB, RB are buttons 0-5 (Back, Start, the
  // Xbox button, L3 and R3 are 6-10); the sticks are axes 0, 1 and 3, 4, the triggers axes 2 and 5, the D-pad axes 6 and 7.
  xbox: { name: 'Xbox', buttons: [0, 1, 2, 3, 4, 5], sticks: [0, 1, 3, 4], triggers: [2, 5], dpad: dpadAxes,
    products: ['028e', '028f', '0291', '02a1', '0719', '02d1', '02dd', '02e3', '02ea', '02fd', '0b00', '0b12', '0b13'] },
  // Sony's DualShock 4 and DualSense through hid-sony: Cross, Circle, Triangle, Square, L1, R1 are buttons 0-5 (Triangle and
  // Square are the standard mapping's Y and X: swapped), L2 and R2 buttons 6 and 7 and also axes 3 and 4; the sticks are axes 0,
  // 1 and 2, 5; the D-pad is a hat on axis 9, or axes 6 and 7 on a pad of eight axes.
  playstation: { name: 'PlayStation', buttons: [0, 1, 3, 2, 4, 5, 6, 7], sticks: [0, 1, 2, 5], triggers: [3, 4],
    dpad: (pad) => (pad.axes.length > 9 ? hat(pad.axes[9]) : dpadAxes(pad)),
    products: ['05c4', '09cc', '0ba0', '0ce6', '0df2'] },
};
const VENDORS = { '045e': RAW_LAYOUTS.xbox, '054c': RAW_LAYOUTS.playstation };

// The layout of a pad that has no standard mapping, if its vendor and product are known.
function rawLayout(pad) {
  const found = /Vendor: ([0-9a-f]{4}) Product: ([0-9a-f]{4})/i.exec(pad.id) ?? /^([0-9a-f]{4})-([0-9a-f]{4})-/i.exec(pad.id);
  const layout = found && VENDORS[found[1].toLowerCase()];
  return layout?.products.includes(found[2].toLowerCase()) ? layout : null;
}

// The pad as one of the standard mapping would read: its axes [left x, left y, right x, right y] and its buttons 0-15, the
// triggers and the D-pad (12 up, 13 down, 14 left, 15 right) among them.
function standardized(pad, layout) {
  const axis = (i) => pad.axes[i] ?? 0;
  const pressed = new Set();
  layout.buttons.forEach((raw, standard) => { if (pad.buttons[raw]?.pressed) pressed.add(standard); });
  layout.triggers.forEach((raw, i) => { if (axis(raw) > TRIGGER_PULLED) pressed.add(6 + i); });
  const [x, y] = layout.dpad(pad);
  if (y < -DPAD_PUSHED) pressed.add(12);
  if (y > DPAD_PUSHED) pressed.add(13);
  if (x < -DPAD_PUSHED) pressed.add(14);
  if (x > DPAD_PUSHED) pressed.add(15);
  return { axes: layout.sticks.map(axis), buttons: Array.from({ length: 16 }, (_, i) => ({ pressed: pressed.has(i) })) };
}

const padName = (pad) => pad.id.replace(/^[0-9a-f]{4}-[0-9a-f]{4}-/i, '').replace(/\s*\(.*$/, '').trim() || 'Gamepad';
// The pads the browser lists: a browser without the Gamepad API, or one that keeps it to secure pages (a page served over
// plain http from a host that is not localhost), lists none.
const connectedPads = () => [...(navigator.getGamepads?.() ?? [])].filter((pad) => pad && pad.connected);

export class GamepadInput {
  // input: the PlayerInput the pad feeds; say(text): a notice for the player.
  constructor(input, say) {
    Object.assign(this, { input, say });
    this.told = new Set();         // the pads the player was told of, by index: each is told once
    this.layouts = new Map();      // the pads in use, by index -> their raw layout (null: the standard mapping)
    this.down = new Set();         // the buttons down at the last poll, by index: the weapon buttons act when they go down
    this.connected = (e) => this.connect(e.gamepad);
    this.disconnected = (e) => this.disconnect(e.gamepad);
    window.addEventListener('gamepadconnected', this.connected);
    window.addEventListener('gamepaddisconnected', this.disconnected);
    for (const pad of connectedPads()) this.connect(pad);     // one that connected before this page
  }

  // A pad connected: said once, whether the page saw it connect or found it there.
  connect(pad) {
    if (this.told.has(pad.index)) return;
    this.told.add(pad.index);
    const standard = pad.mapping === 'standard';
    const layout = standard ? null : rawLayout(pad);
    if (standard || layout) this.layouts.set(pad.index, layout);
    if (standard) this.say(`${padName(pad)} connected`);
    else this.say(layout ? `${padName(pad)} connected (${layout.name} layout)` : `${padName(pad)} has no standard mapping: not used`);
  }

  disconnect(pad) {
    this.told.delete(pad.index);
    this.layouts.delete(pad.index);
    this.down.clear();
    this.input.hold('pad', [], performance.now());
    this.say(`${padName(pad)} disconnected`);
  }

  // At every animation frame (`dt` s since the last), while the player plays: the pad's buttons, the weapon buttons that
  // went down and the look stick's turn over `dt`. Nothing is polled while no pad in use is connected.
  update(now, dt) {
    if (!this.layouts.size) return;
    const found = connectedPads().find((p) => this.layouts.has(p.index));
    if (!found) return;
    const pad = this.layouts.get(found.index) ? standardized(found, this.layouts.get(found.index)) : found;
    const held = stickHeld(pad.axes[LEFT_X], pad.axes[LEFT_Y]);
    for (const [index, button] of Object.entries(GAMEPAD_BUTTONS)) if (pad.buttons[index]?.pressed) held.add(button);
    this.input.hold('pad', held, now);
    const down = new Set(pad.buttons.flatMap((b, i) => (b.pressed ? [i] : [])));
    for (const index of down) {
      if (this.down.has(index)) continue;
      if (index in BUMPERS) this.input.cycle(BUMPERS[index], now);
      if (index in DPAD_WEAPONS) this.input.select(DPAD_WEAPONS[index], now);
    }
    this.down = down;
    const [dp, dy] = stickLook(pad.axes[RIGHT_X], pad.axes[RIGHT_Y], this.input.maxRate);
    if (dp || dy) this.input.lookAtRate(dp, dy, dt, now);
  }

  // The player pauses: no weapon button counts as down.
  release() {
    this.down.clear();
  }

  close() {
    window.removeEventListener('gamepadconnected', this.connected);
    window.removeEventListener('gamepaddisconnected', this.disconnected);
  }
}

const icon = (paths) => `<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round">${paths}</svg>`;
const ICONS = {
  jump: icon('<path d="M12 19V6M6 11l6-6 6 6"/>'),
  duck: icon('<path d="M12 5v13M6 13l6 6 6-6"/>'),
  attack: icon('<circle cx="12" cy="12" r="6"/><path d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4"/>'),
  next: icon('<path d="M4 12a8 8 0 0 1 13.5-5.8L20 8M20 3v5h-5M20 12a8 8 0 0 1-13.5 5.8L4 16M4 21v-5h5"/>'),
};
// A button of the pad: a model button it holds (data-b), or `next`, the weapon after the one held; its caption is
// said under it.
const button = (kind, label, caption) => `<button class="pad-btn" type="button" ${kind === 'next' ? 'data-next' : `data-b="${kind}"`}
  aria-label="${label}"><span class="disc">${ICONS[kind]}</span><span class="pad-cap">${caption}</span></button>`;
const stick = (name, caption) => `<div class="stick" data-stick="${name}"><div class="stick-base">
  <svg viewBox="0 0 100 100" aria-hidden="true"><path d="M50 9l7 9H43zM50 91l7-9H43zM9 50l9-7v14zM91 50l-9-7v14z"/></svg><i class="stick-knob"></i></div>
  <span class="pad-cap">${caption}</span></div>`;

// The on-screen pad's markup: movement, jump and crouch on the left; look, fire and the next weapon on the right.
export const padMarkup = () => `<div class="pad" id="pad" aria-label="On-screen controls">
  <div class="pad-side pad-left"><div class="pad-buttons">${button('jump', 'Jump', 'Jump')}${button('duck', 'Crouch', 'Crouch')}</div>
    ${stick('move', 'Move')}</div>
  <div class="pad-side pad-right"><div class="pad-buttons">${button('attack', 'Fire', 'Fire')}${button('next', 'Next weapon', 'Weapon')}</div>
    ${stick('look', 'Look')}</div></div>`;

export class TouchPad {
  // root: the pad's element (padMarkup); input: the PlayerInput it feeds.
  constructor(root, input) {
    Object.assign(this, { root, input });
    this.fingers = new Map();      // button element -> the pointers down on it
    this.sticks = [...root.querySelectorAll('.stick')].map((el) => this.stick(el));
    for (const el of root.querySelectorAll('.pad-btn')) this.button(el);
    root.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  // A stick: its finger (the pointer that went down on it, captured until it lifts), where that finger went down (the
  // stick's origin: it works from where the thumb lands, so that a thumb that lands off the middle turns nothing until it
  // moves), and its deflection (x, y) over its travel from there, each -1 to 1 (right and down are positive).
  stick(el) {
    const stick = { name: el.dataset.stick, base: el.querySelector('.stick-base'), knob: el.querySelector('.stick-knob'),
      pointer: null, origin: [0, 0], x: 0, y: 0 };
    stick.base.addEventListener('pointerdown', (e) => {
      if (stick.pointer !== null) return;
      e.preventDefault();
      stick.pointer = e.pointerId;
      stick.origin = [e.clientX, e.clientY];
      stick.base.setPointerCapture(e.pointerId);
      stick.base.classList.add('on');
      this.drag(stick, e);
    });
    stick.base.addEventListener('pointermove', (e) => { if (e.pointerId === stick.pointer) this.drag(stick, e); });
    for (const type of ['pointerup', 'pointercancel']) {
      stick.base.addEventListener(type, (e) => { if (e.pointerId === stick.pointer) this.rest(stick); });
    }
    return stick;
  }

  // The finger at e: the stick's deflection over its travel (the base's radius less the knob's) from the origin, at most
  // 1 whichever way; the knob is drawn that far from the middle of the base.
  drag(stick, e) {
    const travel = (stick.base.getBoundingClientRect().width - stick.knob.getBoundingClientRect().width) / 2;
    let x = (e.clientX - stick.origin[0]) / travel;
    let y = (e.clientY - stick.origin[1]) / travel;
    const deflection = Math.hypot(x, y);
    if (deflection > 1) [x, y] = [x / deflection, y / deflection];
    Object.assign(stick, { x, y });
    stick.knob.style.transform = `translate(${x * travel}px, ${y * travel}px)`;
  }

  // The finger lifts: the stick is at rest.
  rest(stick) {
    Object.assign(stick, { pointer: null, x: 0, y: 0 });
    stick.knob.style.transform = '';
    stick.base.classList.remove('on');
  }

  // A button: down while a finger is on it (the pointer is captured, so a finger that slides off still lifts it; with two
  // fingers on it, it is up when the last lifts); the weapon button takes the next weapon whenever a finger goes down.
  button(el) {
    const next = 'next' in el.dataset;
    const fingers = new Set();
    this.fingers.set(el, fingers);
    el.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      fingers.add(e.pointerId);
      el.classList.add('on');
      if (next) this.input.cycle(1, e.timeStamp);
      else if (fingers.size === 1) this.input.press(el.dataset.b, true, e.timeStamp, 'touch');
    });
    const up = (e) => {
      fingers.delete(e.pointerId);
      if (fingers.size) return;
      el.classList.remove('on');
      if (!next) this.input.press(el.dataset.b, false, e.timeStamp, 'touch');
    };
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
  }

  // At every animation frame (`dt` s since the last), while the player plays: the left stick's movement buttons and the
  // right stick's turn over `dt`.
  update(now, dt) {
    for (const stick of this.sticks) {
      if (stick.name === 'move') {
        this.input.hold('stick', stickHeld(stick.x, stick.y), now);
      } else {
        const [dp, dy] = stickLook(stick.x, stick.y, this.input.maxRate);
        if (dp || dy) this.input.lookAtRate(dp, dy, dt, now);
      }
    }
  }

  // The player pauses: the sticks are at rest and no button is down.
  release() {
    for (const stick of this.sticks) this.rest(stick);
    for (const el of this.root.querySelectorAll('.pad-btn')) el.classList.remove('on');
    for (const fingers of this.fingers.values()) fingers.clear();
  }
}
