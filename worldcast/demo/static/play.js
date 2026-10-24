// The play view: the generated view at its own size, each frame drawn as it was generated (turned at once by the turn
// it does not contain yet, as far as the frames shown reach, while the instant turn is on, which it is not by default;
// scaled and shifted by the move it does not contain yet only while the player has the motion feedback on), keyboard,
// mouse, gamepad and, on a touch device, the on-screen pad in (the mouse looks unless the pause panel's Mouse look is
// off, and turns the view as far as it moves; the arrow keys and the sticks turn at a rate), and around the view, never
// over it, the HUD (who and the clock, fps, input-to-photon latency, network), the key caps, the pad, the weapons and
// the players; the latency breakdown, which says how the own position is known, opens from the HUD's numbers. Over the
// view while one plays: the crosshair and the field's markers (M).

import { $, clock, esc, joined, page, playerColor, store, weaponLabel, wsUrl } from './common.js';
import {
  ARROWS, INSPECT_BUTTON, MAX_FRAME_STEP_S, MAX_TURN_DEGREES_PER_SECOND, MAX_TURN_RANGE, MAX_TURN_STEP, PlayerInput,
  TURN_DEGREES_PER_SECOND, TURN_SPEED_RANGE, TURN_SPEED_STEP, arrowLook,
} from './input.js';
import { FEEDBACK_DEFAULT, FEEDBACK_SHARE, MotionFeedback } from './feedback.js';
import { LatencyMeter } from './latency.js';
import { GamepadInput, TouchPad, padMarkup } from './pad.js';
import { FrameStream } from './stream.js';
import { INSTANT_TURN_DEFAULT } from './turn.js';

const KEYS = {
  KeyW: 'forward', KeyS: 'back', KeyA: 'move_left', KeyD: 'move_right', Space: 'jump', ControlLeft: 'duck',
  KeyC: 'duck', ShiftLeft: 'speed', KeyR: 'reload', KeyF: 'look_at_weapon',
};
const MOUSE = { 0: 'attack', 2: 'attack2' };
const WEAPON_KEYS = /^Digit[1-4]$/;
const CAPS = [['forward', 'W'], ['move_left', 'A'], ['back', 'S'], ['move_right', 'D']];
// The row's caps: [button, key, class, title]. F inspects with a knife only (input.js INSPECT_BUTTON): its cap and its
// line of the legend are greyed while another weapon is held.
const INSPECT_SAYS = 'F — inspect (knife)';
const CAPS_ROW = [['jump', 'Space', 'cap-w'], ['duck', 'Ctrl'], ['speed', 'Shift'], ['attack', 'LMB'], ['attack2', 'RMB'],
  ['reload', 'R'], [INSPECT_BUTTON, 'F', '', INSPECT_SAYS]];
const ARROW_CAPS = [['ArrowUp', '↑'], ['ArrowLeft', '←'], ['ArrowDown', '↓'], ['ArrowRight', '→']];

const crosshair = `<svg class="crosshair" viewBox="0 0 24 24" aria-hidden="true"><g stroke="#fff" stroke-width="2"
  stroke-linecap="round"><path d="M12 3v5M12 16v5M3 12h5M16 12h5"/></g></svg>`;
// A field marker is a ring at the player's feet with its label above, an outline that shows on any picture: its
// horizontal radius in CSS pixels is MARKER_SCALE over the player's depth (u), within MARKER_RADIUS, so that the ring is
// as wide as the figure up close (60 px across at most) and never narrower than 2 * MARKER_RADIUS[0] however far the
// player is; its vertical radius is MARKER_FLAT of the horizontal one, at least MARKER_MIN_HEIGHT.
const MARKER_SCALE = 4000;
const MARKER_RADIUS = [12, 30];
const MARKER_FLAT = 0.45;
const MARKER_MIN_HEIGHT = 7;
// The window height the pause panel needs on one side of the frame rate and latency, its margins included: its title,
// its Play button and a part of its list. CARD_MARGIN_PX lies between the panel and them.
const PANEL_ROOM_PX = 380;
const CARD_MARGIN_PX = 12;
const UNDRAWN_ALPHA = 0.45;    // a marker of a player the field does not draw
// A player state row of a frame header is [slot, t, x, y, z, yaw, pitch, alive, vx, vy, src, drawn] (clients.py
// LivePlayerState.wire); drawn is 1 where the field draws the player, 0 where it does not, null where unknown.
const SLOT = 0;
const DRAWN = 11;

const pauseIcon = `<svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/>
  <rect x="14" y="5" width="4" height="14" rx="1"/></svg>`;

// The view's markup. The stage holds the view and nothing else but the crosshair and the markers; the HUD sits in the
// grid around its bezel, the on-screen pad of a touch device outside it, and the pause and start panels (the overlay)
// over the whole page, hidden while one plays.
function playMarkup(join, color, touch) {
  const [width, height] = page.frame_pixels;
  return `
  <div class="play ${touch ? 'touch' : ''}" style="--acc:${color};--frame-w:${width}px;--frame-ratio:${width / height}">
    <div class="backdrop" aria-hidden="true"><i class="reg tr"></i><i class="reg bl"></i><i class="reg br"></i>
      <span class="mark"><img src="/static/favicon.svg" alt="">WorldCast <b>Live</b></span></div>
    <header class="bar">
      <div class="who plate"><span class="dot"></span><b class="who-name">${esc(join.name)}</b>
        <span class="tag">P${join.seat.slot + 1}</span>
        <span class="sub where">${esc(join.round_start.map_label)} · ${esc(join.round_start.label)}</span>
        <span class="sub mono" id="clock">0:00:00</span>
        ${touch ? `<button class="pause-btn" id="pause" type="button" aria-label="Pause">${pauseIcon}</button>` : ''}</div>
      <button class="stats plate" id="stats" type="button" title="Latency breakdown" aria-expanded="false">
        <span class="stat"><b id="fps">0</b><small>fps</small></span>
        <span class="stat"><b id="lat">—</b><small>${touch ? 'ms end to end' : 'ms input→photon'}</small></span>
        <span class="bars" id="bars" data-q="2"><i></i><i></i><i></i></span><i class="more" aria-hidden="true"></i></button>
    </header>
    <div class="deck">
      <div class="bezel">
        <i class="bracket tl" aria-hidden="true"></i><i class="bracket tr" aria-hidden="true"></i>
        <i class="bracket bl" aria-hidden="true"></i><i class="bracket br" aria-hidden="true"></i>
        <i class="ruler ruler-l" aria-hidden="true"></i><i class="ruler ruler-r" aria-hidden="true"></i>
        <div class="stage" id="stage">
          <canvas class="view" id="view"></canvas><canvas class="view marks" id="marks" hidden></canvas>${crosshair}
        </div>
      </div>
    </div>
    ${touch ? padMarkup() : ''}
    <footer class="below">
      <div class="under">
        <span class="caption mono" title="The generated frame's size in pixels">${width} × ${height}</span>
        <div class="toast" id="toast" role="status"></div>
        <span class="hint">Esc pauses · M marks the players</span>
      </div>
      <div class="keys plate" id="caps">
        <div class="kg kg-wasd">${CAPS.map(([b, k]) => `<span class="cap" data-b="${b}">${k}</span>`).join('')}</div>
        <div class="kg kg-wasd">${ARROW_CAPS.map(([k, label]) => `<span class="cap" data-k="${k}">${label}</span>`).join('')}</div>
        <div class="kg kg-row">${CAPS_ROW.map(([b, k, c, title]) => `<span class="cap ${c || ''}" data-b="${b}"${title
          ? ` title="${title}"` : ''}>${k}</span>`).join('')}</div>
      </div>
    </footer>
    <aside class="flank flank-left">
      <section class="plate"><h3 class="label">Players</h3><ul class="roster" id="roster"></ul>
        <p class="alone" id="alone" hidden>Others appear in your view when they take a seat in this room.</p></section>
    </aside>
    <aside class="flank flank-right">
      <section class="plate weapons-card"><h3 class="label">Loadout</h3><div class="weapons" id="weapons"></div></section>
    </aside>
    <div class="lat-panel plate" id="lat-panel" hidden></div>
    <div class="overlay" id="overlay"></div>
  </div>`;
}

// What runs the player's client, for the page's sentences: a mock worker has no GPU and plays pre-rendered frames, and
// the page says so.
const runnerKind = (worker) => (worker.client === 'mock' ? 'mock client' : 'GPU worker');
const runner = (worker) => `${runnerKind(worker)} <b>${esc(worker.id)}</b>`;
const source = (worker) => (worker.client === 'mock'
  ? `The ${runner(worker)} plays pre-rendered frames: no GPU generates your view.`
  : `Your view is generated live on ${runner(worker)}.`);

// What Shift does and what the Walk by default setting says, for its row of the panel, the legend and the key cap.
const shiftSays = (walk) => (walk ? 'run' : 'walk');
const walkSays = (walk) => (walk ? 'You walk while a movement key is held; Shift runs' : 'You run; Shift walks');

const caps = (...texts) => texts.map((text) => `<span class="cap ${text === 'Space' ? 'cap-w' : ''}">${text}</span>`).join('');

// The pause overlay's markup: the controls and the settings. A touch device is told how the on-screen pad plays, and has no
// settings; the others have the mouse look switch (off for a browser that cannot lock the pointer), the mouse
// sensitivity, the turn speed of the arrow keys, the maximum turn speed of the keys and the sticks, the instant turn and
// the motion feedback.
// inspect: whether the weapon held inspects (F's line is greyed if not).
function pauseMarkup(worker, settings, sens, touch, canLock, inspect) {
  const greyed = inspect ? '' : ' unavailable';
  const keys = touch ? `
      <dt>${caps('Left stick')}</dt><dd>Move</dd>
      <dt>${caps('Right stick')}</dt><dd>Look</dd>
      <dt>${caps('Buttons')}</dt><dd>Jump, crouch, fire, next weapon</dd>` : `
      <dt>${caps('W', 'A', 'S', 'D')}</dt><dd>Move</dd>
      <dt>${caps('←', '↑', '↓', '→')}</dt><dd>Turn and look up or down</dd>
      <dt>${caps('Mouse')}</dt><dd>Turns the view while Mouse look is on; fire (left), alt-fire (right)</dd>
      <dt>${caps('Space', 'C', 'Shift')}</dt><dd>Jump, crouch, <span id="shift-says">${shiftSays(settings.walk)}</span></dd>
      <dt>${caps('1', '2', '3', '4', 'Q')}</dt><dd>Weapons, last weapon</dd>
      <dt>${caps('R', 'Esc')}</dt><dd>Reload, pause</dd>
      <dt class="inspect${greyed}" title="${INSPECT_SAYS}">${caps('F')}</dt><dd class="inspect${greyed}">Inspect (knife)</dd>
      <dt>${caps('M')}</dt><dd>Rings on the view where the model places the other players</dd>`;
  const [low, high] = TURN_SPEED_RANGE;
  const [maxLow, maxHigh] = MAX_TURN_RANGE;
  const sliders = touch ? '' : `
    <div class="settings">
      <label class="switch-row"><span>Mouse look<small>${canLock ? 'The mouse is captured and turns the view; Esc lets it go'
        : 'This browser cannot capture the mouse'}</small></span>
        <input id="mouselook" type="checkbox" role="switch" ${settings.mouseLook ? 'checked' : ''} ${canLock ? '' : 'disabled'}><i class="switch" aria-hidden="true"></i></label>
      <label class="slider"><span>Mouse</span><input id="sens" type="range" min="0.01" max="0.15" step="0.005" value="${sens}"
        ${settings.mouseLook ? '' : 'disabled'}><output id="sens-v">${sens.toFixed(3)}°</output></label>
      <label class="slider"><span>Turn speed<small>arrow keys</small></span><input id="turn" type="range" min="${low}" max="${high}" step="${TURN_SPEED_STEP}"
        value="${settings.turn}"><output id="turn-v">${settings.turn}°/s</output></label>
      <label class="slider"><span>Max turn speed<small>keys and sticks</small></span><input id="maxturn" type="range" min="${maxLow}" max="${maxHigh}" step="${MAX_TURN_STEP}"
        value="${settings.maxTurn}"><output id="maxturn-v">${settings.maxTurn}°/s</output></label>
      <label class="switch-row"><span>Walk by default<small id="walk-says">${walkSays(settings.walk)}</small></span>
        <input id="walk" type="checkbox" role="switch" ${settings.walk ? 'checked' : ''}><i class="switch" aria-hidden="true"></i></label>
      <label class="switch-row"><span>Instant turn<small>The view turns at once as far as what it has seen reaches</small></span>
        <input id="instant" type="checkbox" role="switch" ${settings.instant ? 'checked' : ''}><i class="switch" aria-hidden="true"></i></label>
      <div class="choice" role="radiogroup" aria-labelledby="motionfb-l"><span id="motionfb-l">Motion feedback<small>movement keys</small></span>
        <div class="segments">${Object.keys(FEEDBACK_SHARE).map((value) => `<label><input type="radio" name="motionfb" value="${value}"
          ${value === settings.motion ? 'checked' : ''}><span>${value[0].toUpperCase() + value.slice(1)}</span></label>`).join('')}</div></div>
    </div>`;
  return `<div class="panel panel-pause"><h2>${touch ? 'Tap' : 'Click'} to play</h2>
    <div class="panel-scroll"><p>${source(worker)}</p>
    <dl class="legend">${keys}
      <dt>${caps('Gamepad')}</dt><dd>Sticks move and look; A jump, B crouch, X reload, Y inspect (knife); triggers fire; bumpers and
        <span class="nb">D-pad</span> take weapons</dd>
    </dl>${sliders}</div>
    <div class="row"><button class="pill pill-light" id="go" type="button">Play</button>
      <button class="pill" id="leave" type="button">Leave seat</button></div></div>`;
}

export function renderPlay(app, roomId) {
  const join = joined(roomId);
  if (!join) { location.hash = `#/room/${roomId}`; return () => {}; }
  document.body.classList.add('is-play');
  const seat = join.seat;
  const touch = matchMedia('(pointer: coarse)').matches && !matchMedia('(any-pointer: fine)').matches;   // not a laptop with a touch screen
  app.innerHTML = playMarkup(join, playerColor(seat.slot), touch);

  const play = $('.play');
  const stage = $('#stage');
  const overlay = $('#overlay');
  // Shown at its own size on a display of a whole number of device pixels per CSS pixel, the frame is drawn pixel for
  // pixel (app.css .native): the browser's smoothing would only blur it.
  const crisp = () => stage.classList.toggle('native',
    Math.abs(stage.getBoundingClientRect().width - page.frame_pixels[0]) < 0.01 && Number.isInteger(window.devicePixelRatio));
  crisp();
  window.addEventListener('resize', crisp);
  const state = {
    engaged: false, status: 'starting', roster: new Map(), frames: 0, framesPerBlock: '?', markers: false,
    unlocked: false,       // the browser refused the pointer lock: the keyboard plays until a click on the view asks again
    position: null,        // how the own position is known (the newest frame's st), for the latency breakdown
    shown: null,           // the frame on screen: its header h, and drawn, whether the field draws each of its players
  };
  const meter = new LatencyMeter();
  // What the pause panel keeps in localStorage (wc.arrowturn, wc.turncap, wc.mouselook, wc.motion, wc.walk,
  // wc.instantturn; wc.sens is the mouse's): the yaw rate of the arrow keys and the most the engine is asked to turn
  // the yaw, each within its range, whether the mouse looks, how much the view answers a movement key before the frames
  // do (Off, Light or Strong), whether a movement key walks, and whether the view turns at once.
  // The mouse looks only with a mouse and a browser that can lock the pointer (iPadOS Safari cannot): a touch device has no
  // switch. Where the pointer is a fine one, the mouse looks until the player turns it off.
  const within = (value, [low, high]) => Math.min(Math.max(value, low), high);
  const canLock = 'requestPointerLock' in Element.prototype;
  const settings = {
    turn: within(Number(store('wc.arrowturn')) || TURN_DEGREES_PER_SECOND, TURN_SPEED_RANGE),
    maxTurn: within(Number(store('wc.turncap')) || MAX_TURN_DEGREES_PER_SECOND, MAX_TURN_RANGE),
    mouseLook: !touch && canLock && (store('wc.mouselook') ?? matchMedia('(pointer: fine)').matches),
    motion: Object.hasOwn(FEEDBACK_SHARE, store('wc.motion')) ? store('wc.motion') : FEEDBACK_DEFAULT,
    walk: store('wc.walk') !== false,
    instant: store('wc.instantturn') ?? INSTANT_TURN_DEFAULT,       // the view turns at once (turn.js)
  };
  const feedback = new MotionFeedback(settings.motion);
  const input = new PlayerInput(seat.loadout, store('wc.sens'), (seq, onsets, sample) => {
    meter.sample(seq, performance.now(), onsets);
    stream.send(sample);
  }, repaint, (...edges) => feedback.buttons(...edges));
  const stream = new FrameStream(wsUrl(join.play_url), $('#view'), {
    unsent: () => input.unsent(), instant: settings.instant, moved: (now) => feedback.at(now), onFrame, onText, onClose,
  });
  const arrows = new Map();       // the arrow keys held -> when each went down
  const touchPad = touch ? new TouchPad($('#pad'), input) : null;
  let toastTimer;
  function toast(text) {
    const el = $('#toast');
    el.textContent = text;
    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('show'), 2600);
  }
  input.maxRate = settings.maxTurn;
  input.walk = settings.walk;
  $('.cap[data-b=speed]').title = `Shift ${shiftSays(settings.walk)}s`;
  const gamepad = new GamepadInput(input, toast);

  // A hidden page still acknowledges every frame, so the worker is told when it is hidden: it releases a seat whose
  // page stays hidden (play.away_s).
  const sendVisibility = () => stream.send({ t: 'vis', h: document.hidden });
  document.addEventListener('visibilitychange', sendVisibility);

  // ------------------------------------------------------------------------------------------- overlays
  function showOverlay(kind, message = '') {
    state.status = kind;
    play.classList.toggle('ended', kind === 'ended');
    play.classList.toggle('starting', kind === 'starting');
    if (kind === 'none') { overlay.hidden = true; return; }
    overlay.hidden = false;
    if (kind === 'starting') {
      overlay.innerHTML = `<div class="panel"><div class="spinner"></div><h2>Starting your client</h2>
        <p>Seat ${seat.slot + 1} on ${runner(join.worker)}. ${esc(message)}</p></div>`;
    } else if (kind === 'ended') {
      openPanel(false);
      overlay.innerHTML = `<div class="panel"><h2>Session ended</h2><p>${esc(message || 'The connection closed.')}</p>
        <div class="row"><a class="pill pill-light" href="#/">Back to the lobby</a>
          <a class="pill" href="#/room/${esc(roomId)}">Take a seat again</a></div></div>`;
    } else {
      overlay.innerHTML = pauseMarkup(join.worker, settings, input.sens, touch, canLock, input.canInspect());
      $('#go').onclick = engage;
      $('#leave').onclick = leave;
      $('.panel-scroll').onscroll = hintScroll;
      if (!touch) bindSettings();
    }
    placeCard();
  }

  function bindSettings() {
    $('#mouselook').onchange = (e) => {
      settings.mouseLook = e.target.checked;
      $('#sens').disabled = !settings.mouseLook;
      store('wc.mouselook', settings.mouseLook);
    };
    $('#sens').oninput = (e) => {
      input.sens = Number(e.target.value);
      $('#sens-v').textContent = `${input.sens.toFixed(3)}°`;
      store('wc.sens', input.sens);
    };
    $('#turn').oninput = (e) => {
      settings.turn = Number(e.target.value);
      $('#turn-v').textContent = `${settings.turn}°/s`;
      store('wc.arrowturn', settings.turn);
    };
    $('#maxturn').oninput = (e) => {
      settings.maxTurn = Number(e.target.value);
      input.maxRate = settings.maxTurn;
      $('#maxturn-v').textContent = `${settings.maxTurn}°/s`;
      store('wc.turncap', settings.maxTurn);
    };
    $('#walk').onchange = (e) => {
      settings.walk = e.target.checked;
      input.walk = settings.walk;
      $('#walk-says').textContent = walkSays(settings.walk);
      $('#shift-says').textContent = shiftSays(settings.walk);
      $('.cap[data-b=speed]').title = `Shift ${shiftSays(settings.walk)}s`;
      store('wc.walk', settings.walk);
    };
    $('#instant').onchange = (e) => {
      settings.instant = stream.instant = e.target.checked;
      store('wc.instantturn', settings.instant);
    };
    for (const radio of document.querySelectorAll('input[name=motionfb]')) {
      radio.onchange = (e) => {
        settings.motion = e.target.value;
        feedback.set(settings.motion);
        store('wc.motion', settings.motion);
      };
    }
  }

  // The frame rate and latency stay above the pause panel (the mouse is free then), so that the latency breakdown opens
  // from them: the panel sits under them, or over them where the window has more room there. A window with room for the
  // panel (PANEL_ROOM_PX) on neither side gives the panel the whole window and the frame rate and latency go under it
  // (the play view's `cramped`): the panel's list scrolls in a window too low for it, never its Play button.
  function placeCard() {
    // A hidden overlay keeps its padding, so that a panel shown again in the same window is at once where it was: for a
    // reader who asks for less motion every change is a transition (app.css), which shows the old padding for a frame.
    if (overlay.hidden) return;
    const { top, bottom } = $('#stats').getBoundingClientRect();
    const beside = state.status === 'pause' && !touch;
    const under = beside && innerHeight - bottom >= PANEL_ROOM_PX;
    const over = beside && !under && top >= PANEL_ROOM_PX;
    // Scrolled out of the window, they leave the panel at its edge: a negative padding is refused, the old one stays.
    overlay.style.paddingTop = under ? `${Math.max(0, Math.ceil(bottom)) + CARD_MARGIN_PX}px` : '';
    overlay.style.paddingBottom = over ? `${Math.max(0, Math.ceil(innerHeight - top)) + CARD_MARGIN_PX}px` : '';
    play.classList.toggle('cramped', beside && !under && !over);
    hintScroll();
  }

  // A panel body with more under its lower edge fades there, so that a list that scrolls says so.
  function hintScroll() {
    const body = $('.panel-scroll');
    if (body) body.classList.toggle('more', body.scrollHeight - body.scrollTop - body.clientHeight > 2);
  }
  showOverlay('starting', 'Loading the round start…');

  // ---------------------------------------------------------------------------------------- messages
  function onText(m) {
    if (m.t === 'welcome') {
      input.welcome(m);
      state.framesPerBlock = m.frames_per_block;
      renderWeapons();
      repaint();                                   // F's cap greyed unless the weapon inspects
      for (const p of m.roster) state.roster.set(p.slot, p);
      renderRoster();
      stream.send({ t: 'markers', on: state.markers });   // a new socket of the session: as the page has them
      sendVisibility();
    } else if (m.t === 'ended') {
      disengage();
      showOverlay('ended', m.message);
    } else if (m.t === 'roster') {
      const now = new Map(m.roster.map((p) => [p.slot, p]));
      for (const [s, p] of now) if (!state.roster.has(s) && s !== seat.slot) toast(`${p.name} joined seat ${s + 1}`);
      for (const [s, p] of state.roster) if (!now.has(s)) toast(`${p.name} left`);
      state.roster = now;
      renderRoster();
    }
  }

  function onClose() {
    if (state.status !== 'ended') { disengage(); showOverlay('ended', `Lost the connection to the ${runnerKind(join.worker)}.`); }
  }

  // The frame's rows of the player state field: the HUD shows the room time of the own row, the latency breakdown how it
  // is known, and the view the field's markers while M has them on, dimmed where the field does not draw the player.
  function onFrame(h, t) {
    state.frames += 1;
    if (state.status === 'starting') showOverlay('pause');
    $('#clock').textContent = clock(h.s[1]);                 // the room time of the player's own state
    state.position = h.st;
    state.shown = { h, drawn: new Map(h.p.map((row) => [row[SLOT], row[DRAWN]])) };
    meter.frame(h, t);
    feedback.arrived(h.q, t.shown);
  }

  // At every animation frame, a ring and a label at the feet of every other player in front of the camera, where the
  // field puts it (h.m: [slot, u, v, depth], u and v fractions of the frame) and the instant turn and the motion feedback
  // move it, dimmed where the field does not draw it. The ring is an outline of the player's colour on a dark one
  // (MARKER_RADIUS): it shows at any distance.
  function drawMarkers() {
    const canvas = $('#marks');
    canvas.hidden = !state.markers;
    if (!state.markers || !state.shown) return;
    const { h, drawn } = state.shown;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    const [width, height] = [Math.round(rect.width * dpr), Math.round(rect.height * dpr)];
    if (canvas.width !== width || canvas.height !== height) Object.assign(canvas, { width, height });
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);
    ctx.font = `700 ${12 * dpr}px Inter, system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.lineJoin = 'round';
    for (const [slot, u, v, depth] of h.m || []) {
      const at = stream.place(u, v);
      if (!at) continue;
      const [x, y] = [at[0] * width, at[1] * height];
      const rx = Math.min(MARKER_RADIUS[1], Math.max(MARKER_RADIUS[0], MARKER_SCALE / depth)) * dpr;
      const ry = Math.max(MARKER_MIN_HEIGHT * dpr, rx * MARKER_FLAT);
      const label = `P${slot + 1}`;
      ctx.globalAlpha = drawn.get(slot) === 0 ? UNDRAWN_ALPHA : 1;
      ctx.beginPath();
      ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(0, 0, 0, .7)';
      ctx.lineWidth = 5 * dpr;
      ctx.stroke();
      ctx.strokeStyle = ctx.fillStyle = playerColor(slot);
      ctx.lineWidth = 2.5 * dpr;
      ctx.stroke();
      ctx.strokeStyle = 'rgba(0, 0, 0, .75)';
      ctx.lineWidth = 3.5 * dpr;
      ctx.strokeText(label, x, y - ry - 5 * dpr);
      ctx.fillText(label, x, y - ry - 5 * dpr);
    }
    ctx.globalAlpha = 1;
  }

  function toggleMarkers() {
    state.markers = !state.markers;
    stream.send({ t: 'markers', on: state.markers });
    toast(state.markers ? 'Field markers on: where the model places the other players' : 'Field markers off');
  }

  // ------------------------------------------------------------------------------------------- input
  // At every animation frame while one plays: the turn the held arrow keys and the sticks make over the time since the
  // frame before (a frame that stalled counts as MAX_FRAME_STEP_S at most) joins the mouse's turn since, then the
  // sample (at most SAMPLE_RATE a second: a display faster than that sends one at some of its frames, with the turn of
  // the frames between).
  let lastFrame = performance.now();
  let raf = requestAnimationFrame(function frame(now) {
    raf = requestAnimationFrame(frame);
    const previous = lastFrame;
    const step = Math.min((now - previous) / 1000, MAX_FRAME_STEP_S);
    lastFrame = now;
    if (state.engaged) {
      const [dp, dy] = arrowLook(arrows.keys(), settings.turn);
      if (dp || dy) {
        const pressed = Math.max(...arrows.values());     // a turn that starts with a key press began with the press
        input.lookAtRate(dp, dy, step, now, pressed > previous ? pressed : now);
      }
      gamepad.update(now, step);
      touchPad?.update(now, step);
    }
    if (state.engaged || touch) input.tick(step);    // a phone sends idle samples: its end-to-end delay stays measurable
    drawMarkers();
  });

  // The key caps, the pad's weapon and the loadout show what the input holds, whichever source changed it; F's cap and line
  // of the legend are greyed, and its cap does not light, while the weapon held does not inspect.
  let shownWeapon = -1;
  function repaint() {
    const inspect = input.canInspect();
    for (const cap of document.querySelectorAll('.cap[data-b]')) {
      cap.classList.toggle('on', input.pressed.has(cap.dataset.b) && (inspect || cap.dataset.b !== INSPECT_BUTTON));
    }
    for (const el of document.querySelectorAll(`.cap[data-b=${INSPECT_BUTTON}], .legend .inspect`)) {
      el.classList.toggle('unavailable', !inspect);
    }
    for (const cap of document.querySelectorAll('.cap[data-k]')) cap.classList.toggle('on', arrows.has(cap.dataset.k));
    if (shownWeapon !== input.weapon) renderWeapons();
  }

  function renderWeapons() {
    shownWeapon = input.weapon;
    $('#weapons').innerHTML = input.loadout.map((w, i) =>
      `<div class="weapon ${i === input.weapon ? 'on' : ''}"><b>${i + 1}</b>${esc(weaponLabel(w))}</div>`).join('');
    const next = $('[data-next] .pad-cap');
    if (next) next.textContent = weaponLabel(input.loadout[input.weapon]);
  }

  // Every key the page binds is taken, so that Ctrl+1-4 (held to crouch) do not switch tabs. macOS sends no keyup for a key
  // pressed or held while Cmd is down: Cmd lets go of the keys, and a key pressed with Cmd (a browser shortcut) holds nothing.
  const onKey = (e) => {
    if (!state.engaged) return;
    const down = e.type === 'keydown';
    if (e.code === 'MetaLeft' || e.code === 'MetaRight') {
      if (down) {
        arrows.clear();
        input.hold('keys', [], e.timeStamp);
        repaint();
      }
      return;
    }
    if (e.code in ARROWS) {
      e.preventDefault();
      if (!down) arrows.delete(e.code);
      else if (!arrows.has(e.code) && !e.metaKey) arrows.set(e.code, e.timeStamp);
      repaint();
      return;
    }
    if (KEYS[e.code]) {
      e.preventDefault();
      if (!e.repeat && !(down && e.metaKey)) input.press(KEYS[e.code], down, e.timeStamp);
      return;
    }
    if (WEAPON_KEYS.test(e.code) || e.code === 'KeyQ' || e.code === 'KeyM') e.preventDefault();
    if (down && WEAPON_KEYS.test(e.code)) input.select(Number(e.code.slice(5)) - 1, e.timeStamp);
    if (down && e.code === 'KeyQ') input.select(input.lastWeapon, e.timeStamp);
    if (down && !e.repeat && e.code === 'KeyM') toggleMarkers();
    if (down && e.code === 'Escape' && !document.pointerLockElement) disengage();
  };
  // A touch device has the pad's Fire: the mouse events a tap on the view brings along fire nothing.
  const onMouseButton = (e) => {
    if (!state.engaged || touch || !(e.button in MOUSE)) return;
    e.preventDefault();
    input.press(MOUSE[e.button], e.type === 'mousedown', e.timeStamp, 'mouse');
  };
  // The middle button (autoscroll, a paste on X11) and the back and forward buttons of a mouse (they leave the view and drop
  // the seat) do nothing while one plays.
  const onSideButton = (e) => { if (state.engaged && !(e.button in MOUSE)) e.preventDefault(); };
  const onMove = (e) => {
    if (state.engaged && settings.mouseLook && !state.unlocked) input.turn(e.movementX, e.movementY, e.timeStamp);
  };
  // A lock that takes hold may report the pointer's jump to it as a movement: the input drops the next one.
  const onLock = () => {
    if (document.pointerLockElement) input.lockTaken();
    else if (state.engaged) disengage();
  };
  const onLockError = () => { if (eventRefusal) lockRefused(); };
  // A window that loses the focus gets the keys' releases: the player stands, with nothing held and the view not turning,
  // and the view stays uncovered (another window of the same room is where its person plays); a click on it plays again.
  const onBlur = () => { if (state.engaged) disengage(false); };

  // Capture the mouse: with raw movement first, then plainly where the browser or the system has none. A lock the browser
  // refuses (Chrome refuses one within about a second of an Esc exit) leaves the keyboard playing and says so; a click on the
  // view asks again. Chromium answers a refusal by the promise (and the event), the other browsers by the event alone.
  let eventRefusal = false;
  function lockPointer() {
    state.unlocked = false;
    const asked = stage.requestPointerLock({ unadjustedMovement: true });
    eventRefusal = !asked?.catch;
    asked?.catch(() => stage.requestPointerLock().catch(lockRefused));
  }

  function lockRefused() {
    eventRefusal = false;
    if (!state.engaged || document.pointerLockElement) return;
    state.unlocked = true;
    toast('The mouse was not captured: click the view to look with it');
  }

  function engage() {
    state.engaged = true;
    play.classList.remove('unfocused');
    showOverlay('none');
    if (settings.mouseLook) lockPointer();
  }

  function disengage(cover = true) {
    state.engaged = false;
    arrows.clear();
    gamepad.release();
    touchPad?.release();
    input.releaseAll(performance.now());
    repaint();
    if (document.pointerLockElement) document.exitPointerLock();
    if (state.status === 'ended' || !state.frames) return;
    if (cover) showOverlay('pause');
    else play.classList.add('unfocused');
  }

  stage.addEventListener('mousedown', () => {
    if (state.engaged) { if (state.unlocked) lockPointer(); } else if (state.status !== 'ended' && state.frames) engage();
  });
  document.addEventListener('keydown', onKey);
  document.addEventListener('keyup', onKey);
  stage.addEventListener('mousedown', onMouseButton);
  document.addEventListener('mouseup', onMouseButton);
  for (const type of ['mousedown', 'mouseup', 'auxclick']) document.addEventListener(type, onSideButton);
  document.addEventListener('mousemove', onMove);
  document.addEventListener('pointerlockchange', onLock);
  document.addEventListener('pointerlockerror', onLockError);
  window.addEventListener('blur', onBlur);
  stage.addEventListener('contextmenu', (e) => e.preventDefault());
  if (touch) $('#pause').onclick = () => { if (state.engaged) disengage(); };

  // ---------------------------------------------------------------------------------------------- HUD
  // The breakdown opens from the frame rate and latency. Docked, in the grid at the right of the view, it stays while
  // one plays; in a layout without that room (app.css sets --popover there) it is a popover under them, opened while
  // paused (the mouse is free only then) and gone when one plays. A touch device has no breakdown.
  const panel = $('#lat-panel');
  const popover = () => getComputedStyle(play).getPropertyValue('--popover').trim() === '1';
  function placePanel() {
    if (panel.hidden) return;
    if (!popover()) { panel.style.top = panel.style.right = ''; return; }
    const stats = $('#stats').getBoundingClientRect();
    panel.style.top = `${Math.max(8, Math.min(stats.bottom + 8, innerHeight - panel.offsetHeight - 8))}px`;
    panel.style.right = `${Math.max(8, innerWidth - stats.right)}px`;
  }
  function refreshPanel(rtt) {
    panel.innerHTML = meter.panel(rtt, state.framesPerBlock, state.position);
    placePanel();
  }
  // Open or close the breakdown; a popover takes the pause panel's place (the overlay's own card steps aside), a docked
  // one the loadout's.
  function openPanel(open) {
    panel.hidden = !open;
    $('#stats').setAttribute('aria-expanded', String(open));
    play.classList.toggle('measuring', open && popover());
    play.classList.toggle('docked', open && !popover());
    if (open) refreshPanel(stream.stats().rtt);
    else panel.style.top = panel.style.right = '';
  }
  $('#stats').onclick = () => {
    if (!touch && !(state.engaged && popover())) openPanel(panel.hidden);
  };
  overlay.addEventListener('click', (e) => { if (e.target === overlay && !panel.hidden && popover()) openPanel(false); });
  // A window that crosses between the layouts closes the breakdown: it would be a popover over a view being played, or
  // a docked card where a popover was placed. The observer reports after the new layout's style is in force, which the
  // window's resize event does not yet promise.
  let wasPopover = popover();
  const resized = new ResizeObserver(() => {
    if (popover() !== wasPopover) { wasPopover = popover(); openPanel(false); }
    placePanel();
    placeCard();
  });
  resized.observe(play);
  // The frame rate and latency, by which the breakdown and the pause panel are placed, also move while the play view
  // keeps its size: in a narrow window the plate of who plays sits above them in the bar, and its line wrapping anew (a
  // font that arrives, a clock that gains a digit) moves them; and the room beside them is the window's, which the play
  // view's size does not follow where the page is taller than the window (the overlay's border box is the window).
  // Placing the panels changes none of these boxes, so the observer never reports its own work.
  resized.observe($('.bar'));
  resized.observe(overlay, { box: 'border-box' });
  window.addEventListener('scroll', placeCard, { passive: true });     // a narrow window's page scrolls under the panel
  // In a low window the play view, not the page, scrolls: a wheel over the frame rate and latency (above the overlay)
  // scrolls it while paused.
  play.addEventListener('scroll', placeCard, { passive: true });
  const hud = setInterval(() => {
    const s = stream.stats();
    $('#fps').textContent = s.fps;
    const lat = meter.latency(touch);
    $('#lat').textContent = Number.isNaN(lat) ? '—' : Math.round(lat);
    $('#bars').dataset.q = s.quality;
    $('#bars').title = `round trip ${Number.isNaN(s.rtt) ? '—' : Math.round(s.rtt)} ms, ${s.stalls} stall(s) in 10 s`;
    if (!panel.hidden) refreshPanel(s.rtt);
  }, 250);

  // The players of the room; while only the player is in it, a line says that others appear when they take a seat.
  function renderRoster() {
    $('#roster').innerHTML = [...state.roster.values()].sort((x, y) => x.slot - y.slot).map((p) => `<li style="--acc:${playerColor(p.slot)}"
      class="${p.slot === seat.slot ? 'me' : ''}"><span class="dot"></span><span class="r-name">${esc(p.name)}</span>
      <span class="tag">P${p.slot + 1}</span><span class="r-meta">${p.slot === seat.slot ? 'you' : esc(p.team)}</span></li>`).join('');
    $('#alone').hidden = state.roster.size > 1;
  }

  function leave() {
    stream.send({ t: 'bye' });
    joined(roomId, null);
    location.hash = '#/';
  }

  // A tab that closes or reloads while one plays (Ctrl+W, with Ctrl held to crouch) asks first.
  const confirmClose = (e) => {
    if (!state.engaged) return;
    e.preventDefault();
    e.returnValue = '';
  };
  window.addEventListener('beforeunload', confirmClose);

  // The screen stays on while the view is open: a stick or a pad is no input that keeps a display from sleeping. A
  // hidden page loses the lock, so it is asked for again when the page shows.
  let wake = null;
  let open = true;
  const keepAwake = async () => {
    try { wake = await navigator.wakeLock?.request('screen'); } catch { wake = null; }
    if (!open) wake?.release();
  };
  keepAwake();
  document.addEventListener('visibilitychange', keepAwake);

  return () => {
    open = false;
    wake?.release();
    document.removeEventListener('visibilitychange', keepAwake);
    window.removeEventListener('beforeunload', confirmClose);
    cancelAnimationFrame(raf);
    clearInterval(hud);
    stream.close();
    window.removeEventListener('resize', crisp);
    window.removeEventListener('scroll', placeCard);
    resized.disconnect();
    document.removeEventListener('keydown', onKey);
    document.removeEventListener('keyup', onKey);
    document.removeEventListener('mouseup', onMouseButton);
    for (const type of ['mousedown', 'mouseup', 'auxclick']) document.removeEventListener(type, onSideButton);
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('pointerlockchange', onLock);
    document.removeEventListener('pointerlockerror', onLockError);
    window.removeEventListener('blur', onBlur);
    gamepad.close();
    document.removeEventListener('visibilitychange', sendVisibility);
    if (document.pointerLockElement) document.exitPointerLock();
    document.body.classList.remove('is-play');
  };
}
