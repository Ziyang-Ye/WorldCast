// The play view: the generated view, keyboard and mouse in, the HUD (fps, input-to-photon latency, network), the
// minimap and the key caps.

import { $, BIT, clock, esc, median, seatColor, store, weaponLabel, wsUrl } from './common.js';
import { FrameStream } from './stream.js';
import { Minimap, fromWire } from './minimap.js';

const KEYS = {
  KeyW: 'forward', KeyS: 'back', KeyA: 'move_left', KeyD: 'move_right', Space: 'jump', ControlLeft: 'duck',
  KeyC: 'duck', ShiftLeft: 'speed', KeyR: 'reload', KeyF: 'look_at_weapon',
};
const MOUSE = { 0: 'attack', 2: 'attack2' };
const CAPS = [['forward', 'W'], ['move_left', 'A'], ['back', 'S'], ['move_right', 'D']];
const CAPS_ROW = [['jump', 'Space', 'cap-w'], ['duck', 'Ctrl'], ['speed', 'Shift'], ['attack', 'LMB'], ['attack2', 'RMB'],
  ['reload', 'R'], ['look_at_weapon', 'F']];
const LAT_PARTS = [['input', 'Sampling', '#f5f5f7'], ['net', 'Network', '#64d2ff'],
  ['wait', 'Waiting for the next block', '#8e8e93'], ['gen', 'Generating', '#bf5af2'], ['enc', 'Encoding', '#ffd60a'],
  ['buf', 'Jitter buffer', '#30d158']];
const MOUSE_ONSET_GAP_MS = 150;
const KEPT = 24;

const crosshair = `<svg class="crosshair" viewBox="0 0 24 24" aria-hidden="true"><g stroke="#fff" stroke-width="2"
  stroke-linecap="round"><path d="M12 3v5M12 16v5M3 12h5M16 12h5"/></g></svg>`;

export function renderPlay(app, roomId) {
  const join = JSON.parse(sessionStorage.getItem(`wc.join.${roomId}`) || 'null');
  if (!join) { location.hash = `#/room/${roomId}`; return () => {}; }
  document.body.classList.add('is-play');
  const seat = join.seat;
  const color = seatColor(seat.seat);
  const touch = matchMedia('(pointer: coarse)').matches;
  app.innerHTML = `
  <div class="play" style="--acc:${color}">
    <div class="stage" id="stage">
      <canvas class="view" id="view"></canvas>${crosshair}
      <div class="hud-who glass"><span class="dot"></span><span>${esc(join.name)}</span><span class="sep"></span>
        <span class="sub">${esc(join.round_start.map_label)} · ${esc(join.round_start.label)}</span><span class="sep"></span>
        <span class="sub mono" id="clock">0:00</span></div>
      <div class="overlay" id="overlay"></div>
    </div>
    <div class="hud">
      <button class="hud-stats glass" id="stats" type="button" title="Latency breakdown">
        <span class="stat"><b id="fps">0</b><small>fps</small></span>
        <span class="stat"><b id="lat">—</b><small>${touch ? 'ms end to end' : 'ms input→photon'}</small></span>
        <span class="bars" id="bars" data-q="2"><i></i><i></i><i></i></span></button>
      <div class="lat-panel glass" id="lat-panel" hidden></div>
      <div class="minimap glass"><canvas id="map"></canvas></div>
      <div class="keys glass" id="caps">
        <div class="kg kg-wasd">${CAPS.map(([b, k]) => `<span class="cap" data-b="${b}">${k}</span>`).join('')}</div>
        <div class="kg kg-row">${CAPS_ROW.map(([b, k, c]) => `<span class="cap ${c || ''}" data-b="${b}">${k}</span>`).join('')}</div>
      </div>
      <div class="weapons glass" id="weapons"></div>
      <p class="mobile-note glass">You hold <b>seat ${seat.seat + 1}</b>. Playing needs a keyboard and a mouse;
        on this device you watch your seat live.</p>
      <ul class="roster mobile-roster glass" id="roster"></ul>
    </div>
  </div>`;

  const stage = $('#stage');
  const overlay = $('#overlay');
  const map = new Minimap($('#map'), join.round_start, { follow: seat.seat });
  const state = {
    held: new Set(), dp: 0, dy: 0, seq: 0, sent: new Map(), events: [], lat: [], parts: [], engaged: false,
    lastMove: 0, onset: null, lastQ: -1, fresh: [], weapon: 0, lastWeapon: 0, loadout: seat.loadout, weaponIds: {}, status: 'starting',
    sens: store('wc.sens') || join.play.mouse_degrees_per_pixel, roster: new Map(), frames: 0,
  };
  window.wcPlay = state;                       // read by the headless checks (docs/demo.md, "Testing")

  const stream = new FrameStream(wsUrl(join.play_url), $('#view'), {
    fps: join.play.fps, buffer: join.play.jitter_buffer_frames, onFrame, onText, onClose,
  });

  // ------------------------------------------------------------------------------------------- overlays
  function showOverlay(kind, message = '') {
    state.status = kind === 'pause' ? state.status : kind;
    if (kind === 'none' || (touch && kind === 'pause')) { overlay.hidden = true; return; }
    overlay.hidden = false;
    if (kind === 'starting') {
      overlay.innerHTML = `<div class="panel"><div class="spinner"></div><h2>Starting your client</h2>
        <p>Seat ${seat.seat + 1} on GPU worker <b>${esc(join.worker.id)}</b>. ${esc(message)}</p></div>`;
    } else if (kind === 'ended') {
      overlay.innerHTML = `<div class="panel"><h2>Session ended</h2><p>${esc(message || 'The connection closed.')}</p>
        <div class="row"><a class="pill pill-light" href="#/">Back to the lobby</a></div></div>`;
    } else {
      overlay.innerHTML = `<div class="panel"><h2>Click to play</h2>
        <p>Your view is generated live on GPU worker <b>${esc(join.worker.id)}</b>.</p>
        <dl class="legend">
          <dt><span class="cap">W</span><span class="cap">A</span><span class="cap">S</span><span class="cap">D</span></dt><dd>Move</dd>
          <dt><span class="cap">Mouse</span></dt><dd>Look, fire (left), alt-fire (right)</dd>
          <dt><span class="cap cap-w">Space</span><span class="cap">C</span><span class="cap">Shift</span></dt><dd>Jump, crouch, walk</dd>
          <dt><span class="cap">1</span><span class="cap">2</span><span class="cap">3</span><span class="cap">4</span><span class="cap">Q</span></dt><dd>Weapons, last weapon</dd>
          <dt><span class="cap">R</span><span class="cap">F</span><span class="cap">Esc</span></dt><dd>Reload, inspect, pause</dd>
        </dl>
        <label class="sens">Mouse<input id="sens" type="range" min="0.015" max="0.15" step="0.005" value="${state.sens}">
          <output id="sens-v">${state.sens.toFixed(3)}°</output></label>
        <div class="row"><button class="pill pill-light" id="go" type="button">Play</button>
          <button class="pill" id="leave" type="button">Leave seat</button></div></div>`;
      $('#go').onclick = engage;
      $('#leave').onclick = leave;
      $('#sens').oninput = (e) => {
        state.sens = Number(e.target.value);
        $('#sens-v').textContent = `${state.sens.toFixed(3)}°`;
        store('wc.sens', state.sens);
      };
    }
  }
  showOverlay('starting', 'Loading the round start…');

  // ---------------------------------------------------------------------------------------- messages
  function onText(m) {
    if (m.t === 'welcome') {
      state.weaponIds = m.weapon_ids;
      state.loadout = m.seat.loadout;
      renderWeapons();
      for (const p of m.roster) state.roster.set(p.seat, p);
      renderRoster();
    } else if (m.t === 'status') {
      if (m.status === 'ended') { disengage(); showOverlay('ended', m.message); }
    } else if (m.t === 'roster') {
      const now = new Map(m.roster.map((p) => [p.seat, p]));
      for (const [s, p] of now) if (!state.roster.has(s) && s !== seat.seat) toast(`${p.name} joined seat ${s + 1}`);
      for (const [s, p] of state.roster) if (!now.has(s)) { toast(`${p.name} left`); map.remove(s); }
      state.roster = now;
      renderRoster();
    }
  }

  function onClose() {
    if (state.status !== 'ended') { disengage(); showOverlay('ended', 'Lost the connection to the GPU worker.'); }
  }

  function onFrame(h, t) {
    state.frames += 1;
    state.last = h;
    if (state.status === 'starting') showOverlay(touch ? 'none' : 'pause');
    const players = [fromWire(h.s), ...h.p.map(fromWire)];
    for (const p of players) p.label = p.seat === seat.seat ? '' : (state.roster.get(p.seat)?.name || `P${p.seat + 1}`);
    map.update(players);
    $('#clock').textContent = clock(h.rt);
    if (h.q > state.lastQ && state.sent.has(h.q)) {          // end to end: the newest input tick to its first frame
      state.lastQ = h.q;
      state.fresh.push(t.shown - state.sent.get(h.q));
      if (state.fresh.length > KEPT) state.fresh.shift();
    }
    // input -> photon: each key / button / mouse-move onset, at the first frame that folds in its input tick. The
    // worker times the frame's newest tick q; an earlier tick e waited (sent q - sent e) longer for the block.
    while (state.events.length && state.events[0].seq <= h.q) {
      const ev = state.events.shift();
      const total = t.shown - ev.t;
      state.lat.push(total);
      if (state.lat.length > KEPT) state.lat.shift();
      const sentE = state.sent.get(ev.seq);
      const sentQ = state.sent.get(h.q);
      if (h.tm[0] === null || sentE === undefined || sentQ === undefined) continue;
      const [waitQ, gen, enc] = h.tm;
      const part = { total, input: sentE - ev.t, wait: waitQ + sentQ - sentE, gen, enc, buf: t.shown - t.recv };
      part.net = Math.max(0, total - part.input - part.wait - gen - enc - part.buf);
      state.parts.push(part);
      if (state.parts.length > KEPT) state.parts.shift();
    }
  }

  // ------------------------------------------------------------------------------------------- input
  function mask() {
    let m = 0;
    for (const b of state.held) m |= BIT[b];
    return m;
  }

  function sendTick(eventTime) {
    state.seq += 1;
    const now = performance.now();
    state.sent.set(state.seq, now);
    if (state.sent.size > 600) state.sent.delete(state.seq - 600);
    if (eventTime !== undefined) state.events.push({ seq: state.seq, t: eventTime });
    if (state.onset !== null) { state.events.push({ seq: state.seq, t: state.onset }); state.onset = null; }
    if (state.events.length > 64) state.events.shift();
    stream.send({ t: 'in', q: state.seq, b: mask(), dp: +state.dp.toFixed(4), dy: +state.dy.toFixed(4),
      w: state.weaponIds[state.loadout[state.weapon]] ?? 0 });
    state.dp = 0;
    state.dy = 0;
  }

  let raf = requestAnimationFrame(function sample() {
    raf = requestAnimationFrame(sample);
    if (state.engaged || touch) sendTick();      // a phone sends idle ticks: its end-to-end delay stays measurable
  });

  function press(button, down, e) {
    if (down === state.held.has(button)) return;
    if (down) state.held.add(button); else state.held.delete(button);
    $(`.cap[data-b="${button}"]`)?.classList.toggle('on', down);
    sendTick(e.timeStamp);
  }

  function selectWeapon(i, e) {
    if (i < 0 || i >= state.loadout.length || i === state.weapon) return;
    state.lastWeapon = state.weapon;
    state.weapon = i;
    renderWeapons();
    sendTick(e.timeStamp);
  }

  function renderWeapons() {
    $('#weapons').innerHTML = state.loadout.map((w, i) =>
      `<div class="weapon ${i === state.weapon ? 'on' : ''}"><b>${i + 1}</b>${esc(weaponLabel(w))}</div>`).join('');
  }

  const onKey = (e) => {
    if (!state.engaged) return;
    const down = e.type === 'keydown';
    if (KEYS[e.code]) { e.preventDefault(); if (!e.repeat) press(KEYS[e.code], down, e); return; }
    if (down && /^Digit[1-4]$/.test(e.code)) selectWeapon(Number(e.code.slice(5)) - 1, e);
    if (down && e.code === 'KeyQ') selectWeapon(state.lastWeapon, e);
    if (down && e.code === 'Escape' && !document.pointerLockElement) disengage();
  };
  const onMouseButton = (e) => {
    if (!state.engaged || !(e.button in MOUSE)) return;
    e.preventDefault();
    press(MOUSE[e.button], e.type === 'mousedown', e);
  };
  const onMove = (e) => {
    if (!state.engaged) return;
    if (e.timeStamp - state.lastMove > MOUSE_ONSET_GAP_MS && (e.movementX || e.movementY)) state.onset = e.timeStamp;
    state.lastMove = e.timeStamp;
    state.dy -= e.movementX * state.sens;     // mouse right turns right: yaw (counter-clockwise) goes down
    state.dp += e.movementY * state.sens;     // mouse down looks down: pitch goes up
  };
  const onLock = () => { if (!document.pointerLockElement && state.engaged) disengage(); };

  function engage() {
    state.engaged = true;
    showOverlay('none');
    const lock = stage.requestPointerLock?.({ unadjustedMovement: true });
    if (lock && lock.catch) lock.catch(() => stage.requestPointerLock?.()?.catch?.(() => {}));
  }

  function disengage() {
    state.engaged = false;
    for (const b of [...state.held]) press(b, false, { timeStamp: performance.now() });
    if (document.pointerLockElement) document.exitPointerLock();
    if (state.status !== 'ended' && state.frames) showOverlay('pause');
  }

  stage.addEventListener('mousedown', (e) => { if (!state.engaged && state.status !== 'ended' && state.frames && e.target.closest('.overlay') === null) engage(); });
  document.addEventListener('keydown', onKey);
  document.addEventListener('keyup', onKey);
  stage.addEventListener('mousedown', onMouseButton);
  document.addEventListener('mouseup', onMouseButton);
  document.addEventListener('mousemove', onMove);
  document.addEventListener('pointerlockchange', onLock);
  stage.addEventListener('contextmenu', (e) => e.preventDefault());

  // ---------------------------------------------------------------------------------------------- HUD
  $('#stats').onclick = () => { $('#lat-panel').hidden = !$('#lat-panel').hidden; };
  const hud = setInterval(() => {
    const s = stream.stats();
    $('#fps').textContent = s.fps;
    const lat = median(touch ? state.fresh : state.lat);
    $('#lat').textContent = Number.isNaN(lat) ? '—' : Math.round(lat);
    $('#bars').dataset.q = s.quality;
    $('#bars').title = `round trip ${Number.isNaN(s.rtt) ? '—' : Math.round(s.rtt)} ms, ${s.stalls} stall(s) in 10 s`;
    state.hud = { fps: s.fps, latency: lat, rtt: s.rtt, stalls: s.stalls, quality: s.quality };
    if (!$('#lat-panel').hidden) renderLatency(s);
  }, 250);

  function renderLatency(s) {
    const mean = (k) => state.parts.reduce((a, p) => a + p[k], 0) / state.parts.length;
    const parts = LAT_PARTS.map(([k, label, c]) => [k, label, c, mean(k)]);
    const total = mean('total');
    const ok = state.parts.length > 0;
    $('#lat-panel').innerHTML = `<h4>INPUT → PHOTON${ok ? ` · MEAN ${Math.round(total)} MS` : ''}</h4>
      <div class="lat-bar">${ok ? parts.map(([, , c, v]) => `<i style="background:${c};width:${(100 * v) / total}%"></i>`).join('') : ''}</div>
      <div class="lat-rows">${parts.map(([, label, c, v]) => `<span class="sw" style="background:${c}"></span><span>${label}</span>
        <span class="v">${ok ? Math.round(v) : '—'}</span>`).join('')}
        <span></span><span>Round trip</span><span class="v">${Number.isNaN(s.rtt) ? '—' : Math.round(s.rtt)}</span></div>
      <p class="lat-note">From each key press, click or mouse-move onset to the first frame that shows it, over the
        last ${state.parts.length} inputs (the HUD shows their median). A block is ${join.worker.frames_per_step || '?'}
        frames: an input waits for the next block to start.</p>`;
  }

  function renderRoster() {
    $('#roster').innerHTML = [...state.roster.values()].sort((x, y) => x.seat - y.seat).map((p) => `<li style="--acc:${seatColor(p.seat)}">
      <span class="dot"></span><span class="r-name">${esc(p.name)}</span><span class="tag">P${p.seat + 1}</span>
      <span class="r-meta">${p.seat === seat.seat ? 'you' : esc(p.team)}</span></li>`).join('');
  }

  let toastTimer;
  function toast(text) {
    let el = $('.toast');
    if (!el) { el = document.createElement('div'); el.className = 'toast glass'; stage.append(el); }
    el.textContent = text;
    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('show'), 2600);
  }

  function leave() {
    stream.send({ t: 'bye' });
    sessionStorage.removeItem(`wc.join.${roomId}`);
    location.hash = '#/';
  }

  return () => {
    cancelAnimationFrame(raf);
    clearInterval(hud);
    stream.close();
    document.removeEventListener('keydown', onKey);
    document.removeEventListener('keyup', onKey);
    document.removeEventListener('mouseup', onMouseButton);
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('pointerlockchange', onLock);
    if (document.pointerLockElement) document.exitPointerLock();
    document.body.classList.remove('is-play');
  };
}
