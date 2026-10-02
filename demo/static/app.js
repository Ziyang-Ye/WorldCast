// WorldCast Live: the router, the lobby feed, the lobby and the seat picker. Play and watch live in their modules.

import { $, api, esc, seatColor, store, thumb } from './common.js';
import { renderPlay } from './play.js';
import { renderWatch } from './watch.js';

const app = $('#app');
let catalog = null;          // GET /api/lobby: round starts, play settings
let cleanup = () => {};

// ------------------------------------------------------------------------------------------ lobby feed
const feed = {
  state: { rooms: [], workers: { total: 0, free: 0, list: [] } },
  listeners: new Set(),
  subscribe(fn) { this.listeners.add(fn); fn(this.state); return () => this.listeners.delete(fn); },
  connect() {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/lobby`);
    ws.onmessage = (e) => {
      this.state = JSON.parse(e.data);
      renderGpuPill(this.state.workers);
      this.listeners.forEach((fn) => fn(this.state));
    };
    ws.onclose = () => { renderGpuPill(null); setTimeout(() => this.connect(), 1500); };
  },
};

function renderGpuPill(workers) {
  const pill = $('#gpu-pill');
  pill.classList.toggle('is-busy', !!workers && workers.total > 0 && workers.free === 0);
  pill.classList.toggle('is-none', !workers || workers.total === 0);
  if (!workers) { $('#gpu-text').textContent = 'Reconnecting'; return; }
  const engines = [...new Set(workers.list.map((w) => w.engine))];
  $('#gpu-text').textContent = workers.total ? `${workers.free} of ${workers.total} GPUs free` : 'No GPU workers';
  pill.title = engines.includes('mock') ? 'Mock engine: workers play pre-rendered frames (no GPU)' : `Engine: ${engines.join(', ')}`;
}

// ------------------------------------------------------------------------------------------------ lobby
function renderLobby() {
  const rounds = catalog.rounds;
  let selected = store('wc.round') || rounds[0].id;
  if (!rounds.some((r) => r.id === selected)) selected = rounds[0].id;
  let sync = catalog.play.default_sync;
  const mock = () => feed.state.workers.list.some((w) => w.engine === 'mock');
  app.innerHTML = `<div class="wrap">
    <section class="hero"><p class="eyebrow">WorldCast Live</p>
      <h1 class="headline">Play inside a world model.</h1>
      <p class="lead">Every player's view is generated on their own GPU. The clients share only player state and scene
        state: no game engine renders this, and no recorded positions steer it.</p></section>
    <div class="lobby-grid">
      <section class="card"><div class="card-head"><div><h2 class="card-title">Start a room</h2>
        <p class="card-sub">Pick a recorded round start. Each seat is one player, on one GPU.</p></div>
        <span class="count">${rounds.length} round start${rounds.length > 1 ? 's' : ''}</span></div>
        <div class="rounds" id="rounds"></div>
        <div class="create-row"><div class="seg" id="sync" role="group" aria-label="Synchronisation">
          <button type="button" data-sync="async">Asynchronous</button><button type="button" data-sync="lockstep">Lock-step</button></div>
          <span class="hint" id="sync-hint"></span>
          <button class="pill pill-light" id="create" type="button">Create room</button></div>
        <div class="error" id="create-error"></div></section>
      <section class="card"><div class="card-head"><div><h2 class="card-title">Live rooms</h2>
        <p class="card-sub">Join a free seat, or watch every view at once.</p></div><span class="count" id="room-count"></span></div>
        <div class="rooms" id="rooms"></div></section>
    </div></div>
    <footer class="foot"><div class="wrap"><p id="foot"></p></div></footer>`;

  const hints = { async: 'Each client steps on its peers\' latest states, extrapolated over a block: nobody waits.',
    lockstep: 'Every client waits for every peer\'s previous block, as in the paper\'s evaluation.' };
  const paint = () => {
    $('#rounds').innerHTML = rounds.map((r) => `<button class="round" type="button" data-id="${esc(r.id)}" aria-pressed="${r.id === selected}">
      <div class="thumb">${thumb(r)}<span class="chip round-map">${esc(r.map_label)}</span></div>
      <div class="round-label">${esc(r.label)}</div><div class="round-meta">${r.seats.length} seats${r.note ? ` · ${esc(r.note)}` : ''}</div></button>`).join('');
    $('#sync').querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.sync === sync));
    $('#sync-hint').textContent = hints[sync];
  };
  paint();
  $('#rounds').onclick = (e) => {
    const b = e.target.closest('.round');
    if (b) { selected = b.dataset.id; store('wc.round', selected); paint(); }
  };
  $('#sync').onclick = (e) => { const b = e.target.closest('button'); if (b) { sync = b.dataset.sync; paint(); } };
  $('#create').onclick = async () => {
    try {
      const room = await api('/api/rooms', { round: selected, sync });
      location.hash = `#/room/${room.id}`;
    } catch (err) { $('#create-error').textContent = err.message; }
  };

  const byId = Object.fromEntries(rounds.map((r) => [r.id, r]));
  const unsubscribe = feed.subscribe((lobby) => {
    $('#room-count').textContent = lobby.rooms.length ? `${lobby.rooms.length} live` : '';
    $('#rooms').innerHTML = lobby.rooms.length ? lobby.rooms.map((room) => {
      const r = byId[room.round];
      const full = r && room.players.length >= r.seats.length;
      return `<div class="room-row"><div class="thumb">${r ? thumb(r) : ''}</div><div>
        <div class="room-name">${esc(room.name)}</div>
        <div class="room-meta"><span class="room-dots">${room.players.map((p) => `<span class="dot" style="background:${seatColor(p.seat)}" title="${esc(p.name)}"></span>`).join('')}</span>
          ${room.players.length}/${r ? r.seats.length : '?'} · ${room.sync === 'lockstep' ? 'lock-step' : 'async'}</div></div>
        <div class="room-actions"><a class="pill pill-xs" href="#/room/${esc(room.id)}" ${full ? 'aria-disabled="true"' : ''}>Join</a>
          <a class="pill pill-xs" href="#/watch/${esc(room.id)}">Watch</a></div></div>`;
    }).join('') : '<div class="empty"><strong>No rooms yet</strong>Start one, then share its link.</div>';
    $('#foot').textContent = mock()
      ? 'These workers run the mock engine: they play pre-rendered WorldCast frames, and positions come from a stand-in state model driven by your controls. On GPU workers the frames are generated live.'
      : 'Frames are generated live by the WorldCast model on each player\'s GPU; positions come from each client\'s state model.';
  });
  return unsubscribe;
}

// ------------------------------------------------------------------------------------------ seat picker
function renderRoom(roomId) {
  let room = null;
  let chosen = null;
  app.innerHTML = '<div class="wrap"><div class="page-head"><div><a class="back" href="#/">‹ Lobby</a><h1 class="page-title">Loading</h1></div></div></div>';
  const paint = () => {
    const r = room.round_start;
    const taken = new Map(room.players.map((p) => [p.seat, p]));
    if (chosen !== null && taken.has(chosen)) chosen = null;
    $('#seats').innerHTML = r.seats.map((s) => {
      const p = taken.get(s.seat);
      return `<button class="seat" type="button" data-seat="${s.seat}" style="--acc:${seatColor(s.seat)}" ${p ? 'disabled' : ''} aria-pressed="${s.seat === chosen}">
        <div class="thumb">${thumb(r, s.seat)}<span class="chip"><span class="dot"></span>P${s.seat + 1}</span>
        ${p ? `<span class="taken chip"><span class="dot"></span>${esc(p.name)}</span>` : ''}</div>
        <div class="seat-line"><span class="tag tag-${s.team.toLowerCase()}">${esc(s.team)}</span>${p ? `<b>${esc(p.name)}</b> is playing` : 'Free'}</div></button>`;
    }).join('');
    $('#join').disabled = chosen === null;
    $('#join').textContent = chosen === null ? 'Pick a seat' : `Join as P${chosen + 1}`;
    $('#room-sub').textContent = `${r.map_label} · ${r.label} · ${room.sync === 'lockstep' ? 'lock-step' : 'asynchronous'} · ${room.players.length} of ${r.seats.length} seats taken`;
  };
  api(`/api/rooms/${encodeURIComponent(roomId)}`).then((r) => {
    room = r;
    app.innerHTML = `<div class="wrap"><div class="page-head"><div><a class="back" href="#/">‹ Lobby</a>
        <h1 class="page-title">Pick a seat.</h1><p class="page-sub" id="room-sub"></p></div>
        <div class="head-actions"><button class="pill pill-sm" id="copy" type="button">Copy link</button>
          <a class="pill pill-sm" href="#/watch/${esc(roomId)}">Watch</a></div></div>
      <div class="seats" id="seats"></div></div>
      <form class="joinbar" id="joinbar"><input id="name" maxlength="24" placeholder="Your name" autocomplete="nickname"
        value="${esc(store('wc.name') || '')}"><button class="pill pill-light" id="join" type="submit" disabled>Pick a seat</button></form>
      <div class="wrap"><div class="error" id="join-error"></div></div>`;
    paint();
    $('#seats').onclick = (e) => {
      const b = e.target.closest('.seat');
      if (b && !b.disabled) { chosen = Number(b.dataset.seat); paint(); }
    };
    $('#copy').onclick = async () => {
      try { await navigator.clipboard.writeText(location.href); $('#copy').textContent = 'Copied'; } catch { $('#copy').textContent = location.href; }
    };
    $('#joinbar').onsubmit = async (e) => {
      e.preventDefault();
      const name = $('#name').value.trim() || `Player ${chosen + 1}`;
      store('wc.name', name);
      $('#join').disabled = true;
      $('#join').textContent = 'Joining…';
      try {
        const join = await api(`/api/rooms/${encodeURIComponent(roomId)}/join`, { seat: chosen, name });
        sessionStorage.setItem(`wc.join.${roomId}`, JSON.stringify(join));
        location.hash = `#/play/${roomId}`;
      } catch (err) {
        $('#join-error').textContent = err.message;
        paint();
      }
    };
  }).catch((e) => {
    app.innerHTML = `<div class="wrap"><div class="page-head"><div><a class="back" href="#/">‹ Lobby</a>
      <h1 class="page-title">Room not found</h1><p class="page-sub">${esc(e.message)}</p></div></div></div>`;
  });
  return feed.subscribe((lobby) => {
    const r = lobby.rooms.find((x) => x.id === roomId);
    if (room && r) { room.players = r.players; paint(); }
  });
}

// ----------------------------------------------------------------------------------------------- router
function route() {
  cleanup();
  const [, view, id] = location.hash.replace(/^#/, '').split('/');
  window.scrollTo(0, 0);
  document.title = 'WorldCast Live';
  if (view === 'room' && id) cleanup = renderRoom(id);
  else if (view === 'play' && id) cleanup = renderPlay(app, id);
  else if (view === 'watch' && id) cleanup = renderWatch(app, id, feed);
  else cleanup = renderLobby();
}

async function boot() {
  feed.connect();
  catalog = await api('/api/lobby');
  window.addEventListener('hashchange', route);
  route();
}

boot().catch((e) => { app.innerHTML = `<div class="wrap"><div class="empty"><strong>${esc(e.message)}</strong></div></div>`; });
