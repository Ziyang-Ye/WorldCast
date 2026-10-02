// Spectator mode: every player's live view in a grid, and one minimap with everyone.

import { $, api, esc, seatColor, wsUrl } from './common.js';
import { FrameStream } from './stream.js';
import { Minimap, fromWire } from './minimap.js';

const columns = (n) => (n <= 1 ? 1 : n <= 4 ? 2 : n <= 9 ? 3 : 4);

export function renderWatch(app, roomId, feed) {
  const tiles = new Map();            // seat -> {stream, el, stats}
  let map = null;
  let room = null;
  let closed = false;

  app.innerHTML = `<div class="wrap watch"><div class="watch-head">
      <div><a class="back" href="#/">‹ Lobby</a><h1 id="w-title">Watching</h1><p class="page-sub" id="w-meta">Loading</p></div>
      <a class="pill pill-sm" href="#/room/${esc(roomId)}">Take a seat</a></div>
    <div class="watch-grid"><div class="tiles" id="tiles"></div>
      <aside class="side"><div class="card"><div class="side-map"><canvas id="w-map"></canvas></div></div>
        <div class="card"><ul class="roster" id="w-roster"></ul></div></aside></div></div>`;

  api(`/api/rooms/${encodeURIComponent(roomId)}`).then((r) => {
    if (closed) return;
    room = r;
    map = new Minimap($('#w-map'), r.round_start);
    $('#w-title').textContent = `Watching ${r.round_start.map_label}`;
    sync(r.players);
  }).catch((e) => { $('#tiles').innerHTML = `<div class="empty"><strong>${esc(e.message)}</strong></div>`; });

  function sync(players) {
    const seats = new Set(players.map((p) => p.seat));
    for (const [seat, tile] of tiles) {
      if (!seats.has(seat)) { tile.stream.close(); tile.el.remove(); tiles.delete(seat); map?.remove(seat); }
    }
    for (const p of players) if (!tiles.has(p.seat) && p.watch_url) addTile(p);
    $('#tiles').style.setProperty('--cols', columns(tiles.size));
    if (!tiles.size) $('#tiles').innerHTML = '<div class="empty"><strong>Nobody is playing</strong>Take a seat to start.</div>';
    $('#w-meta').textContent = `${room.round_start.map_label} · ${room.round_start.label} · ${players.length} playing · ${room.sync === 'lockstep' ? 'lock-step' : 'asynchronous'}`;
    $('#w-roster').innerHTML = players.map((p) => `<li style="--acc:${seatColor(p.seat)}"><span class="dot"></span>
      <span class="r-name">${esc(p.name)}</span><span class="tag">P${p.seat + 1}</span>
      <span class="r-meta" data-seat="${p.seat}">—</span></li>`).join('');
  }

  function addTile(p) {
    $('#tiles .empty')?.remove();
    const el = document.createElement('div');
    el.className = 'tile';
    el.style.setProperty('--acc', seatColor(p.seat));
    el.innerHTML = `<canvas></canvas><div class="waiting">Connecting…</div>
      <span class="chip"><span class="dot"></span>${esc(p.name)}<span class="chip-t">P${p.seat + 1}</span></span>
      <span class="chip tile-stats">—</span>`;
    $('#tiles').append(el);
    const tile = { el, gen: [] };
    tile.stream = new FrameStream(wsUrl(p.watch_url), $('canvas', el), {
      buffer: 3,
      onFrame: (h) => {
        $('.waiting', el)?.remove();
        tile.gen.push(h.tm[1]);
        if (tile.gen.length > 16) tile.gen.shift();
        map?.update([{ ...fromWire(h.s), label: p.name }]);
      },
      onClose: () => { if ($('.waiting', el) === null) el.insertAdjacentHTML('beforeend', '<div class="waiting">Ended</div>'); },
    });
    tiles.set(p.seat, tile);
  }

  const stats = setInterval(() => {
    for (const [seat, tile] of tiles) {
      const s = tile.stream.stats();
      const text = `${s.fps} fps`;
      $('.tile-stats', tile.el).textContent = text;
      const row = $(`.r-meta[data-seat="${seat}"]`);
      if (row) row.textContent = `${s.fps} fps · ${Number.isNaN(s.rtt) ? '—' : Math.round(s.rtt)} ms`;
    }
  }, 500);

  const unsubscribe = feed.subscribe((lobby) => {
    const r = lobby.rooms.find((x) => x.id === roomId);
    if (r && room) sync(r.players);
  });

  return () => {
    closed = true;
    clearInterval(stats);
    unsubscribe();
    for (const tile of tiles.values()) tile.stream.close();
  };
}
