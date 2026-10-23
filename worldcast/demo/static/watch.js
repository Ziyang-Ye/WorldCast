// Spectator mode: every player's live view in a grid.

import { $, api, esc, playerColor, wsUrl } from './common.js';
import { FrameStream } from './stream.js';

const columns = (n) => (n <= 1 ? 1 : n <= 4 ? 2 : n <= 9 ? 3 : 4);

export function renderWatch(app, roomId, feed) {
  const tiles = new Map();            // player slot -> {stream, el}
  let room = null;
  let closed = false;

  app.innerHTML = `<div class="wrap watch"><div class="watch-head">
      <div><a class="back" href="#/">‹ Lobby</a><h1 id="w-title">Watching</h1><p class="page-sub" id="w-meta">Loading</p></div>
      <a class="pill pill-sm" href="#/room/${esc(roomId)}">Take a seat</a></div>
    <div class="watch-grid"><div class="tiles" id="tiles"></div>
      <aside class="side"><div class="card"><ul class="roster" id="w-roster"></ul></div></aside></div></div>`;

  api(`/api/rooms/${encodeURIComponent(roomId)}`).then((r) => {
    if (closed) return;
    room = r;
    $('#w-title').textContent = `Watching ${r.round_start.map_label}`;
    sync(r.players);
  }).catch((e) => { $('#tiles').innerHTML = `<div class="empty"><strong>${esc(e.message)}</strong></div>`; });

  function sync(players) {
    const slots = new Set(players.map((p) => p.slot));
    for (const [slot, tile] of tiles) {
      if (!slots.has(slot)) { tile.stream.close(); tile.el.remove(); tiles.delete(slot); }
    }
    for (const p of players) if (!tiles.has(p.slot) && p.watch_url) addTile(p);
    $('#tiles').style.setProperty('--cols', columns(tiles.size));
    if (!tiles.size) $('#tiles').innerHTML = '<div class="empty"><strong>Nobody is playing</strong>Take a seat to start.</div>';
    $('#w-meta').textContent = `${room.round_start.map_label} · ${room.round_start.label} · ${players.length} playing`;
    $('#w-roster').innerHTML = players.map((p) => `<li style="--acc:${playerColor(p.slot)}"><span class="dot"></span>
      <span class="r-name">${esc(p.name)}</span><span class="tag">P${p.slot + 1}</span>
      <span class="r-meta" data-slot="${p.slot}">—</span></li>`).join('');
  }

  function addTile(p) {
    $('#tiles .empty')?.remove();
    const el = document.createElement('div');
    el.className = 'tile';
    el.style.setProperty('--acc', playerColor(p.slot));
    el.innerHTML = `<canvas></canvas><div class="waiting">Connecting…</div>
      <span class="chip"><span class="dot"></span>${esc(p.name)}<span class="chip-t">P${p.slot + 1}</span></span>
      <span class="chip tile-stats">—</span>`;
    $('#tiles').append(el);
    const tile = { el };
    tile.stream = new FrameStream(wsUrl(p.watch_url), $('canvas', el), {
      buffer: 3,
      onFrame: () => { $('.waiting', el)?.remove(); },
      onClose: () => { if ($('.waiting', el) === null) el.insertAdjacentHTML('beforeend', '<div class="waiting">Ended</div>'); },
    });
    tiles.set(p.slot, tile);
  }

  const stats = setInterval(() => {
    for (const [slot, tile] of tiles) {
      const s = tile.stream.stats();
      $('.tile-stats', tile.el).textContent = `${s.fps} fps`;
      const row = $(`.r-meta[data-slot="${slot}"]`);
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
