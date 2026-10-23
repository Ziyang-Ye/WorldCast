// The map picker of the lobby: a map's overview; a click anywhere on it starts a room at the seat nearest to it, which the
// pointer's hover marks (POST /api/start). `maps` is GET /api/maps: per map its image (or none: a plain panel), its floors, and its seats as
// [u, v, yaw, floor], the position as fractions of the image and the heading in degrees.

import { $, api, esc, joined, page, store } from './common.js';

const DOT = 4;                 // css px: the radius of the marker of the seat a click would start at
const TICK = 12;               // css px: the length of its heading
const NEAR_COLOR = '#2997ff';  // --link

export function mountMapPicker(root, maps) {
  let current = Math.max(0, maps.findIndex((m) => m.map === store('wc.map')));
  let floor = 0;
  let near = -1;               // the seat nearest to the pointer, in maps[current].seats
  let pointer = null;          // the pointer's place in the view, css px
  let starting = false;

  root.innerHTML = `<div class="map-bar"><div class="map-tabs" id="map-tabs" role="group" aria-label="Map"></div>
      <div class="map-floors" id="map-floors" role="group" aria-label="Floor"></div>
      <input class="map-name" id="map-name" maxlength="${page.player_name_length}" placeholder="Your name" autocomplete="nickname"
        value="${esc(store('wc.name') || '')}"></div>
    <div class="map-stage"><div class="map-view" id="map-view">
      <img class="map-img" id="map-img" alt="" draggable="false"><canvas class="map-dots" id="map-dots" role="img"></canvas>
      <span class="map-hint" id="map-hint"></span>
    </div></div>
    <div class="error" id="map-error"></div>`;
  const view = $('#map-view', root);
  const canvas = $('#map-dots', root);

  const seats = () => maps[current].seats;
  const place = (seat) => [seat[0] * view.clientWidth, seat[1] * view.clientHeight];

  function hint(text) { $('#map-hint', root).textContent = text; }

  function paintChrome() {
    const m = maps[current];
    $('#map-tabs', root).innerHTML = maps.map((x, i) => `<button class="pill pill-sm" type="button" data-map="${i}"
      aria-pressed="${i === current}">${esc(x.label)}</button>`).join('');
    $('#map-floors', root).innerHTML = m.floors.length > 1 ? m.floors.map((f, i) => `<button class="pill pill-xs" type="button"
      data-floor="${i}" aria-pressed="${i === floor}">${esc(f.name)}</button>`).join('') : '';
    const shown = m.floors[Math.min(floor, m.floors.length - 1)];
    const img = $('#map-img', root);
    if (shown.image) img.src = shown.image; else img.removeAttribute('src');
    img.hidden = !shown.image;
    view.classList.toggle('is-plain', !shown.image);
    view.style.aspectRatio = `${m.size[0]} / ${m.size[1]}`;
    canvas.setAttribute('aria-label', `${m.label}: ${m.seats.length} starting spots. Click to start at the nearest.`);
    hint('Click anywhere on the map');
  }

  function draw() {
    const w = view.clientWidth;
    const h = view.clientHeight;
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    const ctx = canvas.getContext('2d');
    ctx.scale(scale, scale);
    const heading = (seat, [x, y], length) => {
      const a = (seat[2] * Math.PI) / 180;
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.cos(a) * length, y - Math.sin(a) * length);   // the image's y runs down, the world's up
    };
    ctx.lineCap = 'round';
    if (near < 0) return;
    const seat = seats()[near];
    const at = place(seat);
    if (pointer) {   // from the pointer to the seat the click would start at
      ctx.beginPath();
      ctx.moveTo(pointer[0], pointer[1]);
      ctx.lineTo(at[0], at[1]);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = 'rgba(41, 151, 255, 0.8)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.beginPath();
    heading(seat, at, TICK * 1.8);
    ctx.strokeStyle = NEAR_COLOR;
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(at[0], at[1], DOT * 2, 0, 2 * Math.PI);
    ctx.fillStyle = NEAR_COLOR;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#fff';
    ctx.stroke();
  }

  function nearest(x, y) {
    let best = -1;
    let least = Infinity;
    seats().forEach((seat, i) => {
      if (seat[3] !== floor) return;
      const [sx, sy] = place(seat);
      const d = (sx - x) ** 2 + (sy - y) ** 2;
      if (d < least) { least = d; best = i; }
    });
    return best;
  }

  const at = (e) => {
    const box = view.getBoundingClientRect();
    return [e.clientX - box.left, e.clientY - box.top];
  };
  view.onpointermove = (e) => {
    pointer = at(e);
    near = nearest(...pointer);
    draw();
  };
  view.onpointerleave = () => { pointer = null; near = -1; draw(); };
  view.onclick = async (e) => {
    if (starting) return;
    const [x, y] = at(e);
    const m = maps[current];
    starting = true;
    $('#map-error', root).textContent = '';
    hint('Starting…');
    const name = $('#map-name', root).value.trim();
    try {
      if (name) store('wc.name', name);
      const started = await api('/api/start', {
        map: m.map, u: x / view.clientWidth, v: y / view.clientHeight, floor: m.floors.length > 1 ? floor : null, name,
      });
      joined(started.room.id, started);
      location.hash = `#/play/${started.room.id}`;
    } catch (err) {
      $('#map-error', root).textContent = err.message;
      starting = false;
      paintChrome();
    }
  };
  root.onclick = (e) => {
    const tab = e.target.closest('[data-map]');
    const pill = e.target.closest('[data-floor]');
    if (!tab && !pill) return;
    if (tab) { current = Number(tab.dataset.map); floor = 0; store('wc.map', maps[current].map); }
    if (pill) floor = Number(pill.dataset.floor);
    near = -1;
    paintChrome();
    draw();
  };

  paintChrome();
  const resize = new ResizeObserver(draw);
  resize.observe(view);
  $('#map-img', root).onload = draw;
  return () => resize.disconnect();
}
