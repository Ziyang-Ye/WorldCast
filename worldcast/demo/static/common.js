// Shared constants and helpers.

// What the page takes from the server (GET /api/lobby, `page`; set once at boot): the colour of each player slot,
// the longest player name, the pixel size of the generated frame, [width, height], and the camera's tangents of its half
// angles, [across, up], and its pitch range, degrees (the instant turn, turn.js).
export const page = {
  player_colors: ['#86868b'], player_name_length: 24, frame_pixels: [672, 384], camera_tans: [1, 1],
  pitch_range_degrees: [-90, 90],
};
export const playerColor = (slot) => page.player_colors[slot % page.player_colors.length];

const WEAPON_LABELS = {
  ak47: 'AK-47', m4a1: 'M4A4', m4a1_silencer: 'M4A1-S', awp: 'AWP', famas: 'FAMAS', galilar: 'Galil AR', aug: 'AUG',
  sg556: 'SG 553', ssg08: 'SSG 08', mp9: 'MP9', mac10: 'MAC-10', mp7: 'MP7', ump45: 'UMP-45', p90: 'P90',
  glock: 'Glock-18', usp_silencer: 'USP-S', hkp2000: 'P2000', p250: 'P250', deagle: 'Desert Eagle',
  fiveseven: 'Five-SeveN', tec9: 'Tec-9', elite: 'Dual Berettas', cz75a: 'CZ75-Auto', knife: 'Knife',
  hegrenade: 'HE Grenade', smokegrenade: 'Smoke', flashbang: 'Flashbang', molotov: 'Molotov', incgrenade: 'Incendiary',
  decoy: 'Decoy', c4: 'C4',
};
export const weaponLabel = (name) => WEAPON_LABELS[name] || name.replace(/_/g, ' ');

export const $ = (selector, root = document) => root.querySelector(selector);
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.text()) || res.statusText);
  return res.json();
}

// A worker URL from the coordinator: absolute (direct) or a path on this host (proxied).
export function wsUrl(url) {
  if (/^wss?:\/\//.test(url)) return url;
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}${url}`;
}

// h:mm:ss: a session has no end.
export function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const two = (n) => String(n).padStart(2, '0');
  return `${Math.floor(s / 3600)}:${two(Math.floor(s / 60) % 60)}:${two(s % 60)}`;
}

export const seatCount = (n) => `${n} seat${n === 1 ? '' : 's'}`;

export const median = (xs) => {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

export function store(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(key));
    localStorage.setItem(key, JSON.stringify(value));
  } catch { /* private mode: settings are not kept */ }
  return null;
}

// The seat a page holds in a room (the answer to its join): in sessionStorage, so that a reloaded page keeps it, and in
// memory, for a browser that lets the page keep nothing. `join` undefined: read it; null: let it go.
const joins = new Map();
export function joined(roomId, join) {
  const key = `wc.join.${roomId}`;
  if (join === undefined) {
    try { return JSON.parse(sessionStorage.getItem(key)) ?? joins.get(roomId) ?? null; } catch { return joins.get(roomId) ?? null; }
  }
  if (join === null) joins.delete(roomId); else joins.set(roomId, join);
  try {
    if (join === null) sessionStorage.removeItem(key); else sessionStorage.setItem(key, JSON.stringify(join));
  } catch { /* blocked storage: the memory has it */ }
  return join;
}

// Round-start thumbnail: the first frame of the player of `slot`, or the synthetic arena's drawn tile.
export function thumb(roundStart, slot = roundStart.cover) {
  const s = roundStart.seats.find((x) => x.slot === slot) || roundStart.seats[0];
  return s && s.preview ? `<img src="/library/${esc(s.preview)}" alt="" loading="lazy">` : '<div class="arena-art"></div>';
}
