// Shared constants and helpers.

// The paper model's buttons, in its order (worldcast/data/actions.py PAPER_ACTION_BUTTONS): bit i of an input tick.
export const BUTTONS = ['forward', 'back', 'move_left', 'move_right', 'jump', 'duck', 'speed', 'attack', 'attack2',
  'reload', 'look_at_weapon'];
export const BIT = Object.fromEntries(BUTTONS.map((b, i) => [b, 1 << i]));

// Seat colours: the project page's client colours, then Apple system colours (demo/mock_engine.py SEAT_COLORS).
export const SEAT_COLORS = ['#E67F27', '#3D94E8', '#3FB576', '#BF5AF2', '#FF375F', '#64D2FF', '#FFD60A', '#AC8E68',
  '#5E5CE6', '#30D158'];
export const seatColor = (seat) => SEAT_COLORS[seat % SEAT_COLORS.length];

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

export function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

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

// Round-start thumbnail: the seat's first frame, or the synthetic arena's drawn tile.
export function thumb(round, seat = round.cover) {
  const s = round.seats.find((x) => x.seat === seat) || round.seats[0];
  return s && s.preview ? `<img src="/library/${esc(s.preview)}" alt="" loading="lazy">` : '<div class="arena-art"></div>';
}
