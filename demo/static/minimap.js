// The minimap: every player where the state models put them (north up; world +x right, +y up).

import { seatColor } from './common.js';

const HFOV = 106.26;
const GRID_UNITS = 256;

export class Minimap {
  // follow: the seat at the centre (play view), or null to fit everyone (spectators). unitsAcross: zoom when following.
  constructor(canvas, round, { follow = null, unitsAcross = 1900 } = {}) {
    Object.assign(this, { canvas, follow, unitsAcross });
    this.ctx = canvas.getContext('2d');
    this.players = new Map();
    this.center = null;
    this.radar = round.radar;
    if (this.radar) {
      this.image = new Image();
      this.image.onload = () => this.draw();
      this.image.src = `/library/${this.radar.image}`;
    }
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  // players: [{seat, x, y, yaw, alive}]; replaces what is known about those seats.
  update(players) {
    for (const p of players) this.players.set(p.seat, p);
    this.draw();
  }

  remove(seat) { this.players.delete(seat); this.draw(); }

  draw() {
    const { canvas, ctx } = this;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    const W = Math.max(1, Math.round(rect.width * dpr));
    const H = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== W || canvas.height !== H) Object.assign(canvas, { width: W, height: H });
    const players = [...this.players.values()];
    const [cx, cy, scale] = this.view(players, W, H);
    ctx.fillStyle = '#0b0b0d';
    ctx.fillRect(0, 0, W, H);
    const toPx = (x, y) => [W / 2 + (x - cx) * scale, H / 2 - (y - cy) * scale];
    if (this.image && this.image.complete && this.image.naturalWidth) {
      const r = this.radar;
      const [dx, dy] = toPx(r.x0 - r.cx / r.scale, r.y0 + r.cy / r.scale);
      ctx.globalAlpha = 0.95;
      ctx.drawImage(this.image, dx, dy, this.image.naturalWidth * scale / r.scale, this.image.naturalHeight * scale / r.scale);
      ctx.globalAlpha = 1;
    } else {
      this.grid(toPx, cx, cy, W, H, scale);
    }
    const order = players.sort((a, b) => (a.seat === this.follow) - (b.seat === this.follow));
    for (const p of order) this.player(p, toPx(p.x, p.y), W, H, dpr);
  }

  view(players, W, H) {
    const target = this.follow !== null ? this.players.get(this.follow) : null;
    let tx; let ty; let scale;
    if (target) {
      [tx, ty] = [target.x, target.y];
      scale = Math.min(W, H) / this.unitsAcross;
    } else if (players.length) {
      const xs = players.map((p) => p.x); const ys = players.map((p) => p.y);
      [tx, ty] = [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2];
      const extent = Math.max(900, Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
      scale = (0.72 * Math.min(W, H)) / extent;
    } else {
      [tx, ty, scale] = [0, 0, Math.min(W, H) / this.unitsAcross];
    }
    if (!this.center) this.center = [tx, ty];
    this.center = [this.center[0] + (tx - this.center[0]) * 0.5, this.center[1] + (ty - this.center[1]) * 0.5];
    this.scale = this.scale ? this.scale + (scale - this.scale) * 0.2 : scale;
    return [this.center[0], this.center[1], this.scale];
  }

  grid(toPx, cx, cy, W, H, scale) {
    const { ctx } = this;
    const step = GRID_UNITS;
    const half = Math.max(W, H) / scale / 2 + step;
    ctx.strokeStyle = 'rgba(168, 190, 220, .07)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = Math.floor((cx - half) / step) * step; x <= cx + half; x += step) {
      const [px] = toPx(x, 0); ctx.moveTo(px, 0); ctx.lineTo(px, H);
    }
    for (let y = Math.floor((cy - half) / step) * step; y <= cy + half; y += step) {
      const [, py] = toPx(0, y); ctx.moveTo(0, py); ctx.lineTo(W, py);
    }
    ctx.stroke();
  }

  player(p, [px, py], W, H, dpr) {
    const { ctx } = this;
    const color = seatColor(p.seat);
    const inset = 9 * dpr;
    const outside = px < inset || py < inset || px > W - inset || py > H - inset;
    px = Math.min(W - inset, Math.max(inset, px));
    py = Math.min(H - inset, Math.max(inset, py));
    const heading = (-p.yaw * Math.PI) / 180;      // yaw is counter-clockwise from +x; the canvas y axis points down
    const me = p.seat === this.follow;
    if (!outside && p.alive) {
      const r = (me ? 34 : 26) * dpr;
      const half = (HFOV / 2) * (Math.PI / 180);
      const g = ctx.createRadialGradient(px, py, 0, px, py, r);
      g.addColorStop(0, `${color}55`);
      g.addColorStop(1, `${color}00`);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.arc(px, py, r, heading - half, heading + half);
      ctx.closePath();
      ctx.fill();
    }
    ctx.lineWidth = 1.5 * dpr;
    ctx.strokeStyle = 'rgba(255, 255, 255, .92)';
    ctx.fillStyle = p.alive ? color : '#636366';
    if (me) {
      const s = 7 * dpr;
      ctx.save();
      ctx.translate(px, py);
      ctx.rotate(heading);
      ctx.beginPath();
      ctx.moveTo(s * 1.35, 0);
      ctx.lineTo(-s * 0.8, s * 0.85);
      ctx.lineTo(-s * 0.35, 0);
      ctx.lineTo(-s * 0.8, -s * 0.85);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    } else {
      ctx.beginPath();
      ctx.arc(px, py, (outside ? 3.5 : 5) * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    if (!outside && !me) {
      ctx.font = `600 ${10.5 * dpr}px Inter, system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(245, 245, 247, .92)';
      ctx.shadowColor = 'rgba(0, 0, 0, .8)';
      ctx.shadowBlur = 3 * dpr;
      ctx.fillText(p.label || `P${p.seat + 1}`, px, py - 9 * dpr);
      ctx.shadowBlur = 0;
    }
  }
}

// A wire state (demo/engine.py PlayerState.wire) as the minimap's player record.
export const fromWire = (w) => ({ seat: w[0], t: w[1], x: w[2], y: w[3], z: w[4], yaw: w[5], pitch: w[6], alive: !!w[7] });
