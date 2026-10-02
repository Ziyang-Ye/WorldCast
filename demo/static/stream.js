// A worker's frame stream on a canvas: WebSocket in, JPEG decode, a small jitter buffer, an even 16 fps out.

import { median } from './common.js';

const PING_MS = 1000;
// The buffer target grows by a frame after a stall (up to EXTRA more) and shrinks after CALM_MS without one.
const EXTRA = 2;
const CALM_MS = 8000;
// Frames arrive a block at a time, so the queue saw-tooths. Only when even its lowest point over TROUGH_MS stays above
// the target is there surplus delay, and playback then runs a little faster until it is gone.
const TROUGH_MS = 2000;

export class FrameStream {
  // url: the worker's play or watch socket; canvas: where frames go.
  // onFrame(header, {shown, recv}) after a frame is drawn; onText(message) for JSON messages.
  constructor(url, canvas, { fps = 16, buffer = 2, onFrame = () => {}, onText = () => {}, onClose = () => {} } = {}) {
    Object.assign(this, { canvas, fps, onFrame, onText, onClose });
    this.base = Math.max(1, buffer);
    this.target = this.base;
    this.calmSince = performance.now();
    this.period = 1000 / fps;
    this.queue = [];
    this.playing = false;
    this.next = 0;
    this.shownAt = [];
    this.depths = [];
    this.stalls = [];
    this.rtts = [];
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';
    this.ws.onmessage = (e) => (typeof e.data === 'string' ? this.text(JSON.parse(e.data)) : this.binary(e.data));
    this.ws.onclose = (e) => { this.closed = true; this.onClose(e); };
    this.pinger = setInterval(() => this.send({ t: 'ping', c: performance.now() }), PING_MS);
    this.raf = requestAnimationFrame((t) => this.tick(t));
    this.resize = new ResizeObserver(() => this.fit());
    this.resize.observe(canvas);
  }

  send(message) {
    if (this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(message));
  }

  close() {
    clearInterval(this.pinger);
    cancelAnimationFrame(this.raf);
    this.resize.disconnect();
    this.ws.onclose = null;
    this.ws.close();
    this.queue.forEach((f) => f.bitmap.close());
  }

  text(message) {
    if (message.t === 'pong') {
      this.rtts.push(performance.now() - message.c);
      if (this.rtts.length > 5) this.rtts.shift();
    }
    this.onText(message);
  }

  async binary(data) {
    const recv = performance.now();
    const n = new DataView(data).getUint32(0);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 4, n)));
    const bitmap = await createImageBitmap(new Blob([new Uint8Array(data, 4 + n)], { type: 'image/jpeg' }));
    if (this.closed) return bitmap.close();
    const frame = { header, bitmap, recv };
    let i = this.queue.length;               // decodes may finish out of order: keep the queue sorted by frame
    while (i > 0 && this.queue[i - 1].header.k > header.k) i -= 1;
    this.queue.splice(i, 0, frame);
  }

  fit() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const { width, height } = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.round(width * dpr));
    this.canvas.height = Math.max(1, Math.round(height * dpr));
    if (this.last) this.draw(this.last, false);
  }

  draw(bitmap, keep = true) {
    this.ctx.imageSmoothingQuality = 'high';
    this.ctx.drawImage(bitmap, 0, 0, this.canvas.width, this.canvas.height);
    if (keep) {
      if (this.last) this.last.close();
      this.last = bitmap;
    }
  }

  // Playout: start once `target` frames wait, then one frame per period; an empty queue is a stall (rebuffer).
  tick(now) {
    this.raf = requestAnimationFrame((t) => this.tick(t));
    if (!this.playing && this.queue.length >= this.target) {
      this.playing = true;
      this.next = now;
    }
    if (!this.playing || now < this.next - 2) return;
    const frame = this.queue.shift();
    if (!frame) {
      this.playing = false;
      this.stalls.push(now);
      this.target = Math.min(this.base + EXTRA, this.target + 1);
      this.calmSince = now;
      return;
    }
    if (now - this.calmSince > CALM_MS && this.target > this.base) {
      this.target -= 1;
      this.calmSince = now;
    }
    this.draw(frame.bitmap);
    this.next += this.period;
    if (now - this.next > this.period) this.next = now + this.period;
    this.depths.push([now, this.queue.length]);
    while (now - this.depths[0][0] > TROUGH_MS) this.depths.shift();
    const trough = Math.min(...this.depths.map((d) => d[1]));
    if (now - this.depths[0][0] > TROUGH_MS * 0.9 && trough > this.target) this.next -= this.period * 0.25;
    this.shownAt.push(now);
    while (this.shownAt.length && now - this.shownAt[0] > 1000) this.shownAt.shift();
    this.onFrame(frame.header, { shown: now, recv: frame.recv });
  }

  // fps shown, round trip (ms), stalls in the last 10 s, and a 1-3 quality score.
  stats() {
    const now = performance.now();
    this.stalls = this.stalls.filter((t) => now - t < 10000);
    const rtt = median(this.rtts);
    const stalls = this.stalls.length;
    const byRtt = Number.isNaN(rtt) ? 2 : rtt < 60 ? 3 : rtt < 150 ? 2 : 1;
    const byStall = stalls === 0 ? 3 : stalls <= 3 ? 2 : 1;
    return { fps: this.shownAt.length, rtt, stalls, quality: Math.min(byRtt, byStall), buffered: this.queue.length,
      target: this.target };
  }
}
