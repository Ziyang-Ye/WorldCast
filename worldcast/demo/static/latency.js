// Input to photon in the play view: from each key press, click or mouse-move onset to the first frame that folds in
// its input sample, split into its parts with the times the worker reports per frame.

import { esc, median } from './common.js';

// How the own position is known (a frame header's st, set by the client from its estimate of the position), in a plain
// sentence. The estimate follows the player's movement as far as the generated view followed it, corrected by the
// model's reading of its own video.
const POSITIONS = {
  guarded: "Your position follows your keys, corrected by the model's reading of where you are, which is checked before it is used.",
  reacquiring: "The model's reading of where you are has disagreed with your keys for some time, and your position is being moved to it, a little each frame.",
  degraded: "The model's reading of where you are keeps disagreeing with your movement, so it is ignored for now and your position follows your movement as far as the view followed it.",
};
const PARTS = [['input', 'Sampling', '#f5f5f7'], ['net', 'Network', '#64d2ff'],
  ['wait', 'Waiting for the next block', '#8e8e93'], ['gen', 'Generating', '#bf5af2'], ['enc', 'Encoding', '#ffd60a'],
  ['buf', 'Jitter buffer', '#30d158']];
const KEPT = 24;              // inputs the HUD's median and the breakdown run over
const SAMPLES_KEPT_MS = 10000;     // how long a sample's send time is remembered, whatever the display's rate
const ONSETS_KEPT = 64;       // onsets that wait for their frame

export class LatencyMeter {
  constructor() {
    this.sent = new Map();     // sample -> when it was sent
    this.onsets = [];          // {seq, t}: an onset and the sample that carries it, until its frame is shown
    this.totals = [];          // onset to photon of the last inputs, ms
    this.parts = [];           // their breakdowns
    this.fresh = [];           // from a sample sent to the first frame that folds it in (end to end, on a phone)
    this.lastQ = -1;
  }

  // Sample `seq` goes out at `now`; onsets: when the inputs it carries began.
  sample(seq, now, onsets) {
    this.sent.set(seq, now);
    for (const [old, at] of this.sent) {
      if (now - at <= SAMPLES_KEPT_MS) break;
      this.sent.delete(old);
    }
    for (const t of onsets) this.onsets.push({ seq, t });
    while (this.onsets.length > ONSETS_KEPT) this.onsets.shift();
  }

  // A frame with header h was shown at t.shown, having arrived at t.recv. The worker times the frame's newest sample
  // q (h.tm: its wait for the block's controls, the generation, the encoding); an earlier sample e waited
  // (sent q - sent e) longer for the block.
  frame(h, t) {
    if (h.q > this.lastQ && this.sent.has(h.q)) {
      this.lastQ = h.q;
      this.fresh.push(t.shown - this.sent.get(h.q));
      if (this.fresh.length > KEPT) this.fresh.shift();
    }
    while (this.onsets.length && this.onsets[0].seq <= h.q) {
      const onset = this.onsets.shift();
      const total = t.shown - onset.t;
      this.totals.push(total);
      if (this.totals.length > KEPT) this.totals.shift();
      const sentE = this.sent.get(onset.seq);
      const sentQ = this.sent.get(h.q);
      if (h.tm[0] === null || sentE === undefined || sentQ === undefined) continue;
      const [waitQ, gen, enc] = h.tm;
      const part = { total, input: sentE - onset.t, wait: waitQ + sentQ - sentE, gen, enc, buf: t.shown - t.recv };
      part.net = Math.max(0, total - part.input - part.wait - gen - enc - part.buf);
      this.parts.push(part);
      if (this.parts.length > KEPT) this.parts.shift();
    }
  }

  // The HUD's number, ms (NaN: none yet): the median input to photon, or end to end on a device without inputs.
  latency(endToEnd) {
    return median(endToEnd ? this.fresh : this.totals);
  }

  // The breakdown panel's markup. rtt: the socket's round trip, ms; framesPerBlock: the client's block; position: how
  // the own position is known (a frame header's st), with its meaning in a sentence.
  panel(rtt, framesPerBlock, position) {
    const mean = (k) => this.parts.reduce((a, p) => a + p[k], 0) / this.parts.length;
    const parts = PARTS.map(([k, label, c]) => [label, c, mean(k)]);
    const total = mean('total');
    const ok = this.parts.length > 0;
    return `<h4>INPUT → PHOTON${ok ? ` · MEAN ${Math.round(total)} MS` : ''}</h4>
      <div class="lat-bar">${ok ? parts.map(([, c, v]) => `<i style="background:${c};width:${(100 * v) / total}%"></i>`).join('') : ''}</div>
      <div class="lat-rows">${parts.map(([label, c, v]) => `<span class="sw" style="background:${c}"></span><span>${label}</span>
        <span class="v">${ok ? Math.round(v) : '—'}</span>`).join('')}
        <span></span><span>Round trip</span><span class="v">${Number.isNaN(rtt) ? '—' : Math.round(rtt)}</span></div>
      <p class="lat-note">${ok ? `From each key press, click or mouse-move onset to the first frame that shows it, over the
        last ${this.parts.length} input${this.parts.length === 1 ? '' : 's'} (the HUD shows their median).` : `Press a key, click or move the mouse: the
        breakdown fills in as the inputs reach a frame.`} A block is ${framesPerBlock} frames: an input waits for the
        next block to start.</p>
      <h4 class="lat-pos">YOUR POSITION${position ? ` · ${esc(position).toUpperCase()}` : ''}</h4>
      <p class="lat-note">${esc(Object.hasOwn(POSITIONS, position) ? POSITIONS[position] : position ? `The client reports “${position}”.` : 'It shows with the first frames.')}</p>`;
  }
}
