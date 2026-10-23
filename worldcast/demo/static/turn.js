// The instant turn: the play view turns with the player's hand at once, ahead of the frames that hold the turn. A turn
// about the eye needs no depth: each pixel of the view at the player's orientation now shows what was seen along the
// same ray, in the newest frame (taken at its own orientation, its header's tn and cp) or in a recent one. The frames
// shown are kept in a panorama, an equirectangular texture over every direction (yaw 360 degrees, pitch -90 to 90),
// each written at its own orientation over the older ones, with the frame each texel comes from. A texel is used while
// its frame is recent (PANORAMA_AGE_S) and the player has stayed near where it was taken (PANORAMA_RADIUS_U: a move
// changes the parallax, which a turn cannot show). No pixel is invented: where neither the newest frame nor the
// panorama has the ray, the view turns only as far as the known pixels reach, and the rest of the turn shows when the
// frames that hold it arrive. The first-person viewmodel stays where the frame has it (VIEWMODEL): the newest frame's
// pixels there are drawn as they are, over the turned view, and never go into the panorama, but for those that match
// what the panorama saw there beside the viewmodel (the scene behind it, which turns with the view). While the viewmodel
// may be away from its pieces (a knife carried at a run, a shot, a reload, a weapon drawn, an inspect: ViewmodelMotion,
// from the buttons and the weapon of the frame's row of the controls), the panorama takes only the band of the frame
// above where it may reach, or none of the frame. Moving (WASD) is not shown ahead: without depth only a turn is. A
// frame that holds the whole turn is drawn as it was generated.
//
// A view turned between the orientations of the frames it is drawn from lacks a sliver at its edges (a frame's top
// and bottom edges are great circles, lower at its sides than at its centre, and the frames' pitches differ a little):
// the turned view is then magnified by the least that puts what it lacks past the canvas, up to MAX_MAGNIFY, and eases
// back as the frames catch up (MAGNIFY_EASE_S); more than that is lacking, the turn stops short.
//
// Two halves, one geometry: the GPU (WebGL2) writes and reads the panorama per texel and per pixel; the CPU tests a
// view against the frames before it is drawn (Coverage): every pixel of it is to be in the newest frame or, a margin of
// half a texel inside, in a frame of the panorama in use. Directions are in the camera model of the generator's camera
// (worldcast.data.camera): the frame spans the angles whose tangents are page.camera_tans across and up; yaw counter-
// clockwise from +x and pitch > 0 looking down, as the frame header's tn and cp, and the input's turn (pitch > 0 looks
// down, yaw > 0 turns left).

import { page } from './common.js';
import { MOVEMENT_BUTTONS } from './input.js';

// The setting's default: off, the view turns with the generated frames only.
export const INSTANT_TURN_DEFAULT = false;
// A frame in the panorama is used for this long after it was shown, s: a few turns back and forth, while what it saw
// has seldom changed.
export const PANORAMA_AGE_S = 8;
// ... and while every frame shown since was taken within this distance of where the player is now (the newest frame's
// own position), engine units: a step of 16 u moves a wall 300 u away by 3 degrees at most.
export const PANORAMA_RADIUS_U = 16;
// The panorama's texels per degree (fewer where the GPU's textures are smaller): the frames have 4.4 pixels a degree
// at their centre and 12 at their sides.
export const PANORAMA_TEXELS_PER_DEGREE = 8;
// The first-person viewmodel (the weapon and the hands), fixed on the screen as in the game: convex pieces, their
// vertices as fractions of the frame from its top left, reaching past the frame where a piece meets its edge. The
// weapon held in the right hand fills the right piece from the bottom right corner (rifles, pistols, knives at rest),
// the left hand the left piece (a knife's run). Drawn over the generated frames of the examples (examples/), the
// pieces hold the viewmodel but in an inspect or draw animation (a raised knife) and a viewmodel held on the left.
export const VIEWMODEL = [
  [[1.3, 0.34], [0.8, 0.45], [0.63, 0.61], [0.52, 0.81], [0.47, 1.3], [1.3, 1.3]],
  [[-0.3, 0.7], [0.14, 0.76], [0.26, 0.92], [0.28, 1.3], [-0.3, 1.3]],
];
// The viewmodel's soft edge, frame pixels: the newest frame's pixels fade into the turned view over it.
const VIEWMODEL_FEATHER_PX = 8;
// While the viewmodel moves it may be anywhere below this, as a fraction of the frame from its top (a knife swung at a
// run, a gun's reload, a weapon drawn; the examples' generated frames): the panorama then takes the band above it.
export const VIEWMODEL_TOP = 0.42;
// How long the viewmodel may move after the buttons that move it are held, s, in the frames' rows: a knife's slash and
// a shot (attack, attack2), a reload, a weapon drawn (its weapon changes), a knife carried at a run (a movement button
// held with a knife); an inspect (look_at_weapon, a knife's) may take it anywhere, the panorama then takes nothing.
export const VIEWMODEL_MOVES_S = { attack: 0.5, attack2: 0.7, reload: 3.5, weapon: 1.2, run: 0.4 };
export const VIEWMODEL_INSPECT_S = 5;
// A pixel of the newest frame in the viewmodel's pieces is the scene where the panorama has the same around it (the
// mean difference over 3 x 3 pixels, 2 apart, as a share of the full scale): the scene below SCENE_DIFFERENCE (JPEG and
// the panorama's texels), the viewmodel above VIEWMODEL_DIFFERENCE, a mix between.
const SCENE_DIFFERENCE = 0.04;
const VIEWMODEL_DIFFERENCE = 0.1;
// The newest frame's own pixels count as the scene this far inside the viewmodel's pieces, radians (about 6 pixels of
// the frame there): the pieces bound the viewmodel with room to spare, and the test of a view, a pixel at a time, needs
// twice a pixel's diagonal to pass the newest frame itself.
const VIEWMODEL_INSET = 0.015;
// The most the turned view is magnified so that what it lacks at its edges is past the canvas, and how long it takes
// to ease from there back to none, s. A look around while standing, the pitch wandering by 2 degrees, has the view at
// the hand 73% of the time at 2%, 80% at 3% and 84% at 5%.
export const MAX_MAGNIFY = 1.03;
const MAGNIFY_EASE_S = 0.25;
// The test of a view: blocks of BLOCK_CELLS x BLOCK_CELLS cells of CELL_PX pixels, each all in one frame or not
// needed (all in the viewmodel); a cell that is neither, its pixels one by one.
const CELL_PX = 4;
const BLOCK_CELLS = 8;
// Frames whose forward axes are further than this from the view's (its cosine) cannot reach it: twice the angle from a
// frame's centre to its corner, 57 degrees.
const REACH_COS = Math.cos(115 * (Math.PI / 180));
// A frame shown at the orientation of a frame before it, within this many degrees, takes its place in the test.
const SAME_DEGREES = 0.01;
// A ray is inside a constraint of a frame of the panorama where it is more than this inside it (the display pass's
// float32).
const INSIDE = 1e-6;
// A turn of less than this, degrees, is none.
const NO_TURN_DEGREES = 1e-4;
// The search for the farthest turn shown: steps along the turn of at most this many degrees from the whole turn down,
// at most SEARCH_STEPS of them, then halvings between the last step shown and the one after it.
const SEARCH_STEP_DEGREES = 10;
const SEARCH_STEPS = 8;
const SEARCH_HALVINGS = 7;
// A WebGL context that is lost and not given back within this many ms (in a page in view) is given up: the frames are
// then drawn as they were generated.
const RESTORE_WAIT_MS = 3000;
// The most pieces and edges of the viewmodel the shaders take.
const MAX_PIECES = 4;
const MAX_EDGES = 16;

const RAD = Math.PI / 180;
const glsl = (x) => (Number.isInteger(x) ? x.toFixed(1) : String(x));

// ---------------------------------------------------------------------------------------------------------- geometry
// The rotation of a camera at `yaw` and `pitch` degrees, row-major: its columns are the camera's right, up and forward
// axes in the world (x, y horizontal, z up), so a camera ray (x right, y up, z forward) r is R r in the world.
export function rotation(yaw, pitch) {
  const [cy, sy, cp, sp] = [Math.cos(yaw * RAD), Math.sin(yaw * RAD), Math.cos(pitch * RAD), Math.sin(pitch * RAD)];
  return [sy, cy * sp, cy * cp, -cy, sy * sp, sy * cp, 0, cp, -sp];
}

const transpose = (a) => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
const apply = (a, v) => [0, 3, 6].map((i) => a[i] * v[0] + a[i + 1] * v[1] + a[i + 2] * v[2]);
function product(a, b) {
  const out = new Array(9);
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) out[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
  }
  return out;
}
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => { const n = Math.hypot(...a); return a.map((x) => x / n); };
const centroid = (piece) => [0, 1].map((i) => piece.reduce((s, p) => s + p[i], 0) / piece.length);

// The camera ray through the point (u, v) of the frame, fractions from its top left.
const frameRay = ([u, v], [across, up]) => [(2 * u - 1) * across, (1 - 2 * v) * up, 1];

// The planes through the eye along the frame's edges, as unit normals into the frame (camera space): a unit ray d is in
// the frame where every n . d >= 0, and n . d is the sine of its angle from that edge.
function frameEdges([across, up]) {
  const [h, v] = [Math.hypot(1, across), Math.hypot(1, up)];
  return [[-1 / h, 0, across / h], [1 / h, 0, across / h], [0, -1 / v, up / v], [0, 1 / v, up / v]];
}

// The planes through the eye along the edges of each piece of the viewmodel, as unit normals out of it (camera
// space): a unit ray is outside a piece where n . d > 0 for one of its edges, and outside the viewmodel where it is
// outside each.
function viewmodelPlanes(pieces, tans) {
  return pieces.map((piece) => {
    const centre = frameRay(centroid(piece), tans);
    return piece.map((vertex, i) => {
      const n = unit(cross(frameRay(vertex, tans), frameRay(piece[(i + 1) % piece.length], tans)));
      return dot(n, centre) > 0 ? n.map((x) => -x) : n;
    });
  });
}

// The edges of each piece of the viewmodel on a view of `size` pixels, as lines [nx, ny, x, y] (a unit normal out of
// the piece and a point): a pixel point p is inside a piece where n . (p - point) <= 0 for each of its lines.
function viewmodelLines(pieces, [width, height]) {
  return pieces.map((piece) => {
    const [cx, cy] = centroid(piece);
    return piece.map(([u, v], i) => {
      const [u2, v2] = piece[(i + 1) % piece.length];
      const [ax, ay, bx, by] = [u * width, v * height, u2 * width, v2 * height];
      const length = Math.hypot(bx - ax, by - ay);
      const [nx, ny] = [(by - ay) / length, -(bx - ax) / length];
      const out = nx * (cx * width - ax) + ny * (cy * height - ay) > 0 ? -1 : 1;
      return [out * nx, out * ny, ax, ay];
    });
  });
}

// The viewmodel's weight at each pixel of a view of `size`, 0-255 (255 inside a piece, falling to 0 over the soft
// edge outside it), row by row from the top.
function viewmodelMask(pieces, size) {
  const [width, height] = size;
  const lines = viewmodelLines(pieces, size);
  const mask = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let weight = 0;
      for (const piece of lines) {
        let distance = -Infinity;
        for (const [nx, ny, ax, ay] of piece) distance = Math.max(distance, nx * (x + 0.5 - ax) + ny * (y + 0.5 - ay));
        weight = Math.max(weight, distance <= 0 ? 1 : Math.max(0, 1 - distance / VIEWMODEL_FEATHER_PX));
      }
      mask[y * width + x] = Math.round(255 * weight);
    }
  }
  return mask;
}

// What of each frame the panorama takes, by what the viewmodel may be doing in it, from the buttons and the weapon of
// the frame's row of the controls (its header's `b` and `w`): all but the viewmodel's pieces ('outside'), the band above
// VIEWMODEL_TOP while it may be moving ('above'), none of it while it may be anywhere ('none'). `welcome`: the worker's
// (the bit of each button of `b`, the knives, the frames a second).
export class ViewmodelMotion {
  constructor(welcome) {
    this.bit = Object.fromEntries(welcome.buttons.map((b, i) => [b, 1 << i]));
    this.knives = new Set(welcome.inspect_weapons);
    this.fps = welcome.fps;
    this.since = {};                         // what moved the viewmodel -> the newest frame it was held in
    this.weapon = null;
  }

  // The part of the frame of `header` (its index k) the panorama takes.
  part({ k, b = 0, w = 0 }) {
    const held = (name) => (b & this.bit[name]) !== 0;
    const saw = (name) => { this.since[name] = Math.max(this.since[name] ?? -Infinity, k); };
    for (const name of ['attack', 'attack2', 'reload', 'look_at_weapon']) if (held(name)) saw(name);
    if (this.weapon !== null && w !== this.weapon) saw('weapon');
    this.weapon = w;
    if (this.knives.has(w) && MOVEMENT_BUTTONS.some(held)) saw('run');
    const within = (name, seconds) => k - (this.since[name] ?? -Infinity) < seconds * this.fps;
    if (within('look_at_weapon', VIEWMODEL_INSPECT_S)) return 'none';
    return Object.entries(VIEWMODEL_MOVES_S).some(([name, seconds]) => within(name, seconds)) ? 'above' : 'outside';
  }
}

// ---------------------------------------------------------------------------------------------------------- the CPU
// The frames the view is drawn from and the test of a view against them. `tans`: the camera's; `size`: the view's
// pixels [width, height]; `margin`: how far inside a frame of the panorama a direction is to be, radians, so that the
// panorama has a texel of the frame next to it (half a texel's diagonal).
export class Coverage {
  constructor(tans, size, margin) {
    Object.assign(this, { tans, size, margin });
    this.edges = frameEdges(tans);
    this.pieces = viewmodelPlanes(VIEWMODEL, tans);
    this.mask = viewmodelMask(VIEWMODEL, size);
    // the cells of CELL_PX all of whose pixels are the viewmodel's, row by row
    const [width, height] = size;
    this.still = [];
    for (let y = 0; y < height; y += CELL_PX) {
      for (let x = 0; x < width; x += CELL_PX) this.still.push(this.allStill(x, y, CELL_PX));
    }
    // the planes of a frame's constraints in one list: its edges, then its viewmodel's piece after piece, ending at
    // `ends`, then the band above VIEWMODEL_TOP (a unit normal into it)
    const [, up] = tans;
    this.band = unit([0, 1, -(1 - 2 * VIEWMODEL_TOP) * up]);
    this.normals = [...this.edges, ...this.pieces.flat(), this.band];
    this.ends = [];
    this.pieces.reduce((end, piece) => { this.ends.push(end + piece.length); return end + piece.length; }, 4);
    this.frames = [];          // {serial, shown, at, yaw, pitch, toWorld, forward}, by serial; one per orientation
  }

  // A frame shown, the newest: `serial` (the panorama's), shown at `shown` (ms), taken at {yaw, pitch, at}, the
  // panorama taking `part` of it (ViewmodelMotion). The frames shown longer ago than PANORAMA_AGE_S are let go, and one
  // before it at its orientation of which the panorama took no more (the new one's texels are in its place).
  add(serial, shown, pose) {
    const { yaw, pitch, at, part } = pose;
    const toWorld = rotation(yaw, pitch);
    const covers = { outside: ['outside', 'above', 'none'], above: ['above', 'none'], none: ['none'] }[part];
    this.frames = this.frames.filter((f) => shown - f.shown <= PANORAMA_AGE_S * 1000 && !(covers.includes(f.part)
      && Math.abs(f.yaw - yaw) < SAME_DEGREES && Math.abs(f.pitch - pitch) < SAME_DEGREES));
    this.frames.push({ serial, shown, at, yaw, pitch, part, toWorld, forward: [toWorld[2], toWorld[5], toWorld[8]] });
  }

  // The newest frame.
  get newest() {
    return this.frames[this.frames.length - 1];
  }

  // The oldest frame of the panorama in use at `now`: every frame shown since was shown within PANORAMA_AGE_S and
  // taken within PANORAMA_RADIUS_U of the newest frame's position.
  oldest(now) {
    const here = this.newest.at;
    let oldest = this.newest.serial;
    for (let i = this.frames.length - 1; i >= 0; i -= 1) {
      const { serial, shown, at } = this.frames[i];
      if (now - shown > PANORAMA_AGE_S * 1000 || Math.hypot(at[0] - here[0], at[1] - here[1], at[2] - here[2])
        > PANORAMA_RADIUS_U) break;
      oldest = serial;
    }
    return oldest;
  }

  // The least magnification of the view at `view` (its rotation), drawn at `magnify` already, that has every pixel it
  // shows outside the viewmodel from a frame: the newest one, or one of the panorama's from `oldest` on. 1: none more;
  // Infinity: more than `most`. A pixel of the view at a magnification shows the ray of the point that much nearer its
  // centre; the viewmodel stays. The view is taken in blocks, the blocks not settled in cells, the cells not settled
  // pixel by pixel. A block or a cell is settled where its corners are all in one piece of the view's viewmodel (it
  // needs nothing), or where their rays are all in one frame (`within`: each region is convex, so every ray between
  // the corners is in it too); a pixel where its centre is, as the display pass takes it. A pixel from no frame is put
  // past the canvas by magnifying the view; one that cannot be within `most` ends the test. `order`: the blocks
  // [x, y] in the order they are tested (blocks).
  covered(view, oldest, magnify, most, order = this.blocks()) {
    const [width, height] = this.size;
    const [across, up] = this.tans;
    const { margin, newest, normals, ends } = this;
    const fromWorld = transpose(view);
    const forward = [view[2], view[5], view[8]];
    const frames = [];
    for (const f of this.frames) {
      const own = f === newest;
      if (f.serial < oldest || dot(forward, f.forward) < REACH_COS || (!own && f.part === 'none')) continue;
      const toView = product(fromWorld, f.toWorld);
      const planes = new Float64Array(3 * normals.length);
      normals.forEach((n, i) => planes.set(apply(toView, n), 3 * i));
      frames.push({ own, above: !own && f.part === 'above', near: own ? 2 : dot(forward, f.forward), planes });
    }
    frames.sort((a, b) => b.near - a.near);
    const rays = new Float64Array(12);       // the rays tested: a square's corners or a pixel's centre
    let count = 0;
    // Whether every ray between the `count` rays is in frame f: inside its edges and outside its viewmodel (and in the
    // band above VIEWMODEL_TOP, of a frame the panorama took that of), constraint by constraint the least over the rays
    // (for a piece of the viewmodel, its edge with the most). The newest frame is drawn from itself, its viewmodel but
    // for VIEWMODEL_INSET; a frame of the panorama through the four texels around
    // the ray: one of them is in the frame where the ray is inside all of its constraints but one by `margin` (half a
    // texel's diagonal) and inside that one at all (a texel square across a straight edge has a corner on each side).
    const within = (f) => {
      const p = f.planes;
      const least = (e) => {
        let s = Infinity;
        for (let j = 0; j < 3 * count; j += 3) {
          s = Math.min(s, p[3 * e] * rays[j] + p[3 * e + 1] * rays[j + 1] + p[3 * e + 2] * rays[j + 2]);
        }
        return s;
      };
      let tight = 0;
      for (let e = 0; e < 4; e += 1) {
        const s = least(e);
        if (f.own ? s < 0 : s <= INSIDE) return false;
        if (s < margin) tight += 1;
      }
      let e = 4;
      for (const end of ends) {
        let s = -Infinity;
        for (; e < end; e += 1) s = Math.max(s, least(e));
        if (s <= (f.own ? -VIEWMODEL_INSET : INSIDE)) return false;
        if (s < margin) tight += 1;
      }
      if (f.above) {
        const s = least(e);
        if (s <= INSIDE) return false;
        if (s < margin) tight += 1;
      }
      return f.own || tight <= 1;
    };
    let last = 0;                            // the frame of the last square found: its neighbours mostly are in it too
    const found = () => {
      if (last < frames.length && within(frames[last])) return true;
      for (let i = 0; i < frames.length; i += 1) {
        if (i !== last && within(frames[i])) {
          last = i;
          return true;
        }
      }
      return false;
    };
    const add = (x, y) => {
      const a = (((2 * x) / width - 1) * across) / magnify;
      const b = ((1 - (2 * y) / height) * up) / magnify;
      const n = Math.sqrt(a * a + b * b + 1);
      rays[3 * count] = a / n;
      rays[3 * count + 1] = b / n;
      rays[3 * count + 2] = 1 / n;
      count += 1;
    };
    // the square of `size` pixels at (x, y) settled: all in the view's viewmodel (`still`), or its corners' rays in a
    // frame
    const settled = (x, y, size, still) => {
      if (still) return true;
      const [x1, y1] = [Math.min(x + size, width), Math.min(y + size, height)];
      count = 0;
      add(x, y);
      add(x1, y);
      add(x, y1);
      add(x1, y1);
      return found();
    };
    // the pixel at (x, y) settled: in the view's viewmodel, or its centre's ray in a frame
    const pixel = (x, y) => {
      if (this.mask[y * width + x] === 255) return true;
      count = 0;
      add(x + 0.5, y + 0.5);
      return found();
    };
    // the magnification that puts the square of `size` pixels at (x, y) past the canvas (its point nearest the centre)
    const reach = (a, size, n) => (2 * a < n && 2 * (a + size) > n ? 0
      : Math.min(Math.abs((2 * a) / n - 1), Math.abs((2 * (a + size)) / n - 1)));
    const past = (x, y, size) => {
      const out = Math.max(reach(x, size, width), reach(y, size, height));
      return out > 0 ? 1 / out : Infinity;
    };
    const block = CELL_PX * BLOCK_CELLS;
    const cellsAcross = Math.ceil(width / CELL_PX);
    let need = 1;                            // a square past the canvas at this magnification is not tested
    for (const [bx, by, still] of order) {
      if (past(bx, by, block) <= need || settled(bx, by, block, still)) continue;
      for (let y = by; y < Math.min(by + block, height); y += CELL_PX) {
        for (let x = bx; x < Math.min(bx + block, width); x += CELL_PX) {
          const cell = this.still[(y / CELL_PX) * cellsAcross + x / CELL_PX];
          if (past(x, y, CELL_PX) <= need || settled(x, y, CELL_PX, cell)) continue;
          for (let py = y; py < Math.min(y + CELL_PX, height); py += 1) {
            for (let px = x; px < Math.min(x + CELL_PX, width); px += 1) {
              const hidden = past(px, py, 1);
              if (hidden <= need || pixel(px, py)) continue;
              if (hidden > most) return Infinity;
              need = hidden;
            }
          }
        }
      }
    }
    return need;
  }

  // The blocks of the view's test, [x, y, still] (pixels, and whether all its pixels are the viewmodel's), those on the
  // side of `lead` first: a direction across the view [x right, y up], where a turn brings what the view has not seen.
  blocks(lead = [0, 0]) {
    const [width, height] = this.size;
    const block = CELL_PX * BLOCK_CELLS;
    const order = [];
    for (let y = 0; y < height; y += block) {
      for (let x = 0; x < width; x += block) {
        const side = lead[0] * (x + block / 2 - width / 2) - lead[1] * (y + block / 2 - height / 2);
        order.push([x, y, this.allStill(x, y, block), side]);
      }
    }
    return order.sort((a, b) => b[3] - a[3]);
  }

  // Whether every pixel of the square of `size` pixels at (x, y) is the viewmodel's (its weight whole: the display pass
  // draws the newest frame's own pixel there).
  allStill(x, y, size) {
    const [width, height] = this.size;
    for (let py = y; py < Math.min(y + size, height); py += 1) {
      for (let px = x; px < Math.min(x + size, width); px += 1) if (this.mask[py * width + px] !== 255) return false;
    }
    return true;
  }
}

// ---------------------------------------------------------------------------------------------------------- the GPU
const VERTEX = `#version 300 es
in vec2 corner;
uniform vec4 rect;
out vec2 ndc;
void main() {
  ndc = mix(rect.xy, rect.zw, corner * 0.5 + 0.5);
  gl_Position = vec4(ndc, 0.0, 1.0);
}`;

// What both passes share: the frame's edges and the viewmodel's (camera space; frameEdges, viewmodelPlanes) and the
// point of the frame on a ray.
const PLANES = `#version 300 es
precision highp float;
precision highp int;
uniform vec2 tans;
uniform vec3 frameEdges[4];
uniform vec3 hull[${MAX_EDGES}];
uniform int pieceEnd[${MAX_PIECES}];
uniform int pieces;
in vec2 ndc;
float inFrame(vec3 d) {
  return min(min(dot(frameEdges[0], d), dot(frameEdges[1], d)), min(dot(frameEdges[2], d), dot(frameEdges[3], d)));
}
float outside(vec3 d) {
  float o = 1e9;
  int e = 0;
  for (int k = 0; k < ${MAX_PIECES}; k++) {
    if (k >= pieces) break;
    float side = -1e9;
    for (; e < pieceEnd[k]; e++) side = max(side, dot(hull[e], d));
    o = min(o, side);
  }
  return o;
}
vec2 framePoint(vec3 d) { return vec2(0.5 + 0.5 * d.x / (d.z * tans.x), 0.5 - 0.5 * d.y / (d.z * tans.y)); }
`;

// The panorama's write: each texel of the frame's footprint whose direction is in the frame, outside its viewmodel and
// above `top` (a fraction of the frame from its top) takes the frame's colour and its serial (24 bits in rgb, a = 1:
// written).
const WRITE = `${PLANES}
uniform sampler2D frame;
uniform mat3 toFrame;
uniform vec3 code;
uniform vec2 panorama;
uniform float top;
layout(location = 0) out vec4 colour;
layout(location = 1) out vec4 serial;
void main() {
  float yaw = gl_FragCoord.x / panorama.x * 6.28318530718;
  float down = (gl_FragCoord.y / panorama.y - 0.5) * 3.14159265359;
  vec3 d = toFrame * vec3(cos(yaw) * cos(down), sin(yaw) * cos(down), -sin(down));
  if (inFrame(d) < 0.0 || outside(d) <= 0.0) discard;
  vec2 point = framePoint(d);
  if (point.y >= top) discard;
  colour = vec4(texture(frame, point).rgb, 1.0);
  serial = vec4(code, 1.0);
}`;

// The view: each pixel outside the viewmodel from the newest frame where its ray (the point `magnify` times nearer the
// centre) is in it, outside the frame's viewmodel but for the inset, else from the panorama, its four nearest texels of
// frames from `oldest` on; the viewmodel's pixels are the newest frame's own, at its soft edge mixed with the turned
// view. In the viewmodel's pieces a pixel that is the scene behind them (`held`) turns as the view does where a frame
// has it. The motion feedback scales and shifts the whole view (mirrored past the edges, as the 2D canvas draws it).
// debug 1: a pixel from no frame is magenta; debug 2: each pixel says where it is from (r the newest frame, g the
// panorama, b none, a the viewmodel's weight).
const DISPLAY = `${PLANES}
uniform sampler2D frame;
uniform sampler2D colours;
uniform sampler2D serials;
uniform sampler2D mask;
uniform mat3 turn;
uniform mat3 view;
uniform mat3 base;
uniform vec2 pixel;
uniform float inset;
uniform float oldest;
uniform float magnify;
uniform vec2 panorama;
uniform float zoom;
uniform vec2 shift;
uniform int debug;
out vec4 colour;
vec4 cached(vec3 w) {
  vec2 at = vec2(fract(atan(w.y, w.x) / 6.28318530718) * panorama.x,
    (asin(clamp(-w.z, -1.0, 1.0)) / 3.14159265359 + 0.5) * panorama.y) - 0.5;
  vec2 low = floor(at);
  vec2 f = at - low;
  ivec2 size = ivec2(panorama);
  vec3 sum = vec3(0.0);
  float weight = 0.0;
  for (int j = 0; j < 2; j++) {
    for (int i = 0; i < 2; i++) {
      ivec2 texel = ivec2(int(low.x) + i, int(low.y) + j);
      texel.x += texel.x < 0 ? size.x : (texel.x >= size.x ? -size.x : 0);
      texel.y = clamp(texel.y, 0, size.y - 1);
      vec4 s = texelFetch(serials, texel, 0);
      float serial = dot(floor(s.rgb * 255.0 + 0.5), vec3(1.0, 256.0, 65536.0));
      float w = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
      if (s.a > 0.5 && serial >= oldest && w > 0.0) {
        sum += w * texelFetch(colours, texel, 0).rgb;
        weight += w;
      }
    }
  }
  return weight > 0.0 ? vec4(sum / weight, 1.0) : vec4(0.0);
}
// How much of the newest frame's pixel at uv, in the viewmodel's pieces, is the viewmodel's: none where the panorama
// has what it has around it (the scene behind the pieces, seen by frames whose viewmodel was elsewhere), all where it
// differs, or where the panorama has not seen all of it.
float held(vec2 uv) {
  float diff = 0.0;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec2 q = clamp(uv + 2.0 * vec2(i, j) * pixel, 0.5 * pixel, 1.0 - 0.5 * pixel);
      vec4 behind = cached(base * normalize(vec3(vec2(2.0 * q.x - 1.0, 1.0 - 2.0 * q.y) * tans, 1.0)));
      if (behind.a == 0.0) return 1.0;
      diff += dot(abs(texture(frame, q).rgb - behind.rgb), vec3(1.0 / 3.0));
    }
  }
  return smoothstep(${glsl(SCENE_DIFFERENCE)}, ${glsl(VIEWMODEL_DIFFERENCE)}, diff / 9.0);
}
void main() {
  vec2 uv = vec2(0.5, -0.5) * (ndc - shift) / zoom + 0.5;
  uv = 1.0 - abs(1.0 - mod(uv, 2.0));
  vec2 p = vec2(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
  float piece = texture(mask, uv).r;
  float m = piece > 0.0 ? piece * held(uv) : 0.0;
  vec3 still = texture(frame, uv).rgb;
  vec3 scene = still;
  int source = 0;
  if (m < 1.0) {
    vec3 ray = normalize(vec3(p * tans / magnify, 1.0));
    vec3 d = turn * ray;
    float side = outside(d);
    if (inFrame(d) >= 0.0 && (side > -inset || held(framePoint(d)) < 1.0)) {
      scene = texture(frame, framePoint(d)).rgb;
      source = 1;
    } else {
      vec4 known = cached(view * ray);
      source = known.a > 0.0 ? 2 : 3;
      if (source == 2) scene = known.rgb;
    }
  }
  // the scene a piece of the viewmodel held and the panorama has not turned yet stays as it is
  bool nowhere = source == 3 && piece < 1.0;
  if (debug == 2) {
    colour = vec4(source == 1 ? 1.0 : 0.0, source == 2 ? 1.0 : 0.0, nowhere ? 1.0 : 0.0, m);
    return;
  }
  if (debug == 1 && nowhere) scene = vec3(1.0, 0.0, 1.0);
  colour = vec4(mix(scene, still, m), 1.0);
}`;

// The instant turn on a canvas (WebGL2): the panorama, the view drawn from it and the newest frame, and the Coverage
// that says how far the view may turn. A lost context (a phone that put the page away, a GPU reset) is asked back and
// set up again, the panorama empty; the view is drawn again at the next animation frame. One that does not come back
// within RESTORE_WAIT_MS, or whose programs no longer build, is given up: `giveUp()`.
export class InstantTurn {
  // The instant turn on `canvas`, or null where the browser has no WebGL2. A shader that does not compile, or a program
  // that does not link, gives it up (`giveUp`, at the first frame): `canvas` is then replaced by one that gives a 2D
  // context.
  static on(canvas, giveUp) {
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, depth: false });
    return gl ? new InstantTurn(gl, canvas, giveUp) : null;
  }

  constructor(gl, canvas, giveUp) {
    Object.assign(this, { gl, canvas, giveUp });
    this.size = [canvas.width, canvas.height];
    this.waiting = null;                     // the timer of a lost context's return
    canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();                    // so that the browser gives it back
      // a page put away gets it back when it shows: only the loss of a page in view is waited out
      if (!document.hidden) this.waiting = setTimeout(() => !document.hidden && giveUp(), RESTORE_WAIT_MS);
    });
    canvas.addEventListener('webglcontextrestored', () => {
      clearTimeout(this.waiting);
      this.init();
    });
    this.init();
  }

  // The quad, the textures and the programs: once, and again when a lost context comes back (the panorama then starts
  // empty). The programs are built without waiting for them (a GPU builds them while the page goes on, a software one
  // for a second or two); `ready()` takes them at the first frame.
  init() {
    const { gl } = this;
    const texels = Math.min(PANORAMA_TEXELS_PER_DEGREE, Math.floor(gl.getParameter(gl.MAX_TEXTURE_SIZE) / 360));
    this.panorama = [360 * texels, 180 * texels];
    this.coverage = new Coverage(page.camera_tans, this.size, (Math.SQRT1_2 / texels) * RAD);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const texture = (unit, filter, size, format, pixels = null) => {
      const t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      for (const [key, value] of [[gl.TEXTURE_MIN_FILTER, filter], [gl.TEXTURE_MAG_FILTER, filter],
        [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) {
        gl.texParameteri(gl.TEXTURE_2D, key, value);
      }
      const [internal, kind] = format === 'r' ? [gl.R8, gl.RED] : [gl.RGBA8, gl.RGBA];
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, size[0], size[1], 0, kind, gl.UNSIGNED_BYTE, pixels);
      return t;
    };
    this.frameTexture = texture(0, gl.LINEAR, this.size, 'rgba');
    const colours = texture(1, gl.NEAREST, this.panorama, 'rgba');
    const serials = texture(2, gl.NEAREST, this.panorama, 'rgba');
    texture(3, gl.LINEAR, this.size, 'r', this.coverage.mask);
    this.target = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.target);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, colours, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, serials, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.building = Object.fromEntries([['write', WRITE], ['display', DISPLAY]].map(([name, fragment]) => {
      const program = gl.createProgram();
      for (const [type, source] of [[gl.VERTEX_SHADER, VERTEX], [gl.FRAGMENT_SHADER, fragment]]) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        gl.attachShader(program, shader);
      }
      gl.linkProgram(program);
      return [name, program];
    }));
    this.programs = null;
    this.serial = 0;
    this.key = '';                           // what the view's turn was found for: the frame, the look, the panorama
    this.drawn = '';                         // what the canvas shows
    this.view = null;                        // the orientation {yaw, pitch} the view is drawn at
    this.reached = { share: 0, magnify: 1 };
    this.magnify = 1;                        // the magnification drawn, and when
    this.magnifiedAt = 0;
  }

  // Whether the programs are there to draw with: at the first call, once they are built (it waits for them), their
  // uniforms that do not change are set; a program that does not link gives the instant turn up.
  ready() {
    const { gl } = this;
    if (this.programs) return true;
    if (Object.values(this.building).some((program) => !gl.getProgramParameter(program, gl.LINK_STATUS))) {
      setTimeout(this.giveUp);
      return false;
    }
    this.programs = this.building;
    const { edges, pieces } = this.coverage;
    this.uniforms = {};
    for (const [name, program] of Object.entries(this.programs)) {
      gl.useProgram(program);
      const corner = gl.getAttribLocation(program, 'corner');
      gl.enableVertexAttribArray(corner);
      gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
      const at = (u) => gl.getUniformLocation(program, u);
      gl.uniform2fv(at('tans'), page.camera_tans);
      gl.uniform3fv(at('frameEdges'), edges.flat());
      gl.uniform3fv(at('hull'), pieces.flat(2));
      let end = 0;
      gl.uniform1iv(at('pieceEnd'), pieces.map((piece) => { end += piece.length; return end; }));
      gl.uniform1i(at('pieces'), pieces.length);
      gl.uniform1i(at('frame'), 0);
      gl.uniform2fv(at('panorama'), this.panorama);
      this.uniforms[name] = Object.fromEntries(['rect', 'toFrame', 'code', 'top', 'turn', 'view', 'base', 'oldest',
        'magnify', 'zoom', 'shift', 'debug'].map((u) => [u, at(u)]));
      if (name === 'display') {
        gl.uniform1i(at('colours'), 1);
        gl.uniform1i(at('serials'), 2);
        gl.uniform1i(at('mask'), 3);
        gl.uniform1f(at('inset'), 1.01 * VIEWMODEL_INSET);
        gl.uniform4f(at('rect'), -1, -1, 1, 1);
        gl.uniform2f(at('pixel'), 1 / this.size[0], 1 / this.size[1]);
      }
    }
    return true;
  }

  // The newest frame's orientation {yaw, pitch} and rotation, once a frame is shown.
  get base() {
    return this.coverage.newest;
  }

  // A frame shown: `bitmap` taken at `pose` {yaw, pitch, at: [x, y, z], part} at `now` (ms); it is the newest frame from
  // now on, and the panorama has the `part` of it ViewmodelMotion says.
  shown(bitmap, pose, now) {
    const { gl } = this;
    if (gl.isContextLost() || !this.ready()) return;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.frameTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    this.serial += 1;
    this.coverage.add(this.serial, now, pose);
    if (pose.part === 'none') return;
    const area = this.footprint(this.base.toWorld);
    const [width, height] = this.panorama;
    const per = width / 360;
    gl.useProgram(this.programs.write);
    const u = this.uniforms.write;
    gl.uniformMatrix3fv(u.toFrame, false, this.base.toWorld);    // column-major: its transpose, the world to the frame
    gl.uniform3fv(u.code, [0, 8, 16].map((bits) => ((this.serial >> bits) & 255) / 255));
    gl.uniform1f(u.top, pose.part === 'above' ? VIEWMODEL_TOP : 1);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.target);
    gl.viewport(0, 0, width, height);
    const [y0, y1] = [(area.down0 + 90) * per, (area.down1 + 90) * per].map((y) => (2 * y) / height - 1);
    const x0 = area.yaw0 * per;
    const x1 = x0 + area.yaws * per;
    for (const [a, b] of x1 > width ? [[x0, width], [0, x1 - width]] : [[x0, x1]]) {
      gl.uniform4f(u.rect, (2 * a) / width - 1, y0, (2 * b) / width - 1, y1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // The panorama's texels a frame at `toWorld` (its rotation) may cover: {yaw0, yaws, down0, down1}, whole degrees,
  // yaw0 + yaws past 360 going round; from its border, 32 points an edge, and the poles it holds.
  footprint(toWorld) {
    const { tans } = this.coverage;
    const toFrame = transpose(toWorld);
    const edges = frameEdges(tans);
    const poles = [-90, 90].filter((down) => {
      const d = apply(toFrame, [0, 0, -Math.sign(down)]);
      return edges.every((n) => dot(n, d) >= 0);
    });
    const centre = Math.atan2(toWorld[5], toWorld[2]) / RAD;
    let [low, high, top, bottom] = [Infinity, -Infinity, Infinity, -Infinity];
    for (let i = 0; i <= 32; i += 1) {
      for (const point of [[i / 32, 0], [i / 32, 1], [0, i / 32], [1, i / 32]]) {
        const w = unit(apply(toWorld, frameRay(point, tans)));
        const yaw = ((Math.atan2(w[1], w[0]) / RAD - centre + 540) % 360) - 180;
        const down = Math.asin(Math.max(-1, Math.min(1, -w[2]))) / RAD;
        [low, high] = [Math.min(low, yaw), Math.max(high, yaw)];
        [top, bottom] = [Math.min(top, down), Math.max(bottom, down)];
      }
    }
    if (poles.length) [low, high] = [-180, 180];
    if (poles.includes(-90)) top = -90;
    if (poles.includes(90)) bottom = 90;
    const yaw0 = Math.floor(centre + low - 2);
    return {
      yaw0: ((yaw0 % 360) + 360) % 360,
      yaws: Math.min(360, Math.ceil(centre + high + 2) - yaw0),
      down0: Math.max(-90, Math.floor(top - 2)),
      down1: Math.min(90, Math.ceil(bottom + 2)),
    };
  }

  // How far the view turns from the newest frame towards `look` {yaw, pitch}, {share, magnify}: the share of the way,
  // the whole turn where the view there has every pixel (magnified by at most MAX_MAGNIFY); else the farthest share
  // that has, searched in steps from the whole turn down (a view may have every pixel past a stretch that has not: a
  // turn back to where the player looked), then halved between the step that has and the one after it. No share has
  // every pixel but none (the newest frame).
  reach(look, oldest) {
    const { base } = this;
    const [dy, dp] = [look.yaw - base.yaw, look.pitch - base.pitch];
    const angle = Math.hypot(dy, dp);
    if (angle < NO_TURN_DEGREES) return { share: 1, magnify: 1 };
    // the side of the view a turn brings in: a turn left (yaw up) the left side, one down the bottom
    const order = this.coverage.blocks([-dy / angle, -dp / angle]);
    const fit = (share) => {
      const view = rotation(base.yaw + share * dy, base.pitch + share * dp);
      let magnify = 1;
      for (let i = 0; i < 3; i += 1) {
        const more = this.coverage.covered(view, oldest, magnify, MAX_MAGNIFY / magnify, order);
        if (more === 1) return magnify;
        if (more === Infinity) break;
        magnify *= more;
      }
      return Infinity;
    };
    let magnify = fit(1);
    if (magnify <= MAX_MAGNIFY) return { share: 1, magnify };
    const steps = Math.min(SEARCH_STEPS, Math.max(2, Math.ceil(angle / SEARCH_STEP_DEGREES)));
    let [low, high] = [0, 1 / steps];
    magnify = 1;
    for (let j = steps - 1; j >= 1; j -= 1) {
      const m = fit(j / steps);
      if (m <= MAX_MAGNIFY) { [low, high, magnify] = [j / steps, (j + 1) / steps, m]; break; }
    }
    for (let i = 0; i < SEARCH_HALVINGS; i += 1) {
      const middle = (low + high) / 2;
      const m = fit(middle);
      if (m <= MAX_MAGNIFY) [low, magnify] = [middle, m]; else high = middle;
    }
    return { share: low, magnify };
  }

  // Draw the newest frame turned towards `look` {yaw, pitch} (as far as the known pixels reach) and moved by `move`
  // {zoom, shift} (the motion feedback's) at `now` (ms); nothing when that is what the canvas shows, or while the
  // context is lost. Returns the orientation the view is drawn at.
  draw(look, move, now) {
    const { gl, base } = this;
    if (gl.isContextLost() || !base || !this.ready()) return this.view;
    const oldest = this.coverage.oldest(now);
    const key = `${this.serial} ${look.yaw} ${look.pitch} ${oldest}`;
    if (key !== this.key) {
      this.key = key;
      this.reached = this.reach(look, oldest);
      const { share } = this.reached;
      const along = (key) => base[key] + share * (look[key] - base[key]);
      this.view = { yaw: along('yaw'), pitch: along('pitch') };
    }
    const eased = this.magnify - ((MAX_MAGNIFY - 1) * (now - this.magnifiedAt)) / (MAGNIFY_EASE_S * 1000);
    this.magnify = Math.max(this.reached.magnify, eased);
    this.magnifiedAt = now;
    const drawn = `${key} ${this.magnify} ${move.zoom} ${move.shift}`;
    if (drawn === this.drawn) return this.view;
    this.drawn = drawn;
    this.render(move, oldest, 0, null);
    return this.view;
  }

  // The display pass into `target` (null: the canvas) with `debug` (0: none).
  render(move, oldest, debug, target) {
    const { gl } = this;
    const view = rotation(this.view.yaw, this.view.pitch);
    const turn = product(transpose(this.base.toWorld), view);
    gl.useProgram(this.programs.display);
    const u = this.uniforms.display;
    gl.uniformMatrix3fv(u.turn, false, transpose(turn));
    gl.uniformMatrix3fv(u.view, false, transpose(view));
    gl.uniformMatrix3fv(u.base, false, transpose(this.base.toWorld));
    gl.uniform1f(u.oldest, oldest);
    gl.uniform1f(u.magnify, this.magnify);
    gl.uniform1f(u.zoom, move.zoom);
    gl.uniform2f(u.shift, 2 * move.shift, 0);
    gl.uniform1i(u.debug, debug);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target);
    gl.viewport(0, 0, this.size[0], this.size[1]);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // Where the point (u, v) of the newest frame, fractions from its top left, shows in the view drawn (before the motion
  // feedback); null where it is behind the view.
  place(u, v) {
    if (!this.view) return [u, v];
    const toView = product(transpose(rotation(this.view.yaw, this.view.pitch)), this.base.toWorld);
    const [x, y, z] = apply(toView, frameRay([u, v], page.camera_tans));
    const [across, up] = page.camera_tans;
    const scale = this.magnify / (2 * z);
    return z > 0 ? [0.5 + (x / across) * scale, 0.5 - (y / up) * scale] : null;
  }

  // For the tests: the view drawn again with `debug` (0: as it is; 1: magenta where no frame has the pixel; 2: where
  // each pixel is from, as the display pass says) and `move`, its pixels read back at once, row by row from the bottom.
  read(debug, move, now) {
    const { gl } = this;
    const [width, height] = this.size;
    const pixels = new Uint8Array(4 * width * height);
    let [target, texture] = [null, null];
    if (debug === 2) {
      target = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, target);
      texture = gl.createTexture();
      gl.activeTexture(gl.TEXTURE4);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, width, height);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    }
    this.render(move, this.coverage.oldest(now), debug, target);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (target) {
      gl.deleteFramebuffer(target);
      gl.deleteTexture(texture);
    }
    this.drawn = '';
    return pixels;
  }
}
