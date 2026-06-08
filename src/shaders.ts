// All WGSL authored here so struct/uniform defs are shared across passes and
// compile-time constants (workgroup size, max particles per cell) get injected.

export const MAX_PER_CELL = 16; // bucket capacity for Mode B shared-memory load
export const WG = 64; // workgroup size for per-particle passes
const SHARED = MAX_PER_CELL * 9; // 3x3 neighbourhood capacity in Mode B
const INVALID = "0xffffffffu";

// ---- shared declarations injected at top of every module ----------------
const COMMON = /* wgsl */ `
struct Particle {
  pos   : vec2<f32>,
  speed : vec2<f32>,
  size  : f32,
  temp  : f32,
};

struct Constants {
  numParticles : u32,
  gridW        : u32,
  gridH        : u32,
  maxPerCell   : u32,
  cellSize     : f32,
  dt           : f32,
  aspect       : f32,
  tempDecay    : f32,
  tempGain     : f32,
  restitution  : f32,
  worldSize    : f32,
  viewSize     : f32,  // side of the on-screen window into the world (camera)
};

fn cellCoord(p: vec2<f32>) -> vec2<i32> {
  let g = vec2<i32>(floor(p / C.cellSize));
  return clamp(g, vec2<i32>(0), vec2<i32>(i32(C.gridW) - 1, i32(C.gridH) - 1));
}
fn cellIndex(c: vec2<i32>) -> u32 {
  return u32(c.y) * C.gridW + u32(c.x);
}

// elastic impulse on particle a from particle b (mass = size^2).
// returns delta velocity for a; accumulates collision heat into *heat.
fn collidePair(pa: vec2<f32>, va: vec2<f32>, ra: f32,
               pb: vec2<f32>, vb: vec2<f32>, rb: f32,
               heat: ptr<function, f32>) -> vec2<f32> {
  let d = pa - pb;
  let dist2 = dot(d, d);
  let rsum = ra + rb;
  if (dist2 >= rsum * rsum || dist2 < 1e-12) { return vec2<f32>(0.0); }
  let dist = sqrt(dist2);
  let n = d / dist;
  // gentle positional push so deep overlaps don't stick and churn impulses
  let sep = n * (rsum - dist) * 1.5;
  let vrel = dot(va - vb, n);
  if (vrel >= 0.0) { return sep; } // already separating: only de-penetrate
  let ma = ra * ra;
  let mb = rb * rb;
  let j = -(1.0 + C.restitution) * vrel * (mb / (ma + mb));
  *heat += abs(j) * C.tempGain;
  return j * n + sep;
}
`;

// ---- uniform + particle bindings present in most passes -----------------
const BIND_SIM = /* wgsl */ `
@group(0) @binding(0) var<uniform> C : Constants;
@group(0) @binding(1) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(2) var<storage, read_write> newSpeed  : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> newTemp   : array<f32>;
`;

// =========================================================================
// Shared passes (both modes)
// =========================================================================

// integrate motion + bounce off the unit-box walls.
export const INTEGRATE_WGSL = COMMON + BIND_SIM + /* wgsl */ `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= C.numParticles) { return; }
  var p = particles[i];
  p.pos += p.speed * C.dt;
  let r = p.size;
  let w = C.worldSize;
  if (p.pos.x < r)      { p.pos.x = r;      p.speed.x =  abs(p.speed.x); }
  if (p.pos.x > w - r)  { p.pos.x = w - r;  p.speed.x = -abs(p.speed.x); }
  if (p.pos.y < r)      { p.pos.y = r;      p.speed.y =  abs(p.speed.y); }
  if (p.pos.y > w - r)  { p.pos.y = w - r;  p.speed.y = -abs(p.speed.y); }
  particles[i] = p;
}
`;

// apply collision results computed in the collide pass.
export const APPLY_WGSL = COMMON + BIND_SIM + /* wgsl */ `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= C.numParticles) { return; }
  var v = newSpeed[i];
  let m = length(v);
  let vmax = C.worldSize * 0.4;
  if (m > vmax) { v *= vmax / m; } // safety clamp against runaway collisions
  particles[i].speed = v;
  particles[i].temp  = max(newTemp[i] * C.tempDecay, 0.0);
}
`;

// =========================================================================
// MODE A — classic per-particle linked-list grid (Diligent Tutorial14)
// =========================================================================

const BIND_A = /* wgsl */ `
@group(0) @binding(4) var<storage, read_write> gridHead : array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> gridNext : array<i32>;
`;

export const CLEAR_A_WGSL = COMMON + BIND_SIM + BIND_A + /* wgsl */ `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= C.gridW * C.gridH) { return; }
  atomicStore(&gridHead[i], -1);
}
`;

// insert each particle into the head-linked list of its cell.
export const BIN_A_WGSL = COMMON + BIND_SIM + BIND_A + /* wgsl */ `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= C.numParticles) { return; }
  let cell = cellIndex(cellCoord(particles[i].pos));
  let prev = atomicExchange(&gridHead[cell], i32(i));
  gridNext[i] = prev;
}
`;

// per particle: walk the 3x3 neighbourhood linked lists.
export const COLLIDE_A_WGSL = COMMON + BIND_SIM + BIND_A + /* wgsl */ `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= C.numParticles) { return; }
  let me = particles[i];
  let cc = cellCoord(me.pos);
  var dv = vec2<f32>(0.0);
  var heat = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let nc = cc + vec2<i32>(dx, dy);
      if (nc.x < 0 || nc.y < 0 || nc.x >= i32(C.gridW) || nc.y >= i32(C.gridH)) { continue; }
      var j = atomicLoad(&gridHead[cellIndex(nc)]);
      var guard = 0u;
      loop {
        if (j < 0 || guard > C.numParticles) { break; }
        let uj = u32(j);
        if (uj != i) {
          let o = particles[uj];
          dv += collidePair(me.pos, me.speed, me.size, o.pos, o.speed, o.size, &heat);
        }
        j = gridNext[uj];
        guard++;
      }
    }
  }
  newSpeed[i] = me.speed + dv;
  newTemp[i]  = me.temp + heat;
}
`;

// =========================================================================
// MODE B — iterate by occupied bucket, neighbours staged in shared memory
// =========================================================================

const BIND_B = /* wgsl */ `
@group(0) @binding(4) var<storage, read_write> cellCount     : array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> cellParticles : array<u32>;
@group(0) @binding(6) var<storage, read_write> occupied      : array<u32>;
@group(0) @binding(7) var<storage, read_write> dispatchArgs  : array<atomic<u32>>; // [x,y,z]
`;

export const CLEAR_B_WGSL = COMMON + BIND_SIM + BIND_B + /* wgsl */ `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  // thread 0 resets the indirect dispatch args (x = occupied count, y=z=1).
  if (i == 0u) {
    atomicStore(&dispatchArgs[0], 0u);
    atomicStore(&dispatchArgs[1], 1u);
    atomicStore(&dispatchArgs[2], 1u);
  }
  if (i >= C.gridW * C.gridH) { return; }
  atomicStore(&cellCount[i], 0u);
}
`;

// count particles per cell, store up to maxPerCell, register first-touch cells.
export const BIN_B_WGSL = COMMON + BIND_SIM + BIND_B + /* wgsl */ `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= C.numParticles) { return; }
  let cell = cellIndex(cellCoord(particles[i].pos));
  let slot = atomicAdd(&cellCount[cell], 1u);
  if (slot == 0u) {
    // first particle in this cell -> append to the occupied list
    let oi = atomicAdd(&dispatchArgs[0], 1u);
    occupied[oi] = cell;
  }
  if (slot < C.maxPerCell) {
    cellParticles[cell * C.maxPerCell + slot] = i;
  }
}
`;

// collide pass binds the bucket buffers read-only-ish (no dispatchArgs): the
// indirect buffer can't also be a storage binding in the same sync scope.
const BIND_B_COLLIDE = /* wgsl */ `
@group(0) @binding(4) var<storage, read_write> cellCount     : array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> cellParticles : array<u32>;
@group(0) @binding(6) var<storage, read_write> occupied      : array<u32>;
`;

// one workgroup per occupied bucket; ${MAX_PER_CELL} threads load the 3x3
// neighbourhood (${SHARED} entries) into shared memory, then collide.
export const COLLIDE_B_WGSL = COMMON + BIND_SIM + BIND_B_COLLIDE + /* wgsl */ `
var<workgroup> sPos  : array<vec2<f32>, ${SHARED}>;
var<workgroup> sVel  : array<vec2<f32>, ${SHARED}>;
var<workgroup> sSize : array<f32,       ${SHARED}>;
var<workgroup> sIdx  : array<u32,       ${SHARED}>;

@compute @workgroup_size(${MAX_PER_CELL})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  let cell = occupied[wid.x];
  let cx = i32(cell % C.gridW);
  let cy = i32(cell / C.gridW);

  // collaboratively stage the 3x3 neighbourhood: thread lid loads slot lid of
  // each of the 9 cells -> ${SHARED} total entries.
  for (var k = 0u; k < 9u; k++) {
    let dx = i32(k % 3u) - 1;
    let dy = i32(k / 3u) - 1;
    let nx = cx + dx;
    let ny = cy + dy;
    let dst = k * C.maxPerCell + lid;
    var idx = ${INVALID};
    if (nx >= 0 && ny >= 0 && nx < i32(C.gridW) && ny < i32(C.gridH)) {
      let nc = u32(ny) * C.gridW + u32(nx);
      let cnt = min(atomicLoad(&cellCount[nc]), C.maxPerCell);
      if (lid < cnt) { idx = cellParticles[nc * C.maxPerCell + lid]; }
    }
    if (idx != ${INVALID}) {
      let p = particles[idx];
      sPos[dst]  = p.pos;
      sVel[dst]  = p.speed;
      sSize[dst] = p.size;
    }
    sIdx[dst] = idx;
  }
  workgroupBarrier();

  // "my" particle = slot lid of the centre cell (k=4 -> 4*maxPerCell+lid)
  let myStaged = 4u * C.maxPerCell + lid;
  let me = sIdx[myStaged];
  if (me == ${INVALID}) { return; }

  let mp = sPos[myStaged];
  let mv = sVel[myStaged];
  let mr = sSize[myStaged];
  var dv = vec2<f32>(0.0);
  var heat = 0.0;
  let total = 9u * C.maxPerCell;
  for (var s = 0u; s < total; s++) {
    let oi = sIdx[s];
    if (oi == ${INVALID} || oi == me) { continue; }
    dv += collidePair(mp, mv, mr, sPos[s], sVel[s], sSize[s], &heat);
  }
  newSpeed[me] = mv + dv;
  newTemp[me]  = particles[me].temp + heat;
}
`;

// =========================================================================
// Render — instanced circles coloured by temperature
// =========================================================================

export const RENDER_WGSL = COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> C : Constants;
@group(0) @binding(1) var<storage, read> particles : array<Particle>;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) local : vec2<f32>,
  @location(1) temp  : f32,
};

fn fit(c: vec2<f32>) -> vec2<f32> {
  var o = c;
  if (C.aspect > 1.0) { o.x /= C.aspect; } else { o.y *= C.aspect; }
  return o;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> VSOut {
  // unit quad
  let corners = array<vec2<f32>, 6>(
    vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
    vec2(-1.0, 1.0), vec2(1.0,-1.0), vec2( 1.0,1.0));
  let q = corners[vi];
  let p = particles[inst];
  // camera: a viewSize-wide window centred on the world maps to NDC [-1,1]
  let vh = C.viewSize * 0.5;
  let half = C.worldSize * 0.5;
  let center = (p.pos - vec2<f32>(half)) / vh;
  let offset = q * (p.size / vh);
  // colour by kinetic energy (speed) plus a collision-heat flash
  let speedNorm = length(p.speed) / (C.worldSize * 0.35);
  var out: VSOut;
  out.clip  = vec4<f32>(fit(center + offset), 0.0, 1.0);
  out.local = q;
  out.temp  = clamp(speedNorm + p.temp, 0.0, 1.0);
  return out;
}

fn heatColor(t: f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0);
  let cold = vec3<f32>(0.25, 0.65, 0.95); // blue  (slow / cold)
  let mid  = vec3<f32>(0.55, 0.80, 0.55); // green
  let hot  = vec3<f32>(0.98, 0.82, 0.25); // yellow (fast / hot)
  let a = mix(cold, mid, smoothstep(0.0, 0.5, x));
  return mix(a, hot, smoothstep(0.5, 1.0, x));
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let d = length(in.local);
  if (d > 1.0) { discard; }
  // soft radial blob, opaque core fading to a glow at the rim
  let alpha = smoothstep(1.0, 0.25, d);
  return vec4<f32>(heatColor(in.temp), alpha);
}
`;
