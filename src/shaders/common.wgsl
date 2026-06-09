const WG             : u32 = 64;
const MAX_PER_CELL   : u32 = 16;
const SHARED         : u32 = MAX_PER_CELL * 9u;   // 3x3 neighbourhood, one bucket
const BUCKETS_PER_WG : u32 = 4u;                  // occupied cells handled per workgroup
const INVALID        : u32 = 0xffffffffu;

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
  viewSize     : f32,
  viewCenter   : vec2<f32>,
  mode         : u32,
  maxObs       : f32,
};

// Flatten a 2D workgroup grid (used when 1D group count exceeds the
// 65535 per-dimension dispatch limit) back to a linear thread index.
fn linearId(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
  return gid.y * nwg.x * WG + gid.x;
}

fn cellCoord(p: vec2<f32>) -> vec2<i32> {
  let g = vec2<i32>(floor(p / C.cellSize));
  return clamp(g, vec2<i32>(0), vec2<i32>(i32(C.gridW) - 1, i32(C.gridH) - 1));
}
fn cellIndex(c: vec2<i32>) -> u32 {
  return u32(c.y) * C.gridW + u32(c.x);
}

fn collidePair(pa: vec2<f32>, va: vec2<f32>, ra: f32,
               pb: vec2<f32>, vb: vec2<f32>, rb: f32,
               heat: ptr<function, f32>) -> vec2<f32> {
  let d = pa - pb;
  let dist2 = dot(d, d);
  let rsum = ra + rb;
  if (dist2 >= rsum * rsum || dist2 < 1e-12) { return vec2<f32>(0.0); }
  let dist = sqrt(dist2);
  let n = d / dist;
  let sep = n * (rsum - dist) * 1.5;
  let vrel = dot(va - vb, n);
  if (vrel >= 0.0) { return sep; }
  let ma = ra * ra;
  let mb = rb * rb;
  let j = -(1.0 + C.restitution) * vrel * (mb / (ma + mb));
  *heat += abs(j) * C.tempGain;
  return j * n + sep;
}
