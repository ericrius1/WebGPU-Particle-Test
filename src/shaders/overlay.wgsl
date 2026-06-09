@group(0) @binding(0) var<uniform> C : Constants;
@group(0) @binding(1) var<storage, read> statCount : array<u32>;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv  : vec2<f32>,
  @location(1) occ : f32,
  @location(2) cnt : f32,
  @location(3) cap : f32,
};

fn fit(c: vec2<f32>) -> vec2<f32> {
  var o = c;
  if (C.aspect > 1.0) { o.x /= C.aspect; } else { o.y *= C.aspect; }
  return o;
}
fn heatB(t: f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0);
  return mix(vec3<f32>(0.10, 0.45, 0.95), vec3<f32>(0.98, 0.30, 0.15), x);
}
fn heatA(t: f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0);
  return mix(vec3<f32>(0.10, 0.75, 0.62), vec3<f32>(0.85, 0.15, 0.80), x);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> VSOut {
  let corners = array<vec2<f32>, 6>(
    vec2(0.0,0.0), vec2(1.0,0.0), vec2(0.0,1.0),
    vec2(0.0,1.0), vec2(1.0,0.0), vec2(1.0,1.0));
  let q = corners[vi];
  let cx = f32(inst % C.gridW);
  let cy = f32(inst / C.gridW);
  let world = (vec2<f32>(cx, cy) + q) * C.cellSize;
  let vh = C.viewSize * 0.5;
  var out: VSOut;
  out.clip = vec4<f32>(fit((world - C.viewCenter) / vh), 0.0, 1.0);
  out.uv  = q;
  let cnt = statCount[inst];
  out.cnt = f32(cnt);
  out.cap = 0.0;
  if (C.mode == 1u) {
    out.occ = f32(cnt) / f32(C.maxPerCell);
    if (cnt >= C.maxPerCell) { out.cap = 1.0; }
  } else {
    out.occ = f32(cnt) / max(C.maxObs, 1.0);
  }
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let modeB = C.mode == 1u;
  let t = clamp(in.occ, 0.0, 1.0);
  let edge = min(min(in.uv.x, 1.0 - in.uv.x), min(in.uv.y, 1.0 - in.uv.y));
  let line = 1.0 - smoothstep(0.0, 0.04, edge);

  if (modeB && in.cap > 0.5) {
    let stripe = step(0.5, fract((in.uv.x + in.uv.y) * 6.0));
    let warn = mix(vec3<f32>(1.0, 0.85, 0.10), vec3<f32>(1.0, 0.20, 0.0), stripe);
    return vec4<f32>(warn, 0.85);
  }

  let base = select(heatA(t), heatB(t), modeB);
  let lineCol = select(vec3<f32>(0.04, 0.07, 0.05), vec3<f32>(0.05, 0.05, 0.10), modeB);
  let fillA = select(0.0, 0.12 + 0.5 * t, in.cnt > 0.0);
  let col = mix(base, lineCol, line);
  let a = max(fillA, line * 0.30);
  return vec4<f32>(col, a);
}
