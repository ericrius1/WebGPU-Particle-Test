// Grid overlay — one instanced quad per cell, tinted by occupancy + grid lines
@group(0) @binding(0) var<uniform> C : Constants;
@group(0) @binding(1) var<storage, read> statCount : array<u32>;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv  : vec2<f32>,
  @location(1) occ : f32,
};

fn fit(c: vec2<f32>) -> vec2<f32> {
  var o = c;
  if (C.aspect > 1.0) { o.x /= C.aspect; } else { o.y *= C.aspect; }
  return o;
}
fn heatColor(t: f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0);
  return mix(vec3<f32>(0.10, 0.45, 0.95), vec3<f32>(0.98, 0.30, 0.15), x);
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
  out.occ = f32(statCount[inst]) / f32(C.maxPerCell);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let t = clamp(in.occ, 0.0, 1.0);
  let edge = min(min(in.uv.x, 1.0 - in.uv.x), min(in.uv.y, 1.0 - in.uv.y));
  let line = 1.0 - smoothstep(0.0, 0.04, edge);     // grid line near cell border
  let fillA = select(0.0, 0.12 + 0.5 * t, in.occ > 0.0);
  let col = mix(heatColor(t), vec3<f32>(0.05, 0.05, 0.08), line);
  let a = max(fillA, line * 0.30);
  return vec4<f32>(col, a);
}
