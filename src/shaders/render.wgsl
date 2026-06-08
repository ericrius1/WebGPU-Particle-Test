// Render — instanced circles coloured by temperature
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
  // camera: a viewSize-wide window centred on viewCenter maps to NDC [-1,1]
  let vh = C.viewSize * 0.5;
  let center = (p.pos - C.viewCenter) / vh;
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
