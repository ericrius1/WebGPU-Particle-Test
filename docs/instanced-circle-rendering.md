# Instanced circle rendering

Draw N particles as soft circles in a single instanced draw call — no per-circle
geometry, no texture. One quad, `N` instances, the circle shaped in the fragment
shader.

## One draw call, a quad per instance

`draw(6, N)` issues 6 vertices (two triangles = a quad) for each of `N`
instances. The vertex shader builds the quad from `vertex_index` and positions it
from `instance_index`:

```wgsl
let corners = array<vec2<f32>, 6>(
  vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
  vec2(-1.0, 1.0), vec2(1.0,-1.0), vec2( 1.0,1.0));
let q = corners[vi];                 // unit quad corner, range [-1,1]
let p = particles[inst];             // this instance's particle
```

Particle data is read from a `read-only-storage` buffer bound to the vertex
stage — no vertex buffers needed.

## Camera: a window into the world

A `viewSize`-wide window centred on `viewCenter` maps to NDC `[-1, 1]`. Particle
world position and radius are both divided by the half-window so zoom affects
size and position consistently:

```wgsl
let vh = C.viewSize * 0.5;
let center = (p.pos - C.viewCenter) / vh;   // particle centre in NDC
let offset = q * (p.size / vh);             // quad corner scaled to radius
out.clip = vec4<f32>(fit(center + offset), 0.0, 1.0);
out.local = q;                              // pass quad coord to fragment for the circle mask
```

See [camera-pan-zoom](camera-pan-zoom.md) for how `viewSize`/`viewCenter` are
driven by input and why the world grows with N.

## Aspect-ratio fit (keep circles round)

Letterbox to a square so circles don't stretch on non-square canvases — shrink
the longer axis:

```wgsl
fn fit(c: vec2<f32>) -> vec2<f32> {
  var o = c;
  if (C.aspect > 1.0) { o.x /= C.aspect; } else { o.y *= C.aspect; }
  return o;
}
```

## Circle mask + soft edge (fragment)

The quad's local coordinate `q ∈ [-1,1]²` is reused as a radial coordinate.
Discard outside the unit disc; fade alpha near the rim for a soft blob:

```wgsl
let d = length(in.local);
if (d > 1.0) { discard; }
let alpha = smoothstep(1.0, 0.25, d);   // opaque core -> glow at the rim
return vec4<f32>(heatColor(in.temp), alpha);
```

Use premultiplied/standard alpha blending in the pipeline so the glow composites
over the background.

## Colour by energy + collision heat

The vertex shader computes a 0–1 "temperature" from normalised speed plus the
collision-heat term (`temp`), and the fragment maps it through a cold→hot ramp:

```wgsl
let speedNorm = length(p.speed) / (C.worldSize * 0.35);
out.temp = clamp(speedNorm + p.temp, 0.0, 1.0);
```

```wgsl
fn heatColor(t: f32) -> vec3<f32> {
  let cold = vec3<f32>(0.25, 0.65, 0.95);  // blue   (slow / cold)
  let mid  = vec3<f32>(0.55, 0.80, 0.55);  // green
  let hot  = vec3<f32>(0.98, 0.82, 0.25);  // yellow (fast / hot)
  let a = mix(cold, mid, smoothstep(0.0, 0.5, t));
  return mix(a, hot, smoothstep(0.5, 1.0, t));
}
```

Two `smoothstep`/`mix` segments give a smooth three-stop gradient. Collision heat
(`p.temp`) briefly pushes a particle toward the hot end on impact, then decays
(see [collision-response](collision-response.md) and
[particle-integration](particle-integration.md)).
