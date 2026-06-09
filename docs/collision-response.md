# Collision response — elastic impulse with mass ∝ size²

A velocity-only (no rotation, no persistent contacts) elastic collision between
two circles. Cheap enough to run per candidate pair inside the collision kernel,
stable enough to survive thousands of simultaneous contacts on the GPU.

## The kernel

```wgsl
// mass = size^2 (area in 2D). Returns delta velocity for particle a;
// accumulates collision heat into *heat.
fn collidePair(pa: vec2<f32>, va: vec2<f32>, ra: f32,
               pb: vec2<f32>, vb: vec2<f32>, rb: f32,
               heat: ptr<function, f32>) -> vec2<f32> {
  let d = pa - pb;
  let dist2 = dot(d, d);
  let rsum = ra + rb;
  if (dist2 >= rsum * rsum || dist2 < 1e-12) { return vec2<f32>(0.0); }
  let dist = sqrt(dist2);
  let n = d / dist;
  let sep = n * (rsum - dist) * 1.5;               // positional de-penetration
  let vrel = dot(va - vb, n);
  if (vrel >= 0.0) { return sep; }                 // separating: push only
  let ma = ra * ra;
  let mb = rb * rb;
  let j = -(1.0 + C.restitution) * vrel * (mb / (ma + mb));
  *heat += abs(j) * C.tempGain;
  return j * n + sep;
}
```

## Design notes

- **Impulse formula.** Along the contact normal `n`:
  `j = -(1 + restitution) · v_rel · (m_b / (m_a + m_b))`, with mass `= size²`
  (2D area). The mass ratio gives the correct momentum split for unequal sizes;
  `restitution` (0–1) controls bounciness.
- **Early outs.** Skip when not overlapping (`dist2 >= rsum²`) and when nearly
  coincident (`dist2 < 1e-12`) to avoid a divide-by-zero normal.
- **Positional push (`sep`).** A small overcorrection (`×1.5`) along the normal
  separates deep overlaps. Without it, particles that interpenetrate stick and
  churn impulses forever. Pairs that are already separating (`vrel >= 0`) get
  *only* the push — no impulse — so they don't gain energy.
- **Heat.** Each impulse magnitude accumulates into a per-particle `temp` scaled
  by `tempGain`; `temp` decays each frame (`tempDecay`). Rendering flashes hot on
  impact. See [particle-integration](particle-integration.md) for decay and
  [instanced-circle-rendering](instanced-circle-rendering.md) for the colour map.

## Stability

Because many neighbours can hit one particle in a single frame, the summed delta
velocity can blow up. The `apply` pass clamps speed to a fraction of the world
size as a safety net against runaway feedback — see
[particle-integration](particle-integration.md).
