@group(0) @binding(0) var<uniform> C : Constants;
@group(0) @binding(1) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(2) var<storage, read_write> newSpeed  : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> newTemp   : array<f32>;
