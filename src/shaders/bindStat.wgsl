// Occupancy stats bindings
@group(0) @binding(0) var<uniform> C : Constants;
@group(0) @binding(1) var<storage, read> particles : array<Particle>;
@group(0) @binding(2) var<storage, read_write> statCount : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> statMeta  : array<atomic<u32>>; // [occupied,max,overflow,_]
