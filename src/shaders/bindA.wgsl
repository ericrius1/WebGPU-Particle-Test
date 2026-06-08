// Mode A: per-particle linked-list grid
@group(0) @binding(4) var<storage, read_write> gridHead : array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> gridNext : array<i32>;
