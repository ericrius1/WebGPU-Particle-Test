@group(0) @binding(4) var<storage, read_write> cellCount     : array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> cellParticles : array<u32>;
@group(0) @binding(6) var<storage, read_write> occupied      : array<u32>;
@group(0) @binding(7) var<storage, read_write> dispatchArgs  : array<atomic<u32>>;
