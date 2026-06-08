// Mode B collide pass binds the bucket buffers without dispatchArgs: the
// indirect buffer can't also be a storage binding in the same sync scope.
@group(0) @binding(4) var<storage, read_write> cellCount     : array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> cellParticles : array<u32>;
@group(0) @binding(6) var<storage, read_write> occupied      : array<u32>;
