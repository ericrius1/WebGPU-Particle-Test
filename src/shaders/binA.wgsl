// insert each particle into the head-linked list of its cell.
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= C.numParticles) { return; }
  let cell = cellIndex(cellCoord(particles[i].pos));
  let prev = atomicExchange(&gridHead[cell], i32(i));
  gridNext[i] = prev;
}
