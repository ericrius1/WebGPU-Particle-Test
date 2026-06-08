@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= C.numParticles) { return; }
  let cell = cellIndex(cellCoord(particles[i].pos));
  let old = atomicAdd(&statCount[cell], 1u);
  if (old == 0u) { atomicAdd(&statMeta[0], 1u); }       // occupied cells
  atomicMax(&statMeta[1], old + 1u);                     // max per cell
  if (old >= C.maxPerCell) { atomicAdd(&statMeta[2], 1u); } // overflow (dropped in B)
}
