@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linearId(gid, nwg);
  if (i >= C.numParticles) { return; }
  let cell = cellIndex(cellCoord(particles[i].pos));
  let slot = atomicAdd(&cellCount[cell], 1u);
  if (slot == 0u) {
    let oi = atomicAdd(&dispatchArgs[0], 1u);
    occupied[oi] = cell;
  }
  if (slot < C.maxPerCell) {
    cellParticles[cell * C.maxPerCell + slot] = i;
  }
}
