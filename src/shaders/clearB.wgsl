@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  // thread 0 resets the indirect dispatch args (x = occupied count, y=z=1).
  if (i == 0u) {
    atomicStore(&dispatchArgs[0], 0u);
    atomicStore(&dispatchArgs[1], 1u);
    atomicStore(&dispatchArgs[2], 1u);
  }
  if (i >= C.gridW * C.gridH) { return; }
  atomicStore(&cellCount[i], 0u);
}
