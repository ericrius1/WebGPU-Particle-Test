@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linearId(gid, nwg);
  if (i == 0u) {
    atomicStore(&dispatchArgs[0], 0u);
    atomicStore(&dispatchArgs[1], 1u);
    atomicStore(&dispatchArgs[2], 1u);
  }
  if (i >= C.gridW * C.gridH) { return; }
  atomicStore(&cellCount[i], 0u);
}
