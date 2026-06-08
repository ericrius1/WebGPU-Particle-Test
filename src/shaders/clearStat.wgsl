@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i == 0u) {
    atomicStore(&statMeta[0], 0u);
    atomicStore(&statMeta[1], 0u);
    atomicStore(&statMeta[2], 0u);
    atomicStore(&statMeta[3], 0u);
  }
  if (i >= C.gridW * C.gridH) { return; }
  atomicStore(&statCount[i], 0u);
}
