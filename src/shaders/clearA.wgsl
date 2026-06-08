@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= C.gridW * C.gridH) { return; }
  atomicStore(&gridHead[i], -1);
}
