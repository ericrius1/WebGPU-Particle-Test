// Convert the occupied-cell count produced by binB into an indirect
// dispatch that respects the 65535 per-dimension workgroup limit.
// dispatchArgs[0] holds the count after binB; we stash it in [3] (so
// collideB can bound-check) then split it across x/y.
@compute @workgroup_size(1)
fn main() {
  let c = atomicLoad(&dispatchArgs[0]);
  atomicStore(&dispatchArgs[3], c);
  atomicStore(&dispatchArgs[0], min(c, 65535u));
  atomicStore(&dispatchArgs[1], (c + 65534u) / 65535u);
}
