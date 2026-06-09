// Convert the occupied-cell count produced by binB into an indirect
// dispatch that respects the 65535 per-dimension workgroup limit.
// dispatchArgs[0] holds the count after binB; we stash it in [3] (so
// collideB can bound-check) then split the *workgroup* count across x/y.
// Each workgroup chews BUCKETS_PER_WG occupied cells, so it needs
// ceil(count / BUCKETS_PER_WG) workgroups, not `count`.
@compute @workgroup_size(1)
fn main() {
  let c = atomicLoad(&dispatchArgs[0]);
  atomicStore(&dispatchArgs[3], c);
  let wg = (c + BUCKETS_PER_WG - 1u) / BUCKETS_PER_WG;
  atomicStore(&dispatchArgs[0], min(wg, 65535u));
  atomicStore(&dispatchArgs[1], (wg + 65534u) / 65535u);
}
