// per particle: walk the 3x3 neighbourhood linked lists.
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= C.numParticles) { return; }
  let me = particles[i];
  let cc = cellCoord(me.pos);
  var dv = vec2<f32>(0.0);
  var heat = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let nc = cc + vec2<i32>(dx, dy);
      if (nc.x < 0 || nc.y < 0 || nc.x >= i32(C.gridW) || nc.y >= i32(C.gridH)) { continue; }
      var j = atomicLoad(&gridHead[cellIndex(nc)]);
      var guard = 0u;
      loop {
        if (j < 0 || guard > C.numParticles) { break; }
        let uj = u32(j);
        if (uj != i) {
          let o = particles[uj];
          dv += collidePair(me.pos, me.speed, me.size, o.pos, o.speed, o.size, &heat);
        }
        j = gridNext[uj];
        guard++;
      }
    }
  }
  newSpeed[i] = me.speed + dv;
  newTemp[i]  = me.temp + heat;
}
