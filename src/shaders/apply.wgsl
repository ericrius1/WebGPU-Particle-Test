// apply collision results computed in the collide pass.
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= C.numParticles) { return; }
  var v = newSpeed[i];
  let m = length(v);
  let vmax = C.worldSize * 0.4;
  if (m > vmax) { v *= vmax / m; } // safety clamp against runaway collisions
  particles[i].speed = v;
  particles[i].temp  = max(newTemp[i] * C.tempDecay, 0.0);
}
