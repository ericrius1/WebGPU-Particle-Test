@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = linearId(gid, nwg);
  if (i >= C.numParticles) { return; }
  var p = particles[i];
  p.pos += p.speed * C.dt;
  let r = p.size;
  let w = C.worldSize;
  if (p.pos.x < r)      { p.pos.x = r;      p.speed.x =  abs(p.speed.x); }
  if (p.pos.x > w - r)  { p.pos.x = w - r;  p.speed.x = -abs(p.speed.x); }
  if (p.pos.y < r)      { p.pos.y = r;      p.speed.y =  abs(p.speed.y); }
  if (p.pos.y > w - r)  { p.pos.y = w - r;  p.speed.y = -abs(p.speed.y); }
  particles[i] = p;
}
