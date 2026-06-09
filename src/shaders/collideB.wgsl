var<workgroup> sPos  : array<vec2<f32>, SHARED>;
var<workgroup> sVel  : array<vec2<f32>, SHARED>;
var<workgroup> sSize : array<f32,       SHARED>;
var<workgroup> sIdx  : array<u32,       SHARED>;

@compute @workgroup_size(MAX_PER_CELL)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  let slot = wid.y * nwg.x + wid.x;
  if (slot >= dispatchArgs[3]) { return; }
  let cell = occupied[slot];
  let cx = i32(cell % C.gridW);
  let cy = i32(cell / C.gridW);

  for (var k = 0u; k < 9u; k++) {
    let dx = i32(k % 3u) - 1;
    let dy = i32(k / 3u) - 1;
    let nx = cx + dx;
    let ny = cy + dy;
    let dst = k * C.maxPerCell + lid;
    var idx = INVALID;
    if (nx >= 0 && ny >= 0 && nx < i32(C.gridW) && ny < i32(C.gridH)) {
      let nc = u32(ny) * C.gridW + u32(nx);
      let cnt = min(atomicLoad(&cellCount[nc]), C.maxPerCell);
      if (lid < cnt) { idx = cellParticles[nc * C.maxPerCell + lid]; }
    }
    if (idx != INVALID) {
      let p = particles[idx];
      sPos[dst]  = p.pos;
      sVel[dst]  = p.speed;
      sSize[dst] = p.size;
    }
    sIdx[dst] = idx;
  }
  workgroupBarrier();

  let myStaged = 4u * C.maxPerCell + lid;
  let me = sIdx[myStaged];
  if (me == INVALID) { return; }

  let mp = sPos[myStaged];
  let mv = sVel[myStaged];
  let mr = sSize[myStaged];
  var dv = vec2<f32>(0.0);
  var heat = 0.0;
  let total = 9u * C.maxPerCell;
  for (var s = 0u; s < total; s++) {
    let oi = sIdx[s];
    if (oi == INVALID || oi == me) { continue; }
    dv += collidePair(mp, mv, mr, sPos[s], sVel[s], sSize[s], &heat);
  }
  newSpeed[me] = mv + dv;
  newTemp[me]  = particles[me].temp + heat;
}
