// One workgroup handles BUCKETS_PER_WG occupied cells (lanes packed
// 64 = 4 buckets x 16 slots) so waves run full instead of 1/4 idle.
// Each bucket stages its 3x3 neighbourhood into shared memory *compactly*
// (atomic append, no INVALID holes) and every survivor then loops only the
// real neighbour count, not a fixed 9*MAX_PER_CELL.
var<workgroup> sPos   : array<vec2<f32>, SHARED * BUCKETS_PER_WG>;
var<workgroup> sVel   : array<vec2<f32>, SHARED * BUCKETS_PER_WG>;
var<workgroup> sSize  : array<f32,       SHARED * BUCKETS_PER_WG>;
var<workgroup> sIdx   : array<u32,       SHARED * BUCKETS_PER_WG>;
var<workgroup> sCount : array<atomic<u32>, BUCKETS_PER_WG>;   // compact length per bucket

@compute @workgroup_size(MAX_PER_CELL * BUCKETS_PER_WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  let b    = lid / C.maxPerCell;          // which bucket in this workgroup
  let slot = lid % C.maxPerCell;          // slot within that bucket
  let wgLinear = wid.y * nwg.x + wid.x;
  let occIdx = wgLinear * BUCKETS_PER_WG + b;
  let inRange = occIdx < dispatchArgs[3];

  if (lid < BUCKETS_PER_WG) { atomicStore(&sCount[lid], 0u); }
  workgroupBarrier();

  // Stage: this lane loads its slot from each of the 9 neighbour cells and
  // appends the valid ones into bucket b's compact shared region.
  if (inRange) {
    let cell = occupied[occIdx];
    let cx = i32(cell % C.gridW);
    let cy = i32(cell / C.gridW);
    let region = b * SHARED;
    for (var k = 0u; k < 9u; k++) {
      let nx = cx + (i32(k % 3u) - 1);
      let ny = cy + (i32(k / 3u) - 1);
      if (nx >= 0 && ny >= 0 && nx < i32(C.gridW) && ny < i32(C.gridH)) {
        let nc = u32(ny) * C.gridW + u32(nx);
        let cnt = min(atomicLoad(&cellCount[nc]), C.maxPerCell);
        if (slot < cnt) {
          let idx = cellParticles[nc * C.maxPerCell + slot];
          let d = region + atomicAdd(&sCount[b], 1u);
          let p = particles[idx];
          sPos[d]  = p.pos;
          sVel[d]  = p.speed;
          sSize[d] = p.size;
          sIdx[d]  = idx;
        }
      }
    }
  }
  workgroupBarrier();

  if (!inRange) { return; }

  // "My" particle = slot `slot` of the centre cell.
  let center = occupied[occIdx];
  let centerCount = min(atomicLoad(&cellCount[center]), C.maxPerCell);
  if (slot >= centerCount) { return; }
  let me = cellParticles[center * C.maxPerCell + slot];
  let mp = particles[me].pos;
  let mv = particles[me].speed;
  let mr = particles[me].size;

  var dv = vec2<f32>(0.0);
  var heat = 0.0;
  let region = b * SHARED;
  let n = atomicLoad(&sCount[b]);
  for (var s = 0u; s < n; s++) {
    let d = region + s;
    if (sIdx[d] == me) { continue; }
    dv += collidePair(mp, mv, mr, sPos[d], sVel[d], sSize[d], &heat);
  }
  newSpeed[me] = mv + dv;
  newTemp[me]  = particles[me].temp + heat;
}
