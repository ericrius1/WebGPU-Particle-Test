# Mode B — counting buckets + indirect dispatch + shared-memory staging

A second way to record cell membership in a
[spatial-hash grid](spatial-hash-grid.md). Iterates by **occupied cell** rather
than by particle, and reads each neighbour from global memory **once per
workgroup** instead of once per particle.

## Data structures

```wgsl
cellCount     : array<atomic<u32>>;  // particles in each cell
cellParticles : array<u32>;          // cellCount entries packed [cell * MAX_PER_CELL + slot]
occupied      : array<u32>;          // compact list of cell indices that have >=1 particle
dispatchArgs  : array<atomic<u32>>;  // [x, y, z] indirect-dispatch args; x = occupied count
```

`MAX_PER_CELL` (e.g. 16) is a **fixed bucket capacity** — the design budget for
particles per cell. Cells that exceed it drop the overflow (counted as a metric,
see [grid-occupancy-overlay](grid-occupancy-overlay.md)).

## Passes

### clearB

Reset per-cell counts to 0, and (thread 0 only) reset the indirect args so the
occupied count starts at 0 with `y = z = 1`:

```wgsl
if (i == 0u) {
  atomicStore(&dispatchArgs[0], 0u);   // x = occupied cell count (filled in by binB)
  atomicStore(&dispatchArgs[1], 1u);   // y
  atomicStore(&dispatchArgs[2], 1u);   // z
}
if (i < numCells) { atomicStore(&cellCount[i], 0u); }
```

### binB — count, store, register occupied cells

```wgsl
let cell = cellIndex(cellCoord(particles[i].pos));
let slot = atomicAdd(&cellCount[cell], 1u);
if (slot == 0u) {
  let oi = atomicAdd(&dispatchArgs[0], 1u);  // first arrival -> append cell, bump dispatch x
  occupied[oi] = cell;
}
if (slot < C.maxPerCell) {
  cellParticles[cell * C.maxPerCell + slot] = i;  // store up to the cap
}
```

Two atomics do the work: `cellCount` assigns each particle a slot in its bucket;
the `slot == 0` test detects the **first** particle in a cell and appends that
cell to the `occupied` list. That same append counter *is* the `x` of the
indirect-dispatch buffer — so the collide pass automatically dispatches exactly
as many workgroups as there are non-empty cells. Empty cells cost nothing.

### collideB — one workgroup per occupied bucket, shared-memory neighbours

Dispatched indirectly: `dispatchWorkgroupsIndirect(dispatchBuf, 0)`. Workgroup
size = `MAX_PER_CELL`, so there is one thread per bucket slot.

```wgsl
var<workgroup> sPos  : array<vec2<f32>, SHARED>;   // SHARED = MAX_PER_CELL * 9 (the 3x3 nbhd)
var<workgroup> sVel  : array<vec2<f32>, SHARED>;
var<workgroup> sSize : array<f32,       SHARED>;
var<workgroup> sIdx  : array<u32,       SHARED>;   // INVALID marks empty slots

// 1. Collaboratively stage the 3x3 neighbourhood. Thread `lid` loads slot `lid`
//    of each of the 9 cells -> 9 * MAX_PER_CELL entries total.
for (var k = 0u; k < 9u; k++) { /* load cell (cx+dx, cy+dy), slot lid -> sIdx[k*cap+lid] */ }
workgroupBarrier();

// 2. "My" particle = slot lid of the centre cell (k = 4).
let me = sIdx[4u * C.maxPerCell + lid];
if (me == INVALID) { return; }

// 3. Collide against every staged neighbour (read from shared memory).
for (var s = 0u; s < 9u * C.maxPerCell; s++) {
  let oi = sIdx[s];
  if (oi == INVALID || oi == me) { continue; }
  dv += collidePair(mp, mv, mr, sPos[s], sVel[s], sSize[s], &heat);
}
newSpeed[me] = mv + dv;
newTemp[me]  = particles[me].temp + heat;
```

The key win: the `9 * MAX_PER_CELL` (= 144) neighbour entries are read from
global memory **once**, cooperatively, by the workgroup — then every thread
collides against the shared copy. In mode A each thread re-reads its neighbours
independently.

## The indirect-buffer pass split (a WebGPU constraint)

`dispatchArgs` is written **as `storage`** in `binB` but read **as `INDIRECT`**
in `collideB`. WebGPU rejects a buffer used as both writable-storage and indirect
in the same synchronisation scope. The fix: put the two uses in **separate
compute passes** (the bin pass ends, then the collide pass begins). Consequently
the collide pass binds a *different* binding block (`bindBCollide`) that omits
`dispatchArgs` entirely — it only consumes the buffer as the indirect argument to
the dispatch call, never as a shader binding. This is why bindings are split into
their own modules (see [wgsl-module-assembly](wgsl-module-assembly.md)).

## Characteristics

- **Empty cells are free** — never dispatched.
- **Capacity cap** — drops collisions in cells over `MAX_PER_CELL`. Tune the cap
  to the expected max occupancy; the overlay shows where buckets fill/overflow.
- **Wins when dense** — the fixed 144-entry shared load amortises over many
  particles. **Loses when sparse** — every occupied bucket pays the full staging
  cost even for near-empty neighbourhoods, where [mode A](mode-a-linked-list-grid.md)
  just walks a couple of short lists.
