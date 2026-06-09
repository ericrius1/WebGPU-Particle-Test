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

### collideB — many occupied buckets per workgroup, compact shared staging

Dispatched indirectly: `dispatchWorkgroupsIndirect(dispatchBuf, 0)`. Workgroup
size = `MAX_PER_CELL * BUCKETS_PER_WG` (= 16 * 4 = **64**), so one workgroup
chews `BUCKETS_PER_WG` occupied cells at once. Lane `lid` splits into
`b = lid / MAX_PER_CELL` (which bucket) and `slot = lid % MAX_PER_CELL` (slot
within it). `fixB` therefore dispatches `ceil(occupied / BUCKETS_PER_WG)`
workgroups, not one per cell.

Two deliberate choices, both fixing why the naive one-bucket-per-workgroup
version *lost* to [mode A](mode-a-linked-list-grid.md):

1. **Pack buckets to fill the wave.** A 16-thread workgroup leaves 1/2–3/4 of
   every 32/64-lane wave idle, every dispatch, regardless of density. Packing
   four buckets into a 64-thread workgroup runs the wave full.

2. **Compact the staging — no INVALID holes.** Each lane appends its valid
   neighbours into bucket `b`'s shared region via a workgroup atomic, so the
   collide loop walks the *real* neighbour count, not a fixed `9 * MAX_PER_CELL`
   (= 144) with `continue`-skips. Sparse cells stop paying the full 144.

```wgsl
var<workgroup> sPos   : array<vec2<f32>, SHARED * BUCKETS_PER_WG>;
var<workgroup> sVel   : array<vec2<f32>, SHARED * BUCKETS_PER_WG>;
var<workgroup> sSize  : array<f32,       SHARED * BUCKETS_PER_WG>;
var<workgroup> sIdx   : array<u32,       SHARED * BUCKETS_PER_WG>;
var<workgroup> sCount : array<atomic<u32>, BUCKETS_PER_WG>;   // compact length per bucket

// 1. Stage: this lane loads its slot from each of the 9 neighbour cells and
//    *appends* the valid ones into bucket b's region (atomic bump, no holes).
let region = b * SHARED;
for (var k = 0u; k < 9u; k++) {
  // ... if (slot < cnt) { let d = region + atomicAdd(&sCount[b], 1u); store ... }
}
workgroupBarrier();

// 2. "My" particle = slot `slot` of the centre cell, loaded direct from global.
let me = cellParticles[center * C.maxPerCell + slot];

// 3. Collide against the compact staged list — real length, not 144.
let n = atomicLoad(&sCount[b]);
for (var s = 0u; s < n; s++) {
  if (sIdx[region + s] == me) { continue; }
  dv += collidePair(mp, mv, mr, sPos[region+s], sVel[region+s], sSize[region+s], &heat);
}
```

The key win still holds — each cell's particle data is read from global memory
**once** by the workgroup, then shared — but now the workgroup is wave-sized and
the collide loop is occupancy-sized. Shared budget: `4 * 144` entries ×
(pos 8 + vel 8 + size 4 + idx 4) ≈ **13.8 KB**, under the 16 KB floor.

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
- **Full waves** — `BUCKETS_PER_WG` packs the workgroup to 64 lanes, so it does
  not waste 1/2–3/4 of every wave the way a 16-lane workgroup does.
- **Occupancy-sized collide** — compact staging walks the real neighbour count,
  not a flat 144 per particle.
- **Still loses when sparse** — these two fixes shrink B's overhead but do not
  remove its fundamental cost: B launches one workgroup per `BUCKETS_PER_WG`
  *occupied cells* (64 threads each), while [mode A](mode-a-linked-list-grid.md)
  launches one thread per *particle*. At low occupancy that's a large constant
  multiple of wasted threads staging near-empty neighbourhoods.

### Measured (1M particles, this machine)

| avg particles / cell | mode A | mode B |
| --- | --- | --- |
| 0.2 (sparse, `cellScale` 1) | **4.9 ms** | 21.3 ms |
| 13 (dense, `cellScale` 8)   | 24.8 ms | **7.3 ms** |

B wins decisively only once cells are near `MAX_PER_CELL` full — exactly where
the shared-memory reuse amortises. When cells are nearly empty, A's
one-thread-per-particle walk is unbeatable. Pick the mode to match occupancy
(raise `cellScale` to make cells fuller).
