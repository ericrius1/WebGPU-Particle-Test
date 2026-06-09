# Spatial-hash grid for neighbour queries

## Problem

Pairwise interaction (collision, attraction, SPH) is `O(N²)` if every element
tests every other. For short-range interaction most pairs are too far apart to
matter.

## Approach

Overlay a **uniform grid** on the world. Each element hashes to the cell
containing it; interaction only tests the element's own cell plus the immediate
neighbour cells. Cost drops to ~`O(N)` when occupancy per cell is bounded.

### Cell size = largest interaction radius

Set the cell side to the **diameter of the largest particle** (`2 * maxSize`).
Then any two overlapping particles are guaranteed to share a cell or sit in
adjacent cells, so testing the **3×3 neighbourhood** of a particle's cell finds
every possible contact. Particles may be any size up to that max.

An optional `cellScale` multiplier trades grid resolution for neighbour-list
length (bigger cells = fewer cells but more candidates per cell).

```wgsl
cellSize = max(2 * maxSize * cellScale, 1e-3);  // never zero
```

### Coordinate ↔ index helpers

Clamp to the grid bounds so out-of-box positions still map to a valid edge cell
(cheaper and safer than a branch):

```wgsl
fn cellCoord(p: vec2<f32>) -> vec2<i32> {
  let g = vec2<i32>(floor(p / C.cellSize));
  return clamp(g, vec2<i32>(0), vec2<i32>(i32(C.gridW) - 1, i32(C.gridH) - 1));
}
fn cellIndex(c: vec2<i32>) -> u32 {
  return u32(c.y) * C.gridW + u32(c.x);
}
```

### Neighbourhood iteration

Walk `dx, dy ∈ {-1, 0, 1}` and skip cells outside the grid:

```wgsl
for (var dy = -1; dy <= 1; dy++) {
  for (var dx = -1; dx <= 1; dx++) {
    let nc = cc + vec2<i32>(dx, dy);
    if (nc.x < 0 || nc.y < 0 || nc.x >= i32(C.gridW) || nc.y >= i32(C.gridH)) { continue; }
    // ... test candidates in cell nc ...
  }
}
```

## Keeping occupancy bounded (constant density)

The grid only stays near-`O(N)` if particles-per-cell stays bounded as `N`
grows. Achieve that by **growing the world with N** instead of packing more
particles into a fixed box. After picking sizes, set the box side so the covered
fraction (`density`/`coverage`) is constant:

```
L = sqrt(totalParticleArea / density)
gridW = gridH = floor(L / cellSize)
```

Now 8 000 and 400 000 particles look equally sparse and occupancy stays in the
same range — see [camera-pan-zoom](camera-pan-zoom.md) for how the camera frames
a fixed window into the growing world.

## Two ways to store cell membership

The grid is the same; how you record "which particles are in this cell" gives
two implementations with different performance profiles:

- [mode-a-linked-list-grid.md](mode-a-linked-list-grid.md) — per-particle linked
  list (no per-cell cap; one thread per particle re-reads global memory).
- [mode-b-bucket-shared-memory.md](mode-b-bucket-shared-memory.md) — counting
  buckets with a fixed cap; one workgroup per occupied cell stages neighbours in
  shared memory.

Neither is unconditionally faster: dense cells favour B (amortised shared load),
sparse cells favour A (short lists, no fixed staging cost).
