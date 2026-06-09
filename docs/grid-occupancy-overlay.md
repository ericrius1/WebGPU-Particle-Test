# Grid occupancy overlay + stats

A debug visualisation that draws the hash grid and colours each cell by how full
it is — so you can *see* how particles bin, where buckets fill toward capacity,
and where [mode B](mode-b-bucket-shared-memory.md) would drop overflow.

## A mode-independent count pass

Occupancy is gathered by a tiny stats pair that runs regardless of which
collision mode is live, so the overlay means the same thing in both:

- **clearStat** — zero a per-cell count buffer and a 4-entry meta buffer.
- **binStat** — re-bin every particle into the count buffer and accumulate
  aggregate metrics atomically:

```wgsl
let old = atomicAdd(&statCount[cell], 1u);
if (old == 0u)             { atomicAdd(&statMeta[0], 1u); }  // occupied cells
atomicMax(&statMeta[1], old + 1u);                           // max particles in any cell
if (old >= C.maxPerCell)   { atomicAdd(&statMeta[2], 1u); }  // overflow (mode B would drop)
```

Run it **only while the debug panel is open**, and **outside the timed compute
region**, so it never pollutes the `compute ms` measurement (see
[gpu-profiling-and-readback](gpu-profiling-and-readback.md)).

## Overlay draw — one instanced quad per cell

Like the particle renderer, the overlay is one instanced draw (`draw(6,
numCells)`), one quad per cell, positioned from `instance_index`:

```wgsl
let cx = f32(inst % C.gridW);
let cy = f32(inst / C.gridW);
let world = (vec2<f32>(cx, cy) + q) * C.cellSize;     // cell quad in world space
out.clip = vec4<f32>(fit((world - C.viewCenter) / vh), 0.0, 1.0);  // same camera as particles
```

## The look reflects the live algorithm

The overlay reads a `mode` flag from the uniforms and changes meaning so it
honestly represents the cost of whichever algorithm is running:

- **Mode A (linked list)** — no per-cell cap, so heat shows *relative list
  length* = traversal work. Scale by the busiest observed cell (`maxObs`, fed
  back from last frame's stats) to keep the gradient readable. Teal→magenta
  "cost" ramp.
- **Mode B (bucket)** — fixed capacity, so heat is a *capacity meter*:
  `count / MAX_PER_CELL`. Blue→red ramp. Cells at/over capacity get a bold red
  stripe pattern — exactly where B silently drops collisions.

```wgsl
if (C.mode == 1u) {                       // B: occupancy vs fixed capacity
  out.occ = f32(cnt) / f32(C.maxPerCell);
  if (cnt >= C.maxPerCell) { out.cap = 1.0; }
} else {                                  // A: scale by busiest cell
  out.occ = f32(cnt) / max(C.maxObs, 1.0);
}
```

The fragment shader draws grid lines (distance-to-cell-edge `smoothstep`), fills
proportional to occupancy, and for over-capacity B cells draws an animated stripe
warning. Blend over the scene with alpha.

## Feeding stats back to the UI

The meta buffer (`[occupied, maxPerCell, overflow, _]`) is read back
asynchronously and surfaced as live metrics; `maxObs` also feeds next frame's
overlay scale for mode A. Readback is non-blocking and double-buffered — see
[gpu-profiling-and-readback](gpu-profiling-and-readback.md).
