# WebGPU Particles — Spatial-Hash Collisions

Pure WebGPU + TypeScript port of Diligent Engine's
[Tutorial14 ComputeShader](https://github.com/DiligentGraphics/DiligentSamples/tree/master/Tutorials/Tutorial14_ComputeShader):
thousands of different-sized circles bouncing in a box, colliding via a spatial
hash grid, all on the GPU. Colour = collision temperature.

```bash
npm install
npm run dev      # http://localhost:5190 (or whatever Vite picks)
```

Needs a WebGPU browser (Chrome/Edge 113+).

Press **`/`** for the debug panel (Tweakpane): FPS, **compute ms/frame** (GPU
timestamp-query), collision mode, particle count, density, zoom, speed,
restitution, heat, particle size, pause.

## Density & camera

To stay sparse like the reference at any count, the **world grows with the
particle count**: after picking sizes, the box side is
`L = sqrt(totalParticleArea / density)`, so the covered fraction is constant
regardless of `N` (8000 and 40000 look equally sparse). A fixed-size **view
window** (`zoom`) is rendered into that world, so particles stay big on screen —
you watch a window into a larger sim. The grid scales with `L`, so bucket
occupancy stays bounded.

Colour = kinetic energy (speed → blue…green…yellow) plus a collision-heat flash.

## GPU timing — the A vs B tradeoff

`compute ms` is wall-clock GPU time for the compute passes (timestamp-query).
Mode B is **not** unconditionally faster:

- **Dense** cells (near `MAX_PER_CELL`): B wins — the 144-entry shared load is
  amortised over many particles, neighbours read once per workgroup.
- **Sparse** cells (≪1 particle/cell, as in this reference-density default): B
  *loses* — every occupied bucket still pays the full shared-memory load for
  mostly-empty neighbourhoods, while A just walks a couple of short lists.

So the two modes bracket the design space; switch density/count and watch the
`compute ms` graph cross over.

## Two collision modes

The grid is a uniform hash: cell size = diameter of the **largest** particle, so
a particle only ever touches its own cell + the 8 neighbours. Particles may be
any size up to that max.

### Mode A — per particle (linked list)
The original Diligent approach.

1. `clear`   — grid head pointers → −1
2. `integrate` — move + bounce off walls
3. `bin`     — each particle `atomicExchange`es itself into its cell's head-linked list
4. `collide` — **one thread per particle** walks the 3×3 neighbourhood's linked
   lists, accumulating elastic impulses
5. `apply`   — commit new velocity + temperature

One thread per particle; every thread re-traverses its neighbour lists from
global memory.

### Mode B — per occupied bucket (shared memory)
Iterates by **bucket** instead of by particle, the alternative you described.

- `bin` uses counting buckets: `atomicAdd` on a per-cell counter, storing up to
  `MAX_PER_CELL` (16) particle indices per cell. The **first** particle to land
  in a cell (`slot == 0`) appends the cell to an `occupied` list via an atomic
  counter — that counter doubles as the `x` of an **indirect dispatch** buffer.
- `collide` dispatches **one workgroup per occupied bucket**
  (`dispatchWorkgroupsIndirect`). Its 16 threads cooperatively stage the whole
  3×3 neighbourhood — `16 × 9 = 144` particles — into **workgroup shared
  memory**, barrier, then each thread collides its own particle against the
  shared set. Neighbours are read from global memory once per workgroup instead
  of once per particle.

Empty cells cost nothing (never dispatched). Cells over capacity drop the
overflow (the 16-per-cell budget the design assumes).

> The indirect-args buffer is written as storage in the bin pass and read as
> `INDIRECT` in the collide pass — those uses live in **separate compute
> passes**, otherwise WebGPU rejects the buffer for mixing writable-storage and
> indirect usage in one sync scope.

## Layout

- `src/shaders.ts` — all WGSL (shared structs + both pipelines), compile-time
  constants (`WG`, `MAX_PER_CELL`) injected.
- `src/engine.ts` — buffers, bind groups, pipelines, per-frame pass scheduling.
- `src/main.ts` — device init, canvas, Tweakpane debug, main loop.

## Notes

- Velocity-only elastic response (mass ∝ size²) with a small positional push so
  deep overlaps de-penetrate instead of sticking; a speed clamp guards against
  runaway collision feedback.
- Simulation runs in a unit `[0,1]²` box, rendered letterboxed to a square so
  circles stay round on any aspect.
