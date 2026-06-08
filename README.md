# WebGPU Particles — Spatial-Hash Collisions

A pure **WebGPU + TypeScript** port of Diligent Engine's
[Tutorial14 ComputeShader](https://github.com/DiligentGraphics/DiligentSamples/tree/master/Tutorials/Tutorial14_ComputeShader):
thousands of different-sized circles bouncing in a box, colliding through a
spatial-hash grid — entirely on the GPU. No physics library, no CPU step; every
particle integrates, bins, collides, and shades in compute/render passes.
Colour encodes kinetic energy plus a collision-heat flash.

```bash
npm install
npm run dev      # http://localhost:5190 (or whatever Vite picks)
```

Needs a WebGPU browser (Chrome/Edge 113+). Press **`/`** for the debug UI.

---

## The algorithm

Collision detection is the cost. Naive all-pairs is `O(N²)`; this uses a
**uniform spatial hash** to make it near-`O(N)`.

The grid cell size = **diameter of the largest particle**, so any particle can
only overlap something in its own cell or the 8 neighbours. Collision becomes:
hash each particle into a cell, then test only the 3×3 neighbourhood. Particles
may be any size up to that max.

Two interchangeable implementations of the same grid bracket the performance
design space — toggle between them at runtime and watch `compute ms` cross over.

### Mode A — per particle (linked list)

The original Diligent approach. Per frame:

1. `clear`     — grid head pointers → −1
2. `integrate` — move + bounce off walls
3. `bin`       — each particle `atomicExchange`es itself into its cell's
   head-linked list
4. `collide`   — **one thread per particle** walks the 3×3 neighbourhood's
   linked lists, accumulating elastic impulses
5. `apply`     — commit new velocity + temperature

One thread per particle; every thread re-traverses its neighbour lists straight
from global memory.

### Mode B — per occupied bucket (shared memory)

Iterates by **bucket** instead of by particle.

- `bin` uses counting buckets: `atomicAdd` on a per-cell counter stores up to
  `MAX_PER_CELL` (16) particle indices per cell. The **first** particle to land
  in a cell (`slot == 0`) appends that cell to an `occupied` list via an atomic
  counter — and that counter doubles as the `x` of an **indirect-dispatch**
  buffer.
- `collide` dispatches **one workgroup per occupied bucket**
  (`dispatchWorkgroupsIndirect`). Its 16 threads cooperatively stage the entire
  3×3 neighbourhood — `16 × 9 = 144` particles — into **workgroup shared
  memory**, `workgroupBarrier()`, then each thread collides its own particle
  against the shared set. Neighbours are read from global memory **once per
  workgroup** instead of once per particle.

Empty cells cost nothing (never dispatched). Cells over capacity drop the
overflow — the 16-per-cell budget the design assumes.

> The indirect-args buffer is written as storage in `bin` and read as `INDIRECT`
> in `collide`. Those uses live in **separate compute passes**, otherwise WebGPU
> rejects the buffer for mixing writable-storage and indirect usage in one sync
> scope.

### A vs B — the tradeoff

Mode B is **not** unconditionally faster. `compute ms` (timestamp-query
wall-clock GPU time) shows where each wins:

- **Dense** cells (near `MAX_PER_CELL`): **B wins** — the 144-entry shared load
  amortises over many particles; neighbours read once per workgroup.
- **Sparse** cells (≪1 particle/cell, the reference-density default): **B
  loses** — every occupied bucket still pays the full shared-memory load for
  mostly-empty neighbourhoods, while A just walks a couple of short lists.

Change density/count and the two curves cross.

---

## Collision response

`collidePair` (in `common.wgsl`) is a velocity-only elastic impulse with
**mass ∝ size²**:

```
j = -(1 + restitution) · v_rel · (m_b / (m_a + m_b))
```

applied along the contact normal. A small positional push (`sep`) de-penetrates
deep overlaps so they don't stick and churn impulses; pairs already separating
get only the push. A speed clamp guards against runaway collision feedback.
Each impulse magnitude feeds **collision heat** (`temp`), which decays each
frame — that's the red flash on impact.

---

## Density & camera

To stay sparse like the reference at any particle count, the **world grows with
N**: after sizes are picked, the box side is

```
L = sqrt(totalParticleArea / density)
```

so covered fraction is constant regardless of `N` (8 000 and 40 000 look equally
sparse). The grid scales with `L`, keeping bucket occupancy bounded.

A fixed-size **view window** (`viewSize` = zoom, `viewCenter` = pan) is rendered
into that world, so particles stay big on screen — you watch a window into a
larger sim. Rendering is letterboxed to a square so circles stay round on any
aspect.

---

## Debug UI (`/`)

Two panes:

- **controls** (right): collision mode (A/B), particle count, density, zoom,
  speed, restitution, heat, particle size, **grid overlay**, pause.
- **metrics** (left): FPS, frame ms (CPU), compute ms (GPU timestamp-query),
  JS heap + GPU buffer memory, and live grid occupancy (cells, occupied,
  occupied %, avg/occupied cell, max/cell, overflow drops).

### Grid overlay

Toggle it to draw the hash grid: cell lines + an occupancy heatmap (blue→red).
Lets you *see* the algorithm — how particles bin, where buckets fill toward
`MAX_PER_CELL`, and where Mode B would drop overflow. Occupancy comes from a
tiny, mode-independent count pass that runs **only** while the panel is open
(outside the timed region, so it doesn't pollute `compute ms`).

---

## Layout

- `src/shaders/` — all WGSL, one file per pass. `common.wgsl` holds the shared
  structs, grid helpers, and `collidePair`; per-pass binding blocks + bodies are
  glued onto it in `index.ts`. Compile-time constants (`WG = 64`,
  `MAX_PER_CELL = 16`) live in `common.wgsl` and are mirrored in `index.ts` for
  host-side dispatch/buffer math — they **must** stay in sync.
- `src/engine.ts` — buffers, bind groups, pipelines, per-frame pass scheduling.
- `src/main.ts` — device init, canvas, Tweakpane debug, main loop.

Vite loads `.wgsl` as raw strings (`?raw` + `assetsInclude` in
`vite.config.ts`).
