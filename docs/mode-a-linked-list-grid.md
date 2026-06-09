# Mode A — per-particle linked-list grid

One way to record cell membership in a [spatial-hash grid](spatial-hash-grid.md).
No fixed per-cell capacity; one GPU thread per particle.

## Data structures

```wgsl
gridHead : array<atomic<i32>>;  // one entry per cell, head of that cell's list, -1 = empty
gridNext : array<i32>;          // one entry per particle, next index in the same cell's list
```

A classic intrusive singly-linked list, but the "next" pointers live in a
parallel array indexed by particle, and the heads live in a per-cell array.

## Passes

### clear

Reset every cell head to `-1` (empty):

```wgsl
atomicStore(&gridHead[i], -1);
```

### bin — atomic prepend

Each particle prepends itself to its cell's list with a single atomic. The old
head becomes this particle's `next`:

```wgsl
let cell = cellIndex(cellCoord(particles[i].pos));
let prev = atomicExchange(&gridHead[cell], i32(i));
gridNext[i] = prev;
```

`atomicExchange` makes insertion lock-free and order-independent — multiple
particles racing to insert into the same cell each get a consistent chain.

### collide — one thread per particle walks the 3×3 lists

```wgsl
for (var dy = -1; dy <= 1; dy++) {
  for (var dx = -1; dx <= 1; dx++) {
    // skip out-of-grid neighbours
    var j = atomicLoad(&gridHead[cellIndex(nc)]);
    var guard = 0u;
    loop {
      if (j < 0 || guard > C.numParticles) { break; }   // -1 terminates; guard = corruption safety
      let uj = u32(j);
      if (uj != i) {
        let o = particles[uj];
        dv += collidePair(me.pos, me.speed, me.size, o.pos, o.speed, o.size, &heat);
      }
      j = gridNext[uj];
      guard++;
    }
  }
}
newSpeed[i] = me.speed + dv;
newTemp[i]  = me.temp + heat;
```

The `guard` counter caps the walk at the particle count so a corrupted pointer
(should never happen) can't hang the GPU in an infinite loop.

## Characteristics

- **No per-cell cap** — a cell can hold arbitrarily many particles; nothing is
  dropped. The overlay scales heat by the observed busiest cell since there is no
  fixed maximum.
- **Memory reads from global** — every thread re-reads its neighbours' particle
  data straight from global memory, with no sharing between threads in a
  workgroup.
- **Wins when sparse** — short lists mean little work; no fixed staging cost.
  Loses to [mode B](mode-b-bucket-shared-memory.md) when cells are dense, because
  B amortises neighbour reads across a workgroup.
