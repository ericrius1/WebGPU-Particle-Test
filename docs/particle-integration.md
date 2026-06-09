# Integrate / apply passes — motion, walls, and double-buffered velocity

The simulation step is split so that collision results don't fight with each
other while being computed. Three concerns, three passes:

1. **integrate** — advance position, bounce off walls.
2. **collide** — read current state, *write proposed* new velocity/heat to
   separate buffers (mode A or B; see those docs).
3. **apply** — commit the proposed velocity/heat back onto the particles.

## Why separate `newSpeed` / `newTemp` buffers

The collide pass reads every neighbour's **current** velocity. If it wrote
directly back into the particle buffer, a neighbour processed later in the same
pass would read an already-modified velocity — order-dependent, race-prone
results. Writing proposals to side buffers (`newSpeed`, `newTemp`) and applying
them in a later pass makes the collide step a pure function of the frame's
starting state.

## Integrate

```wgsl
// move + bounce off the box walls (axis-aligned, world side = w)
p.pos += p.speed * C.dt;
let r = p.size;
let w = C.worldSize;
if (p.pos.x < r)     { p.pos.x = r;     p.speed.x =  abs(p.speed.x); }
if (p.pos.x > w - r) { p.pos.x = w - r; p.speed.x = -abs(p.speed.x); }
if (p.pos.y < r)     { p.pos.y = r;     p.speed.y =  abs(p.speed.y); }
if (p.pos.y > w - r) { p.pos.y = w - r; p.speed.y = -abs(p.speed.y); }
```

Wall handling clamps position to the wall *and* forces the velocity sign
outward (`abs` / `-abs`) rather than negating — this prevents a particle that is
pushed past a wall by a collision from getting stuck vibrating against it.

## Apply

```wgsl
var v = newSpeed[i];
let m = length(v);
let vmax = C.worldSize * 0.4;
if (m > vmax) { v *= vmax / m; }                 // clamp runaway collisions
particles[i].speed = v;
particles[i].temp  = max(newTemp[i] * C.tempDecay, 0.0);
```

- **Speed clamp.** Dense simultaneous collisions can sum to an enormous delta;
  capping speed at a fraction of the world size keeps the sim stable without
  visibly affecting normal motion.
- **Heat decay.** `temp` is multiplied by `tempDecay` (e.g. 0.92) each frame and
  floored at 0, producing the fade-out of the collision flash.

## Frame-rate safety

On the host, clamp `dt` before feeding it in (`min(dt, 1/30) * speed`) so a
dropped frame or a paused tab doesn't teleport particles through walls. `paused`
is implemented as `dt = 0` (still renders, no motion).
