# Camera — a fixed window into a world that grows with N

## World grows with particle count

To keep visual density constant as `N` changes, the **world box grows with N**
rather than packing more particles into a fixed box. After particle sizes are
chosen, the box side is set from the target covered fraction (`density`):

```
L = max( sqrt(totalParticleArea / density), 4 * maxSize )
```

so the fraction of the box covered by particles is constant — 8 000 and 400 000
particles look equally sparse, and grid occupancy stays bounded (see
[spatial-hash-grid](spatial-hash-grid.md)). The lower bound keeps the box sane
for tiny counts.

Initial velocities also scale with `L` (`(0.05–0.20) * L`) so motion looks the
same regardless of world size.

## A view window keeps particles big on screen

Because the world can be huge, the camera renders a fixed-size **window** into
it, defined by two values passed to the shader:

- `viewSize` — side length of the on-screen window in world units (zoom).
- `viewCenter` — world point at the centre of the screen (pan).

A `viewSize`-wide window centred on `viewCenter` maps to NDC `[-1,1]`. You watch a
window into a larger sim; particles stay large on screen at any `N`. On rebuild,
recentre on the world (`viewCenter = L/2`) and clamp `viewSize` to `L`.

## Screen pixel → world coordinate

The inverse of the camera transform, needed for pan and zoom-under-cursor. It
accounts for the same aspect-fit letterboxing the renderer uses:

```ts
function screenToWorld(px, py, vs = engine.viewSize): [number, number] {
  const aspect = canvas.width / canvas.height;
  const ndcX = (px / window.innerWidth) * 2 - 1;
  const ndcY = 1 - (py / window.innerHeight) * 2;   // y is flipped vs screen
  const vh = vs * 0.5;
  const wx = viewCenterX + ndcX * vh * (aspect > 1 ? aspect : 1);
  const wy = viewCenterY + ndcY * vh * (aspect > 1 ? 1 : 1 / aspect);
  return [wx, wy];
}
```

## Drag to pan

Convert both the previous and current pointer positions to world space and shift
the centre by their difference, so the world point under the cursor follows the
cursor exactly:

```ts
const [wx0, wy0] = screenToWorld(lastPx, lastPy);
const [wx1, wy1] = screenToWorld(e.clientX, e.clientY);
viewCenterX -= wx1 - wx0;
viewCenterY -= wy1 - wy0;
```

Use pointer capture so a drag that leaves the canvas keeps tracking.

## Wheel / pinch to zoom under the cursor

Trackpad pinch arrives as `ctrlKey + wheel`; a normal wheel also zooms (with a
gentler factor). Zoom is exponential in `deltaY` so it feels uniform. To keep the
point under the cursor fixed, compute its world position at the old and new
`viewSize` and shift the centre by the difference:

```ts
const factor = Math.exp(e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
const newVs = clamp(viewSize * factor, 0.05, worldSize * 2);
const [wx,  wy ] = screenToWorld(e.clientX, e.clientY, viewSize);
const [wx2, wy2] = screenToWorld(e.clientX, e.clientY, newVs);
viewCenterX += wx - wx2;
viewCenterY += wy - wy2;
viewSize = newVs;
```

Call `preventDefault()` (passive: false) so the page doesn't scroll/zoom.
