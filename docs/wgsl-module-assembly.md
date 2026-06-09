# WGSL module assembly

## Problem

WGSL has no `#include`. A multi-pass GPU program duplicates the same struct
definitions, grid helpers, and binding declarations across every shader. Copy-
paste drifts out of sync.

## Approach

Keep one WGSL file per pass, plus shared fragments, and concatenate them on the
host at load time. A shader module = **shared header** + **binding block** +
**entry-point body**, joined with newlines:

```
module = common.wgsl + bind<X>.wgsl + <pass>.wgsl
```

- `common.wgsl` — everything shared: `struct` defs, the uniform `Constants`
  layout, grid helper fns, compile-time `const`s, and the collision kernel. It
  is prepended to *every* module.
- `bind*.wgsl` — just the `@group/@binding` declarations for a family of passes.
  Splitting bindings out lets several passes share one binding set, and lets one
  pass swap binding sets (e.g. a collide pass that must *not* bind an indirect
  buffer as storage — see [mode-b](mode-b-bucket-shared-memory.md)).
- `<pass>.wgsl` — only the `@compute`/`@vertex`/`@fragment` entry point and its
  body. No structs, no bindings — those come from the parts above.

A trivial linker is enough:

```ts
const link = (...parts: string[]) => parts.join("\n");
export const COLLIDE_A_WGSL = link(common, bindSim, bindA, collideA);
```

### Loading raw shader text

With Vite, import `.wgsl` as a raw string via the `?raw` suffix, and register
the extension in `vite.config.ts`:

```ts
// vite.config.ts
assetsInclude: ["**/*.wgsl"]
```

```ts
import common from "./common.wgsl?raw";
```

Other bundlers have equivalents (`raw-loader`, `?raw`, asset/source). The point
is the shader stays a real `.wgsl` file (editor highlighting, no JS escaping),
loaded as text.

## Host/device constant mirroring — the sync trap

Compile-time sizes that the **shader** needs as `const` (workgroup size, per-cell
capacity) are *also* needed on the **host** to compute dispatch counts and buffer
sizes. They live in two places:

```wgsl
// common.wgsl  (device side)
const WG           : u32 = 64;
const MAX_PER_CELL : u32 = 16;
```

```ts
// shaders/index.ts  (host side)
export const WG = 64;
export const MAX_PER_CELL = 16;
```

**These must stay equal.** If they diverge, dispatch counts and buffer
allocations no longer match what the shader indexes, producing silent
out-of-bounds or under-utilisation rather than a clean error. There is no
language-level link between them — treat the pair as a single value edited in
lockstep, and document it loudly (this doc, plus adjacent placement in both
files).

Derived device-only consts can be expressed in terms of the mirrored ones, e.g.
`const SHARED : u32 = MAX_PER_CELL * 9u;` (3×3 neighbourhood capacity).
