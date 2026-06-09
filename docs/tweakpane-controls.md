# Single-source-of-truth controls + live metrics (Tweakpane)

A pattern for wiring a debug UI to a simulation without scattering defaults,
ranges, and change-handlers across the codebase.

## One control table drives everything

Declare each tunable **once**, with its default value sitting right next to its
slider range and behaviour. The params object, the UI bindings, and the
rebuild-on-change wiring are all derived from this single list:

```ts
type Control = {
  key: string;
  value?: string | number | boolean;     // default; omitted = bind to a live engine field
  target?: "engine";                      // bind to the engine instead of the params object
  folder?: string;                        // group under a collapsible folder
  rebuild?: "last" | "always";            // reallocate buffers on change
  opts?: Record<string, unknown>;         // min/max/step/label/options passed to the pane
};

const CONTROLS: Control[] = [
  { key: "numParticles", value: 8000, rebuild: "last", opts: { min: 100, max: 1000000, step: 1000 } },
  { key: "viewSize",     target: "engine", opts: { min: 0.1, max: 6, step: 0.05, label: "zoom" } },
  // ...
];
```

Derive the runtime params object from the entries that have a default:

```ts
const params = Object.fromEntries(
  CONTROLS.filter(c => c.value !== undefined).map(c => [c.key, c.value]),
);
```

Then build the pane in a loop, binding each control to either `params` or the
live `engine`, into an optional folder:

```ts
for (const c of CONTROLS) {
  const obj = c.target === "engine" ? engine : params;
  const parent = c.folder ? (folders[c.folder] ??= pane.addFolder({ title: c.folder })) : pane;
  const b = parent.addBinding(obj, c.key, c.opts ?? {});
  if (c.rebuild === "last")   b.on("change", e => { if (e.last) engine.rebuild(); });
  if (c.rebuild === "always") b.on("change", () => engine.rebuild());
}
```

Adding a tunable = adding one row. Default and range never drift apart.

### `rebuild` semantics

Some changes (particle count, sizes, density) require **reallocating GPU
buffers** — expensive. Others (speed, restitution) just feed into the next
frame's uniforms and need no rebuild.

- `rebuild: "last"` — rebuild on slider *release* only (`ev.last`), so dragging a
  count slider doesn't reallocate every intermediate value.
- `rebuild: "always"` — rebuild on every change (for cheap-to-rebuild things like
  grid cell scale).
- omitted — no rebuild; the value is read fresh each frame.

## Metrics pane: numeric readout + graph per row

For each live metric, add two bindings to the same field: a formatted numeric
readout, then a graph view with a blank label so they read as one row. Mark them
`readonly` and `refresh()` the pane periodically (e.g. 4×/sec), not every frame:

```ts
met.addBinding(m, "fps", { readonly: true, format: v => v.toFixed(0) });
met.addBinding(m, "fps", { readonly: true, view: "graph", min: 0, max: 165, label: " " });
```

Gate expensive metric collection (and the stats GPU passes that feed it) behind
the panel's visibility, so a hidden HUD costs nothing — see
[grid-occupancy-overlay](grid-occupancy-overlay.md) and
[gpu-profiling-and-readback](gpu-profiling-and-readback.md).

## Graceful capability degradation

Probe optional capabilities and only add the corresponding rows when present:
`performance.memory` (JS heap, Chromium-only) and `timestamp-query` (GPU compute
ms). The UI adapts to what the browser actually supports.
