# Approach docs — GPU particle simulation techniques

Modular, project-agnostic write-ups of the techniques used in this WebGPU
particle sim. Each file documents **one technique** so it can be lifted into a
different project (or handed to another AI) on its own. They were extracted from
the inline source comments so the code stays clean and the knowledge stays
reusable.

Read order for a newcomer: `wgsl-module-assembly` → `spatial-hash-grid` →
`collision-response` → one of the two grid modes → `instanced-circle-rendering`.

| Doc | Technique |
| --- | --- |
| [wgsl-module-assembly.md](wgsl-module-assembly.md) | Splitting WGSL into reusable modules; host/device constant mirroring |
| [spatial-hash-grid.md](spatial-hash-grid.md) | Uniform grid for near-`O(N)` neighbour queries |
| [collision-response.md](collision-response.md) | Velocity-only elastic impulse with mass ∝ size² |
| [particle-integration.md](particle-integration.md) | Integrate + wall bounce + apply pass, speed clamp, heat decay |
| [mode-a-linked-list-grid.md](mode-a-linked-list-grid.md) | Per-particle linked-list binning |
| [mode-b-bucket-shared-memory.md](mode-b-bucket-shared-memory.md) | Counting buckets, indirect dispatch, shared-memory neighbour staging |
| [instanced-circle-rendering.md](instanced-circle-rendering.md) | Instanced quads → soft circles, camera, heat colour, aspect fit |
| [camera-pan-zoom.md](camera-pan-zoom.md) | View window into a world that grows with N; pan/zoom under cursor |
| [grid-occupancy-overlay.md](grid-occupancy-overlay.md) | Visualising grid occupancy / algorithm cost |
| [gpu-profiling-and-readback.md](gpu-profiling-and-readback.md) | `timestamp-query` timing; non-blocking async buffer readback |
| [tweakpane-controls.md](tweakpane-controls.md) | Single-source-of-truth control table + live metrics pane |

The repo-level overview (what the demo *is*) lives in the top-level
[../README.md](../README.md). These docs are the *how/why* of each piece.
