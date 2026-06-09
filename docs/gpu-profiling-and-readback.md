# GPU profiling + non-blocking buffer readback

Two related techniques for getting numbers *back* from the GPU without stalling
the render loop: `timestamp-query` for compute time, and double-buffered async
`mapAsync` for stats.

## Compute timing with `timestamp-query`

The feature is optional — request it only if the adapter has it, and degrade
gracefully:

```ts
const wantTs = adapter.features.has("timestamp-query");
const device = await adapter.requestDevice({
  requiredFeatures: wantTs ? ["timestamp-query"] : [],
});
```

Create a 2-slot query set and two small buffers (resolve target + mappable
readback). Wrap the compute pass(es) you want to time with `timestampWrites`,
writing the timestamp at the **beginning** of the first pass and the **end** of
the last:

```ts
querySet = device.createQuerySet({ type: "timestamp", count: 2 });
// pass 1: beginningOfPassWriteIndex: 0
// last pass: endOfPassWriteIndex: 1
```

After the passes, resolve and copy to a mappable buffer:

```ts
enc.resolveQuerySet(querySet, 0, 2, tsResolve, 0);
enc.copyBufferToBuffer(tsResolve, 0, tsResult, 0, 16);
```

The two timestamps are `u64` nanoseconds; the delta is GPU wall-clock time for
the timed region. Smooth it (EWMA) for a stable readout:

```ts
const ns = Number(t[1] - t[0]);              // BigInt64Array
if (ns > 0) gpuMs = gpuMs * 0.85 + (ns / 1e6) * 0.15;
```

**Keep untimed work out of the timed region.** Occupancy stats and the render
pass run *after* the timestamped passes so they don't inflate the compute number.

## Non-blocking readback (don't `await` in the loop)

`mapAsync` returns a promise that resolves *frames later*. Awaiting it would stall
the render loop. Instead, fire-and-forget with a **pending flag** so only one
readback is in flight at a time, and read whatever resolves whenever it resolves:

```ts
if (!pending) {
  pending = true;
  buf.mapAsync(GPUMapMode.READ).then(() => {
    const m = new Uint32Array(buf.getMappedRange().slice(0));  // copy out before unmap
    occupied = m[0]; maxCell = m[1]; overflow = m[2];
    buf.unmap();
    pending = false;
  }).catch(() => { pending = false; });
}
```

Notes:
- `.slice(0)` copies the mapped range into a detached array buffer *before*
  `unmap()` — the mapped view is invalidated by `unmap`.
- The pending flag prevents queuing a second `mapAsync` on an already-mapped
  buffer (which throws). You read slightly stale stats; for a debug HUD that's
  fine.
- Apply the same pattern to the timestamp result buffer.

## Rough GPU memory accounting

There's no API for "how much VRAM did I allocate," so sum your own buffer sizes
when you allocate them and surface that as a metric (particles + side buffers +
grid A + grid B + stats). It's an approximation but tracks real allocation.
