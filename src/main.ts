import { Pane } from "tweakpane";
import { Engine, type SimParams } from "./engine";

async function boot() {
  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const noGpu = document.getElementById("nowebgpu")!;

  if (!navigator.gpu) { noGpu.style.display = "grid"; return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { noGpu.style.display = "grid"; return; }
  const wantTs = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: wantTs ? ["timestamp-query"] : [],
    // Default limits cap buffers at 256MB; 1M particles needs bigger grid
    // buffers. Request the most this adapter allows.
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
  device.lost.then((info) => console.error("device lost:", info.message));

  const ctx = canvas.getContext("webgpu")!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  type Control = {
    key: string;
    value?: string | number | boolean;
    target?: "engine";
    folder?: string;
    rebuild?: "last" | "always";
    opts?: Record<string, unknown>;
  };
  const CONTROLS: Control[] = [
    { key: "mode", value: "auto", opts: { options: { "Auto (switch A/B)": "auto", "A — per particle (linked list)": "A", "B — per bucket (shared mem)": "B" } } },
    { key: "numParticles", value: 1000000, rebuild: "last", opts: { min: 100, max: 2000000, step: 100 } },
    // Target fraction of the sim box covered by particles. Sets worldSize
    // (box shrinks as this rises -> tighter packing). Applied on slider release.
    { key: "coverage", value: 0.3, rebuild: "last", opts: { min: 0.02, max: 0.7, step: 0.01, label: "world density" } },
    // Grid cell is fixed at the largest particle's diameter (see engine).
    // Auto-switch knob: gpuParallel ~ workgroups B needs (ceil(occupied/4)) to
    // saturate this GPU; below it the sim falls back to mode A.
    { key: "gpuParallel", value: 2048, folder: "auto switch", opts: { min: 64, max: 8192, step: 64, label: "gpu parallel (wg)" } },
    { key: "viewSize", target: "engine", opts: { min: 0.1, max: 6, step: 0.05, label: "zoom (view)" } },
    { key: "speed", value: 0.02, opts: { min: 0, max: 0.1, step: 0.01 } },
    { key: "restitution", value: 1.0, opts: { min: 0, max: 1, step: 0.02 } },
    { key: "tempGain", value: 0.012, opts: { min: 0, max: 0.3, step: 0.005 } },
    { key: "tempDecay", value: 0.92, opts: { min: 0.8, max: 1, step: 0.005 } },
    { key: "minSize", value: 0.004, folder: "particle size", rebuild: "last", opts: { min: 0.002, max: 0.02, step: 0.001 } },
    { key: "maxSize", value: 0.011, folder: "particle size", rebuild: "last", opts: { min: 0.004, max: 0.03, step: 0.001 } },
    { key: "showGrid", value: true, opts: { label: "grid overlay" } },
    { key: "paused", value: false },
  ];

  const params = Object.fromEntries(
    CONTROLS.filter((c) => c.value !== undefined).map((c) => [c.key, c.value]),
  ) as unknown as SimParams;

  const engine = new Engine(device, ctx, format, params);
  (window as any).engine = engine;
  (window as any).params = params;

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(window.innerWidth * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    engine.setAspect(canvas.width / canvas.height);
  }
  resize();
  window.addEventListener("resize", resize);

  const hasHeap = !!(performance as any).memory;

  const ctrlWrap = document.createElement("div");
  ctrlWrap.style.cssText = "position:fixed;top:10px;right:10px;z-index:20;display:none;width:300px;";
  document.body.appendChild(ctrlWrap);
  const pane = new Pane({ container: ctrlWrap, title: "controls" });

  const folders: Record<string, any> = {};
  let modeBinding: any;
  for (const c of CONTROLS) {
    const parent = c.folder
      ? (folders[c.folder] ??= pane.addFolder({ title: c.folder, expanded: false }))
      : pane;
    // numParticles drives per-frame dispatch/draw counts and buffer sizes.
    // Bind to a holder and only commit (+rebuild) on slider release, so
    // dragging never runs the sim against mismatched buffers (avoids the
    // hitch and out-of-bounds access while scrubbing).
    if (c.key === "numParticles") {
      const holder = { numParticles: params.numParticles };
      const b = parent.addBinding(holder, "numParticles", c.opts ?? {});
      b.on("change", (ev: { last: boolean }) => {
        if (!ev.last) return;
        params.numParticles = holder.numParticles;
        engine.rebuild();
      });
      continue;
    }
    const obj = c.target === "engine" ? engine : params;
    const b = parent.addBinding(obj, c.key, c.opts ?? {});
    if (c.rebuild === "last") b.on("change", (ev: { last: boolean }) => { if (ev.last) engine.rebuild(); });
    else if (c.rebuild === "always") b.on("change", () => engine.rebuild());
    if (c.key === "mode") modeBinding = b;
  }

  const m = {
    active: "B",
    fps: 0, frameMs: 0, computeMs: 0,
    jsHeapMB: 0, gpuMemMB: 0,
    cells: 0, occupied: 0, occupiedPct: 0, avgPerCell: 0, maxPerCell: 0, overflow: 0,
    worldSize: 0, inView: 0,
  };
  const metWrap = document.createElement("div");
  metWrap.style.cssText = "position:fixed;top:10px;left:10px;z-index:20;display:none;width:280px;";
  document.body.appendChild(metWrap);
  const met = new Pane({ container: metWrap, title: "metrics" });

  met.addBinding(m, "active", { readonly: true, label: "active mode" });
  met.addBinding(m, "fps", { readonly: true, format: (v: number) => v.toFixed(0) });
  met.addBinding(m, "fps", { readonly: true, view: "graph", min: 0, max: 165, label: " " });
  met.addBinding(m, "frameMs", { readonly: true, format: (v: number) => v.toFixed(1) + " ms", label: "frame ms (cpu)" });
  met.addBinding(m, "frameMs", { readonly: true, view: "graph", min: 0, max: 33, label: " " });
  if (engine.canTimestamp) {
    met.addBinding(m, "computeMs", { readonly: true, format: (v: number) => v.toFixed(2) + " ms", label: "compute ms (gpu)" });
    met.addBinding(m, "computeMs", { readonly: true, view: "graph", min: 0, max: 8, label: " " });
  }
  const fMem = met.addFolder({ title: "memory" });
  if (hasHeap) fMem.addBinding(m, "jsHeapMB", { readonly: true, format: (v: number) => v.toFixed(1) + " MB", label: "js heap" });
  fMem.addBinding(m, "gpuMemMB", { readonly: true, format: (v: number) => v.toFixed(1) + " MB", label: "gpu buffers" });
  const fGrid = met.addFolder({ title: "grid occupancy" });
  fGrid.addBinding(m, "cells", { readonly: true, format: (v: number) => v.toFixed(0) });
  fGrid.addBinding(m, "occupied", { readonly: true, format: (v: number) => v.toFixed(0) });
  fGrid.addBinding(m, "occupiedPct", { readonly: true, format: (v: number) => v.toFixed(1) + " %", label: "occupied %" });
  fGrid.addBinding(m, "avgPerCell", { readonly: true, format: (v: number) => v.toFixed(2), label: "avg / occ cell" });
  fGrid.addBinding(m, "maxPerCell", { readonly: true, format: (v: number) => v.toFixed(0), label: "max / cell" });
  fGrid.addBinding(m, "overflow", { readonly: true, format: (v: number) => v.toFixed(0), label: "overflow (B drops)" });
  const fView = met.addFolder({ title: "world / view" });
  fView.addBinding(m, "worldSize", { readonly: true, format: (v: number) => v.toFixed(2), label: "world size" });
  fView.addBinding(m, "inView", { readonly: true, format: (v: number) => v.toFixed(0), label: "particles in view" });

  const hint = document.getElementById("hint");
  let debug = true;
  const applyDebug = () => {
    ctrlWrap.style.display = debug ? "block" : "none";
    metWrap.style.display = debug ? "block" : "none";
    if (hint) hint.style.display = debug ? "none" : "block";
    engine.collectStats = debug;
  };
  applyDebug();
  window.addEventListener("keydown", (e) => {
    if (e.key === "/") {
      e.preventDefault();
      debug = !debug;
      applyDebug();
    } else if (e.key === "m" || e.key === "M") {
      const order = ["auto", "A", "B"] as const;
      params.mode = order[(order.indexOf(params.mode as any) + 1) % order.length];
      modeBinding.refresh();
    }
  });

  function screenToWorld(px: number, py: number, vs = engine.viewSize): [number, number] {
    const aspect = canvas.width / canvas.height;
    const ndcX = (px / window.innerWidth) * 2 - 1;
    const ndcY = 1 - (py / window.innerHeight) * 2;
    const vh = vs * 0.5;
    const wx = engine.viewCenterX + ndcX * vh * (aspect > 1 ? aspect : 1);
    const wy = engine.viewCenterY + ndcY * vh * (aspect > 1 ? 1 : 1 / aspect);
    return [wx, wy];
  }

  let dragging = false;
  let lastPx = 0, lastPy = 0;
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true; lastPx = e.clientX; lastPy = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const [wx0, wy0] = screenToWorld(lastPx, lastPy);
    const [wx1, wy1] = screenToWorld(e.clientX, e.clientY);
    engine.viewCenterX -= wx1 - wx0;
    engine.viewCenterY -= wy1 - wy0;
    lastPx = e.clientX; lastPy = e.clientY;
  });
  const endDrag = (e: PointerEvent) => { dragging = false; canvas.releasePointerCapture?.(e.pointerId); };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = Math.exp(e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
    const newVs = Math.min(Math.max(engine.viewSize * factor, 0.05), engine.worldSize * 2);
    const [wx, wy] = screenToWorld(e.clientX, e.clientY, engine.viewSize);
    const [wx2, wy2] = screenToWorld(e.clientX, e.clientY, newVs);
    engine.viewCenterX += wx - wx2;
    engine.viewCenterY += wy - wy2;
    engine.viewSize = newVs;
  }, { passive: false });

  let last = performance.now();
  let acc = 0, frames = 0, frameMsSmooth = 0;
  function loop(now: number) {
    const dt = (now - last) / 1000;
    last = now;
    frameMsSmooth = frameMsSmooth * 0.9 + dt * 1000 * 0.1;
    engine.frame(dt);

    acc += dt; frames++;
    if (acc >= 0.25) {
      if (debug) {
        m.active = params.mode === "auto" ? `${engine.activeMode} (auto)` : engine.activeMode;
        m.fps = frames / acc;
        m.frameMs = frameMsSmooth;
        m.computeMs = engine.gpuMs;
        if (hasHeap) m.jsHeapMB = (performance as any).memory.usedJSHeapSize / 1048576;
        m.gpuMemMB = engine.gpuBytes / 1048576;
        m.cells = engine.numCells;
        m.occupied = engine.occupied;
        m.occupiedPct = engine.numCells ? (engine.occupied / engine.numCells) * 100 : 0;
        m.avgPerCell = engine.occupied ? params.numParticles / engine.occupied : 0;
        m.maxPerCell = engine.maxCell;
        m.overflow = engine.overflow;
        m.worldSize = engine.worldSize;
        // Fraction of the world the current view window covers, x particle count.
        const frac = Math.min(engine.viewSize / engine.worldSize, 1);
        m.inView = params.numParticles * frac * frac;
        met.refresh();
      }
      acc = 0; frames = 0;
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

boot();
