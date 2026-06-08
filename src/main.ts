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
  });
  device.lost.then((info) => console.error("device lost:", info.message));

  const ctx = canvas.getContext("webgpu")!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  const params: SimParams = {
    mode: "A",
    numParticles: 8000,
    speed: 1.0,
    restitution: 1.0,
    tempGain: 0.012,
    tempDecay: 0.92,
    minSize: 0.004,
    maxSize: 0.011,
    coverage: 0.08,
    cellScale: 1.0,
    paused: false,
    showGrid: false,
  };

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

  // ---- controls pane (top-right), hidden until "/" ----------------------
  const ctrlWrap = document.createElement("div");
  ctrlWrap.style.cssText = "position:fixed;top:10px;right:10px;z-index:20;display:none;width:300px;";
  document.body.appendChild(ctrlWrap);
  const pane = new Pane({ container: ctrlWrap, title: "controls" });

  const modeBinding = pane.addBinding(params, "mode", {
    options: { "A — per particle (linked list)": "A", "B — per bucket (shared mem)": "B" },
  });
  pane.addBinding(params, "numParticles", { min: 100, max: 40000, step: 100 }).on("change", (ev) => {
    if (ev.last) engine.rebuild();
  });
  pane.addBinding(params, "coverage", { min: 0.02, max: 0.3, step: 0.01, label: "density" }).on("change", (ev) => {
    if (ev.last) engine.rebuild();
  });
  pane.addBinding(params, "cellScale", { min: 1, max: 8, step: 0.5, label: "grid cell ×" }).on("change", (ev) => {
    if (ev.last) engine.rebuild();
  });
  pane.addBinding(engine, "viewSize", { min: 0.1, max: 6, step: 0.05, label: "zoom (view)" });
  pane.addBinding(params, "speed", { min: 0, max: 3, step: 0.05 });
  pane.addBinding(params, "restitution", { min: 0, max: 1, step: 0.02 });
  pane.addBinding(params, "tempGain", { min: 0, max: 0.3, step: 0.005 });
  pane.addBinding(params, "tempDecay", { min: 0.8, max: 1, step: 0.005 });
  const fSize = pane.addFolder({ title: "particle size", expanded: false });
  const onSize = (ev: { last: boolean }) => { if (ev.last) engine.rebuild(); };
  fSize.addBinding(params, "minSize", { min: 0.002, max: 0.02, step: 0.001 }).on("change", onSize);
  fSize.addBinding(params, "maxSize", { min: 0.004, max: 0.03, step: 0.001 }).on("change", onSize);
  pane.addBinding(params, "showGrid", { label: "grid overlay" });
  pane.addBinding(params, "paused");

  // ---- metrics pane (top-left) ------------------------------------------
  const m = {
    fps: 0, frameMs: 0, computeMs: 0,
    jsHeapMB: 0, gpuMemMB: 0,
    cells: 0, occupied: 0, occupiedPct: 0, avgPerCell: 0, maxPerCell: 0, overflow: 0,
  };
  const metWrap = document.createElement("div");
  metWrap.style.cssText = "position:fixed;top:10px;left:10px;z-index:20;display:none;width:280px;";
  document.body.appendChild(metWrap);
  const met = new Pane({ container: metWrap, title: "metrics" });

  met.addBinding(m, "fps", { readonly: true, view: "graph", min: 0, max: 165 });
  met.addBinding(m, "frameMs", { readonly: true, view: "graph", min: 0, max: 33, label: "frame ms (cpu)" });
  if (engine.canTimestamp) {
    met.addBinding(m, "computeMs", { readonly: true, view: "graph", min: 0, max: 8, label: "compute ms (gpu)" });
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

  // ---- keys: "/" debug, "M" toggle mode ---------------------------------
  let debug = false;
  window.addEventListener("keydown", (e) => {
    if (e.key === "/") {
      e.preventDefault();
      debug = !debug;
      ctrlWrap.style.display = debug ? "block" : "none";
      metWrap.style.display = debug ? "block" : "none";
      engine.collectStats = debug; // only run stat passes while panel is open
    } else if (e.key === "m" || e.key === "M") {
      params.mode = params.mode === "A" ? "B" : "A";
      modeBinding.refresh();
    }
  });

  // ---- camera: drag to pan, wheel / pinch to zoom -----------------------
  // screen pixel -> world coords using the current camera (optionally a custom zoom)
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
    engine.viewCenterX -= wx1 - wx0; // content follows the cursor
    engine.viewCenterY -= wy1 - wy0;
    lastPx = e.clientX; lastPy = e.clientY;
  });
  const endDrag = (e: PointerEvent) => { dragging = false; canvas.releasePointerCapture?.(e.pointerId); };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    // trackpad pinch arrives as ctrlKey+wheel; normal wheel also zooms
    const factor = Math.exp(e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
    const newVs = Math.min(Math.max(engine.viewSize * factor, 0.05), engine.worldSize * 2);
    // keep the world point under the cursor fixed
    const [wx, wy] = screenToWorld(e.clientX, e.clientY, engine.viewSize);
    const [wx2, wy2] = screenToWorld(e.clientX, e.clientY, newVs);
    engine.viewCenterX += wx - wx2;
    engine.viewCenterY += wy - wy2;
    engine.viewSize = newVs;
  }, { passive: false });

  // ---- main loop --------------------------------------------------------
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
        met.refresh();
      }
      acc = 0; frames = 0;
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

boot();
