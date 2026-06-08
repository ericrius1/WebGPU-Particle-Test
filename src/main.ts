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
    paused: false,
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

  // ---- debug panel (tweakpane), hidden until "/" ------------------------
  const stats = { fps: 0, computeMs: 0 };
  const wrap = document.createElement("div");
  wrap.style.cssText = "position:fixed;top:10px;right:10px;z-index:20;display:none;width:300px;";
  document.body.appendChild(wrap);
  const pane = new Pane({ container: wrap, title: "particles / debug" });

  pane.addBinding(stats, "fps", { readonly: true, view: "graph", min: 0, max: 165 });
  if (engine.canTimestamp) {
    pane.addBinding(stats, "computeMs", { readonly: true, view: "graph", min: 0, max: 8, label: "compute ms" });
  }
  pane.addBinding(params, "mode", {
    options: { "A — per particle (linked list)": "A", "B — per bucket (shared mem)": "B" },
  });
  pane.addBinding(params, "numParticles", { min: 100, max: 40000, step: 100 }).on("change", (ev) => {
    if (ev.last) engine.rebuild();
  });
  pane.addBinding(params, "coverage", { min: 0.02, max: 0.3, step: 0.01, label: "density" }).on("change", (ev) => {
    if (ev.last) engine.rebuild();
  });
  pane.addBinding(engine, "viewSize", { min: 0.3, max: 4, step: 0.05, label: "zoom (view)" });
  pane.addBinding(params, "speed", { min: 0, max: 3, step: 0.05 });
  pane.addBinding(params, "restitution", { min: 0, max: 1, step: 0.02 });
  pane.addBinding(params, "tempGain", { min: 0, max: 0.3, step: 0.005 });
  pane.addBinding(params, "tempDecay", { min: 0.8, max: 1, step: 0.005 });
  const fSize = pane.addFolder({ title: "particle size", expanded: false });
  const onSize = (ev: { last: boolean }) => { if (ev.last) engine.rebuild(); };
  fSize.addBinding(params, "minSize", { min: 0.002, max: 0.02, step: 0.001 }).on("change", onSize);
  fSize.addBinding(params, "maxSize", { min: 0.004, max: 0.03, step: 0.001 }).on("change", onSize);
  pane.addBinding(params, "paused");

  // ---- "/" toggles debug ------------------------------------------------
  let debug = false;
  window.addEventListener("keydown", (e) => {
    if (e.key === "/") {
      e.preventDefault();
      debug = !debug;
      wrap.style.display = debug ? "block" : "none";
    }
  });

  // ---- main loop --------------------------------------------------------
  let last = performance.now();
  let acc = 0, frames = 0;
  function loop(now: number) {
    const dt = (now - last) / 1000;
    last = now;
    engine.frame(dt);

    acc += dt; frames++;
    if (acc >= 0.25) {
      stats.fps = frames / acc;
      stats.computeMs = engine.gpuMs;
      acc = 0; frames = 0;
      pane.refresh();
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

boot();
