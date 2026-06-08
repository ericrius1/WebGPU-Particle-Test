import {
  WG,
  MAX_PER_CELL,
  INTEGRATE_WGSL,
  APPLY_WGSL,
  CLEAR_A_WGSL,
  BIN_A_WGSL,
  COLLIDE_A_WGSL,
  CLEAR_B_WGSL,
  BIN_B_WGSL,
  COLLIDE_B_WGSL,
  RENDER_WGSL,
} from "./shaders";

export type Mode = "A" | "B";

export interface SimParams {
  mode: Mode;
  numParticles: number;
  speed: number; // dt multiplier
  restitution: number;
  tempGain: number;
  tempDecay: number;
  minSize: number;
  maxSize: number;
  coverage: number; // target fraction of the box covered by particles (density)
  paused: boolean;
}

const PARTICLE_FLOATS = 6; // pos.xy, speed.xy, size, temp
const PARTICLE_BYTES = PARTICLE_FLOATS * 4; // 24

export class Engine {
  device: GPUDevice;
  ctx: GPUCanvasContext;
  format: GPUTextureFormat;

  params: SimParams;
  aspect = 1;

  // grid sizing
  gridW = 1;
  gridH = 1;
  numCells = 1;
  cellSize = 0.02;
  worldSize = 1; // side length of the square sim box (grows with particle count)
  viewSize = 0.75; // on-screen window into the world; keeps particles big like the reference

  // buffers
  private constBuf!: GPUBuffer;
  private particleBuf!: GPUBuffer;
  private newSpeedBuf!: GPUBuffer;
  private newTempBuf!: GPUBuffer;
  // mode A
  private gridHeadBuf!: GPUBuffer;
  private gridNextBuf!: GPUBuffer;
  // mode B
  private cellCountBuf!: GPUBuffer;
  private cellPartBuf!: GPUBuffer;
  private occupiedBuf!: GPUBuffer;
  private dispatchBuf!: GPUBuffer;

  // pipelines
  private pIntegrate!: GPUComputePipeline;
  private pApply!: GPUComputePipeline;
  private pClearA!: GPUComputePipeline;
  private pBinA!: GPUComputePipeline;
  private pCollideA!: GPUComputePipeline;
  private pClearB!: GPUComputePipeline;
  private pBinB!: GPUComputePipeline;
  private pCollideB!: GPUComputePipeline;
  private pRender!: GPURenderPipeline;

  // layouts
  private bglSim!: GPUBindGroupLayout;
  private bglA!: GPUBindGroupLayout;
  private bglB!: GPUBindGroupLayout;
  private bglBCollide!: GPUBindGroupLayout;
  private bglRender!: GPUBindGroupLayout;

  // bind groups
  private bgSim!: GPUBindGroup;
  private bgA!: GPUBindGroup;
  private bgB!: GPUBindGroup;
  private bgBCollide!: GPUBindGroup;
  private bgRender!: GPUBindGroup;

  private constArray = new ArrayBuffer(48);
  private constU32 = new Uint32Array(this.constArray);
  private constF32 = new Float32Array(this.constArray);

  // GPU timing (compute ms/frame) via timestamp-query
  canTimestamp = false;
  gpuMs = 0;
  private querySet?: GPUQuerySet;
  private tsResolve?: GPUBuffer;
  private tsResult?: GPUBuffer;
  private tsPending = false;

  constructor(device: GPUDevice, ctx: GPUCanvasContext, format: GPUTextureFormat, params: SimParams) {
    this.device = device;
    this.ctx = ctx;
    this.format = format;
    this.params = params;
    this.canTimestamp = device.features.has("timestamp-query");
    if (this.canTimestamp) {
      this.querySet = device.createQuerySet({ type: "timestamp", count: 2 });
      this.tsResolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.tsResult = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    this.createLayouts();
    this.createPipelines();
    this.rebuild();
  }

  private createLayouts() {
    const d = this.device;
    const buf = (binding: number, type: GPUBufferBindingType, vis = GPUShaderStage.COMPUTE): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: vis,
      buffer: { type },
    });
    this.bglSim = d.createBindGroupLayout({
      entries: [
        buf(0, "uniform"),
        buf(1, "storage"),
        buf(2, "storage"),
        buf(3, "storage"),
      ],
    });
    this.bglA = d.createBindGroupLayout({
      entries: [
        buf(0, "uniform"), buf(1, "storage"), buf(2, "storage"), buf(3, "storage"),
        buf(4, "storage"), buf(5, "storage"),
      ],
    });
    this.bglB = d.createBindGroupLayout({
      entries: [
        buf(0, "uniform"), buf(1, "storage"), buf(2, "storage"), buf(3, "storage"),
        buf(4, "storage"), buf(5, "storage"), buf(6, "storage"), buf(7, "storage"),
      ],
    });
    this.bglBCollide = d.createBindGroupLayout({
      entries: [
        buf(0, "uniform"), buf(1, "storage"), buf(2, "storage"), buf(3, "storage"),
        buf(4, "storage"), buf(5, "storage"), buf(6, "storage"),
      ],
    });
    this.bglRender = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });
  }

  private compute(code: string, layout: GPUBindGroupLayout): GPUComputePipeline {
    const d = this.device;
    return d.createComputePipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: d.createShaderModule({ code }), entryPoint: "main" },
    });
  }

  private createPipelines() {
    const d = this.device;
    this.pIntegrate = this.compute(INTEGRATE_WGSL, this.bglSim);
    this.pApply = this.compute(APPLY_WGSL, this.bglSim);
    this.pClearA = this.compute(CLEAR_A_WGSL, this.bglA);
    this.pBinA = this.compute(BIN_A_WGSL, this.bglA);
    this.pCollideA = this.compute(COLLIDE_A_WGSL, this.bglA);
    this.pClearB = this.compute(CLEAR_B_WGSL, this.bglB);
    this.pBinB = this.compute(BIN_B_WGSL, this.bglB);
    this.pCollideB = this.compute(COLLIDE_B_WGSL, this.bglBCollide);

    const mod = d.createShaderModule({ code: RENDER_WGSL });
    this.pRender = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.bglRender] }),
      vertex: { module: mod, entryPoint: "vs" },
      fragment: {
        module: mod,
        entryPoint: "fs",
        targets: [
          {
            format: this.format,
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  /** (re)allocate buffers + bind groups for current particle count / sizes. */
  rebuild() {
    const d = this.device;
    const N = this.params.numParticles;

    // pick particle sizes first, then size the world so total coverage (density)
    // stays constant as the count changes — sparse like the reference.
    const sizes = new Float32Array(N);
    let area = 0;
    for (let i = 0; i < N; i++) {
      const s = this.params.minSize + Math.random() * (this.params.maxSize - this.params.minSize);
      sizes[i] = s;
      area += Math.PI * s * s;
    }
    this.worldSize = Math.max(Math.sqrt(area / Math.max(this.params.coverage, 0.01)), 4 * this.params.maxSize);
    const L = this.worldSize;

    // grid: cell must fit the largest particle (diameter = 2*maxSize)
    this.cellSize = Math.max(2 * this.params.maxSize, 1e-3);
    this.gridW = Math.max(1, Math.floor(L / this.cellSize));
    this.gridH = this.gridW;
    this.numCells = this.gridW * this.gridH;

    // free previous
    for (const b of [
      this.constBuf, this.particleBuf, this.newSpeedBuf, this.newTempBuf,
      this.gridHeadBuf, this.gridNextBuf, this.cellCountBuf, this.cellPartBuf,
      this.occupiedBuf, this.dispatchBuf,
    ]) b?.destroy?.();

    this.constBuf = d.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const init = new Float32Array(N * PARTICLE_FLOATS);
    for (let i = 0; i < N; i++) {
      const o = i * PARTICLE_FLOATS;
      const size = sizes[i];
      init[o + 0] = size + Math.random() * (L - 2 * size); // pos.x inside walls
      init[o + 1] = size + Math.random() * (L - 2 * size); // pos.y
      const ang = Math.random() * Math.PI * 2;
      const spd = (0.05 + Math.random() * 0.15) * L; // scale with box so motion looks the same
      init[o + 2] = Math.cos(ang) * spd; // speed.x
      init[o + 3] = Math.sin(ang) * spd; // speed.y
      init[o + 4] = size;
      init[o + 5] = 0;
    }
    this.particleBuf = d.createBuffer({
      size: N * PARTICLE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    d.queue.writeBuffer(this.particleBuf, 0, init);

    this.newSpeedBuf = d.createBuffer({ size: N * 8, usage: GPUBufferUsage.STORAGE });
    this.newTempBuf = d.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE });

    // mode A
    this.gridHeadBuf = d.createBuffer({ size: this.numCells * 4, usage: GPUBufferUsage.STORAGE });
    this.gridNextBuf = d.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE });

    // mode B
    this.cellCountBuf = d.createBuffer({ size: this.numCells * 4, usage: GPUBufferUsage.STORAGE });
    this.cellPartBuf = d.createBuffer({ size: this.numCells * MAX_PER_CELL * 4, usage: GPUBufferUsage.STORAGE });
    this.occupiedBuf = d.createBuffer({ size: this.numCells * 4, usage: GPUBufferUsage.STORAGE });
    this.dispatchBuf = d.createBuffer({
      size: 12,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });

    // bind groups
    const e = (binding: number, buffer: GPUBuffer) => ({ binding, resource: { buffer } });
    this.bgSim = d.createBindGroup({
      layout: this.bglSim,
      entries: [e(0, this.constBuf), e(1, this.particleBuf), e(2, this.newSpeedBuf), e(3, this.newTempBuf)],
    });
    this.bgA = d.createBindGroup({
      layout: this.bglA,
      entries: [
        e(0, this.constBuf), e(1, this.particleBuf), e(2, this.newSpeedBuf), e(3, this.newTempBuf),
        e(4, this.gridHeadBuf), e(5, this.gridNextBuf),
      ],
    });
    this.bgB = d.createBindGroup({
      layout: this.bglB,
      entries: [
        e(0, this.constBuf), e(1, this.particleBuf), e(2, this.newSpeedBuf), e(3, this.newTempBuf),
        e(4, this.cellCountBuf), e(5, this.cellPartBuf), e(6, this.occupiedBuf), e(7, this.dispatchBuf),
      ],
    });
    this.bgBCollide = d.createBindGroup({
      layout: this.bglBCollide,
      entries: [
        e(0, this.constBuf), e(1, this.particleBuf), e(2, this.newSpeedBuf), e(3, this.newTempBuf),
        e(4, this.cellCountBuf), e(5, this.cellPartBuf), e(6, this.occupiedBuf),
      ],
    });
    this.bgRender = d.createBindGroup({
      layout: this.bglRender,
      entries: [e(0, this.constBuf), e(1, this.particleBuf)],
    });
  }

  setAspect(a: number) { this.aspect = a; }

  async debugRead(n = 8): Promise<number[][]> {
    const d = this.device;
    const bytes = n * PARTICLE_BYTES;
    const stg = d.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.particleBuf, 0, stg, 0, bytes);
    d.queue.submit([enc.finish()]);
    await stg.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(stg.getMappedRange().slice(0));
    stg.unmap(); stg.destroy();
    const out: number[][] = [];
    for (let i = 0; i < n; i++) out.push(Array.from(f.subarray(i * PARTICLE_FLOATS, i * PARTICLE_FLOATS + PARTICLE_FLOATS)));
    return out;
  }

  private writeConstants(dt: number) {
    const u = this.constU32, f = this.constF32, p = this.params;
    u[0] = p.numParticles;
    u[1] = this.gridW;
    u[2] = this.gridH;
    u[3] = MAX_PER_CELL;
    f[4] = this.cellSize;
    f[5] = dt;
    f[6] = this.aspect;
    f[7] = p.tempDecay;
    f[8] = p.tempGain;
    f[9] = p.restitution;
    f[10] = this.worldSize;
    f[11] = Math.min(this.viewSize, this.worldSize);
    this.device.queue.writeBuffer(this.constBuf, 0, this.constArray);
  }

  frame(dtSeconds: number) {
    const d = this.device;
    const p = this.params;
    const dt = p.paused ? 0 : Math.min(dtSeconds, 1 / 30) * p.speed;
    this.writeConstants(dt);

    const enc = d.createCommandEncoder();
    const ts = this.canTimestamp && !this.tsPending;

    if (!p.paused) {
      const cellGroups = Math.ceil(this.numCells / WG);
      const partGroups = Math.ceil(p.numParticles / WG);

      if (p.mode === "A") {
        const cp = enc.beginComputePass(
          ts ? { timestampWrites: { querySet: this.querySet!, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined,
        );
        cp.setBindGroup(0, this.bgA);
        cp.setPipeline(this.pClearA); cp.dispatchWorkgroups(cellGroups);
        cp.setBindGroup(0, this.bgSim);
        cp.setPipeline(this.pIntegrate); cp.dispatchWorkgroups(partGroups);
        cp.setBindGroup(0, this.bgA);
        cp.setPipeline(this.pBinA); cp.dispatchWorkgroups(partGroups);
        cp.setPipeline(this.pCollideA); cp.dispatchWorkgroups(partGroups);
        cp.setBindGroup(0, this.bgSim);
        cp.setPipeline(this.pApply); cp.dispatchWorkgroups(partGroups);
        cp.end();
      } else {
        // pass 1: clear + integrate + bin (writes dispatchArgs as storage)
        const cp = enc.beginComputePass(
          ts ? { timestampWrites: { querySet: this.querySet!, beginningOfPassWriteIndex: 0 } } : undefined,
        );
        cp.setBindGroup(0, this.bgB);
        cp.setPipeline(this.pClearB); cp.dispatchWorkgroups(Math.max(cellGroups, 1));
        cp.setBindGroup(0, this.bgSim);
        cp.setPipeline(this.pIntegrate); cp.dispatchWorkgroups(partGroups);
        cp.setBindGroup(0, this.bgB);
        cp.setPipeline(this.pBinB); cp.dispatchWorkgroups(partGroups);
        cp.end();
        // pass 2: collide (dispatchArgs used only as indirect) + apply
        const cp2 = enc.beginComputePass(
          ts ? { timestampWrites: { querySet: this.querySet!, endOfPassWriteIndex: 1 } } : undefined,
        );
        cp2.setBindGroup(0, this.bgBCollide);
        cp2.setPipeline(this.pCollideB); cp2.dispatchWorkgroupsIndirect(this.dispatchBuf, 0);
        cp2.setBindGroup(0, this.bgSim);
        cp2.setPipeline(this.pApply); cp2.dispatchWorkgroups(partGroups);
        cp2.end();
      }

      if (ts) {
        enc.resolveQuerySet(this.querySet!, 0, 2, this.tsResolve!, 0);
        enc.copyBufferToBuffer(this.tsResolve!, 0, this.tsResult!, 0, 16);
      }
    }

    const view = this.ctx.getCurrentTexture().createView();
    const rp = enc.beginRenderPass({
      colorAttachments: [
        { view, clearValue: { r: 0.62, g: 0.62, b: 0.62, a: 1 }, loadOp: "clear", storeOp: "store" },
      ],
    });
    rp.setPipeline(this.pRender);
    rp.setBindGroup(0, this.bgRender);
    rp.draw(6, p.numParticles);
    rp.end();

    d.queue.submit([enc.finish()]);

    if (ts && !p.paused) {
      this.tsPending = true;
      const rb = this.tsResult!;
      rb.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigInt64Array(rb.getMappedRange().slice(0));
        const ns = Number(t[1] - t[0]);
        if (ns > 0) this.gpuMs = this.gpuMs * 0.85 + (ns / 1e6) * 0.15; // smooth
        rb.unmap();
        this.tsPending = false;
      });
    }
  }
}
