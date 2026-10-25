// render/device.ts
async function initDevice() {
  const gpu = navigator.gpu;
  if (!gpu) throw new Error("WebGPU not available (need Chrome/Edge/Safari or Deno --unstable-webgpu)");
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const want = ["float32-filterable", "timestamp-query", "shader-f16"].filter((f) => adapter.features.has(f));
  const lim = adapter.limits;
  const requiredLimits = {};
  const raise = (k) => {
    const v = lim[k];
    if (typeof v === "number") requiredLimits[k] = v;
  };
  raise("maxBufferSize");
  raise("maxStorageBufferBindingSize");
  raise("maxTextureDimension3D");
  const device = await adapter.requestDevice({ requiredFeatures: want, requiredLimits });
  return { adapter, device, features: new Set(want) };
}

// render/mat4.ts
function identity() {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}
function multiply(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  }
  return o;
}
function perspectiveZO(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[11] = -1;
  m[10] = far / (near - far);
  m[14] = far * near / (near - far);
  return m;
}
function perspectiveZOTile(fovy, viewW, viewH, x, y2, w, h, near, far) {
  const t = near * Math.tan(fovy / 2), b = -t;
  const r = t * (viewW / viewH), l = -r;
  const l2 = l + (r - l) * x / viewW, r2 = l + (r - l) * (x + w) / viewW;
  const t2 = t - (t - b) * y2 / viewH, b2 = t - (t - b) * (y2 + h) / viewH;
  const m = new Float32Array(16);
  m[0] = 2 * near / (r2 - l2);
  m[5] = 2 * near / (t2 - b2);
  m[8] = (r2 + l2) / (r2 - l2);
  m[9] = (t2 + b2) / (t2 - b2);
  m[10] = far / (near - far);
  m[11] = -1;
  m[14] = far * near / (near - far);
  return m;
}
function lookAt(eye, center, up) {
  let zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
  let zl = Math.hypot(zx, zy, zz) || 1;
  zx /= zl;
  zy /= zl;
  zz /= zl;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  let xl = Math.hypot(xx, xy, xz) || 1;
  xx /= xl;
  xy /= xl;
  xz /= xl;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  const m = new Float32Array(16);
  m[0] = xx;
  m[4] = xy;
  m[8] = xz;
  m[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  m[1] = yx;
  m[5] = yy;
  m[9] = yz;
  m[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  m[2] = zx;
  m[6] = zy;
  m[10] = zz;
  m[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  m[15] = 1;
  return m;
}
function invert(a) {
  const m = a;
  const b00 = m[0] * m[5] - m[1] * m[4], b01 = m[0] * m[6] - m[2] * m[4];
  const b02 = m[0] * m[7] - m[3] * m[4], b03 = m[1] * m[6] - m[2] * m[5];
  const b04 = m[1] * m[7] - m[3] * m[5], b05 = m[2] * m[7] - m[3] * m[6];
  const b06 = m[8] * m[13] - m[9] * m[12], b07 = m[8] * m[14] - m[10] * m[12];
  const b08 = m[8] * m[15] - m[11] * m[12], b09 = m[9] * m[14] - m[10] * m[13];
  const b10 = m[9] * m[15] - m[11] * m[13], b11 = m[10] * m[15] - m[11] * m[14];
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return identity();
  det = 1 / det;
  const o = new Float32Array(16);
  o[0] = (m[5] * b11 - m[6] * b10 + m[7] * b09) * det;
  o[1] = (m[2] * b10 - m[1] * b11 - m[3] * b09) * det;
  o[2] = (m[13] * b05 - m[14] * b04 + m[15] * b03) * det;
  o[3] = (m[10] * b04 - m[9] * b05 - m[11] * b03) * det;
  o[4] = (m[6] * b08 - m[4] * b11 - m[7] * b07) * det;
  o[5] = (m[0] * b11 - m[2] * b08 + m[3] * b07) * det;
  o[6] = (m[14] * b02 - m[12] * b05 - m[15] * b01) * det;
  o[7] = (m[8] * b05 - m[10] * b02 + m[11] * b01) * det;
  o[8] = (m[4] * b10 - m[5] * b08 + m[7] * b06) * det;
  o[9] = (m[1] * b08 - m[0] * b10 - m[3] * b06) * det;
  o[10] = (m[12] * b04 - m[13] * b02 + m[15] * b00) * det;
  o[11] = (m[9] * b02 - m[8] * b04 - m[11] * b00) * det;
  o[12] = (m[5] * b07 - m[4] * b09 - m[6] * b06) * det;
  o[13] = (m[0] * b09 - m[1] * b07 + m[2] * b06) * det;
  o[14] = (m[13] * b01 - m[12] * b03 - m[14] * b00) * det;
  o[15] = (m[8] * b03 - m[9] * b01 + m[10] * b00) * det;
  return o;
}
function patientToTexture(dims, spacing, center = [0, 0, 0]) {
  const m = new Float32Array(16);
  for (let a = 0; a < 3; a++) {
    const s = 1 / (spacing[a] * dims[a]);
    m[a * 4 + a] = s;
    m[12 + a] = 0.5 - center[a] * s;
  }
  m[15] = 1;
  return m;
}
function volumeAABB(dims, spacing, center = [0, 0, 0]) {
  const ext = [dims[0] * spacing[0] / 2, dims[1] * spacing[1] / 2, dims[2] * spacing[2] / 2];
  return [
    [center[0] - ext[0], center[1] - ext[1], center[2] - ext[2]],
    [center[0] + ext[0], center[1] + ext[1], center[2] + ext[2]]
  ];
}
function transpose4(m) {
  const o = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) o[c * 4 + r] = m[r * 4 + c];
  return o;
}
function patientToTextureFromIjkToRAS(ijkToRAS, dims) {
  return invert(texToRASFromIjkToRAS(ijkToRAS, dims));
}
function texToRASFromIjkToRAS(ijkToRAS, dims) {
  const M = transpose4(ijkToRAS);
  const A = new Float32Array(16);
  for (let a = 0; a < 3; a++) {
    A[a * 4 + a] = dims[a];
    A[12 + a] = -0.5;
  }
  A[15] = 1;
  return multiply(M, A);
}
function volumeAABBFromIjkToRAS(ijkToRAS, dims) {
  const t2r = texToRASFromIjkToRAS(ijkToRAS, dims);
  const lo2 = [Infinity, Infinity, Infinity], hi2 = [-Infinity, -Infinity, -Infinity];
  for (let c = 0; c < 8; c++) {
    const u = c & 1, v = c >> 1 & 1, w = c >> 2 & 1;
    for (let r = 0; r < 3; r++) {
      const p = t2r[r] * u + t2r[4 + r] * v + t2r[8 + r] * w + t2r[12 + r];
      if (p < lo2[r]) lo2[r] = p;
      if (p > hi2[r]) hi2[r] = p;
    }
  }
  return [lo2, hi2];
}
function applyMat4(m, p) {
  const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
  const y2 = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
  const z2 = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15] || 1;
  return [x / w, y2 / w, z2 / w];
}
function spacingFromIjkToRAS(ijkToRAS) {
  const col = (c) => Math.hypot(ijkToRAS[c], ijkToRAS[4 + c], ijkToRAS[8 + c]);
  return [col(0), col(1), col(2)];
}

// render/scene-renderer.ts
var DEFAULT_FORMAT = "rgba8unorm-srgb";
var SCENE_FLOATS = 16;
var CLIP_FLOATS = 36;
var MESH_WGSL = (
  /* wgsl */
  `
struct MU { view_proj : mat4x4<f32>, eye : vec4<f32>, color : vec4<f32> };
@group(0) @binding(0) var<uniform> mu : MU;
struct VO { @builtin(position) pos : vec4<f32>, @location(0) wp : vec3<f32> };
@vertex fn vs_mesh(@location(0) p : vec3<f32>) -> VO { var o : VO; o.pos = mu.view_proj * vec4<f32>(p, 1.0); o.wp = p; return o; }
struct FO { @location(0) col : vec4<f32>, @location(1) depth : vec4<f32> };
@fragment fn fs_mesh(i : VO) -> FO {
  let n = normalize(cross(dpdx(i.wp), dpdy(i.wp)));       // flat face normal (no normals on the wire)
  let l = normalize(mu.eye.xyz - i.wp);                   // headlight
  let lam = 0.25 + 0.75 * abs(dot(n, l));
  let a = mu.color.a;
  var o : FO;
  o.col = vec4<f32>(mu.color.rgb * lam * a, a);           // premultiplied
  o.depth = vec4<f32>(distance(mu.eye.xyz, i.wp), 0.0, 0.0, 1.0);
  return o;
}`
);
var SceneRenderer = class _SceneRenderer {
  // ── surface meshes (models): rasterised before each trace into colour+depth targets the march composites ──
  meshPipeline;
  gpuMeshes = [];
  meshTargetsBySize = /* @__PURE__ */ new Map();
  viewProj = new Float32Array(16);
  eyePos = [0, 0, 0];
  /** Replace the surface meshes (world/RAS float32 xyz + uint32 triangles, colour, opacity). */
  setMeshes(meshes) {
    for (const m of this.gpuMeshes) {
      m.vbuf.destroy();
      m.ibuf.destroy();
      m.ubuf.destroy();
    }
    this.gpuMeshes = meshes.filter((m) => m.indices.length >= 3).map((m) => {
      const vbuf = this.dev.createBuffer({ size: Math.ceil(m.positions.byteLength / 4) * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      this.dev.queue.writeBuffer(vbuf, 0, m.positions);
      const ibuf = this.dev.createBuffer({ size: Math.ceil(m.indices.byteLength / 4) * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
      this.dev.queue.writeBuffer(ibuf, 0, m.indices);
      const ubuf = this.dev.createBuffer({ size: 24 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      return { vbuf, ibuf, count: m.indices.length, ubuf, color: m.color, opacity: m.opacity };
    });
  }
  hasMeshes() {
    return this.gpuMeshes.length > 0;
  }
  ensureMeshPipeline() {
    if (this.meshPipeline) return;
    const mod = this.dev.createShaderModule({ code: MESH_WGSL });
    this.meshPipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: mod, entryPoint: "vs_mesh", buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }] },
      fragment: { module: mod, entryPoint: "fs_mesh", targets: [{ format: "rgba16float" }, { format: "r32float" }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }
    });
  }
  /** Colour/depth targets (+ the group-1 bind group of the trace pipeline) for a given trace size. */
  meshTargets(w, h) {
    const key = w + "x" + h;
    let t = this.meshTargetsBySize.get(key);
    if (!t) {
      if (this.meshTargetsBySize.size > 4) {
        for (const old of this.meshTargetsBySize.values()) {
          old.col.destroy();
          old.depth.destroy();
          old.z.destroy();
        }
        this.meshTargetsBySize.clear();
      }
      t = {
        w,
        h,
        col: this.dev.createTexture({ size: [w, h], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
        depth: this.dev.createTexture({ size: [w, h], format: "r32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
        z: this.dev.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT })
      };
      this.meshTargetsBySize.set(key, t);
    }
    if (!t.bind) t.bind = this.dev.createBindGroup({ layout: this.pipeline.getBindGroupLayout(1), entries: [{ binding: 0, resource: t.col.createView() }, { binding: 1, resource: t.depth.createView() }] });
    return t;
  }
  /** Rasterise the meshes for this frame's trace size; returns the bind group the trace pass needs. */
  meshPass(enc, w, h) {
    const t = this.meshTargets(w, h);
    const pass = enc.beginRenderPass({
      colorAttachments: [
        { view: t.col.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } },
        { view: t.depth.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 1e30, g: 0, b: 0, a: 1 } }
      ],
      depthStencilAttachment: { view: t.z.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" }
    });
    if (this.gpuMeshes.length) {
      this.ensureMeshPipeline();
      pass.setPipeline(this.meshPipeline);
      for (const m of this.gpuMeshes) {
        const u = new Float32Array(24);
        u.set(this.viewProj, 0);
        u[16] = this.eyePos[0];
        u[17] = this.eyePos[1];
        u[18] = this.eyePos[2];
        u[19] = 1;
        u[20] = m.color[0];
        u[21] = m.color[1];
        u[22] = m.color[2];
        u[23] = m.opacity;
        this.dev.queue.writeBuffer(m.ubuf, 0, u);
        pass.setBindGroup(0, this.dev.createBindGroup({ layout: this.meshPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: m.ubuf } }] }));
        pass.setVertexBuffer(0, m.vbuf);
        pass.setIndexBuffer(m.ibuf, "uint32");
        pass.drawIndexed(m.count);
      }
    }
    pass.end();
    return t.bind;
  }
  dev;
  format;
  placed = [];
  pipeline;
  sampler;
  camBuf;
  matBuf;
  mat;
  bind;
  // PICK pass: a 1x1 ray-trace that reuses the field compositing to find the RAS point where
  // front-to-back opacity first crosses 50% (Slicer's 3D volume pick). Ghost handles excluded.
  pickPipeline;
  pickBind;
  pickOff = 0;
  // mat[] offset of the pick_cursor uniform (NDC)
  pickTarget;
  // 1x1 rgba32float (wp.xyz, hit)
  pickReadBuf;
  // PRODUCER→RECONSTRUCTOR seam (docs/UNIFIED-RENDERING-PLAN.md M1). The ray-march writes the
  // premultiplied composited sample into `traceTex` (rgba32float, lossless); `resolvePipeline`
  // composites it over the background into the output view. 1:1 for now (byte-identical); the
  // resolve pass is where spatial upsample + temporal accumulation (time-averaged AA) will live.
  resolvePipeline;
  resolveBind;
  resolveBgBuf;
  traceTex;
  traceView;
  traceW = 0;
  traceH = 0;
  // TEMPORAL ACCUMULATION (M2a, docs/UNIFIED-RENDERING-PLAN.md §3). When the view is still, each
  // frame jitters the CAMERA sub-pixel (Halton, via a clip-space translation of invVP — the shader
  // is untouched, so a non-jittered frame is byte-identical) and the Reconstructor folds it into a
  // running mean, converging to a supersampled, time-averaged-AA image. Ping-pong accum + running n.
  baseInvVP = new Float32Array(16);
  // last setCamera invVP (unjittered)
  focalPx = 1;
  // last setCamera focal (view→pixels); used to keep screen-space handles view-sized under low-res trace
  accumPipeline;
  // MRT: trace + prev-accum -> new-accum + presented view
  accumBind = [void 0, void 0];
  accumUniformBuf;
  // (bg.rgb, blend)
  accumTex = [void 0, void 0];
  accumView = [void 0, void 0];
  accumPing = 0;
  accumN = 0;
  lastAccumCam = new Float32Array(16);
  // camera (invVP) of the last accumulated frame
  lastAccumValid = false;
  // false forces a reset (after a rebuild / first frame)
  streamPipeline;
  // trace -> rgba8unorm, for compact sample readback (remote)
  streamBind;
  // its OWN bind group (auto-layout differs from this.pipeline's)
  // RESOLUTION-SCALED reconstruction (M2b): while interacting, trace at a fraction of the view
  // (BudgetController) and Catmull-Rom UPSAMPLE the low-res trace to the view — the client-superres
  // ported from the Python spike. A settled view renders native + accumulates instead.
  superresPipeline;
  superresBind;
  superresBuf;
  // (traceW, traceH, viewW, viewH)
  // The moving/upscale path traces into its OWN low-res target so it never resizes/destroys the
  // full-size traceTex the accumulation bind groups reference (that sharing caused destroyed-texture
  // submits + MRT attachment-size mismatches → 3D flicker/blank during interaction).
  lowTex;
  lowView;
  lowW = 0;
  lowH = 0;
  accumW = 0;
  accumH = 0;
  /** Emit a default AABB-distance skip for fields that don't supply their own bound.
   *
   *  OFF because it MEASURED AS A NET LOSS (render/test/profile-boxskip.ts, 448², M-series):
   *      MultiVolume +8.7%   Volume+Fiducials +7.3%   Segmentation +96.5%   SingleVolume -15.5%
   *  The appealing theory — "Panoramix sits +200mm R of CTACardio, so rays spend much of the
   *  scene box outside one volume" — is true but worthless: ImageField's out-of-box sample was
   *  ALREADY nearly free (it early-returns on the texture-bounds test), so there was no per-step
   *  cost to remove. Meanwhile every field pays a box distance + horizon bookkeeping at every
   *  step it is INSIDE its box, which is most of the march since the scene box is the union of
   *  the field boxes. Fields with their own cheap early-out are hurt worst — SegmentField
   *  (`v<=0.02||v>=0.98`) nearly doubles. The lone SingleVolume win survives warm-up but has no
   *  algorithmic explanation (the box IS the scene box there, so the bound is 0 at every sample)
   *  and is almost certainly a shader-compiler/occupancy artifact — not something to bank on.
   *
   *  Kept behind a flag rather than deleted so the negative result stays reproducible, and
   *  because it may behave differently on other GPUs (NVIDIA/AMD) — re-measure before enabling.
   *  The real win for dense volumes is an occupancy grid over air INSIDE the box, not the box. */
  static boxSkip = false;
  canTime;
  clipOff = 0;
  constructor(gpu, format = DEFAULT_FORMAT) {
    this.dev = gpu.device;
    this.format = format;
    this.canTime = gpu.features.has("timestamp-query");
    this.sampler = this.dev.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    this.camBuf = this.dev.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.resolveBgBuf = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const rmod = this.dev.createShaderModule({ code: this.resolveWgsl() });
    this.resolvePipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: rmod, entryPoint: "vs_resolve" },
      fragment: { module: rmod, entryPoint: "fs_resolve", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.accumUniformBuf = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const amod = this.dev.createShaderModule({ code: this.accumWgsl() });
    this.accumPipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: amod, entryPoint: "vs_resolve" },
      fragment: { module: amod, entryPoint: "fs_accum", targets: [{ format: "rgba32float" }, { format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.superresBuf = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const smod = this.dev.createShaderModule({ code: this.superresWgsl() });
    this.superresPipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: smod, entryPoint: "vs_resolve" },
      fragment: { module: smod, entryPoint: "fs_superres", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
  }
  /** RECONSTRUCTOR (upsampling): Catmull-Rom (bicubic, 9 bilinear taps) reconstruction of the
   *  low-res premultiplied trace, composited over the background — the client-superres from the
   *  Python spike (435b28d), on WebGPU. Slight edge sharpening from the negative lobes; premultiplied
   *  so the alpha reconstructs correctly. Used only when the trace is smaller than the view. */
  superresWgsl() {
    return (
      /* wgsl */
      `
@group(0) @binding(0) var t_trace : texture_2d<f32>;
@group(0) @binding(1) var s_lin : sampler;
@group(0) @binding(2) var<uniform> u_sr : vec4<f32>;   // (traceW, traceH, viewW, viewH)
@group(0) @binding(3) var<uniform> u_bg : vec4<f32>;
fn srgb2physical(c : vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92;
  let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(lo, hi, c > vec3<f32>(0.04045));
}
// Catmull-Rom via 9 bilinear taps (Sigg/Hadwiger form).
fn cr(uv : vec2<f32>, texSize : vec2<f32>) -> vec4<f32> {
  let sp = uv * texSize;
  let tp1 = floor(sp - 0.5) + 0.5;
  let f = sp - tp1;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let off12 = w2 / w12;
  let inv = 1.0 / texSize;
  let p0 = (tp1 - 1.0) * inv;
  let p3 = (tp1 + 2.0) * inv;
  let p12 = (tp1 + off12) * inv;
  var r = vec4<f32>(0.0);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p0.x,  p0.y),  0.0) * (w0.x  * w0.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p12.x, p0.y),  0.0) * (w12.x * w0.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p3.x,  p0.y),  0.0) * (w3.x  * w0.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p0.x,  p12.y), 0.0) * (w0.x  * w12.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p12.x, p12.y), 0.0) * (w12.x * w12.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p3.x,  p12.y), 0.0) * (w3.x  * w12.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p0.x,  p3.y),  0.0) * (w0.x  * w3.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p12.x, p3.y),  0.0) * (w12.x * w3.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p3.x,  p3.y),  0.0) * (w3.x  * w3.y);
  return r;
}
struct RV { @builtin(position) position : vec4<f32> };
@vertex
fn vs_resolve(@builtin(vertex_index) vi : u32) -> RV {
  let x = select(-1.0, 3.0, vi == 1u);
  let y = select(-1.0, 3.0, vi == 2u);
  var o : RV; o.position = vec4<f32>(x, y, 0.0, 1.0); return o;
}
@fragment
fn fs_superres(v : RV) -> @location(0) vec4<f32> {
  let uv = v.position.xy / u_sr.zw;
  let s = cr(uv, u_sr.xy);
  let a = clamp(s.a, 0.0, 1.0);
  let bg = srgb2physical(u_bg.rgb);
  return vec4<f32>(mix(bg, s.rgb, a), 1.0);
}`
    );
  }
  /** Accumulating RECONSTRUCTOR: fold this frame's traced sample into the running mean (blend =
   *  1/n; blend=1 on reset → mean=this frame) and present it over the background. MRT so one pass
   *  updates the accumulation texture AND the swap-chain view. Frame N jitters the ray sub-pixel,
   *  so the mean over N frames is a supersampled, time-averaged-AA image (still camera). */
  accumWgsl() {
    return (
      /* wgsl */
      `
@group(0) @binding(0) var t_trace : texture_2d<f32>;
@group(0) @binding(1) var t_accum : texture_2d<f32>;
@group(0) @binding(2) var<uniform> u_ra : vec4<f32>;   // (bg.r, bg.g, bg.b, blend)
fn srgb2physical(c : vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92;
  let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(lo, hi, c > vec3<f32>(0.04045));
}
struct RV { @builtin(position) position : vec4<f32> };
@vertex
fn vs_resolve(@builtin(vertex_index) vi : u32) -> RV {
  let x = select(-1.0, 3.0, vi == 1u);
  let y = select(-1.0, 3.0, vi == 2u);
  var o : RV; o.position = vec4<f32>(x, y, 0.0, 1.0); return o;
}
struct FO { @location(0) accum : vec4<f32>, @location(1) present : vec4<f32> };
@fragment
fn fs_accum(v : RV) -> FO {
  let p = vec2<i32>(v.position.xy);
  let cur = textureLoad(t_trace, p, 0);
  let prev = textureLoad(t_accum, p, 0);
  let acc = mix(prev, cur, u_ra.w);        // blend=1 on reset -> acc = cur
  let bg = srgb2physical(u_ra.rgb);
  var o : FO;
  o.accum = acc;
  o.present = vec4<f32>(mix(bg, acc.rgb, acc.a), 1.0);
  return o;
}`
    );
  }
  /** RECONSTRUCTOR (M1: identity resolve). Composites the traced premultiplied sample over the
   *  background — the exact `mix(bg, rgb, a)` the fused fs_main used. `textureLoad` at integer
   *  coords is a 1:1 fetch (no filtering), so the output is byte-identical to the fused path.
   *  M2 replaces this with a spatial-upsample + temporal-accumulate resolve. */
  resolveWgsl() {
    return (
      /* wgsl */
      `
@group(0) @binding(0) var t_trace : texture_2d<f32>;
@group(0) @binding(1) var<uniform> u_bg : vec4<f32>;
fn srgb2physical(c : vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92;
  let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(lo, hi, c > vec3<f32>(0.04045));
}
struct RV { @builtin(position) position : vec4<f32> };
@vertex
fn vs_resolve(@builtin(vertex_index) vi : u32) -> RV {
  let x = select(-1.0, 3.0, vi == 1u);
  let y = select(-1.0, 3.0, vi == 2u);
  var o : RV; o.position = vec4<f32>(x, y, 0.0, 1.0); return o;
}
@fragment
fn fs_resolve(v : RV) -> @location(0) vec4<f32> {
  let s = textureLoad(t_trace, vec2<i32>(v.position.xy), 0);
  let bg = srgb2physical(u_bg.rgb);
  return vec4<f32>(mix(bg, s.rgb, s.a), 1.0);
}`
    );
  }
  /** (Re)allocate the trace target + resolve bind group when the view size changes. */
  ensureTrace(width, height) {
    if (this.traceTex && this.traceW === width && this.traceH === height) return;
    this.traceTex?.destroy();
    this.traceTex = this.dev.createTexture({
      size: [width, height],
      format: "rgba32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    });
    this.traceView = this.traceTex.createView();
    this.traceW = width;
    this.traceH = height;
    this.resolveBind = this.dev.createBindGroup({
      layout: this.resolvePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: this.traceView }, { binding: 1, resource: { buffer: this.resolveBgBuf } }]
    });
  }
  /** (Re)allocate the low-res trace target + superres bind group when the moving render size changes.
   *  Separate from traceTex so a moving frame never disturbs the accumulation textures. */
  ensureLow(width, height) {
    if (this.lowTex && this.lowW === width && this.lowH === height) return;
    this.lowTex?.destroy();
    this.lowTex = this.dev.createTexture({ size: [width, height], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.lowView = this.lowTex.createView();
    this.lowW = width;
    this.lowH = height;
    this.superresBind = this.dev.createBindGroup({
      layout: this.superresPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.lowView },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.superresBuf } },
        { binding: 3, resource: { buffer: this.resolveBgBuf } }
      ]
    });
  }
  /** Adaptive (moving-frame) render: trace at `renderW×renderH` and Catmull-Rom upsample to the
   *  `viewW×viewH` output. The caller MUST have set the camera size to renderW×renderH (so the
   *  low-res rays fill the same frustum). Single frame, no accumulation — use while interacting;
   *  switch to renderAccum when the view settles. */
  renderUpscaled(view, renderW, renderH, viewW, viewH) {
    this.ensureLow(renderW, renderH);
    this.flush();
    this.dev.queue.writeBuffer(this.camBuf, 72, new Float32Array([this.focalPx * (viewH / renderH)]));
    this.dev.queue.writeBuffer(this.superresBuf, 0, new Float32Array([renderW, renderH, viewW, viewH]));
    this.dev.queue.writeBuffer(this.resolveBgBuf, 0, this.mat.subarray(12, 16));
    const enc = this.dev.createCommandEncoder();
    const mb = this.meshPass(enc, renderW, renderH);
    const tp = enc.beginRenderPass({ colorAttachments: [{ view: this.lowView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.pipeline);
    tp.setBindGroup(0, this.bind);
    tp.setBindGroup(1, mb);
    tp.draw(3);
    tp.end();
    const sp = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    sp.setPipeline(this.superresPipeline);
    sp.setBindGroup(0, this.superresBind);
    sp.draw(3);
    sp.end();
    this.dev.queue.submit([enc.finish()]);
  }
  /** Encode trace (producer) + resolve (reconstructor) into `enc`, output to `outView`. */
  encodeFrame(enc, outView) {
    const mb = this.meshPass(enc, this.traceW, this.traceH);
    const tp = enc.beginRenderPass({ colorAttachments: [{ view: this.traceView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.pipeline);
    tp.setBindGroup(0, this.bind);
    tp.setBindGroup(1, mb);
    tp.draw(3);
    tp.end();
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: outView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    rp.setPipeline(this.resolvePipeline);
    rp.setBindGroup(0, this.resolveBind);
    rp.draw(3);
    rp.end();
  }
  /** (Re)allocate the ping-pong accumulation targets + their bind groups on a size change. Tracks its
   *  OWN size and always rebuilds accumBind against the current traceView (which ensureTrace, called
   *  first in renderAccum, has just refreshed) — so the bind never dangles on a destroyed trace. */
  ensureAccum(width, height) {
    if (this.accumTex[0] && this.accumW === width && this.accumH === height) return;
    this.accumW = width;
    this.accumH = height;
    for (let k = 0; k < 2; k++) {
      this.accumTex[k]?.destroy();
      this.accumTex[k] = this.dev.createTexture({ size: [width, height], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      this.accumView[k] = this.accumTex[k].createView();
    }
    for (let k = 0; k < 2; k++) {
      this.accumBind[k] = this.dev.createBindGroup({
        layout: this.accumPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.traceView },
          { binding: 1, resource: this.accumView[k] },
          { binding: 2, resource: { buffer: this.accumUniformBuf } }
        ]
      });
    }
    this.accumN = 0;
    this.accumPing = 0;
  }
  /** Reset temporal accumulation — call when the view changes (camera move, scene edit, resize). */
  resetAccumulation() {
    this.accumN = 0;
  }
  /** Frames accumulated since the last reset (0 before the first accumulated frame). */
  accumCount() {
    return this.accumN;
  }
  /** Accumulating render: trace this frame (sub-pixel jittered) and fold it into the running mean,
   *  presenting the mean over the background. `reset` (or a view change) restarts the mean at this
   *  frame (n=1, no jitter — byte-identical to renderToView). Call repeatedly while the view is
   *  still to converge to a supersampled, time-averaged-AA image. */
  renderAccum(view, width, height, reset) {
    this.ensureTrace(width, height);
    this.ensureAccum(width, height);
    let camChanged = !this.lastAccumValid;
    const cam = this.baseInvVP;
    for (let i = 0; i < 16 && !camChanged; i++) if (cam[i] !== this.lastAccumCam[i]) camChanged = true;
    if (camChanged) reset = true;
    this.lastAccumCam.set(cam);
    this.lastAccumValid = true;
    if (reset) this.accumN = 0;
    this.accumN += 1;
    const n = this.accumN;
    if (n > 1) {
      const jx = _SceneRenderer.halton(n, 2) - 0.5, jy = _SceneRenderer.halton(n, 3) - 0.5;
      const T = new Float32Array(16);
      T[0] = T[5] = T[10] = T[15] = 1;
      T[12] = 2 * jx / width;
      T[13] = -2 * jy / height;
      this.dev.queue.writeBuffer(this.camBuf, 0, multiply(this.baseInvVP, T));
    } else {
      this.dev.queue.writeBuffer(this.camBuf, 0, this.baseInvVP);
    }
    this.dev.queue.writeBuffer(this.camBuf, 76, new Float32Array([n - 1]));
    this.flush();
    this.dev.queue.writeBuffer(this.accumUniformBuf, 0, new Float32Array([this.mat[12], this.mat[13], this.mat[14], 1 / n]));
    const prev = this.accumPing, next = 1 - this.accumPing;
    const enc = this.dev.createCommandEncoder();
    const mb = this.meshPass(enc, width, height);
    const tp = enc.beginRenderPass({ colorAttachments: [{ view: this.traceView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.pipeline);
    tp.setBindGroup(0, this.bind);
    tp.setBindGroup(1, mb);
    tp.draw(3);
    tp.end();
    const ap = enc.beginRenderPass({ colorAttachments: [
      { view: this.accumView[next], loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } },
      { view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }
    ] });
    ap.setPipeline(this.accumPipeline);
    ap.setBindGroup(0, this.accumBind[prev]);
    ap.draw(3);
    ap.end();
    this.dev.queue.submit([enc.finish()]);
    this.accumPing = next;
  }
  /** (Re)build the pipeline for a set of fields. */
  build(fields) {
    const kindCount = {};
    let uoff = SCENE_FLOATS, bbase = 3;
    this.placed = fields.map((field) => {
      const slot = kindCount[field.kind] ?? 0;
      kindCount[field.kind] = slot + 1;
      const p = { field, slot, uoff, bbase };
      uoff += field.uniformFloats();
      bbase += field.bindingCount;
      return p;
    });
    this.clipOff = uoff;
    this.pickOff = uoff + CLIP_FLOATS;
    this.mat = new Float32Array(uoff + CLIP_FLOATS + 12);
    this.matBuf = this.dev.createBuffer({ size: (uoff + CLIP_FLOATS + 12) * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    for (const t of this.meshTargetsBySize.values()) t.bind = void 0;
    const module = this.dev.createShaderModule({ code: this.wgsl() });
    this.pipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_trace", targets: [{ format: "rgba32float" }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.pickPipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_pick", targets: [{ format: "rgba32float" }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.streamPipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_trace", targets: [{ format: "rgba8unorm" }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.bind = this.dev.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: this.bindGroupEntries() });
    this.streamBind = this.dev.createBindGroup({ layout: this.streamPipeline.getBindGroupLayout(0), entries: this.bindGroupEntries() });
    if (this.pickPipeline) this.pickBind = this.dev.createBindGroup({ layout: this.pickPipeline.getBindGroupLayout(0), entries: this.bindGroupEntries() });
    this.setBackground(0.07, 0.08, 0.12);
    const step = this.placed.length ? Math.min(...this.placed.map((p) => p.field.sampleStep())) : 1;
    this.setSampleStep(step * 0.7);
    this.recomputeBounds();
    for (const p of this.placed) p.field.fillUniforms(this.mat, p.uoff);
    this.accumN = 0;
    this.lastAccumValid = false;
  }
  wgsl() {
    const members = this.placed.map((p) => p.field.structMembers(p.slot)).join("\n");
    const decls = this.placed.map((p) => p.field.declareBindings(p.slot, p.bbase)).join("\n");
    const modifiers = this.placed.filter((p) => p.field.modifier);
    const receivers = this.placed.filter((p) => !p.field.modifier);
    const modFns = modifiers.map((p) => p.field.samplingWGSL(p.slot)).join("\n");
    const slotOf = new Map(this.placed.map((p) => [p.field, p.slot]));
    const tpFns = receivers.map((p) => {
      const tf = p.field.transform;
      const tfSlot = tf && tf.modifier ? slotOf.get(tf) : void 0;
      const body = tfSlot === void 0 ? "  return wp;" : `  return wp + displacement_grid${tfSlot}(wp);`;
      return `fn transform_point_${p.field.kind}${p.slot}(wp : vec3<f32>) -> vec3<f32> {
${body}
}`;
    }).join("\n");
    const fieldFns = receivers.map((p) => p.field.samplingWGSL(p.slot)).join("\n");
    const wf = (v) => (Number.isFinite(v) ? v : 0).toFixed(6);
    const boxSkipWGSL = (p) => {
      const [lo2, hi2] = p.field.aabb();
      return `
fn skip_${p.field.kind}${p.slot}(wp : vec3<f32>) -> f32 {
  let q = max(vec3<f32>(${wf(lo2[0])}, ${wf(lo2[1])}, ${wf(lo2[2])}) - wp,
              wp - vec3<f32>(${wf(hi2[0])}, ${wf(hi2[1])}, ${wf(hi2[2])}));
  return length(max(q, vec3<f32>(0.0)));   // 0 inside the box, exact distance outside
}`;
    };
    const ghostFields = receivers.filter((p) => p.field.ghost);
    const normalReceivers = receivers.filter((p) => !p.field.ghost);
    const clipGuard = (p, expr) => p.field.clippable === false ? expr : `if (!clipped) { ${expr} }`;
    const sampleInto = (nm, ghost) => ghost ? `let c = sample_field_${nm}(wp, rd); if (c.a > g_op) { g_op = c.a; g_col = c.rgb / max(c.a, 1e-4); }` : `let c = sample_field_${nm}(wp, rd); sum += c;`;
    const skipBranch = (p, clip, ghost = false) => {
      const nm = `${p.field.kind}${p.slot}`;
      const smp = sampleInto(nm, ghost);
      return `    if (t >= resume_${nm}) {
      let d_${nm} = max(skip_${nm}(wp) - step, 0.0);
      if (d_${nm} > 0.0) { resume_${nm} = t + d_${nm}; }
      else { ${clip ? clipGuard(p, smp) : smp} }
    }
    if (t < resume_${nm}) { jump_t = min(jump_t, resume_${nm}); } else { all_defer = false; }`;
    };
    const plainBranch = (p, clip, ghost = false) => {
      const nm = `${p.field.kind}${p.slot}`;
      const smp = sampleInto(nm, ghost);
      return `    { ${clip ? clipGuard(p, smp) : smp} all_defer = false; }`;
    };
    const normalSkippers = normalReceivers.filter((p) => !p.field.transform).filter((p) => _SceneRenderer.boxSkip || p.field.providesSkip && p.field.skipWGSL);
    const ghostSkippers = ghostFields.filter((p) => p.field.providesSkip && p.field.skipWGSL);
    const canSkip = new Set(normalSkippers.map((p) => p.field));
    const ghostCanSkip = new Set(ghostSkippers.map((p) => p.field));
    const skipFns = [
      ...normalSkippers.map((p) => p.field.providesSkip && p.field.skipWGSL ? p.field.skipWGSL(p.slot) : boxSkipWGSL(p)),
      ...ghostSkippers.map((p) => p.field.skipWGSL(p.slot))
    ].join("\n");
    const fns = [modFns, tpFns, fieldFns, skipFns].filter((s) => s.trim()).join("\n");
    const skipInit = [...normalSkippers, ...ghostSkippers].map((p) => `  var resume_${p.field.kind}${p.slot} : f32 = -1.0e30;`).join("\n");
    const dispatch = normalReceivers.map(
      (p) => canSkip.has(p.field) ? skipBranch(p, true) : plainBranch(p, true)
    ).join("\n");
    const ghostDispatch = ghostFields.map(
      (p) => ghostCanSkip.has(p.field) ? skipBranch(p, false, true) : plainBranch(p, false, true)
    ).join("\n");
    const hasGhost = ghostFields.length > 0;
    const pickDispatch = normalReceivers.map(
      (p) => `    ${clipGuard(p, `{ let c = sample_field_${p.field.kind}${p.slot}(wp, rd); sum += c; }`)}`
    ).join("\n");
    return (
      /* wgsl */
      `
struct Camera { inv_view_proj : mat4x4<f32>, size : vec4<f32>, eye : vec4<f32> };
struct Material {
  bmin : vec4<f32>,
  bmax : vec4<f32>,
  scene : vec4<f32>,   // sample_step, _, _, _
  bg : vec4<f32>,
${members}
  clip_planes : array<vec4<f32>, 8>,   // (nx, ny, nz, offset) inward; tail so field offsets are stable
  clip_count : vec4<f32>,              // (count, _, _, _)
  pick_cursor : vec4<f32>,             // (ndc_x, ndc_y, _, _) \u2014 the ray for fs_pick
  probe_origin : vec4<f32>,            // explicit-ray probe: world origin
  probe_dir : vec4<f32>,               // (dx, dy, dz, enabled) \u2014 w>0 uses this ray instead of the cursor
};
@group(0) @binding(0) var<uniform> u_cam : Camera;
@group(0) @binding(1) var<uniform> u_material : Material;
// Rasterised surface meshes (models): nearest-surface colour (premultiplied) + its distance along the
// ray, produced by the mesh pass before each trace. The march composites the surface at that depth,
// so volumes in front occlude it and it occludes what is behind \u2014 the depth-composite seam.
@group(1) @binding(0) var t_mesh_col : texture_2d<f32>;
@group(1) @binding(1) var t_mesh_depth : texture_2d<f32>;
${this.usesSampler() ? "@group(0) @binding(2) var s_lin : sampler;" : ""}
${decls}

struct Varyings { @builtin(position) position : vec4<f32> };
@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> Varyings {
  let x = select(-1.0, 3.0, vi == 1u);
  let y = select(-1.0, 3.0, vi == 2u);
  var o : Varyings; o.position = vec4<f32>(x, y, 0.0, 1.0); return o;
}
fn srgb2physical(c : vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92;
  let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(lo, hi, c > vec3<f32>(0.04045));
}
fn ndc_to_world(ndc : vec4<f32>) -> vec3<f32> { let w = u_cam.inv_view_proj * ndc; return w.xyz / w.w; }
fn ign(p : vec2<f32>) -> f32 { return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715)))); }
${fns}

// PRODUCER (fs_trace): march the ray and return the composited PREMULTIPLIED sample
// (integrated.rgb, integrated.a) BEFORE the background composite \u2014 a "traced pixel". The
// Reconstructor (fs_resolve / reconstructor.ts) composites it over the background. Splitting
// trace from assemble is the seam the unified local/remote pipeline turns on (see
// docs/UNIFIED-RENDERING-PLAN.md); the background composite is identical to the fused path, so
// output is byte-identical at full density. An empty slab returns transparent (0) \u2192 resolve = bg.
@fragment
fn fs_trace(v : Varyings) -> @location(0) vec4<f32> {
  let size = u_cam.size.xy;
  let ndc_x = (v.position.x / size.x) * 2.0 - 1.0;
  let ndc_y = 1.0 - (v.position.y / size.y) * 2.0;
  let ro = ndc_to_world(vec4<f32>(ndc_x, ndc_y, 0.0, 1.0));
  let rd = normalize(ndc_to_world(vec4<f32>(ndc_x, ndc_y, 1.0, 1.0)) - ro);

  let mpix = vec2<i32>(v.position.xy);
  let mesh_c = textureLoad(t_mesh_col, mpix, 0);          // premultiplied surface colour (0 = no mesh)
  let mesh_t = textureLoad(t_mesh_depth, mpix, 0).r;      // distance along the ray (1e30 = none)
  var mesh_done = mesh_c.a <= 0.0;

  let inv = vec3<f32>(1.0) / rd;
  let tb = (u_material.bmin.xyz - ro) * inv;
  let tt = (u_material.bmax.xyz - ro) * inv;
  let tmn = min(tt, tb); let tmx = max(tt, tb);
  var t_near = max(max(tmn.x, tmn.y), tmn.z);
  var t_far  = min(min(tmx.x, tmx.y), tmx.z);
  if (t_far <= t_near || t_far <= 0.0) { return mesh_c; }

  let step = max(u_material.scene.x, 1e-3);
  t_near = max(t_near + step, 0.0);
  t_far  = t_far - step;
  if (t_far <= t_near) { return mesh_c; }
  let seed = ign(v.position.xy);
  var t = t_near;
  var integrated = vec4<f32>(0.0);
  var safety : i32 = 0;
  var saturated = false;   // LATCH: once opaque, normal fields stay off even after a ghost
                           // handle dims the accumulation (else the volume behind the handle
                           // would re-opaque over it and re-bury the shine-through).
  var g_op = 0.0;          // ghost (handle) surface: max opacity along the ray (0.5 inactive /
  var g_col = vec3<f32>(0.0);  // 1.0 active) and its colour \u2014 tracked, never accumulated.
${skipInit}
  loop {
    if (t >= t_far || safety >= 5000${hasGhost ? "" : " || integrated.a >= 0.99"}) { break; }
    // Per-(pixel, step, ACCUM FRAME) ray-offset jitter. The frame term (u_cam.size.w, the
    // accumulation index) is what makes temporal AA actually converge: with a frame-invariant
    // offset the jitter turns banding into FIXED-PATTERN noise that averaging can never remove
    // (measured: 32 samples was as grainy as 1). Varying it per frame decorrelates the samples
    // so the mean approaches the true integral \u2014 no banding AND no noise. size.w is 0 for every
    // non-accumulating path, so frame 1 stays byte-identical to a plain renderToView.
    // Base offset: decorrelated per (pixel, step) so a single frame shows noise, not banding.
    let jbase = fract(sin(dot(v.position.xy + vec2<f32>(f32(safety) * 0.7548, f32(safety) * 0.5698), vec2<f32>(12.9898, 78.233))) * 43758.5453);
    // Advance it across accumulation frames by the golden-ratio additive recurrence
    // (Cranley-Patterson rotation). MEASURED: this converges at the same 1/sqrt(n) rate as an
    // independent random offset per frame (high-freq energy 1.36 vs 1.31 at n=64) \u2014 the low-
    // discrepancy walk is NOT faster here, because the variance is dominated by the step size
    // against a sharp transfer function, not by the sequence. Kept because it is deterministic
    // and costs nothing; reduce sampleStep if you need less residual speckle.
    // At size.w = 0 this is exactly jbase, so the first accumulated frame stays byte-identical
    // to a plain renderToView \u2014 the property render/test baselines depend on.
    let js = fract(jbase + u_cam.size.w * 0.6180339887) - 0.5;
    if (!mesh_done && t + 0.5 * step >= mesh_t) {         // the ray reaches the surface: composite it here
      integrated = integrated + (1.0 - integrated.a) * mesh_c;
      mesh_done = true;
    }
    let wp = ro + rd * (t + js * step);
    var sum = vec4<f32>(0.0);
    var all_defer = true;        // every field guarantees emptiness here -> we may leap
    var jump_t = 1.0e30;         // nearest field horizon
    var clipped = false;         // ROI clip: sample on the negative side of any active plane
    let ccount = u32(u_material.clip_count.x);
    for (var ci = 0u; ci < ccount; ci = ci + 1u) {
      let cp = u_material.clip_planes[ci];
      if (dot(wp, cp.xyz) + cp.w < 0.0) { clipped = true; break; }
    }
    // Normal fields stop being sampled once the ray is opaque (latched); GHOST fields keep
    // their skip horizons and keep going, so a handle behind an opaque region still shines
    // through and the ray LEAPS between handles on the ghost skip (early-termination kept).
${hasGhost ? "    if (integrated.a >= 0.99) { saturated = true; }\n    if (!saturated) {" : ""}
${dispatch}
      if (sum.a > 0.0) { integrated = integrated + (1.0 - integrated.a) * vec4<f32>(sum.rgb, clamp(sum.a, 0.0, 1.0)); }
${hasGhost ? "    }" : ""}
${ghostDispatch}
    if (all_defer && jump_t > t + step) { t = jump_t; } else { t = t + step; }
    safety = safety + 1;
  }
  if (!mesh_done) { integrated = integrated + (1.0 - integrated.a) * mesh_c; }   // surface beyond the slab
  // GHOST x-ray, applied ONCE (never compounding): the volume IN FRONT of a handle is shown
  // at residual = 1 - handle_opacity (50% for an inactive handle at opacity 0.5, 0% for an
  // active/hovered handle at opacity 1.0), then the handle (colour g_col at opacity g_op)
  // draws over it.
  if (g_op > 0.001) {
    let ga = clamp(g_op, 0.0, 1.0);
    let residual = 1.0 - ga;
    let fA = integrated.a * residual;
    integrated = vec4<f32>(integrated.rgb * residual + (1.0 - fA) * g_col * ga, fA + (1.0 - fA) * ga);
  }
  return integrated;   // premultiplied (rgb, a); resolve composites over the background
}

// PICK: trace the cursor ray (pick_cursor NDC) through the SAME field compositing and return the
// world (RAS) position where front-to-back opacity first crosses 50% \u2014 Slicer's 3D volume pick.
// Output: (wp.x, wp.y, wp.z, hit). hit=0 means the ray never reached 50% (empty/miss).
@fragment
fn fs_pick() -> @location(0) vec4<f32> {
  // Two ray sources: the screen cursor (pick) or an explicit world ray (probe). The explicit
  // form exists because the cursor ray can only ever probe what is ON SCREEN \u2014 useless for
  // "how much room is BEHIND me?", which endovascular navigation needs for reverse and for
  // lateral clearance.
  var ro = ndc_to_world(vec4<f32>(u_material.pick_cursor.x, u_material.pick_cursor.y, 0.0, 1.0));
  var rd = normalize(ndc_to_world(vec4<f32>(u_material.pick_cursor.x, u_material.pick_cursor.y, 1.0, 1.0)) - ro);
  if (u_material.probe_dir.w > 0.5) {
    ro = u_material.probe_origin.xyz;
    rd = normalize(u_material.probe_dir.xyz);
  }
  let inv = vec3<f32>(1.0) / rd;
  let tb = (u_material.bmin.xyz - ro) * inv;
  let tt = (u_material.bmax.xyz - ro) * inv;
  let tmn = min(tt, tb); let tmx = max(tt, tb);
  var t_near = max(max(tmn.x, tmn.y), tmn.z);
  var t_far  = min(min(tmx.x, tmx.y), tmx.z);
  if (t_far <= t_near || t_far <= 0.0) { return vec4<f32>(0.0); }
  let step = max(u_material.scene.x, 1e-3);
  t_near = max(t_near + step, 0.0);
  t_far  = t_far - step;
  var t = t_near;
  var acc = 0.0;
  var safety : i32 = 0;
  loop {
    if (t >= t_far || safety >= 5000 || acc >= 0.5) { break; }
    let wp = ro + rd * t;
    var clipped = false;
    let ccount = u32(u_material.clip_count.x);
    for (var ci = 0u; ci < ccount; ci = ci + 1u) {
      let cp = u_material.clip_planes[ci];
      if (dot(wp, cp.xyz) + cp.w < 0.0) { clipped = true; break; }
    }
    var sum = vec4<f32>(0.0);
${pickDispatch}
    if (sum.a > 0.0) {
      let a_new = acc + (1.0 - acc) * clamp(sum.a, 0.0, 1.0);
      if (a_new >= 0.5) { return vec4<f32>(wp, 1.0); }   // 50% crossing -> the pick point
      acc = a_new;
    }
    t = t + step;
  }
  return vec4<f32>(0.0);
}`
    );
  }
  setBackground(r, g, b) {
    this.mat[12] = r;
    this.mat[13] = g;
    this.mat[14] = b;
    this.mat[15] = 1;
  }
  setSampleStep(step) {
    this.mat[8] = step;
  }
  /** Van der Corput / Halton radical inverse in `base`. */
  static halton(i, base) {
    let f = 1, r = 0;
    while (i > 0) {
      f /= base;
      r += f * (i % base);
      i = Math.floor(i / base);
    }
    return r;
  }
  /** Set up to 8 clip planes (nx,ny,nz,offset), inward-normal, keep-side `dot(wp,n)+offset>=0`.
   *  Written into the uniform tail — a Tier-A update the next flush() uploads; no rebuild. */
  setClipPlanes(planes) {
    const n = Math.min(planes.length, 8);
    for (let i = 0; i < n; i++) this.mat.set(planes[i], this.clipOff + i * 4);
    this.mat[this.clipOff + 32] = n;
  }
  clearClip() {
    this.mat[this.clipOff + 32] = 0;
  }
  /** Axis-aligned RAS crop box [lo,hi] → 6 inward planes. offset = -dot(faceOrigin, n). */
  setClipBox(lo2, hi2) {
    this.setClipPlanes([
      [1, 0, 0, -lo2[0]],
      [-1, 0, 0, hi2[0]],
      // keep lo.x <= x <= hi.x
      [0, 1, 0, -lo2[1]],
      [0, -1, 0, hi2[1]],
      [0, 0, 1, -lo2[2]],
      [0, 0, -1, hi2[2]]
    ]);
  }
  /** Scene AABB = union of field AABBs; also picks a default sample step from the smallest field extent. */
  recomputeBounds() {
    if (!this.placed.length) return;
    let mn2 = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (const p of this.placed) {
      const [a, b] = p.field.aabb();
      for (let i = 0; i < 3; i++) {
        mn2[i] = Math.min(mn2[i], a[i]);
        mx[i] = Math.max(mx[i], b[i]);
      }
    }
    this.mat[0] = mn2[0];
    this.mat[1] = mn2[1];
    this.mat[2] = mn2[2];
    this.mat[4] = mx[0];
    this.mat[5] = mx[1];
    this.mat[6] = mx[2];
  }
  /** Tier-A interactive update: re-pack every field's uniform block into the resident
   *  material buffer WITHOUT recompiling the pipeline or rebuilding the bind group. This is
   *  the render-side of the interaction architecture (ARCHITECTURE-2026-07-24 §7): a
   *  lightweight drag — clip planes, ROI box geometry, fiducial position, TPS displacement
   *  grid — mutates node state, the field re-derives its uniforms, and the SAME per-frame
   *  flush() the renderer already does uploads them. Cost is a CPU re-pack; no shader build.
   *
   *  Also refreshes the scene AABB (which is uniform-resident), so a moved field's ray-clip
   *  bounds stay correct. REQUIRES the field SET and each field's uniformFloats() to be
   *  unchanged since build() — geometry/appearance may change, STRUCTURE may not. A structural
   *  change (add/remove a field, a field that resizes its uniform block, or a texture swap
   *  needing refreshBindings) still goes through build()/refreshBindings(). This is exactly
   *  why moving geometry must be uniform-resident, never baked into generated WGSL — see the
   *  box-skip note above and RENDER-PERFORMANCE.md. */
  syncUniforms() {
    for (const p of this.placed) p.field.fillUniforms(this.mat, p.uoff);
    this.recomputeBounds();
  }
  /** Rebuild the bind group from the fields' current resources (e.g. after a field
   *  swapped a texture) without recompiling the pipeline. Field set/structure must be unchanged. */
  refreshBindings() {
    this.bind = this.dev.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: this.bindGroupEntries() });
    this.streamBind = this.dev.createBindGroup({ layout: this.streamPipeline.getBindGroupLayout(0), entries: this.bindGroupEntries() });
    if (this.pickPipeline) this.pickBind = this.dev.createBindGroup({ layout: this.pickPipeline.getBindGroupLayout(0), entries: this.bindGroupEntries() });
  }
  /** Only fields with texture bindings use the shared sampler. `layout: "auto"` derives the
   *  layout from what the shader ACTUALLY references, so in a scene of purely procedural
   *  fields (e.g. fiducials/markups only) binding 2 is absent from the layout — supplying it
   *  anyway fails validation and the whole view silently renders nothing. Emit the sampler
   *  declaration and its bind entry under the SAME condition so the two can't drift. */
  usesSampler() {
    return this.placed.some((p) => p.field.bindingCount > 0);
  }
  bindGroupEntries() {
    const entries = [
      { binding: 0, resource: { buffer: this.camBuf } },
      { binding: 1, resource: { buffer: this.matBuf } }
    ];
    if (this.usesSampler()) entries.push({ binding: 2, resource: this.sampler });
    for (const p of this.placed) entries.push(...p.field.bindEntries(p.slot, p.bbase));
    return entries;
  }
  setCamera(eye, center, up, fovyDeg, width, height) {
    const view = lookAt(eye, center, up);
    const proj = perspectiveZO(fovyDeg * Math.PI / 180, width / height, 1, 1e5);
    const invVP = invert(multiply(proj, view));
    this.baseInvVP = invVP;
    this.viewProj = multiply(proj, view);
    this.eyePos = eye;
    const cam = new Float32Array(24);
    cam.set(invVP, 0);
    this.focalPx = height / 2 / Math.tan(fovyDeg * Math.PI / 360);
    cam[16] = width;
    cam[17] = height;
    cam[18] = height / 2 / Math.tan(fovyDeg * Math.PI / 360);
    cam[19] = 0;
    cam[20] = eye[0];
    cam[21] = eye[1];
    cam[22] = eye[2];
    this.dev.queue.writeBuffer(this.camBuf, 0, cam);
  }
  /** Camera for ONE TILE of the view: the same rays the full frame would cast for `rect`, into a
   *  rect.w×rect.h target. Screen-space glyph sizing stays keyed to the FULL view height, so a
   *  patch of the gizmo is drawn at exactly the size the full frame drew it. Pair with
   *  traceSamples(rect.w, rect.h) — its focal rewrite is then a no-op. */
  setCameraTile(eye, center, up, fovyDeg, viewW, viewH, rect) {
    const view = lookAt(eye, center, up);
    const proj = perspectiveZOTile(fovyDeg * Math.PI / 180, viewW, viewH, rect.x, rect.y, rect.w, rect.h, 1, 1e5);
    const invVP = invert(multiply(proj, view));
    this.baseInvVP = invVP;
    this.viewProj = multiply(proj, view);
    this.eyePos = eye;
    const cam = new Float32Array(24);
    cam.set(invVP, 0);
    this.focalPx = viewH / 2 / Math.tan(fovyDeg * Math.PI / 360);
    cam[16] = rect.w;
    cam[17] = rect.h;
    cam[18] = this.focalPx;
    cam[19] = 0;
    cam[20] = eye[0];
    cam[21] = eye[1];
    cam[22] = eye[2];
    this.dev.queue.writeBuffer(this.camBuf, 0, cam);
  }
  flush() {
    this.dev.queue.writeBuffer(this.matBuf, 0, this.mat);
  }
  /** Ray-trace the cursor (u,v in [0,1], y down) through the composited fields and return the
   *  RAS point where front-to-back opacity first reaches 50% — Slicer's 3D volume pick. Traces
   *  whatever renders (DVR volumes, SegmentField iso shells, RGBA), EXCLUDING ghost handles.
   *  Uses the camera set by the last setCamera(); returns null if the ray never reaches 50%. */
  async pick(u, v) {
    if (!this.pickPipeline || !this.pickBind || !this.placed.length) return null;
    return this.serialise(async () => {
      this.mat[this.pickOff] = u * 2 - 1;
      this.mat[this.pickOff + 1] = 1 - v * 2;
      this.mat[this.pickOff + 11] = 0;
      this.flush();
      return await this.tracePick();
    });
  }
  /** Trace an EXPLICIT world ray and return the distance (mm) to the first point where
   *  front-to-back opacity reaches 50%, or Infinity if it never does. Unlike pick(), the ray
   *  is independent of the camera, so it can look backwards and sideways — which is what makes
   *  collision "rails" possible in a first-person flythrough. */
  async probe(origin, dir) {
    if (!this.pickPipeline || !this.pickBind || !this.placed.length) return Infinity;
    const l = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    return this.serialise(async () => {
      this.mat[this.pickOff + 4] = origin[0];
      this.mat[this.pickOff + 5] = origin[1];
      this.mat[this.pickOff + 6] = origin[2];
      this.mat[this.pickOff + 8] = dir[0] / l;
      this.mat[this.pickOff + 9] = dir[1] / l;
      this.mat[this.pickOff + 10] = dir[2] / l;
      this.mat[this.pickOff + 11] = 1;
      this.flush();
      const hit = await this.tracePick();
      this.mat[this.pickOff + 11] = 0;
      this.flush();
      if (!hit) return Infinity;
      return Math.hypot(hit[0] - origin[0], hit[1] - origin[1], hit[2] - origin[2]);
    });
  }
  /** Serialises pick/probe. They share ONE uniform buffer and ONE readback buffer, so
   *  concurrent calls would overwrite each other's ray and double-map the buffer — a
   *  Promise.all of probes silently returns garbage. Callers may fire as many as they like;
   *  they queue here. */
  pickChain = Promise.resolve();
  serialise(fn2) {
    const next = this.pickChain.then(fn2, fn2);
    this.pickChain = next.catch(() => {
    });
    return next;
  }
  /** The shared 1x1 render + readback behind pick() and probe(). */
  async tracePick() {
    if (!this.pickTarget) {
      this.pickTarget = this.dev.createTexture({ size: [1, 1], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      this.pickReadBuf = this.dev.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    const enc = this.dev.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.pickTarget.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    pass.setPipeline(this.pickPipeline);
    pass.setBindGroup(0, this.pickBind);
    pass.draw(3);
    pass.end();
    enc.copyTextureToBuffer({ texture: this.pickTarget }, { buffer: this.pickReadBuf, bytesPerRow: 256, rowsPerImage: 1 }, [1, 1]);
    this.dev.queue.submit([enc.finish()]);
    await this.pickReadBuf.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(this.pickReadBuf.getMappedRange().slice(0, 16));
    this.pickReadBuf.unmap();
    return r[3] > 0.5 ? [r[0], r[1], r[2]] : null;
  }
  renderToView(view, width, height) {
    this.ensureTrace(width, height);
    this.flush();
    this.dev.queue.writeBuffer(this.resolveBgBuf, 0, this.mat.subarray(12, 16));
    const enc = this.dev.createCommandEncoder();
    this.encodeFrame(enc, view);
    this.dev.queue.submit([enc.finish()]);
  }
  /** Exact GPU time of the ray-march pass (median ms over `iters`), via timestamp-query.
   *  Times ONLY the render pass — no texture copy/readback — so it reflects shader cost.
   *  Returns NaN if the device lacks timestamp-query. Deno gives full-resolution timestamps;
   *  Chrome quantizes them unless cross-origin isolated, so profile headless for sharp numbers. */
  async timePass(width, height, iters = 40) {
    if (!this.canTime) return NaN;
    this.flush();
    const target = this.dev.createTexture({ size: [width, height], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const view = target.createView();
    const qs2 = this.dev.createQuerySet({ type: "timestamp", count: 2 });
    const resolve = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const samples = [];
    for (let i = 0; i < iters; i++) {
      const enc = this.dev.createCommandEncoder();
      const mb = this.meshPass(enc, width, height);
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
        timestampWrites: { querySet: qs2, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bind);
      pass.setBindGroup(1, mb);
      pass.draw(3);
      pass.end();
      enc.resolveQuerySet(qs2, 0, 2, resolve, 0);
      enc.copyBufferToBuffer(resolve, 0, read, 0, 16);
      this.dev.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(read.getMappedRange());
      const ms2 = Number(t[1] - t[0]) / 1e6;
      read.unmap();
      if (ms2 > 0 && Number.isFinite(ms2)) samples.push(ms2);
    }
    target.destroy();
    qs2.destroy();
    resolve.destroy();
    read.destroy();
    if (!samples.length) return NaN;
    samples.sort((a, b) => a - b);
    return samples[samples.length >> 1];
  }
  async renderToRGBA(width, height) {
    this.ensureTrace(width, height);
    this.flush();
    this.dev.queue.writeBuffer(this.resolveBgBuf, 0, this.mat.subarray(12, 16));
    const target = this.dev.createTexture({ size: [width, height], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const enc = this.dev.createCommandEncoder();
    this.encodeFrame(enc, target.createView());
    const bpr = Math.ceil(width * 4 / 256) * 256;
    const buf = this.dev.createBuffer({ size: bpr * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyTextureToBuffer({ texture: target }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: height }, [width, height]);
    this.dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(width * height * 4);
    for (let y2 = 0; y2 < height; y2++) out.set(padded.subarray(y2 * bpr, y2 * bpr + width * 4), y2 * width * 4);
    buf.unmap();
    target.destroy();
    buf.destroy();
    return out;
  }
  /** REMOTE PRODUCER (M3): trace at width×height and read back the PREMULTIPLIED sample (pre-
   *  background) as tightly-packed rgba8 — the bytes streamed to the remote client, which runs the
   *  same reconstruction (upsample + background composite) the local resolve does. The caller sets
   *  the camera to width×height first (like renderUpscaled). Returns width*height*4 bytes. */
  async traceSamples(width, height, viewH = height) {
    this.flush();
    this.dev.queue.writeBuffer(this.camBuf, 64, new Float32Array([width, height]));
    this.dev.queue.writeBuffer(this.camBuf, 72, new Float32Array([this.focalPx * (viewH / height)]));
    const target = this.dev.createTexture({ size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const enc = this.dev.createCommandEncoder();
    this.meshPass(enc, width, height);
    const smt = this.meshTargets(width, height);
    const streamMb = this.dev.createBindGroup({ layout: this.streamPipeline.getBindGroupLayout(1), entries: [{ binding: 0, resource: smt.col.createView() }, { binding: 1, resource: smt.depth.createView() }] });
    const tp = enc.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.streamPipeline);
    tp.setBindGroup(0, this.streamBind);
    tp.setBindGroup(1, streamMb);
    tp.draw(3);
    tp.end();
    const bpr = Math.ceil(width * 4 / 256) * 256;
    const buf = this.dev.createBuffer({ size: bpr * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyTextureToBuffer({ texture: target }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: height }, [width, height]);
    this.dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(width * height * 4);
    for (let y2 = 0; y2 < height; y2++) out.set(padded.subarray(y2 * bpr, y2 * bpr + width * 4), y2 * width * 4);
    buf.unmap();
    target.destroy();
    buf.destroy();
    return out;
  }
};

// render/slice-renderer.ts
var DEFAULT_FORMAT2 = "rgba8unorm-srgb";
var SHADER = (
  /* wgsl */
  `
struct U {
  p2t : mat4x4<f32>,     // RAS -> texture[0,1] (folds in ijkToRAS: rotation + anisotropy)
  origin : vec4<f32>,    // RAS of the plane center (for the current scrub offset)
  uvec : vec4<f32>,      // RAS vector spanning the view width  (isotropic mm)
  vvec : vec4<f32>,      // RAS vector spanning the view height (isotropic mm)
  params : vec4<f32>,    // win, lev, fillOpacity, outlineOpacity
  size : vec4<f32>,      // sizeX, sizeY, labelOverlayMode, bgLutMode (0 gray, 1 LUT row 0)
  // \u2500\u2500 Slicer slice-composite layers (vtkMRMLSliceCompositeNode): a FOREGROUND volume blended over the
  //    background with its own geometry, W/L and LUT, and a LABEL volume coloured through a colour table.
  p2tFg : mat4x4<f32>,   // RAS -> foreground texture[0,1]
  fgParams : vec4<f32>,  // win, lev, opacity (0 = no foreground), compositing (0 alpha,1 reverse alpha,2 add,3 subtract)
  p2tLabel : mat4x4<f32>,// RAS -> label texture[0,1]
  labelParams : vec4<f32>, // opacity (0 = no label layer), lutEntries, fgLutMode (0 gray, 1 LUT row 1), _
};
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var s_lin : sampler;
@group(0) @binding(2) var t_scalar : texture_3d<f32>;
@group(0) @binding(3) var t_overlay : texture_3d<f32>;
@group(0) @binding(4) var s_nn : sampler;   // NEAREST \u2014 labelmap overlay is per-voxel crisp (matches Slicer)
// Label-overlay mode (size.z > 0.5): instead of a pre-coloured rgba volume, take the segment
// number from a u8 label volume and its colour+opacity from the same 256x2 palette the
// ColorizeField uses. A coloured overlay of a 509x365x299 CT would be 222 MB; label + palette
// is 55 MB and, because it shares the palette, hiding an organ group in 3D hides it here too.
@group(0) @binding(5) var t_labels : texture_3d<u32>;
@group(0) @binding(6) var t_palette : texture_2d<f32>;
@group(0) @binding(7) var t_fg : texture_3d<f32>;       // foreground scalar volume
@group(0) @binding(8) var t_lut : texture_2d<f32>;      // 256x2 colour LUTs: row 0 background, row 1 foreground (sampled over the W/L ramp)
@group(0) @binding(9) var t_labelVol : texture_3d<f32>; // label volume (integer values stored as float)
@group(0) @binding(10) var t_labelLut : texture_2d<f32>;// Nx1 colour table indexed by label value

struct V { @builtin(position) position : vec4<f32> };
@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> V {
  let x = select(-1.0, 3.0, vi == 1u);
  let y = select(-1.0, 3.0, vi == 2u);
  var o : V; o.position = vec4<f32>(x, y, 0.0, 1.0); return o;
}
fn srgb2physical(c : vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92; let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(lo, hi, c > vec3<f32>(0.04045));
}
/** The overlay colour at a texture coordinate, from whichever source is configured. */
fn ov_tex(t : vec3<f32>) -> vec4<f32> {
  if (u.size.z > 0.5) {
    let d = vec3<f32>(textureDimensions(t_labels));
    let vi = vec3<i32>(clamp(floor(t * d), vec3<f32>(0.0), d - vec3<f32>(1.0)));
    let lab = i32(textureLoad(t_labels, vi, 0).r);
    if (lab == 0) { return vec4<f32>(0.0); }
    return textureLoad(t_palette, vec2<i32>(lab, 1), 0);
  }
  return textureSampleLevel(t_overlay, s_nn, t, 0.0);
}
fn ov_at(ras : vec3<f32>) -> vec4<f32> {   // overlay at a RAS point (0 outside the volume)
  let t = (u.p2t * vec4<f32>(ras, 1.0)).xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return vec4<f32>(0.0); }
  return ov_tex(t);
}
@fragment
fn fs_main(v : V) -> @location(0) vec4<f32> {
  let uv = v.position.xy / u.size.xy;                 // [0,1], y down
  let ras = u.origin.xyz + u.uvec.xyz * (uv.x - 0.5) + u.vvec.xyz * (0.5 - uv.y);
  let t4 = u.p2t * vec4<f32>(ras, 1.0);
  let tex = t4.xyz;
  if (any(tex < vec3<f32>(0.0)) || any(tex > vec3<f32>(1.0))) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
  let val = textureSampleLevel(t_scalar, s_lin, tex, 0.0).r;
  let win = max(u.params.x, 1e-6);
  let g = clamp((val - (u.params.y - win * 0.5)) / win, 0.0, 1.0);
  var col = vec3<f32>(g);
  if (u.size.w > 0.5) { col = textureLoad(t_lut, vec2<i32>(i32(g * 255.0), 0), 0).rgb; }
  // \u2500\u2500 foreground layer (Slicer's vtkImageBlend semantics per compositing mode) \u2500\u2500
  if (u.fgParams.z > 0.0) {
    let tf = (u.p2tFg * vec4<f32>(ras, 1.0)).xyz;
    if (all(tf >= vec3<f32>(0.0)) && all(tf <= vec3<f32>(1.0))) {
      let fv = textureSampleLevel(t_fg, s_lin, tf, 0.0).r;
      let fwin = max(u.fgParams.x, 1e-6);
      let fg = clamp((fv - (u.fgParams.y - fwin * 0.5)) / fwin, 0.0, 1.0);
      var fcol = vec3<f32>(fg);
      if (u.labelParams.z > 0.5) { fcol = textureLoad(t_lut, vec2<i32>(i32(fg * 255.0), 1), 0).rgb; }
      let a = u.fgParams.z;
      let mode = i32(u.fgParams.w + 0.5);
      if (mode == 0) { col = mix(col, fcol, a); }                       // alpha: fg over bg
      else if (mode == 1) { col = mix(fcol, col, a); }                  // reverse alpha: bg over fg
      else if (mode == 2) { col = clamp(col + fcol * a, vec3<f32>(0.0), vec3<f32>(1.0)); }   // add
      else { col = clamp(col - fcol * a, vec3<f32>(0.0), vec3<f32>(1.0)); }                   // subtract
    }
  }
  // \u2500\u2500 label layer: integer label -> colour table entry, blended at labelOpacity (label 0 = transparent) \u2500\u2500
  if (u.labelParams.x > 0.0) {
    let tl = (u.p2tLabel * vec4<f32>(ras, 1.0)).xyz;
    if (all(tl >= vec3<f32>(0.0)) && all(tl <= vec3<f32>(1.0))) {
      let lv = i32(textureSampleLevel(t_labelVol, s_nn, tl, 0.0).r + 0.5);
      let nEntries = i32(u.labelParams.y);
      if (lv > 0 && lv < nEntries) {
        let lc = textureLoad(t_labelLut, vec2<i32>(lv, 0), 0);
        col = mix(col, lc.rgb, clamp(lc.a * u.labelParams.x, 0.0, 1.0));
      }
    }
  }
  let ov = ov_tex(tex);
  // Slicer-style 2D segmentation: a semi-transparent per-voxel FILL plus a brighter boundary
  // OUTLINE, with independent opacities (params.z = fill, params.w = outline). The outline is
  // screen-space (constant pixel width under zoom), drawn in the segment's own colour along its
  // inner edge \u2014 at both label\u2194label and label\u2194background boundaries.
  let fillA = clamp(ov.a * u.params.z, 0.0, 1.0);
  var outA = 0.0;
  if (u.params.w > 0.0) {
    let du = u.uvec.xyz / u.size.x * 1.5;   // ~1.5 px right, in RAS
    let dv = u.vvec.xyz / u.size.y * 1.5;   // ~1.5 px up
    let n0 = ov_at(ras + du); let n1 = ov_at(ras - du); let n2 = ov_at(ras + dv); let n3 = ov_at(ras - dv);
    let e = max(max(distance(n0.rgb, ov.rgb) + abs(n0.a - ov.a), distance(n1.rgb, ov.rgb) + abs(n1.a - ov.a)),
                max(distance(n2.rgb, ov.rgb) + abs(n2.a - ov.a), distance(n3.rgb, ov.rgb) + abs(n3.a - ov.a)));
    let edge = clamp((e - 0.03) * 12.0, 0.0, 1.0);   // 0 in the interior, 1 at a colour/label edge
    // Label-overlay (Colorize) mode: contours draw at full strength for every VISIBLE segment,
    // independent of its fill opacity, so faint groups (lungs/muscle at ~5%) still read as shapes.
    let segA = select(ov.a, select(0.0, 1.0, ov.a > 0.0), u.size.z > 0.5);
    outA = clamp(segA * u.params.w * edge, 0.0, 1.0);
  }
  col = mix(col, ov.rgb, max(fillA, outA));
  return vec4<f32>(srgb2physical(col), 1.0);
}
`
);
var BASES = {
  axial: { uDir: [-1, 0, 0], vDir: [0, 1, 0], nDir: [0, 0, 1] },
  coronal: { uDir: [-1, 0, 0], vDir: [0, 0, 1], nDir: [0, 1, 0] },
  sagittal: { uDir: [0, -1, 0], vDir: [0, 0, 1], nDir: [1, 0, 0] },
  // ~45° tilt of axial about R–L (default dental / reformat demo plane).
  oblique: {
    uDir: [-1, 0, 0],
    vDir: [0, Math.SQRT1_2, Math.SQRT1_2],
    nDir: [0, -Math.SQRT1_2, Math.SQRT1_2]
  }
};
var dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
var SliceRenderer = class {
  dev;
  format;
  pipeline;
  sampler;
  nnSampler;
  ubuf;
  u = new Float32Array(80);
  // p2t(16) origin(4) uvec(4) vvec(4) params(4) size(4) | p2tFg(16) fgParams(4) p2tLabel(16) labelParams(4)
  bind;
  // Adaptive downsample (moving frames): render the reslice into a low-res target, then bilinear-blit
  // it up to the view — the 2D analogue of SceneRenderer.renderUpscaled. Lets a slice cell degrade
  // resolution under load to keep interactive latency low, snapping back to native when settled.
  blitPipeline;
  lowTex;
  lowView;
  lowW = 0;
  lowH = 0;
  blitBind;
  overlay;
  labels;
  palette;
  scalarTex;
  fgTex;
  lutTex;
  // 256x2 rgba8: row 0 bg LUT, row 1 fg LUT
  labelVolTex;
  labelLutTex;
  // actual in-plane extents (mm) spanned by the LAST rendered viewport, aspect-corrected so
  // pixels stay isotropic on a non-square view (0 until first render → fall back to the square span).
  uSpanMm = 0;
  vSpanMm = 0;
  // volume geometry + current plane
  p2t = new Float32Array(16);
  rasLo = [-1, -1, -1];
  rasHi = [1, 1, 1];
  orient = "axial";
  offset01 = 0.5;
  // Per-orientation pan (mm along the plane's uDir/vDir) + zoom (1 = fitted). Slicer-style
  // slice navigation: pan translates the in-plane view centre, zoom scales the field of view.
  viewState = {
    axial: { panU: 0, panV: 0, zoom: 1 },
    coronal: { panU: 0, panV: 0, zoom: 1 },
    sagittal: { panU: 0, panV: 0, zoom: 1 },
    oblique: { panU: 0, panV: 0, zoom: 1 }
  };
  cX = [0, 0, 0];
  // in-plane centre of the LAST rendered frame (for viewToTex picking)
  // Optional per-orientation basis override (reslice along a volume's own axes). null = the
  // anatomical preset.
  basisOverride = {};
  constructor(gpu, format = DEFAULT_FORMAT2) {
    this.dev = gpu.device;
    this.format = format;
    const m = this.dev.createShaderModule({ code: SHADER });
    this.pipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: m, entryPoint: "vs_main" },
      fragment: { module: m, entryPoint: "fs_main", targets: [{ format }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.sampler = this.dev.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    this.nnSampler = this.dev.createSampler({ magFilter: "nearest", minFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    this.ubuf = this.dev.createBuffer({ size: this.u.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.setWindowLevel(255, 127);
    this.setOverlayOpacity(0.55);
  }
  /** 1x1x1 stand-ins so the label-overlay bindings always exist. The pipeline layout is fixed,
   *  so every caller must bind them even when it only wants a plain MPR. */
  emptyLabels;
  emptyPalette;
  noLabels() {
    if (!this.emptyLabels) {
      this.emptyLabels = this.dev.createTexture({ size: [1, 1, 1], dimension: "3d", format: "r8uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.dev.queue.writeTexture({ texture: this.emptyLabels }, new Uint8Array(1), { bytesPerRow: 1, rowsPerImage: 1 }, [1, 1, 1]);
    }
    return this.emptyLabels;
  }
  noPalette() {
    if (!this.emptyPalette) {
      this.emptyPalette = this.dev.createTexture({ size: [256, 2], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.dev.queue.writeTexture({ texture: this.emptyPalette }, new Uint8Array(256 * 2 * 4), { bytesPerRow: 256 * 4 }, [256, 2]);
    }
    return this.emptyPalette;
  }
  emptyScalar;
  noScalar() {
    if (!this.emptyScalar) {
      this.emptyScalar = this.dev.createTexture({ size: [1, 1, 1], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.dev.queue.writeTexture({ texture: this.emptyScalar }, new Float32Array(1), { bytesPerRow: 4, rowsPerImage: 1 }, [1, 1, 1]);
    }
    return this.emptyScalar;
  }
  emptyLut;
  noLut() {
    if (!this.emptyLut) {
      this.emptyLut = this.dev.createTexture({ size: [256, 2], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.dev.queue.writeTexture({ texture: this.emptyLut }, new Uint8Array(256 * 2 * 4), { bytesPerRow: 256 * 4 }, [256, 2]);
    }
    return this.emptyLut;
  }
  /** Foreground layer: a second scalar volume with its own RAS->texture mapping, W/L, opacity and
   *  compositing mode (Slicer's slice composite node). Pass null to remove. */
  setForeground(tex, p2t, win, lev, opacity, compositing = 0) {
    this.fgTex = tex ?? void 0;
    if (p2t) this.u.set(p2t, 36);
    this.u[52] = win;
    this.u[53] = lev;
    this.u[54] = tex ? opacity : 0;
    this.u[55] = compositing;
    if (this.scalarTex) this.rebind();
  }
  /** Colour LUTs over the W/L ramp for the background (row 0) and foreground (row 1): 256 rgba8 entries
   *  each, or null for the grayscale ramp. */
  setLayerLUTs(bg, fg) {
    if (!bg && !fg) {
      this.lutTex = void 0;
      this.u[35] = 0;
      this.u[58] = 0;
      if (this.scalarTex) this.rebind();
      return;
    }
    if (!this.lutTex) this.lutTex = this.dev.createTexture({ size: [256, 2], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const gray = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      gray[i * 4] = gray[i * 4 + 1] = gray[i * 4 + 2] = i;
      gray[i * 4 + 3] = 255;
    }
    this.dev.queue.writeTexture({ texture: this.lutTex, origin: [0, 0] }, bg ?? gray, { bytesPerRow: 256 * 4 }, [256, 1]);
    this.dev.queue.writeTexture({ texture: this.lutTex, origin: [0, 1] }, fg ?? gray, { bytesPerRow: 256 * 4 }, [256, 1]);
    this.u[35] = bg ? 1 : 0;
    this.u[58] = fg ? 1 : 0;
    if (this.scalarTex) this.rebind();
  }
  /** Label layer: a label volume (integer values in a float texture) coloured through a colour table
   *  (rgba8 entries, index = label value), blended at `opacity`. Pass null to remove. */
  setLabelLayer(tex, p2t, table, opacity) {
    this.labelVolTex = tex ?? void 0;
    if (p2t) this.u.set(p2t, 60);
    const n = table ? table.length / 4 : 0;
    if (table && n > 0) {
      if (!this.labelLutTex || this.labelLutTex.width !== n) {
        this.labelLutTex?.destroy();
        this.labelLutTex = this.dev.createTexture({ size: [n, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      }
      this.dev.queue.writeTexture({ texture: this.labelLutTex }, table, { bytesPerRow: n * 4 }, [n, 1]);
    }
    this.u[76] = tex && table ? opacity : 0;
    this.u[77] = n;
    if (this.scalarTex) this.rebind();
  }
  emptyOverlay;
  transparentOverlay() {
    if (!this.emptyOverlay) {
      this.emptyOverlay = this.dev.createTexture({ size: [1, 1, 1], dimension: "3d", format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.dev.queue.writeTexture({ texture: this.emptyOverlay }, new Uint16Array(4), { bytesPerRow: 8, rowsPerImage: 1 }, [1, 1, 1]);
    }
    return this.emptyOverlay;
  }
  /** Reslice this orientation along an arbitrary RAS basis instead of the anatomical preset.
   *  Pass null to restore. The vectors should be unit length and mutually orthogonal; they are
   *  used verbatim, so the caller owns the display convention for a non-anatomical frame. */
  setBasis(orient, basis) {
    this.basisOverride[orient] = basis;
  }
  /** offset01 (the setPlane scrub coordinate) for a RAS point, along the plane's current normal — the
   *  inverse of what setPlane does internally, so a caller holding a position in mm (a slice node's
   *  centre, a crosshair) can address the same slice for anatomical AND oblique bases. */
  offset01Along(orient, ras) {
    const n = this.basisOf(orient).nDir;
    const { lo: lo2, hi: hi2 } = this.extentAlong(n);
    return Math.max(0, Math.min(1, (dot3(ras, n) - lo2) / Math.max(hi2 - lo2, 1e-6)));
  }
  basisOf(orient) {
    return this.basisOverride[orient] ?? BASES[orient];
  }
  /** Extent of the volume's RAS bounding box projected onto a direction — the generalisation
   *  of "rasHi[axis] - rasLo[axis]" to an oblique axis. Reduces to exactly that for the
   *  anatomical bases, since projecting an axis-aligned box on its own axis is the axis span. */
  extentAlong(d) {
    let lo2 = Infinity, hi2 = -Infinity;
    for (let i = 0; i < 8; i++) {
      const c = [
        i & 1 ? this.rasHi[0] : this.rasLo[0],
        i & 2 ? this.rasHi[1] : this.rasLo[1],
        i & 4 ? this.rasHi[2] : this.rasLo[2]
      ];
      const t = dot3(c, d);
      if (t < lo2) lo2 = t;
      if (t > hi2) hi2 = t;
    }
    return { lo: lo2, hi: hi2 };
  }
  /** Volume geometry: patientToTexture (RAS->tex[0,1], encodes ijkToRAS) + the RAS
   *  bounding box (for plane extents/scrub range). Get both from the ImageField. */
  setVolume(p2t, rasLo, rasHi) {
    this.p2t = p2t;
    this.rasLo = rasLo;
    this.rasHi = rasHi;
    this.u.set(p2t, 0);
  }
  /** Set the grayscale scalar (r32float 3d) and, optionally, a colored overlay
   *  (rgba16float 3d) — which MUST share the same geometry (ijkToRAS/dims) so the
   *  same RAS->tex mapping addresses both. Omit overlay for a plain MPR. */
  setTextures(scalar, overlay) {
    this.overlay = overlay ?? this.transparentOverlay();
    this.scalarTex = scalar;
    this.rebind();
  }
  /** Colour the overlay from a u8 label volume + the 256x2 palette (row 1 = colour/opacity),
   *  instead of a pre-coloured rgba volume. Same geometry requirement as setTextures. Pass
   *  nulls to go back to the rgba overlay. */
  setLabelOverlay(labels, palette) {
    this.labels = labels ?? void 0;
    this.palette = palette ?? void 0;
    this.u[34] = labels && palette ? 1 : 0;
    if (this.scalarTex) this.rebind();
  }
  rebind() {
    if (!this.scalarTex) return;
    this.bind = this.dev.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ubuf } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: this.scalarTex.createView() },
        { binding: 3, resource: (this.overlay ?? this.transparentOverlay()).createView() },
        { binding: 4, resource: this.nnSampler },
        { binding: 5, resource: (this.labels ?? this.noLabels()).createView() },
        { binding: 6, resource: (this.palette ?? this.noPalette()).createView() },
        { binding: 7, resource: (this.fgTex ?? this.noScalar()).createView() },
        { binding: 8, resource: (this.lutTex ?? this.noLut()).createView() },
        { binding: 9, resource: (this.labelVolTex ?? this.noScalar()).createView() },
        { binding: 10, resource: (this.labelLutTex ?? this.noPalette()).createView() }
      ]
    });
  }
  // Uniform float layout: p2t[0..15] origin[16..19] uvec[20..23] vvec[24..27] params[28..31] size[32..35]
  /** Select the anatomical plane and scrub position (0..1 along the plane normal, RAS bbox). */
  setPlane(orient, offset01) {
    this.orient = orient;
    this.offset01 = Math.max(0, Math.min(1, offset01));
  }
  setWindowLevel(win, lev) {
    this.u[28] = win;
    this.u[29] = lev;
  }
  /** Overlay FILL opacity (per-voxel coloured regions). 0 hides the fill. */
  setOverlayOpacity(o) {
    this.u[30] = o;
  }
  /** Overlay OUTLINE opacity (boundary line, composited over the fill). 0 hides the outline. */
  setOutlineOpacity(o) {
    this.u[31] = o;
  }
  /** Convenience toggle: outline on (opacity 1) / off (0). Composites over the fill. */
  setOverlayOutline(on2) {
    this.u[31] = on2 ? 1 : 0;
  }
  /** Physical size (mm) of the square view for the current plane (isotropic, letterboxed).
   *  Matches Slicer's FitSliceToBackground: the field of view is exactly the volume's
   *  extent along the limiting in-plane axis — NO extra margin. (Verified against
   *  Slicer: Red FOV=[891.78,256] at viewport 634x182 -> vertical FOV == the 256mm
   *  A-extent, horizontal follows viewport aspect.) */
  viewSpanMm() {
    const b = this.basisOf(this.orient);
    const u = this.extentAlong(b.uDir), v = this.extentAlong(b.vDir);
    return Math.max(u.hi - u.lo, v.hi - v.lo);
  }
  /** The fitted in-plane extent (mm) used for a given orientation — the value directly
   *  comparable to a Slicer slice node's fitted fieldOfView. */
  spanMmFor(orient) {
    const prev = this.orient;
    this.orient = orient;
    const s = this.viewSpanMm();
    this.orient = prev;
    return s;
  }
  /** Letterbox fit at zoom=1 (Slicer's FitSliceToVolume): the in-plane FOV (uS0×vS0) that
   *  exactly contains the slice's bounding box in a viewport of the given aspect — the whole
   *  slice is visible and the LIMITING axis touches the window edge (so the largest fitting
   *  axis fills the window, no needless margin). Replaces the old max(uExt,vExt) span, which
   *  under-zoomed whenever the larger extent wasn't on the viewport's limiting axis. */
  fitUV(orient, aspectWH) {
    const b = this.basisOf(orient);
    const u = this.extentAlong(b.uDir), v = this.extentAlong(b.vDir);
    const uExt = u.hi - u.lo, vExt = v.hi - v.lo;
    const uS0 = Math.max(uExt, vExt * aspectWH);
    return { uS0, vS0: uS0 / aspectWH };
  }
  /** The complete in-plane view frame for an orientation at a given viewport aspect, folding
   *  in pan (mm along uDir/vDir) + zoom. Single source of truth shared by drawInto, rasToView,
   *  viewToRas — so the rendered image and the markup projection stay pixel-aligned under
   *  pan/zoom. Returns the plane centre `c` (RAS, incl. scrub offset + pan) and the half-... no:
   *  uS/vS are the FULL in-plane extents mapped across the viewport width/height. */
  frameFor(orient, offset01, aspectWH) {
    const b = this.basisOf(orient);
    const vs2 = this.viewState[orient];
    const { uS0, vS0 } = this.fitUV(orient, aspectWH);
    const uS = uS0 / vs2.zoom, vS = vS0 / vs2.zoom;
    const c = [(this.rasLo[0] + this.rasHi[0]) / 2, (this.rasLo[1] + this.rasHi[1]) / 2, (this.rasLo[2] + this.rasHi[2]) / 2];
    const nx = this.extentAlong(b.nDir);
    const want = nx.lo + Math.max(0, Math.min(1, offset01)) * (nx.hi - nx.lo);
    const have = dot3(c, b.nDir);
    c[0] += b.nDir[0] * (want - have);
    c[1] += b.nDir[1] * (want - have);
    c[2] += b.nDir[2] * (want - have);
    c[0] += b.uDir[0] * vs2.panU + b.vDir[0] * vs2.panV;
    c[1] += b.uDir[1] * vs2.panU + b.vDir[1] * vs2.panV;
    c[2] += b.uDir[2] * vs2.panU + b.vDir[2] * vs2.panV;
    return { b, c, uS, vS };
  }
  /** Zoom factor for an orientation (1 = fitted). */
  zoom(orient) {
    return this.viewState[orient].zoom;
  }
  /** Pan the in-plane view by a pixel delta (drag): the anatomy under the cursor follows it. */
  panByPixels(orient, dxPx, dyPx, w, h) {
    const z2 = this.viewState[orient].zoom;
    const { uS0, vS0 } = this.fitUV(orient, w / h);
    const uS = uS0 / z2, vS = vS0 / z2;
    this.viewState[orient].panU -= dxPx / w * uS;
    this.viewState[orient].panV += dyPx / h * vS;
  }
  /** Zoom by `factor` (>1 zooms in) about a pivot (u,v in [0,1]); the pivot point stays fixed. */
  zoomAbout(orient, factor, pu, pv, w, h) {
    const vs2 = this.viewState[orient];
    const { uS0, vS0 } = this.fitUV(orient, w / h);
    const z2 = Math.max(0.2, Math.min(50, vs2.zoom * factor));
    vs2.panU += (pu - 0.5) * (uS0 / vs2.zoom - uS0 / z2);
    vs2.panV += (0.5 - pv) * (vS0 / vs2.zoom - vS0 / z2);
    vs2.zoom = z2;
  }
  /** Reset pan/zoom for an orientation to the fitted view. */
  resetView(orient) {
    this.viewState[orient] = { panU: 0, panV: 0, zoom: 1 };
  }
  /** Snapshot per-orientation pan+zoom (e.g. to persist a view across reloads). */
  getViewState() {
    return structuredClone(this.viewState);
  }
  /** Restore a (possibly partial) snapshot from getViewState(). */
  setViewState(vs2) {
    for (const k of Object.keys(vs2)) {
      const v = vs2[k];
      if (v && Number.isFinite(v.zoom) && v.zoom > 0) this.viewState[k] = { ...v };
    }
  }
  /** Mirror Slicer's in-plane navigation for an orientation: drive pan + zoom from the slice
   *  node's RAS centre and field of view (mm). zoom = extent/FOV on the limiting axis (== 1 when
   *  Slicer is fitted, per FitSliceToBackground's no-margin fit), so SlicerLive tracks Slicer's
   *  zoom proportionally; pan is the centre's offset from the volume centre projected onto the
   *  plane's in-plane axes. The out-of-plane offset is applied separately via setPlane. */
  setMirrorFrame(orient, centerRAS, fovX, fovY) {
    const b = this.basisOf(orient);
    const aspect = fovX / Math.max(fovY, 1e-6);
    const { uS0 } = this.fitUV(orient, aspect);
    const zoom = Math.max(1e-3, uS0 / Math.max(fovX, 1e-6));
    const volC = [(this.rasLo[0] + this.rasHi[0]) / 2, (this.rasLo[1] + this.rasHi[1]) / 2, (this.rasLo[2] + this.rasHi[2]) / 2];
    const d = [centerRAS[0] - volC[0], centerRAS[1] - volC[1], centerRAS[2] - volC[2]];
    const panU = d[0] * b.uDir[0] + d[1] * b.uDir[1] + d[2] * b.uDir[2];
    const panV = d[0] * b.vDir[0] + d[1] * b.vDir[1] + d[2] * b.vDir[2];
    this.viewState[orient] = { panU, panV, zoom };
  }
  /** The current pan/zoom of a plane expressed the way Slicer's slice node stores it: in-plane centre
   *  (RAS, without the out-of-plane offset which the caller owns) + field of view (mm) — the inverse
   *  of setMirrorFrame, so a local pan/zoom can be written back to the app as a slice frame. */
  mirrorFrame(orient, aspectWH) {
    const b = this.basisOf(orient);
    const st = this.viewState[orient];
    const { uS0, vS0 } = this.fitUV(orient, aspectWH);
    const fovX = uS0 / st.zoom, fovY = vS0 / st.zoom;
    const volC = [(this.rasLo[0] + this.rasHi[0]) / 2, (this.rasLo[1] + this.rasHi[1]) / 2, (this.rasLo[2] + this.rasHi[2]) / 2];
    const centerRAS = [
      volC[0] + b.uDir[0] * st.panU + b.vDir[0] * st.panV,
      volC[1] + b.uDir[1] * st.panU + b.vDir[1] * st.panV,
      volC[2] + b.uDir[2] * st.panU + b.vDir[2] * st.panV
    ];
    return { centerRAS, fovX, fovY };
  }
  /** Map a view (u,v) in [0,1] (y down) to normalized texture coords for the current
   *  plane — for click picking. Returns the tex coord; the caller converts to IJK via
   *  ijk = tex*dims - 0.5. Anisotropy/rotation are handled by the same p2t the shader uses. */
  viewToTex(u, v) {
    const b = this.basisOf(this.orient);
    const uS = this.uSpanMm || this.viewSpanMm();
    const vS = this.vSpanMm || this.viewSpanMm();
    const c = this.cX;
    const ras = [
      c[0] + b.uDir[0] * (u - 0.5) * uS + b.vDir[0] * (0.5 - v) * vS,
      c[1] + b.uDir[1] * (u - 0.5) * uS + b.vDir[1] * (0.5 - v) * vS,
      c[2] + b.uDir[2] * (u - 0.5) * uS + b.vDir[2] * (0.5 - v) * vS
    ];
    return applyMat4(this.p2t, ras);
  }
  /** Project a RAS point onto a plane's view: returns u,v in [0,1] (y down, matching the
   *  rendered pixels for a viewport of aspect w/h) and the signed distance (mm) from the
   *  point to the plane along its normal. Inverse of viewToTex; used to place 2D markup
   *  glyphs and hit-test clicks on them. */
  rasToView(orient, offset01, ras, aspectWH) {
    const { b, c, uS, vS } = this.frameFor(orient, offset01, aspectWH);
    const d = [ras[0] - c[0], ras[1] - c[1], ras[2] - c[2]];
    const u = 0.5 + (d[0] * b.uDir[0] + d[1] * b.uDir[1] + d[2] * b.uDir[2]) / uS;
    const v = 0.5 - (d[0] * b.vDir[0] + d[1] * b.vDir[1] + d[2] * b.vDir[2]) / vS;
    return { u, v, distMm: dot3(d, b.nDir) };
  }
  /** Map a view (u,v in [0,1], y down) on a plane back to a RAS point ON that plane —
   *  the exact inverse of rasToView (same pan/zoom/aspect). Used to drag a 2D markup:
   *  the point lands on the current slice (its out-of-plane coord becomes the plane offset). */
  viewToRas(orient, offset01, u, v, aspectWH) {
    const { b, c, uS, vS } = this.frameFor(orient, offset01, aspectWH);
    const du = (u - 0.5) * uS, dv = (0.5 - v) * vS;
    return [
      c[0] + b.uDir[0] * du + b.vDir[0] * dv,
      c[1] + b.uDir[1] * du + b.vDir[1] * dv,
      c[2] + b.uDir[2] * du + b.vDir[2] * dv
    ];
  }
  drawInto(view, w, h) {
    const { b, c, uS, vS } = this.frameFor(this.orient, this.offset01, w / h);
    this.uSpanMm = uS;
    this.vSpanMm = vS;
    this.cX = c;
    this.u.set(this.p2t, 0);
    this.u[16] = c[0];
    this.u[17] = c[1];
    this.u[18] = c[2];
    this.u[19] = 0;
    this.u[20] = b.uDir[0] * uS;
    this.u[21] = b.uDir[1] * uS;
    this.u[22] = b.uDir[2] * uS;
    this.u[23] = 0;
    this.u[24] = b.vDir[0] * vS;
    this.u[25] = b.vDir[1] * vS;
    this.u[26] = b.vDir[2] * vS;
    this.u[27] = 0;
    this.u[32] = w;
    this.u[33] = h;
    this.dev.queue.writeBuffer(this.ubuf, 0, this.u);
    const enc = this.dev.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind);
    pass.draw(3);
    pass.end();
    this.dev.queue.submit([enc.finish()]);
  }
  renderToView(view, w, h) {
    this.drawInto(view, w, h);
  }
  /** Bilinear-blit pipeline (fullscreen triangle) that upsamples the low-res reslice to the view. */
  ensureBlit() {
    if (this.blitPipeline) return;
    const m = this.dev.createShaderModule({
      code: (
        /* wgsl */
        `
struct VO { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o: VO; o.pos = vec4<f32>(p[i], 0.0, 1.0);
  o.uv = vec2<f32>((p[i].x + 1.0) * 0.5, (1.0 - p[i].y) * 0.5); return o;
}
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment fn fs(in: VO) -> @location(0) vec4<f32> { return textureSample(src, samp, in.uv); }`
      )
    });
    this.blitPipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: m, entryPoint: "vs" },
      fragment: { module: m, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
  }
  ensureLow(w, h) {
    this.ensureBlit();
    if (this.lowTex && this.lowW === w && this.lowH === h) return;
    this.lowTex?.destroy();
    this.lowTex = this.dev.createTexture({ size: [w, h], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.lowView = this.lowTex.createView();
    this.lowW = w;
    this.lowH = h;
    this.blitBind = this.dev.createBindGroup({
      layout: this.blitPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: this.lowView }, { binding: 1, resource: this.sampler }]
    });
  }
  /** Adaptive (moving-frame) render: reslice at `rw×rh` into an off-screen target, then bilinear-blit
   *  up to the `vw×vh` view. Single frame, no accumulation — use while interacting; call renderToView
   *  (native) when the view settles. At rw==vw/rh==vh this is a native render plus a pass-through blit. */
  renderUpscaled(view, rw, rh, vw, vh) {
    if (rw >= vw && rh >= vh) {
      this.drawInto(view, vw, vh);
      return;
    }
    this.ensureLow(rw, rh);
    this.drawInto(this.lowView, rw, rh);
    const enc = this.dev.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(this.blitPipeline);
    pass.setBindGroup(0, this.blitBind);
    pass.draw(3);
    pass.end();
    this.dev.queue.submit([enc.finish()]);
  }
  async renderToRGBA(w, h) {
    const target = this.dev.createTexture({ size: [w, h], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    this.drawInto(target.createView(), w, h);
    const bpr = Math.ceil(w * 4 / 256) * 256;
    const buf = this.dev.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: target }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: h }, [w, h]);
    this.dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(w * h * 4);
    for (let y2 = 0; y2 < h; y2++) out.set(padded.subarray(y2 * bpr, y2 * bpr + w * 4), y2 * w * 4);
    buf.unmap();
    target.destroy();
    buf.destroy();
    return out;
  }
};

// render/vtk-camera.ts
var sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
var add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
var scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
var cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0]
];
var dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
var norm = (a) => Math.hypot(a[0], a[1], a[2]);
var normalize = (a) => {
  const n = norm(a) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
};
function rotateAboutAxis(v, axis, deg) {
  const k = normalize(axis);
  const t = deg * Math.PI / 180;
  const c = Math.cos(t), s = Math.sin(t);
  const kv = cross(k, v);
  const kd = dot(k, v);
  return [
    v[0] * c + kv[0] * s + k[0] * kd * (1 - c),
    v[1] * c + kv[1] * s + k[1] * kd * (1 - c),
    v[2] * c + kv[2] * s + k[2] * kd * (1 - c)
  ];
}
var VtkCamera = class _VtkCamera {
  position;
  focalPoint;
  viewUp;
  viewAngle;
  // degrees (vtkCamera default 30)
  parallelProjection = false;
  parallelScale = 1;
  constructor(position = [0, 0, 1], focalPoint = [0, 0, 0], viewUp = [0, 1, 0], viewAngle = 30) {
    this.position = [...position];
    this.focalPoint = [...focalPoint];
    this.viewUp = [...viewUp];
    this.viewAngle = viewAngle;
  }
  /** Slicer's default 3D camera (vtkMRMLCameraNode): (0,500,0) -> origin, +S up, 30 deg. */
  static slicerDefault() {
    return new _VtkCamera([0, 500, 0], [0, 0, 0], [0, 0, 1], 30);
  }
  clone() {
    const c = new _VtkCamera(this.position, this.focalPoint, this.viewUp, this.viewAngle);
    c.parallelProjection = this.parallelProjection;
    c.parallelScale = this.parallelScale;
    return c;
  }
  get distance() {
    return norm(sub(this.focalPoint, this.position));
  }
  /** normalize(focalPoint - position) — vtkCamera::DirectionOfProjection. */
  get directionOfProjection() {
    return normalize(sub(this.focalPoint, this.position));
  }
  /** Rows of the view transform, per vtkTransform::SetupCamera. */
  basis(viewUp = this.viewUp) {
    const back = normalize(sub(this.position, this.focalPoint));
    const right = normalize(cross(viewUp, back));
    const up = cross(back, right);
    return { right, up, back };
  }
  /** vtkCamera::Azimuth — rotate position about viewUp through the focal point. */
  azimuth(deg) {
    const rel = sub(this.position, this.focalPoint);
    this.position = add(this.focalPoint, rotateAboutAxis(rel, this.viewUp, deg));
  }
  /** vtkCamera::Elevation — rotate position about -right through the focal point.
   *  Returns the rotated view-up VTK uses internally (see class comment); callers that
   *  mirror Slicer follow with orthogonalizeViewUp(rotatedUp). */
  elevation(deg) {
    const axis = scale(this.basis().right, -1);
    const rotatedUp = rotateAboutAxis(this.viewUp, axis, deg);
    const rel = sub(this.position, this.focalPoint);
    this.position = add(this.focalPoint, rotateAboutAxis(rel, axis, deg));
    return rotatedUp;
  }
  /** vtkCamera::OrthogonalizeViewUp — viewUp = row1 of the view transform. */
  orthogonalizeViewUp(usingUp = this.viewUp) {
    this.viewUp = this.basis(usingUp).up;
  }
  /** vtkCamera::Dolly — factor > 1 moves the camera toward the focal point. */
  dolly(factor) {
    if (factor <= 0) return;
    if (this.parallelProjection) {
      this.parallelScale = this.parallelScale / factor;
      return;
    }
    const d = this.distance / factor;
    const dop = this.directionOfProjection;
    this.position = sub(this.focalPoint, scale(dop, d));
  }
  /** Translate both position and focal point (used by pan). */
  translate(v) {
    this.position = add(this.position, v);
    this.focalPoint = add(this.focalPoint, v);
  }
  /** Half-height of the view plane at the focal point (perspective). */
  focalPlaneHalfHeight() {
    return this.parallelProjection ? this.parallelScale : this.distance * Math.tan(this.viewAngle * Math.PI / 360);
  }
  /** Pan by a display-space delta, moving the world under the cursor 1:1 at focal depth.
   *  Equivalent to vtkMRMLCameraWidget::ProcessTranslate's focal-depth unprojection, but
   *  expressed directly in the camera basis (exact for a centred perspective view).
   *  dxDisplay/dyDisplay are in VTK display convention (y UP). */
  panByDisplayDelta(dxDisplay, dyDisplay, viewportWidth, viewportHeight) {
    const halfH = this.focalPlaneHalfHeight();
    const mmPerPixel = 2 * halfH / viewportHeight;
    const { right, up } = this.basis();
    const motion = add(scale(right, -dxDisplay * mmPerPixel), scale(up, -dyDisplay * mmPerPixel));
    this.translate(motion);
  }
  /** Project a world (RAS) point to display pixels (y DOWN, origin top-left) for a w×h viewport.
   *  Vertical-FOV perspective matching SceneRenderer.setCamera (perspectiveZO(fovy, w/h)). `depth`
   *  is the distance along the view direction (>0 in front of the camera). Used to hit-test
   *  screen-space markup glyphs. */
  worldToDisplay(p, w, h) {
    const { right, up } = this.basis();
    const dop = this.directionOfProjection;
    const rel = sub(p, this.position);
    const depth = dot(rel, dop);
    const halfH = Math.max(1e-6, depth) * Math.tan(this.viewAngle * Math.PI / 360);
    const aspect = w / h;
    const ndcx = dot(rel, right) / (halfH * aspect);
    const ndcy = dot(rel, up) / halfH;
    return { x: (ndcx * 0.5 + 0.5) * w, y: (0.5 - ndcy * 0.5) * h, depth };
  }
  /** Inverse of worldToDisplay at a FIXED view-depth: the world point under display pixel (x,y)
   *  lying in the plane perpendicular to the view at `depth`. Dragging a 3D handle in this plane
   *  keeps its distance from the camera, so it tracks the cursor without depth ambiguity. */
  displayToWorldAtDepth(x, y2, depth, w, h) {
    const { right, up } = this.basis();
    const dop = this.directionOfProjection;
    const halfH = Math.max(1e-6, depth) * Math.tan(this.viewAngle * Math.PI / 360);
    const aspect = w / h;
    const ndcx = x / w * 2 - 1;
    const ndcy = 1 - y2 / h * 2;
    const offset = add(scale(right, ndcx * halfH * aspect), scale(up, ndcy * halfH));
    return add(add(this.position, scale(dop, depth)), offset);
  }
  /** vtkCamera-comparable snapshot for the harness. */
  state() {
    return {
      position: [...this.position],
      focalPoint: [...this.focalPoint],
      viewUp: [...this.viewUp],
      viewAngle: this.viewAngle,
      distance: this.distance
    };
  }
};

// render/budget-controller.ts
var BudgetController = class {
  budgetPx;
  targetMs;
  minPx;
  maxPx;
  constructor(opts = {}) {
    this.targetMs = opts.targetMs ?? 16;
    this.minPx = opts.minPx ?? 3e4;
    this.maxPx = opts.maxPx ?? 8e6;
    this.budgetPx = opts.startPx ?? 35e4;
  }
  /** Nudge the budget toward hitting targetMs. Multiplicative, clamped per step (0.8–1.25×) so the
   *  loop is stable, and bounded to [minPx, maxPx]. Faster-than-target grows it; slower shrinks it. */
  update(measuredMs) {
    if (!(measuredMs > 0) || !Number.isFinite(measuredMs)) return;
    const adj = Math.max(0.35, Math.min(1.2, this.targetMs / measuredMs));
    this.budgetPx = Math.max(this.minPx, Math.min(this.maxPx, this.budgetPx * adj));
  }
  /** Resolution scale for a `w×h` view: sqrt(budget / area), clamped to [0.25, 1]. 1 when the view
   *  already fits the budget (small window); a fraction for a big/retina window under load. */
  scale(w, h) {
    const area = Math.max(1, w * h);
    return Math.max(0.25, Math.min(1, Math.sqrt(this.budgetPx / area)));
  }
};

// render/demos/accum-loop.ts
function mountAdaptiveLoop(opts) {
  const target = opts.target ?? 32;
  const idleGap = opts.idleGapMs ?? 120;
  const paced = () => Promise.race([
    new Promise((r) => requestAnimationFrame(() => r())),
    new Promise((r) => setTimeout(r, 33))
  ]);
  const sync = opts.sync ?? (() => Promise.resolve());
  let running = false, stopped = false, lastKick = -1e12, wasMoving = false;
  const step = () => {
    if (performance.now() - lastKick < idleGap) {
      opts.renderMoving();
      wasMoving = true;
      return true;
    }
    if (wasMoving) {
      wasMoving = false;
      opts.renderSettled(true);
      return true;
    }
    if (opts.count() < target) {
      opts.renderSettled(false);
      return true;
    }
    return false;
  };
  const run = async () => {
    running = true;
    stopped = false;
    while (!stopped && step()) await Promise.all([sync(), paced()]);
    running = false;
  };
  return {
    kick() {
      lastKick = performance.now();
      if (!running) run();
    },
    // run() renders the 1st frame synchronously
    stop() {
      stopped = true;
    }
  };
}
function mountAdaptive3d(opts) {
  const budget = new BudgetController({ targetMs: opts.targetMs ?? 16 });
  const DBG = typeof location !== "undefined" && new URLSearchParams(location.search).has("perf");
  let dbgN = 0, dbgMoving = 0, dbgSettled = 0, dbgLast = 0;
  const dbgTick = (kind, ms2, s) => {
    if (!DBG) return;
    dbgN++;
    if (kind === "mov") dbgMoving += ms2;
    else dbgSettled += ms2;
    const now = performance.now();
    if (now - dbgLast > 500) {
      console.log(`[perf] mov=${dbgMoving.toFixed(0)}ms/${dbgN}f settled=${dbgSettled.toFixed(0)}ms lastScale=${s.toFixed(2)} last=${ms2.toFixed(1)}ms`);
      dbgLast = now;
      dbgMoving = dbgSettled = dbgN = 0;
    }
  };
  const movingCap = opts.movingScaleCap ?? 1;
  const renderMoving = () => {
    const sc = opts.scene();
    if (!sc) return;
    const { w: vw, h: vh } = opts.size();
    if (!vw || !vh) return;
    const s = Math.min(movingCap, budget.scale(vw, vh)), t0 = performance.now();
    if (s > 0.98) {
      opts.setCamera(sc, vw, vh);
      sc.renderToView(opts.view(), vw, vh);
    } else {
      const rw = Math.max(16, Math.round(vw * s)), rh = Math.max(16, Math.round(vh * s));
      opts.setCamera(sc, rw, rh);
      sc.renderUpscaled(opts.view(), rw, rh, vw, vh);
    }
    opts.gpu.device.queue.onSubmittedWorkDone().then(() => {
      const ms2 = performance.now() - t0;
      budget.update(ms2);
      dbgTick("mov", ms2, s);
    });
    opts.onFrame?.();
  };
  const renderSettled = (reset) => {
    const sc = opts.scene();
    if (!sc) return;
    const { w: vw, h: vh } = opts.size();
    if (!vw || !vh) return;
    const t0 = performance.now();
    opts.setCamera(sc, vw, vh);
    sc.renderAccum(opts.view(), vw, vh, reset);
    if (DBG) opts.gpu.device.queue.onSubmittedWorkDone().then(() => dbgTick("set", performance.now() - t0, 1));
    opts.onFrame?.();
  };
  const loop = mountAdaptiveLoop({
    renderMoving,
    renderSettled,
    count: () => opts.scene()?.accumCount() ?? 1e9,
    target: opts.target ?? 24,
    idleGapMs: opts.idleGapMs,
    sync: () => opts.gpu.device.queue.onSubmittedWorkDone()
    // GPU-paced: no backlog, input preempts
  });
  let kickN = 0, kickLast = 0;
  const draw = () => {
    if (DBG) {
      kickN++;
      const now = performance.now();
      if (now - kickLast > 500) {
        console.log(`[perf] kicks=${kickN} in 500ms`);
        kickN = 0;
        kickLast = now;
      }
    }
    loop.kick();
  };
  return { draw, budget, renderSettled, renderMoving, loop };
}

// render/rate-limiter.ts
var Coalescer = class {
  /**
   * @param intervalMs minimum wall time between wire flushes (e.g. 33 = ~30Hz).
   * @param flush       called with the coalesced batch (latest value per key) to put on the wire.
   * @param now         clock (overridable for tests); defaults to performance.now.
   */
  constructor(intervalMs, flush, now = () => performance.now()) {
    this.intervalMs = intervalMs;
    this.flush = flush;
    this.now = now;
  }
  pending = /* @__PURE__ */ new Map();
  timer = null;
  lastFlush = -Infinity;
  /** Record the latest value for `key`. Cheap — safe to call every animation frame. */
  update(key, value) {
    this.pending.set(key, value);
    this.schedule();
  }
  schedule() {
    if (this.timer !== null || this.pending.size === 0) return;
    const wait = Math.max(0, this.intervalMs - (this.now() - this.lastFlush));
    this.timer = setTimeout(() => this.doFlush(), wait);
  }
  doFlush() {
    this.timer = null;
    if (this.pending.size === 0) return;
    this.lastFlush = this.now();
    const batch = this.pending;
    this.pending = /* @__PURE__ */ new Map();
    this.flush(batch);
    this.schedule();
  }
  /** Force the pending batch out now (e.g. on pointer-up: the authoritative final value must land
   *  without waiting out the interval). Resets the rate window so the next update can flush at once. */
  flushNow() {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.lastFlush = -Infinity;
    this.doFlush();
  }
  /** Drop any pending values without sending (e.g. interaction cancelled). */
  clear() {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.pending.clear();
  }
  get pendingCount() {
    return this.pending.size;
  }
};

// render/livesync.ts
function opKey(op) {
  if (op.op === "cmd") return `${op.id}:${op.cmd}:${op.args?.index ?? ""}`;
  if (op.op === "patch") return `${op.id}:${op.path}`;
  return `${op.id}:${op.op}`;
}
var LiveSync = class {
  constructor(scene, transport, opts = {}) {
    this.scene = scene;
    this.transport = transport;
    this.peerId = opts.peerId ?? "remote";
    this.relay = !!opts.relay;
    this.relayTypes = opts.relayTypes ? new Set(opts.relayTypes) : void 0;
    this.now = opts.now ?? (() => Date.now());
    this.out = new Coalescer(
      opts.intervalMs ?? 33,
      (batch) => {
        const ops = [...batch.values()];
        const tag = ++this.tag;
        this.pending.set(tag, ops);
        this.transport.send({ op: "applyOps", ops, tag });
      },
      opts.now ?? (() => typeof performance !== "undefined" ? performance.now() : Date.now())
    );
    transport.onMessage = (m) => this.onMessage(m);
    transport.onOpen = () => this.onOpen();
    transport.onClose = () => this.onClose();
  }
  onStatus;
  out;
  tag = 0;
  /** Sent batches not yet acknowledged by the peer (tag -> ops). Re-sent after a reconnect so a burst
   *  that raced the drop is not lost; cleared by OpAck. */
  pending = /* @__PURE__ */ new Map();
  /** Highest peer seq seen (a checkpoint for resumable reconnects). */
  lastSeq = 0;
  everConnected = false;
  unsub;
  inq = Promise.resolve();
  // serialize inbound handling in arrival order
  // event-driven reconnect (no heartbeat)
  backoff = { base: 1e3, factor: 2, max: 3e4 };
  attempt = 0;
  retryTimer;
  wantClose = false;
  firstOpen;
  now;
  peerId;
  relayTypes;
  relay;
  relayed = /* @__PURE__ */ new Map();
  // node id -> last content relayed to this peer
  relayedDel = /* @__PURE__ */ new Set();
  relayCount = 0;
  emit(s) {
    try {
      this.onStatus?.(s);
    } catch {
    }
  }
  /** Connect and keep connected. Resolves on the FIRST open; later drops auto-reconnect. No reject —
   *  a down peer just keeps retrying (progress via onStatus). */
  connect() {
    this.wantClose = false;
    return new Promise((resolve) => {
      this.firstOpen = resolve;
      this.transport.connect();
    });
  }
  onOpen() {
    this.attempt = 0;
    const reconnect = this.everConnected;
    this.everConnected = true;
    this.transport.send({ op: "subscribe", types: this.scene.subscribedTypes(), localBulk: this.scene.localBulk(), lastSeq: this.lastSeq });
    this.unsub?.();
    this.unsub = this.scene.subscribe((c) => this.onLocalChange(c));
    if (reconnect) {
      this.reconcileAfter = true;
      this.reconcileSnapshot = JSON.parse(JSON.stringify(Object.fromEntries(this.scene.nodes)));
    }
    this.emit({ state: "connected" });
    this.firstOpen?.();
    this.firstOpen = void 0;
  }
  reconcileAfter = false;
  reconcileSnapshot = {};
  /** After a reconnect the peer (a restarted app, or one that diverged while we were away) re-snapshots;
   *  authority inversion means OUR node map wins: send it as a reconcile, then re-send unacked batches. */
  reconcile() {
    this.reconcileAfter = false;
    this.transport.send({ op: "reconcile", nodes: this.reconcileSnapshot });
    this.reconcileSnapshot = {};
    for (const [tag, ops] of this.pending) this.transport.send({ op: "applyOps", ops, tag });
  }
  onLocalChange(c) {
    if (c.origin === this.peerId) return;
    if (this.relayTypes && c.type && !this.relayTypes.has(c.type)) return;
    if (c.op && c.origin === this.scene.origin) {
      this.out.update(opKey(c.op), c.op);
      return;
    }
    if (!this.relay || c.origin === this.scene.origin) return;
    if (c.kind === "upsert" && c.node) {
      const body = JSON.stringify(c.node);
      if (this.relayed.get(c.id) === body) return;
      this.relayed.set(c.id, body);
      if (++this.relayCount > 5e3) {
        console.warn("LiveSync: relay cap reached \u2014 possible loop; relay disabled for", this.peerId);
        this.relay = false;
        return;
      }
      this.out.update(`${c.id}:put`, { op: "put", id: c.id, node: c.node, origin: c.origin, role: "module" });
    } else if (c.kind === "remove") {
      if (!this.relayed.has(c.id) && this.relayedDel.has(c.id)) return;
      this.relayed.delete(c.id);
      this.relayedDel.add(c.id);
      this.out.update(`${c.id}:del`, { op: "del", id: c.id, origin: c.origin });
    }
  }
  onMessage(m) {
    const msg = m;
    if (!msg || typeof msg !== "object" || !("event" in msg)) return;
    if (typeof msg.seq === "number" && msg.seq > this.lastSeq) this.lastSeq = msg.seq;
    if (msg.event === "OpAck") {
      if (typeof msg.tag === "number") this.pending.delete(msg.tag);
      if (msg.created) for (const [clientId, realId] of Object.entries(msg.created)) this.scene.aliasNode(clientId, realId);
      return;
    }
    if (msg.event === "Reconciled") return;
    if (this.relay && msg.event === "NodeAdded" && msg.node?.id) {
      const n = msg.node;
      this.relayed.set(n.id, JSON.stringify(n));
    }
    this.inq = this.inq.then(() => this.scene.receiveEvent(msg, this.peerId));
    if (msg.event === "SnapshotComplete" && this.reconcileAfter) this.inq = this.inq.then(() => this.reconcile());
  }
  onClose() {
    this.unsub?.();
    this.unsub = void 0;
    if (!this.wantClose) this.scheduleReconnect();
  }
  scheduleReconnect() {
    if (this.wantClose || this.retryTimer !== void 0) return;
    const delay = Math.min(this.backoff.max, this.backoff.base * this.backoff.factor ** this.attempt);
    this.attempt++;
    this.emit({ state: "waiting", attempt: this.attempt, nextRetryAt: this.now() + delay });
    this.retryTimer = setTimeout(() => {
      this.retryTimer = void 0;
      this.emit({ state: "connecting", attempt: this.attempt });
      this.transport.connect();
    }, delay);
  }
  /** Force an immediate reconnect (the "Try now" button); does not reset the backoff schedule. */
  reconnectNow() {
    if (this.retryTimer !== void 0) {
      clearTimeout(this.retryTimer);
      this.retryTimer = void 0;
    }
    this.emit({ state: "connecting", attempt: this.attempt });
    this.transport.connect();
  }
  /** Direct outbound (e.g. a pointer drag that owns its own optimistic local update): coalesced onto
   *  the wire exactly like a feed-driven write. Ops are stamped local-origin if unset. */
  sendOps(ops) {
    for (const o of ops) {
      const op = o.origin ? o : { ...o, origin: this.scene.origin };
      this.out.update(opKey(op), op);
    }
  }
  /** Flush pending outbound now (e.g. on pointer-up — the authoritative final value must not wait). */
  flush() {
    this.out.flushNow();
  }
  close() {
    this.wantClose = true;
    if (this.retryTimer !== void 0) {
      clearTimeout(this.retryTimer);
      this.retryTimer = void 0;
    }
    this.unsub?.();
    this.unsub = void 0;
    this.out.clear();
    this.transport.close();
  }
};

// render/demos/sl-logo.ts
var SL_LOGO = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAA8CAIAAABTt4VhAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAARGVYSWZNTQAqAAAACAABh2kABAAAAAEAAAAaAAAAAAADoAEAAwAAAAEAAQAAoAIABAAAAAEAAAA5oAMABAAAAAEAAAA8AAAAAH9xBdAAAAHLaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA2LjAuMCI+CiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOmV4aWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vZXhpZi8xLjAvIj4KICAgICAgICAgPGV4aWY6Q29sb3JTcGFjZT4xPC9leGlmOkNvbG9yU3BhY2U+CiAgICAgICAgIDxleGlmOlBpeGVsWERpbWVuc2lvbj41MDA8L2V4aWY6UGl4ZWxYRGltZW5zaW9uPgogICAgICAgICA8ZXhpZjpQaXhlbFlEaW1lbnNpb24+NTIwPC9leGlmOlBpeGVsWURpbWVuc2lvbj4KICAgICAgPC9yZGY6RGVzY3JpcHRpb24+CiAgIDwvcmRmOlJERj4KPC94OnhtcG1ldGE+ConTBbQAABmbSURBVGgFjZpZkB3XWcd7775919k1o2VGsjZLthw5sR3HiZ3EGMcJJqRIXKmiqAKTByh4yEN4pQJFUVBUUUWRQIViCVQZQ0IWJyGLYyeyY8lYkrXYlmxJtnbNSLPeO3fpvZvfd/pKOOSFnjt97+0+fc7//L/1fOfqjbFZjaPItCLVNU3Xdc3gj3fd0DTP1bZtsD/5yNwnfvmef/23Z1860VlY1dJcM7lrFAYnXePEWeMD/ci/HEWha3mRa0XOvzrzIS+0jAtylbci40XDnNZyq5B3+aQ+FwXf5UKhGZZhWHzmX/qnZ66XAGV0EHNNF0zcWm/3gl77Pfu2Hzv9qmvrZloYprQBpMnEmJl6VB4qDzUEGPJc5yNnBTTPcs3IDb5lZVPOQDZkPvBCe8Ys6FDgCgAQ8cctdegWX9QMNMWQ3FT3ChCYhuHYuu9oFUczbbtSqTiW5jl6bOiWyWRBWWItcUuHJVohhpdg5SVEpvDFma8pf0ZKC0CkhW7qaUY/Q7jyCA8K3BK6dKJkLV0LrxyCT+YgQKESoUKbY2sNX9805e3fv3V6Zvrgy6fTwrRNgGoiFUvOMiU4pj0Pqlmq7mVIAXoTIoCyTLCm4ABcljNKyoBZgexuwlWS5aKSP9wpphU+JXWwCtsykCioUG+YcGwAqOrokyPW3u2jO7ZtPvDcz469di0rTMfKhVRLGticBavBiw+i63QFCvQyK7IsFzqBqFAmqZamWmxqZponCUgYi3ENmYGhw7Rp6Ggw3YhGiB4Ig5kisTxZdF8OUZJiWgIUtmxT8z1jy7R//727PN+bv97pBaDQKq6JyoLSsQ3bpmGexPFgEIVRVuhukhZx1KfNxFijWa8oA9IS5J4VcWrESR4lehzroQ7jJqwqmoZwmd5NuAgEFRGzM0WPwSiHJQrB7ORrYSFNaBV11BxL91297pueYw36Sa8fMgfXFi4rHmLPozBY6iS5XhuZ2rVtz50zs7tWFuePHzmQDHrdOFq9tLBhJKj6HijLl/AK9gLNoU+hLBTicAoiVdFoU3g1DQQCHmYzhDtEClbhVb1MET0odT5Ylu7aYDJqnul7rikM5r4nU6FBp7PeT5wNs+/7yKOP7L7j3ompGdetHD/83Ksv/3Dj3I5N2/ZkSfzakefnz70wmsSmLXqH5on+pdpqR69WR5rNeoG0tVzBzZRkFVxRAww2x2UoZcALiVqWhA5tCxC4J+hSQDXXQsrwao6P1sbGm1EQjTTsRi1N1vpLq9nc3k/c/9FP37bzTsdx0ixlBudOHzl04JmK39i8dXeep8y56lc/+Uj1yc94k2OmIOUwtYtXsy9/NT56yvBdA8srncWQXTUXYVcwgga4eDOQivaXWnATq5iUbg2lbyBoDz9q6J313oWzp2a3TG/dMnH2/Fm/Pvdrv/q597z/Y7phpmmSxBEeYNDvvnH8hTgJ53bthSrRe8ZCmUTxbzoHdRHDgA68XsUBKxDQRkg1IiEO/kTaelrgK7RM/IVosKalQ5sd+iycAGqk44ag0zEN18FhQbTWWY+uXl3ZvHlza2LLvQ+9d27/b/mtqTCK0SnBhEaZ1qV3Xl9cuDQ6MeN5Xs51y0jiJEtjaFOjCIKhGBGdobmOgdWimhIsS5UFtD60+JhOgYswMoQsuqsCgIx2k1elqaipY4kBebZZcXXQOo7tN0ZTb+vcPR/d6t7WG6QAJSLghiQiaXo86IE1yZKZiRlINU1z+frFs68fXJ4/v2cy0XVXoSn1TXiDaSSmdAAlEF4hEEPiM7qqJUDSY2YmpmakWY4Ry6RktrewltI3DczfdUwClecaWOv6enDqfDFz9z3TznbbKUZsK0n0fpAPgiwU3TLnb1xaWbpWq7dc16O7teX5tcV3PvLoY1E4qAZfL4qB2I2SrRouJ3zg8iouvIrPQimEUZErnyEPt4CvykRHyRW4esuzacRYDon7OFSxJ/ChADhI39GhMNEmP3zP79rNfaud0HW1mm/7vm1ZZCD0h/YXN+bfjsLe2Mw2IKVxtL5y5Y79+x9+/LOHnvuOvohzgkZDghMHUcATh0LnjSrxBJUTKXO2yBJkSim806ucxR4JXzxLhCs1SOUugMWr2ZBqa+KqHMNz9SxLB2nr8c/+4W17HgjCSCZO+EmwpxS4vm+iBmvd3tryVRwyvBI/1teut1r1Pe+5x/ereZaeOBEvXFz3wIoQcax60etrb17Q1oP2wmqfuBDTYZrzgZ5rPtruaY6oiWp9k10BXuqy4pX7+CxIdS3TJVNxUZK0EzoPfvzJ/fc9HEdRSTxqJGZuGJ5XQI/j6v2ri0Fv1a/WLNfN8QvR+ty22Y2z2wiuzP6iN3O8vrPwHMlfoAatNIJG5dIHvcHeCQ/CEDSEM4eXLnXPDiIbZw5nkkVIWig3RBUM8gc5iiGvotigRJMwUkvPuqG2bc/DDz76RBbH0pz4LpqihClaispIgOl3FopsUG+Mo4W9oOs41syW2Ypfy1S8d2cn6w9s11pVBgcrfMXX2yNZ8IFs9Vd2jyghy2VMi4zg+uWImKOyR5UIKE1QMaSARzmwfvUmgUB4BautR3Hm1Gcf/dTv2KaJJgBxeICaQ0IGc8NEsjxYdu3cqzfwROGgU6/7k9ObyvboH27HSBMjlsjKRZI3DFsX3sRuJAcT8oj2wiVai99FUXISXFEaU/EDzMyIlXUKr5JJig64eCvUpcjDzH3woU/PbNyaJJHcEnzqrKByIgA6tuO6ccUJm3XH8v2E/DMNmiMbGiPjknGopJ1YYNEvmi6iFIQSGjF5rnBwFlqV9SvLxk74qlIG8a44LeWARYgKhdIBZoEQbVtSliQuWlO33/fAY6jfLaDSNYcaIsvzGOJtLDcZHXEnJ0YCw0n7oWXmrdaI79eAypggkexWnhF0opsEMz5zCZZ5Q4cFrjQRazElmDFJ1g5K19Rgco9Qp8RS6gCXeRbp83ysVR64/+P15ojEz5v4pEFJreqBcXECg07btu2JDdNrfbfbbaPojdaIaVpERnlOVji6acmbkrp8BRj4FVbVtUxAeVjSZfFl3MQ0WMYRI5iaoAY3CiZTHtqW+FchlVy42tpy5/4PlqSqptJM/hVKnpJE2zFJSRcXFhbnr3jNyVartrhwuVpx6s0WvnBImcTzXB9E0IJJih5wI4iZZZDlnVDWh6Wp4h9CyOSruGwWaASFgvQQrKaZu5iWUI/DLW2LSQC2yJPc2rHzfSPjk2kSl6SWQAWq4hVxMkuiKylpd319ZbU9WZ30G0TiXDfdarXO9IczxFhPX/BefQ3wspCB21zDGvtB9JSWffOtNqsGXBLZCfq9FqS91Dh1LeGSePG8kNVOWmRJNDPVwI5Up6XPktgrXk0za7vuuLfksRzy3Wexk5JlIY/0fhCEKZGtbpnVipnpVsX3S6wwQYcP+v5DFaeJvDB55TDnw/Qn9vrohPPAhqr4VyaWSch85uyN7+n3J9uf0L0max8UDF1OoqBz8K9M85Kon5LqMHcRdSh0vzG1acuOoZ9St8EmpArJqJaIgyhgu5bGiiWLmF6sXECrUclNslkyFdoOHYFvGWOm07JsoRQiioKgUo3MlmfN1ByqA9JpVtiG1nQM26gWjUnDHUO8KmTpWhSYdsVhTNFeOYb+lXmTK05MztWbo9K1Okq0fBS46ookYg7GwUFCnRFmcDFREFarrlUBlXjIUglKr4W7vPmwqBy30CLOQjQdStMyoaYVOs01EpZyYkyZxBXzEBnSLU1RJtqxuGFQc3JqzrJthWp4EogKpuKXWUg7wYGqa6lKGomvdsWvVKo+CaGotaiKWBiwRBDDLyIXeYmHEMg3LUA6lkvyJy+aq2/DrwRUMSVuqFvyJnqu22OSg0qj4fG/n4YXUPwwTHECzN7QEvxUrVaR2JiReOJJBGt5CAABd2tkASfHrSvDccoH1GU1NN/LS9KQ8IFPHnYpa0NhVqAaOJ3RoQzVLG+OerOtij2ie6ziqR74BqtcX1KTrNdZa7gteJWkQ/5wF3AqVRmxMrnAY+I+cbi2wdIDkUv+hYWxbqe2wEx1y9UdT5IDmuZ4vIwuZHbclWOYuzAEKyjL86tiBD9/DKel3mAK7mRMwxyfmjKN0PEtsvc4iuM4AR/kpcyFSJknl/udF9LYN230RkJmXqxEydlB75pmkWSSnIiNyAKwOL7Y6xanU/1bpldP44wkETWPozBpXzWmUbYh2GF9QKYoM8Dkfh4rEEuUcl+0EHsKw0TLY6cyMrXZ0e1We2WZp8AjKk9BRtfOvXFw8dLr1p73XW9M4JqUKLgpvmYaq8yKC8AU+QCKa8X4Lv0xLeuszL/6ysENm2+vtybgLnGTE2a/34soTMEBT9/yWaKzjDaEpqgVkDIH8VZIDNqk2GHiWQvKLL2FS426Pr5lDgJcqjSS0kjKeePiievvHLrvQ4/ddd/jlo0X+/8eT/3T3/Qi8/HPfmF60w4MqdNZ+os/+tyNS0dTTRZtgFG8ok64NVYPqaTV7z5KW+EMXRWP9IJUTWOmcaK3O2EU9OqTA782Um00yGXo4ezJFxbePrT37kd33vUI0/p5Gb27Y/l8a6iV5cWn/+XLL7/44mOf/nxrbJqKjunYF8+9QegOo6TAMynNHPIKGkJs0O8qQx12Kpyq/jAP0sVqxSYwBv00UdXJdjdei1ZGZ5Zbo5Ou6wRZeubYs5fPHNl3z8d27HsYoP8X2rvAlbdYEF25eO6F57934NnvVurTTzz5x5u37SFvwEMlUfSz5/8TlSNxG6gUGomXWLmLx4w7ndVbc72FV9hXwxC4Cd/EbHISPAERK+xH/V7YGNHwyldePxDl9fse/Mzu/R8dPvsLb7KG6/dWlheuXDh37MiLb7/5xura+uTMzo/9+hd27n2/5bhpmmH6eN/vf+Pvz711bPfeexcuHhksD8qewKroE+1NVxav/qLU5Iq4IQ5RVgk5fCUl1Vg1gJ5ZZzguM1tyLXdt6cLRF//DIzEo9F4/CPphf9Drddc7a8urK0urq8u9bjfLdNcf2bxt/4Mf/4ONc7vrzQkaU2vCXBzHW19f/vbTXzr002c+9RufX1ldWb58WHyvHGWuTRop4SFbWrgYRQF1PLFYoVo1UWeUAd+eoKpizuIzSFVt03I9CjTR2NSWOz/4uUsXrl5552wQDlzPxpGtrQUR9RMWO07Nr8/M7b5rb3OiOTJVq4+7fs0yHeaZMt2c3M9iqb+6ev3oS99//vtPkcz+9u//6Z3v+/A//90XWYxJ3FJcyRpGOJPVRb66fLmzsjQ+NSOZGYe6BUrJUVItCDIUXRZtgtv0XIoItu/lKIBR3bYJf7u9CMKM+hDuhFwPhVHec9jT0HOJ78JdMeFcSsx53qb48darJw8/f/bNo65Xf+jRJ/bd+0u2V11dW1ycP2+TwKG2Uk26WX+FRqiO+8uXLp6enNlS5vYKrYyEL4yoUElEKXMOKrVEV+qGPVOLdWdbWjTJvOjBccRiw1A6EHCCWcQjCwDeSfuLPA7D9fby5fNvXDh3kvPq8oLr1bbffu9v/t6fTc/uYr3e6barGMDFc8H6jaZyeqUeqhhLB7laCxWDM6+/8t77HhGANw+GZGIyttIKhsQtoDPNVi1OSAoW29f+e3F+vjD8xti0ZXuF7iQpNdqAtJoENwwHfTx6t9NpL62t3GivLGDBlML9amty4233fOiTc9vvGp3cjDJQGFnrLKIUCM51/TOvHzGLLhESfRSKWLnwzovp08IysitvH1+8fmVyw2ZSXlFZhCXuTJIcnpGgwEVZ8+Tj03d4mzY6btFbu9bvzK+tXO8utyyTOnFg2NVOp3vt2o3lVVwFqYBrOnXPH5mcnt2x9wOt8U0jY9ONkSnLcoT5HJRtJJ5mlBbFjhyn0mu3z5896lmUupwsT2jG6EP/SrxOMqlZ9/sLJw7/5LFPPYlh3mRWJsOBDIENXFJB1vJ16m3eBPO2/F07Rx+MQtZJGRlsGW4lj9H1KMolHstiFb8xrKmI92DXQ9bDIV0TQQZBT+wAz6oGoueXD/xX0L4yUiWDkS07IbKMXTRgiRPH8Gy6ZnTy8A+Xb8xLxULZ1q0zzQBKRbJeNeuUBWydZVkah6QrttRxkiSFVOmExdAgZM+jIPay6QDZOG9CTSqvME0jkAkAgaCHYRhFAyVYIY99tCgIjr/8o4odwoOsO8rSnYgWHVAuE3dEVkEFrr924eBPvknapSYpHap/xavUxIlh7MwZJETRgGIcvBSsbLnITEoIfOBiGGe8sPVG1W7WbCr6koiJTqlD+pUKX7/fRlmHjGhaqzV++KUfri+dpWIJnoikSykAzcHKWfwLtZxQseLb6YlXvvvOmZOGI/p062AMSV/YaWQhiRQpEMtOFUU3yZfQH1Y4Mj56b4s7QsvDMO8PWD9mTJyCISUItQ12q0ut11sP40BZj5Baq43cmL/yyoFv1NyYmgFABSteTh0q4ZUcSVZ5Ucw9IamIrv/gm18ZdNfF5ktNUApAzYtKVlamw2p3hnSaxXGMULAJxVwUp1yxHanQYZOULHv9dL1LcCiASw9lGMJIB2G/P+iINxbBivRd2/3Bt/6hCOc9147TIkyKNFELXigo2AesNGCDujtLRlkhienIxsby4rVeP96z736ZklqUU0RyqAQVuYxH7FA1lSQpgohqIrVO2aBDXlEoO29MybJMqKI+gvvHHhgFr0xn3MXpxkm01l6KE2xfgJqmPT624dnvPvXmkWdGG7INisaTkYdJ3h0Amlaa6QrWguE9h5xWvBQ4bJvNwezi+TOGVdu+e7/Ka6U2g7PDpFlmIWmaAQIRR5iK4BDRE6mQGhMAEFegFj5lvU9YZheO9QpyyrU4zVbXrkdxKEFMgFoT45uOHPzxT7/3ldFabJpOEOZBDFCknVNgpQrBIbzyRq4MtaiMLFCGBkSMjt86fdL1R+e23yHkiPdTBVqEyxLANJKsYGsTKExa7dBiK6JIYmFlqUq1pBn4FLOF45DkZ+vd1W6/SxtaWqY9ObHp5JEXv/P0X7a8vuu4gzgbwGgEnVS9tUEsEZ7JgrWupIMHkDgr61v546AiDd3hqZNHTNvfumMfRsxq3WP3EDnGFIVAruOk8ASIFs8AkTCK3mPWokvskNEglRK7BCOokHNGjTlNCWokLoXr+ePjmw7/7EfffurPW26v4ntBlPdDfEheGg8SCGMtUT84oLTRQDo4UwSICig75g0Sh3AdMzx14tCg179t936/wu6shFz2iOGVFxIX7TQpMuP+hWlWOGgtD7MWR4OhmMb0BotR0o8Ga6aZsCcFmkp1vOK1fvTMvz737b8eq8aViofoYTTAG7JMQpdSIlQepSiSToQ3SW1QUoa3JGoIQNEEtIvOuY8qOw4VoGvnjyxdfWvL3M6xiQ0sNMFHLsuM8BuJ1AYk9SmnihcTy0abdKKlbEHCaxhHvd7aoL8G0bJB7lT8+vTywo2n//FPXnv561OjFiuzQYToxVIxgIiAi4hwv5kWZ7htYdF0PIp7yJPtJrDKv1CreEXp1FeUwZoac2v6dbe47Pq+4UzabgW9plofBPzOQsIg6QNPUs5gomxpQy2Mor4sw5Kk224v9QckImatRuVzynUqLz//79/46hc7i6fZ8kUwfYCK6DPUFKBRIls00g/7c9K/xAHTFl5BhxMQrCVQSBGgwjH2rrEqvG1z8yMPbB+pJm+/9tzlc8ebo2N+fTLOXbIIqcLIIbMkHFDEQMmIt0EYkDpFYcd1igqa6I3Um1PY55VzR8+f/Nq5Y1/TsoFhV9BONvewejRVRD8EmknPKHqBDFmMCLhhIAUT9Eh1RPZZuYFtmypjlqIIKQ6lYAzjjdOXmq1GEZ+7dvJvTx8eC/VtzZm7JzbuqtZHTEruLAIkjJLni4sypHLokZJ7ZKNG3Lv89qGDPz519Plg/Z27947duXv62BsLK71gEGb9EHsHJT6ESSIWAYqgZLtPORmhQWEVOsXJqAqwxjCSg0MwgsV95ZEsHSRqtdfDziC3K/xew/XMYKX9+pm3f3r24pcKZ3JseufG2Z2bNs7Obtk4MTWJ3SyvdLu4pc5K0LlSs1Y3TmYLl86cOXk6ilEff2E5uWtfa9NMcHUxxAGHsUhfWZIAJXxgDzg6ASq5tRKZYBWogrrMqWnEFoTQCjPiYwyP3/1QHPDddifIdWdhJUqj0NuzwXKrptWfnHD6Qbdz7dDyxQOnjGJqrHrnHXOMcfy1CzdW+mQwjZp51+7JPVPbalvclZWpMxf764MMZ7yy1t26ZfzM+dVrSxFJotgov+CQX5ugoyVQngaoYJMDXNiWsl+UUwK/cgMKPcahdkqoDGyc9G/fMbGy2tMN+613VtbWIxxts1ltd5P5ZRwlqy/X8/x6vdqoOfxEhh9GIFam6VaqFZ+9V61R9xr1yko7WGCnMciIi9UK6YE9PjFyfam/uBqhCfAqQCVvk7CnGBUmyTJLOpW+ivsTtkHK5gKCV7RK9i8Pi96I11xY6jdHmgQ9FoBhInkJ/bIOG4TMCh0lcuqUDVHTMNWYxlovgS1+4sGPdAjoCzfW6brZ9NZ6HeTeaLZOnLpG4Z4opVyp+GxxxIpU8EqchE2BJaRygFW+wKksp8TqpZEUd9SeDe+o/Hova/eSxbVwEGu3bZs69tolSi8EQ5vapMCVWEWyQneu50IDsJbbUW+QkyslhD629Ayj0+1R8JmbHcWD9np9KphBbBw7cXWxk+EH3g1UlgiCRSmrAqpg6/8DnlUhNsYFwKsAAAAASUVORK5CYII=";

// render/demos/sl-chrome.ts
var DEFAULT_HELP = [
  { title: "3D view", rows: [
    ["Left-drag", "Rotate"],
    ["Right-drag", "Zoom"],
    ["Middle / Shift+Left-drag", "Pan"],
    ["Wheel / two-finger", "Zoom (dolly)"],
    ["Double-click", "Maximize / restore"],
    ["Shift + move", "Pick \u2192 jump slices to the point"]
  ] },
  { title: "Endovascular flight (fly-inside / endo demo)", rows: [
    ["Up / Down", "Move in / out along the view axis"],
    ["Left / Right", "Yaw"],
    ["Shift + Left/Right", "Pitch"],
    ["Ctrl + Left/Right", "Roll"],
    ["Space", "Toggle forward cruise"],
    ["Shift + Space", "Toggle reverse cruise"],
    ["Escape", "Stop"],
    ["Left-drag", "Look around"],
    ["Shift + click", "Autopilot target"],
    ["Speed slider", "Travel speed in mm/s (live, applies mid-flight)"]
  ] },
  { title: "Slice views", rows: [
    ["Wheel / Left-drag", "Scroll through slices"],
    ["Right-drag / \u2318-wheel", "Zoom this slice"],
    ["Middle / Shift+Left-drag", "Pan"],
    ["Double-click", "Maximize / restore"],
    ["R", "Reset pan/zoom"],
    ["Shift + move", "Jump the other views to the point under the cursor"]
  ] }
];
function glass(el2, extra = "") {
  el2.style.cssText += ";background:linear-gradient(135deg,rgba(58,64,88,.55),rgba(20,24,38,.66));backdrop-filter:blur(20px) saturate(1.6);-webkit-backdrop-filter:blur(20px) saturate(1.6);border:1px solid rgba(255,255,255,.2);box-shadow:0 18px 50px rgba(0,0,0,.55);" + extra;
}
var helpEl = null;
function escClose(e) {
  if (e.key === "Escape") closeChromeHelp();
}
function closeChromeHelp() {
  if (helpEl) {
    helpEl.remove();
    helpEl = null;
    document.removeEventListener("keydown", escClose, true);
  }
}
function openChromeHelp(host = document.body, sheets = DEFAULT_HELP, actions = []) {
  if (helpEl) return;
  helpEl = document.createElement("div");
  helpEl.style.cssText = "position:fixed;inset:0;z-index:96;display:flex;align-items:center;justify-content:center;background:rgba(6,8,14,.55);font:13px/1.5 -apple-system,system-ui,sans-serif;color:#e8eeff;";
  helpEl.addEventListener("mousedown", (e) => {
    if (e.target === helpEl) closeChromeHelp();
  });
  const panel = document.createElement("div");
  panel.style.cssText = "max-width:min(640px,92vw);max-height:86vh;overflow-y:auto;padding:22px 26px;border-radius:16px;color:#eaf0ff;";
  glass(panel);
  panel.innerHTML = `<div style="font:800 20px -apple-system,system-ui,sans-serif;margin-bottom:4px">SlicerLive \u2014 controls</div>`;
  for (const sec of sheets) {
    const rows = sec.rows.map(([k, d]) => `<div style="font:600 12px ui-monospace,Menlo,monospace;color:#fff5d6;white-space:nowrap">${k}</div><div style="color:rgba(232,238,255,.85)">${d}</div>`).join("");
    panel.innerHTML += `<div style="margin-top:14px;padding:12px 14px;border-radius:10px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08)"><div style="font:700 11px -apple-system,system-ui,sans-serif;letter-spacing:1.1px;text-transform:uppercase;color:#9fe9ff;margin-bottom:9px">${sec.title}</div><div style="display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;align-items:baseline">${rows}</div></div>`;
  }
  if (actions.length) {
    const actionsRow = document.createElement("div");
    actionsRow.style.cssText = "margin-top:16px;display:flex;flex-wrap:wrap;gap:8px;align-items:center";
    for (const action of actions) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = action.label;
      btn.disabled = Boolean(action.disabled);
      if (action.title) btn.title = action.title;
      btn.style.cssText = "cursor:pointer;padding:7px 12px;border-radius:8px;border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.08);color:#eaf0ff;font:650 12px/1 -apple-system,system-ui,sans-serif;";
      if (btn.disabled) {
        btn.style.opacity = "0.45";
        btn.style.cursor = "not-allowed";
      }
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (btn.disabled) return;
        closeChromeHelp();
        action.onClick();
      });
      actionsRow.append(btn);
    }
    panel.append(actionsRow);
  }
  const dismiss = document.createElement("div");
  dismiss.style.cssText = "margin-top:16px;font-size:12px;color:rgba(232,238,255,.55)";
  dismiss.innerHTML = `Press <b style="color:#fff5d6">esc</b> or click outside to dismiss.`;
  panel.append(dismiss);
  helpEl.appendChild(panel);
  host.appendChild(helpEl);
  document.addEventListener("keydown", escClose, true);
}
function installChrome(opts) {
  const controls = opts.controls ?? [];
  const host = opts.container ?? document.body;
  const help = opts.help === false ? DEFAULT_HELP : opts.help ?? DEFAULT_HELP;
  const openHelp = () => openChromeHelp(host, help);
  let helpBtn = null;
  if (opts.help !== false) {
    helpBtn = document.createElement("button");
    helpBtn.textContent = "?";
    helpBtn.title = "Controls & key bindings";
    helpBtn.style.cssText = "position:fixed;top:12px;left:12px;z-index:74;width:32px;height:32px;padding:0;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;border-radius:50%;color:#cfe6ff;font:700 15px -apple-system,system-ui,sans-serif;";
    glass(helpBtn);
    helpBtn.onclick = openHelp;
    host.appendChild(helpBtn);
  }
  const logo = document.createElement("div");
  logo.id = "sl-badge";
  logo.title = "SlicerLive \u2014 visualization";
  logo.style.cssText = "position:fixed;z-index:74;cursor:pointer;user-select:none;display:flex;flex-direction:column;align-items:center;gap:4px;padding:7px 12px 6px;border-radius:14px;background:#121826;border:1px solid rgba(255,255,255,.12);box-shadow:0 10px 30px rgba(0,0,0,.55),inset 0 1px 0 rgba(255,255,255,.06);transition:transform 120ms ease-out;";
  const mark = document.createElement("img");
  mark.src = SL_LOGO;
  mark.alt = "SlicerLive";
  mark.style.cssText = "height:40px;width:auto;display:block;filter:drop-shadow(0 0 5px rgba(255,200,80,.5));";
  const word = document.createElement("div");
  word.innerHTML = 'Slicer<b style="color:#ffd34d">Live</b>';
  word.style.cssText = "font:800 12px/1 -apple-system,system-ui,sans-serif;letter-spacing:.5px;color:#eef7ff;text-shadow:0 0 14px rgba(255,210,90,.4);";
  logo.appendChild(mark);
  logo.appendChild(word);
  host.appendChild(logo);
  const place = () => {
    const a = opts.anchor;
    const r = a && a.getClientRects().length ? a.getBoundingClientRect() : null;
    if (r && r.width > 2 && r.height > 2) {
      logo.style.top = Math.round(r.top + 8) + "px";
      logo.style.right = Math.round(globalThis.innerWidth - r.right + 8) + "px";
    } else {
      logo.style.top = "10px";
      logo.style.right = "12px";
    }
  };
  place();
  requestAnimationFrame(place);
  globalThis.addEventListener("resize", place);
  const anchorRO = opts.anchor && "ResizeObserver" in globalThis ? new ResizeObserver(place) : null;
  anchorRO?.observe(opts.anchor);
  const pop = document.createElement("div");
  pop.id = "sl-popup";
  pop.style.cssText = "position:fixed;z-index:73;min-width:210px;max-width:300px;max-height:84vh;overflow-y:auto;padding:10px 12px;border-radius:12px;color:#eaf0ff;font:13px -apple-system,system-ui,sans-serif;opacity:0;pointer-events:none;transform:translateY(-6px);transition:opacity 120ms ease-out,transform 120ms ease-out;";
  glass(pop);
  host.appendChild(pop);
  const paintSw = (sw, on2) => {
    sw.style.background = on2 ? "linear-gradient(180deg,#9fe9ff,#54c6f0)" : "rgba(255,255,255,.18)";
    sw.innerHTML = `<span style="position:absolute;top:2px;left:${on2 ? 17 : 2}px;width:15px;height:15px;border-radius:50%;background:#fff;transition:left 120ms;box-shadow:0 1px 3px rgba(0,0,0,.4)"></span>`;
  };
  const afterPaint = (fn2) => requestAnimationFrame(() => requestAnimationFrame(fn2));
  const paintTri = (box, level, color) => {
    const pct = Math.round(level * 100);
    const c = `rgb(${Math.round(color[0] * 255)},${Math.round(color[1] * 255)},${Math.round(color[2] * 255)})`;
    box.style.opacity = level < 0.02 ? "0.75" : "1";
    box.innerHTML = `<span style="position:absolute;left:0;top:0;bottom:0;width:${pct}%;background:${c};opacity:.9"></span><span style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font:700 10px -apple-system,system-ui,sans-serif;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.75)">${pct}%</span>`;
  };
  const triNext = (v) => v > 0.66 ? 0.5 : v > 0.04 ? 0 : 1;
  const attachOpacity = (box, get, set, color, onChange) => {
    box.style.cursor = "ew-resize";
    box.title = "Click: 100% \u2192 50% \u2192 off \xB7 Drag sideways for a live opacity slider";
    const paint = () => paintTri(box, get(), color);
    paint();
    let startX = 0, startV = 0, dragged = false, id = -1;
    box.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      startX = e.clientX;
      startV = get();
      dragged = false;
      id = e.pointerId;
      try {
        box.setPointerCapture(id);
      } catch {
      }
    });
    box.addEventListener("pointermove", (e) => {
      if (id < 0) return;
      const dx = e.clientX - startX;
      if (Math.abs(dx) > 3) dragged = true;
      if (dragged) {
        set(Math.max(0, Math.min(1, startV + dx / 130)));
        paint();
        onChange();
      }
    });
    const end = () => {
      if (id < 0) return;
      if (!dragged) {
        set(triNext(get()));
        paint();
        onChange();
      }
      try {
        box.releasePointerCapture(id);
      } catch {
      }
      id = -1;
    };
    box.addEventListener("pointerup", end);
    box.addEventListener("pointercancel", end);
    return paint;
  };
  const OPBOX_CSS = "width:44px;height:18px;border-radius:6px;position:relative;overflow:hidden;flex:0 0 auto;background:rgba(255,255,255,.14);box-shadow:inset 0 0 0 1px rgba(255,255,255,.18);touch-action:none;";
  const heading = (text, first) => {
    const h = document.createElement("div");
    h.textContent = text;
    h.style.cssText = "font:700 10px -apple-system,system-ui,sans-serif;letter-spacing:1.1px;text-transform:uppercase;color:#9fe9ff;margin:" + (first ? "0 0 8px" : "12px 0 6px") + ";" + (first ? "" : "border-top:1px solid rgba(255,255,255,.12);padding-top:10px;");
    pop.appendChild(h);
  };
  const selects = opts.selects ?? [];
  const selEls = [];
  let sectionSeen = null;
  let firstHead = true;
  for (const c of selects) {
    const sec = c.section ?? "Visualization";
    if (sec !== sectionSeen) {
      heading(sec, firstHead);
      sectionSeen = sec;
      firstHead = false;
    }
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:5px 0;";
    const lab = document.createElement("span");
    lab.textContent = c.label;
    const sel = document.createElement("select");
    sel.style.cssText = "flex:1 1 auto;max-width:60%;border-radius:7px;padding:4px 6px;cursor:pointer;font:500 12px -apple-system,system-ui,sans-serif;color:#e8eeff;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.20);";
    for (const o of c.options) {
      const op = document.createElement("option");
      op.value = o.value;
      op.textContent = o.label;
      op.style.cssText = "background:#1b2030;color:#e8eeff;";
      sel.appendChild(op);
    }
    sel.value = c.get();
    sel.onclick = (e) => e.stopPropagation();
    sel.onchange = () => {
      c.set(sel.value);
      opts.onChange?.();
      refresh();
    };
    row.appendChild(lab);
    row.appendChild(sel);
    pop.appendChild(row);
    selEls.push({ c, el: sel });
  }
  const rows = [];
  if (controls.length) {
    for (const c of controls) {
      const sec = c.section ?? "Visualization";
      if (sec !== sectionSeen) {
        heading(sec, firstHead);
        sectionSeen = sec;
        firstHead = false;
      }
      const row = document.createElement("div");
      row.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:14px;padding:5px 0;";
      if (c.slider) {
        row.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:6px 0;";
        const top = document.createElement("div");
        top.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
        const lab2 = document.createElement("span");
        lab2.textContent = c.label;
        const val = document.createElement("span");
        val.style.cssText = "font:600 11px ui-monospace,Menlo,monospace;color:#9fe9ff;font-variant-numeric:tabular-nums;";
        top.appendChild(lab2);
        top.appendChild(val);
        const inp = document.createElement("input");
        inp.type = "range";
        inp.min = String(c.slider.min);
        inp.max = String(c.slider.max);
        inp.step = String(c.slider.step ?? 1);
        inp.value = String(c.slider.get());
        inp.style.cssText = "width:100%;accent-color:#54c6f0;cursor:pointer;";
        const fmt = c.slider.format ?? ((v) => String(Math.round(v)));
        const paint = () => {
          val.textContent = fmt(c.slider.get());
        };
        inp.oninput = () => {
          c.slider.set(parseFloat(inp.value));
          paint();
          opts.onChange?.();
        };
        inp.onpointerdown = (e) => e.stopPropagation();
        paint();
        row.appendChild(top);
        row.appendChild(inp);
        pop.appendChild(row);
        rows.push({ c, row, repaint: () => {
          inp.value = String(c.slider.get());
          paint();
        } });
        continue;
      }
      const lab = document.createElement("span");
      lab.textContent = c.label;
      row.appendChild(lab);
      if (c.button) {
        const pill = document.createElement("span");
        pill.style.cssText = "max-width:60%;border-radius:7px;padding:4px 10px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font:600 12px -apple-system,system-ui,sans-serif;color:#eaf0ff;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.20);";
        pill.textContent = c.button.text();
        pill.onclick = (e) => {
          e.stopPropagation();
          c.button.run();
        };
        pill.onpointerdown = (e) => e.stopPropagation();
        row.appendChild(pill);
        pop.appendChild(row);
        rows.push({ c, row, repaint: () => {
          pill.textContent = c.button.text();
        } });
        continue;
      }
      if (c.getOpacity && c.setOpacity) {
        const box = document.createElement("span");
        box.style.cssText = OPBOX_CSS;
        row.appendChild(box);
        const paint = attachOpacity(box, c.getOpacity, (o) => c.setOpacity(o), c.color ?? [0.62, 0.9, 1], () => opts.onChange?.());
        rows.push({ c, row, repaint: paint });
      } else {
        row.style.cursor = "pointer";
        const sw = document.createElement("span");
        sw.style.cssText = "width:34px;height:19px;border-radius:999px;position:relative;transition:background 120ms;flex:0 0 auto;";
        row.appendChild(sw);
        row.onclick = () => {
          if (c.disabled?.()) return;
          const next = !c.get();
          paintSw(sw, next);
          afterPaint(() => {
            c.set(next);
            opts.onChange?.();
            refresh();
          });
        };
        rows.push({ c, row, sw });
      }
      pop.appendChild(row);
    }
  } else if (opts.about === false && !opts.segments && !selects.length) {
    pop.textContent = "SlicerLive \u2014 WebGPU renderer";
  }
  const segHost = document.createElement("div");
  pop.appendChild(segHost);
  const segRows = [];
  function buildSegments() {
    const S = opts.segments;
    segRows.length = 0;
    segHost.innerHTML = "";
    if (!S) return;
    const list = S.list();
    if (!list.length) return;
    const wrap = document.createElement("div");
    wrap.style.cssText = "margin-top:6px;border-top:1px solid rgba(255,255,255,.12);padding-top:6px;" + (list.length > 6 ? "max-height:210px;overflow-y:auto;" : "");
    for (const s of list) {
      const row = document.createElement("div");
      row.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:4px 2px;";
      const left = document.createElement("span");
      left.style.cssText = "display:flex;align-items:center;gap:8px;min-width:0;";
      const swatch = document.createElement("span");
      swatch.style.cssText = `flex:0 0 auto;width:11px;height:11px;border-radius:3px;box-shadow:0 0 0 1px rgba(255,255,255,.25);background:rgb(${Math.round(s.color[0] * 255)},${Math.round(s.color[1] * 255)},${Math.round(s.color[2] * 255)})`;
      const lab = document.createElement("span");
      lab.textContent = s.name;
      lab.style.cssText = "font:500 12.5px -apple-system,system-ui,sans-serif;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
      left.appendChild(swatch);
      left.appendChild(lab);
      const box = document.createElement("span");
      box.style.cssText = OPBOX_CSS;
      row.appendChild(left);
      row.appendChild(box);
      const paint = attachOpacity(box, () => S.get(s.num), (o) => {
        if (!(S.enabled && !S.enabled())) S.set(s.num, o);
      }, s.color, () => opts.onChange?.());
      wrap.appendChild(row);
      segRows.push({ num: s.num, box, color: s.color, paint });
    }
    segHost.appendChild(wrap);
    paintSegments();
  }
  function paintSegments() {
    const S = opts.segments;
    if (!S) return;
    const dis = S.enabled ? !S.enabled() : false;
    segHost.style.opacity = dis ? "0.4" : "1";
    for (const r of segRows) r.paint();
  }
  if (opts.about !== false) {
    const about = document.createElement("div");
    const aLabel = opts.about?.label ?? "About SlicerLive";
    const aURL = opts.about?.url ?? "https://github.com/pieper/SlicerLive";
    about.textContent = aLabel;
    about.style.cssText = "cursor:pointer;border-radius:9px;padding:9px 8px 3px;margin-top:4px;" + (controls.length || opts.segments ? "border-top:1px solid rgba(255,255,255,.12);" : "") + "font:600 13px -apple-system,system-ui,sans-serif;color:#9fe9ff;";
    about.onmouseenter = () => {
      about.style.background = "rgba(255,255,255,.07)";
    };
    about.onmouseleave = () => {
      about.style.background = "transparent";
    };
    about.onclick = (e) => {
      e.stopPropagation();
      globalThis.open(aURL, "_blank", "noopener");
    };
    pop.appendChild(about);
  }
  function refresh() {
    for (const { c, el: el2 } of selEls) {
      const v = c.get();
      if (el2.value !== v) el2.value = v;
    }
    for (const { c, row, sw, repaint } of rows) {
      const dis = c.disabled?.() ?? false;
      row.style.opacity = dis ? "0.4" : "1";
      if (repaint) {
        repaint();
        continue;
      }
      const on2 = c.get();
      row.style.cursor = dis ? "default" : "pointer";
      sw.style.background = on2 ? "linear-gradient(180deg,#9fe9ff,#54c6f0)" : "rgba(255,255,255,.18)";
      sw.innerHTML = `<span style="position:absolute;top:2px;left:${on2 ? 17 : 2}px;width:15px;height:15px;border-radius:50%;background:#fff;transition:left 120ms;box-shadow:0 1px 3px rgba(0,0,0,.4)"></span>`;
    }
    paintSegments();
  }
  refresh();
  const show = () => {
    buildSegments();
    refresh();
    const b = logo.getBoundingClientRect();
    pop.style.top = Math.round(b.bottom + 6) + "px";
    pop.style.right = Math.round(globalThis.innerWidth - b.right) + "px";
    pop.style.opacity = "1";
    pop.style.pointerEvents = "auto";
    pop.style.transform = "translateY(0)";
  };
  const hide = () => {
    pop.style.opacity = "0";
    pop.style.pointerEvents = "none";
    pop.style.transform = "translateY(-6px)";
  };
  let pinned = false;
  logo.onmouseenter = () => {
    logo.style.transform = "scale(1.08)";
    show();
  };
  logo.onclick = () => {
    pinned = !pinned;
    pinned ? show() : hide();
  };
  logo.onmouseleave = () => {
    logo.style.transform = "scale(1)";
    if (!pinned) setTimeout(() => {
      if (!pop.matches(":hover") && !pinned) hide();
    }, 120);
  };
  pop.onmouseleave = () => {
    if (!pinned) hide();
  };
  const destroy = () => {
    globalThis.removeEventListener("resize", place);
    anchorRO?.disconnect();
    closeChromeHelp();
    helpBtn?.remove();
    logo.remove();
    pop.remove();
  };
  return { refresh, destroy, openHelp };
}

// render/recorder.ts
var clone = (v) => structuredClone(v);
function seekFrames(frames, t) {
  let base = -1;
  for (let i = 0; i < frames.length; i++) {
    if (frames[i].t > t) break;
    if (frames[i].k === "key" || frames[i].k === "reset") base = i;
  }
  const nodes = /* @__PURE__ */ new Map();
  let start = 0;
  if (base >= 0) {
    const b = frames[base];
    if (b.k === "key") for (const [id, n] of Object.entries(b.nodes)) nodes.set(id, clone(n));
    start = base + 1;
  }
  for (let i = start; i < frames.length; i++) {
    const e = frames[i];
    if (e.t > t) break;
    if (e.k === "up") nodes.set(e.id, clone(e.node));
    else if (e.k === "rm") nodes.delete(e.id);
    else if (e.k === "reset") nodes.clear();
    else if (e.k === "key") {
      nodes.clear();
      for (const [id, n] of Object.entries(e.nodes)) nodes.set(id, clone(n));
    }
  }
  return nodes;
}
function nearestThumbOf(thumbs, t) {
  let best, bestD = Infinity;
  for (const th of thumbs) {
    const d = Math.abs(th.t - t);
    if (d < bestD) {
      bestD = d;
      best = th;
    }
  }
  return best;
}

// render/commits.ts
function canonicalize(v) {
  if (v === null || typeof v === "boolean" || typeof v === "number") return JSON.stringify(v);
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(canonicalize).join(",") + "]";
  if (typeof v === "object") {
    const o = v;
    const keys = Object.keys(o).filter((k) => o[k] !== void 0).sort();
    return "{" + keys.map((k) => JSON.stringify(k) + ":" + canonicalize(o[k])).join(",") + "}";
  }
  throw new Error("cannot canonicalize " + typeof v);
}
async function sha256Hex(s) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
async function hashCommit(c) {
  const { hash: _omit, ...rest } = c;
  return "sha256-" + await sha256Hex(canonicalize(rest));
}
async function sealStream(deltas, opts = {}) {
  const interval = opts.intervalMs ?? 1e3;
  const iso = opts.isoFromMs ?? ((ms2) => new Date(ms2).toISOString());
  const commits = [];
  let parent = null;
  let bundle = [];
  let windowStart = null;
  const seal = async () => {
    const last = bundle[bundle.length - 1];
    const c = {
      parents: parent ? [parent] : [],
      ...commits.length === 0 && opts.base ? { base: opts.base } : {},
      t: iso(last.t ?? windowStart ?? 0),
      ...opts.author ? { author: opts.author } : {},
      ...opts.role ? { role: opts.role } : {},
      ops: bundle
    };
    const hash = await hashCommit(c);
    commits.push({ ...c, hash });
    parent = hash;
    bundle = [];
    windowStart = null;
  };
  for (const d of deltas) {
    const t = d.t ?? windowStart ?? 0;
    if (windowStart !== null && t - windowStart >= interval) await seal();
    if (windowStart === null) windowStart = t;
    bundle.push(d);
  }
  if (bundle.length) await seal();
  return commits;
}
function branchPointAtIndex(commits, globalIndex) {
  let cum = 0, baseCommit = "", baseCum = 0;
  for (const c of commits) {
    const next = cum + c.ops.length;
    if (next <= globalIndex) {
      baseCommit = c.hash;
      baseCum = next;
      cum = next;
    } else break;
  }
  return { commit: baseCommit, offset: globalIndex - baseCum };
}
function branchPointAtTime(commits, tMs) {
  let n = 0;
  for (const c of commits) for (const op of c.ops) {
    if ((op.t ?? 0) <= tMs) n++;
    else return branchPointAtIndex(commits, n);
  }
  return branchPointAtIndex(commits, n);
}
async function verifyChain(commits) {
  let parent = null;
  for (let i = 0; i < commits.length; i++) {
    const recomputed = await hashCommit(commits[i]);
    if (recomputed !== commits[i].hash) return { ok: false, badAt: i, reason: "hash mismatch (content altered)" };
    const expectedParents = parent ? [parent] : [];
    if (canonicalize(commits[i].parents) !== canonicalize(expectedParents)) return { ok: false, badAt: i, reason: "broken parent link" };
    parent = commits[i].hash;
  }
  return { ok: true };
}

// render/recording.ts
function reduceEvent(nodes, ev) {
  const t = ev.t;
  switch (ev.event) {
    case "NodeAdded": {
      if (!ev.node) return null;
      const node = structuredClone(ev.node);
      nodes.set(node.id, node);
      return { t, k: "up", id: node.id, node };
    }
    case "NodeRemoved": {
      const id = ev.sourceId;
      nodes.delete(id);
      return { t, k: "rm", id };
    }
    case "CameraModified": {
      const id = ev.sourceId;
      const node = nodes.get(id);
      if (!node) return null;
      for (const k of ["position", "focalPoint", "viewUp", "viewAngle", "parallelScale"]) {
        if (k in ev) node[k] = ev[k];
      }
      return { t, k: "up", id, node: structuredClone(node) };
    }
    case "SegmentationDisplayModified": {
      const id = ev.sourceId;
      const node = nodes.get(id);
      if (!node || !ev.display) return null;
      for (const k of ["visible", "opacity", "fill2D", "outline2D", "segments"]) {
        if (k in ev.display) node[k] = ev.display[k];
      }
      return { t, k: "up", id, node: structuredClone(node) };
    }
    case "SceneClosed":
      nodes.clear();
      return { t, k: "reset" };
    default:
      return null;
  }
}
var Recording = class _Recording {
  // segment-editor intent strokes (M1b), for the replay stroke overlay
  constructor(base, session) {
    this.base = base;
    this.session = session;
  }
  // git-style history over the event stream (see commits.ts). Present when the recording was sealed on
  // disk (render/tools/seal-recording.ts); otherwise computed deterministically on load, so every
  // recording "has" a verifiable commit chain and a stable head hash.
  commits = [];
  headCommit;
  // NB: distinct from head() (the timeline's max frame time)
  rootCommit;
  strokes = [];
  /** Fetch + compile a finalized recording at `base` (…/mrson/rec/<name>/). */
  static async load(base) {
    if (!base.endsWith("/")) base += "/";
    const man = await (await fetch(base + "recording.json")).json();
    const keyDocs = await Promise.all(
      man.keyframes.map((k) => fetch(base + k.scene).then((r) => r.json()))
    );
    const items = [];
    man.keyframes.forEach((k, i) => items.push({ t: k.t, key: keyDocs[i].nodes }));
    for (const e of man.events) items.push({ t: e.t, ev: e });
    items.sort((a, b) => a.t - b.t || (a.key ? 0 : 1) - (b.key ? 0 : 1));
    const frames = [];
    const cur = /* @__PURE__ */ new Map();
    for (const it of items) {
      if (it.key) {
        frames.push({ t: it.t, k: "key", nodes: structuredClone(it.key) });
        cur.clear();
        for (const [id, n] of Object.entries(it.key)) cur.set(id, structuredClone(n));
      } else if (it.ev) {
        const fr2 = reduceEvent(cur, it.ev);
        if (fr2) frames.push(fr2);
      }
    }
    const thumbs = man.thumbs.map((th) => ({ t: th.t, url: base + th.file }));
    const session = { id: man.id, startedAt: man.startedAt, frames, thumbs, marks: man.marks ?? [] };
    const rec = new _Recording(base, session);
    rec.strokes = man.events.filter((e) => e.event === "SegEdit" && e.edit).map((e) => ({ t: e.t, edit: e.edit }));
    rec.commits = man.commits && man.commits.length ? man.commits : await sealStream(man.events ?? [], { intervalMs: 1e3, role: "module" });
    rec.rootCommit = man.root ?? rec.commits[0]?.hash;
    rec.headCommit = man.head ?? rec.commits[rec.commits.length - 1]?.hash;
    return rec;
  }
  // Same query surface as SceneRecorder, so the timeline UI drives either interchangeably.
  seek(t) {
    return seekFrames(this.session.frames, t);
  }
  nearestThumb(t) {
    return nearestThumbOf(this.session.thumbs, t);
  }
  head() {
    const f = this.session.frames;
    return f.length ? f[f.length - 1].t : this.session.startedAt;
  }
  span() {
    return [this.session.startedAt, this.head()];
  }
  /** Times of every recorded frame (keyframe/delta), ascending — used to skip idle gaps in playback. */
  frameTimes() {
    return this.session.frames.map((f) => f.t);
  }
  /** Intent strokes committed within the last `windowMs` up to `t` — for the fading replay overlay. */
  strokesInWindow(t, windowMs) {
    return this.strokes.filter((s) => s.t <= t && s.t > t - windowMs);
  }
  /** blobBase for a LiveScene replaying this recording (so ImageField/zarr fetch the recording's blobs). */
  blobBase() {
    return this.base;
  }
  // ── git-style history ────────────────────────────────────────────────────
  /** The `(commit, offset)` branch point at a timeline position — the address to fork from here. */
  branchPointAt(tMs) {
    return branchPointAtTime(this.commits, tMs);
  }
  /** Re-hash the commit chain: detects any altered delta (integrity). */
  verify() {
    return verifyChain(this.commits);
  }
};

// render/vtk-interactor.ts
var MOTION_FACTOR = 10;
var MOUSE_WHEEL_MOTION_FACTOR = 1;
function actionForButton(button, m = {}) {
  const shift = !!m.shift, ctrl = !!m.ctrl, alt = !!m.alt;
  if (button === 0) {
    if (shift && ctrl) return "scale";
    if (ctrl) return "spin";
    if (shift) return "translate";
    return "rotate";
  }
  if (button === 1) return "translate";
  if (button === 2) return "scale";
  return "none";
}
var CameraInteractor = class _CameraInteractor {
  camera;
  action = "none";
  prev = null;
  // previous position, VTK display coords
  onChange;
  constructor(camera, onChange) {
    this.camera = camera;
    this.onChange = onChange;
  }
  /** Convert browser (cssX, cssY within the view) to VTK display coords (y up). */
  static toDisplay(cssX, cssY, height) {
    return [cssX, height - cssY];
  }
  start(button, cssX, cssY, height, m = {}) {
    this.action = actionForButton(button, m);
    this.prev = _CameraInteractor.toDisplay(cssX, cssY, height);
  }
  end() {
    this.action = "none";
    this.prev = null;
  }
  /** Mouse move while dragging. width/height are the view size in CSS pixels. */
  move(cssX, cssY, width, height) {
    if (this.action === "none" || !this.prev) return;
    const [x, y2] = _CameraInteractor.toDisplay(cssX, cssY, height);
    const dx = x - this.prev[0];
    const dy = y2 - this.prev[1];
    if (dx === 0 && dy === 0) return;
    switch (this.action) {
      case "rotate":
        this.rotate(dx, dy, width, height);
        break;
      case "translate":
        this.camera.panByDisplayDelta(dx, dy, width, height);
        break;
      case "scale":
        this.scale(dy, height);
        break;
      case "spin":
        this.spin(x, y2, this.prev[0], this.prev[1], width, height);
        break;
    }
    this.prev = [x, y2];
    this.onChange?.();
  }
  /** vtkMRMLCameraWidget::ProcessRotate */
  rotate(dx, dy, width, height) {
    const deltaAzimuth = -20 / width;
    const deltaElevation = -20 / height;
    const rxf = dx * deltaAzimuth * MOTION_FACTOR;
    const ryf = dy * deltaElevation * MOTION_FACTOR;
    this.camera.azimuth(rxf);
    const rotatedUp = this.camera.elevation(ryf);
    this.camera.orthogonalizeViewUp(rotatedUp);
  }
  /** vtkMRMLCameraWidget::ProcessScale — note the sign flip vs plain VTK. */
  scale(dy, height) {
    const centerY = height / 2;
    const dyf = MOTION_FACTOR * dy / centerY;
    this.camera.dolly(Math.pow(1.1, -dyf));
  }
  /** vtkMRMLCameraWidget::ProcessSpin — roll about the view plane normal. */
  spin(x, y2, px, py, width, height) {
    const cx = width / 2, cy = height / 2;
    const newAngle = Math.atan2(y2 - cy, x - cx) * 180 / Math.PI;
    const oldAngle = Math.atan2(py - cy, px - cx) * 180 / Math.PI;
    this.roll(newAngle - oldAngle);
  }
  /** vtkCamera::Roll — rotate viewUp about the direction of projection. */
  roll(deg) {
    const cam = this.camera;
    const axis = cam.directionOfProjection;
    const t = deg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
    const v = cam.viewUp;
    const k = axis;
    const kv = [k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0]];
    const kd = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
    cam.viewUp = [
      v[0] * c + kv[0] * s + k[0] * kd * (1 - c),
      v[1] * c + kv[1] * s + k[1] * kd * (1 - c),
      v[2] * c + kv[2] * s + k[2] * kd * (1 - c)
    ];
    cam.orthogonalizeViewUp();
    this.onChange?.();
  }
  /** Mouse wheel. `forward` = wheel away from the user = zoom in. */
  wheel(forward) {
    const e = 0.2 * MOTION_FACTOR * MOUSE_WHEEL_MOTION_FACTOR;
    this.camera.dolly(Math.pow(1.1, forward ? e : -e));
    this.onChange?.();
  }
};

// render/demos/slice-control.ts
function attachSliceControls(canvas, cfg) {
  const SCROLL_PX = cfg.scrollPx ?? 7;
  const h = cfg.hooks ?? {};
  const uv = (e) => {
    const r = canvas.getBoundingClientRect();
    return { u: (e.clientX - r.left) / r.width, v: (e.clientY - r.top) / r.height, w: r.width, h: r.height };
  };
  let lastDown = 0, lastX = 0, lastY = 0;
  let view = null;
  let scroll = null;
  let grabbed = null;
  let wlDrag = null;
  const onContext = (e) => e.preventDefault();
  const onWheel = (e) => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      const { u, v, w, h: hh } = uv(e);
      cfg.getSlice().zoomAbout(cfg.orient, Math.exp(-e.deltaY * 15e-4), u, v, w, hh);
      cfg.redraw();
      h.onZoom?.();
      return;
    }
    cfg.step(e.deltaY < 0);
    cfg.redraw();
    h.onScroll?.(e.deltaY < 0);
  };
  const onDown = (e) => {
    if (e.button === 0) {
      const now = e.timeStamp, dbl = now - lastDown < 350 && Math.hypot(e.clientX - lastX, e.clientY - lastY) < 6;
      lastDown = dbl ? 0 : now;
      lastX = e.clientX;
      lastY = e.clientY;
      if (dbl && (e.ctrlKey || e.metaKey) && cfg.wl?.enabled() && cfg.wl.reset) {
        e.preventDefault();
        cfg.wl.reset();
        cfg.redraw();
        return;
      }
      if (dbl && h.onDoubleClick?.()) {
        e.preventDefault();
        return;
      }
    }
    const wantPan = e.button === 1 || e.button === 0 && e.shiftKey;
    const wantZoom = e.button === 2;
    if (wantPan || wantZoom) {
      e.preventDefault();
      const { u: u2, v: v2 } = uv(e);
      view = { mode: wantZoom ? "zoom" : "pan", x: e.clientX, y: e.clientY, pu: u2, pv: v2 };
      canvas.style.cursor = wantZoom ? "ns-resize" : "grabbing";
      canvas.setPointerCapture(e.pointerId);
      return;
    }
    if (e.button !== 0) return;
    e.preventDefault();
    const { u, v, w, h: hh } = uv(e);
    const mode = cfg.leftMode?.() ?? (cfg.wl?.enabled() ? "wl" : "scroll");
    if (h.onLeftGrab?.(u, v, w, hh)) {
      grabbed = { moved: 0 };
    } else if (mode === "wl" && cfg.wl) {
      const [win, lev] = cfg.wl.get();
      wlDrag = { x: e.clientX, y: e.clientY, win, lev };
      canvas.style.cursor = "crosshair";
    } else if (mode === "zoom" || mode === "pan") {
      view = { mode, x: e.clientX, y: e.clientY, pu: u, pv: v };
      canvas.style.cursor = mode === "zoom" ? "ns-resize" : "grabbing";
    } else scroll = { x: e.clientX, y: e.clientY, acc: 0 };
    canvas.setPointerCapture(e.pointerId);
  };
  const onMove = (e) => {
    if (view) {
      const dx = e.clientX - view.x, dy = e.clientY - view.y;
      const r = canvas.getBoundingClientRect();
      if (view.mode === "pan") cfg.getSlice().panByPixels(cfg.orient, dx, dy, r.width, r.height);
      else cfg.getSlice().zoomAbout(cfg.orient, Math.exp(dy * 6e-3), 0.5, 0.5, r.width, r.height);
      view.x = e.clientX;
      view.y = e.clientY;
      cfg.redraw();
      return;
    }
    if (grabbed) {
      grabbed.moved += Math.abs(e.movementX) + Math.abs(e.movementY);
      const { u, v, w, h: hh } = uv(e);
      h.onLeftDrag?.(u, v, w, hh);
      return;
    }
    if (wlDrag && cfg.wl) {
      const [lo2, hi2] = cfg.wl.range();
      const r = canvas.getBoundingClientRect();
      const gain = (hi2 - lo2) / Math.max(1, Math.min(r.width, r.height));
      let win = wlDrag.win + gain * (e.clientX - wlDrag.x);
      if (win < 0) win = 0;
      let lev = wlDrag.lev + gain * (wlDrag.y - e.clientY);
      if (lev < lo2 - win / 2) lev = lo2 - win / 2;
      if (lev > hi2 + win / 2) lev = hi2 + win / 2;
      cfg.wl.set(win, lev);
      wlDrag = { x: e.clientX, y: e.clientY, win, lev };
      cfg.redraw();
      return;
    }
    if (scroll) {
      scroll.acc += e.clientX - scroll.x - (e.clientY - scroll.y);
      scroll.x = e.clientX;
      scroll.y = e.clientY;
      while (Math.abs(scroll.acc) >= SCROLL_PX) {
        const f = scroll.acc > 0;
        cfg.step(f);
        scroll.acc -= f ? SCROLL_PX : -SCROLL_PX;
      }
      cfg.redraw();
      return;
    }
    if (e.buttons === 0 && h.onHover) {
      const { u, v, w, h: hh } = uv(e);
      h.onHover(u, v, w, hh);
    }
  };
  const onUp = (e) => {
    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch {
    }
    if (view) {
      view = null;
      canvas.style.cursor = "default";
      return;
    }
    if (grabbed) {
      const m = grabbed.moved;
      grabbed = null;
      h.onLeftDrop?.(m);
      return;
    }
    if (wlDrag) {
      wlDrag = null;
      canvas.style.cursor = "default";
      return;
    }
    scroll = null;
  };
  canvas.addEventListener("contextmenu", onContext);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);
  return {
    resetView() {
      cfg.getSlice().resetView(cfg.orient);
      cfg.redraw();
    },
    detach() {
      canvas.removeEventListener("contextmenu", onContext);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
    }
  };
}

// render/capsule-field.ts
var MAX = 256;
var CapsuleField = class {
  kind = "cap";
  bindingCount = 0;
  segA = new Float32Array(MAX * 4);
  // (ax,ay,az,radius)
  segB = new Float32Array(MAX * 4);
  // (bx,by,bz,_)
  colors = new Float32Array(MAX * 4);
  n = 0;
  maxR = 0;
  clippable;
  ghost;
  providesSkip = true;
  screen;
  sh;
  ka;
  kd;
  ks;
  light;
  constructor(segments = [], opts = {}) {
    this.setSegments(segments);
    this.sh = opts.shininess ?? 40;
    this.ka = opts.kAmbient ?? 0.35;
    this.kd = opts.kDiffuse ?? 0.8;
    this.ks = opts.kSpecular ?? 0.35;
    this.light = opts.lightColor ?? [1, 1, 1];
    this.clippable = opts.clippable ?? true;
    this.ghost = opts.ghost ?? false;
    this.screen = opts.screenSpace ?? false;
  }
  setSegments(list) {
    this.n = Math.min(list.length, MAX);
    this.segA.fill(0);
    this.segB.fill(0);
    this.colors.fill(0);
    this.maxR = 0;
    for (let i = 0; i < this.n; i++) {
      const s = list[i];
      this.segA.set([s.a[0], s.a[1], s.a[2], s.radius], i * 4);
      this.segB.set([s.b[0], s.b[1], s.b[2], 0], i * 4);
      this.colors.set(s.color, i * 4);
      this.maxR = Math.max(this.maxR, s.radius);
    }
  }
  get count() {
    return this.n;
  }
  uniformFloats() {
    return 12 + MAX * 4 * 3;
  }
  // params + params2 + light + segA + segB + colors
  sampleStep() {
    return 1;
  }
  aabb() {
    if (this.n === 0) return [[-1, -1, -1], [1, 1, 1]];
    const lo2 = [Infinity, Infinity, Infinity], hi2 = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < this.n; i++) {
      const r = this.screen ? 0 : this.segA[i * 4 + 3];
      for (let a = 0; a < 3; a++) {
        lo2[a] = Math.min(lo2[a], this.segA[i * 4 + a] - r, this.segB[i * 4 + a] - r);
        hi2[a] = Math.max(hi2[a], this.segA[i * 4 + a] + r, this.segB[i * 4 + a] + r);
      }
    }
    if (this.screen) {
      const diag = Math.hypot(hi2[0] - lo2[0], hi2[1] - lo2[1], hi2[2] - lo2[2]);
      const m = Math.max(40, diag * 0.15);
      for (let a = 0; a < 3; a++) {
        lo2[a] -= m;
        hi2[a] += m;
      }
    }
    return [lo2, hi2];
  }
  structMembers(s) {
    return [
      `  cap${s}_params : vec4<f32>,`,
      // n_segments, visible, shininess, k_ambient
      `  cap${s}_params2 : vec4<f32>,`,
      // k_diffuse, k_specular, max_radius, _
      `  cap${s}_light : vec4<f32>,`,
      // light_color.rgb, _
      `  cap${s}_segA : array<vec4<f32>, ${MAX}>,`,
      `  cap${s}_segB : array<vec4<f32>, ${MAX}>,`,
      `  cap${s}_colors : array<vec4<f32>, ${MAX}>,`
    ].join("\n");
  }
  declareBindings(_s2, _base) {
    return "";
  }
  bindEntries(_s2, _base) {
    return [];
  }
  skipWGSL(s) {
    return (
      /* wgsl */
      `
fn cap_closest${s}(p : vec3<f32>, a : vec3<f32>, b : vec3<f32>) -> vec3<f32> {
  let ba = b - a;
  let h = clamp(dot(p - a, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
  return a + ba * h;
}
fn skip_cap${s}(wp : vec3<f32>) -> f32 {
  let n = i32(u_material.cap${s}_params.x);
  if (n <= 0) { return 1.0e6; }
  var best = 1.0e12;
  for (var k = 0; k < n; k = k + 1) {
    let A = u_material.cap${s}_segA[k];
    if (A.w <= 0.0) { continue; }
    let c = cap_closest${s}(wp, A.xyz, u_material.cap${s}_segB[k].xyz);
    ${this.screen ? `let r = A.w * length(u_cam.eye.xyz - c) / max(u_cam.size.z, 1.0);` : `let r = A.w;`}
    best = min(best, length(wp - c) - r);
  }
  return max(best, 0.0);
}`
    );
  }
  samplingWGSL(s) {
    return (
      /* wgsl */
      `
fn sample_field_cap${s}(wp : vec3<f32>, rd : vec3<f32>) -> vec4<f32> {
  let wp_r = transform_point_cap${s}(wp);
  let n = i32(u_material.cap${s}_params.x);
  var best_depth = -1.0;
  var best_c = vec3<f32>(0.0);
  var best_color = vec4<f32>(0.0);
  var found = false;
  for (var k = 0; k < n; k = k + 1) {
    let A = u_material.cap${s}_segA[k];
    if (A.w <= 0.0) { continue; }
    let c = cap_closest${s}(wp_r, A.xyz, u_material.cap${s}_segB[k].xyz);
    ${this.screen ? `let r = A.w * length(u_cam.eye.xyz - c) / max(u_cam.size.z, 1.0);` : `let r = A.w;`}
    let depth = r - length(wp_r - c);
    if (depth > best_depth) { best_depth = depth; best_c = c; best_color = u_material.cap${s}_colors[k]; found = true; }
  }
  if (!found || best_depth <= 0.0) { return vec4<f32>(0.0); }

  let to_wp = wp_r - best_c;
  var n_hat = to_wp / max(length(to_wp), 1e-6);
  if (dot(n_hat, -rd) < 0.0) { n_hat = -n_hat; }
  let view_dir = normalize(-rd);
  let ldotn = max(dot(view_dir, n_hat), 0.0);
  let refl = normalize(2.0 * ldotn * n_hat - view_dir);
  let rdotv = max(dot(refl, view_dir), 0.0);
  let sh = u_material.cap${s}_params.z;
  let ka = u_material.cap${s}_params.w; let kd = u_material.cap${s}_params2.x; let ks = u_material.cap${s}_params2.y;
  let base = best_color.rgb;
  let highlight = mix(base, u_material.cap${s}_light.rgb, 0.85);
  let lit = base * ka + base * (kd * ldotn) + highlight * (ks * pow(rdotv, sh));
  let col = srgb2physical(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)));
  ${this.ghost ? `let ghostScale = 0.5;` : `let ghostScale = 1.0;`}
  let opacity = clamp(best_color.a, 0.0, 1.0) * ghostScale;
  return vec4<f32>(col * opacity, opacity);
}`
    );
  }
  fillUniforms(out, off) {
    out[off + 0] = this.n;
    out[off + 1] = 1;
    out[off + 2] = this.sh;
    out[off + 3] = this.ka;
    out[off + 4] = this.kd;
    out[off + 5] = this.ks;
    out[off + 6] = this.maxR;
    out[off + 7] = -1;
    out[off + 8] = this.light[0];
    out[off + 9] = this.light[1];
    out[off + 10] = this.light[2];
    out.set(this.segA, off + 12);
    out.set(this.segB, off + 12 + MAX * 4);
    out.set(this.colors, off + 12 + MAX * 8);
  }
};

// render/textures.ts
function writeScalarTexture3DSlices(dev, texture, data, dims, zStart, zCount) {
  const [dx, dy, dz] = dims;
  if (zCount <= 0 || zStart < 0 || zStart >= dz) {
    return;
  }
  const depth = Math.min(zCount, dz - zStart);
  if (depth <= 0) {
    return;
  }
  const bytesPerRow = dx * 4;
  const rowsPerImage = dy;
  const bytesPerSlice = bytesPerRow * rowsPerImage;
  const maxBuffer = Math.max(1, Number(dev.limits.maxBufferSize) || 256 * 1024 * 1024);
  const maxChunkBytes = Math.max(bytesPerSlice, Math.floor(maxBuffer * 0.9));
  const slicesPerChunk = Math.max(1, Math.floor(maxChunkBytes / bytesPerSlice));
  let z2 = zStart;
  const zEnd = zStart + depth;
  while (z2 < zEnd) {
    const chunkDepth = Math.min(slicesPerChunk, zEnd - z2);
    const offsetFloats = z2 * dx * dy;
    const chunk = data.subarray(offsetFloats, offsetFloats + chunkDepth * dx * dy);
    dev.queue.writeTexture(
      { texture, origin: [0, 0, z2] },
      chunk,
      { bytesPerRow, rowsPerImage },
      [dx, dy, chunkDepth]
    );
    z2 += chunkDepth;
  }
}
function writeScalarTexture3DRegion(dev, texture, data, dims, xStart, yStart, zStart, xCount, yCount, zCount) {
  const [dx, dy, dz] = dims;
  if (xCount <= 0 || yCount <= 0 || zCount <= 0) return;
  if (xStart < 0 || yStart < 0 || zStart < 0) return;
  if (xStart >= dx || yStart >= dy || zStart >= dz) return;
  const w = Math.min(xCount, dx - xStart);
  const h = Math.min(yCount, dy - yStart);
  const d = Math.min(zCount, dz - zStart);
  if (w <= 0 || h <= 0 || d <= 0) return;
  const bytesPerRow = dx * 4;
  const rowsPerImage = dy;
  const offsetFloats = (zStart * dy + yStart) * dx + xStart;
  dev.queue.writeTexture(
    { texture, origin: [xStart, yStart, zStart] },
    data,
    { offset: offsetFloats * 4, bytesPerRow, rowsPerImage },
    [w, h, d]
  );
}

// render/fields.ts
function transformedAABB(m, lo2, hi2) {
  const mn2 = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < 8; i++) {
    const c = applyMat4(m, [i & 1 ? hi2[0] : lo2[0], i & 2 ? hi2[1] : lo2[1], i & 4 ? hi2[2] : lo2[2]]);
    for (let a = 0; a < 3; a++) {
      mn2[a] = Math.min(mn2[a], c[a]);
      mx[a] = Math.max(mx[a], c[a]);
    }
  }
  return [mn2, mx];
}
var ImageField = class {
  kind = "img";
  bindingCount = 2;
  // volume (3d) + lut (2d)
  volTex;
  lutTex;
  dev;
  p2t;
  clim;
  shade;
  unit;
  stepMm;
  box;
  normScale = 1;
  // r8unorm samples return raw/255; clim is packed /normScale so shader math is unchanged
  dims;
  constructor(dev, data, dims, spacing, lut, opts) {
    this.dims = dims;
    const center = opts.center ?? [0, 0, 0];
    let src = data, fmt = "r32float", bpe = 4;
    this.normScale = 1;
    if (data instanceof Uint8Array) {
      fmt = "r8unorm";
      bpe = 1;
      this.normScale = 255;
    } else if (data instanceof Uint16Array) {
      src = Float32Array.from(data);
      fmt = "r32float";
      bpe = 4;
    }
    this.volTex = dev.createTexture({ size: dims, dimension: "3d", format: fmt, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    {
      const bytesPerRow = dims[0] * bpe, rowsPerImage = dims[1], sliceBytes = bytesPerRow * rowsPerImage;
      const CHUNK = 256 * 1024 * 1024;
      const slab = Math.max(1, Math.min(dims[2], Math.floor(CHUNK / Math.max(1, sliceBytes))));
      for (let z2 = 0; z2 < dims[2]; z2 += slab) {
        const depth = Math.min(slab, dims[2] - z2);
        dev.queue.writeTexture(
          { texture: this.volTex, origin: { x: 0, y: 0, z: z2 } },
          src,
          { offset: z2 * sliceBytes, bytesPerRow, rowsPerImage },
          [dims[0], dims[1], depth]
        );
      }
    }
    this.lutTex = dev.createTexture({ size: [256, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    dev.queue.writeTexture({ texture: this.lutTex }, lut, { bytesPerRow: 256 * 4 }, [256, 1]);
    if (opts.ijkToRAS) {
      this.p2t = patientToTextureFromIjkToRAS(opts.ijkToRAS, dims);
      this.box = volumeAABBFromIjkToRAS(opts.ijkToRAS, dims);
      this.stepMm = Math.min(...spacingFromIjkToRAS(opts.ijkToRAS));
    } else {
      this.p2t = patientToTexture(dims, spacing, center);
      this.box = volumeAABB(dims, spacing, center);
      this.stepMm = Math.min(...spacing);
    }
    this.clim = opts.clim;
    this.shade = opts.shade ?? [0.35, 0.75, 0.35, 20];
    this.unit = opts.opacityUnitDistance ?? this.stepMm;
    this.dev = dev;
  }
  /** Replace the 256-entry rgba8 color/opacity LUT in place (no texture/bind-group churn).
   *  The bind group holds a stable view of lutTex, so the next render uses the new LUT. */
  setLUT(lut) {
    this.dev.queue.writeTexture({ texture: this.lutTex }, lut, { bytesPerRow: 256 * 4 }, [256, 1]);
  }
  /**
   * Patch a contiguous z-range of the scalar volume texture from a full-volume
   * Float32Array (r32float ImageFields). No pipeline rebuild — caller redraws.
   */
  updateScalars(data, zStart, zCount) {
    if (this.normScale !== 1) {
      throw new Error("ImageField.updateScalars requires an r32float volume");
    }
    writeScalarTexture3DSlices(this.dev, this.volTex, data, this.dims, zStart, zCount);
  }
  /**
   * Patch an axis-aligned box (one zarr chunk) from a full-volume Float32Array.
   * Enables MPR mid-plane updates before a full z-slab is assembled.
   */
  updateScalarsRegion(data, xStart, yStart, zStart, xCount, yCount, zCount) {
    if (this.normScale !== 1) {
      throw new Error("ImageField.updateScalarsRegion requires an r32float volume");
    }
    writeScalarTexture3DRegion(
      this.dev,
      this.volTex,
      data,
      this.dims,
      xStart,
      yStart,
      zStart,
      xCount,
      yCount,
      zCount
    );
  }
  /** Free GPU textures (call when replacing the field on a coarse→fine LOD upgrade). */
  destroy() {
    this.volTex.destroy();
    this.lutTex.destroy();
  }
  /** The scalar range the LUT spans — window/level for the volume rendering. Re-packed into
   *  the material uniform on the next syncUniforms()/render, so no pipeline rebuild. */
  setClim(lo2, hi2) {
    this.clim = [lo2, hi2];
  }
  getClim() {
    return [this.clim[0], this.clim[1]];
  }
  /** Phong shading tuple [ka, kd, ks, shininess] — re-packed into the material uniform next
   *  render (VR presets carry their own lighting). [1,0,0,1] = flat emission (no shading). */
  setShade(shade) {
    this.shade = [shade[0], shade[1], shade[2], shade[3]];
  }
  origP2t;
  // sampling matrix + box at identity, for setWorldTransform
  origBox;
  uniformFloats() {
    return 28;
  }
  // mat4(16) + clim(4) + shade(4) + params(4)
  aabb() {
    return this.box;
  }
  sampleStep() {
    return this.stepMm;
  }
  /** The r32float 3D scalar texture (e.g. to share with a SliceRenderer for MPR). */
  volumeTexture() {
    return this.volTex;
  }
  /** r8unorm volumes sample /255, so clim is packed /normScale in the shader; a slice plane sharing this
   *  texture must use the same factor. 1 for f32 volumes. */
  normScaleOf() {
    return this.normScale;
  }
  /** Centre of the volume in world (RAS) at identity — a natural pivot for a transform widget. */
  worldCenter() {
    const [lo2, hi2] = this.origBox ?? this.box;
    return [(lo2[0] + hi2[0]) / 2, (lo2[1] + hi2[1]) / 2, (lo2[2] + hi2[2]) / 2];
  }
  /** Place the volume in the world by a rigid transform M (worldFromLocal): the ray samples
   *  at p2t·M⁻¹·wp, so the volume appears moved/rotated. A Tier-A interactive update — caller
   *  does scene.syncUniforms() (which re-packs p2t AND refreshes the ray-entry AABB). */
  setWorldTransform(m) {
    if (!this.origP2t) {
      this.origP2t = this.p2t;
      this.origBox = this.box;
    }
    this.p2t = multiply(this.origP2t, invert(m));
    this.box = transformedAABB(m, this.origBox[0], this.origBox[1]);
  }
  /** RAS(patient) -> texture[0,1] matrix (encodes the real ijkToRAS geometry). */
  patientToTexture() {
    return this.p2t;
  }
  /** Re-place the volume in RAS without re-uploading voxels (a parent transform moved it). */
  setIjkToRAS(ijkToRAS) {
    this.p2t = patientToTextureFromIjkToRAS(ijkToRAS, this.dims);
    this.box = volumeAABBFromIjkToRAS(ijkToRAS, this.dims);
    this.stepMm = Math.min(...spacingFromIjkToRAS(ijkToRAS));
  }
  structMembers(s) {
    return [
      `  img${s}_p2t : mat4x4<f32>,`,
      `  img${s}_clim : vec4<f32>,`,
      // lo, hi, _, _
      `  img${s}_shade : vec4<f32>,`,
      // ka, kd, ks, shininess
      `  img${s}_params : vec4<f32>,`
      // opacity_unit_distance, _, _, _
    ].join("\n");
  }
  declareBindings(s, base) {
    return [
      `@group(0) @binding(${base}) var t_vol_img${s} : texture_3d<f32>;`,
      `@group(0) @binding(${base + 1}) var t_lut_img${s} : texture_2d<f32>;`
    ].join("\n");
  }
  samplingWGSL(s) {
    return (
      /* wgsl */
      `
fn sampc_img${s}(wp : vec3<f32>) -> f32 {
  let t4 = u_material.img${s}_p2t * vec4<f32>(transform_point_img${s}(wp), 1.0);
  return textureSampleLevel(t_vol_img${s}, s_lin, clamp(t4.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).r;
}
fn sample_field_img${s}(wp : vec3<f32>, rd : vec3<f32>) -> vec4<f32> {
  let t4 = u_material.img${s}_p2t * vec4<f32>(transform_point_img${s}(wp), 1.0);
  let tex = t4.xyz;
  if (any(tex < vec3<f32>(0.0)) || any(tex > vec3<f32>(1.0))) { return vec4<f32>(0.0); }
  let val = textureSampleLevel(t_vol_img${s}, s_lin, tex, 0.0).r;
  let lo = u_material.img${s}_clim.x; let hi = u_material.img${s}_clim.y;
  let tf = textureSampleLevel(t_lut_img${s}, s_lin, vec2<f32>(clamp((val - lo) / max(hi - lo, 1e-6), 0.0, 1.0), 0.5), 0.0);
  let step = u_material.scene.x;
  let unit = max(u_material.img${s}_params.x, 1e-3);
  let opacity = clamp(1.0 - pow(1.0 - clamp(tf.a, 0.0, 1.0), step / unit), 0.0, 1.0);
  if (opacity <= 0.001) { return vec4<f32>(0.0); }
  let h = step * 2.0;   // wider central difference -> smoother normals (less shading aliasing on coarse volumes)
  let g = vec3<f32>(
    sampc_img${s}(wp + vec3<f32>(h,0,0)) - sampc_img${s}(wp - vec3<f32>(h,0,0)),
    sampc_img${s}(wp + vec3<f32>(0,h,0)) - sampc_img${s}(wp - vec3<f32>(0,h,0)),
    sampc_img${s}(wp + vec3<f32>(0,0,h)) - sampc_img${s}(wp - vec3<f32>(0,0,h))) / (2.0 * h);
  let glen = length(g);
  let ka = u_material.img${s}_shade.x; let kd = u_material.img${s}_shade.y;
  let ks = u_material.img${s}_shade.z; let sh = u_material.img${s}_shade.w;
  var lit_srgb = tf.rgb * ka;
  if (glen > 1e-6) {
    var n = g / glen;
    if (dot(n, -rd) < 0.0) { n = -n; }
    let view_dir = normalize(-rd);
    let ldotn = dot(view_dir, n);
    if (ldotn > 0.0) {
      let refl = normalize(2.0 * ldotn * n - view_dir);
      let rdotv = max(0.0, dot(refl, view_dir));
      lit_srgb = tf.rgb * (ka + kd * ldotn) + vec3<f32>(ks * pow(rdotv, sh));
    }
  }
  let lit = srgb2physical(clamp(lit_srgb, vec3<f32>(0.0), vec3<f32>(1.0)));
  return vec4<f32>(lit * opacity, opacity);
}`
    );
  }
  fillUniforms(out, off) {
    out.set(this.p2t, off);
    out[off + 16] = this.clim[0] / this.normScale;
    out[off + 17] = this.clim[1] / this.normScale;
    out[off + 20] = this.shade[0];
    out[off + 21] = this.shade[1];
    out[off + 22] = this.shade[2];
    out[off + 23] = this.shade[3];
    out[off + 24] = this.unit;
  }
  bindEntries(_s2, base) {
    return [
      { binding: base, resource: this.volTex.createView() },
      { binding: base + 1, resource: this.lutTex.createView() }
    ];
  }
};
var SegmentField = class {
  kind = "seg";
  bindingCount;
  // 1 (value texture) + 1 when an sdf attr (opacity) texture is bound
  clippable;
  tex;
  attrTex;
  // sdf per-voxel attributes (.r = opacity)
  p2t;
  box;
  color;
  opacity;
  shade;
  bandMm;
  stepMm;
  mode;
  colorFromTex;
  interfaceMode;
  voxelMm = 1;
  providesSkip;
  constructor(tex, dims, spacing, opts) {
    this.tex = tex;
    const center = opts.center ?? [0, 0, 0];
    let voxelMm;
    if (opts.ijkToRAS) {
      this.p2t = patientToTextureFromIjkToRAS(opts.ijkToRAS, dims);
      this.box = volumeAABBFromIjkToRAS(opts.ijkToRAS, dims);
      voxelMm = Math.min(...spacingFromIjkToRAS(opts.ijkToRAS));
    } else {
      this.p2t = patientToTexture(dims, spacing, center);
      this.box = volumeAABB(dims, spacing, center);
      voxelMm = Math.min(...spacing);
    }
    this.color = opts.color;
    this.opacity = opts.opacity ?? 1;
    this.shade = opts.shade ?? [0.2, 0.85, 0.3, 32];
    this.bandMm = opts.bandMm ?? voxelMm;
    this.stepMm = opts.sampleStepMm ?? Math.max(0.5 * voxelMm, 0.1);
    this.clippable = opts.clippable ?? true;
    this.voxelMm = voxelMm;
    this.mode = opts.mode ?? "iso";
    this.providesSkip = this.mode === "sdf";
    this.colorFromTex = opts.colorFromTexture ?? false;
    this.interfaceMode = this.mode === "sdf" && (opts.interfaceMode ?? false);
    this.attrTex = this.mode === "sdf" ? opts.attrTexture : void 0;
    this.bindingCount = this.attrTex ? 2 : 1;
  }
  /** Field-level opacity (multiplies every segment's per-label opacity in the shader). Live global
   *  segmentation opacity — the caller does scene.syncUniforms() + redraw to apply. */
  setOpacity(o) {
    this.opacity = Math.max(0, Math.min(1, o));
  }
  uniformFloats() {
    return 36;
  }
  // mat4(16) + color(4) + shade(4) + params(4) + bmin(4) + bmax(4)
  aabb() {
    return this.box;
  }
  sampleStep() {
    return this.stepMm;
  }
  setTexture(tex, destroyPrev = true) {
    if (destroyPrev && this.tex !== tex) this.tex.destroy();
    this.tex = tex;
  }
  /** Empty-space skip for "sdf" mode: the texture .a is a TRUE distance-to-surface (mm), so the ray can
   *  leap |sdf| minus the shell band toward the surface — sphere tracing. A one-voxel safety margin
   *  absorbs the JFA distance approximation so the leap never overshoots a thin shell (image unchanged,
   *  just far fewer march steps). Self-contained (no dependency on the sampling fns' emission order). */
  skipWGSL(s) {
    return (
      /* wgsl */
      `
fn skip_seg${s}(wp : vec3<f32>) -> f32 {
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) {
    // OUTSIDE the SDF grid: leap only to the seg's AABB (a contract-valid lower bound), never past it.
    let cen = (u_material.seg${s}_bmin.xyz + u_material.seg${s}_bmax.xyz) * 0.5;
    let ext = (u_material.seg${s}_bmax.xyz - u_material.seg${s}_bmin.xyz) * 0.5;
    let q = abs(transform_point_seg${s}(wp) - cen) - ext;
    return max(0.0, length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0));
  }
  let d = abs(textureSampleLevel(t_seg${s}, s_lin, t, 0.0).a);                  // |distance to surface| (mm)
  return max(0.0, d - u_material.seg${s}_params.x - u_material.seg${s}_params.y);   // leap toward the shell (band + 1 voxel safe)
}`
    );
  }
  structMembers(s) {
    return [
      `  seg${s}_p2t : mat4x4<f32>,`,
      `  seg${s}_color : vec4<f32>,`,
      // rgb, opacity
      `  seg${s}_shade : vec4<f32>,`,
      // ka, kd, ks, shininess
      `  seg${s}_params : vec4<f32>,`,
      // band_mm, voxel_mm, _, _
      `  seg${s}_bmin : vec4<f32>,`,
      // aabb min (RAS) — for a contract-valid skip outside the texture
      `  seg${s}_bmax : vec4<f32>,`
    ].join("\n");
  }
  declareBindings(s, base) {
    const value = `@group(0) @binding(${base}) var t_seg${s} : texture_3d<f32>;`;
    return this.attrTex ? `${value}
@group(0) @binding(${base + 1}) var t_attr${s} : texture_3d<f32>;` : value;
  }
  samplingWGSL(s) {
    if (this.mode === "sdf") {
      return (
        /* wgsl */
        `
fn v_seg${s}(wp : vec3<f32>) -> f32 {   // signed distance (mm)
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return 1e3; }   // far outside \u2192 culled
  return textureSampleLevel(t_seg${s}, s_lin, t, 0.0).a;
}
fn vgrad_seg${s}(wp : vec3<f32>) -> f32 {   // signed distance for the NORMAL finite-difference
  // A gradient tap that steps just outside the (padded) SDF texture must read BACKGROUND, not the
  // out-of-volume cull sentinel (1e3) \u2014 a huge sentinel would fabricate an enormous fake gradient that
  // points out through the volume face and unlights the surface (black speckle at the seg/boundary
  // interface). Clamp to the texture edge: with the padded grid that edge IS background, so the normal
  // near a cap stays correct. This is the "artificial background boundary sample" for the normal.
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = clamp(t4.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
  return textureSampleLevel(t_seg${s}, s_lin, t, 0.0).a;
}
fn col_seg${s}(wp : vec3<f32>) -> vec3<f32> {   // per-label colour of the nearest region
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return vec3<f32>(0.0); }
  let pm = textureSampleLevel(t_seg${s}, s_lin, t, 0.0).rgb;   // PREMULTIPLIED (rgb\xB7opacity)${this.attrTex ? `
  let a = textureSampleLevel(t_attr${s}, s_lin, t, 0.0).r;      // per-segment opacity (crisp) \u2014 un-premultiply to the true colour, so a hidden neighbour's colour doesn't bleed in
  return pm / max(a, 1e-3);` : `
  return pm;`}
}${this.attrTex ? `
fn attr_seg${s}(wp : vec3<f32>) -> vec2<f32> {   // per-segment (.x = opacity, .y = shading mode)
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return vec2<f32>(0.0); }
  return textureSampleLevel(t_attr${s}, s_lin, t, 0.0).rg;
}` : ""}${this.interfaceMode ? `
fn bdist_seg${s}(wp : vec3<f32>) -> f32 {   // SMOOTH (seam-blurred) interface distance from attr.b \u2014 for the normal only
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = clamp(t4.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
  return textureSampleLevel(t_attr${s}, s_lin, t, 0.0).b;
}
fn pres_seg${s}(wp : vec3<f32>) -> f32 {   // CRISP in-segment presence (attr.a), linear-sampled for ~1-voxel AA
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return 0.0; }
  return textureSampleLevel(t_attr${s}, s_lin, t, 0.0).a;
}
fn g1_seg${s}(c : f32, wp : vec3<f32>, dir : vec3<f32>, h : f32) -> f32 {
  // One-sided difference of the SMOOTH interface distance along dir, picking the STEEPER side. The
  // unsigned distance has a V-crease at the surface: a central difference straddling it cancels (fake
  // zero gradient \u2192 degenerate normal). Taking the steeper one-sided difference follows the true \xB11
  // slope AWAY from the interface, so the normal stays well-defined right at the shell \u2014 computed on
  // the fly only at shell samples, no stored normal. Uses the blurred distance (attr.b) so the normal
  // is smooth (no JFA facets); the SHARP sdfTex.a still drives shell membership (surface stays at 0).
  let dp = bdist_seg${s}(wp + dir * h) - c;
  let dm = c - bdist_seg${s}(wp - dir * h);
  return select(dm, dp, abs(dp) > abs(dm)) / h;
}` : ""}
// Shell (surface) contribution at wp: crisp Phong shell around sdf=0. Weighted by (1-mode) so it
// morphs smoothly into the volume contribution across a blurred surface\u2194volume boundary.
fn surface_seg${s}(wp : vec3<f32>, rd : vec3<f32>, sdf : f32, band : f32, step : f32, seg_op : f32, op0 : f32) -> vec4<f32> {
  let d_mm = abs(sdf);
  if (d_mm > band + step) { return vec4<f32>(0.0); }
  let T = clamp(op0 * seg_op, 0.0, 1.0);      // TARGET surface opacity (per-segment \xD7 field)
  if (T <= 0.0) { return vec4<f32>(0.0); }
  let h = step;
${this.interfaceMode ? `  let hg = 1.5 * step;
  let dc = bdist_seg${s}(wp);
  let g = vec3<f32>(
    g1_seg${s}(dc, wp, vec3<f32>(1,0,0), hg),
    g1_seg${s}(dc, wp, vec3<f32>(0,1,0), hg),
    g1_seg${s}(dc, wp, vec3<f32>(0,0,1), hg));` : `  let g = vec3<f32>(
    vgrad_seg${s}(wp + vec3<f32>(h,0,0)) - vgrad_seg${s}(wp - vec3<f32>(h,0,0)),
    vgrad_seg${s}(wp + vec3<f32>(0,h,0)) - vgrad_seg${s}(wp - vec3<f32>(0,h,0)),
    vgrad_seg${s}(wp + vec3<f32>(0,0,h)) - vgrad_seg${s}(wp - vec3<f32>(0,0,h))) / (2.0 * h);`}
  let glen = length(g);
  if (glen < 1e-5) { return vec4<f32>(0.0); }
  var n = g / glen;
  if (dot(n, -rd) < 0.0) { n = -n; }
  // SURFACE opacity (Slicer polydata parity): the shell is a THIN surface of opacity T, not a solid
  // band. A raymarch crosses it in several samples; giving each \u03B1=T lets the front-to-back OVER
  // saturate toward opaque (50% looked like ~100%). Instead accumulate OPTICAL DEPTH with a shell
  // profile \u03C1 = a/band that integrates to 1 across the crossing, scaled by -ln(1-T): \u03A3d\u03C4 = -ln(1-T),
  // so net opacity = 1-e^(-\u03A3d\u03C4) = T EXACTLY \u2014 independent of band thickness and sample rate, and T\u21921
  // stays crisply opaque. |dot(rd,n)| converts ray-step to shell-normal distance (\u21920 at grazing =
  // built-in silhouette AA).
  let a = max(1.0 - d_mm / band, 0.0);
  if (a <= 0.0) { return vec4<f32>(0.0); }
  // Convert ray-step to d_mm-distance. Outer: RAW gradient projection |dot(rd,g)| = |d(d_mm)/ds|
  // (includes |grad sdf|, which the distance blur pulls below 1). Interface: the re-signed gradient has
  // an arbitrary magnitude (a sign jump at the interface), so use the UNIT normal cosine |dot(rd,n)| \u2014
  // still \u21920 at grazing (silhouette AA), and keeps opacity thickness-consistent.
  let rate = max(abs(dot(rd, ${this.interfaceMode ? "n" : "g"})), 1e-3);
  let tau = -log(1.0 - min(T, 0.9999)) * (a / band) * (step * rate);
  var op = 1.0 - exp(-tau);
${this.interfaceMode ? `  op = op * pres_seg${s}(wp);   // gate to GENUINELY in-segment voxels \u2014 kills the bled colour/opacity halo beyond the real edge` : ""}
  if (op <= 0.0004) { return vec4<f32>(0.0); }
  let ka = u_material.seg${s}_shade.x; let kd = u_material.seg${s}_shade.y;
  let ks = u_material.seg${s}_shade.z; let sh = u_material.seg${s}_shade.w;
  let ldn = max(dot(-rd, n), 0.0);
  let refl = normalize(2.0 * ldn * n + rd);
  let rdv = max(dot(refl, -rd), 0.0);
  let col = col_seg${s}(wp);
  var lit = col * ka + col * (kd * ldn) + vec3<f32>(ks * pow(rdv, max(sh, 1.0)));
  lit = srgb2physical(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)));
  return vec4<f32>(lit * op, op);
}
fn sample_field_seg${s}(wp : vec3<f32>, rd : vec3<f32>) -> vec4<f32> {
  let op0 = u_material.seg${s}_color.a;
  if (op0 <= 0.0) { return vec4<f32>(0.0); }
  let sdf = v_seg${s}(wp);
  let band = max(u_material.seg${s}_params.x, 1e-3);
  let step = max(u_material.scene.x, 1e-3);
${this.attrTex ? `  let at = attr_seg${s}(wp);        // (opacity, shading mode)
  let seg_op = at.x;
  if (seg_op <= 0.0) { return vec4<f32>(0.0); }
  let mode = clamp(at.y, 0.0, 1.0);
  // Surface and volume are BLENDED by the (seam-blurred, fractional) mode, so an opaque-surface
  // segment and a translucent-volume segment meet with a smooth transition instead of a jagged,
  // voxel-quantized classification edge.
  var acc = vec4<f32>(0.0);
  if (mode > 0.001 && sdf < 0.0) {
    // VOLUME: translucent DVR fill of the interior (~24 mm opacity-unit-distance).
    let vop = clamp(op0 * seg_op * step / 24.0, 0.0, 1.0);
    if (vop > 0.0) {
      let vcol = srgb2physical(clamp(col_seg${s}(wp), vec3<f32>(0.0), vec3<f32>(1.0)));
      acc += mode * vec4<f32>(vcol * vop, vop);
    }
  }
  if (mode < 0.999) {
    acc += (1.0 - mode) * surface_seg${s}(wp, rd, sdf, band, step, seg_op, op0);
  }
  return acc;` : `  return surface_seg${s}(wp, rd, sdf, band, step, 1.0, op0);`}
}`
      );
    }
    const alphaWGSL = this.mode === "surface" ? (
      /* wgsl */
      `
  let step = max(u_material.scene.x, 1e-3);
  let op = clamp(op0 * glen * step, 0.0, 1.0);
  if (op <= 0.0) { return vec4<f32>(0.0); }`
    ) : (
      /* wgsl */
      `
  // Local first-order signed distance to the v=0.5 isosurface (mm), then a
  // 1-voxel opacity band around it: crisp opaque shell, sub-voxel anti-aliased.
  let d_mm = abs((v - 0.5) / glen);
  let band = max(u_material.seg${s}_params.x, 1e-3);
  let a = 1.0 - clamp(d_mm / band, 0.0, 1.0);
  if (a <= 0.0) { return vec4<f32>(0.0); }
  let op = clamp(a * op0, 0.0, 1.0);`
    );
    const colWGSL = this.colorFromTex ? (
      /* wgsl */
      `
fn col_seg${s}(wp : vec3<f32>) -> vec3<f32> {
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return vec3<f32>(0.0); }
  return textureSampleLevel(t_seg${s}, s_lin, t, 0.0).rgb;
}`
    ) : "";
    const colExpr = this.colorFromTex ? `col_seg${s}(wp)` : `u_material.seg${s}_color.rgb`;
    return (
      /* wgsl */
      `
fn v_seg${s}(wp : vec3<f32>) -> f32 {
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return 0.0; }
  return textureSampleLevel(t_seg${s}, s_lin, t, 0.0).a;   // Gaussian-smoothed presence in .a
}${colWGSL}
fn sample_field_seg${s}(wp : vec3<f32>, rd : vec3<f32>) -> vec4<f32> {
  let op0 = u_material.seg${s}_color.a;
  if (op0 <= 0.0) { return vec4<f32>(0.0); }
  let v = v_seg${s}(wp);
  // Skip deep interior / exterior: |grad| ~ 0 there so no shell to emit.
  if (v <= 0.02 || v >= 0.98) { return vec4<f32>(0.0); }
  let h = max(u_material.scene.x, 1e-3);
  let g = vec3<f32>(
    v_seg${s}(wp + vec3<f32>(h,0,0)) - v_seg${s}(wp - vec3<f32>(h,0,0)),
    v_seg${s}(wp + vec3<f32>(0,h,0)) - v_seg${s}(wp - vec3<f32>(0,h,0)),
    v_seg${s}(wp + vec3<f32>(0,0,h)) - v_seg${s}(wp - vec3<f32>(0,0,h))) / (2.0 * h);
  let glen = length(g);
  if (glen < 1e-5) { return vec4<f32>(0.0); }${alphaWGSL}
  // Phong from the same gradient, normal flipped to face the camera.
  var n = g / glen;
  if (dot(n, -rd) < 0.0) { n = -n; }
  let ka = u_material.seg${s}_shade.x; let kd = u_material.seg${s}_shade.y;
  let ks = u_material.seg${s}_shade.z; let sh = u_material.seg${s}_shade.w;
  let ldn = max(dot(-rd, n), 0.0);
  let refl = normalize(2.0 * ldn * n + rd);
  let rdv = max(dot(refl, -rd), 0.0);
  let col = ${colExpr};
  var lit = col * ka + col * (kd * ldn) + vec3<f32>(ks * pow(rdv, max(sh, 1.0)));
  lit = srgb2physical(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)));
  return vec4<f32>(lit * op, op);
}`
    );
  }
  fillUniforms(out, off) {
    out.set(this.p2t, off);
    out[off + 16] = this.color[0];
    out[off + 17] = this.color[1];
    out[off + 18] = this.color[2];
    out[off + 19] = this.opacity;
    out[off + 20] = this.shade[0];
    out[off + 21] = this.shade[1];
    out[off + 22] = this.shade[2];
    out[off + 23] = this.shade[3];
    out[off + 24] = this.bandMm;
    out[off + 25] = this.voxelMm;
    out[off + 28] = this.box[0][0];
    out[off + 29] = this.box[0][1];
    out[off + 30] = this.box[0][2];
    out[off + 32] = this.box[1][0];
    out[off + 33] = this.box[1][1];
    out[off + 34] = this.box[1][2];
  }
  bindEntries(_s2, base) {
    const e = [{ binding: base, resource: this.tex.createView() }];
    if (this.attrTex) e.push({ binding: base + 1, resource: this.attrTex.createView() });
    return e;
  }
};
var RGBAVolumeField = class {
  kind = "rgba";
  bindingCount = 1;
  // baked rgba texture (sampler shared)
  clippable;
  tex;
  p2t;
  shade;
  unit;
  stepMm;
  box;
  constructor(tex, dims, spacing, opts = {}) {
    const center = opts.center ?? [0, 0, 0];
    this.tex = tex;
    if (opts.ijkToRAS) {
      this.p2t = patientToTextureFromIjkToRAS(opts.ijkToRAS, dims);
      this.box = volumeAABBFromIjkToRAS(opts.ijkToRAS, dims);
      this.stepMm = Math.min(...spacingFromIjkToRAS(opts.ijkToRAS));
    } else {
      this.p2t = patientToTexture(dims, spacing, center);
      this.box = volumeAABB(dims, spacing, center);
      this.stepMm = Math.min(...spacing);
    }
    this.shade = opts.shade ?? [0.3, 0.75, 0.45, 24];
    this.unit = opts.opacityUnitDistance ?? this.stepMm;
    this.clippable = opts.clippable ?? true;
  }
  uniformFloats() {
    return 24;
  }
  // mat4(16) + params(4) + shade(4)
  aabb() {
    return this.box;
  }
  sampleStep() {
    return this.stepMm;
  }
  /** Swap the baked texture in place (e.g. after re-baking an updated mask). The
   *  geometry is unchanged; the caller refreshes the SceneRenderer bind group. */
  setTexture(tex, destroyPrev = true) {
    if (destroyPrev && this.tex !== tex) this.tex.destroy();
    this.tex = tex;
  }
  get texture() {
    return this.tex;
  }
  structMembers(s) {
    return [
      `  rgba${s}_p2t : mat4x4<f32>,`,
      `  rgba${s}_params : vec4<f32>,`,
      // opacity_unit_distance, _, _, _
      `  rgba${s}_shade : vec4<f32>,`
      // ka, kd, ks, shininess
    ].join("\n");
  }
  declareBindings(s, base) {
    return `@group(0) @binding(${base}) var t_rgba${s} : texture_3d<f32>;`;
  }
  samplingWGSL(s) {
    return (
      /* wgsl */
      `
fn alpha_rgba${s}(wp : vec3<f32>) -> f32 {
  let t4 = u_material.rgba${s}_p2t * vec4<f32>(transform_point_rgba${s}(wp), 1.0);
  return textureSampleLevel(t_rgba${s}, s_lin, clamp(t4.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).a;
}
fn sample_field_rgba${s}(wp : vec3<f32>, rd : vec3<f32>) -> vec4<f32> {
  let t4 = u_material.rgba${s}_p2t * vec4<f32>(transform_point_rgba${s}(wp), 1.0);
  let tex = t4.xyz;
  if (any(tex < vec3<f32>(0.0)) || any(tex > vec3<f32>(1.0))) { return vec4<f32>(0.0); }
  let c = textureSampleLevel(t_rgba${s}, s_lin, tex, 0.0);
  let step = u_material.scene.x;
  let unit = max(u_material.rgba${s}_params.x, 1e-3);
  let opacity = clamp(1.0 - pow(1.0 - clamp(c.a, 0.0, 1.0), step / unit), 0.0, 1.0);
  if (opacity <= 0.001) { return vec4<f32>(0.0); }
  let h = step * 2.0;   // wider central difference -> smoother normals (less shading aliasing on coarse volumes)
  let g = vec3<f32>(
    alpha_rgba${s}(wp + vec3<f32>(h,0,0)) - alpha_rgba${s}(wp - vec3<f32>(h,0,0)),
    alpha_rgba${s}(wp + vec3<f32>(0,h,0)) - alpha_rgba${s}(wp - vec3<f32>(0,h,0)),
    alpha_rgba${s}(wp + vec3<f32>(0,0,h)) - alpha_rgba${s}(wp - vec3<f32>(0,0,h))) / (2.0 * h);
  let glen = length(g);
  let ka = u_material.rgba${s}_shade.x; let kd = u_material.rgba${s}_shade.y;
  let ks = u_material.rgba${s}_shade.z; let sh = u_material.rgba${s}_shade.w;
  var lit_srgb = c.rgb * ka;
  if (glen > 1e-6) {
    var n = g / glen;
    if (dot(n, -rd) < 0.0) { n = -n; }
    let view_dir = normalize(-rd);
    let ldotn = dot(view_dir, n);
    if (ldotn > 0.0) {
      let refl = normalize(2.0 * ldotn * n - view_dir);
      let rdotv = max(0.0, dot(refl, view_dir));
      lit_srgb = c.rgb * (ka + kd * ldotn) + vec3<f32>(ks * pow(rdotv, sh));
    }
  }
  let lit = srgb2physical(clamp(lit_srgb, vec3<f32>(0.0), vec3<f32>(1.0)));
  return vec4<f32>(lit * opacity, opacity);
}`
    );
  }
  fillUniforms(out, off) {
    out.set(this.p2t, off);
    out[off + 16] = this.unit;
    out[off + 20] = this.shade[0];
    out[off + 21] = this.shade[1];
    out[off + 22] = this.shade[2];
    out[off + 23] = this.shade[3];
  }
  bindEntries(_s2, base) {
    return [{ binding: base, resource: this.tex.createView() }];
  }
};

// render/fiducial-field.ts
var MAX2 = 64;
var FiducialField = class {
  kind = "fid";
  bindingCount = 0;
  // procedural — all state lives in the uniform block
  spheres = new Float32Array(MAX2 * 4);
  // (cx,cy,cz,radius)
  colors = new Float32Array(MAX2 * 4);
  // (r,g,b,a)
  n = 0;
  maxR = 0;
  // largest radius in this field (for the skip bound)
  active = -1;
  // hovered/active sphere index (ghost mode: it goes full opacity)
  clippable;
  ghost;
  providesSkip;
  // off in screen-space mode (radius varies with the camera)
  screen;
  sh;
  ka;
  kd;
  ks;
  light;
  constructor(spheres = [], opts = {}) {
    this.setSpheres(spheres);
    this.sh = opts.shininess ?? 80;
    this.ka = opts.kAmbient ?? 0.2;
    this.kd = opts.kDiffuse ?? 0.85;
    this.ks = opts.kSpecular ?? 0.5;
    this.light = opts.lightColor ?? [1, 1, 1];
    this.clippable = opts.clippable ?? true;
    this.ghost = opts.ghost ?? false;
    this.screen = opts.screenSpace ?? false;
    this.providesSkip = true;
  }
  setSpheres(list) {
    this.n = Math.min(list.length, MAX2);
    this.spheres.fill(0);
    this.colors.fill(0);
    this.maxR = 0;
    for (let i = 0; i < this.n; i++) {
      const s = list[i];
      this.spheres.set([s.center[0], s.center[1], s.center[2], s.radius], i * 4);
      this.colors.set(s.color, i * 4);
      this.maxR = Math.max(this.maxR, s.radius);
    }
  }
  get count() {
    return this.n;
  }
  /** Hovered/active sphere (ghost mode only): it renders at full opacity while the others stay
   *  half-visible (partially hidden inside the volume). Pass null/-1 to clear. */
  setActive(i) {
    this.active = i ?? -1;
  }
  get activeIndex() {
    return this.active;
  }
  uniformFloats() {
    return 12 + MAX2 * 4 * 2;
  }
  // params(4)+params2(4)+light(4) + spheres + colors
  sampleStep() {
    return 1;
  }
  aabb() {
    if (this.n === 0) return [[-1, -1, -1], [1, 1, 1]];
    const lo2 = [Infinity, Infinity, Infinity], hi2 = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < this.n; i++) {
      const r = this.screen ? 0 : this.spheres[i * 4 + 3];
      for (let a = 0; a < 3; a++) {
        lo2[a] = Math.min(lo2[a], this.spheres[i * 4 + a] - r);
        hi2[a] = Math.max(hi2[a], this.spheres[i * 4 + a] + r);
      }
    }
    if (this.screen) {
      const diag = Math.hypot(hi2[0] - lo2[0], hi2[1] - lo2[1], hi2[2] - lo2[2]);
      const m = Math.max(40, diag * 0.15);
      for (let a = 0; a < 3; a++) {
        lo2[a] -= m;
        hi2[a] += m;
      }
    }
    return [lo2, hi2];
  }
  structMembers(s) {
    return [
      `  fid${s}_params : vec4<f32>,`,
      // n_spheres, visible, shininess, k_ambient
      `  fid${s}_params2 : vec4<f32>,`,
      // k_diffuse, k_specular, max_radius, _
      `  fid${s}_light : vec4<f32>,`,
      // light_color.rgb, _
      `  fid${s}_spheres : array<vec4<f32>, ${MAX2}>,`,
      `  fid${s}_colors : array<vec4<f32>, ${MAX2}>,`
    ].join("\n");
  }
  declareBindings(_s2, _base) {
    return "";
  }
  bindEntries(_s2, _base) {
    return [];
  }
  // --- empty-space skipping -------------------------------------------------
  // The spheres are an exact SDF, so we can hand the ray-marcher a real distance to
  // leap. Conservative form: nearest-CENTRE distance minus the field's LARGEST radius.
  // Since min_j(d_j) <= d_k and max_r >= r_k for every k, this never exceeds the true
  // min_k(d_k - r_k) — so it can't skip over a sphere — and it costs only squared
  // distances in the loop plus ONE sqrt at the end (cheaper than the sampling loop).
  // (providesSkip is false in screen-space mode — the world radius varies with the camera.)
  skipWGSL(s) {
    if (this.screen) {
      return (
        /* wgsl */
        `
fn skip_fid${s}(wp : vec3<f32>) -> f32 {
  let n = i32(u_material.fid${s}_params.x);
  if (n <= 0) { return 1.0e6; }
  var best = 1.0e12;
  for (var k = 0; k < n; k = k + 1) {
    let sp = u_material.fid${s}_spheres[k];
    if (sp.w <= 0.0) { continue; }
    let r = sp.w * length(u_cam.eye.xyz - sp.xyz) / max(u_cam.size.z, 1.0);
    best = min(best, length(wp - sp.xyz) - r);
  }
  return max(best, 0.0);
}`
      );
    }
    return (
      /* wgsl */
      `
fn skip_fid${s}(wp : vec3<f32>) -> f32 {
  let n = i32(u_material.fid${s}_params.x);
  if (n <= 0) { return 1.0e6; }        // nothing here: unbounded empty space
  var min_d2 = 1.0e12;
  for (var k = 0; k < n; k = k + 1) {
    let sp = u_material.fid${s}_spheres[k];
    if (sp.w <= 0.0) { continue; }
    let dv = wp - sp.xyz;
    min_d2 = min(min_d2, dot(dv, dv));
  }
  return max(sqrt(min_d2) - u_material.fid${s}_params2.z, 0.0);
}`
    );
  }
  samplingWGSL(s) {
    return (
      /* wgsl */
      `
fn sample_field_fid${s}(wp : vec3<f32>, rd : vec3<f32>) -> vec4<f32> {
  // an attached TransformField warps where the spheres appear (slicer_wgpu parity)
  let wp_r = transform_point_fid${s}(wp);
  let n = i32(u_material.fid${s}_params.x);
  var best_depth = -1.0;
  var best_center = vec3<f32>(0.0);
  var best_color = vec4<f32>(0.0);
  var best_k = -1;
  var found = false;
  for (var k = 0; k < n; k = k + 1) {
    let sp = u_material.fid${s}_spheres[k];
    if (sp.w <= 0.0) { continue; }
    // screen-space: sp.w is a PIXEL radius -> world radius = px * distance(eye) / focal_px,
    // so the sphere stays a constant size on screen. Otherwise sp.w is a world radius.
    ${this.screen ? `let r = sp.w * length(u_cam.eye.xyz - sp.xyz) / max(u_cam.size.z, 1.0);` : `let r = sp.w;`}
    let depth = r - length(wp_r - sp.xyz);   // > 0 -> inside this sphere
    if (depth > best_depth) { best_depth = depth; best_center = sp.xyz; best_color = u_material.fid${s}_colors[k]; best_k = k; found = true; }
  }
  if (!found || best_depth <= 0.0) { return vec4<f32>(0.0); }

  let to_wp = wp_r - best_center;
  var n_hat = to_wp / max(length(to_wp), 1e-6);
  if (dot(n_hat, -rd) < 0.0) { n_hat = -n_hat; }
  let view_dir = normalize(-rd);            // headlight (== normalize(ray_origin - wp) for t>0)
  let ldotn = max(dot(view_dir, n_hat), 0.0);
  let refl = normalize(2.0 * ldotn * n_hat - view_dir);
  let rdotv = max(dot(refl, view_dir), 0.0);

  let sh = u_material.fid${s}_params.z;
  let ka = u_material.fid${s}_params.w; let kd = u_material.fid${s}_params2.x; let ks = u_material.fid${s}_params2.y;
  let base = best_color.rgb;
  let highlight = mix(base, u_material.fid${s}_light.rgb, 0.85);
  let lit = base * ka + base * (kd * ldotn) + highlight * (ks * pow(rdotv, sh));
  let col = srgb2physical(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)));
  // Ghost mode: a non-active glyph emits HALF opacity so the ghost compositor leaves 50% of the
  // volume in front of it (partially hidden inside the render); the hovered one emits full (0%
  // residual -> fully visible). Same trick the transform gizmo uses for its active handle.
  ${this.ghost ? `let ghostScale = select(0.5, 1.0, best_k == i32(u_material.fid${s}_params2.w));` : `let ghostScale = 1.0;`}
  let opacity = clamp(best_color.a, 0.0, 1.0) * ghostScale;
  return vec4<f32>(col * opacity, opacity);
}`
    );
  }
  fillUniforms(out, off) {
    out[off + 0] = this.n;
    out[off + 1] = 1;
    out[off + 2] = this.sh;
    out[off + 3] = this.ka;
    out[off + 4] = this.kd;
    out[off + 5] = this.ks;
    out[off + 6] = this.maxR;
    out[off + 7] = this.active;
    out[off + 8] = this.light[0];
    out[off + 9] = this.light[1];
    out[off + 10] = this.light[2];
    out.set(this.spheres, off + 12);
    out.set(this.colors, off + 12 + MAX2 * 4);
  }
};

// render/roi-box-field.ts
var RoiBoxField = class {
  kind = "roi";
  bindingCount = 0;
  // procedural — all state in the uniform block
  clippable = false;
  // the frame sits on the clip planes; never clip it
  providesSkip = true;
  // sparse SDF -> cheap via empty-space skipping
  center;
  half;
  color;
  opacity;
  bar;
  constructor(center, half, opts = {}) {
    this.center = [...center];
    this.half = [...half];
    this.color = opts.color ?? [1, 0.85, 0.25];
    this.opacity = opts.opacity ?? 1;
    this.bar = opts.barHalfMm ?? 1.5;
  }
  /** Update the box (a drag) — caller does scene.syncUniforms() + redraw. */
  setBox(center, half) {
    this.center = [...center];
    this.half = [...half];
  }
  get boxCenter() {
    return [...this.center];
  }
  get boxHalf() {
    return [...this.half];
  }
  uniformFloats() {
    return 16;
  }
  // center(4) + half(4) + color(4) + params(4)
  sampleStep() {
    return Math.max(0.5 * this.bar, 0.25);
  }
  aabb() {
    const m = this.bar + 0.5;
    return [
      [this.center[0] - this.half[0] - m, this.center[1] - this.half[1] - m, this.center[2] - this.half[2] - m],
      [this.center[0] + this.half[0] + m, this.center[1] + this.half[1] + m, this.center[2] + this.half[2] + m]
    ];
  }
  structMembers(s) {
    return [
      `  roi${s}_center : vec4<f32>,`,
      // cx,cy,cz,_
      `  roi${s}_half : vec4<f32>,`,
      // hx,hy,hz,_
      `  roi${s}_color : vec4<f32>,`,
      // rgb, opacity
      `  roi${s}_params : vec4<f32>,`
      // bar_half, _, _, _
    ].join("\n");
  }
  declareBindings() {
    return "";
  }
  bindEntries() {
    return [];
  }
  samplingWGSL(s) {
    return (
      /* wgsl */
      `
fn sd_box_frame${s}(p0 : vec3<f32>, b : vec3<f32>, e : f32) -> f32 {
  let p = abs(p0) - b;
  let q = abs(p + vec3<f32>(e)) - vec3<f32>(e);
  return min(min(
    length(max(vec3<f32>(p.x, q.y, q.z), vec3<f32>(0.0))) + min(max(p.x, max(q.y, q.z)), 0.0),
    length(max(vec3<f32>(q.x, p.y, q.z), vec3<f32>(0.0))) + min(max(q.x, max(p.y, q.z)), 0.0)),
    length(max(vec3<f32>(q.x, q.y, p.z), vec3<f32>(0.0))) + min(max(q.x, max(q.y, p.z)), 0.0));
}
fn sd_roi${s}(wp : vec3<f32>) -> f32 {
  return sd_box_frame${s}(wp - u_material.roi${s}_center.xyz, u_material.roi${s}_half.xyz, u_material.roi${s}_params.x);
}
fn skip_roi${s}(wp : vec3<f32>) -> f32 {
  // exact exterior distance to the bars, minus a bar-width margin (stays conservative)
  return max(sd_roi${s}(wp) - u_material.roi${s}_params.x, 0.0);
}
fn sample_field_roi${s}(wp : vec3<f32>, rd : vec3<f32>) -> vec4<f32> {
  let op0 = u_material.roi${s}_color.a;
  if (op0 <= 0.0) { return vec4<f32>(0.0); }
  let sd = sd_roi${s}(wp);
  // crisp opaque bar: ~1 inside, AA-ramp to 0 across ~half a sample step at the surface
  let op = clamp(0.5 - sd / max(u_material.scene.x, 1e-3), 0.0, 1.0) * op0;
  if (op <= 0.0) { return vec4<f32>(0.0); }
  let col = srgb2physical(u_material.roi${s}_color.rgb);   // flat/unlit, the Slicer widget look
  return vec4<f32>(col * op, op);
}`
    );
  }
  skipWGSL(s) {
    return "";
  }
  // skip_roi<s> is emitted by samplingWGSL above
  fillUniforms(out, off) {
    out[off + 0] = this.center[0];
    out[off + 1] = this.center[1];
    out[off + 2] = this.center[2];
    out[off + 4] = this.half[0];
    out[off + 5] = this.half[1];
    out[off + 6] = this.half[2];
    out[off + 8] = this.color[0];
    out[off + 9] = this.color[1];
    out[off + 10] = this.color[2];
    out[off + 11] = this.opacity;
    out[off + 12] = this.bar;
  }
};

// render/bake.ts
var INIT_WGSL = (
  /* wgsl */
  `
struct U { dims : vec4<u32> };
@group(0) @binding(0) var t_label : texture_3d<u32>;
@group(0) @binding(1) var t_out : texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u_pal : array<vec4<f32>, 256>;
@group(0) @binding(3) var<uniform> u : U;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let label = textureLoad(t_label, vec3<i32>(gid), 0).r;
  let pal = u_pal[label & 255u];
  let present = select(0.0, 1.0, label != 0u);
  textureStore(t_out, vec3<i32>(gid), vec4<f32>(pal.rgb, present * pal.a));
}`
);
var BLUR_WGSL = (
  /* wgsl */
  `
struct U { dims : vec4<u32>, axis_r : vec4<u32>, w : array<vec4<f32>, 4> };  // axis, radius; half-kernel weights
@group(0) @binding(0) var t_in : texture_3d<f32>;
@group(0) @binding(1) var t_out : texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u : U;
fn wt(i : u32) -> f32 { return u.w[i >> 2u][i & 3u]; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let dmax = vec3<i32>(u.dims.xyz) - vec3<i32>(1);
  var av = vec3<i32>(0);
  if (u.axis_r.x == 0u) { av = vec3<i32>(1,0,0); } else if (u.axis_r.x == 1u) { av = vec3<i32>(0,1,0); } else { av = vec3<i32>(0,0,1); }
  let center = textureLoad(t_in, c, 0);
  var asum = center.a * wt(0u);
  let R = i32(u.axis_r.y);
  for (var i = 1; i <= R; i = i + 1) {
    let o = av * i;
    let p1 = clamp(c + o, vec3<i32>(0), dmax);
    let p2 = clamp(c - o, vec3<i32>(0), dmax);
    asum = asum + wt(u32(i)) * (textureLoad(t_in, p1, 0).a + textureLoad(t_in, p2, 0).a);
  }
  textureStore(t_out, c, vec4<f32>(center.rgb, asum));
}`
);
function gaussHalfKernel(sigma) {
  const radius = Math.max(1, Math.min(15, Math.ceil(3 * sigma)));
  const raw = new Float32Array(radius + 1);
  let total = 0;
  for (let i = 0; i <= radius; i++) {
    raw[i] = Math.exp(-(i * i) / (2 * sigma * sigma));
    total += (i === 0 ? 1 : 2) * raw[i];
  }
  const w = new Float32Array(16);
  for (let i = 0; i <= radius; i++) w[i] = raw[i] / total;
  return { radius, w };
}
var ColorizeBaker = class {
  /** `label` is either a CPU labelmap (baker allocates + uploads its own r8uint texture, the classic
   *  path) OR an EXTERNAL r8uint 3D texture the baker only READS (the shared-buffer path used by
   *  `algorithms/EditableSegmentation` — a compute effect writes the label texture on-GPU and the baker
   *  re-colorizes from it, no CPU round-trip). An external texture must be `r8uint` with at least
   *  TEXTURE_BINDING usage; the baker never writes or destroys it. */
  constructor(dev, label, dims) {
    this.dev = dev;
    this.dims = dims;
    const [dx, dy, dz] = dims;
    if (label instanceof GPUTexture) {
      this.labelTex = label;
      this.ownsLabel = false;
    } else {
      this.labelTex = dev.createTexture({ size: dims, dimension: "3d", format: "r8uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      dev.queue.writeTexture({ texture: this.labelTex }, label, { bytesPerRow: dx, rowsPerImage: dy }, dims);
      this.ownsLabel = true;
    }
    this.palBuf = dev.createBuffer({ size: 256 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dimsBuf = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(this.dimsBuf, 0, new Uint32Array([dx, dy, dz, 0]));
    this.initPipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: INIT_WGSL }), entryPoint: "main" } });
    this.blurPipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: BLUR_WGSL }), entryPoint: "main" } });
    this.g = [Math.ceil(dx / 4), Math.ceil(dy / 4), Math.ceil(dz / 4)];
  }
  labelTex;
  ownsLabel;
  // false when the label texture is owned externally (shared buffer)
  scratch;
  // blur ping-pong (lazy; only when sigma > 0)
  palBuf;
  dimsBuf;
  initPipe;
  blurPipe;
  g;
  /** Re-upload an EDITED labelmap (same dims). Follow with bakeInto() to re-colorize into the caller's
   *  existing output textures — an in-place replace (no re-allocation, so a segmentation edit updates
   *  smoothly with no flash). */
  updateLabelmap(labelmap) {
    if (!this.ownsLabel) throw new Error("ColorizeBaker.updateLabelmap: label texture is external (write it via the owner, e.g. a compute effect), then call bakeInto()");
    const [dx, dy] = this.dims;
    this.dev.queue.writeTexture({ texture: this.labelTex }, labelmap, { bytesPerRow: dx, rowsPerImage: dy }, this.dims);
  }
  /** Allocate an output texture sized/typed for this baker's labelmap (caller owns it). */
  output() {
    return this.dev.createTexture({ size: this.dims, dimension: "3d", format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
  }
  /** (Re)colorize into `out` with `palette` (256*4 f32: rgb + presence*opacity) and Gaussian
   *  `sigmaVoxels` (0 = crisp, for the 2D slice overlay). In place — reuses everything resident. */
  bakeInto(out, palette, sigmaVoxels = 1.5) {
    const dev = this.dev, [gx, gy, gz] = this.g, [dx, dy, dz] = this.dims;
    const palData = new Float32Array(256 * 4);
    palData.set(palette.subarray(0, Math.min(palette.length, 256 * 4)));
    dev.queue.writeBuffer(this.palBuf, 0, palData);
    const enc = dev.createCommandEncoder();
    const smooth = sigmaVoxels > 0;
    if (smooth && !this.scratch) this.scratch = this.output();
    const initDst = smooth ? this.scratch : out;
    const initBind = dev.createBindGroup({ layout: this.initPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.labelTex.createView() },
      { binding: 1, resource: initDst.createView() },
      { binding: 2, resource: { buffer: this.palBuf } },
      { binding: 3, resource: { buffer: this.dimsBuf } }
    ] });
    {
      const p = enc.beginComputePass();
      p.setPipeline(this.initPipe);
      p.setBindGroup(0, initBind);
      p.dispatchWorkgroups(gx, gy, gz);
      p.end();
    }
    if (smooth) {
      const s = this.scratch;
      const { radius, w } = gaussHalfKernel(sigmaVoxels);
      const passes = [[s, out, 0], [out, s, 1], [s, out, 2]];
      for (const [src, dst, axis] of passes) {
        const ub = dev.createBuffer({ size: 16 + 16 + 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        dev.queue.writeBuffer(ub, 0, new Uint32Array([dx, dy, dz, 0, axis, radius, 0, 0]));
        dev.queue.writeBuffer(ub, 32, w);
        const b = dev.createBindGroup({ layout: this.blurPipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: src.createView() },
          { binding: 1, resource: dst.createView() },
          { binding: 2, resource: { buffer: ub } }
        ] });
        const p = enc.beginComputePass();
        p.setPipeline(this.blurPipe);
        p.setBindGroup(0, b);
        p.dispatchWorkgroups(gx, gy, gz);
        p.end();
      }
    }
    dev.queue.submit([enc.finish()]);
  }
  destroy() {
    if (this.ownsLabel) this.labelTex.destroy();
    this.scratch?.destroy();
    this.palBuf.destroy();
    this.dimsBuf.destroy();
  }
};

// render/zarr.ts
var ZDT = {
  "<f4": Float32Array,
  "<f8": Float64Array,
  "<i4": Int32Array,
  "<u4": Uint32Array,
  "<i2": Int16Array,
  "<u2": Uint16Array,
  "|i1": Int8Array,
  "|u1": Uint8Array,
  "<i1": Int8Array,
  "<u1": Uint8Array
};
async function inflateDeflate(buf) {
  const ds2 = new DecompressionStream("deflate");
  return await new Response(new Response(buf).body.pipeThrough(ds2)).arrayBuffer();
}
var blobFetch = (url) => fetch(url);
function setBlobFetch(f) {
  blobFetch = f ?? ((url) => fetch(url));
}
function normalizeFetchOpts(onBytesOrOpts, concurrency = 12) {
  if (onBytesOrOpts && typeof onBytesOrOpts === "object") {
    return {
      concurrency: onBytesOrOpts.concurrency ?? 12,
      onBytes: onBytesOrOpts.onBytes,
      onSlab: onBytesOrOpts.onSlab,
      onChunk: onBytesOrOpts.onChunk,
      into: onBytesOrOpts.into
    };
  }
  return {
    concurrency,
    onBytes: onBytesOrOpts
  };
}
async function fetchZarrVolume(blobBase, z2, onBytesOrOpts, concurrency = 12) {
  const opts = normalizeFetchOpts(onBytesOrOpts, concurrency);
  if (opts.onSlab || opts.onChunk || opts.into) {
    return await fetchZarrVolumeF32(blobBase, z2, opts);
  }
  const zv = await fetchZarrVolumeNative(blobBase, z2, opts);
  const data = zv.data instanceof Float32Array ? zv.data : Float32Array.from(zv.data);
  return { data, dims: zv.dims, range: zv.range };
}
async function fetchZarrVolumeNative(blobBase, z2, onBytesOrOpts, concurrency = 12) {
  const opts = normalizeFetchOpts(onBytesOrOpts, concurrency);
  const Ctor = ZDT[z2.dtype] ?? Int16Array;
  return await assembleZarr(blobBase, z2, Ctor, opts);
}
async function fetchZarrVolumeF32(blobBase, z2, opts) {
  const zv = await assembleZarr(blobBase, z2, Float32Array, opts);
  return { data: zv.data, dims: zv.dims, range: zv.range };
}
function orderChunkJobsMprFirst(ncz, ncy, ncx) {
  const midK = ncz - 1 >> 1;
  const midJ = ncy - 1 >> 1;
  const midI = ncx - 1 >> 1;
  const dist = (kk, jj, ii2) => Math.abs(kk - midK) + Math.abs(jj - midJ) + Math.abs(ii2 - midI);
  const byCenter = (a, b) => dist(a[0], a[1], a[2]) - dist(b[0], b[1], b[2]) || a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
  const axial = [];
  const cross2 = [];
  const rest = [];
  for (let kk = 0; kk < ncz; kk++) {
    for (let jj = 0; jj < ncy; jj++) {
      for (let ii2 = 0; ii2 < ncx; ii2++) {
        const job = [kk, jj, ii2];
        if (kk === midK) axial.push(job);
        else if (jj === midJ || ii2 === midI) cross2.push(job);
        else rest.push(job);
      }
    }
  }
  axial.sort(byCenter);
  cross2.sort(byCenter);
  rest.sort(byCenter);
  return [...axial, ...cross2, ...rest];
}
async function assembleZarr(blobBase, z2, Ctor, opts) {
  const [nz, ny, nx] = z2.shape, [cz, cy, cx] = z2.chunks, [ncz, ncy, ncx] = z2.chunkGrid;
  const hashes = z2.chunkHashes;
  const posBase = blobBase + z2.dir + "/" + z2.dataset + "/";
  const chunkUrl = (kk, jj, ii2) => hashes ? blobBase + hashes[kk + "." + jj + "." + ii2] : posBase + kk + "." + jj + "." + ii2;
  const expected = nz * ny * nx;
  let out;
  if (Ctor === Float32Array && opts.into) {
    if (opts.into.length !== expected) {
      throw new Error(`zarr into length ${opts.into.length} != ${expected}`);
    }
    out = opts.into;
  } else {
    out = new Ctor(expected);
  }
  let lo2 = Infinity, hi2 = -Infinity;
  const onBytes = opts.onBytes;
  const onSlab = opts.onSlab;
  const onChunk = opts.onChunk;
  const dims = [nx, ny, nz];
  const remaining = new Int32Array(ncz);
  const perSlab = ncy * ncx;
  for (let kk = 0; kk < ncz; kk++) remaining[kk] = perSlab;
  const jobs = orderChunkJobsMprFirst(ncz, ncy, ncx);
  let idx = 0;
  const worker = async () => {
    while (idx < jobs.length) {
      const [kk, jj, ii2] = jobs[idx++];
      const resp = await blobFetch(chunkUrl(kk, jj, ii2));
      let gz;
      if (resp.body && onBytes) {
        const parts = [];
        const rd = resp.body.getReader();
        let total = 0;
        for (; ; ) {
          const { done, value } = await rd.read();
          if (done) break;
          parts.push(value);
          total += value.byteLength;
          onBytes(value.byteLength);
        }
        const all = new Uint8Array(total);
        let o = 0;
        for (const p of parts) {
          all.set(p, o);
          o += p.byteLength;
        }
        gz = all.buffer;
      } else {
        gz = await resp.arrayBuffer();
        onBytes?.(gz.byteLength);
      }
      const SrcCtor = ZDT[z2.dtype] ?? Int16Array;
      const chunk = new SrcCtor(await inflateDeflate(gz));
      const z0 = kk * cz, y0 = jj * cy, x0 = ii2 * cx;
      const zw = Math.min(cz, nz - z0), yw = Math.min(cy, ny - y0), xw = Math.min(cx, nx - x0);
      for (let zz = 0; zz < zw; zz++) {
        for (let yy = 0; yy < yw; yy++) {
          const src = (zz * cy + yy) * cx;
          const dst = ((z0 + zz) * ny + (y0 + yy)) * nx + x0;
          for (let xx = 0; xx < xw; xx++) {
            const v = chunk[src + xx];
            out[dst + xx] = v;
            if (v < lo2) lo2 = v;
            if (v > hi2) hi2 = v;
          }
        }
      }
      onChunk?.({
        xStart: x0,
        yStart: y0,
        zStart: z0,
        xCount: xw,
        yCount: yw,
        zCount: zw,
        data: out,
        dims
      });
      if (onSlab) {
        remaining[kk]--;
        if (remaining[kk] === 0) {
          onSlab({
            zStart: z0,
            zCount: zw,
            data: out,
            dims
          });
        }
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(opts.concurrency, jobs.length) }, worker));
  return { data: out, dtype: z2.dtype, dims, range: [lo2, hi2] };
}

// logic/transforms.ts
var IDENTITY4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function rowMul(a, b) {
  const out = new Array(16).fill(0);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[r * 4 + k] * b[k * 4 + c];
    out[r * 4 + c] = s;
  }
  return out;
}
var parentTransformId = (n) => (n?.refs?.transform ?? [])[0];
function worldMatrix(transformId, nodes, _seen = /* @__PURE__ */ new Set()) {
  if (!transformId) return IDENTITY4.slice();
  if (_seen.has(transformId)) return IDENTITY4.slice();
  _seen.add(transformId);
  const t = nodes.get(transformId);
  if (!t || t.type !== "transform") return IDENTITY4.slice();
  const local = t.matrix ?? IDENTITY4;
  const parent = parentTransformId(t);
  return parent ? rowMul(worldMatrix(parent, nodes, _seen), local) : local.slice();
}
function worldForNode(node, nodes) {
  return worldMatrix(parentTransformId(node), nodes);
}

// render/scene-volume.ts
function interpTF(tf, s, comps) {
  if (!tf.length) return new Array(comps).fill(0);
  if (s <= tf[0][0]) return tf[0].slice(1, 1 + comps);
  const last = tf[tf.length - 1];
  if (s >= last[0]) return last.slice(1, 1 + comps);
  for (let i = 1; i < tf.length; i++) {
    if (s <= tf[i][0]) {
      const a = tf[i - 1], b = tf[i];
      const u = (s - a[0]) / Math.max(b[0] - a[0], 1e-9);
      return Array.from({ length: comps }, (_, c) => a[1 + c] + u * (b[1 + c] - a[1 + c]));
    }
  }
  return last.slice(1, 1 + comps);
}
function lutFromTransferFunctions(colorTF, opacityTF, clim) {
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const s = clim[0] + i / 255 * (clim[1] - clim[0]);
    const [r, g, b] = interpTF(colorTF, s, 3);
    const [a] = interpTF(opacityTF, s, 1);
    lut[i * 4 + 0] = Math.round(Math.max(0, Math.min(1, r)) * 255);
    lut[i * 4 + 1] = Math.round(Math.max(0, Math.min(1, g)) * 255);
    lut[i * 4 + 2] = Math.round(Math.max(0, Math.min(1, b)) * 255);
    lut[i * 4 + 3] = Math.round(Math.max(0, Math.min(1, a)) * 255);
  }
  return lut;
}

// render/liveops.ts
function pointerKeys(path) {
  return path.replace(/^#/, "").split("/").filter((k) => k.length > 0);
}
function setByPointer(obj, path, value) {
  const keys = pointerKeys(path);
  if (keys.length === 0) return false;
  let cur = obj;
  for (let i = 0; i < keys.length - 1; i++) {
    const k = keys[i];
    if (cur == null || typeof cur !== "object") return false;
    const container = cur;
    let next = container[k];
    if (next == null || typeof next !== "object") {
      next = /^\d+$/.test(keys[i + 1]) ? [] : {};
      container[k] = next;
    }
    cur = next;
  }
  if (cur == null || typeof cur !== "object") return false;
  cur[keys[keys.length - 1]] = value;
  return true;
}
var CMDS = {
  // move one markup control point (SlicerLive drag). World (RAS) coords, latest-wins.
  setControlPoint(node, args) {
    const i = Number(args.index ?? 0);
    const pos = args.position;
    const cps = node.controlPoints;
    if (!cps || !cps[i] || !Array.isArray(pos)) return false;
    cps[i].position = [Number(pos[0]), Number(pos[1]), Number(pos[2])];
    return true;
  },
  setCameraPose(node, args) {
    let did = false;
    for (const k of ["position", "focalPoint", "viewUp"]) {
      if (Array.isArray(args[k])) {
        node[k] = [...args[k]];
        did = true;
      }
    }
    return did;
  },
  setRoi(node, args) {
    let did = false;
    for (const k of ["center", "size"]) {
      if (Array.isArray(args[k])) {
        node[k] = [...args[k]];
        did = true;
      }
    }
    return did;
  }
};
function applyOp(nodes, op) {
  const id = op.id;
  switch (op.op) {
    case "put": {
      if (!op.node || typeof op.node !== "object") return { changed: false, id, kind: "noop" };
      nodes.set(id, { ...op.node, id });
      return { changed: true, id, kind: "put" };
    }
    case "del": {
      return { changed: nodes.delete(id), id, kind: "del" };
    }
    case "patch": {
      const node = nodes.get(id);
      if (!node) return { changed: false, id, kind: "noop", path: op.path };
      const ok = setByPointer(node, op.path, op.value);
      return { changed: ok, id, kind: ok ? "patch" : "noop", path: op.path };
    }
    case "cmd": {
      const node = nodes.get(id);
      const handler = CMDS[op.cmd];
      if (!node || !handler) return { changed: false, id, kind: "noop" };
      const ok = handler(node, op.args ?? {});
      return { changed: ok, id, kind: ok ? "cmd" : "noop" };
    }
    default:
      return { changed: false, id, kind: "noop" };
  }
}

// render/livescene.ts
var LiveScene = class {
  // LiveScene is the pure data model — no wire. A LiveSync (render/livesync.ts) owns the transport,
  // reconnect, coalescing, and echo suppression; it drives this model via receiveEvent()/applyRemote()
  // and observes it via subscribe(). httpBase stays only so managers can resolve blob URLs (blobBase).
  constructor(httpBase, managers) {
    this.httpBase = httpBase;
    this.managers = managers;
  }
  nodes = /* @__PURE__ */ new Map();
  view;
  // the renderer surface managers drive
  /** This place's origin id — stamps local writes and drives echo suppression. */
  origin = "local";
  seq = 0;
  // monotonic _changes sequence
  changeSubs = /* @__PURE__ */ new Set();
  blobBase() {
    return new URL("blobs/", this.httpBase).href;
  }
  find(type) {
    for (const n of this.nodes.values()) if (n.type === type) return n;
    return void 0;
  }
  /** The union of node types the DisplayableManagers care about; LiveSync subscribes the peer to
   *  these on (re)connect. Public because LiveSync — not the model — owns the wire. */
  subscribedTypes() {
    return [...new Set(this.managers.flatMap((m) => m.interestedTypes))];
  }
  /** Union of node types the managers reproduce locally — the peer skips re-streaming their bulk updates
   *  (ARCHITECTURE: consumer-declared local authority over deterministic bulk). LiveSync sends it on subscribe. */
  localBulk() {
    return [...new Set(this.managers.flatMap((m) => m.localBulkTypes ?? []))];
  }
  interested(type) {
    return type ? this.managers.filter((m) => m.interestedTypes.includes(type)) : [];
  }
  // ── local authority + the _changes feed (ARCHITECTURE-2026-08-02) ───────────
  /** Observe the `_changes` feed. Controls use it to reflect current node state; LiveSync uses it to
   *  replicate out. Returns an unsubscribe function. */
  subscribe(cb) {
    this.changeSubs.add(cb);
    return () => {
      this.changeSubs.delete(cb);
    };
  }
  feed(c) {
    for (const cb of this.changeSubs) {
      try {
        cb(c);
      } catch {
      }
    }
  }
  /** LOCAL authoritative write — how a Control or Interactor changes the scene. Applies the op to the
   *  model IMMEDIATELY (optimistic), notifies displayers, emits on the `_changes` feed, and — if the
   *  op is locally-originated — queues it for sync. Standalone and connected run the identical path;
   *  "connected" only adds a LiveSync peer downstream. Echo suppression is by construction: only
   *  local-origin ops are sent out, and an inbound remote op is applied via `applyRemote`, never here. */
  write(op) {
    const stamped = { ...op, origin: op.origin ?? this.origin, v: op.v ?? ++this.seq, role: op.role ?? "human" };
    const r = applyOp(this.nodes, stamped);
    if (r.changed) this.applied(r, stamped.origin, stamped.v, stamped);
  }
  writeMany(ops) {
    for (const o of ops) this.write(o);
  }
  /** Apply an op that arrived from a peer (inbound remote). Same mutation + notify as a local write,
   *  but NOT re-sent (echo suppression). Not yet on the wire path (inbound is still event-shaped in
   *  `handle`); present so Controls/tests exercise the symmetric remote path. */
  applyRemote(op) {
    const r = applyOp(this.nodes, op);
    if (r.changed) this.applied(r, op.origin ?? "remote", op.v ?? ++this.seq);
  }
  /** Fan a completed op mutation out to displayers + the `_changes` feed. `op` is set only for LOCAL
   *  writes so LiveSync replicates them (echo suppression: remote/event changes carry no op). */
  applied(r, origin, v, op) {
    if (r.kind === "del") {
      for (const m of this.managers) m.onNodeRemoved?.(r.id, this);
      this.feed({ id: r.id, kind: "remove", origin, v, op });
      return;
    }
    const node = this.nodes.get(r.id);
    if (!node) return;
    for (const m of this.interested(node.type)) m.onNodeAdded?.(node, this);
    this.feed({ id: r.id, type: node.type, kind: "upsert", origin, v, node, op });
  }
  // ── replay (SceneRecorder) ──────────────────────────────────────────────────
  // When applyView is false the model + _changes feed still update on every inbound event (so a
  // SceneRecorder keeps a LOSSLESS record of the live session), but the DISPLAYABLE MANAGERS are NOT
  // driven — the view is under replay control via applySnapshot(). Resuming live re-attaches the view
  // to the current model. This is the DVR head advancing while you scrub the past.
  applyView = true;
  /** Enter/leave replay mode. Leaving does NOT itself repaint — the caller reconciles the view to the
   *  desired node map (present or a seeked past) with applySnapshot(). */
  setLive(on2) {
    this.applyView = on2;
  }
  /** Drive the displayable managers so the VIEW reflects `target` (a full node map — the live model, or
   *  a SceneRecorder.seek(t) reconstruction), reconciling from `from` (what the view currently shows).
   *  Emits nothing on the _changes feed and does NOT mutate this.nodes — replay must not pollute the
   *  recording nor the authoritative model. Removes gone nodes, (re)adds new/changed ones (JSON-diff);
   *  heavy GPU resources keyed by id are reused by the managers, so scrubbing is cheap after the first
   *  fetch. */
  async applySnapshot(target, from, opts) {
    for (const [id, node] of from) {
      if (!target.has(id)) for (const m of this.interested(node.type)) m.onNodeRemoved?.(id, this);
    }
    for (const [id, node] of target) {
      const prev = from.get(id);
      if (!prev || JSON.stringify(prev) !== JSON.stringify(node) || opts?.force?.(node)) {
        for (const m of this.interested(node.type)) await m.onNodeAdded?.(node, this);
      }
    }
  }
  /** Apply one inbound event from a peer (via LiveSync). Slicer sends event-shaped changes (NodeAdded
   *  upsert / NodeRemoved / CameraModified / SceneClosed); each mutates the model, notifies displayers
   *  (unless replay froze the view), and emits on the `_changes` feed with a remote origin so Controls
   *  and the SceneRecorder reflect it. */
  /** A node this place created with a provisional id (put) now has the peer's real id: move it. */
  aliasNode(clientId, realId) {
    if (clientId === realId) return;
    const node = this.nodes.get(clientId);
    if (!node) return;
    this.nodes.delete(clientId);
    node.id = realId;
    this.nodes.set(realId, node);
    if (this.applyView) for (const m of this.interested(node.type)) {
      m.onNodeRemoved?.(clientId, this);
      m.onNodeAdded?.(node, this);
    }
    this.feed({ id: clientId, type: node.type, kind: "remove", origin: "remote", v: ++this.seq });
    this.feed({ id: realId, type: node.type, kind: "upsert", origin: "remote", v: ++this.seq, node });
  }
  async receiveEvent(ev, origin = "remote") {
    const e = ev.event;
    const live = this.applyView;
    if (e === "NodeAdded" && ev.node) {
      const node = ev.node;
      if (typeof ev.clientId === "string") this.aliasNode(ev.clientId, node.id);
      this.nodes.set(node.id, node);
      if (live) for (const m of this.interested(node.type)) await m.onNodeAdded?.(node, this);
      this.feed({ id: node.id, type: node.type, kind: "upsert", origin, v: ++this.seq, node });
    } else if (e === "NodeRemoved") {
      const id = ev.sourceId;
      const node = this.nodes.get(id);
      this.nodes.delete(id);
      if (live) for (const m of this.interested(node?.type)) m.onNodeRemoved?.(id, this);
      this.feed({ id, type: node?.type, kind: "remove", origin, v: ++this.seq });
    } else if (e === "SnapshotComplete") {
    } else if (e === "Snapshot") {
      const scene = ev.scene;
      const incoming = scene?.nodes;
      if (!incoming || typeof incoming !== "object") return;
      const nextIds = new Set(Object.keys(incoming));
      for (const [id, node] of [...this.nodes.entries()]) {
        if (nextIds.has(id)) continue;
        this.nodes.delete(id);
        if (live) for (const m of this.interested(node.type)) m.onNodeRemoved?.(id, this);
        this.feed({ id, type: node.type, kind: "remove", origin, v: ++this.seq });
      }
      for (const [id, node] of Object.entries(incoming)) {
        if (!node || typeof node !== "object") continue;
        const n = { ...node, id: node.id || id };
        this.nodes.set(n.id, n);
        if (live) for (const m of this.interested(n.type)) await m.onNodeAdded?.(n, this);
        this.feed({ id: n.id, type: n.type, kind: "upsert", origin, v: ++this.seq, node: n });
      }
    } else if (e === "SceneClosed") {
      this.nodes.clear();
      if (live) for (const m of this.managers) m.onSceneClosed?.(this);
      this.feed({ id: "", kind: "reset", origin, v: ++this.seq });
    } else if (e === "SegmentationDisplayModified") {
      const id = ev.sourceId;
      const node = this.nodes.get(id);
      const disp = ev.display;
      if (node && disp) {
        for (const k of ["visible", "opacity", "fill2D", "outline2D", "segments"]) {
          if (k in disp) node[k] = disp[k];
        }
      }
      if (live) for (const m of this.interested("segmentation")) await m.onEvent?.(ev, this);
      if (node) this.feed({ id, type: "segmentation", kind: "upsert", origin, v: ++this.seq, node });
    } else if (e === "CameraModified") {
      const id = ev.sourceId;
      const node = this.nodes.get(id);
      if (node) {
        for (const k of ["position", "focalPoint", "viewUp", "viewAngle", "parallelScale"]) {
          if (k in ev) node[k] = ev[k];
        }
      }
      if (live) for (const m of this.interested("camera")) await m.onEvent?.(ev, this);
      if (node) this.feed({ id, type: "camera", kind: "upsert", origin, v: ++this.seq, node });
    } else {
      const t = this.nodes.get(ev.sourceId)?.type;
      if (live) for (const m of this.interested(t)) await m.onEvent?.(ev, this);
    }
  }
};
var CameraDisplayableManager = class {
  interestedTypes = ["camera"];
  last;
  apply(n, scene) {
    this.last = {
      position: n.position,
      focalPoint: n.focalPoint,
      viewUp: n.viewUp,
      viewAngle: n.viewAngle,
      parallelScale: n.parallelScale
    };
    scene.view?.setCamera(this.last);
  }
  onNodeAdded(node, scene) {
    this.apply(node, scene);
  }
  onEvent(ev, scene) {
    if (ev.event === "CameraModified") this.apply(ev, scene);
  }
};
var MarkupsDisplayableManager = class {
  interestedTypes = ["markup"];
  nodes = /* @__PURE__ */ new Map();
  // markup id -> its full node (points + geometry)
  field;
  // control-point glyphs (all markup types)
  lines;
  // connectors: line/angle/curve/plane geometry
  spheresFor(node) {
    const col = node.color ?? [1, 0.85, 0.2, 1];
    const cps = node.controlPoints ?? [];
    const radius = 3 * (node.glyphScale ?? 3);
    return cps.map((cp) => ({ center: cp.position, radius, color: [col[0], col[1], col[2], 1] }));
  }
  roiFields = /* @__PURE__ */ new Map();
  syncRoi(node, scene) {
    const key = "roi:" + node.id;
    if (node.markupType !== "roi" || node.visible === false || !node.center || !node.size) {
      if (this.roiFields.delete(key)) scene.view?.removeField(key);
      return;
    }
    const c = node.center, sz = node.size, col = node.color ?? [1, 0.85, 0.25, 1];
    const f = new RoiBoxField(c, [sz[0] / 2, sz[1] / 2, sz[2] / 2], { color: [col[0], col[1], col[2]] });
    this.roiFields.set(key, f);
    scene.view?.setField(key, f);
  }
  /** Per-type connector geometry: line/angle connect consecutive control points; curve/closedCurve
   *  use Slicer's interpolated world polyline (closedCurve wraps); plane uses its 4 world corners. */
  segmentsFor(node) {
    const t = node.markupType;
    const col = node.color ?? [1, 0.85, 0.2, 1];
    const c = [col[0], col[1], col[2], 1];
    const cps = (node.controlPoints ?? []).map((p) => p.position);
    let pts = [];
    let closed = false;
    if (t === "line" || t === "angle") pts = cps;
    else if (t === "curve") pts = node.linePoints ?? cps;
    else if (t === "closedCurve") {
      pts = node.linePoints ?? cps;
      closed = true;
    } else if (t === "plane") {
      pts = node.corners ?? [];
      closed = true;
    } else return [];
    const segs = [];
    for (let i = 0; i + 1 < pts.length; i++) segs.push({ a: pts[i], b: pts[i + 1], radius: 3, color: c });
    if (closed && pts.length > 2) segs.push({ a: pts[pts.length - 1], b: pts[0], radius: 3, color: c });
    return segs;
  }
  allSpheres() {
    const out = [];
    for (const n of this.nodes.values()) out.push(...this.spheresFor(n));
    return out;
  }
  allSegments() {
    const out = [];
    for (const n of this.nodes.values()) out.push(...this.segmentsFor(n));
    return out;
  }
  /** 2D overlay items for the slice views: every control point (drawn in-plane or as a projection)
   *  and the connector polylines. Mirrors vtkMRMLMarkupsDisplayableManager's slice-view actors. */
  overlayItems() {
    const out = [];
    for (const n of this.nodes.values()) {
      const col = n.color ?? [1, 0.85, 0.2, 1];
      if (n.visible === false) continue;
      const cps = n.controlPoints ?? [];
      const radiusPx = Math.max(2, (n.glyphScale ?? 3) * 2);
      for (const cp of cps) out.push({ kind: "point", ras: cp.position, color: col, radiusPx, label: cp.label });
      const segs = this.segmentsFor(n);
      if (segs.length) {
        const pts = [segs[0].a, ...segs.map((sg) => sg.b)];
        out.push({ kind: "polyline", points: pts, color: col, widthPx: 2 });
      }
    }
    return out;
  }
  refresh(scene, first = false) {
    scene.view?.setOverlay?.("*", "markups", this.overlayItems());
    if (!this.field) this.field = new FiducialField(this.allSpheres(), { screenSpace: true, ghost: true, shininess: 60 });
    else this.field.setSpheres(this.allSpheres());
    if (!this.lines) this.lines = new CapsuleField(this.allSegments(), { screenSpace: true, ghost: true });
    else this.lines.setSegments(this.allSegments());
    if (first) {
      scene.view?.setField("markups", this.field);
      scene.view?.setField("markupLines", this.lines);
    } else scene.view?.redraw();
  }
  /** Every draggable control point, in the same order allSpheres() lays them out. */
  handles() {
    const out = [];
    for (const n of this.nodes.values()) {
      const cps = n.controlPoints ?? [];
      cps.forEach((cp, index) => out.push({ id: n.id, index, ras: cp.position }));
    }
    return out;
  }
  /** Optimistic local move of one control point (SlicerLive drag), before Slicer echoes it back.
   *  Keeps the glyph under the cursor with zero round-trip latency. */
  moveLocal(id, index, ras, scene) {
    const n = this.nodes.get(id);
    const cps = n?.controlPoints;
    if (!cps || !cps[index]) return;
    cps[index].position = [...ras];
    this.refresh(scene);
  }
  // ORIGIN / echo suppression: while the user drags a control point locally, that point is the
  // authoritative source — suppress the (stale) echo of our OWN move so it can't rubber-band the
  // glyph. AUTO-EXPIRING (a deadline, not a sticky flag): `touch()` on every drag frame extends the
  // window; it lapses ~holdMs after the last move, so a drag that never cleanly releases (pointer
  // left the canvas, JS error) can NEVER permanently freeze a markup's sync. The final flushed op's
  // echo (arriving well within holdMs) then re-syncs to the same value → no jump.
  heldUntil = /* @__PURE__ */ new Map();
  // "id:index" -> perf.now() deadline
  touch(id, index, holdMs = 250) {
    this.heldUntil.set(id + ":" + index, performance.now() + holdMs);
  }
  isHeld(id) {
    const now = performance.now();
    let held = false;
    for (const [k, t] of this.heldUntil) {
      if (t <= now) {
        this.heldUntil.delete(k);
        continue;
      }
      if (k.startsWith(id + ":")) held = true;
    }
    return held;
  }
  onNodeAdded(node, scene) {
    if (node.markupType === "roi") {
      this.syncRoi(node, scene);
      return;
    }
    if (node.visible === false) {
      this.onNodeRemoved(node.id, scene);
      return;
    }
    if (this.isHeld(node.id)) return;
    const first = !this.field;
    this.nodes.set(node.id, node);
    this.refresh(scene, first);
  }
  onNodeRemoved(id, scene) {
    if (this.roiFields.delete("roi:" + id)) scene.view?.removeField("roi:" + id);
    if (!this.nodes.delete(id) || !this.field) return;
    this.field.setSpheres(this.allSpheres());
    this.lines?.setSegments(this.allSegments());
    scene.view?.setOverlay?.("*", "markups", this.overlayItems());
    scene.view?.redraw();
  }
  onSceneClosed(scene) {
    this.nodes.clear();
    this.field = void 0;
    this.lines = void 0;
    scene.view?.setOverlay?.("*", "markups", []);
    scene.view?.removeField("markupLines");
    scene.view?.removeField("markups");
  }
};
var RoiCropDisplayableManager = class {
  interestedTypes = ["volumeRenderingDisplay", "markup"];
  crop = { enabled: false };
  rois = /* @__PURE__ */ new Map();
  recompute(scene) {
    const r = this.crop.enabled && this.crop.roiId ? this.rois.get(this.crop.roiId) : void 0;
    if (r) {
      const c = r.center, s = r.size;
      scene.view?.setClipBox(
        [c[0] - s[0] / 2, c[1] - s[1] / 2, c[2] - s[2] / 2],
        [c[0] + s[0] / 2, c[1] + s[1] / 2, c[2] + s[2] / 2]
      );
    } else {
      scene.view?.setClipBox(null);
    }
  }
  onNodeAdded(node, scene) {
    if (node.type === "volumeRenderingDisplay") {
      this.crop = { enabled: !!node.cropEnabled, roiId: node.refs?.roi?.[0] };
      this.recompute(scene);
    } else if (node.markupType === "roi" && node.center && node.size) {
      this.rois.set(node.id, { center: node.center, size: node.size });
      this.recompute(scene);
    }
  }
  onNodeRemoved(id, scene) {
    let changed = this.rois.delete(id);
    if (this.crop.roiId === id) {
      this.crop.roiId = void 0;
      changed = true;
    }
    if (changed) this.recompute(scene);
  }
  onSceneClosed(scene) {
    this.crop = { enabled: false };
    this.rois.clear();
    scene.view?.setClipBox(null);
  }
};
var SliceDisplayableManager = class _SliceDisplayableManager {
  interestedTypes = ["view"];
  static ORIENT = { Axial: "axial", Coronal: "coronal", Sagittal: "sagittal" };
  onNodeAdded(node, scene) {
    if (node.type !== "view" || node.kind !== "slice") return;
    const cell = node.layoutName;
    const m = node.sliceToRAS;
    if (!cell || !m || m.length < 16) return;
    const col = (c) => [m[c], m[4 + c], m[8 + c]];
    const norm2 = (v) => {
      const l = Math.hypot(v[0], v[1], v[2]) || 1;
      return [v[0] / l, v[1] / l, v[2] / l];
    };
    const nDir = norm2(col(2)), uDir = norm2(col(0)), vDir = norm2(col(1));
    const trans = [m[3], m[7], m[11]];
    const ax = [Math.abs(nDir[0]), Math.abs(nDir[1]), Math.abs(nDir[2])];
    const axis = ax[2] >= ax[0] && ax[2] >= ax[1] ? 2 : ax[1] >= ax[0] ? 1 : 0;
    const orient = _SliceDisplayableManager.ORIENT[node.orientation] ?? ["sagittal", "coronal", "axial"][axis];
    const anatomical = ax[axis] > 0.9999 && _SliceDisplayableManager.ORIENT[node.orientation] !== void 0;
    const plane = anatomical ? { orient, posMm: trans[axis] } : { orient, posMm: trans[0] * nDir[0] + trans[1] * nDir[1] + trans[2] * nDir[2], basis: { uDir, vDir, nDir } };
    const fov = node.fieldOfView;
    if (fov && fov.length >= 2 && fov[0] > 0 && fov[1] > 0) {
      plane.centerRAS = trans;
      plane.fovX = fov[0];
      plane.fovY = fov[1];
    }
    plane.chrome = { orientationMarkerType: node.orientationMarkerType ?? 0, orientationMarkerSize: node.orientationMarkerSize ?? 20, rulerType: node.rulerType ?? 0 };
    scene.view?.setSlicePlane(cell, plane);
  }
};
var LayoutDisplayableManager = class {
  interestedTypes = ["layout"];
  onNodeAdded(node, scene) {
    if (node.type === "layout") scene.view?.setLayout(node.arrangementName ?? "fourUp");
  }
};
function slice2DOpacities(node, visible) {
  if (!visible) return [0, 0];
  const overall = typeof node.opacity === "number" ? node.opacity : 1;
  const f = node.fill2D;
  const o = node.outline2D;
  const fill = f?.visible ?? true ? overall * (f?.opacity ?? 0.5) : 0;
  const outline = o?.visible ?? true ? overall * (o?.opacity ?? 1) : 0;
  return [fill, outline];
}
function segPalette(segments) {
  const p = new Float32Array(256 * 4);
  for (const s of segments ?? []) {
    const lv = s.labelValue;
    if (lv > 0 && lv < 256 && s.visible !== false) {
      p[lv * 4] = s.color[0];
      p[lv * 4 + 1] = s.color[1];
      p[lv * 4 + 2] = s.color[2];
      p[lv * 4 + 3] = 1;
    }
  }
  return p;
}
function paletteKey(segments) {
  return (segments ?? []).map((s) => `${s.labelValue}:${s.color.map((x) => x.toFixed(3)).join(",")}:${s.visible !== false}`).join("|");
}
var SegmentationDisplayableManager = class {
  // is the 3D field currently in the view?
  constructor(dev, sigma = 1.5, onBytes) {
    this.dev = dev;
    this.sigma = sigma;
    this.onBytes = onBytes;
  }
  interestedTypes = ["segmentation"];
  baker;
  // resident: labelmap uploaded once, re-colorized in place
  overlayTex;
  // crisp (σ=0) — 2D slice overlay (reused every re-bake)
  volTex;
  // smoothed (σ) — 3D colorized field (reused every re-bake)
  field;
  segId;
  blobBaseHref = "";
  dims;
  ijkToRAS;
  palKey = "";
  added = false;
  zarrSig = "";
  // signature of the current labelmap; changes when the segmentation is EDITED
  async onNodeAdded(node, scene) {
    if (node.type !== "segmentation" || !node.zarr) return;
    const sig = JSON.stringify(node.zarr);
    if (this.baker && sig === this.zarrSig) {
      this.apply(node, scene);
      return;
    }
    this.blobBaseHref = scene.blobBase();
    const zv = await fetchZarrVolume(this.blobBaseHref, node.zarr, this.onBytes);
    const lab = Uint8Array.from(zv.data);
    const segments = node.segments ?? [];
    const sameDims = !!(this.baker && this.dims && zv.dims[0] === this.dims[0] && zv.dims[1] === this.dims[1] && zv.dims[2] === this.dims[2]);
    if (this.baker && sameDims) {
      this.zarrSig = sig;
      this.baker.updateLabelmap(lab);
      this.palKey = paletteKey(segments);
      this.recolorize(segPalette(segments));
      this.apply(node, scene);
      return;
    }
    if (this.baker) this.reset(scene);
    this.zarrSig = sig;
    this.segId = node.id;
    this.dims = zv.dims;
    this.ijkToRAS = node.ijkToRAS;
    this.baker = new ColorizeBaker(this.dev, lab, zv.dims);
    this.overlayTex = this.baker.output();
    this.volTex = this.baker.output();
    this.palKey = paletteKey(segments);
    this.recolorize(segPalette(segments));
    this.field = new RGBAVolumeField(this.volTex, zv.dims, [1, 1, 1], { ijkToRAS: this.ijkToRAS, shade: [0.3, 0.78, 0.5, 28], clippable: false });
    this.apply(node, scene);
  }
  /** Live display change (opacity/visibility/colour). Re-colorize IN PLACE only when the palette
   *  (colour or per-segment visibility) changed — the bulk labelmap is never re-fetched or
   *  re-uploaded, and the output textures are reused, so the 3D field + slice bind stay valid
   *  (a redraw suffices). Opacity-only changes skip the bake entirely. */
  onEvent(ev, scene) {
    if (ev.event !== "SegmentationDisplayModified" || ev.sourceId !== this.segId || !this.baker) return;
    const d = ev.display;
    const key = paletteKey(d.segments ?? []);
    if (key !== this.palKey) {
      this.palKey = key;
      this.recolorize(segPalette(d.segments ?? []));
    }
    this.apply(d, scene);
  }
  /** Push current visibility/opacity to the view. The output textures are stable objects, so an
   *  in-place re-colorize needs only a redraw (no setField / scene rebuild); visibility flips add
   *  or remove the 3D field. */
  apply(disp, scene) {
    const visible = disp.visible !== false;
    const [fill, outline] = slice2DOpacities(disp, visible);
    scene.view?.setSegmentationOverlay(visible ? this.overlayTex : null, fill, outline);
    if (visible) {
      if (!this.added) {
        scene.view?.setField("seg:" + this.segId, this.field);
        this.added = true;
      } else scene.view?.redraw();
    } else if (this.added) {
      scene.view?.removeField("seg:" + this.segId);
      this.added = false;
    }
  }
  /** Re-colorize the crisp slice overlay (σ=0) + smoothed 3D field (σ) into the resident output
   *  textures from a palette — reuses the baker's uploaded labelmap, pipelines, and scratch. */
  recolorize(palette) {
    this.baker.bakeInto(this.overlayTex, palette, 0);
    this.baker.bakeInto(this.volTex, palette, this.sigma);
  }
  onNodeRemoved(id, scene) {
    if (id === this.segId) this.reset(scene);
  }
  onSceneClosed(scene) {
    this.reset(scene);
  }
  reset(scene) {
    if (this.added && this.segId) scene.view?.removeField("seg:" + this.segId);
    scene.view?.setSegmentationOverlay(null, 0, 0);
    this.baker?.destroy();
    this.overlayTex?.destroy();
    this.volTex?.destroy();
    this.baker = void 0;
    this.overlayTex = void 0;
    this.volTex = void 0;
    this.field = void 0;
    this.segId = void 0;
    this.added = false;
    this.palKey = "";
    this.zarrSig = "";
  }
};
var VolumeRenderingDisplayableManager = class {
  constructor(dev, onBytes) {
    this.dev = dev;
    this.onBytes = onBytes;
  }
  interestedTypes = ["image", "volumeRenderingDisplay", "scalarVolumeDisplay", "transferFunction", "transform"];
  image;
  tf;
  scalarDisp;
  vrDisplayId;
  vrVisible = false;
  zv;
  field;
  building = false;
  blobBaseHref = "";
  view;
  async onNodeAdded(node, scene) {
    this.blobBaseHref = scene.blobBase();
    this.view = scene.view;
    if (node.type === "image") {
      if (!this.image) this.image = node;
      else if (node.id === this.image.id) {
        const prev = this.effSig;
        this.image = node;
        this.replaceGeometry(scene, prev);
      }
    } else if (node.type === "transform") {
      if (this.image) {
        const prev = this.effSig;
        this.replaceGeometry(scene, prev);
      }
    } else if (node.type === "volumeRenderingDisplay") {
      this.vrDisplayId = node.id;
      this.vrVisible = !!node.visible;
    } else if (node.type === "transferFunction") {
      this.tf = node;
      this.reLUT();
    } else if (node.type === "scalarVolumeDisplay") {
      const mine = (this.image?.refs?.display ?? []).includes(node.id);
      if (mine || !this.image && !this.scalarDisp) {
        this.scalarDisp = node;
        this.pushVolume();
      }
    }
    await this.ensureField(scene);
    this.view?.showVolume3D(!!(this.field && this.vrVisible));
  }
  onEvent() {
  }
  onNodeRemoved(id, scene) {
    if (id === this.image?.id) {
      this.reset(scene);
      return;
    }
    if (id === this.vrDisplayId) {
      this.vrVisible = false;
      this.vrDisplayId = void 0;
      scene.view?.showVolume3D(false);
    }
  }
  onSceneClosed(scene) {
    this.reset(scene);
  }
  reset(scene) {
    this.image = this.tf = this.scalarDisp = void 0;
    this.zv = this.field = void 0;
    this.vrVisible = false;
    this.vrDisplayId = void 0;
    scene.view?.setVolumeField(null);
    scene.view?.showVolume3D(false);
  }
  effSig = "";
  /** world(transform chain) · base ijkToRAS for the tracked image. */
  effIjk(scene) {
    return this.image ? rowMul(worldForNode(this.image, scene.nodes), this.image.ijkToRAS) : IDENTITY4.slice();
  }
  /** Re-place the 3D field if the effective geometry changed (base moved or a transform edited). */
  replaceGeometry(scene, prevSig) {
    const eff = this.effIjk(scene);
    const sig = JSON.stringify(eff);
    if (sig === prevSig) return;
    this.effSig = sig;
    if (this.field) {
      this.field.setIjkToRAS(eff);
      this.view?.setVolumeField(this.field, this.wl());
      this.view?.redraw();
    }
  }
  wl() {
    const range = this.zv?.range ?? [0, 1];
    const win = this.scalarDisp?.window ?? range[1] - range[0];
    const lev = this.scalarDisp?.level ?? (range[0] + range[1]) / 2;
    return { win, lev };
  }
  pushVolume() {
    if (this.field) this.view?.setVolumeField(this.field, this.wl());
  }
  reLUT() {
    if (this.field && this.zv) {
      this.field.setLUT(this.buildLUT(this.zv.range));
      this.view?.redraw();
    }
  }
  async ensureField(scene) {
    if (this.field || this.building || !this.image?.zarr) return;
    this.building = true;
    if (!this.zv) this.zv = await fetchZarrVolume(this.blobBaseHref, this.image.zarr, this.onBytes);
    const ijkToRAS = rowMul(worldForNode(this.image, scene.nodes), this.image.ijkToRAS);
    this.effSig = JSON.stringify(ijkToRAS);
    this.field = new ImageField(this.dev, this.zv.data, this.zv.dims, [1, 1, 1], this.buildLUT(this.zv.range), { clim: this.zv.range, ijkToRAS, shade: [0.25, 0.75, 0.5, 24] });
    this.building = false;
    this.view?.setVolumeField(this.field, this.wl());
  }
  // 256-entry rgba8 LUT sampled across the DATA RANGE (clim is fixed to that range).
  buildLUT(range) {
    const cs2 = this.tf?.colorStops;
    const os2 = this.tf?.scalarOpacity;
    if (cs2?.length && os2?.length) {
      const colorTF = cs2.map((s) => [s.value, s.rgba[0], s.rgba[1], s.rgba[2]]);
      const opac = os2.map((s) => [s.value, s.opacity]);
      return lutFromTransferFunctions(colorTF, opac, range);
    }
    const win = this.scalarDisp?.window ?? range[1] - range[0];
    const lev = this.scalarDisp?.level ?? (range[0] + range[1]) / 2;
    const lo2 = lev - win / 2, hi2 = lev + win / 2;
    const lut = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      const v = range[0] + i / 255 * (range[1] - range[0]);
      const g = Math.max(0, Math.min(1, (v - lo2) / Math.max(hi2 - lo2, 1e-6)));
      lut[i * 4] = lut[i * 4 + 1] = lut[i * 4 + 2] = Math.round(g * 255);
      lut[i * 4 + 3] = Math.round(Math.max(0, Math.min(1, (g - 0.15) / 0.85)) * 200);
    }
    return lut;
  }
};

// algorithms/geom.ts
function transpose42(m) {
  const o = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) o[c * 4 + r] = m[r * 4 + c];
  return o;
}
function spacingFromIjkToRAS2(ijkToRAS) {
  const col = (c) => Math.hypot(ijkToRAS[c], ijkToRAS[4 + c], ijkToRAS[8 + c]);
  return [col(0), col(1), col(2)];
}

// algorithms/editable-segmentation.ts
var EditableSegmentation = class {
  dims;
  ijkToRAS;
  device;
  // effects (algorithms/effects/*) build their own pipelines against this
  labelTex;
  // master (r32uint, STORAGE) — the shared buffer effects write
  dirtyCbs = [];
  constructor(device, dims, opts) {
    this.device = device;
    this.dims = dims;
    this.ijkToRAS = Array.from(opts.ijkToRAS);
    this.labelTex = device.createTexture({
      size: dims,
      dimension: "3d",
      format: "r32uint",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC
    });
  }
  /** The master labelmap (r32uint storage). The logic layer reads it (to derive a presence texture);
   *  editing effects write it on-GPU (A-1+). */
  masterTexture() {
    return this.labelTex;
  }
  /** Register a callback fired after any edit — the logic layer rebakes + redraws. Returns an
   *  unsubscribe (so a logic can be swapped/disposed without leaking a stale rebake). */
  onDirty(cb) {
    this.dirtyCbs.push(cb);
    return () => {
      const i = this.dirtyCbs.indexOf(cb);
      if (i >= 0) this.dirtyCbs.splice(i, 1);
    };
  }
  /** Signal that the master was edited (effects call this after writing the label texture on-GPU). */
  markDirty() {
    for (const cb of this.dirtyCbs) cb();
  }
  /** Voxel spacing (mm) from the geometry — for mm↔voxel effect params. */
  spacingMm() {
    return spacingFromIjkToRAS2(this.ijkToRAS);
  }
  /** Load a full labelmap (ids 0..255) from CPU into the master, then notify. */
  loadLabelmap(data) {
    const [dx, dy, dz] = this.dims;
    const u32 = data instanceof Uint32Array ? data : Uint32Array.from(data);
    this.device.queue.writeTexture({ texture: this.labelTex }, u32, { bytesPerRow: dx * 4, rowsPerImage: dy }, [dx, dy, dz]);
    this.markDirty();
  }
  /** Region-limited labelmap write (`lo`/`size` in label-grid ijk; data x-fastest, tightly packed).
   *  Deliberately NO dirty notification: the caller pairs it with a region-limited rebake
   *  (SegmentationLogic.rebakeShellRegion) — an onDirty full rebake would defeat the point. */
  writeLabelRegion(data, lo2, size) {
    this.device.queue.writeTexture(
      { texture: this.labelTex, origin: lo2 },
      data,
      { bytesPerRow: size[0] * 4, rowsPerImage: size[1] },
      size
    );
  }
  /** Read the master labelmap back to CPU (ids per voxel, x-fastest). Handles WebGPU's 256-byte
   *  bytesPerRow alignment. For tests + zarr serialization (A-7); not on the interactive path. */
  async readLabelmap() {
    const [dx, dy, dz] = this.dims;
    const bpr = Math.ceil(dx * 4 / 256) * 256;
    const rowU32 = bpr / 4;
    const buf = this.device.createBuffer({ size: bpr * dy * dz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.labelTex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: dy }, [dx, dy, dz]);
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint32Array(buf.getMappedRange());
    const out = new Uint32Array(dx * dy * dz);
    for (let z2 = 0; z2 < dz; z2++) for (let y2 = 0; y2 < dy; y2++) {
      const src = (z2 * dy + y2) * rowU32, dst = (z2 * dy + y2) * dx;
      for (let x = 0; x < dx; x++) out[dst + x] = padded[src + x];
    }
    buf.unmap();
    buf.destroy();
    return out;
  }
  destroy() {
    this.labelTex.destroy();
  }
};

// algorithms/effects/paint.ts
var PAINT_WGSL = (
  /* wgsl */
  `
struct U {
  ijkToRAS : mat4x4<f32>,   // column-major (transpose of the row-major host matrix)
  dims     : vec4<u32>,
  params   : vec4<f32>,     // x=radiusMm, y=id, z=mode(0 add/1 remove), w=pointCount
};
@group(0) @binding(0) var t_label : texture_storage_3d<r32uint, write>;
@group(0) @binding(1) var<uniform> u : U;
@group(0) @binding(2) var<storage, read> pts : array<vec4<f32>>;   // xyz = RAS sample points

fn seg_dist(p : vec3<f32>, a : vec3<f32>, b : vec3<f32>) -> f32 {
  let ab = b - a;
  let denom = max(dot(ab, ab), 1e-8);
  let t = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
  return length(p - (a + t * ab));
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let p = (u.ijkToRAS * vec4<f32>(vec3<f32>(gid), 1.0)).xyz;
  let n = u32(u.params.w);
  if (n == 0u) { return; }
  var dmin = 1e30;
  if (n == 1u) {
    dmin = length(p - pts[0].xyz);
  } else {
    for (var i = 0u; i < n - 1u; i = i + 1u) {
      dmin = min(dmin, seg_dist(p, pts[i].xyz, pts[i + 1u].xyz));
    }
  }
  if (dmin <= u.params.x) {
    let id = select(u32(u.params.y), 0u, u.params.z > 0.5);   // remove \u2192 0
    textureStore(t_label, vec3<i32>(gid), vec4<u32>(id, 0u, 0u, 0u));
  }
}`
);
var PaintEffect = class {
  constructor(seg) {
    this.seg = seg;
    const dev = seg.device;
    this.dev = dev;
    this.pipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: PAINT_WGSL }), entryPoint: "main" } });
    this.uni = dev.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }
  dev;
  pipe;
  uni;
  ptsBuf;
  ptsCap = 0;
  /** Rasterize a stroke (RAS polyline + spherical brush) into the master, interpolating between
   *  samples, then mark the segmentation dirty (one redraw). A single point = a sphere dab. */
  stampStroke(points, opts) {
    if (points.length === 0) return;
    const dev = this.dev, dims = this.seg.dims;
    const need = points.length * 4 * 4;
    if (!this.ptsBuf || this.ptsCap < points.length) {
      this.ptsBuf?.destroy();
      this.ptsCap = Math.max(points.length, 64);
      this.ptsBuf = dev.createBuffer({ size: this.ptsCap * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    const pd = new Float32Array(points.length * 4);
    for (let i = 0; i < points.length; i++) {
      pd[i * 4] = points[i][0];
      pd[i * 4 + 1] = points[i][1];
      pd[i * 4 + 2] = points[i][2];
    }
    dev.queue.writeBuffer(this.ptsBuf, 0, pd, 0, points.length * 4);
    const ab = new ArrayBuffer(96);
    const f = new Float32Array(ab), uu = new Uint32Array(ab);
    f.set(transpose42(this.seg.ijkToRAS), 0);
    uu[16] = dims[0];
    uu[17] = dims[1];
    uu[18] = dims[2];
    uu[19] = 0;
    f[20] = opts.radiusMm;
    f[21] = opts.id ?? 1;
    f[22] = opts.mode === "remove" ? 1 : 0;
    f[23] = points.length;
    dev.queue.writeBuffer(this.uni, 0, ab);
    const bind = dev.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.seg.masterTexture().createView() },
      { binding: 1, resource: { buffer: this.uni } },
      { binding: 2, resource: { buffer: this.ptsBuf, size: need } }
    ] });
    const [gx, gy, gz] = [Math.ceil(dims[0] / 4), Math.ceil(dims[1] / 4), Math.ceil(dims[2] / 4)];
    const enc = dev.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(this.pipe);
    p.setBindGroup(0, bind);
    p.dispatchWorkgroups(gx, gy, gz);
    p.end();
    dev.queue.submit([enc.finish()]);
    this.seg.markDirty();
  }
  /** Incremental segment: weld a capsule from `prev` to `next` (one pointer move). */
  extend(prev, next, opts) {
    this.stampStroke([prev, next], opts);
  }
  destroy() {
    this.uni.destroy();
    this.ptsBuf?.destroy();
  }
};

// algorithms/effects/scissors.ts
var SCISSORS_WGSL = (
  /* wgsl */
  `
struct U {
  ijkToRAS : mat4x4<f32>,    // column-major
  uAxis    : vec4<f32>,      // in-plane basis (xyz)
  vAxis    : vec4<f32>,
  dims     : vec4<u32>,
  params   : vec4<f32>,      // x=vertexCount, y=id, z=op(0 fillInside/1 eraseInside/2 eraseOutside)
};
@group(0) @binding(0) var t_label : texture_storage_3d<r32uint, write>;
@group(0) @binding(1) var<uniform> u : U;
@group(0) @binding(2) var<storage, read> poly : array<vec2<f32>>;   // contour projected to (u,v)

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let p = (u.ijkToRAS * vec4<f32>(vec3<f32>(gid), 1.0)).xyz;
  let q = vec2<f32>(dot(p, u.uAxis.xyz), dot(p, u.vAxis.xyz));   // project the voxel onto the view plane
  let n = u32(u.params.x);
  if (n < 3u) { return; }
  // even-odd (crossing-number) point-in-polygon on the projected contour.
  var inside = false;
  var j = n - 1u;
  for (var i = 0u; i < n; i = i + 1u) {
    let a = poly[i]; let b = poly[j];
    if ((a.y > q.y) != (b.y > q.y)) {
      let xcross = (b.x - a.x) * (q.y - a.y) / (b.y - a.y) + a.x;
      if (q.x < xcross) { inside = !inside; }
    }
    j = i;
  }
  let op = u.params.z;
  let affected = select(inside, !inside, op > 1.5);        // eraseOutside acts where NOT inside
  if (affected) {
    let val = select(0u, u32(u.params.y), op < 0.5);       // fillInside \u2192 id; erase \u2192 0
    textureStore(t_label, vec3<i32>(gid), vec4<u32>(val, 0u, 0u, 0u));
  }
}`
);
var ScissorsEffect = class {
  constructor(seg) {
    this.seg = seg;
    const dev = seg.device;
    this.dev = dev;
    this.pipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: SCISSORS_WGSL }), entryPoint: "main" } });
    this.uni = dev.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }
  dev;
  pipe;
  uni;
  polyBuf;
  polyCap = 0;
  /** Carve the labelmap with a closed RAS contour on the (u,v) view plane, then mark dirty. */
  apply(contourRAS, opts) {
    if (contourRAS.length < 3) return;
    const dev = this.dev, dims = this.seg.dims;
    const [u, v] = [opts.u, opts.v];
    const proj = new Float32Array(contourRAS.length * 2);
    for (let i = 0; i < contourRAS.length; i++) {
      const p2 = contourRAS[i];
      proj[i * 2] = p2[0] * u[0] + p2[1] * u[1] + p2[2] * u[2];
      proj[i * 2 + 1] = p2[0] * v[0] + p2[1] * v[1] + p2[2] * v[2];
    }
    const need = contourRAS.length * 8;
    if (!this.polyBuf || this.polyCap < contourRAS.length) {
      this.polyBuf?.destroy();
      this.polyCap = Math.max(contourRAS.length, 64);
      this.polyBuf = dev.createBuffer({ size: this.polyCap * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    dev.queue.writeBuffer(this.polyBuf, 0, proj);
    const op = opts.operation ?? "eraseInside";
    const ab = new ArrayBuffer(128);
    const f = new Float32Array(ab), uu = new Uint32Array(ab);
    f.set(transpose42(this.seg.ijkToRAS), 0);
    f[16] = u[0];
    f[17] = u[1];
    f[18] = u[2];
    f[19] = 0;
    f[20] = v[0];
    f[21] = v[1];
    f[22] = v[2];
    f[23] = 0;
    uu[24] = dims[0];
    uu[25] = dims[1];
    uu[26] = dims[2];
    uu[27] = 0;
    f[28] = contourRAS.length;
    f[29] = opts.id ?? 1;
    f[30] = op === "fillInside" ? 0 : op === "eraseInside" ? 1 : 2;
    f[31] = 0;
    dev.queue.writeBuffer(this.uni, 0, ab);
    const bind = dev.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.seg.masterTexture().createView() },
      { binding: 1, resource: { buffer: this.uni } },
      { binding: 2, resource: { buffer: this.polyBuf, size: need } }
    ] });
    const [gx, gy, gz] = [Math.ceil(dims[0] / 4), Math.ceil(dims[1] / 4), Math.ceil(dims[2] / 4)];
    const enc = dev.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(this.pipe);
    p.setBindGroup(0, bind);
    p.dispatchWorkgroups(gx, gy, gz);
    p.end();
    dev.queue.submit([enc.finish()]);
    this.seg.markDirty();
  }
  destroy() {
    this.uni.destroy();
    this.polyBuf?.destroy();
  }
};

// algorithms/effects/growcut.ts
var INIT_WGSL2 = (
  /* wgsl */
  `
@group(0) @binding(0) var t_label : texture_3d<u32>;
@group(0) @binding(1) var t_dst : texture_storage_3d<rg32float, write>;
struct U { dims : vec4<u32>, params : vec4<f32> };
@group(0) @binding(2) var<uniform> u : U;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let lbl = textureLoad(t_label, vec3<i32>(gid), 0).r;
  textureStore(t_dst, vec3<i32>(gid), vec4<f32>(f32(lbl), select(0.0, 1.0, lbl != 0u), 0.0, 0.0));   // seed \u2192 strength 1
}`
);
var ITER_WGSL = (
  /* wgsl */
  `
@group(0) @binding(0) var t_img : texture_3d<f32>;
@group(0) @binding(1) var t_src : texture_3d<f32>;
@group(0) @binding(2) var t_dst : texture_storage_3d<rg32float, write>;
struct U { dims : vec4<u32>, params : vec4<f32> };   // params.x = edgeHi (t1), params.y = 1/(t1-t0)
@group(0) @binding(3) var<uniform> u : U;
fn img(c : vec3<i32>) -> f32 { let d = vec3<i32>(u.dims.xyz); return textureLoad(t_img, clamp(c, vec3<i32>(0), d - vec3<i32>(1)), 0).r; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let st = textureLoad(t_src, c, 0);   // (label, strength)
  var bestLabel = st.r;
  var bestStr = st.g;
  let myI = img(c);
  let t1 = u.params.x; let invSpan = u.params.y;
  let offs = array<vec3<i32>, 6>(vec3<i32>(1,0,0), vec3<i32>(-1,0,0), vec3<i32>(0,1,0), vec3<i32>(0,-1,0), vec3<i32>(0,0,1), vec3<i32>(0,0,-1));
  for (var i = 0; i < 6; i = i + 1) {
    let nc = c + offs[i];
    if (any(nc < vec3<i32>(0)) || any(nc >= vec3<i32>(u.dims.xyz))) { continue; }
    let ns = textureLoad(t_src, nc, 0);
    if (ns.g <= 0.0) { continue; }
    // THRESHOLDED similarity: g=1 below the noise floor t0 (no decay inside a region), \u21920 at the edge
    // contrast t1. Lets a label flood homogeneous tissue and stall at a boundary \u2014 robust to noise,
    // unlike the raw 1\u2212d/range which decays every step and can't cross a large homogeneous region.
    let g = clamp((t1 - abs(myI - img(nc))) * invSpan, 0.0, 1.0);
    let attack = ns.g * g;
    if (attack > bestStr) { bestStr = attack; bestLabel = ns.r; }
  }
  textureStore(t_dst, c, vec4<f32>(bestLabel, bestStr, 0.0, 0.0));
}`
);
var FINAL_WGSL = (
  /* wgsl */
  `
@group(0) @binding(0) var t_src : texture_3d<f32>;
@group(0) @binding(1) var t_out : texture_storage_3d<r32uint, write>;
struct U { dims : vec4<u32>, params : vec4<f32> };
@group(0) @binding(2) var<uniform> u : U;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let lbl = u32(textureLoad(t_src, vec3<i32>(gid), 0).r + 0.5);
  textureStore(t_out, vec3<i32>(gid), vec4<u32>(lbl, 0u, 0u, 0u));
}`
);
var GrowCutEffect = class {
  /** `imageTex` is the r32float intensity volume aligned to the segmentation grid (same dims/ijkToRAS). */
  constructor(seg, imageTex) {
    this.seg = seg;
    this.imageTex = imageTex;
    const dev = seg.device;
    this.dev = dev;
    const mk = (code) => dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint: "main" } });
    this.initPipe = mk(INIT_WGSL2);
    this.iterPipe = mk(ITER_WGSL);
    this.finalPipe = mk(FINAL_WGSL);
    const state = () => dev.createTexture({ size: seg.dims, dimension: "3d", format: "rg32float", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.a = state();
    this.b = state();
    this.uni = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const [dx, dy, dz] = seg.dims;
    this.g = [Math.ceil(dx / 4), Math.ceil(dy / 4), Math.ceil(dz / 4)];
  }
  dev;
  initPipe;
  iterPipe;
  finalPipe;
  a;
  b;
  uni;
  g;
  writeUni(t1, invSpan) {
    const ab = new ArrayBuffer(32);
    const u = new Uint32Array(ab), f = new Float32Array(ab);
    u[0] = this.seg.dims[0];
    u[1] = this.seg.dims[1];
    u[2] = this.seg.dims[2];
    u[3] = 0;
    f[4] = t1;
    f[5] = invSpan;
    f[6] = 0;
    f[7] = 0;
    this.dev.queue.writeBuffer(this.uni, 0, ab);
  }
  pass(pipe, entries) {
    const enc = this.dev.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(pipe);
    p.setBindGroup(0, this.dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries }));
    p.dispatchWorkgroups(...this.g);
    p.end();
    this.dev.queue.submit([enc.finish()]);
  }
  /** Grow the current sparse seed labelmap to fill the volume, writing the dense result back to the
   *  master. Returns the iterations actually run (early-stops at convergence). */
  async grow(opts = {}) {
    const dims = this.seg.dims;
    const range = opts.intensityRange ?? await this.imageRange();
    const t0 = (opts.edgeLo ?? 0.15) * range, t1 = (opts.edgeHi ?? 0.5) * range;
    this.writeUni(t1, 1 / Math.max(t1 - t0, 1e-6));
    const maxIter = opts.iterations ?? Math.ceil(Math.max(...dims) * 1.2);
    const checkEvery = opts.checkEvery ?? 16;
    this.pass(this.initPipe, [
      { binding: 0, resource: this.seg.masterTexture().createView() },
      { binding: 1, resource: this.a.createView() },
      { binding: 2, resource: { buffer: this.uni } }
    ]);
    let src = this.a, dst = this.b, ran = 0;
    for (let i = 0; i < maxIter; i++) {
      this.pass(this.iterPipe, [
        { binding: 0, resource: this.imageTex.createView() },
        { binding: 1, resource: src.createView() },
        { binding: 2, resource: dst.createView() },
        { binding: 3, resource: { buffer: this.uni } }
      ]);
      [src, dst] = [dst, src];
      ran++;
      if (checkEvery > 0 && (i + 1) % checkEvery === 0) {
        const filled = await this.filledCount(src);
        if (filled === this.lastFilled) break;
        this.lastFilled = filled;
      }
    }
    this.lastFilled = -1;
    this.pass(this.finalPipe, [
      { binding: 0, resource: src.createView() },
      { binding: 1, resource: this.seg.masterTexture().createView() },
      { binding: 2, resource: { buffer: this.uni } }
    ]);
    this.seg.markDirty();
    return ran;
  }
  lastFilled = -1;
  /** Count labelled voxels in a state texture (readback of the label channel) — for convergence + tests. */
  async filledCount(state) {
    const [dx, dy, dz] = this.seg.dims;
    const bpr = Math.ceil(dx * 8 / 256) * 256;
    const rowF = bpr / 4;
    const buf = this.dev.createBuffer({ size: bpr * dy * dz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: state }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: dy }, [dx, dy, dz]);
    this.dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(buf.getMappedRange());
    let n = 0;
    for (let z2 = 0; z2 < dz; z2++) for (let y2 = 0; y2 < dy; y2++) for (let x = 0; x < dx; x++) if (f[(z2 * dy + y2) * rowF + x * 2] > 0.5) n++;
    buf.unmap();
    buf.destroy();
    return n;
  }
  /** Image intensity range (max−min) for the default similarity scale. */
  async imageRange() {
    const [dx, dy, dz] = this.seg.dims;
    const bpr = Math.ceil(dx * 4 / 256) * 256;
    const rowF = bpr / 4;
    const buf = this.dev.createBuffer({ size: bpr * dy * dz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.imageTex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: dy }, [dx, dy, dz]);
    this.dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(buf.getMappedRange());
    let mn2 = Infinity, mx = -Infinity;
    for (let z2 = 0; z2 < dz; z2++) for (let y2 = 0; y2 < dy; y2++) for (let x = 0; x < dx; x++) {
      const v = f[(z2 * dy + y2) * rowF + x];
      if (v < mn2) mn2 = v;
      if (v > mx) mx = v;
    }
    buf.unmap();
    buf.destroy();
    return mx - mn2;
  }
  destroy() {
    this.a.destroy();
    this.b.destroy();
    this.uni.destroy();
  }
};

// algorithms/seg-edit-driver.ts
var SegEditDriver = class _SegEditDriver {
  constructor(seg, opts = {}) {
    this.seg = seg;
    this.opts = opts;
    this.paint = new PaintEffect(seg);
  }
  paint;
  scissors;
  growcut;
  growcutImg;
  // the image the current growcut was built for
  labels = /* @__PURE__ */ new Map();
  // Slicer segment id → r32uint master label
  nextLabel = 1;
  active;
  /** Normalize any of the three carriers → the bare SegEdit payload (or null). */
  static unwrap(op) {
    const o = op;
    if (o && typeof o === "object") {
      if (o.edit && typeof o.edit === "object") return o.edit;
      if (o.cmd === "segEdit" && o.args && typeof o.args === "object") return o.args;
      if (typeof o.kind === "string") return o;
    }
    return null;
  }
  image() {
    const t = this.opts.imageTex;
    return typeof t === "function" ? t() : t;
  }
  labelFor(segmentId) {
    if (!segmentId) return 1;
    if (this.opts.labelForSegment) return this.opts.labelForSegment(segmentId);
    let id = this.labels.get(segmentId);
    if (id === void 0) {
      id = this.nextLabel++;
      this.labels.set(segmentId, id);
    }
    return id;
  }
  radiusFor(e) {
    return (e.brush?.diameterMm ?? this.opts.defaultDiameterMm ?? 6) / 2;
  }
  modeFor(e) {
    if (e.mode === "remove" || (e.effect ?? "").toLowerCase().startsWith("erase")) return "remove";
    return "add";
  }
  strokeOpts(e) {
    return { radiusMm: this.radiusFor(e), id: this.labelFor(e.segmentId), mode: this.modeFor(e) };
  }
  /** Apply one COMMITTED edit (all its points at once) — the replay/live path. `stroke`/`scissors`
   *  submit synchronously; `seeds` (grow-from-seeds) awaits the CA to converge, so this is async. */
  // deno-lint-ignore require-await
  async applyEdit(op) {
    const e = _SegEditDriver.unwrap(op);
    if (!e) return;
    switch (e.kind) {
      case "stroke": {
        const s = e;
        if (s.points?.length) this.paint.stampStroke(s.points, this.strokeOpts(s));
        return;
      }
      case "scissors":
        return this.applyScissors(e);
      case "seeds":
        return this.applySeeds(e);
      default:
        this.opts.onUnhandled?.(e.kind);
        return;
    }
  }
  applyScissors(e) {
    if (!e.contour?.length || !e.u || !e.v) return;
    this.scissors ??= new ScissorsEffect(this.seg);
    this.scissors.apply(e.contour, {
      u: e.u,
      v: e.v,
      operation: e.operation ?? "eraseInside",
      id: this.labelFor(e.segmentId)
    });
  }
  /** Grow-from-seeds: stamp the sparse scribbles into the master (as seeds, one label each), then run
   *  the intensity-guided CA to flood the volume. Needs the source image (opts.imageTex). */
  async applySeeds(e) {
    if (!e.scribbles?.length) return;
    const img = this.image();
    if (!img) {
      this.opts.onUnhandled?.("seeds(no image)");
      return;
    }
    for (const sc of e.scribbles) {
      if (!sc.points?.length) continue;
      const label = sc.label ?? this.labelFor(sc.segmentId ?? e.segmentId);
      const radiusMm = (sc.brush?.diameterMm ?? this.opts.defaultDiameterMm ?? 6) / 2;
      this.paint.stampStroke(sc.points, { radiusMm, id: label, mode: "add" });
    }
    if (!this.growcut || this.growcutImg !== img) {
      this.growcut?.destroy();
      this.growcut = new GrowCutEffect(this.seg, img);
      this.growcutImg = img;
    }
    await this.growcut.grow({
      edgeLo: e.edgeLo,
      edgeHi: e.edgeHi,
      intensityRange: e.intensityRange,
      iterations: e.iterations
    });
  }
  // ── Incremental live path: begin / addPoint / end, as pointer samples arrive (real-time apply,
  //    no wait for mouse-up). A stroke is a pointer-drag stream, exactly like a camera drag. ──
  /** Start an incremental stroke; `meta` carries the same fields a full edit would (minus points). */
  beginStroke(meta = {}) {
    this.active = { opts: this.strokeOpts({ kind: "stroke", points: [], ...meta }), last: void 0 };
  }
  /** Add one sampled point — welds a capsule from the previous sample (first point = a dab). */
  addPoint(p) {
    if (!this.active) return;
    this.paint.stampStroke(this.active.last ? [this.active.last, p] : [p], this.active.opts);
    this.active.last = p;
  }
  endStroke() {
    this.active = void 0;
  }
  destroy() {
    this.paint.destroy();
    this.scissors?.destroy();
    this.growcut?.destroy();
  }
};

// render/sdf-bake.ts
var INIT_WGSL3 = (
  /* wgsl */
  `
struct U { ijkToRAS : mat4x4<f32>, dims : vec4<u32>, params : vec4<f32>, origin : vec4<i32> };
@group(0) @binding(0) var t_label : texture_3d<u32>;
@group(0) @binding(1) var t_seed_out : texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var<uniform> u : U;
// PADDED SDF GRID: the seed/SDF textures are LARGER than the labelmap by 'pad' voxels on every side
// (dims.w). Coord c is a padded-grid coord; the label lives at c-pad, and everything outside the label
// range is background (0). This gives a real spatial margin of background BEYOND the segmentation, so
// a segment touching the labelmap edge closes as a genuine capped surface with room for the SDF to go
// positive \u2014 and gradient/finite-difference samples near that cap stay in-bounds (they read real
// background) instead of hitting the out-of-volume cull sentinel, which used to poison the normal.
fn labelAt(c : vec3<i32>) -> u32 {
  let pad = i32(u.dims.w);
  let ld = vec3<i32>(u.dims.xyz) - vec3<i32>(2 * pad);   // label dims = padded dims - 2\xB7pad
  let lc = c - vec3<i32>(pad);
  if (any(lc < vec3<i32>(0)) || any(lc >= ld)) { return 0u; }
  return textureLoad(t_label, lc, 0).r;
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = vec3<i32>(gid) + u.origin.xyz;   // region-limited dispatch offsets into the grid
  if (any(c >= vec3<i32>(u.dims.xyz))) { return; }
  let my = labelAt(c);
  let meIn = my != 0u;
  let allMode = u.params.y > 0.5;                   // 0 = outer boundary only; 1 = ANY label change (multi-material interfaces)
  var boundary = false;
  var region = my;                                  // inside voxel \u2192 own label
  let offs = array<vec3<i32>, 6>(vec3<i32>(1,0,0), vec3<i32>(-1,0,0), vec3<i32>(0,1,0), vec3<i32>(0,-1,0), vec3<i32>(0,0,1), vec3<i32>(0,0,-1));
  for (var i = 0; i < 6; i = i + 1) {
    let nl = labelAt(c + offs[i]);
    // outer mode: boundary at inside\u2194outside (segment\u2194background). all mode: boundary at ANY label change
    // (segment\u2194background AND segment\u2194segment) so embedded/nested structures get an interface shell too.
    let isChange = select((nl != 0u) != meIn, nl != my, allMode);
    // outer: a background boundary voxel adopts the neighbour's label (so the outer shell renders on both
    // sides). all: every voxel keeps its OWN label (background stays 0 = transparent), so the region
    // COLOUR changes across EVERY interface \u2014 including the outer one \u2014 which the shader's on-the-fly
    // re-signing needs to recover a clean inside/outside normal there too.
    if (isChange) { boundary = true; if (my == 0u && !allMode) { region = nl; } }
  }
  var seed = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  if (boundary) { seed = vec4<f32>((u.ijkToRAS * vec4<f32>(vec3<f32>(c), 1.0)).xyz, f32(region)); }
  textureStore(t_seed_out, c, seed);
}`
);
var JFA_WGSL = (
  /* wgsl */
  `
struct U { ijkToRAS : mat4x4<f32>, dims : vec4<u32>, params : vec4<f32>, origin : vec4<i32> };
@group(0) @binding(0) var t_seed_in : texture_3d<f32>;
@group(0) @binding(1) var t_seed_out : texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var<uniform> u : U;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = vec3<i32>(gid) + u.origin.xyz;   // region-limited dispatch offsets into the grid
  if (any(c >= vec3<i32>(u.dims.xyz))) { return; }
  let p = (u.ijkToRAS * vec4<f32>(vec3<f32>(c), 1.0)).xyz;
  let step = i32(u.params.x);
  let dmax = vec3<i32>(u.dims.xyz) - vec3<i32>(1);
  var best = textureLoad(t_seed_in, c, 0);
  var bestD = select(1e30, distance(p, best.xyz), best.w > 0.5);
  for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      for (var dx = -1; dx <= 1; dx = dx + 1) {
        if (dx == 0 && dy == 0 && dz == 0) { continue; }
        let nc = clamp(c + vec3<i32>(dx, dy, dz) * step, vec3<i32>(0), dmax);
        let s = textureLoad(t_seed_in, nc, 0);
        if (s.w > 0.5) {
          let d = distance(p, s.xyz);
          if (d < bestD) { bestD = d; best = s; }
        }
      }
    }
  }
  textureStore(t_seed_out, c, best);
}`
);
var FINAL_WGSL2 = (
  /* wgsl */
  `
struct U { ijkToRAS : mat4x4<f32>, dims : vec4<u32>, params : vec4<f32>, origin : vec4<i32> };
@group(0) @binding(0) var t_seed_in : texture_3d<f32>;
@group(0) @binding(1) var t_label : texture_3d<u32>;
@group(0) @binding(2) var t_out : texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var<uniform> u : U;
@group(0) @binding(4) var<uniform> u_pal : array<vec4<f32>, 256>;
@group(0) @binding(5) var t_attr : texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var<uniform> u_mode : array<vec4<f32>, 256>;   // .x = shading mode (0 surface, 1 volume)
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = vec3<i32>(gid) + u.origin.xyz;   // region-limited dispatch offsets into the grid
  if (any(c >= vec3<i32>(u.dims.xyz))) { return; }
  let p = (u.ijkToRAS * vec4<f32>(vec3<f32>(c), 1.0)).xyz;
  let s = textureLoad(t_seed_in, c, 0);
  let valid = s.w > 0.5;
  let dist = select(1e3, distance(p, s.xyz), valid);
  let pad = i32(u.dims.w);                                       // label is offset by pad; pad region = background (outside)
  let lc = c - vec3<i32>(pad);
  let ld = vec3<i32>(u.dims.xyz) - vec3<i32>(2 * pad);
  let inRange = all(lc >= vec3<i32>(0)) && all(lc < ld);
  let ins = inRange && textureLoad(t_label, lc, 0).r != 0u;
  // outer mode: SIGNED (neg inside the union) \u2014 smooth zero-crossing = clean normals. all mode:
  // UNSIGNED distance to the nearest interface (every label change is a wall; no global inside/outside).
  let sdf = select(select(dist, -dist, ins), dist, u.params.y > 0.5);
  let lbl = u32(s.w + 0.5) & 255u;
  let pal = select(vec4<f32>(0.0), u_pal[lbl], valid);
  let mode = select(0.0, u_mode[lbl].x, valid);
  // PREMULTIPLIED colour (rgb\xB7opacity): the colour-seam blur then can't bleed a HIDDEN (opacity 0)
  // segment's colour into a visible neighbour \u2014 an invisible organ was still tinting the organ it
  // abutted (looked like half-opacity). The shader divides by the per-segment opacity to recover the
  // true colour, so a 0-opacity region contributes nothing to the blend.
  textureStore(t_out, c, vec4<f32>(pal.rgb * pal.a, sdf));
  // .r = opacity, .g = shading mode, .b = distance (seam-blurred \u2192 SMOOTH distance for the interface-mode
  // normal; sdfTex.a stays sharp for shell membership), .a = CRISP presence (1 inside a real segment, 0
  // background) \u2014 the FULLBLUR carries it unblurred so the shader can tell a genuine in-segment voxel
  // from one that merely caught BLED colour/opacity outside any segment, and gate the shell to the real edge.
  textureStore(t_attr, c, vec4<f32>(pal.a, mode, sdf, select(0.0, 1.0, ins)));
}`
);
var BLUR_WGSL2 = (
  /* wgsl */
  `
struct BU { dims : vec4<u32>, axis_r : vec4<u32>, w : array<vec4<f32>, 4>, origin : vec4<i32> };
@group(0) @binding(0) var t_in : texture_3d<f32>;
@group(0) @binding(1) var t_out : texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u : BU;
fn wt(i : u32) -> f32 { return u.w[i >> 2u][i & 3u]; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = vec3<i32>(gid) + u.origin.xyz;
  if (any(c >= vec3<i32>(u.dims.xyz))) { return; }
  let dmax = vec3<i32>(u.dims.xyz) - vec3<i32>(1);
  var av = vec3<i32>(0);
  if (u.axis_r.x == 0u) { av = vec3<i32>(1,0,0); } else if (u.axis_r.x == 1u) { av = vec3<i32>(0,1,0); } else { av = vec3<i32>(0,0,1); }
  let center = textureLoad(t_in, c, 0);
  var sum = center.a * wt(0u);
  let R = i32(u.axis_r.y);
  for (var i = 1; i <= R; i = i + 1) {
    sum = sum + wt(u32(i)) * (textureLoad(t_in, clamp(c + av * i, vec3<i32>(0), dmax), 0).a
                            + textureLoad(t_in, clamp(c - av * i, vec3<i32>(0), dmax), 0).a);
  }
  textureStore(t_out, c, vec4<f32>(center.rgb, sum));
}`
);
var COLBLUR_WGSL = (
  /* wgsl */
  `
struct BU { dims : vec4<u32>, axis_r : vec4<u32>, w : array<vec4<f32>, 4>, origin : vec4<i32> };
@group(0) @binding(0) var t_in : texture_3d<f32>;
@group(0) @binding(1) var t_out : texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u : BU;
fn wt(i : u32) -> f32 { return u.w[i >> 2u][i & 3u]; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = vec3<i32>(gid) + u.origin.xyz;
  if (any(c >= vec3<i32>(u.dims.xyz))) { return; }
  let dmax = vec3<i32>(u.dims.xyz) - vec3<i32>(1);
  var av = vec3<i32>(0);
  if (u.axis_r.x == 0u) { av = vec3<i32>(1,0,0); } else if (u.axis_r.x == 1u) { av = vec3<i32>(0,1,0); } else { av = vec3<i32>(0,0,1); }
  let center = textureLoad(t_in, c, 0);
  var sum = center.rgb * wt(0u);
  let R = i32(u.axis_r.y);
  for (var i = 1; i <= R; i = i + 1) {
    sum = sum + wt(u32(i)) * (textureLoad(t_in, clamp(c + av * i, vec3<i32>(0), dmax), 0).rgb
                            + textureLoad(t_in, clamp(c - av * i, vec3<i32>(0), dmax), 0).rgb);
  }
  textureStore(t_out, c, vec4<f32>(sum, center.a));
}`
);
var FULLBLUR_WGSL = (
  /* wgsl */
  `
struct BU { dims : vec4<u32>, axis_r : vec4<u32>, w : array<vec4<f32>, 4>, origin : vec4<i32> };
@group(0) @binding(0) var t_in : texture_3d<f32>;
@group(0) @binding(1) var t_out : texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u : BU;
fn wt(i : u32) -> f32 { return u.w[i >> 2u][i & 3u]; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = vec3<i32>(gid) + u.origin.xyz;
  if (any(c >= vec3<i32>(u.dims.xyz))) { return; }
  let dmax = vec3<i32>(u.dims.xyz) - vec3<i32>(1);
  var av = vec3<i32>(0);
  if (u.axis_r.x == 0u) { av = vec3<i32>(1,0,0); } else if (u.axis_r.x == 1u) { av = vec3<i32>(0,1,0); } else { av = vec3<i32>(0,0,1); }
  let center = textureLoad(t_in, c, 0);
  var gb = center.gb * wt(0u);
  let R = i32(u.axis_r.y);
  for (var i = 1; i <= R; i = i + 1) {
    gb = gb + wt(u32(i)) * (textureLoad(t_in, clamp(c + av * i, vec3<i32>(0), dmax), 0).gb
                          + textureLoad(t_in, clamp(c - av * i, vec3<i32>(0), dmax), 0).gb);
  }
  textureStore(t_out, c, vec4<f32>(center.r, gb.x, gb.y, center.a));
}`
);
function gaussHalfKernel2(sigma) {
  const radius = Math.max(1, Math.min(15, Math.ceil(3 * sigma)));
  const raw = new Float32Array(radius + 1);
  let total = 0;
  for (let i = 0; i <= radius; i++) {
    raw[i] = Math.exp(-(i * i) / (2 * sigma * sigma));
    total += (i === 0 ? 1 : 2) * raw[i];
  }
  const w = new Float32Array(16);
  for (let i = 0; i <= radius; i++) w[i] = raw[i] / total;
  return { radius, w };
}
var JfaSdfBaker = class {
  // 0 = outer boundary (signed); 1 = any label change (unsigned, multi-material)
  constructor(dev, labelTex, dims, ijkToRAS, smoothSigmaVoxels = 1, pad = 2, boundaryMode = "outer") {
    this.labelTex = labelTex;
    this.dims = dims;
    this.ijkToRAS = ijkToRAS;
    this.dev = dev;
    this.smoothSigma = smoothSigmaVoxels;
    this.bmode = boundaryMode === "all" ? 1 : 0;
    this.pad = pad;
    this.labelDims = [dims[0], dims[1], dims[2]];
    this.dims = [dims[0] + 2 * pad, dims[1] + 2 * pad, dims[2] + 2 * pad];
    const m = ijkToRAS.slice();
    for (let r = 0; r < 3; r++) m[r * 4 + 3] -= pad * (m[r * 4] + m[r * 4 + 1] + m[r * 4 + 2]);
    this.ijkToRAS = m;
    const [dx, dy, dz] = this.dims;
    const mk = (fmt, extra = 0) => dev.createTexture({ size: this.dims, dimension: "3d", format: fmt, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | extra });
    const seedUsage = GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    this.seed = [mk("rgba32float", seedUsage), mk("rgba32float", seedUsage)];
    this.sdfTex = mk("rgba16float", GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC);
    this.attrTex = mk("rgba16float", GPUTextureUsage.COPY_DST);
    this.attrScratch = mk("rgba16float", GPUTextureUsage.COPY_SRC);
    this.sdfScratch = mk("rgba16float", GPUTextureUsage.COPY_SRC);
    this.uni = dev.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.palBuf = dev.createBuffer({ size: 256 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.modeBuf = dev.createBuffer({ size: 256 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const mod = (code) => dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint: "main" } });
    this.initPipe = mod(INIT_WGSL3);
    this.jfaPipe = mod(JFA_WGSL);
    this.finalPipe = mod(FINAL_WGSL2);
    this.blurPipe = mod(BLUR_WGSL2);
    this.colBlurPipe = mod(COLBLUR_WGSL);
    this.fullBlurPipe = mod(FULLBLUR_WGSL);
    this.g = [Math.ceil(dx / 4), Math.ceil(dy / 4), Math.ceil(dz / 4)];
    const maxDim = Math.max(dx, dy, dz);
    const steps = [];
    for (let s = 1 << Math.floor(Math.log2(maxDim - 1)); s >= 1; s >>= 1) steps.push(s);
    this.steps = steps;
  }
  dev;
  seed;
  // rgba32float ping-pong (RAS seed xyz + regionLabel)
  sdfTex;
  // rgba16float: .rgb = per-label colour, .a = signed dist (mm) — sampled by SegmentField
  attrTex;
  // rgba16float: .r = per-segment opacity, .g = shading mode — sampled by SegmentField
  attrScratch;
  // rgba16float attr-blur ping-pong
  lastSeed = 0;
  // seed buffer the last sweep finalized from
  sdfScratch;
  // rgba16float blur ping-pong
  uni;
  palBuf;
  // 256 × vec4 label→colour palette (.a = opacity)
  modeBuf;
  // 256 × vec4 label→shading mode (.x = 0 surface / 1 volume)
  initPipe;
  jfaPipe;
  finalPipe;
  blurPipe;
  // blurs .a (distance), carries .rgb
  colBlurPipe;
  // blurs .rgb (colour), carries .a
  fullBlurPipe;
  // blurs all channels — the attr texture (opacity + mode)
  g;
  steps;
  smoothSigma;
  pad;
  // background margin (voxels) padded around the labelmap
  labelDims;
  // original (label) dims, before padding
  // `pad` voxels of background are added on every side so segments touching the labelmap edge get a
  // real cap + in-bounds gradient neighbourhood (docs/ALGORITHMS.md border artifact). The SDF textures,
  // dispatch, seeds, blur and readback all run on the PADDED grid; `dims`/`ijkToRAS` become the padded
  // grid's, and `sdfDims()`/`sdfIjkToRAS()` expose them to the SegmentField.
  bmode;
  /** The resident colorized-SDF texture (rgba16float: .rgb = per-label colour, .a = signed mm).
   *  Identity stable across bakes → the SceneRenderer bind group stays valid; a live edit updates in
   *  place. */
  sdfTexture() {
    return this.sdfTex;
  }
  /** The resident per-segment attribute texture (rgba16float; .r = opacity). Identity stable. */
  attrTexture() {
    return this.attrTex;
  }
  /** The PADDED grid the SDF/attr textures live on (labelDims + 2·pad), and the ijkToRAS that maps it
   *  to RAS — hand these to the SegmentField so its patient→texture transform covers the padded extent. */
  sdfDims() {
    return this.dims;
  }
  sdfIjkToRAS() {
    return this.ijkToRAS;
  }
  /** Background margin (voxels) padded on each side; readDistance() is on the padded grid, so a label
   *  voxel (x,y,z) is at padded (x+pad, y+pad, z+pad). */
  padVoxels() {
    return this.pad;
  }
  /** Read back the per-voxel signed distance (sdfTex .a, mm) to CPU. For accuracy comparison/tests. */
  async readDistance() {
    const [dx, dy, dz] = this.dims;
    const bpr = Math.ceil(dx * 8 / 256) * 256;
    const rowU16 = bpr / 2;
    const buf = this.dev.createBuffer({ size: bpr * dy * dz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.sdfTex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: dy }, [dx, dy, dz]);
    this.dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const u16 = new Uint16Array(buf.getMappedRange());
    const h2f = (h) => {
      const s = h & 32768 ? -1 : 1, e = (h & 31744) >> 10, f = h & 1023;
      if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
      if (e === 31) return f ? NaN : s * Infinity;
      return s * Math.pow(2, e - 15) * (1 + f / 1024);
    };
    const out = new Float32Array(dx * dy * dz);
    for (let z2 = 0; z2 < dz; z2++) for (let y2 = 0; y2 < dy; y2++) for (let x = 0; x < dx; x++) {
      out[(z2 * dy + y2) * dx + x] = h2f(u16[(z2 * dy + y2) * rowU16 + x * 4 + 3]);
    }
    buf.unmap();
    buf.destroy();
    return out;
  }
  /** Set the label→colour palette (256 × rgba f32: rgb = colour, a = opacity). Call before bake(). */
  setPalette(palette) {
    const pal = new Float32Array(256 * 4);
    pal.set(palette.subarray(0, Math.min(palette.length, 256 * 4)));
    this.dev.queue.writeBuffer(this.palBuf, 0, pal);
  }
  /** Set the per-label shading mode palette (256 × vec4; .x = 0 surface shell / 1 volume DVR fill). */
  setModePalette(modes) {
    const m = new Float32Array(256 * 4);
    m.set(modes.subarray(0, Math.min(modes.length, 256 * 4)));
    this.dev.queue.writeBuffer(this.modeBuf, 0, m);
  }
  writeUni(step, origin = [0, 0, 0]) {
    const ab = new ArrayBuffer(112);
    const f = new Float32Array(ab), u = new Uint32Array(ab);
    f.set(transpose4(this.ijkToRAS), 0);
    u[16] = this.dims[0];
    u[17] = this.dims[1];
    u[18] = this.dims[2];
    u[19] = this.pad;
    f[20] = step;
    f[21] = this.bmode;
    f[22] = 0;
    f[23] = 0;
    const i32v = new Int32Array(ab);
    i32v[24] = origin[0];
    i32v[25] = origin[1];
    i32v[26] = origin[2];
    i32v[27] = 0;
    this.dev.queue.writeBuffer(this.uni, 0, ab);
  }
  /** FAST bake for LIVE editing: plain JFA (approximate) + a light distance-only blur (crisp colour
   *  seams). Cheap, so it keeps up with an in-progress stroke; the seams stay voxel-jagged until the
   *  edit settles and refine() runs. */
  bake() {
    this.sweep([], this.smoothSigma, 0);
  }
  /** REFINE for a STATIC labelmap (run once the edit settles): JFA+2 extra passes → a near-exact
   *  Voronoi/SDF (fixes the small JFA mistakes near close/overlapping segments) and a colour-seam blur
   *  so neighbouring-label boundaries are smooth, not a voxel staircase. Distance blur stays at the
   *  same σ (dropping it re-introduces Voronoi facets — crispness comes from the render band, not from
   *  under-smoothing). Higher quality lives in the resident texture, so camera renders stay cheap. */
  refine() {
    this.sweep([2, 1], this.smoothSigma, 1);
  }
  /** REGION-LIMITED refine: re-flood ONLY `regionIjk` (padded-grid coords) after a labelmap edit
   *  confined to it — a per-vertebra visibility flip re-bakes a few % of the grid instead of the
   *  whole volume, which is what makes level stepping feel instant. The seed ping-pong pair is
   *  first made consistent with a full-texture copy, so region passes can ping-pong while JFA
   *  taps read valid exterior seeds (surfaces just outside the region flood in correctly).
   *  Exterior distances that referenced a surface REMOVED inside the region go stale, but only
   *  ≫band away from any visible shell — invisible, and the next full sweep cleans them. */
  refineRegion(regionIjk) {
    const dev = this.dev, [dx, dy, dz] = this.dims;
    const lo2 = [Math.max(0, regionIjk.lo[0]), Math.max(0, regionIjk.lo[1]), Math.max(0, regionIjk.lo[2])];
    const hi2 = [Math.min(dx, regionIjk.hi[0]), Math.min(dy, regionIjk.hi[1]), Math.min(dz, regionIjk.hi[2])];
    const [rx, ry, rz] = [hi2[0] - lo2[0], hi2[1] - lo2[1], hi2[2] - lo2[2]];
    if (rx <= 0 || ry <= 0 || rz <= 0) return;
    const g = [Math.ceil(rx / 4), Math.ceil(ry / 4), Math.ceil(rz / 4)];
    const region = { lo: lo2, hi: hi2 };
    let src = this.lastSeed;
    let enc = dev.createCommandEncoder();
    enc.copyTextureToTexture({ texture: this.seed[src] }, { texture: this.seed[src ^ 1] }, this.dims);
    dev.queue.submit([enc.finish()]);
    this.writeUni(0, lo2);
    enc = dev.createCommandEncoder();
    {
      const b = dev.createBindGroup({ layout: this.initPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.labelTex.createView() },
        { binding: 1, resource: this.seed[src].createView() },
        { binding: 2, resource: { buffer: this.uni } }
      ] });
      const p2 = enc.beginComputePass();
      p2.setPipeline(this.initPipe);
      p2.setBindGroup(0, b);
      p2.dispatchWorkgroups(g[0], g[1], g[2]);
      p2.end();
    }
    dev.queue.submit([enc.finish()]);
    const maxDim = Math.max(rx, ry, rz);
    const steps = [];
    for (let s = 1 << Math.floor(Math.log2(Math.max(2, maxDim - 1))); s >= 1; s >>= 1) steps.push(s);
    steps.push(2, 1);
    for (const step of steps) {
      this.writeUni(step, lo2);
      const dst = src ^ 1;
      enc = dev.createCommandEncoder();
      const b = dev.createBindGroup({ layout: this.jfaPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.seed[src].createView() },
        { binding: 1, resource: this.seed[dst].createView() },
        { binding: 2, resource: { buffer: this.uni } }
      ] });
      const p2 = enc.beginComputePass();
      p2.setPipeline(this.jfaPipe);
      p2.setBindGroup(0, b);
      p2.dispatchWorkgroups(g[0], g[1], g[2]);
      p2.end();
      dev.queue.submit([enc.finish()]);
      src = dst;
    }
    this.lastSeed = src;
    this.writeUni(0, lo2);
    enc = dev.createCommandEncoder();
    const bf = dev.createBindGroup({ layout: this.finalPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.seed[src].createView() },
      { binding: 1, resource: this.labelTex.createView() },
      { binding: 2, resource: this.sdfTex.createView() },
      { binding: 3, resource: { buffer: this.uni } },
      { binding: 4, resource: { buffer: this.palBuf } },
      { binding: 5, resource: this.attrTex.createView() },
      { binding: 6, resource: { buffer: this.modeBuf } }
    ] });
    const p = enc.beginComputePass();
    p.setPipeline(this.finalPipe);
    p.setBindGroup(0, bf);
    p.dispatchWorkgroups(g[0], g[1], g[2]);
    p.end();
    dev.queue.submit([enc.finish()]);
    this.blurStage(this.blurPipe, this.smoothSigma, this.sdfTex, this.sdfScratch, region);
    this.blurStage(this.colBlurPipe, 1, this.sdfTex, this.sdfScratch, region);
    this.blurStage(this.fullBlurPipe, 1, this.attrTex, this.attrScratch, region);
  }
  /** ATTR-ONLY rebake for palette/opacity changes (per-segment visibility): the distance field
   *  doesn't move, so re-run ONLY the finalize (from the last sweep's seed) with its sdf writes
   *  routed to the scratch texture (discarded — the blurred resident sdfTex stays pristine) and
   *  re-blur the attribute seams. ~4 passes instead of the ~20-pass init+JFA+blur sweep, which is
   *  what makes per-vertebra focus switching real-time. */
  rebakeAttr(blurSeams = false, regionIjk) {
    const dev = this.dev, [dx, dy, dz] = this.dims;
    const region = regionIjk && {
      lo: [Math.max(0, regionIjk.lo[0]), Math.max(0, regionIjk.lo[1]), Math.max(0, regionIjk.lo[2])],
      hi: [Math.min(dx, regionIjk.hi[0]), Math.min(dy, regionIjk.hi[1]), Math.min(dz, regionIjk.hi[2])]
    };
    const [gx, gy, gz] = region ? [Math.ceil((region.hi[0] - region.lo[0]) / 4), Math.ceil((region.hi[1] - region.lo[1]) / 4), Math.ceil((region.hi[2] - region.lo[2]) / 4)] : this.g;
    if (region && (gx <= 0 || gy <= 0 || gz <= 0)) return;
    this.writeUni(0, region ? region.lo : [0, 0, 0]);
    const enc = dev.createCommandEncoder();
    const bf = dev.createBindGroup({ layout: this.finalPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.seed[this.lastSeed].createView() },
      { binding: 1, resource: this.labelTex.createView() },
      { binding: 2, resource: this.sdfScratch.createView() },
      // sdf writes discarded
      { binding: 3, resource: { buffer: this.uni } },
      { binding: 4, resource: { buffer: this.palBuf } },
      { binding: 5, resource: this.attrTex.createView() },
      { binding: 6, resource: { buffer: this.modeBuf } }
    ] });
    const p = enc.beginComputePass();
    p.setPipeline(this.finalPipe);
    p.setBindGroup(0, bf);
    p.dispatchWorkgroups(gx, gy, gz);
    p.end();
    dev.queue.submit([enc.finish()]);
    if (blurSeams) this.blurStage(this.fullBlurPipe, 1, this.attrTex, this.attrScratch, region);
  }
  /** Blur the attribute seams of the CURRENT attr texture in place (no re-finalize) — the
   *  cheapest possible settle after a run of rebakeAttr(false) visibility steps. */
  blurAttrOnly() {
    this.blurStage(this.fullBlurPipe, 1, this.attrTex, this.attrScratch);
  }
  /** One full sweep: init → JFA (schedule + extra) → finalize → blur .a → optional blur .rgb. */
  sweep(extraSteps, distSigma, colorSigma) {
    const dev = this.dev, [gx, gy, gz] = this.g;
    this.writeUni(0);
    let enc = dev.createCommandEncoder();
    {
      const b = dev.createBindGroup({ layout: this.initPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.labelTex.createView() },
        { binding: 1, resource: this.seed[0].createView() },
        { binding: 2, resource: { buffer: this.uni } }
      ] });
      const p2 = enc.beginComputePass();
      p2.setPipeline(this.initPipe);
      p2.setBindGroup(0, b);
      p2.dispatchWorkgroups(gx, gy, gz);
      p2.end();
    }
    dev.queue.submit([enc.finish()]);
    let src = 0;
    for (const step of [...this.steps, ...extraSteps]) {
      this.writeUni(step);
      const dst = src ^ 1;
      enc = dev.createCommandEncoder();
      const b = dev.createBindGroup({ layout: this.jfaPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.seed[src].createView() },
        { binding: 1, resource: this.seed[dst].createView() },
        { binding: 2, resource: { buffer: this.uni } }
      ] });
      const p2 = enc.beginComputePass();
      p2.setPipeline(this.jfaPipe);
      p2.setBindGroup(0, b);
      p2.dispatchWorkgroups(gx, gy, gz);
      p2.end();
      dev.queue.submit([enc.finish()]);
      src = dst;
    }
    this.lastSeed = src;
    enc = dev.createCommandEncoder();
    const bf = dev.createBindGroup({ layout: this.finalPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.seed[src].createView() },
      { binding: 1, resource: this.labelTex.createView() },
      { binding: 2, resource: this.sdfTex.createView() },
      { binding: 3, resource: { buffer: this.uni } },
      { binding: 4, resource: { buffer: this.palBuf } },
      { binding: 5, resource: this.attrTex.createView() },
      { binding: 6, resource: { buffer: this.modeBuf } }
    ] });
    const p = enc.beginComputePass();
    p.setPipeline(this.finalPipe);
    p.setBindGroup(0, bf);
    p.dispatchWorkgroups(gx, gy, gz);
    p.end();
    dev.queue.submit([enc.finish()]);
    if (distSigma > 0) this.blurStage(this.blurPipe, distSigma, this.sdfTex, this.sdfScratch);
    if (colorSigma > 0) this.blurStage(this.colBlurPipe, colorSigma, this.sdfTex, this.sdfScratch);
    if (colorSigma > 0) this.blurStage(this.fullBlurPipe, colorSigma, this.attrTex, this.attrScratch);
  }
  /** 3 separable Gaussian passes with the given pipeline (which channels it blurs), tex↔scratch,
   *  ending in scratch → copied back to `tex` so its identity stays stable for the renderer. */
  blurStage(pipe, sigma, tex, scratch, region) {
    const dev = this.dev, [dx, dy, dz] = this.dims;
    const { radius, w } = gaussHalfKernel2(sigma);
    const passes = [[tex, scratch, 0], [scratch, tex, 1], [tex, scratch, 2]];
    const enc = dev.createCommandEncoder();
    let passIdx = 0;
    for (const [srcT, dstT, axis] of passes) {
      const expand = region ? (2 - passIdx) * radius : 0;
      const lo2 = region ? [Math.max(0, region.lo[0] - expand), Math.max(0, region.lo[1] - expand), Math.max(0, region.lo[2] - expand)] : [0, 0, 0];
      const hi2 = region ? [Math.min(dx, region.hi[0] + expand), Math.min(dy, region.hi[1] + expand), Math.min(dz, region.hi[2] + expand)] : [dx, dy, dz];
      const ab = new ArrayBuffer(112);
      const u32 = new Uint32Array(ab), f32 = new Float32Array(ab), i32 = new Int32Array(ab);
      u32[0] = dx;
      u32[1] = dy;
      u32[2] = dz;
      u32[4] = axis;
      u32[5] = radius;
      f32.set(w, 8);
      i32[24] = lo2[0];
      i32[25] = lo2[1];
      i32[26] = lo2[2];
      const ub = dev.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(ub, 0, ab);
      const b = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: srcT.createView() },
        { binding: 1, resource: dstT.createView() },
        { binding: 2, resource: { buffer: ub } }
      ] });
      const bp = enc.beginComputePass();
      bp.setPipeline(pipe);
      bp.setBindGroup(0, b);
      bp.dispatchWorkgroups(Math.ceil((hi2[0] - lo2[0]) / 4), Math.ceil((hi2[1] - lo2[1]) / 4), Math.ceil((hi2[2] - lo2[2]) / 4));
      bp.end();
      passIdx++;
    }
    if (region) {
      const sz = [region.hi[0] - region.lo[0], region.hi[1] - region.lo[1], region.hi[2] - region.lo[2]];
      enc.copyTextureToTexture({ texture: scratch, origin: region.lo }, { texture: tex, origin: region.lo }, sz);
    } else {
      enc.copyTextureToTexture({ texture: scratch }, { texture: tex }, this.dims);
    }
    dev.queue.submit([enc.finish()]);
  }
  destroy() {
    this.seed[0].destroy();
    this.seed[1].destroy();
    this.sdfTex.destroy();
    this.attrTex.destroy();
    this.attrScratch.destroy();
    this.sdfScratch.destroy();
    this.uni.destroy();
    this.palBuf.destroy();
    this.modeBuf.destroy();
  }
};

// logic/segmentation-logic.ts
function invertAffine(m) {
  const r = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
  const i = [
    (r[4] * r[8] - r[5] * r[7]) / det,
    (r[2] * r[7] - r[1] * r[8]) / det,
    (r[1] * r[5] - r[2] * r[4]) / det,
    (r[5] * r[6] - r[3] * r[8]) / det,
    (r[0] * r[8] - r[2] * r[6]) / det,
    (r[2] * r[3] - r[0] * r[5]) / det,
    (r[3] * r[7] - r[4] * r[6]) / det,
    (r[1] * r[6] - r[0] * r[7]) / det,
    (r[0] * r[4] - r[1] * r[3]) / det
  ];
  const t = [m[3], m[7], m[11]];
  return [
    i[0],
    i[1],
    i[2],
    -(i[0] * t[0] + i[1] * t[1] + i[2] * t[2]),
    i[3],
    i[4],
    i[5],
    -(i[3] * t[0] + i[4] * t[1] + i[5] * t[2]),
    i[6],
    i[7],
    i[8],
    -(i[6] * t[0] + i[7] * t[1] + i[8] * t[2]),
    0,
    0,
    0,
    1
  ];
}
var SegmentationLogic = class {
  constructor(device, seg, opts = {}) {
    this.seg = seg;
    this.renderMode = opts.renderMode ?? "sdf";
    this.sigma = opts.sigmaVoxels ?? 1;
    this.bandMm = opts.bandMm;
    this.opacity = opts.opacity ?? 1;
    this.refineDelayMs = opts.refineDelayMs ?? 180;
    this.boundaryMode = opts.boundaryMode ?? "outer";
    this.clippable = opts.clippable ?? false;
    this.setLabelColor(1, opts.color ?? [0.3, 0.85, 0.55]);
    if (this.renderMode === "sdf") {
      this.sdf = new JfaSdfBaker(device, seg.masterTexture(), seg.dims, seg.ijkToRAS, 1, 2, this.boundaryMode);
    } else {
      this.baker = new ColorizeBaker(device, seg.masterTexture(), seg.dims);
      this.presenceTex = this.baker.output();
    }
    this.rebake();
    this.scheduleRefine();
    this.unsubDirty = seg.onDirty(() => {
      this.rebake();
      for (const cb of this.redrawCbs) cb();
      this.scheduleRefine();
    });
  }
  renderMode;
  clippable;
  attrSettleTimer;
  sdf;
  // sdf path
  baker;
  // surface path
  presenceTex;
  sigma;
  bandMm;
  opacity;
  palette = new Float32Array(256 * 4);
  // label id → (r,g,b, opacity); shared by both paths
  modePalette = new Float32Array(256 * 4);
  // label id → (.x = shading mode: 0 surface / 1 volume) — sdf only
  segField;
  redrawCbs = [];
  unsubDirty;
  refineTimer;
  refineDelayMs;
  // quiescence before the settle-refine (sdf mode; capability-tuned)
  boundaryMode;
  /** Assign a display colour to a label id (0..255). Keeps the current opacity (defaults to 1 =
   *  opaque). Takes effect on the next rebake. */
  setLabelColor(id, rgb) {
    if (id < 1 || id > 255) return;
    const o = id * 4;
    this.palette[o] = rgb[0];
    this.palette[o + 1] = rgb[1];
    this.palette[o + 2] = rgb[2];
    if (this.palette[o + 3] === 0) this.palette[o + 3] = 1;
  }
  /** Per-segment opacity (0 = hidden, 1 = opaque) — palette alpha. Enables translucent surface-model
   *  rendering (see through outer segments to inner ones). Rebake/refine to apply. */
  setLabelOpacity(id, opacity) {
    if (id < 1 || id > 255) return;
    this.palette[id * 4 + 3] = Math.max(0, Math.min(1, opacity));
  }
  /** Per-segment shading (sdf mode): "surface" = crisp SDF shell (surface model), "volume" = DVR fill
   *  of the interior (translucent cloud). Rebake/refine to apply. */
  setLabelShading(id, shading) {
    if (id < 1 || id > 255) return;
    this.modePalette[id * 4] = shading === "volume" ? 1 : 0;
  }
  /** Re-derive the render texture from the current master + palette (FAST, in place). */
  rebake() {
    if (this.sdf) {
      this.sdf.setPalette(this.palette);
      this.sdf.setModePalette(this.modePalette);
      this.sdf.bake();
    } else this.baker.bakeInto(this.presenceTex, this.palette, this.sigma);
  }
  /** Schedule the settle-refine after quiescence (debounced; sdf mode only). */
  scheduleRefine() {
    if (!this.sdf) return;
    if (this.refineTimer !== void 0) clearTimeout(this.refineTimer);
    this.refineTimer = setTimeout(() => {
      this.refineTimer = void 0;
      this.refineNow();
    }, this.refineDelayMs);
  }
  /** Run the settle-refine now (JFA+2 + tighter distance blur + colour-seam blur), then redraw.
   *  Public so a test — or an app that knows the edit is done — can force the high-quality bake. */
  refineNow() {
    if (this.refineTimer !== void 0) {
      clearTimeout(this.refineTimer);
      this.refineTimer = void 0;
    }
    if (this.sdf) {
      this.sdf.setPalette(this.palette);
      this.sdf.setModePalette(this.modePalette);
      this.sdf.refine();
      for (const cb of this.redrawCbs) cb();
    }
  }
  /** FAST per-segment opacity refresh: attr-only rebake (no JFA re-sweep) — for visibility
   *  toggles where the labelmap and colours are unchanged.
   *
   *  With `regionRAS` (the bbox of the labels whose opacity changed): the finalize AND the
   *  seam blur run region-limited in one shot — full settled quality lands immediately, no
   *  two-phase. Without it: full-volume fast pass + a debounced full-volume seam blur. */
  /** RAS bbox → padded-SDF-grid ijk bbox with an M-voxel margin (shell band + blur radii). */
  regionToIjk(regionRAS, M = 8) {
    const inv = invertAffine(this.sdf.sdfIjkToRAS());
    const lo2 = [Infinity, Infinity, Infinity];
    const hi2 = [-Infinity, -Infinity, -Infinity];
    for (const x of [regionRAS.lo[0], regionRAS.hi[0]]) for (const y2 of [regionRAS.lo[1], regionRAS.hi[1]]) for (const z2 of [regionRAS.lo[2], regionRAS.hi[2]]) {
      const i = inv[0] * x + inv[1] * y2 + inv[2] * z2 + inv[3];
      const j2 = inv[4] * x + inv[5] * y2 + inv[6] * z2 + inv[7];
      const k = inv[8] * x + inv[9] * y2 + inv[10] * z2 + inv[11];
      lo2[0] = Math.min(lo2[0], i);
      lo2[1] = Math.min(lo2[1], j2);
      lo2[2] = Math.min(lo2[2], k);
      hi2[0] = Math.max(hi2[0], i);
      hi2[1] = Math.max(hi2[1], j2);
      hi2[2] = Math.max(hi2[2], k);
    }
    return {
      lo: [Math.floor(lo2[0]) - M, Math.floor(lo2[1]) - M, Math.floor(lo2[2]) - M],
      hi: [Math.ceil(hi2[0]) + M, Math.ceil(hi2[1]) + M, Math.ceil(hi2[2]) + M]
    };
  }
  /** REGION-LIMITED settle-refine after a labelmap edit confined to `regionRAS` (e.g. a
   *  per-vertebra visibility flip written via EditableSegmentation.writeLabelRegion): the full
   *  refine quality — JFA re-flood, finalize, seam blurs — over just the region, immediately.
   *  Small regions bake in ~ms, so stepping through per-label visibility stays real-time. */
  rebakeShellRegion(regionRAS) {
    if (!this.sdf) {
      this.rebake();
      return;
    }
    if (this.refineTimer !== void 0) {
      clearTimeout(this.refineTimer);
      this.refineTimer = void 0;
    }
    this.sdf.setPalette(this.palette);
    this.sdf.setModePalette(this.modePalette);
    this.sdf.refineRegion(this.regionToIjk(regionRAS));
    for (const cb of this.redrawCbs) cb();
  }
  refreshOpacity(regionRAS) {
    if (!this.sdf) {
      this.rebake();
      return;
    }
    this.sdf.setPalette(this.palette);
    if (regionRAS) {
      this.sdf.rebakeAttr(true, this.regionToIjk(regionRAS));
      for (const cb of this.redrawCbs) cb();
      return;
    }
    this.sdf.rebakeAttr(false);
    for (const cb of this.redrawCbs) cb();
    if (this.attrSettleTimer !== void 0) clearTimeout(this.attrSettleTimer);
    this.attrSettleTimer = setTimeout(() => {
      this.attrSettleTimer = void 0;
      if (this.sdf) {
        this.sdf.blurAttrOnly();
        for (const cb of this.redrawCbs) cb();
      }
    }, 600);
  }
  /** A SegmentField bound to the shared render texture — hand this to the SceneRenderer once; edits
   *  update it in place. Colour comes from the texture (per-label); the uniform supplies opacity. */
  field() {
    if (!this.segField) {
      const tex = this.sdf ? this.sdf.sdfTexture() : this.presenceTex;
      const voxelMm = Math.min(...this.seg.spacingMm());
      const interfaceMode = this.renderMode === "sdf" && this.boundaryMode === "all";
      const band = this.bandMm ?? (this.renderMode === "sdf" ? (interfaceMode ? 1.5 : 0.65) * voxelMm : void 0);
      const fdims = this.sdf ? this.sdf.sdfDims() : this.seg.dims;
      const fijk = this.sdf ? this.sdf.sdfIjkToRAS() : this.seg.ijkToRAS;
      this.segField = new SegmentField(tex, fdims, [1, 1, 1], {
        color: [1, 1, 1],
        opacity: this.opacity,
        ijkToRAS: fijk,
        mode: this.renderMode === "sdf" ? "sdf" : "surface",
        colorFromTexture: true,
        bandMm: band,
        clippable: this.clippable,
        attrTexture: this.sdf ? this.sdf.attrTexture() : void 0,
        // per-segment opacity (sdf)
        interfaceMode
      });
    }
    return this.segField;
  }
  /** Live GLOBAL segmentation opacity (0..1) — the field-level multiplier over every segment's own
   *  opacity. Caller does scene.syncUniforms() + redraw. */
  setGlobalOpacity(o) {
    this.opacity = o;
    this.segField?.setOpacity(o);
  }
  /** Notified after every edit (post-rebake) so the app can redraw. */
  onRedraw(cb) {
    this.redrawCbs.push(cb);
  }
  destroy() {
    if (this.refineTimer !== void 0) clearTimeout(this.refineTimer);
    this.unsubDirty();
    this.sdf?.destroy();
    this.baker?.destroy();
    this.presenceTex?.destroy();
  }
};

// logic/seged-manager.ts
function slice2DOpacities2(node, visible) {
  if (!visible) return [0, 0];
  const overall = typeof node.opacity === "number" ? node.opacity : 1;
  const f = node.fill2D;
  const o = node.outline2D;
  const fill = f?.visible ?? true ? overall * (f?.opacity ?? 0.5) : 0;
  const outline = o?.visible ?? true ? overall * (o?.opacity ?? 1) : 0;
  return [fill, outline];
}
var SegEditDisplayableManager = class {
  // visibility/fill/outline signature — re-push to the view only when it changes
  constructor(dev, opts = {}) {
    this.dev = dev;
    this.opts = opts;
  }
  // "segmentation" routes the initial labelmap + SegEdit events (routed by the event's sourceId type);
  // "segEdit" is not a node type — it's in the subscription union so the Slicer live server starts the
  // stroke-intent capture (mrson_live: `if "segEdit" in types`). It never matches in interested().
  interestedTypes = ["segmentation", "segEdit"];
  // We reproduce the labelmap on-GPU from SegEdit intents, so the peer needn't re-stream the heavy
  // authoritative labelmap on every edit — we only need the INITIAL snapshot (geometry + start).
  localBulkTypes = ["segmentation"];
  seg;
  driver;
  logic;
  overlayBaker;
  overlayTex;
  segId;
  dims;
  labelForId = /* @__PURE__ */ new Map();
  // Slicer segment id → master label value
  overlayPalette = new Float32Array(256 * 4);
  added = false;
  visible = true;
  fill = 0.5;
  outline = 1;
  palKey = "";
  // colours+visibility signature — re-bake the palette only when it changes
  dispKey = "";
  building = false;
  async onNodeAdded(node, scene) {
    if (node.type !== "segmentation") return;
    if (this.seg) {
      this.applyDisplay(node, scene);
      return;
    }
    await this.ensureBuilt(node, scene);
  }
  /** Build the WebGPU segmentation from a seg node's GEOMETRY (dims + ijkToRAS). seged is intent-driven
   *  and the labelmap bulk is suppressed (localBulk), so an ABSENT/empty labelmap is the normal case —
   *  seed from the initial zarr if present, else zeros. Geometry alone is enough to reproduce edits. */
  async ensureBuilt(node, scene) {
    if (this.seg) return true;
    if (this.building) return false;
    const dims = node.dims;
    const ijk = node.ijkToRAS;
    if (!dims || dims[0] < 1 || dims[1] < 1 || dims[2] < 1 || !ijk) {
      this.opts.onEdit?.("no geometry yet");
      return false;
    }
    this.building = true;
    const n = dims[0] * dims[1] * dims[2];
    let lab = new Uint8Array(n);
    if (node.zarr) {
      try {
        const zv = await fetchZarrVolume(scene.blobBase(), node.zarr);
        if (zv.data.length === n) lab = Uint8Array.from(zv.data);
      } catch {
      }
    }
    this.segId = node.id;
    this.dims = dims;
    this.updateLabelMap(node.segments);
    this.seg = new EditableSegmentation(this.dev, dims, { ijkToRAS: ijk });
    this.seg.loadLabelmap(lab);
    this.driver = new SegEditDriver(this.seg, {
      labelForSegment: (id) => this.labelForId.get(id) ?? 1,
      imageTex: this.opts.imageTexForSeeds,
      onUnhandled: (k) => this.opts.onEdit?.("unhandled:" + k)
    });
    this.logic = new SegmentationLogic(this.dev, this.seg, { renderMode: "sdf", boundaryMode: "all", opacity: 1 });
    this.overlayBaker = new ColorizeBaker(this.dev, this.seg.masterTexture(), dims);
    this.overlayTex = this.overlayBaker.output();
    scene.view?.setField("seged:" + this.segId, this.logic.field());
    this.added = true;
    this.logic.onRedraw(() => {
      this.bakeOverlay();
      this.pushToView(scene);
    });
    this.building = false;
    this.opts.onEdit?.(`built ${dims[0]}\xD7${dims[1]}\xD7${dims[2]}`);
    this.applyDisplay(node, scene);
    return true;
  }
  async onEvent(ev, scene) {
    if (ev.event === "SegEdit") {
      if (!this.seg) {
        const n = scene.nodes.get(ev.sourceId);
        if (n) await this.ensureBuilt(n, scene);
      }
      if (this.driver && ev.sourceId === this.segId) {
        const kind = SegEditDriver.unwrap(ev)?.kind ?? "?";
        await this.driver.applyEdit(ev);
        this.opts.onEdit?.("\u270F " + kind);
      } else {
        this.opts.onEdit?.("intent dropped (no seg)");
      }
      return;
    }
    if (ev.event === "SegmentationDisplayModified" && ev.sourceId === this.segId) {
      this.applyDisplay(ev.display ?? ev, scene);
    }
  }
  onNodeRemoved(id, scene) {
    if (id === this.segId) this.reset(scene);
  }
  onSceneClosed(scene) {
    this.reset(scene);
  }
  /** Debug: current segment-id→label map + the non-zero palette colours actually applied. */
  diag() {
    const pal = {};
    for (let lv = 1; lv < 256; lv++) {
      const o = lv * 4;
      if (this.overlayPalette[o + 3] > 0) pal[lv] = [+this.overlayPalette[o].toFixed(3), +this.overlayPalette[o + 1].toFixed(3), +this.overlayPalette[o + 2].toFixed(3)];
    }
    return { built: !!this.seg, segId: this.segId, labelForId: Object.fromEntries(this.labelForId), palette: pal, palKey: this.palKey };
  }
  // ── palette / display ──────────────────────────────────────────────────────
  /** MERGE segment-id → label-value from any node/display carrying ids. Must stay current as segments
   *  are ADDED in Slicer (a new segment's stroke would otherwise fall back to label 1 = the wrong
   *  colour). Display events now carry `id` too (serialize_mrson), so this updates on segment add. */
  updateLabelMap(segs) {
    for (const s of segs ?? []) if (s.id) this.labelForId.set(s.id, s.labelValue);
  }
  /** Set per-label colour/opacity on the logic (3D) and the overlay palette (2D), then re-bake — but
   *  ONLY when the palette actually changed. A paint stroke's authoritative labelmap echo carries the
   *  SAME palette, so without this guard every stroke would trigger a full (expensive) SDF refineNow(),
   *  which competes with rendering and reads as lag. */
  applyPalette(disp) {
    const segs = disp.segments ?? [];
    this.updateLabelMap(segs);
    const key = segs.map((s) => `${s.labelValue}:${s.color.map((x) => x.toFixed(3)).join(",")}:${s.visible !== false}`).join("|");
    if (key === this.palKey) return false;
    this.palKey = key;
    this.overlayPalette.fill(0);
    for (const s of segs) {
      const lv = s.labelValue, on2 = s.visible !== false;
      if (lv > 0 && lv < 256) {
        this.logic?.setLabelColor(lv, [s.color[0], s.color[1], s.color[2]]);
        this.logic?.setLabelOpacity(lv, on2 ? 1 : 0);
        if (on2) {
          const o = lv * 4;
          this.overlayPalette[o] = s.color[0];
          this.overlayPalette[o + 1] = s.color[1];
          this.overlayPalette[o + 2] = s.color[2];
          this.overlayPalette[o + 3] = 1;
        }
      }
    }
    this.logic?.refineNow();
    return true;
  }
  /** A display change (visibility / opacity / colour) from Slicer — no labelmap touch. No-op when
   *  nothing changed (so a per-stroke labelmap echo doesn't re-bake or re-render). */
  applyDisplay(disp, scene) {
    const vis = disp.visible !== false;
    const [f, o] = slice2DOpacities2(disp, vis);
    const paletteChanged = disp.segments ? this.applyPalette(disp) : false;
    const dispKey = `${vis}:${f.toFixed(3)}:${o.toFixed(3)}`;
    if (dispKey !== this.dispKey) {
      this.visible = vis;
      this.fill = f;
      this.outline = o;
      this.dispKey = dispKey;
      this.pushToView(scene);
    } else if (paletteChanged) this.pushToView(scene);
  }
  bakeOverlay() {
    if (this.overlayBaker && this.overlayTex) this.overlayBaker.bakeInto(this.overlayTex, this.overlayPalette, 0);
  }
  pushToView(scene) {
    scene.view?.setSegmentationOverlay(this.visible ? this.overlayTex : null, this.fill, this.outline);
    if (this.visible) {
      if (!this.added && this.logic) {
        scene.view?.setField("seged:" + this.segId, this.logic.field());
        this.added = true;
      } else scene.view?.redraw();
    } else if (this.added) {
      scene.view?.removeField("seged:" + this.segId);
      this.added = false;
    }
  }
  reset(scene) {
    if (this.segId) scene.view?.removeField("seged:" + this.segId);
    scene.view?.setSegmentationOverlay(null, 0, 0);
    this.logic?.destroy();
    this.driver?.destroy();
    this.overlayBaker?.destroy();
    this.overlayTex?.destroy();
    this.seg?.destroy();
    this.seg = void 0;
    this.driver = void 0;
    this.logic = void 0;
    this.overlayBaker = void 0;
    this.overlayTex = void 0;
    this.segId = void 0;
    this.added = false;
    this.building = false;
    this.labelForId.clear();
    this.palKey = "";
    this.dispKey = "";
  }
};

// ../../SlicerHub-js-clients/packages/client/dist/hub-client.js
function Ar(e, t) {
  t.classHierarchy || (t.classHierarchy = []), e.getClassName = () => t.classHierarchy[t.classHierarchy.length - 1] || "HubClient", e.isA = (n) => t.classHierarchy.includes(n), typeof e.delete != "function" && (e.delete = () => {
  });
}
function Ir(e, t, n) {
  n.forEach((i) => {
    const r = `get${i.charAt(0).toUpperCase()}${i.slice(1)}`;
    e[r] = () => t[i];
  });
}
function Ur(e, t) {
  return () => {
    e(), typeof t == "function" && t();
  };
}
function xr(e, t) {
  return function(i = {}) {
    const r = {};
    return e(r, { classHierarchy: [t] }, i), r;
  };
}
var Et = "UNKNOWN";
var Kt = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
function dn(e, t = Et) {
  return String(e || t).trim().replace(/[^a-zA-Z0-9]+/g, "-").replace(/^-+|-+$/g, "") || t;
}
function Hr(e) {
  return `${dn(e)}-`;
}
function Re(e) {
  return typeof crypto < "u" && crypto.randomUUID ? e + crypto.randomUUID().replace(/-/g, "").slice(0, 16) : e + Math.random().toString(36).substring(2, 18);
}
function wt(e = Et) {
  const t = dn(e);
  let n = "";
  for (let i = 0; i < 6; i++) {
    const r = Math.floor(Math.random() * Kt.length);
    n += Kt[r];
  }
  return `${t}-${n}`;
}
var Qt = [
  "dicom",
  "nifti",
  "jpg",
  "png",
  "nrrd",
  "imagingstudy",
  "scene"
  // scene-update (LiveSync HubBlobs multipart)
];
function ve(e) {
  if (typeof e != "string") return false;
  const t = e.trim().toLowerCase();
  if (!t) return false;
  for (let n = 0; n < Qt.length; n++) {
    const i = Qt[n];
    if (t === i || t.startsWith(`${i}-`) || t.startsWith(`${i}_`))
      return true;
  }
  return false;
}
var $r = ve;
function kr(e) {
  return typeof ArrayBuffer < "u" && ArrayBuffer.isView(e);
}
async function Fe(e) {
  if (e instanceof ArrayBuffer)
    return e;
  if (kr(e)) {
    const { buffer: t, byteOffset: n, byteLength: i } = e;
    return t.slice(n, n + i);
  }
  if (typeof Blob < "u" && e instanceof Blob && typeof e.arrayBuffer == "function")
    return e.arrayBuffer();
  throw new Error(
    "HubClient: binary resource.data must be ArrayBuffer, TypedArray, DataView, Blob, or File"
  );
}
function Ae(e) {
  const t = e && e.context;
  if (!t || typeof t != "object" || Array.isArray(t))
    return [];
  const n = t.files;
  return Array.isArray(n) ? n.filter((i) => i && typeof i == "object") : [];
}
function Tt(e) {
  if (!e || typeof e != "object" || e.data != null)
    return { payloadIds: [], chunkByteLengths: [] };
  const t = e.payloadIds;
  if (Array.isArray(t) && t.length) {
    const i = t.map((o) => typeof o == "string" ? o.trim() : "").filter(Boolean), r = Array.isArray(e.chunkByteLengths) ? e.chunkByteLengths.filter((o) => typeof o == "number" && o >= 0) : [];
    if (r.length === i.length)
      return { payloadIds: i, chunkByteLengths: r };
    if (i.length === 1) {
      const o = typeof e.byteLength == "number" && e.byteLength >= 0 ? e.byteLength : null;
      return {
        payloadIds: i,
        chunkByteLengths: o != null ? [o] : []
      };
    }
  }
  const n = typeof e.payloadId == "string" ? e.payloadId.trim() : "";
  if (n) {
    const i = typeof e.byteLength == "number" && e.byteLength >= 0 ? e.byteLength : null;
    return {
      payloadIds: [n],
      chunkByteLengths: i != null ? [i] : []
    };
  }
  return { payloadIds: [], chunkByteLengths: [] };
}
function _r(e) {
  const t = Ae(e);
  for (let n = 0; n < t.length; n++) {
    const i = t[n];
    if (i.data == null) {
      const r = Tt(i);
      if (r.payloadIds.length)
        return {
          kind: "files",
          index: n,
          chunkIndex: 0,
          payloadId: r.payloadIds[0],
          expectedChunkBytes: r.chunkByteLengths[0] ?? null
        };
    }
  }
  return null;
}
function bn(e) {
  return _r(e);
}
function ut(e) {
  return !!bn(e);
}
function lt(e, t) {
  return (typeof e == "string" ? e.trim().toLowerCase() : "").startsWith("nifti") ? t === 0 ? "nifti-send.nii.gz" : `nifti-send-${t + 1}.nii.gz` : t === 0 ? "dicom-send.dcm" : `dicom-send-${t + 1}.dcm`;
}
function ft(e) {
  return (typeof e == "string" ? e.trim().toLowerCase() : "").startsWith("nifti") ? "application/octet-stream" : "application/dicom";
}
function ct(e) {
  const t = e && e.event;
  if (!t || !ve(t["hub.event"]) || Ae(t).length)
    return e;
  const n = t["hub.event"], i = t.context;
  let r = [];
  Array.isArray(i) ? r = i : i != null && (r = [i]);
  const o = [];
  for (let s = 0; s < r.length; s++) {
    const a = r[s];
    if (a && typeof a == "object") {
      const l = a.resource;
      if (l && typeof l == "object" && l.data != null) {
        const h = { ...l };
        (typeof h.fileName != "string" || !h.fileName.trim()) && (h.fileName = lt(n, s)), (typeof h.mimeType != "string" || !h.mimeType.trim()) && (h.mimeType = ft(n)), o.push(h);
      }
    }
  }
  return o.length ? {
    ...e,
    event: {
      ...t,
      context: {
        files: o
      }
    }
  } : e;
}
function Rr(e, t) {
  const n = e && e.event, i = n && n.context;
  let r = [];
  Array.isArray(i) ? r = i : i != null && (r = [i]);
  for (let o = 0; o < r.length; o++) {
    const s = r[o];
    if (s && typeof s == "object") {
      const a = s.resource;
      if (a && typeof a == "object") {
        const l = typeof a.fileName == "string" ? a.fileName.trim() : "";
        if (l)
          return l;
      }
    }
  }
  return t;
}
function De(e) {
  if (!e || typeof e != "object")
    return false;
  const t = typeof e.url == "string" ? e.url.trim() : "";
  if (!t)
    return false;
  const n = t.toLowerCase();
  return n.startsWith("http://") || n.startsWith("https://");
}
function Or(e) {
  const t = [];
  for (let n = 0; n < e.length; n += 1)
    De(e[n]) || t.push(n);
  return t;
}
function Mr(e) {
  const t = e && e.event;
  if (!t || !ve(t["hub.event"]))
    return false;
  const n = Ae(t);
  return n.length ? n.some(
    (i) => i.data != null && !De(i)
  ) : false;
}
async function Pr(e, t = "", n = 0) {
  if (!e || typeof e != "object")
    throw new Error(
      "HubClient: binary batch publish files[] entries must be objects"
    );
  const i = De(e);
  let r = typeof e.byteLength == "number" && e.byteLength >= 0 ? e.byteLength : null;
  if ("data" in e && e.data != null) {
    if (typeof e.data == "string")
      throw new Error(
        "HubClient: binary batch publish string payloads are not supported; pass binary input instead"
      );
    r = (await Fe(e.data)).byteLength;
  }
  if (r == null && i) {
    const s = { ...e };
    return delete s.data, delete s.binaryTransfer, delete s.payloadId, delete s.payloadIds, delete s.chunkByteLengths, delete s.expiresAt, (typeof s.fileName != "string" || !s.fileName.trim()) && (s.fileName = lt(t, n)), (typeof s.mimeType != "string" || !s.mimeType.trim()) && (s.mimeType = ft(t)), s;
  }
  if (r == null)
    throw new Error(
      "HubClient: binary batch publish requires files[].data or files[].byteLength"
    );
  const o = { ...e };
  return delete o.data, delete o.binaryTransfer, i || delete o.url, delete o.payloadId, delete o.payloadIds, delete o.chunkByteLengths, delete o.expiresAt, o.byteLength = r, (typeof o.fileName != "string" || !o.fileName.trim()) && (o.fileName = lt(t, n)), (typeof o.mimeType != "string" || !o.mimeType.trim()) && (o.mimeType = ft(t)), o;
}
async function Fr(e) {
  if (!e.event || typeof e.event != "object")
    throw new Error("HubClient: binary batch publish requires event object");
  const t = e.event;
  if (!ve(t["hub.event"]))
    return e;
  const n = t.context;
  if (!n || typeof n != "object" || Array.isArray(n))
    throw new Error(
      "HubClient: binary batch publish requires event.context object"
    );
  const i = n.files;
  if (!Array.isArray(i) || !i.length)
    throw new Error(
      "HubClient: binary batch publish requires non-empty event.context.files[]"
    );
  const r = t["hub.event"], o = await Promise.all(
    i.map(
      (s, a) => Pr(s, r, a)
    )
  );
  return {
    ...e,
    event: {
      ...t,
      context: {
        ...n,
        files: o
      }
    }
  };
}
async function en(e) {
  const t = e && e.event;
  if (!t)
    return [];
  const i = Ae(t).filter(
    (r) => r && typeof r == "object" && r.data != null && !De(r)
  );
  return Promise.all(i.map((r) => Fe(r.data)));
}
function Dr(e, t, n) {
  const i = new TextEncoder(), r = i.encode(`\r
`), o = [], s = (a) => {
    o.push(i.encode(a));
  };
  s(
    `--${e}\r
Content-Type: application/dicom+json\r
\r
${t}\r
`
  );
  for (let a = 0; a < n.length; a++) {
    const l = n[a], h = l && l.mimeType && String(l.mimeType).trim() || "application/octet-stream";
    s(`--${e}\r
Content-Type: ${h}\r
\r
`), o.push(new Uint8Array(l.buffer)), o.push(r);
  }
  return s(`--${e}--\r
`), new Blob(o);
}
async function Br({
  msg: e,
  fileBytesList: t,
  hub: n,
  messageIdPrefix: i
}) {
  const r = await Fr(e), o = await Promise.all(
    (t || []).map((u) => Fe(u))
  ), s = Ae(r.event), a = Or(s);
  if (o.length !== a.length)
    throw new Error(
      `HubClient: binary batch publish expected ${a.length} file part(s), got ${o.length}`
    );
  const l = o.map((u, d) => {
    const f = a[d];
    return {
      buffer: u,
      mimeType: s[f] && s[f].mimeType || "application/octet-stream"
    };
  }), h = `hub-batch-${Re(i())}`, c = Dr(
    h,
    JSON.stringify(r),
    l
  );
  try {
    return await fetch(n.hub_endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${n.token}`,
        "Content-Type": `multipart/related; boundary="${h}"; type="application/dicom"`
      },
      body: c
    });
  } catch (u) {
    const d = u instanceof Error ? u.message : String(u);
    return console.debug("HubClient:", d), null;
  }
}
function Be(e) {
  const t = {
    responses: [],
    expected: [],
    missing: [],
    timedOut: false,
    ok: false,
    id: null,
    actor: null,
    productName: null
  };
  if (!e || typeof e != "object")
    return t;
  const n = (
    /** @type {Record<string, unknown>} */
    e
  ), i = Array.isArray(n.responses) ? n.responses.filter((s) => s && typeof s == "object") : [], r = Array.isArray(n.expected) ? n.expected.map((s) => String(s)) : [], o = Array.isArray(n.missing) ? n.missing.map((s) => String(s)) : [];
  return {
    responses: i,
    expected: r,
    missing: o,
    timedOut: !!n.timedOut,
    ok: !!n.ok,
    id: n.id != null ? String(n.id) : null,
    actor: n.actor != null ? String(n.actor) : null,
    productName: n.productName != null ? String(n.productName) : null
  };
}
function jr(e) {
  return Be(e).responses;
}
function hn(e, t) {
  if (t === "file:")
    return true;
  const n = String(e || "").toLowerCase();
  return !!(n === "localhost" || n === "127.0.0.1" || n === "[::1]" || /^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(n) || /^192\.168\.\d{1,3}\.\d{1,3}$/.test(n) || /^172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}$/.test(n));
}
function mn(e = typeof window < "u" ? window.location : null) {
  return e ? !hn(e.hostname, e.protocol) : false;
}
function gn(e) {
  try {
    const t = new URL(e);
    return !hn(t.hostname, t.protocol);
  } catch {
    return false;
  }
}
function qr(e, t, n = mn()) {
  const i = t.find((o) => {
    var a;
    const s = (a = e[o]) == null ? void 0 : a.hubEndpoint;
    return s && gn(s) === n;
  });
  if (i)
    return i;
  const r = n ? "cloud" : "local";
  return e[r] ? r : t[0];
}
var Se = "-request";
var Ce = "-response";
var Wr = "imagingstudy-open";
var zr = "imagingstudy-close";
var dt = "status-update";
var Vr = "scene-update";
var Gr = "annotation-update";
var Xr = "annotation-delete";
var Jr = "dicom-send";
var Yr = "nifti-send";
var Zr = "png-send";
var Kr = "idc-claude-send";
var Qr = "conference-chat";
var ei = "dicomsr-update";
var ti = "subscription-removed";
var ni = "heartbeat";
function Nt(e) {
  return typeof e != "string" ? "" : e.trim().toLowerCase();
}
function Ie(e) {
  const t = Nt(e);
  return t ? `${t}${Se}` : "";
}
function pn(e) {
  const t = Nt(e);
  return t ? `${t}${Ce}` : "";
}
function ri(e) {
  return typeof e != "string" ? false : e.endsWith(Se);
}
function ii(e) {
  return typeof e != "string" ? false : e.endsWith(Ce);
}
function oi(e) {
  return typeof e != "string" ? "" : e.endsWith(Se) ? e.slice(0, -Se.length) : e.endsWith(Ce) ? e.slice(0, -Ce.length) : "";
}
function yn(e) {
  if (!(e != null && e.length))
    return ["*"];
  if (e.some((r) => String(r).trim() === "*"))
    return ["*"];
  const t = Ie("STATUS"), n = new Set(
    e.map((r) => String(r).trim().toLowerCase()).filter(Boolean)
  ), i = [];
  return t && !n.has(t.toLowerCase()) && (i.push(t), n.add(t.toLowerCase())), n.has(dt) || i.push(dt), i.length ? [...e, ...i] : [...e];
}
var si = yn;
var Sn = "hub-radio-icon";
var Cn = `<svg class="${Sn}" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9"/><path d="M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5"/><circle cx="12" cy="12" r="2"/><path d="M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5"/><path d="M19.1 4.9C23 8.8 23 15.1 19.1 19"/></svg>`;
var En = "hub-radio-status-icon";
var wn = "hub-radio-slash";
var Tn = "hub-conference-icon";
var ai = `<svg class="${Tn}" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>`;
function ci(e = {}) {
  const t = e.slashId || "castRadioSlash";
  return `<span class="${En}">${Cn}<span id="${t}" class="${wn}" aria-hidden="true" hidden></span></span>`;
}
function ui(e, t = {}) {
  return t == null || t.conferenceActive, e === "connected" ? {
    tone: "connected",
    showSlash: false,
    pulse: false,
    conferencePulse: false
  } : e === "connecting" || e === "token-ready" ? {
    tone: "connecting",
    showSlash: false,
    pulse: true,
    conferencePulse: false
  } : e === "error" || e === "disconnected" ? {
    tone: "error",
    showSlash: true,
    pulse: false,
    conferencePulse: false
  } : {
    tone: "idle",
    showSlash: true,
    pulse: false,
    conferencePulse: false
  };
}
function li() {
  return {
    name: "",
    friendlyName: "",
    version: "",
    hub_endpoint: "",
    authorization_endpoint: "",
    token_endpoint: "",
    client_id: "",
    client_secret: ""
  };
}
function fi() {
  return {
    subscriberName: "",
    productName: "",
    productVersion: "",
    actors: [],
    topic: "",
    events: [],
    lease: 999,
    userName: "",
    defaultTargetActor: ""
  };
}
function Oe(e) {
  if (e == null)
    return;
  const t = String(e).trim();
  if (!(!t || t === "*"))
    return t;
}
function Nn(e) {
  if (e == null)
    return;
  const t = String(e).trim();
  if (!(!t || t === "*"))
    return t;
}
function di() {
  return {
    token: "",
    lastIdToken: "",
    lastPublishedMessageID: "",
    subscribed: false,
    resubscribeRequested: false,
    websocket: null
  };
}
function bi() {
  if (typeof navigator > "u")
    return null;
  const e = {};
  typeof navigator.userAgent == "string" && navigator.userAgent.trim() && (e.userAgent = navigator.userAgent.trim()), typeof navigator.platform == "string" && navigator.platform.trim() && (e.platform = navigator.platform.trim()), typeof navigator.language == "string" && navigator.language.trim() && (e.language = navigator.language.trim());
  try {
    const t = Intl.DateTimeFormat().resolvedOptions().timeZone;
    typeof t == "string" && t.trim() && (e.timezone = t.trim());
  } catch {
  }
  return Object.keys(e).length ? e : null;
}
function tn(e) {
  return e || (typeof window < "u" ? `${window.location.origin}/castCallback` : "");
}
function nn(e, t, n, i = "") {
  const r = new URLSearchParams();
  r.append("hub.mode", e), r.append("hub.channel.type", "websocket"), r.append("hub.callback", n), r.append("hub.events", (t.events || []).toString()), r.append("hub.topic", t.topic || ""), r.append("hub.lease", String(t.lease || 999)), r.append("subscriber.name", t.subscriberName || ""), r.append("subscriber.product.name", t.productName || ""), r.append(
    "subscriber.product.version",
    t.productVersion || i || ""
  );
  const o = (t.actors || []).map((a) => a.trim()).filter(Boolean);
  o.length && r.append("subscriber.actors", JSON.stringify(o));
  const s = bi();
  return s && r.append("subscriber.client_info", JSON.stringify(s)), r;
}
var ae = 25;
function hi(e) {
  try {
    const t = new URL(e);
    return t.hostname.toLowerCase() === "localhost" && (t.hostname = "127.0.0.1"), t.toString();
  } catch {
    return e;
  }
}
function mi(e) {
  if (!(e instanceof Error))
    return false;
  const t = e.name || "";
  return t === "TypeError" || t === "NetworkError";
}
function gi(e, t) {
  if (!e)
    return "";
  try {
    const n = `/api/hub/payloads/${encodeURIComponent(e)}`;
    return hi(new URL(n, t).toString());
  } catch (n) {
    return console.warn("HubClient: invalid payloadId url", e, n), "";
  }
}
function pi(e) {
  delete e.binaryTransfer, delete e.url, delete e.payloadId, delete e.payloadIds, delete e.chunkByteLengths, delete e.expiresAt;
}
function Ln(e, t) {
  const n = e.reduce((o, s) => o + s.byteLength, 0);
  if (typeof t == "number" && t >= 0 && n !== t)
    throw new Error(
      `HubClient: payload size mismatch expected=${t} received=${n}`
    );
  const i = new Uint8Array(n);
  let r = 0;
  for (let o = 0; o < e.length; o++)
    i.set(new Uint8Array(e[o]), r), r += e[o].byteLength;
  return i.buffer;
}
function rn(e, t, n) {
  const i = e && e.event, r = i && i.context, o = r && r.files, s = o && o[t];
  return !s || typeof s != "object" ? false : (pi(s), s.data = n, s.byteLength = n.byteLength, true);
}
async function vn(e, t, { hubEndpoint: n, accessToken: i }) {
  const r = gi(e, n);
  if (!r)
    throw new Error("HubClient.fetchPayload: missing or invalid payloadId");
  const o = {};
  i && (o.Authorization = `Bearer ${i}`);
  async function s() {
    const a = await fetch(r, {
      method: "GET",
      credentials: "omit",
      headers: o
    });
    if (!a.ok)
      throw new Error(
        `HubClient.fetchPayload: GET ${a.status} ${a.statusText}`
      );
    const l = await a.arrayBuffer();
    if (typeof t == "number" && t >= 0 && l.byteLength !== t)
      throw new Error(
        `HubClient.fetchPayload: chunk size mismatch payloadId=${e.slice(
          0,
          8
        )} expected=${t} received=${l.byteLength}`
      );
    return l;
  }
  try {
    return await s();
  } catch (a) {
    if (mi(a))
      return s();
    throw a;
  }
}
async function yi(e, t) {
  const n = Tt(e);
  if (!n.payloadIds.length)
    throw new Error("HubClient.fetchPayload: no payloadIds on file entry");
  const i = [];
  for (let r = 0; r < n.payloadIds.length; r += ae) {
    const o = n.payloadIds.slice(
      r,
      r + ae
    ), s = n.chunkByteLengths.slice(
      r,
      r + ae
    ), a = await Promise.all(
      o.map(
        (l, h) => vn(l, s[h] ?? null, t)
      )
    );
    i.push(...a);
  }
  return Ln(
    i,
    typeof e.byteLength == "number" ? e.byteLength : null
  );
}
async function Si(e, t, n = ae) {
  if (!e.length)
    return [];
  const i = Math.max(1, n | 0), r = new Array(e.length);
  let o = 0;
  async function s() {
    for (; o < e.length; ) {
      const l = o++, h = e[l];
      r[l] = await vn(
        h.payloadId,
        h.expectedLength ?? null,
        t
      );
    }
  }
  const a = Array.from(
    { length: Math.min(i, e.length) },
    () => s()
  );
  return await Promise.all(a), r;
}
function on(e) {
  return JSON.parse(JSON.stringify(e));
}
function Ci(e) {
  const t = () => e();
  return {
    hasPendingPayload(n) {
      const i = n && n.event;
      return !!(i && ut(i));
    },
    async fetchPayload(n) {
      const i = n && n.event;
      if (!i || !ut(i))
        return n;
      const r = bn(i);
      if (!r)
        return n;
      const o = typeof performance < "u" && performance.now ? performance.now() : Date.now(), s = i.context.files[r.index], a = await yi(s, t()), l = on(n);
      if (!rn(l, r.index, a))
        throw new Error("HubClient.fetchPayload: no payload slot on message");
      const c = (typeof performance < "u" && performance.now ? performance.now() : Date.now()) - o;
      return console.debug(
        `HubClient: payload fetched fileIndex=${r.index} bytes=${a.byteLength} elapsed=${(c / 1e3).toFixed(2)}s`
      ), l;
    },
    async fetchAllPayloads(n) {
      const i = n && n.event, r = i && i.context, o = r && typeof r == "object" && !Array.isArray(r) && Array.isArray(r.files) ? r.files : [], s = [];
      for (let f = 0; f < o.length; f++) {
        const m = o[f];
        if (!m || typeof m != "object" || m.data != null)
          continue;
        const g = Tt(m);
        if (g.payloadIds.length)
          for (let p = 0; p < g.payloadIds.length; p++)
            s.push({
              fileIndex: f,
              chunkIndex: p,
              payloadId: g.payloadIds[p],
              expectedLength: typeof g.chunkByteLengths[p] == "number" ? g.chunkByteLengths[p] : null
            });
      }
      if (!s.length)
        return n;
      const a = typeof performance < "u" && performance.now ? performance.now() : Date.now(), l = t(), h = await Si(
        s,
        l,
        ae
      ), c = /* @__PURE__ */ new Map();
      for (let f = 0; f < s.length; f++) {
        const { fileIndex: m, chunkIndex: g } = s[f];
        let p = c.get(m);
        p || (p = [], c.set(m, p)), p[g] = h[f];
      }
      const u = on(n);
      for (const [f, m] of c) {
        const g = u.event.context.files[f], p = typeof g.byteLength == "number" ? g.byteLength : null, S = Ln(m, p);
        if (!rn(u, f, S))
          throw new Error("HubClient.fetchAllPayloads: attach failed");
      }
      const d = (typeof performance < "u" && performance.now ? performance.now() : Date.now()) - a;
      return console.debug(
        `HubClient: fetchAllPayloads files=${c.size} chunks=${s.length} concurrent=${ae} elapsed=${(d / 1e3).toFixed(2)}s`
      ), u;
    }
  };
}
function Ue(e) {
  const t = String(e ?? "").trim();
  if (!t)
    return null;
  try {
    const n = new URL(t);
    return n.protocol === "ws:" ? n.protocol = "http:" : n.protocol === "wss:" && (n.protocol = "https:"), n;
  } catch {
    return null;
  }
}
function Ei(e) {
  const t = Ue(e);
  if (!t)
    return "";
  const n = t.href.endsWith("/") ? t.href : `${t.href}/`;
  return new URL("admin", n).href;
}
function wi(e, t = {}) {
  var l, h, c, u;
  const n = Ue(e);
  if (!n)
    return "";
  const i = new URL("/api/hub/conference-client", n.origin), r = (l = t.subscriberName) == null ? void 0 : l.trim(), o = (h = t.topic) == null ? void 0 : h.trim(), s = ((c = t.theme) == null ? void 0 : c.trim()) || "volview", a = ((u = t.mode) == null ? void 0 : u.trim().toLowerCase()) === "light" ? "light" : "dark";
  return r && i.searchParams.set("subscriberName", r), o && i.searchParams.set("topic", o), i.searchParams.set("theme", s), i.searchParams.set("mode", a), i.href;
}
var Ti = { width: 336, height: 288 };
var Ni = "hubViewer-reporting";
var Li = "hubViewer-classroom";
var vi = "hubViewer-worklist";
function Ai(e = null) {
  const t = typeof globalThis < "u" ? globalThis.screen : null, n = typeof globalThis < "u" && globalThis.window ? globalThis.window : null, i = Number(t == null ? void 0 : t.availLeft) || 0, r = Number(t == null ? void 0 : t.availTop) || 0, o = (t == null ? void 0 : t.availWidth) || (t == null ? void 0 : t.width) || 1200, s = (t == null ? void 0 : t.availHeight) || (t == null ? void 0 : t.height) || 800, a = (e == null ? void 0 : e.left) ?? (n == null ? void 0 : n.screenX) ?? (n == null ? void 0 : n.screenLeft) ?? 0, l = (e == null ? void 0 : e.top) ?? (n == null ? void 0 : n.screenY) ?? (n == null ? void 0 : n.screenTop) ?? 0, h = Math.max(
    1,
    Number(
      (e == null ? void 0 : e.width) ?? (n == null ? void 0 : n.outerWidth) ?? Math.floor(o * 0.5)
    ) || Math.floor(o * 0.5)
  ), c = Math.min(756, Math.max(600, Math.floor(o * 0.444))), u = Math.min(784, Math.max(624, Math.floor(s * 0.688))), d = Math.max(0, Math.floor(h * 0.1));
  let f = a + h - d, m = l;
  return f + c > i + o && (f = Math.max(i, i + o - c)), m + u > r + s && (m = Math.max(r, r + s - u)), f = Math.max(i, Math.min(f, i + o - c)), m = Math.max(r, Math.min(m, r + s - u)), {
    width: c,
    height: u,
    left: Math.floor(f),
    top: Math.floor(m)
  };
}
function Ii(e) {
  return [
    "popup",
    `width=${e.width}`,
    `height=${e.height}`,
    `left=${e.left}`,
    `top=${e.top}`
  ].join(",");
}
function Ui(e, t) {
  if (!(!e || !t))
    try {
      e.resizeTo(t.width, t.height), e.moveTo(t.left, t.top), e.focus();
    } catch {
    }
}
function xi(e, t, n = { width: 800, height: 600 }) {
  if (!e || typeof window > "u")
    return;
  const i = n.width, r = n.height, o = Math.max(0, Math.floor((window.screen.width - i) / 2)), s = Math.max(0, Math.floor((window.screen.height - r) / 2)), a = [
    "popup",
    `width=${i}`,
    `height=${r}`,
    `left=${o}`,
    `top=${s}`,
    "noopener",
    "noreferrer"
  ].join(",");
  window.open(e, t, a);
}
var Hi = 3e4;
var An = 1200;
var In = [
  "ST-444 \xB7 Anatomie humaine",
  "Tumor Board",
  "Case discussion",
  "Pedicle screw"
];
function xe(e, t) {
  const n = Ue(e);
  return n ? new URL(t, n.origin).href : null;
}
function Lt(e) {
  if (!Array.isArray(e))
    return [];
  const t = /* @__PURE__ */ new Set();
  return e.map((n) => String(n ?? "").trim()).filter((n) => !n || t.has(n) ? false : (t.add(n), true));
}
function X(e) {
  return String((e == null ? void 0 : e.hostTopic) ?? (e == null ? void 0 : e.user) ?? "").trim();
}
function vt(e) {
  const t = X(e);
  return String((e == null ? void 0 : e.sceneLeaderTopic) ?? "").trim() || t;
}
function At(e, t) {
  const n = X(t), i = String(e ?? "").trim();
  return !i || !n ? false : i === n || i.toLowerCase() === n.toLowerCase();
}
function Un(e, t) {
  const n = vt(t), i = String(e ?? "").trim();
  return !i || !n ? false : i === n || i.toLowerCase() === n.toLowerCase();
}
function It(e, t, n) {
  const i = String(e ?? "").trim(), r = String(t ?? "").trim(), o = X(n), s = Array.isArray(n == null ? void 0 : n.topics) ? n.topics.map((a) => String(a).trim()).filter(Boolean) : [];
  return !!(i && (i === o || s.includes(i)) || r && r === o);
}
function Ee(e, t, n) {
  return n.find(
    (i) => It(e, t, i)
  ) || null;
}
async function xn(e) {
  const t = xe(e, "/api/hub/conference-topics");
  if (!t)
    return [];
  try {
    const n = await fetch(t);
    if (!n.ok)
      return [];
    const i = await n.json();
    if (!Array.isArray(i))
      return [];
    const r = [];
    for (const o of i) {
      if (o && typeof o == "object") {
        const a = String(o.topic ?? "").trim();
        if (!a || a === "*") continue;
        r.push({
          topic: a,
          timezone: String(o.timezone ?? "").trim()
        });
        continue;
      }
      const s = String(o ?? "").trim();
      !s || s === "*" || r.push({ topic: s, timezone: "" });
    }
    return r;
  } catch {
    return [];
  }
}
async function we(e) {
  const t = Ue(e);
  if (!t)
    return [];
  const n = new URL("/api/hub/conference", t.origin).href;
  try {
    const i = await fetch(n);
    if (!i.ok)
      return [];
    const r = await i.json();
    return Array.isArray(r) ? r : [];
  } catch {
    return [];
  }
}
async function Hn(e, t, n, i, r) {
  const o = xe(e, "/api/hub/conference");
  if (!o)
    throw new Error("Invalid hub endpoint");
  const s = {
    hostTopic: String(t).trim(),
    title: String(n).trim(),
    topics: i
  }, a = String(r ?? "").trim();
  a && (s.hostUserName = a);
  const l = await fetch(o, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(s)
  });
  if (!l.ok) {
    const h = await l.text();
    throw new Error(h || `HTTP ${l.status}`);
  }
}
async function $n(e, t, n, i) {
  const r = xe(e, "/api/hub/conference/join");
  if (!r)
    throw new Error("Invalid hub endpoint");
  const o = {
    hostTopic: String(t).trim(),
    joinTopic: String(n).trim()
  }, s = String(i ?? "").trim();
  s && (o.joinUserName = s);
  const a = await fetch(r, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(o)
  });
  if (!a.ok) {
    const l = await a.text();
    throw new Error(l || `HTTP ${a.status}`);
  }
}
async function Ut(e, t, n) {
  const i = xe(e, "/api/hub/conference");
  if (!i)
    throw new Error("Invalid hub endpoint");
  const r = { hostTopic: String(t).trim() }, o = n == null ? void 0 : n.trim();
  o && (r.leaveTopic = o);
  const s = await fetch(i, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(r)
  });
  if (!s.ok) {
    const a = await s.text();
    throw new Error(a || `HTTP ${s.status}`);
  }
}
async function $i(e, t, n, i) {
  const r = xe(e, "/api/hub/conference/lead");
  if (!r)
    throw new Error("Invalid hub endpoint");
  const o = {
    hostTopic: String(t).trim(),
    sceneLeaderTopic: String(n).trim()
  }, s = String(i ?? "").trim();
  s && (o.leaderUserName = s);
  const a = await fetch(r, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(o)
  });
  if (!a.ok) {
    const l = await a.text();
    throw new Error(l || `HTTP ${a.status}`);
  }
  try {
    return await a.json();
  } catch {
    return { status: "updated" };
  }
}
async function ki(e, t, n) {
  const i = await we(e), r = Ee(t, n, i);
  return {
    active: !!r,
    title: String((r == null ? void 0 : r.title) ?? "").trim(),
    participants: Lt(r == null ? void 0 : r.participants)
  };
}
function xt(e, t) {
  if (!t)
    return null;
  const n = String((e == null ? void 0 : e.topic) ?? "").trim(), i = String((e == null ? void 0 : e.subscriberName) ?? "").trim(), r = String((e == null ? void 0 : e.userName) ?? "").trim(), o = X(t);
  if (!o || !It(n, i, t))
    return null;
  const s = Array.isArray(t.topics) ? t.topics.map((m) => String(m).trim()).filter(Boolean) : [], a = Lt(t.participants), l = [], h = /* @__PURE__ */ new Set();
  for (const m of [n, o, ...s])
    !m || h.has(m) || (h.add(m), l.push(m));
  const c = vt(t), u = At(n, t), d = Un(n, t) ? "leading" : "following", f = l.map((m) => {
    const g = !!n && (m === n || m.toLowerCase() === n.toLowerCase()), p = m === c || m.toLowerCase() === c.toLowerCase(), S = p && d === "following";
    let A = "";
    return g ? A = d === "leading" ? "You \xB7 leading" : "You" : S ? A = "Following" : d === "leading" && (A = "Following you"), {
      placeId: m,
      label: g && r ? r : m,
      isSelf: g,
      isLeading: p,
      isFollowing: S,
      statusLabel: A
    };
  });
  return {
    title: String(t.title ?? "").trim(),
    places: f,
    selfPlaceId: n,
    leadingPlaceId: c,
    selfRole: d,
    isConferenceOwner: u,
    hostTopic: o,
    sceneLeaderTopic: c,
    attendeeTopics: s,
    participants: a
  };
}
async function kn(e, t, n, i) {
  const r = await we(e), o = Ee(t, n, r);
  return xt({ topic: t, subscriberName: n, userName: i }, o);
}
var bt = Object.freeze({
  // Dialog shell
  dialogTitle: "Conferencing",
  close: "Close",
  closeAria: "Close",
  // Session (user / topic)
  sessionSection: "User / topic",
  sessionInfo: "Session info",
  participantsAria: "Participants",
  userLabel: "User:",
  changeUserName: "Change user name",
  changeUserNameTitle: "Change user name",
  userNamePlaceholder: "Enter user name",
  openAnotherUser: "Open another user",
  openAnotherUserTitle: "Open another user session to test conferencing",
  userNameSavedNote: "Saved. Reload the page to apply.",
  done: "Done",
  // Create
  createSection: "Create a conference",
  conferenceTitle: "Conference title",
  selectConferenceTitle: "Select a conference title",
  otherTitle: "Other\u2026",
  customTitle: "Custom title",
  customTitlePlaceholder: "Enter conference title",
  users: "Users",
  loadingUsers: "Loading users\u2026",
  noUsers: "No users available",
  createConference: "Create conference",
  creating: "Creating\u2026",
  // Join
  joinSection: "Join a conference",
  activeConferences: "Active conferences",
  noConferences: "No conferences available",
  selectConference: "Select a conference",
  untitled: "Untitled",
  joinConference: "Join conference",
  joining: "Joining\u2026",
  // Manage
  manageSection: "Manage conference",
  titleLabel: "Title:",
  na: "N/A",
  participants: "Participants",
  noPlaces: "No places",
  leaveConference: "Leave conference",
  endConference: "End conference",
  ending: "Ending\u2026",
  leaving: "Leaving\u2026",
  conferenceEnded: "Conference ended.",
  leftConference: "Left conference.",
  // Errors / status
  connectHubFirst: "Connect to the Slicer hub first.",
  topicRequiredHost: "Hub topic is required to host a conference.",
  topicRequiredJoin: "Hub topic is required to join a conference.",
  titleRequired: "Conference title is required.",
  selectAttendee: "Select at least one attendee topic.",
  selectConferenceToJoin: "Select a conference to join.",
  failedCreate: "Failed to create conference.",
  failedJoin: "Failed to join conference.",
  failedUpdate: "Failed to update conference.",
  // Invite
  inviteTitle: "Conference invitation",
  joinAndFollow: "Join and follow",
  joinNoFollow: "Join but do not follow",
  doNotJoin: "Do not join",
  hostFallback: "host",
  /** {{title}} — join and follow {{host}}? */
  inviteLeadWithTitle: "{{title}} \u2014 join and follow {{host}}?",
  /** Join conference and follow {{host}}? */
  inviteLeadNoTitle: "Join conference and follow {{host}}?",
  /** Join and follow {{host}} */
  joinAndFollowHost: "Join and follow {{host}}",
  // Takeover
  takeoverTitle: "Take over?",
  takeoverBody: "The host and all participants will be notified that you are taking over. You will lead scene updates for this conference.",
  cancel: "Cancel",
  takeOver: "Take over",
  // Header
  conferencing: "Conferencing",
  subscribeHubFirst: "Subscribe to the hub first",
  stopFollowing: "Stop following",
  resumeFollowing: "Resume following"
});
function ye(e, t = {}) {
  return String(e || "").replace(
    /\{\{(\w+)\}\}/g,
    (n, i) => t[i] != null ? String(t[i]) : ""
  );
}
function ee(e) {
  return !e || typeof e != "object" ? { ...bt } : { ...bt, ...e };
}
function F(e, t) {
  return `${e}-${t}`;
}
function Ht(e) {
  var i;
  if (!((i = e == null ? void 0 : e.places) != null && i.length)) return null;
  const t = e.places.find((r) => r && r.isLeading);
  if (t) return t;
  const n = String(e.leadingPlaceId || "").trim();
  return n && e.places.find(
    (r) => r && (r.placeId === n || r.placeId.toLowerCase() === n.toLowerCase())
  ) || null;
}
function _n(e, t = true) {
  return e != null && e.isFollowing ? t ? "Following" : "Skipping" : String((e == null ? void 0 : e.statusLabel) || "").trim();
}
function $t(e, t, n = "short", i = true, r = "") {
  if ((t === "leading" || t === "following" ? t : e != null && e.isLeading ? "leading" : "following") === "leading")
    return n === "full" ? "You \xB7 leading" : "Leading";
  const s = String(r || "").trim();
  if (!i)
    return s ? `Skipping ${s}` : n === "full" ? "You \xB7 skipping" : "Skipping";
  if (s) return `Following ${s}`;
  if (n === "full") {
    const a = String((e == null ? void 0 : e.statusLabel) || "").trim();
    return a || "You";
  }
  return "Following";
}
function Rn(e, t, n = true) {
  if (!e) return "";
  if (e.isSelf) {
    const i = Ht(t);
    return $t(
      e,
      t == null ? void 0 : t.selfRole,
      "short",
      n,
      i && !i.isSelf ? i.label : ""
    );
  }
  return _n(e, n);
}
function _i(e, t) {
  e.classList.remove(t, `${t}-leading`);
  const n = e.querySelector("[data-hub-conference-role]");
  n instanceof HTMLElement && (n.hidden = true, n.textContent = "");
}
function On(e, t, n, i, r) {
  const o = (t.places || []).filter((s) => s && (r || !s.isSelf));
  for (const s of o) {
    const a = document.createElement("div");
    a.className = F(n, "chip"), a.setAttribute("role", "listitem"), s.isSelf && a.classList.add(F(n, "chip-self")), s.isLeading && a.classList.add(F(n, "chip-leading")), s.isFollowing && (a.classList.add(F(n, "chip-following")), i || a.classList.add(F(n, "chip-skipping")));
    const l = Rn(
      s,
      t,
      i
    ), h = document.createElement("span");
    h.className = F(n, "chip-name"), h.textContent = s.label;
    const c = [s.label, l, s.placeId].map((u) => String(u || "").trim()).filter(Boolean).join(" \u2014 ");
    if (h.title = c, a.title = c, a.append(h), l) {
      const u = document.createElement("span");
      u.className = F(n, "chip-status"), u.textContent = l, a.append(u);
    }
    e.append(a);
  }
}
function Ri(e) {
  var p;
  const t = (e == null ? void 0 : e.view) ?? null, n = e == null ? void 0 : e.statusControl, i = (e == null ? void 0 : e.chipsHost) instanceof HTMLElement ? e.chipsHost : null, r = String((e == null ? void 0 : e.classPrefix) || "hub-conference").trim() || "hub-conference", o = String((e == null ? void 0 : e.statusClassPrefix) || F(r, "user")).trim() || F(r, "user"), s = (e == null ? void 0 : e.selfStatusMode) === "full" ? "full" : "short", a = (e == null ? void 0 : e.followHost) !== false, l = F(r, "chip-status");
  if (!(n instanceof HTMLElement))
    return;
  if (!!!((p = t == null ? void 0 : t.places) != null && p.length)) {
    _i(n, o), i && (i.replaceChildren(), i.hidden = true), W();
    return;
  }
  const c = t.places.find((S) => S && S.isSelf) || null, u = Ht(t), d = t.selfRole === "leading" || !!(c != null && c.isLeading), f = u && !u.isSelf ? String(u.label || "").trim() : "";
  n.classList.add(o), n.classList.toggle(`${o}-leading`, d);
  let m = n.querySelector("[data-hub-conference-role]");
  if (m instanceof HTMLElement || (m = document.createElement("span"), m.dataset.hubConferenceRole = "1", n.append(m)), m.className = l, m.hidden = false, m.textContent = $t(
    c,
    t.selfRole,
    s,
    a,
    f
  ), !i)
    return;
  if (i.replaceChildren(), !t.places.filter((S) => S && !S.isSelf).length) {
    i.hidden = true;
    return;
  }
  i.hidden = false, On(i, t, r, a, false);
}
var K = null;
function W() {
  K && (K.cleanup(), K = null);
}
function Oi() {
  return !!K;
}
function Mi(e) {
  var g;
  const t = e == null ? void 0 : e.anchor, n = (e == null ? void 0 : e.view) ?? null, i = (e == null ? void 0 : e.followHost) !== false, r = String((e == null ? void 0 : e.classPrefix) || "hub-conference").trim() || "hub-conference", o = (e == null ? void 0 : e.root) || (typeof document < "u" ? document.body : null);
  if (!(t instanceof HTMLElement) || !o || !((g = n == null ? void 0 : n.places) != null && g.length))
    return W(), { close: W, open: false };
  if (K) {
    const p = K.panel.__castAnchor === t;
    if (W(), p)
      return { close: W, open: false };
  }
  const s = document.createElement("div");
  s.className = F(r, "participants-popup"), s.setAttribute("role", "dialog"), s.setAttribute("aria-label", "Conference participants"), s.dataset.castParticipantsPopup = "1", s.__castAnchor = t;
  const a = document.createElement("div");
  a.className = F(r, "participants-popup-title"), a.textContent = "Participants", s.append(a);
  const l = document.createElement("div");
  l.className = F(r, "participants-popup-list"), l.setAttribute("role", "list"), On(l, n, r, i, true), s.append(l), o.appendChild(s);
  const h = () => {
    const p = t.getBoundingClientRect(), S = 6, A = Math.max(220, Math.min(320, s.offsetWidth || 260));
    let U = p.left, w = p.bottom + S;
    const N = window.innerWidth - A - 8;
    U > N && (U = Math.max(8, N)), w + s.offsetHeight > window.innerHeight - 8 && (w = Math.max(8, p.top - s.offsetHeight - S)), s.style.position = "fixed", s.style.left = `${Math.round(U)}px`, s.style.top = `${Math.round(w)}px`, s.style.zIndex = "10030", s.style.minWidth = "220px", s.style.maxWidth = "320px";
  };
  h();
  const c = (p) => {
    const S = p.target;
    S instanceof Node && (s.contains(S) || t.contains(S) || W());
  }, u = (p) => {
    p.key === "Escape" && W();
  }, d = () => h(), f = window.setTimeout(() => {
    document.addEventListener("pointerdown", c, true);
  }, 0);
  return document.addEventListener("keydown", u, true), window.addEventListener("resize", d), K = { panel: s, cleanup: () => {
    window.clearTimeout(f), document.removeEventListener("pointerdown", c, true), document.removeEventListener("keydown", u, true), window.removeEventListener("resize", d), s.remove();
  } }, requestAnimationFrame(h), { close: W, open: true };
}
function Mn(e = {}) {
  var d;
  const t = ee(e.strings), n = e.view ?? null, i = !!e.followHost, r = !!e.connected, o = !!((d = n == null ? void 0 : n.places) != null && d.length), s = (n == null ? void 0 : n.selfRole) === "leading", l = !!(n != null && n.isConferenceOwner) ? t.endConference : t.leaveConference, h = o && !s && i, c = o && !s && !i, u = r ? t.conferencing : t.subscribeHubFirst;
  return {
    active: o,
    start: {
      // Keep Conferencing available while active so User/topic + Manage stay reachable.
      hidden: false,
      disabled: !r,
      label: t.conferencing,
      title: u,
      ariaLabel: t.conferencing
    },
    leave: {
      hidden: !o,
      disabled: !r || !o,
      label: l,
      title: l,
      ariaLabel: l
    },
    stopFollowing: {
      hidden: !h,
      label: t.stopFollowing,
      title: t.stopFollowing,
      ariaLabel: t.stopFollowing
    },
    resumeFollowing: {
      hidden: !c,
      label: t.resumeFollowing,
      title: t.resumeFollowing,
      ariaLabel: t.resumeFollowing
    },
    takeOver: {
      hidden: !c,
      disabled: !r || !c,
      label: t.takeOver,
      title: t.takeOver,
      ariaLabel: t.takeOver
    }
  };
}
function pe(e, t) {
  e instanceof HTMLElement && (typeof t.hidden == "boolean" && (e.hidden = t.hidden), typeof t.disabled == "boolean" && "disabled" in e && (e.disabled = t.disabled, t.disabled || e.removeAttribute("disabled")), typeof t.label == "string" && !e.querySelector("svg") && (e.textContent = t.label), typeof t.title == "string" && (e.title = t.title), typeof t.ariaLabel == "string" && e.setAttribute("aria-label", t.ariaLabel));
}
function Pi(e = {}) {
  const t = Mn({
    view: e.view,
    followHost: e.followHost,
    connected: e.connected,
    strings: e.strings
  });
  return pe(e.startBtn, t.start), pe(e.endBtn, t.leave), pe(e.stopFollowBtn, t.stopFollowing), pe(e.resumeFollowBtn, t.resumeFollowing), pe(e.takeOverBtn, t.takeOver), t;
}
function kt(e, t, n = "hub-dialog-anchored") {
  e instanceof HTMLElement && e.classList.remove(n), t instanceof HTMLElement && (t.style.top = "", t.style.left = "", t.style.right = "", t.style.bottom = "", t.style.maxWidth = "", t.style.position = "", t.style.margin = "");
}
function Pn(e) {
  const t = e == null ? void 0 : e.overlay, n = e == null ? void 0 : e.dialog, i = e == null ? void 0 : e.anchor, r = String((e == null ? void 0 : e.anchoredClass) || "hub-dialog-anchored").trim(), o = Number.isFinite(e == null ? void 0 : e.gap) ? Number(e.gap) : 6, s = Number.isFinite(e == null ? void 0 : e.pad) ? Number(e.pad) : 8, a = (e == null ? void 0 : e.align) === "end";
  if (!(t instanceof HTMLElement) || !(n instanceof HTMLElement))
    return;
  if (!(i instanceof HTMLElement)) {
    kt(t, n, r);
    return;
  }
  const l = i.getBoundingClientRect();
  t.classList.add(r), n.style.position = "absolute", n.style.margin = "0";
  const h = Math.max(
    280,
    Math.min(n.offsetWidth || 480, window.innerWidth - s * 2)
  );
  let c = a ? l.right - h : l.left, u = l.bottom + o;
  const d = window.innerWidth - h - s;
  c > d && (c = Math.max(s, d)), c < s && (c = s);
  const f = n.offsetHeight || 280;
  u + f > window.innerHeight - s && (u = Math.max(s, l.top - f - o)), n.style.left = `${Math.round(c)}px`, n.style.top = `${Math.round(u)}px`, n.style.maxWidth = `${Math.round(h)}px`;
}
function y(e, t) {
  return `${e}-${t}`;
}
function Z(e) {
  return String(e ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function Fi(e, t) {
  const n = String(e || "").trim();
  if (!n) return false;
  const i = X(t);
  return i && (i === n || i.toLowerCase() === n.toLowerCase()) ? true : (Array.isArray(t == null ? void 0 : t.topics) ? t.topics : []).some((o) => {
    const s = String(o || "").trim();
    return s === n || s.toLowerCase() === n.toLowerCase();
  });
}
function Di(e) {
  const t = typeof (e == null ? void 0 : e.getSession) == "function" ? e.getSession : () => ({}), n = typeof (e == null ? void 0 : e.onConferenceChange) == "function" ? e.onConferenceChange : null, i = typeof (e == null ? void 0 : e.onFollowChange) == "function" ? e.onFollowChange : null, r = String((e == null ? void 0 : e.classPrefix) || "hub-conference").trim(), o = (e == null ? void 0 : e.anchorAlign) === "end" ? "end" : "start", s = (e == null ? void 0 : e.root) || (typeof document < "u" ? document.body : null);
  if (!s)
    throw new Error("createHubConferenceDialog requires a DOM root");
  let a = ee(e == null ? void 0 : e.strings), l = null, h = false, c = null;
  const u = document.createElement("div");
  u.className = y(r, "overlay"), u.hidden = true, u.innerHTML = `
    <div class="${y(r, "dialog")}" role="dialog" aria-modal="true" aria-labelledby="${y(r, "title")}" tabindex="-1">
      <div class="${y(r, "header")}">
        <h2 id="${y(r, "title")}" class="${y(r, "title")}" data-hub-conference-dialog-title></h2>
        <button type="button" class="${y(r, "close")}" data-hub-conference-close aria-label="">\xD7</button>
      </div>
      <div class="${y(r, "body")}">
        <p class="${y(r, "status")}" data-hub-conference-status role="alert" hidden></p>
        <div data-hub-conference-create>
          <div class="${y(r, "section")}" data-hub-conference-create-form>
            <div class="${y(r, "section-title")}" data-hub-conference-create-section></div>
            <div class="${y(r, "field")}">
              <label class="${y(r, "label")}" for="${y(r, "title-select")}" data-hub-conference-title-label></label>
              <select id="${y(r, "title-select")}" class="${y(r, "select")}" data-hub-conference-title-select>
              </select>
            </div>
            <div class="${y(r, "field")}" data-hub-conference-custom-group hidden>
              <label class="${y(r, "label")}" for="${y(r, "custom-title")}" data-hub-conference-custom-label></label>
              <input id="${y(r, "custom-title")}" class="${y(r, "input")}" type="text" data-hub-conference-custom-title />
            </div>
            <div class="${y(r, "field")}">
              <span class="${y(r, "label")}" data-hub-conference-users-label></span>
              <div class="${y(r, "topics")}" data-hub-conference-topics>
                <p class="${y(r, "muted")}" data-hub-conference-loading-users></p>
              </div>
            </div>
            <button type="button" class="${y(r, "primary")}" data-hub-conference-create-btn></button>
          </div>
          <div class="${y(r, "section")}" data-hub-conference-join-section>
            <div class="${y(r, "section-title")}" data-hub-conference-join-section-title></div>
            <div class="${y(r, "field")}">
              <label class="${y(r, "label")}" for="${y(r, "join-select")}" data-hub-conference-active-label></label>
              <select id="${y(r, "join-select")}" class="${y(r, "select")}" data-hub-conference-join-select>
              </select>
            </div>
            <button type="button" class="${y(r, "primary")}" data-hub-conference-join-btn disabled></button>
          </div>
        </div>
        <div data-hub-conference-manage hidden>
          <div class="${y(r, "label")}" data-hub-conference-manage-label></div>
          <div class="${y(r, "info")}" data-hub-conference-info></div>
          <button type="button" class="${y(r, "primary")}" data-hub-conference-exit-btn></button>
        </div>
      </div>
      <div class="${y(r, "section")} ${y(r, "section-session")}" data-hub-conference-session>
        <div class="${y(r, "user-row")}">
          <span class="${y(r, "label")}" data-hub-conference-user-label></span>
          <span
            class="${y(r, "user-display")}"
            data-hub-conference-user-display
            id="sessionUserNameDisplay"
          >\u2014</span>
          <button
            type="button"
            id="sessionChangeUserBtn"
            class="${y(r, "secondary")}"
            data-hub-conference-change-user
          ></button>
        </div>
        <p id="sessionChangeUserNote" class="${y(r, "muted")}" data-hub-conference-user-note hidden></p>
      </div>
    </div>
    <div
      class="${y(r, "change-user-overlay")}"
      data-hub-conference-change-user-overlay
      hidden
    >
      <div
        class="${y(r, "change-user-dialog")}"
        role="dialog"
        aria-modal="true"
        aria-labelledby="${y(r, "change-user-title")}"
        tabindex="-1"
      >
        <div class="${y(r, "change-user-header")}">
          <h3
            id="${y(r, "change-user-title")}"
            class="${y(r, "change-user-title")}"
            data-hub-conference-change-user-title
          ></h3>
          <button
            type="button"
            class="${y(r, "close")}"
            data-hub-conference-change-user-close
            aria-label=""
          >\xD7</button>
        </div>
        <div class="${y(r, "field")}">
          <label
            class="${y(r, "label")}"
            for="sessionUserNameInput"
            data-hub-conference-user-label
          ></label>
          <input
            type="text"
            id="sessionUserNameInput"
            class="${y(r, "input")} ${y(r, "user-input")}"
            autocomplete="username"
            data-hub-conference-user-input
          />
        </div>
        <div class="${y(r, "change-user-actions")}">
          <button
            type="button"
            id="openConferenceTestWorklistBtn"
            class="${y(r, "secondary")} ${y(r, "open-another-user")}"
            data-hub-conference-open-another-user
          ></button>
          <button
            type="button"
            class="${y(r, "primary")}"
            data-hub-conference-change-user-done
          ></button>
        </div>
      </div>
    </div>
  `, s.appendChild(u);
  const d = u.querySelector(`.${y(r, "dialog")}`), f = u.querySelector("[data-hub-conference-status]"), m = u.querySelector(
    "[data-hub-conference-open-another-user]"
  ), g = u.querySelector(
    "[data-hub-conference-user-display]"
  ), p = u.querySelector(
    "[data-hub-conference-change-user]"
  ), S = u.querySelector(
    "[data-hub-conference-change-user-overlay]"
  ), A = u.querySelector(
    "[data-hub-conference-change-user-done]"
  ), U = u.querySelector(
    "[data-hub-conference-change-user-close]"
  ), w = u.querySelector("#sessionUserNameInput"), N = u.querySelector("#sessionChangeUserNote"), E = typeof (e == null ? void 0 : e.onUserNameChange) == "function" ? e.onUserNameChange : null, x = u.querySelector("[data-hub-conference-create]"), H = u.querySelector(
    "[data-hub-conference-create-form]"
  ), Wt = u.querySelector("[data-hub-conference-manage]"), J = u.querySelector("[data-hub-conference-join-section]"), $ = u.querySelector("[data-hub-conference-title-select]"), _e = u.querySelector("[data-hub-conference-custom-group]"), Y = u.querySelector("[data-hub-conference-custom-title]"), ne = u.querySelector("[data-hub-conference-topics]"), I = u.querySelector("[data-hub-conference-join-select]"), zt = u.querySelector("[data-hub-conference-info]"), D = u.querySelector("[data-hub-conference-create-btn]"), _ = u.querySelector("[data-hub-conference-join-btn]"), M = u.querySelector("[data-hub-conference-exit-btn]"), he = u.querySelector("[data-hub-conference-close]");
  function Vt(b) {
    !x || !H || !J || (b ? x.insertBefore(J, H) : x.insertBefore(H, J));
  }
  function dr() {
    if (!($ instanceof HTMLSelectElement)) return;
    const b = String($.value || "");
    $.replaceChildren();
    const C = document.createElement("option");
    C.value = "", C.textContent = a.selectConferenceTitle, $.appendChild(C);
    for (const L of In) {
      const v = document.createElement("option");
      v.value = L, v.textContent = L, L === "Tumor Board" && (v.selected = true), $.appendChild(v);
    }
    const T = document.createElement("option");
    T.value = "other", T.textContent = a.otherTitle, $.appendChild(T), b && [...$.options].some((L) => L.value === b) && ($.value = b);
  }
  function Gt() {
    const b = (C, T) => {
      const L = u.querySelector(C);
      L && (L.textContent = T);
    };
    b("[data-hub-conference-dialog-title]", a.dialogTitle), b("[data-hub-conference-create-section]", a.createSection), b("[data-hub-conference-title-label]", a.conferenceTitle), b("[data-hub-conference-custom-label]", a.customTitle), b("[data-hub-conference-users-label]", a.users), b("[data-hub-conference-loading-users]", a.loadingUsers), b("[data-hub-conference-join-section-title]", a.joinSection), b("[data-hub-conference-active-label]", a.activeConferences), b("[data-hub-conference-manage-label]", a.manageSection), D && (D.textContent = a.createConference), _ && (_.textContent = a.joinConference), M && (M.textContent = l != null && l.isConferenceOwner ? a.endConference : a.leaveConference), u.querySelectorAll("[data-hub-conference-user-label]").forEach((C) => {
      C instanceof HTMLElement && (C.textContent = a.userLabel);
    }), p instanceof HTMLElement && (p.textContent = a.changeUserName, p.title = a.changeUserNameTitle), b("[data-hub-conference-change-user-title]", a.changeUserNameTitle), m instanceof HTMLElement && (m.textContent = a.openAnotherUser, m.title = a.openAnotherUserTitle), A instanceof HTMLElement && (A.textContent = a.done), N instanceof HTMLElement && (N.textContent = a.userNameSavedNote), w instanceof HTMLInputElement && (w.placeholder = a.userNamePlaceholder), Y instanceof HTMLInputElement && (Y.placeholder = a.customTitlePlaceholder), he instanceof HTMLElement && he.setAttribute("aria-label", a.closeAria), U instanceof HTMLElement && U.setAttribute("aria-label", a.closeAria), dr(), Xt();
  }
  function br() {
  }
  function Xt() {
    const b = String(ie().userName || "").trim() || "\u2014";
    g instanceof HTMLElement && (g.textContent = b);
  }
  function hr() {
    const b = new URL(window.location.href);
    b.searchParams.set("freshUser", "1"), b.searchParams.delete("topic"), window.open(b.href, "_blank", "noopener,noreferrer");
  }
  function mr() {
    S instanceof HTMLElement && (w instanceof HTMLInputElement && (w.value = String(ie().userName || "").trim()), N instanceof HTMLElement && (N.hidden = true), S.hidden = false, w instanceof HTMLInputElement && (w.focus(), w.select()));
  }
  function re() {
    S instanceof HTMLElement && (S.hidden = true);
  }
  function Jt() {
    if (!(w instanceof HTMLInputElement)) return;
    const b = w.value.trim();
    E && E(b), g instanceof HTMLElement && (g.textContent = b || "\u2014"), N instanceof HTMLElement && (N.hidden = false, N.textContent = a.userNameSavedNote), re();
  }
  Gt();
  function R(b, C) {
    if (!f)
      return;
    const T = String(C || "").trim();
    if (f.classList.remove(
      y(r, "status-success"),
      y(r, "status-error")
    ), !T) {
      f.hidden = true, f.textContent = "";
      return;
    }
    f.hidden = false, f.textContent = T, b === "success" ? f.classList.add(y(r, "status-success")) : b === "error" && f.classList.add(y(r, "status-error"));
  }
  function it(b) {
    l = b, n && n(b);
  }
  function ie() {
    const b = t() || {};
    return {
      hubEndpoint: String(b.hubEndpoint ?? "").trim(),
      topic: String(b.topic ?? "").trim(),
      subscriberName: String(b.subscriberName ?? "").trim(),
      userName: String(b.userName ?? "").trim(),
      connected: !!b.connected
    };
  }
  function gr() {
    const b = String(($ == null ? void 0 : $.value) || "").trim();
    return b === "other" ? String((Y == null ? void 0 : Y.value) || "").trim() : b;
  }
  function pr() {
    return ne ? Array.from(
      ne.querySelectorAll(
        'input[type="checkbox"][data-hub-conference-topic]:checked'
      )
    ).map((b) => String(b.value || "").trim()).filter(Boolean) : [];
  }
  function yr(b) {
    var T;
    const C = String(b || "").trim();
    if (!C) return "";
    try {
      const v = (T = new Intl.DateTimeFormat("en", {
        timeZone: C,
        timeZoneName: "shortOffset"
      }).formatToParts(/* @__PURE__ */ new Date()).find((k) => k.type === "timeZoneName")) == null ? void 0 : T.value;
      return v ? `${C} (${v})` : C;
    } catch {
      return C;
    }
  }
  function Yt(b) {
    if (ne) {
      if (ne.replaceChildren(), !b.length) {
        const C = document.createElement("p");
        C.className = y(r, "muted"), C.textContent = a.noUsers, ne.appendChild(C);
        return;
      }
      for (const C of b) {
        const T = typeof C == "string" ? String(C).trim() : String((C == null ? void 0 : C.topic) ?? "").trim();
        if (!T) continue;
        const L = typeof C == "string" ? "" : String((C == null ? void 0 : C.timezone) ?? "").trim(), v = document.createElement("label");
        v.className = y(r, "topic-row");
        const k = document.createElement("input");
        k.type = "checkbox", k.dataset.hubConferenceTopic = "1", k.value = T, k.checked = true;
        const q = document.createElement("span");
        q.className = y(r, "topic-label"), q.textContent = T, v.appendChild(k), v.appendChild(q);
        const ge = yr(L);
        if (ge) {
          const G = document.createElement("span");
          G.className = y(r, "topic-timezone"), G.textContent = ge, v.appendChild(G);
        }
        ne.appendChild(v);
      }
    }
  }
  function Zt(b, C) {
    if (!I || !_) return;
    const T = (Array.isArray(b) ? b : []).filter(
      (q) => q && typeof q == "object" && !Fi(
        C,
        /** @type {Record<string, unknown>} */
        q
      )
    ), L = String(I.value || "");
    if (I.replaceChildren(), !T.length) {
      I.replaceChildren(), I.disabled = true, _.disabled = true, J instanceof HTMLElement && (J.hidden = true), Vt(false);
      return;
    }
    J instanceof HTMLElement && (J.hidden = false);
    const v = document.createElement("option");
    v.value = "", v.textContent = a.selectConference, I.appendChild(v);
    const k = [];
    for (const q of T) {
      const ge = (
        /** @type {Record<string, unknown>} */
        q
      ), G = X(ge);
      if (!G) continue;
      k.push(G);
      const vr = String(ge.title || "").trim() || a.untitled, at = document.createElement("option");
      at.value = G, at.textContent = `${vr} \u2014 ${G}`, I.appendChild(at);
    }
    I.disabled = false, L && k.includes(L) ? I.value = L : k.length && (I.value = k[0]), _.disabled = !String(I.value || "").trim(), Vt(true);
  }
  function ot(b) {
    if (x && (x.hidden = !!b), Wt && (Wt.hidden = !b), b && zt) {
      const C = b.places.length ? `<div class="${y(r, "places")}">${b.places.map((T) => {
        const L = [y(r, "place")];
        T.isSelf && L.push(y(r, "place-self")), T.isLeading && L.push(y(r, "place-leading")), T.isFollowing && L.push(y(r, "place-following"));
        const v = T.statusLabel ? `<span class="${y(r, "place-status")}">${Z(
          T.statusLabel
        )}</span>` : "";
        return `<span class="${L.join(
          " "
        )}" title="${Z(T.placeId)}"><span class="${y(
          r,
          "place-name"
        )}">${Z(T.label)}</span>${v}</span>`;
      }).join("")}</div>` : `<div class="${y(r, "muted")}">${Z(a.noPlaces)}</div>`;
      zt.innerHTML = `<div><strong>${Z(a.titleLabel)}</strong> ${Z(
        b.title || a.na
      )}</div><div class="${y(r, "places-label")}">${Z(
        a.participants
      )}</div>${C}`;
    }
    M && (M.textContent = b != null && b.isConferenceOwner ? a.endConference : a.leaveConference);
  }
  async function me() {
    const b = ie();
    if (!b.hubEndpoint || !b.connected)
      return Yt([]), Zt([], b.topic), ot(null), it(null), null;
    const [C, T] = await Promise.all([
      xn(b.hubEndpoint),
      we(b.hubEndpoint)
    ]), L = Ee(
      b.topic,
      b.subscriberName,
      T
    ), v = xt(b, L);
    return L || (Yt(C), Zt(T, b.topic)), ot(v), it(v), v;
  }
  async function Sr() {
    if (h)
      return;
    const b = ie(), C = gr(), T = pr();
    if (!b.connected) {
      R("error", a.connectHubFirst);
      return;
    }
    if (!b.topic) {
      R("error", a.topicRequiredHost);
      return;
    }
    if (!C) {
      R("error", a.titleRequired);
      return;
    }
    if (!T.length) {
      R("error", a.selectAttendee);
      return;
    }
    h = true, D && (D.disabled = true, D.textContent = a.creating), R("", "");
    try {
      await Hn(
        b.hubEndpoint,
        b.topic,
        C,
        T,
        b.userName
      ), await me(), V();
    } catch (L) {
      R(
        "error",
        L instanceof Error ? L.message : a.failedCreate
      );
    } finally {
      h = false, D && (D.disabled = false, D.textContent = a.createConference);
    }
  }
  async function Cr() {
    if (h) return;
    const b = ie(), C = String((I == null ? void 0 : I.value) || "").trim();
    if (!b.connected) {
      R("error", a.connectHubFirst);
      return;
    }
    if (!b.topic) {
      R("error", a.topicRequiredJoin);
      return;
    }
    if (!C) {
      R("error", a.selectConferenceToJoin);
      return;
    }
    h = true, _ && (_.disabled = true, _.textContent = a.joining), R("", "");
    try {
      await $n(
        b.hubEndpoint,
        C,
        b.topic,
        b.userName
      ), i == null || i(true), await me(), V();
    } catch (T) {
      R(
        "error",
        T instanceof Error ? T.message : a.failedJoin
      );
    } finally {
      h = false, _ && (_.textContent = a.joinConference, _.disabled = !String((I == null ? void 0 : I.value) || "").trim());
    }
  }
  async function Er() {
    if (h)
      return;
    const b = ie(), C = await we(b.hubEndpoint), T = Ee(
      b.topic,
      b.subscriberName,
      C
    );
    if (!T) {
      await me();
      return;
    }
    const L = X(T), v = At(b.topic, T);
    h = true, M && (M.disabled = true, M.textContent = v ? a.ending : a.leaving), R("", "");
    try {
      await Ut(
        b.hubEndpoint,
        L,
        v ? void 0 : b.topic
      ), it(null), R(
        "success",
        v ? a.conferenceEnded : a.leftConference
      ), await new Promise((k) => {
        setTimeout(k, An);
      }), V();
    } catch (k) {
      R(
        "error",
        k instanceof Error ? k.message : a.failedUpdate
      );
    } finally {
      h = false, M && (M.disabled = false, M.textContent = v ? a.endConference : a.leaveConference);
    }
  }
  function wr() {
    kt(
      u,
      d instanceof HTMLElement ? d : null,
      y(r, "overlay-anchored")
    );
  }
  function st(b) {
    Pn({
      overlay: u,
      dialog: d instanceof HTMLElement ? d : null,
      anchor: b,
      anchoredClass: y(r, "overlay-anchored"),
      align: o
    });
  }
  function Tr(b) {
    R("", ""), $ && ($.value = "Tumor Board"), Y && (Y.value = ""), _e && (_e.hidden = true), re(), N instanceof HTMLElement && (N.hidden = true), Xt(), c = b instanceof HTMLElement ? b : null, u.hidden = false, st(c), d instanceof HTMLElement && d.focus({ preventScroll: true }), me().then(() => st(c)).catch(() => {
    });
  }
  function V() {
    re(), u.hidden = true, R("", ""), wr(), c = null;
  }
  function Nr() {
    V(), u.remove();
  }
  function Lr(b) {
    a = ee(b), Gt(), l && ot(l);
  }
  return $ == null || $.addEventListener("change", () => {
    _e && (_e.hidden = String($.value || "") !== "other");
  }), I == null || I.addEventListener("change", () => {
    _ && (_.disabled = !String(I.value || "").trim() || h);
  }), D == null || D.addEventListener("click", () => {
    Sr().catch(() => {
    });
  }), _ == null || _.addEventListener("click", () => {
    Cr().catch(() => {
    });
  }), M == null || M.addEventListener("click", () => {
    Er().catch(() => {
    });
  }), he == null || he.addEventListener("click", (b) => {
    b.stopPropagation(), V();
  }), p == null || p.addEventListener("click", (b) => {
    b.stopPropagation(), mr();
  }), A == null || A.addEventListener("click", (b) => {
    b.stopPropagation(), Jt();
  }), U == null || U.addEventListener("click", (b) => {
    b.stopPropagation(), re();
  }), m == null || m.addEventListener("click", (b) => {
    b.stopPropagation(), hr(), V();
  }), w == null || w.addEventListener("keydown", (b) => {
    b.key === "Enter" ? (b.preventDefault(), Jt()) : b.key === "Escape" && (b.preventDefault(), re());
  }), S == null || S.addEventListener("click", (b) => {
    b.target === S && re();
  }), u.addEventListener("click", (b) => {
    b.target === u && V();
  }), window.addEventListener("resize", () => {
    u.hidden || st(c);
  }), {
    open: Tr,
    close: V,
    destroy: Nr,
    refresh: me,
    setStrings: Lr,
    syncSessionControl: br,
    getSessionControl: () => ({
      button: null,
      label: null
    }),
    getView: () => l,
    get element() {
      return u;
    }
  };
}
var ht = "ID";
function Bi(e = {}) {
  const t = Array.isArray(e.selfActors) ? e.selfActors.map((o) => String(o || "").trim()).filter(Boolean) : [], n = Array.isArray(e.connectedActors) ? e.connectedActors.map((o) => String(o || "").trim()).filter(Boolean) : [];
  return t.some(
    (o) => o === ht || o.toUpperCase() === "ID"
  ) ? true : !n.some(
    (o) => o === ht || o.toUpperCase() === "ID"
  );
}
function Fn(e) {
  const t = e && typeof e == "object" ? (
    /** @type {{ event?: Record<string, unknown> }} */
    e.event
  ) : null;
  if (!t || typeof t != "object")
    return null;
  const n = String(t["hub.event"] || "").trim().toLowerCase();
  if (n !== "conference-start" && n !== "conference-end" && n !== "conference-lead")
    return null;
  const i = t.context && typeof t.context == "object" ? (
    /** @type {Record<string, unknown>} */
    t.context
  ) : {}, r = String(
    i.hostTopic || t["hub.topic"] || ""
  ).trim(), o = String(i.title || "").trim(), s = String(i.hostUserName || "").trim(), a = String(i.sceneLeaderTopic || "").trim(), l = String(
    i.leaderUserName || i.leaderLabel || ""
  ).trim(), h = l || a || s || r || "host", c = String(i.joinedTopic || "").trim();
  return {
    hubEvent: n,
    title: o,
    hostTopic: r,
    hostUserName: s,
    hostLabel: s || r || "host",
    sceneLeaderTopic: a,
    leaderUserName: l,
    leaderLabel: h,
    joinedTopic: c
  };
}
function O(e, t) {
  return `${e}-${t}`;
}
function Dn(e = {}) {
  const t = typeof e.onDecision == "function" ? e.onDecision : null, n = String(
    e.classPrefix || "hub-conference-invite"
  ).trim(), i = e.root || (typeof document < "u" ? document.body : null);
  if (!i)
    throw new Error("createHubConferenceInviteDialog requires a DOM root");
  let r = ee(e.strings), o = null, s = false;
  const a = document.createElement("div");
  a.className = O(n, "overlay"), a.hidden = true, a.innerHTML = `
    <div class="${O(n, "dialog")}" role="dialog" aria-modal="true" aria-labelledby="${O(n, "title")}" tabindex="-1">
      <div class="${O(n, "header")}">
        <h2 id="${O(n, "title")}" class="${O(n, "title")}" data-hub-invite-title></h2>
        <button type="button" class="${O(n, "close")}" data-hub-invite-close aria-label="">\xD7</button>
      </div>
      <div class="${O(n, "body")}">
        <p class="${O(n, "lead")}" data-hub-invite-lead></p>
        <p class="${O(n, "status")}" data-hub-invite-status role="alert" hidden></p>
        <div class="${O(n, "actions")}">
          <button type="button" class="${O(n, "primary")}" data-hub-invite-follow></button>
          <button type="button" class="${O(n, "secondary")}" data-hub-invite-join></button>
          <button type="button" class="${O(n, "secondary")}" data-hub-invite-decline></button>
        </div>
      </div>
    </div>
  `, i.appendChild(a);
  const l = a.querySelector(`.${O(n, "dialog")}`), h = a.querySelector("[data-hub-invite-title]"), c = a.querySelector("[data-hub-invite-lead]"), u = a.querySelector("[data-hub-invite-status]"), d = a.querySelector("[data-hub-invite-follow]"), f = a.querySelector("[data-hub-invite-join]"), m = a.querySelector("[data-hub-invite-decline]"), g = a.querySelector("[data-hub-invite-close]");
  function p() {
    h && (h.textContent = r.inviteTitle), f && (f.textContent = r.joinNoFollow), m && (m.textContent = r.doNotJoin), g instanceof HTMLElement && g.setAttribute("aria-label", r.closeAria), d && o ? d.textContent = ye(
      r.joinAndFollowHost,
      { host: o.hostLabel || r.hostFallback }
    ) : d && (d.textContent = r.joinAndFollow);
  }
  p();
  function S(E, x) {
    if (!(u instanceof HTMLElement)) return;
    const H = String(x || "").trim();
    u.hidden = !H, u.textContent = H, u.classList.toggle(O(n, "status-error"), E === "error");
  }
  function A(E) {
    s = E;
    for (const x of [d, f, m])
      x instanceof HTMLButtonElement && (x.disabled = E);
  }
  function U() {
    a.hidden = true, o = null, S("", ""), A(false);
  }
  function w(E) {
    const x = String(
      (E == null ? void 0 : E.hostLabel) || (E == null ? void 0 : E.hostUserName) || (E == null ? void 0 : E.hostTopic) || r.hostFallback
    ).trim() || r.hostFallback, H = String((E == null ? void 0 : E.title) || "").trim();
    o = {
      hubEvent: "conference-start",
      title: H,
      hostTopic: String((E == null ? void 0 : E.hostTopic) || "").trim(),
      hostUserName: String((E == null ? void 0 : E.hostUserName) || "").trim(),
      hostLabel: x
    }, c instanceof HTMLElement && (c.textContent = H ? ye(r.inviteLeadWithTitle, {
      title: H,
      host: x
    }) : ye(r.inviteLeadNoTitle, {
      host: x
    })), d instanceof HTMLElement && (d.textContent = ye(
      r.joinAndFollowHost,
      { host: x }
    )), S("", ""), a.hidden = false, l instanceof HTMLElement && l.focus({ preventScroll: true });
  }
  async function N(E) {
    if (s || !o) return;
    const x = o;
    A(true);
    try {
      t && await t(E, x), U();
    } catch (H) {
      S(
        "error",
        H instanceof Error ? H.message : r.failedUpdate
      ), A(false);
    }
  }
  return d == null || d.addEventListener("click", () => {
    N("join-follow");
  }), f == null || f.addEventListener("click", () => {
    N("join");
  }), m == null || m.addEventListener("click", () => {
    N("decline");
  }), a.querySelectorAll("[data-hub-invite-close]").forEach((E) => {
    E.addEventListener("click", () => U());
  }), a.addEventListener("click", (E) => {
    E.target === a && U();
  }), {
    open: w,
    close: U,
    /**
     * @param {Partial<import('./conferenceStrings.js').CastConferenceStrings> | null | undefined} next
     */
    setStrings(E) {
      r = ee(E), p(), o && !a.hidden && w(o);
    },
    destroy() {
      U(), a.remove();
    },
    isOpen() {
      return !a.hidden;
    },
    getInvite() {
      return o;
    },
    element: a
  };
}
function ji(e) {
  const t = typeof (e == null ? void 0 : e.getSession) == "function" ? e.getSession : () => ({}), n = typeof (e == null ? void 0 : e.shouldShow) == "function" ? e.shouldShow : () => true, i = typeof (e == null ? void 0 : e.onConferenceChange) == "function" ? e.onConferenceChange : null, r = typeof (e == null ? void 0 : e.onFollowChange) == "function" ? e.onFollowChange : null, o = typeof (e == null ? void 0 : e.onLeadChange) == "function" ? e.onLeadChange : null, s = (e == null ? void 0 : e.dialog) || Dn({
    root: e == null ? void 0 : e.root,
    classPrefix: e == null ? void 0 : e.classPrefix,
    strings: e == null ? void 0 : e.strings,
    onDecision: async (c, u) => {
      await l(c, u);
    }
  });
  async function a() {
    const c = t(), u = await kn(
      c.hubEndpoint || "",
      c.topic || "",
      c.subscriberName || "",
      c.userName || ""
    );
    return i == null || i(u), u;
  }
  async function l(c, u) {
    const d = t(), f = String((u == null ? void 0 : u.hostTopic) || "").trim();
    if (!f)
      throw new Error("Missing conference host topic.");
    if (c === "decline") {
      await Ut(
        d.hubEndpoint || "",
        f,
        d.topic || ""
      ), r == null || r(false), i == null || i(null);
      return;
    }
    const m = c === "join-follow";
    r == null || r(m, m ? u : void 0), await a();
  }
  function h(c) {
    const u = Fn(c);
    if (!u) return false;
    const d = t(), f = String(d.topic || "").trim();
    if (u.hubEvent === "conference-lead") {
      const g = String(u.sceneLeaderTopic || "").trim(), p = !!f && !!g && (f === g || f.toLowerCase() === g.toLowerCase());
      return r == null || r(!p, u), a().then(() => {
        o == null || o(u);
      }), true;
    }
    if (u.hubEvent === "conference-end") {
      const g = (
        /** @type {{ context?: Record<string, unknown> }} */
        c.event || {}
      ), p = g.context && typeof g.context == "object" ? g.context : {}, S = String(p.leaveTopic || "").trim();
      return !S || S === f ? (s.close(), r == null || r(false), i == null || i(null)) : a(), true;
    }
    if (f && u.hostTopic && (f === u.hostTopic || f.toLowerCase() === u.hostTopic.toLowerCase()))
      return r == null || r(false), a(), true;
    const m = String(u.joinedTopic || "").trim();
    return f && m && (f === m || f.toLowerCase() === m.toLowerCase()) ? (r == null || r(true, u), a(), true) : n() ? (s.open(u), a(), true) : (a(), true);
  }
  return {
    handleHubMessage: h,
    open: (c) => s.open(c),
    close: () => s.close(),
    destroy: () => s.destroy(),
    refresh: a,
    /**
     * @param {Partial<import('./conferenceStrings.js').CastConferenceStrings> | null | undefined} next
     */
    setStrings(c) {
      var u;
      (u = s.setStrings) == null || u.call(s, c);
    },
    dialog: s
  };
}
function qi(e = {}) {
  return new Promise((t) => {
    if (typeof document > "u") {
      t(false);
      return;
    }
    const n = ee(e.strings), i = String(
      e.classPrefix || "hub-conference-takeover"
    ).trim(), r = e.root || document.body, o = document.createElement("div");
    o.className = `${i}-overlay`, o.setAttribute("role", "presentation"), o.style.cssText = [
      "position:fixed",
      "inset:0",
      "z-index:10040",
      "display:flex",
      "align-items:center",
      "justify-content:center",
      "background:rgba(0,0,0,.45)",
      "padding:16px"
    ].join(";");
    const s = document.createElement("div");
    s.className = `${i}-dialog`, s.setAttribute("role", "dialog"), s.setAttribute("aria-modal", "true"), s.setAttribute("aria-labelledby", `${i}-title`), s.tabIndex = -1, s.style.cssText = [
      "width:min(420px,100%)",
      "background:#121820",
      "color:#e8eef8",
      "border:1px solid #33507e",
      "border-radius:10px",
      "box-shadow:0 16px 40px rgba(0,0,0,.4)",
      "padding:18px 18px 14px",
      "font:14px/1.45 -apple-system,system-ui,sans-serif"
    ].join(";"), s.innerHTML = `
      <h2 id="${i}-title" style="margin:0 0 10px;font:700 16px/1.3 -apple-system,system-ui,sans-serif;"></h2>
      <p data-hub-takeover-body style="margin:0 0 16px;opacity:.92;"></p>
      <div style="display:flex;gap:8px;justify-content:flex-end;flex-wrap:wrap;">
        <button type="button" data-hub-takeover-cancel style="padding:7px 12px;border-radius:6px;border:1px solid #4a628a;background:#1a2332;color:#e8eef8;cursor:pointer;font:600 13px -apple-system,system-ui,sans-serif;"></button>
        <button type="button" data-hub-takeover-confirm style="padding:7px 12px;border-radius:6px;border:1px solid #2b6cb0;background:#2b6cb0;color:#fff;cursor:pointer;font:600 13px -apple-system,system-ui,sans-serif;"></button>
      </div>
    `;
    const a = s.querySelector(`#${i}-title`), l = s.querySelector("[data-hub-takeover-body]"), h = s.querySelector("[data-hub-takeover-cancel]"), c = s.querySelector("[data-hub-takeover-confirm]");
    a && (a.textContent = n.takeoverTitle), l && (l.textContent = n.takeoverBody), h && (h.textContent = n.cancel), c && (c.textContent = n.takeOver), o.appendChild(s), r.appendChild(o);
    let u = false;
    const d = (f) => {
      u || (u = true, o.remove(), t(!!f));
    };
    o.addEventListener("click", (f) => {
      f.target === o && d(false);
    }), h == null || h.addEventListener("click", () => d(false)), c == null || c.addEventListener("click", () => d(true)), s.addEventListener("keydown", (f) => {
      f.key === "Escape" && (f.preventDefault(), d(false));
    }), s.focus({ preventScroll: true });
  });
}
function Wi(e, t = {}) {
  var l;
  const n = String(e || "").trim();
  if (!n || typeof document > "u")
    return;
  const i = Math.max(1e3, Number(t.ms) || 4e3), r = String(t.classPrefix || "hub-info-toast").trim(), o = t.root || (typeof document < "u" ? document.body : null);
  if (!o)
    return;
  const s = (l = o.querySelector) == null ? void 0 : l.call(o, `.${r}`);
  s && s.remove();
  const a = document.createElement("div");
  a.className = r, a.setAttribute("role", "status"), a.textContent = n, a.style.cssText = [
    "position:fixed",
    "left:50%",
    "bottom:28px",
    "transform:translateX(-50%)",
    "z-index:10050",
    "max-width:min(420px,calc(100vw - 32px))",
    "padding:10px 16px",
    "border-radius:8px",
    "background:#1a2332",
    "color:#e8eef8",
    "font:600 13px/1.35 -apple-system,system-ui,sans-serif",
    "box-shadow:0 8px 28px rgba(0,0,0,.35)",
    "pointer-events:none"
  ].join(";"), o.appendChild(a), window.setTimeout(() => {
    a.remove();
  }, i);
}
var He = "http://fhircast.hl7.org/StructureDefinition/fhircast-imaging-study-open";
var $e = "urn:dicom:uid";
var _t = "urn:cast:nifti-url";
var Rt = "urn:cast:nifti-filename";
var je = "urn:cast:volview-sample-id";
var fe = "urn:cast:worklist-sample-id";
var qe = "urn:cast:ohif-mode";
var de = "urn:cast:open-mode";
var Te = "dicomweb";
var We = "dicom-url";
var ce = "files";
var ze = "idc";
var Ve = "local-dicom";
var Ge = "idc";
var Xe = "idc-source-bucket";
var Ot = "urn:cast:dicomweb-root";
function B(e) {
  return typeof e != "string" ? "" : e.trim().replace(/^urn:oid:/i, "");
}
function sn(e) {
  return typeof e == "string" ? e.trim().toLowerCase() : "";
}
function le(e, t) {
  if (!Array.isArray(e))
    return null;
  const n = String(t || "").trim().toLowerCase(), i = e.find(
    (r) => r && typeof r == "object" && typeof r.key == "string" && r.key.trim().toLowerCase() === n
  );
  return !(i != null && i.resource) || typeof i.resource != "object" ? null : i.resource;
}
function j(e, t) {
  const n = le(e, "study");
  if (!n)
    return "";
  const i = Array.isArray(n.identifier) ? n.identifier : [], r = sn(t), o = i.find(
    (s) => sn(s == null ? void 0 : s.system) === r
  );
  return typeof (o == null ? void 0 : o.value) == "string" ? o.value.trim() : "";
}
function se(e) {
  const t = le(e, "study");
  return t ? B(t.uid) || B(j(e, $e)) : "";
}
function oe(e) {
  const t = le(e, "series");
  if (t)
    return B(t.uid);
  const n = le(e, "study"), r = (Array.isArray(n == null ? void 0 : n.series) ? n.series : [])[0];
  return !r || typeof r != "object" ? "" : B(r.uid);
}
function mt(e) {
  const t = j(e, Ot);
  if (!t)
    return "";
  try {
    const n = new URL(t);
    if (n.protocol === "http:" || n.protocol === "https:")
      return n.href.replace(/\/$/, "");
  } catch {
    return "";
  }
  return "";
}
function Ne(e) {
  const t = j(e, _t);
  if (!t)
    return "";
  try {
    const n = new URL(t);
    if (n.protocol === "http:" || n.protocol === "https:")
      return n.href;
  } catch {
    return "";
  }
  return "";
}
function Mt(e) {
  return j(e, Rt);
}
function Bn(e) {
  return j(e, qe);
}
function jn(e) {
  return j(e, je) || j(e, fe);
}
function zi(e) {
  return typeof e.url == "string" ? e.url.trim() : typeof e.uri == "string" ? e.uri.trim() : "";
}
function Vi(e) {
  return typeof e.fileName == "string" ? e.fileName.trim() : typeof e.filename == "string" ? e.filename.trim() : "";
}
var Gi = /* @__PURE__ */ new Set(["http:", "https:", "s3:", "gs:"]);
function Je(e) {
  if (!e || typeof e != "object")
    return null;
  const t = zi(e);
  if (!t)
    return null;
  try {
    const s = new URL(t);
    if (!Gi.has(s.protocol))
      return null;
  } catch {
    return null;
  }
  const n = Vi(e), i = typeof e.mimeType == "string" ? e.mimeType.trim() : "", r = typeof e.role == "string" ? e.role.trim() : "", o = typeof e.label == "string" ? e.label.trim() : "";
  return {
    url: t,
    fileName: n,
    mimeType: i,
    role: r,
    label: o
  };
}
function ue(e) {
  const t = le(e, "files"), i = (Array.isArray(t == null ? void 0 : t.files) ? t.files : []).map((o) => Je(o)).filter((o) => o !== null);
  if (i.length > 0)
    return i;
  const r = Ne(e);
  return r ? [
    {
      url: r,
      fileName: Mt(e),
      mimeType: "",
      role: "",
      label: ""
    }
  ] : [];
}
function gt(e) {
  return B(j(e, Ge));
}
function pt(e) {
  return j(
    e,
    Xe
  ).toLowerCase() === "gcs" ? "gcs" : "aws";
}
function qn(e) {
  const t = j(e, de);
  return t === Te || t === We || t === ce || t === ze || t === Ve ? t : ue(e).length > 0 || Ne(e) ? ce : se(e) && !Ne(e) ? Te : "";
}
function Wn({
  id: e,
  files: t,
  patientReference: n,
  openMode: i,
  includeLegacyNiftiIdentifiers: r = false
}) {
  const o = String(e || "").trim() || "study", s = (Array.isArray(t) ? t : []).map((c) => Je(c)).filter((c) => c !== null), a = [
    { system: de, value: i },
    {
      system: fe,
      value: o
    },
    {
      system: je,
      value: o
    }
  ];
  if (r && s.length === 1) {
    const [c] = s;
    a.push({
      system: _t,
      value: c.url
    }), c.fileName && a.push({
      system: Rt,
      value: c.fileName
    });
  }
  const l = {
    resourceType: "ImagingStudy",
    id: o,
    meta: {
      profile: [He]
    },
    identifier: a,
    status: "available"
  };
  n && (l.subject = { reference: String(n).trim() });
  const h = [
    {
      key: "study",
      resource: l
    }
  ];
  return s.length > 0 && h.push({
    key: "files",
    resource: {
      files: s.map((c) => ({
        url: c.url,
        fileName: c.fileName || void 0,
        mimeType: c.mimeType || void 0,
        role: c.role || void 0,
        label: c.label || void 0
      }))
    }
  }), h;
}
function zn({
  id: e,
  files: t,
  patientReference: n,
  includeLegacyNiftiIdentifiers: i = true
}) {
  return Wn({
    id: e,
    files: t,
    patientReference: n,
    openMode: ce,
    includeLegacyNiftiIdentifiers: i
  });
}
function Xi({
  id: e,
  files: t,
  patientReference: n
}) {
  return Wn({
    id: e,
    files: t,
    patientReference: n,
    openMode: We,
    includeLegacyNiftiIdentifiers: false
  });
}
function Ji({
  id: e,
  studyInstanceUID: t,
  seriesInstanceUID: n,
  dicomwebRoot: i,
  patientReference: r,
  ohifMode: o,
  files: s,
  sourceBucket: a
}) {
  const l = String(e || "").trim() || "study", h = B(t), c = B(n), u = String(i || "").trim().replace(/\/$/, ""), d = [
    { system: de, value: Te },
    { system: $e, value: h },
    {
      system: fe,
      value: l
    }
  ];
  u && d.push({ system: Ot, value: u });
  const f = String(o || "").trim();
  f && d.push({
    system: qe,
    value: f
  });
  const m = (Array.isArray(s) ? s : []).map((S) => Je(S)).filter((S) => S !== null);
  if (m.length > 0) {
    const S = String(a || "aws").trim().toLowerCase() === "gcs" ? "gcs" : "aws";
    d.push({
      system: Xe,
      value: S
    }), c && d.push({ system: Ge, value: c });
  }
  const g = {
    resourceType: "ImagingStudy",
    id: l,
    uid: h,
    meta: {
      profile: [He]
    },
    identifier: d,
    status: "available"
  };
  r && (g.subject = { reference: String(r).trim() });
  const p = [
    {
      key: "study",
      resource: g
    }
  ];
  return c && p.push({
    key: "series",
    resource: {
      resourceType: "ImagingStudy",
      uid: c
    }
  }), m.length > 0 && p.push({
    key: "files",
    resource: {
      files: m.map((S) => ({
        url: S.url,
        fileName: S.fileName || void 0,
        mimeType: S.mimeType || void 0,
        role: S.role || void 0,
        label: S.label || void 0
      }))
    }
  }), p;
}
function Yi({
  id: e,
  studyInstanceUID: t,
  seriesInstanceUID: n,
  patientReference: i
}) {
  const r = String(e || "").trim() || "study", o = B(t), s = B(n), l = {
    resourceType: "ImagingStudy",
    id: r,
    uid: o,
    meta: {
      profile: [He]
    },
    identifier: [
      { system: de, value: Ve },
      { system: $e, value: o },
      {
        system: fe,
        value: r
      }
    ],
    status: "available"
  };
  i && (l.subject = { reference: String(i).trim() });
  const h = [
    {
      key: "study",
      resource: l
    }
  ];
  return s && h.push({
    key: "series",
    resource: {
      resourceType: "ImagingStudy",
      uid: s
    }
  }), h;
}
function Zi({
  id: e,
  studyInstanceUID: t,
  seriesInstanceUID: n,
  sourceBucket: i,
  files: r,
  patientReference: o,
  ohifMode: s
}) {
  const a = String(e || "").trim() || "study", l = B(t), h = B(n), c = String(i || "aws").trim().toLowerCase() === "gcs" ? "gcs" : "aws", u = (Array.isArray(r) ? r : []).map((p) => Je(p)).filter((p) => p !== null), d = [
    { system: de, value: ze },
    { system: $e, value: l },
    {
      system: fe,
      value: a
    },
    {
      system: je,
      value: a
    },
    { system: Xe, value: c }
  ];
  h && d.push({ system: Ge, value: h });
  const f = String(s || "").trim();
  f && d.push({
    system: qe,
    value: f
  });
  const m = {
    resourceType: "ImagingStudy",
    id: a,
    uid: l,
    meta: {
      profile: [He]
    },
    identifier: d,
    status: "available"
  };
  o && (m.subject = { reference: String(o).trim() });
  const g = [
    {
      key: "study",
      resource: m
    }
  ];
  return h && g.push({
    key: "series",
    resource: {
      resourceType: "ImagingStudy",
      uid: h
    }
  }), u.length > 0 && g.push({
    key: "files",
    resource: {
      files: u.map((p) => ({
        url: p.url,
        fileName: p.fileName || void 0,
        mimeType: p.mimeType || void 0,
        role: p.role || void 0,
        label: p.label || void 0
      }))
    }
  }), g;
}
function Ki({
  id: e,
  url: t,
  filename: n,
  patientReference: i
}) {
  return zn({
    id: e,
    files: [{ url: t, fileName: n }],
    patientReference: i,
    includeLegacyNiftiIdentifiers: true
  });
}
function Pt(e) {
  if (Array.isArray(e) || !e || typeof e != "object")
    return e;
  const t = Object.entries(e).reduce((n, [i, r]) => (r && typeof r == "object" && !Array.isArray(r) && n.push({ key: i, resource: r }), n), []);
  return t.length ? t : e;
}
function an(e, t, n) {
  const i = Pt(e), r = ue(i);
  return r.length === 0 ? null : { mode: "files", studyId: t, files: r, ohifMode: n };
}
function Qi(e) {
  const t = Pt(e), n = jn(t) || "study", i = qn(t), r = Bn(t) || void 0;
  if (i === Te) {
    const l = se(t);
    if (!l)
      return null;
    const h = {
      mode: "dicomweb",
      studyId: n,
      studyInstanceUID: l,
      seriesInstanceUID: oe(t) || void 0,
      dicomwebRoot: mt(t) || void 0,
      ohifMode: r
    }, c = ue(t);
    return c.length > 0 && (h.idcFallback = {
      studyInstanceUID: l,
      seriesInstanceUID: gt(t) || oe(t) || void 0,
      sourceBucket: pt(t),
      files: c
    }), h;
  }
  if (i === ce) {
    const l = an(t, n, r);
    if (l)
      return l;
  }
  if (i === We) {
    const l = ue(t);
    return l.length ? { mode: "dicom-url", studyId: n, files: l, ohifMode: r } : null;
  }
  if (i === ze) {
    const l = se(t);
    if (!l)
      return null;
    const h = ue(t);
    return h.length ? {
      mode: "idc",
      studyId: n,
      studyInstanceUID: l,
      seriesInstanceUID: gt(t) || oe(t) || void 0,
      sourceBucket: pt(t),
      files: h,
      ohifMode: r
    } : null;
  }
  if (i === Ve) {
    const l = se(t);
    return l ? {
      mode: "local-dicom",
      studyId: n,
      studyInstanceUID: l,
      seriesInstanceUID: oe(t) || void 0,
      ohifMode: r
    } : null;
  }
  const o = an(t, n, r);
  if (o)
    return o;
  const s = Ne(t);
  if (s) {
    const l = Mt(t);
    return {
      mode: "files",
      studyId: n,
      files: [{ url: s, fileName: l || "volume.nii.gz", label: n }],
      ohifMode: r
    };
  }
  const a = se(t);
  return a ? {
    mode: "dicomweb",
    studyId: n,
    studyInstanceUID: a,
    seriesInstanceUID: oe(t) || void 0,
    dicomwebRoot: mt(t) || void 0,
    ohifMode: r
  } : null;
}
function yt(e) {
  return typeof e != "string" ? "" : e.trim().replace(/^urn:oid:/i, "");
}
function eo(e) {
  if (!e || typeof e != "object")
    return "";
  const t = yt(e.uid);
  if (t)
    return t;
  const i = (Array.isArray(e.identifier) ? e.identifier : []).find(
    (r) => typeof (r == null ? void 0 : r.system) == "string" && r.system.toLowerCase() === "urn:dicom:uid"
  );
  return yt(i == null ? void 0 : i.value);
}
function ke(e) {
  const t = e == null ? void 0 : e["hub.event"];
  return typeof t == "string" ? t.trim().toLowerCase() : "";
}
function Vn(e) {
  return typeof e == "string" ? e.trim().toUpperCase() : !e || typeof e != "object" ? "" : (typeof e.keyword == "string" && e.keyword || typeof e.id == "string" && e.id || typeof e.key == "string" && e.key || "").trim().toUpperCase();
}
function to(e) {
  const t = e == null ? void 0 : e["target.actor"];
  return t == null ? "" : Vn(t);
}
function no(e) {
  if (!Array.isArray(e))
    return [];
  try {
    return structuredClone(e);
  } catch {
    try {
      return JSON.parse(JSON.stringify(e));
    } catch {
      return [...e];
    }
  }
}
function ro(e) {
  const t = e == null ? void 0 : e.event;
  return t && typeof t == "object" && t.context != null ? t.context : [];
}
function io(e) {
  return String(
    (e == null ? void 0 : e["subscriber.name"]) || (e == null ? void 0 : e.subscriber) || ""
  ).trim() || null;
}
function oo(e) {
  return String((e == null ? void 0 : e["subscriber.product.name"]) || "").trim();
}
function so(e) {
  const t = String((e == null ? void 0 : e["subscriber.actor"]) || "").trim();
  if (t)
    return t;
  const n = e == null ? void 0 : e["subscriber.actors"];
  if (Array.isArray(n))
    for (const i of n) {
      const r = String(i || "").trim();
      if (r)
        return r;
    }
  return "";
}
function ao(e) {
  const t = e == null ? void 0 : e.event;
  if (ke(t) !== "status-update")
    return null;
  const n = t != null && t.context && typeof t.context == "object" && !Array.isArray(t.context) ? t.context : {};
  return { message: String(n.message ?? "").trim(), level: String(n.level ?? "info") };
}
function co(e) {
  const t = atob(e), n = new Uint8Array(t.length);
  for (let i = 0; i < t.length; i += 1)
    n[i] = t.charCodeAt(i);
  return n;
}
function Gn(e) {
  try {
    const t = co(e), n = new ArrayBuffer(t.byteLength);
    return new Uint8Array(n).set(t), n;
  } catch {
    return null;
  }
}
function Xn(e) {
  const t = e == null ? void 0 : e.context;
  if (!t || typeof t != "object" || Array.isArray(t))
    return [];
  const n = t.files;
  return Array.isArray(n) ? n.filter((i) => !!(i && typeof i == "object")) : [];
}
function uo(e) {
  const t = e == null ? void 0 : e.context;
  return Array.isArray(t) ? t : t != null ? [t] : [];
}
function cn(e, t, n) {
  return e instanceof ArrayBuffer ? { arrayBuffer: e, fileName: t, mimeType: n } : typeof e == "string" && e ? { fileName: t, data: e, mimeType: n } : null;
}
function Jn(e) {
  return e === "nifti-send" ? "application/vnd.unknown.nifti-1" : "application/dicom";
}
function lo(e, t) {
  return e === "nifti-send" ? `hub-nifti-send-${t + 1}.nii.gz` : `hub-dicom-send-${t + 1}.dcm`;
}
function fo(e, t) {
  return e === "nifti-send" ? `hub-nifti-send-${t + 1}.nii.gz` : "dicom-sr.dcm";
}
function bo(e, t) {
  return typeof e.mimeType == "string" && e.mimeType.trim() ? e.mimeType.trim() : Jn(t);
}
function ho(e, t) {
  return typeof e.mimeType == "string" && e.mimeType.trim() ? e.mimeType.trim() : typeof e.contentType == "string" && e.contentType.trim() ? e.contentType.trim() : Jn(t);
}
function mo(e, t) {
  const n = e == null ? void 0 : e.event;
  if (ke(n) !== t.toLowerCase())
    return [];
  const i = Xn(n);
  if (i.length) {
    const o = i.map((s, a) => {
      const l = bo(s, t), h = typeof s.fileName == "string" && s.fileName.trim() ? s.fileName.trim() : lo(t, a);
      return cn(s.data, h, l);
    }).filter(Boolean);
    if (o.length)
      return o;
  }
  return uo(n).map((o, s) => {
    if (!o || typeof o != "object")
      return null;
    const a = o.resource;
    if (!a || typeof a != "object")
      return null;
    const l = ho(a, t), h = typeof a.fileName == "string" && a.fileName.trim() ? a.fileName.trim() : fo(t, s);
    return cn(a.data, h, l);
  }).filter(Boolean);
}
function Yn(e) {
  return (e == null ? void 0 : e.arrayBuffer) instanceof ArrayBuffer ? e.arrayBuffer : typeof (e == null ? void 0 : e.data) == "string" ? Gn(e.data) : null;
}
function go(e, t, n) {
  const i = Yn(e);
  if (!i)
    return null;
  const r = (e == null ? void 0 : e.fileName) || t, o = (e == null ? void 0 : e.mimeType) || n;
  return new File([i], r, { type: o });
}
var un = 4e3;
var po = 480;
var Me = "[binary/redacted]";
function Pe(e, t) {
  return t > 8 ? "[max-depth]" : e instanceof ArrayBuffer || ArrayBuffer.isView(e) ? Me : typeof e == "string" ? e.length > po ? `${e.slice(0, 160)}\u2026 (${e.length} chars)` : e : Array.isArray(e) ? e.map((n) => Pe(n, t + 1)) : e && typeof e == "object" ? Object.entries(e).reduce((n, [i, r]) => (i.toLowerCase() === "data" && typeof r == "string" && r.length > 200 ? n[i] = Me : n[i] = Pe(r, t + 1), n), {}) : e;
}
function St(e) {
  try {
    const t = JSON.stringify(Pe(e, 0));
    return t.length > un ? `${t.slice(0, un)}\u2026` : t;
  } catch {
    return "[unserializable]";
  }
}
function Ft(e) {
  if (!e)
    return 0;
  const { context: t } = e;
  if (Array.isArray(t))
    return t.length;
  if (t && typeof t == "object") {
    const n = t.files;
    return Array.isArray(n) && n.length ? n.length : 1;
  }
  return 0;
}
function Zn(e, t) {
  if (!e)
    return t === 1 ? "1 object" : `${t} objects`;
  if (t === 0)
    return e;
  const n = e === "dicom-send" || e === "nifti-send" ? "image" : "object", i = t === 1 ? n : `${n}s`;
  return `${e} (${t} ${i})`;
}
function yo(e) {
  const t = e != null && e.event && typeof e.event == "object" ? e.event : void 0, i = ke(t) || "publish", r = Ft(t);
  return { label: Zn(i, r), detail: "" };
}
function So(e) {
  if (!e || typeof e != "object")
    return { label: "raw", detail: St(e) };
  const t = e.event, i = ke(t) || (typeof (t == null ? void 0 : t.event) == "string" ? String(t.event) : "message"), r = Ft(t);
  return { label: Zn(i, r), detail: St(e) };
}
function Co(e) {
  return Pe(e, 0);
}
var Q = "*";
var Ye = "ID";
var Eo = {
  subscriberName: Q,
  subscriberActor: Ye,
  targetActor: Q,
  targetProductName: Q
};
function Le(e) {
  return String(e ?? "").trim() || Q;
}
function wo(e, t) {
  return {
    subscriberName: (e.subscriberName ?? "").trim() || t.subscriberName || Q,
    subscriberActor: (e.subscriberActor ?? "").trim() || Ye,
    targetActor: Le(e.targetActor),
    targetProductName: Le(e.targetProductName)
  };
}
function To(e, t) {
  e["subscriber.name"] = t.subscriberName.trim() || Q, e["subscriber.actor"] = t.subscriberActor.trim() || Ye, e["target.actor"] = Le(t.targetActor), e["target.product.name"] = Le(
    t.targetProductName
  );
}
function Kn(e) {
  const t = e == null ? void 0 : e.event;
  return !!(t && ut(t));
}
async function No(e, t) {
  var r, o;
  if (!(((r = e == null ? void 0 : e.hasPendingPayload) == null ? void 0 : r.call(e, t)) || Kn(t)))
    return t;
  let i = t;
  if (e != null && e.fetchAllPayloads)
    i = await e.fetchAllPayloads(t);
  else if (e != null && e.fetchPayload)
    for (; (o = e.hasPendingPayload) != null && o.call(e, i); )
      i = await e.fetchPayload(i);
  return i;
}
var Qn = [
  "TOTALSEG",
  "TOTAL_SEGMENTATOR",
  "TOTAL-SEGMENTATOR"
];
var er = [
  "LUNGSCREENING",
  "LUNG_SCREENING"
];
var tr = [
  "NEURO_SEG",
  "NEUROSEG",
  "NEURO-SEG"
];
var nr = [
  "DENTAL_SEG",
  "DENTALSEG",
  "DENTAL-SEG",
  "DENTALSEGMENTATOR"
];
var rr = [
  "TORCHXRAYVISION",
  "TXRV",
  "TORCH_XRAY_VISION",
  "TORCH-XRAY-VISION"
];
var Lo = ["FLEXRAY", "FLEX_RAY", "FLEX-RAY"];
var ir = ["MHUB", "MHUB_AI", "MHUBAI"];
function Ct(e) {
  return String(e || "").trim().toUpperCase().replace(/-/g, "_");
}
function te(e, t) {
  const n = Ct(e);
  if (!n)
    return false;
  const i = n.replace(/_/g, "");
  return t.some((r) => {
    const o = Ct(r);
    return o === n || o.replace(/_/g, "") === i;
  });
}
function Ze(e) {
  return te(e, Qn);
}
function Ke(e) {
  return te(e, er);
}
function Qe(e) {
  return te(e, tr);
}
function et(e) {
  return te(e, nr);
}
function tt(e) {
  return te(e, rr);
}
function Dt(e) {
  return te(e, Lo);
}
function nt(e) {
  return te(e, ir);
}
function vo(e) {
  return nt(e) || Ze(e) || Ke(e) || Qe(e) || et(e) || tt(e) || Dt(e);
}
function Bt(e, t) {
  if (!Array.isArray(e))
    return;
  const n = String(t).trim().toLowerCase(), i = e.find((o) => !o || typeof o != "object" ? false : String(o.key ?? "").trim().toLowerCase() === n);
  if (!i)
    return;
  const r = i.value;
  return typeof r == "string" ? r.trim() : void 0;
}
function jt(e) {
  if (!e || typeof e != "object" || e.source !== "status")
    return false;
  const t = Bt(e.items, "availability");
  return t ? t.toLowerCase() === "online" : Array.isArray(e.items) && e.items.length > 0;
}
function rt(e) {
  const t = String((e == null ? void 0 : e.productName) ?? "").trim();
  if (t)
    return t;
  const n = e == null ? void 0 : e.data;
  if (n && typeof n == "object") {
    const i = String(n.product ?? "").trim();
    if (i)
      return i;
  }
  return "";
}
function be(e, t) {
  return Array.isArray(e) ? e.some((n) => {
    const i = rt(n);
    return t(i) && jt(n.data);
  }) : false;
}
function Ao(e) {
  return be(
    e,
    Ze
  );
}
function Io(e) {
  return be(e, Ke);
}
function Uo(e) {
  return be(e, Qe);
}
function xo(e) {
  return be(e, et);
}
function Ho(e) {
  return be(
    e,
    tt
  );
}
function $o(e) {
  return be(e, nt);
}
function ko(e) {
  return typeof e != "string" ? false : e.trim().toUpperCase() === "STATUS";
}
var _o = "mhub";
var Ro = "MHUB";
var Oo = "MHub models";
var Mo = "Boston, USA";
var Po = "Medical Imaging Model Hub \u2014 peer-reviewed deep learning models via standardized Docker I/O.";
var Fo = "https://github.com/MHubAI/";
var Do = "https://mhub.ai";
var Bo = "dicom-send";
var jo = true;
var qo = "https://github.com/MHubAI.png";
var Wo = {
  id: _o,
  product: Ro,
  title: Oo,
  location: Mo,
  summary: Po,
  githubUrl: Fo,
  websiteUrl: Do,
  hubEvent: Bo,
  enabled: jo,
  iconUrl: qo
};
var zo = "lung";
var Vo = "LUNGSCREENING";
var Go = "Lung Screening";
var Xo = "Toronto, Canada";
var Jo = "CT lung screening.";
var Yo = "https://github.com/rphellan/slicerW45";
var Zo = "https://projectweek.na-mic.org/PW45_2026_Boston/Projects/ExtractingDeepLearningFeaturesFromCtImagesOfTheThoracicRegionForLungCancerApplications/";
var Ko = "dicom-send";
var Qo = false;
var es = {
  id: zo,
  product: Vo,
  title: Go,
  location: Xo,
  summary: Jo,
  githubUrl: Yo,
  websiteUrl: Zo,
  hubEvent: Ko,
  enabled: Qo
};
var ts = "totalseg";
var ns = "TOTALSEG";
var rs = "Total Segmentator";
var is = "Basel, Switzerland";
var os = "Whole-body CT/MR multi-organ segmentation.";
var ss = "https://github.com/wasserth/TotalSegmentator";
var as = "https://totalsegmentator.com/";
var cs = "https://www.buymeacoffee.com/totalsegmentator";
var us = "dicom-send";
var ls = true;
var fs = "https://pubs.rsna.org/doi/10.1148/ryai.230024";
var ds = {
  id: ts,
  product: ns,
  title: rs,
  location: is,
  summary: os,
  githubUrl: ss,
  websiteUrl: as,
  buyMeACoffeeUrl: cs,
  hubEvent: us,
  enabled: ls,
  citeUrl: fs
};
var bs = "neuro";
var hs = "NEURO_SEG";
var ms = "Subcortical segmentation";
var gs = "Montreal, Canada";
var ps = "Neuro / subcortical segmentation. Improved white and pial surfaces and parcellation (aparc+aseg) from recon-all pipeline (FreeSurfer 7.4.1) through added dataset";
var ys = "https://projectweek.na-mic.org/PW45_2026_Boston/Projects/ReconAllCorrectionScriptBasedOnManualSubcorticalSegmentationFiles/";
var Ss = "nifti-send";
var Cs = true;
var Es = {
  id: bs,
  product: hs,
  title: ms,
  location: gs,
  summary: ps,
  websiteUrl: ys,
  hubEvent: Ss,
  enabled: Cs
};
var ws = "dental";
var Ts = "DENTAL_SEG";
var Ns = "Dental Segmentator";
var Ls = "Paris, France";
var vs = "Fully automatic CT/CBCT dental segmentation (maxilla, mandible, upper/lower teeth, mandibular canal).";
var As = "https://github.com/gaudot/SlicerDentalSegmentator";
var Is = "https://www.kitware.com/dental-segmentator/";
var Us = "Apache-2.0";
var xs = "https://www.apache.org/licenses/LICENSE-2.0";
var Hs = "dicom-send";
var $s = true;
var ks = {
  id: ws,
  product: Ts,
  title: Ns,
  location: Ls,
  summary: vs,
  githubUrl: As,
  websiteUrl: Is,
  license: Us,
  licenseUrl: xs,
  hubEvent: Hs,
  enabled: $s
};
var _s = "txrv";
var Rs = "TORCHXRAYVISION";
var Os = "TorchXRayVision";
var Ms = "Montreal, Canada";
var Ps = "Chest X-ray pathology classification (18 findings) and lung/heart anatomy segmentation.";
var Fs = "https://github.com/mlmed/torchxrayvision";
var Ds = "https://mlmed.org/torchxrayvision/";
var Bs = "Cohen et al., TorchXRayVision: A library of chest X-ray datasets and models, MIDL 2022";
var js = "https://arxiv.org/abs/2111.00595";
var qs = "dicom-send";
var Ws = true;
var zs = {
  id: _s,
  product: Rs,
  title: Os,
  location: Ms,
  summary: Ps,
  githubUrl: Fs,
  websiteUrl: Ds,
  cite: Bs,
  citeUrl: js,
  hubEvent: qs,
  enabled: Ws
};
var Vs = "flexray";
var Gs = "FLEXRAY";
var Xs = "FleXray";
var Js = "local - in browser";
var Ys = "Foundation model for universal clinical X-ray anatomy segmentation (60 structures) across body regions, views, and acquisition settings. Designed for on-device / local inference.";
var Zs = "https://flexray.csail.mit.edu/";
var Ks = "https://github.com/VictorButoi/FleXray";
var Qs = "CC-BY-NC-4.0";
var ea = "https://creativecommons.org/licenses/by-nc/4.0/";
var ta = "Not for commercial use";
var na = "Victor Ion Butoi, Vivek Gopalakrishnan, John V. Guttag, Adrian V. Dalca, and Neel Dey (2026). FleXray: Universal Clinical X-ray Segmentation. arXiv:2609.26756";
var ra = "https://arxiv.org/abs/2609.26756";
var ia = "dicom-send";
var oa = true;
var sa = true;
var aa = {
  id: Vs,
  product: Gs,
  title: Xs,
  location: Js,
  summary: Ys,
  websiteUrl: Zs,
  githubUrl: Ks,
  license: Qs,
  licenseUrl: ea,
  licenseRestriction: ta,
  cite: na,
  citeUrl: ra,
  hubEvent: ia,
  enabled: oa,
  localCapable: sa
};
var ca = "data:image/svg+xml,%3csvg%20width='35'%20height='50'%20viewBox='0%200%2035%2050'%20fill='none'%20xmlns='http://www.w3.org/2000/svg'%3e%3cpath%20d='M30.7512%2011.6307L30.7171%2011.6106L30.6382%2011.5865C30.6699%2011.6134%2030.7097%2011.6289%2030.7512%2011.6307Z'%20fill='%23FFDD00'/%3e%3cpath%20d='M31.2481%2015.2031L31.21%2015.2138L31.2481%2015.2031Z'%20fill='%23FFDD00'/%3e%3cpath%20d='M30.7659%2011.6253C30.761%2011.6247%2030.7563%2011.6236%2030.7517%2011.6219C30.7514%2011.6251%2030.7514%2011.6283%2030.7517%2011.6314C30.7569%2011.6307%2030.7618%2011.6286%2030.7659%2011.6253Z'%20fill='%23FFDD00'/%3e%3cpath%20d='M30.7515%2011.6314H30.7572V11.6278L30.7515%2011.6314Z'%20fill='%23FFDD00'/%3e%3cpath%20d='M31.2178%2015.1962L31.2753%2015.1634L31.2967%2015.1514L31.3161%2015.1307C31.2796%2015.1464%2031.2463%2015.1686%2031.2178%2015.1962Z'%20fill='%23FFDD00'/%3e%3cpath%20d='M30.8507%2011.7088L30.7945%2011.6553L30.7563%2011.6345C30.7768%2011.6707%2030.8107%2011.6973%2030.8507%2011.7088Z'%20fill='%23FFDD00'/%3e%3cpath%20d='M16.715%2046.3714C16.6701%2046.3908%2016.6307%2046.4212%2016.6006%2046.4597L16.636%2046.4369C16.6601%2046.4149%2016.6942%2046.3888%2016.715%2046.3714Z'%20fill='%23FFDD00'/%3e%3cpath%20d='M24.9235%2044.7473C24.9235%2044.6965%2024.8987%2044.7059%2024.9048%2044.8864C24.9048%2044.8717%2024.9108%2044.857%2024.9134%2044.843C24.9168%2044.8109%2024.9195%2044.7794%2024.9235%2044.7473Z'%20fill='%23FFDD00'/%3e%3cpath%20d='M24.0719%2046.3714C24.027%2046.3908%2023.9877%2046.4212%2023.9575%2046.4597L23.993%2046.4369C24.017%2046.4149%2024.0512%2046.3888%2024.0719%2046.3714Z'%20fill='%23FFDD00'/%3e%3cpath%20d='M10.9344%2046.7574C10.9003%2046.7277%2010.8586%2046.7082%2010.814%2046.7012C10.8501%2046.7186%2010.8862%2046.736%2010.9103%2046.7493L10.9344%2046.7574Z'%20fill='%23FFDD00'/%3e%3cpath%20d='M9.63391%2045.5042C9.62859%2045.4515%209.61242%2045.4005%209.58643%2045.3544C9.60484%2045.4024%209.62025%2045.4516%209.63258%2045.5015L9.63391%2045.5042Z'%20fill='%23FFDD00'/%3e%3cpath%20d='M18.3715%2023.0976C16.5857%2023.8665%2014.5591%2024.7382%2011.9326%2024.7382C10.8339%2024.736%209.74049%2024.5844%208.68213%2024.2875L10.4987%2043.0444C10.563%2043.8284%2010.9181%2044.5594%2011.4935%2045.0923C12.0689%2045.6251%2012.8225%2045.9208%2013.6047%2045.9207C13.6047%2045.9207%2016.1804%2046.0552%2017.0398%2046.0552C17.9648%2046.0552%2020.7385%2045.9207%2020.7385%2045.9207C21.5205%2045.9207%2022.274%2045.6249%2022.8493%2045.092C23.4245%2044.5592%2023.7795%2043.8283%2023.8438%2043.0444L25.7895%2022.3173C24.92%2022.0187%2024.0425%2021.8203%2023.0533%2021.8203C21.3424%2021.8196%2019.964%2022.4122%2018.3715%2023.0976Z'%20fill='%23FFFFFF'/%3e%3cpath%20d='M3.05859%2015.095L3.08936%2015.1238L3.10943%2015.1358C3.09397%2015.1205%203.07693%2015.1068%203.05859%2015.095Z'%20fill='%23FFDD00'/%3e%3cpath%20d='M34.1896%2013.3639L33.916%2011.9762C33.6706%2010.7312%2033.1134%209.55468%2031.8426%209.10468C31.4353%208.96073%2030.9732%208.89885%2030.6608%208.60086C30.3485%208.30288%2030.2562%207.84009%2030.1839%207.41094C30.0502%206.62326%2029.9244%205.83492%2029.7873%205.04859C29.6689%204.37257%2029.5753%203.61315%2029.267%202.99296C28.8657%202.16022%2028.033%201.67322%2027.205%201.35102C26.7807%201.19173%2026.3477%201.05698%2025.9081%200.947426C23.8394%200.398542%2021.6644%200.196746%2019.5362%200.0817228C16.9817%20-0.0600379%2014.4204%20-0.0173274%2011.872%200.209526C9.97525%200.383071%207.97745%200.592939%206.17495%201.25281C5.51616%201.49429%204.8373%201.7842%204.33634%202.29609C3.72169%202.92502%203.52104%203.89768%203.96982%204.68199C4.28886%205.23895%204.82927%205.63245%205.40246%205.89276C6.14906%206.22818%206.92877%206.48341%207.72865%206.65421C9.95585%207.14928%2012.2626%207.34368%2014.538%207.42641C17.0599%207.52878%2019.5859%207.44582%2022.0958%207.1782C22.7165%207.10959%2023.336%207.0273%2023.9545%206.93134C24.6828%206.81901%2025.1503%205.86115%2024.9356%205.19388C24.6788%204.39611%2023.9886%204.08669%2023.208%204.2071C23.093%204.22526%2022.9786%204.24208%2022.8636%204.25889L22.7807%204.271C22.5163%204.30463%2022.2518%204.33602%2021.9874%204.36517C21.4412%204.42437%2020.8937%204.4728%2020.3448%204.51046C19.1155%204.59656%2017.8828%204.63625%2016.6508%204.63827C15.4403%204.63827%2014.229%204.60396%2013.0211%204.52392C12.47%204.48759%2011.9202%204.4414%2011.3718%204.38535C11.1223%204.35912%2010.8735%204.33154%2010.6247%204.3006L10.3879%204.27033L10.3364%204.26293L10.091%204.22728C9.58933%204.15127%209.08771%204.06382%208.59144%203.95822C8.54136%203.94704%208.49657%203.91902%208.46446%203.87879C8.43236%203.83855%208.41486%203.7885%208.41486%203.73691C8.41486%203.68533%208.43236%203.63528%208.46446%203.59504C8.49657%203.55481%208.54136%203.52679%208.59144%203.51561H8.6008C9.03086%203.42346%209.46426%203.34476%209.899%203.27615C10.0439%203.25328%2010.1893%203.23086%2010.3351%203.20888H10.3391C10.6113%203.19072%2010.8849%203.14162%2011.1557%203.10933C13.5125%202.86279%2015.8832%202.77874%2018.2513%202.85776C19.4011%202.89139%2020.5501%202.95933%2021.6945%203.07637C21.9406%203.10193%2022.1854%203.12884%2022.4302%203.15911C22.5238%203.17054%2022.6181%203.18399%2022.7124%203.19543L22.9024%203.22301C23.4562%203.30597%2024.0071%203.40664%2024.5551%203.52503C25.367%203.70261%2026.4097%203.76046%2026.7709%204.65508C26.8859%204.93894%2026.9381%205.25442%2027.0016%205.5524L27.0826%205.93245C27.0847%205.93927%2027.0863%205.94624%2027.0873%205.9533C27.2785%206.85017%2027.4701%207.74704%2027.6618%208.64391C27.6758%208.71017%2027.6762%208.77862%2027.6628%208.84501C27.6493%208.9114%2027.6225%208.97429%2027.5838%209.02977C27.5451%209.08525%2027.4955%209.13212%2027.4381%209.16746C27.3806%209.2028%2027.3165%209.22585%2027.2498%209.23517H27.2444L27.1274%209.25132L27.0117%209.26679C26.6452%209.31477%2026.2782%209.35961%2025.9108%209.40132C25.1871%209.48428%2024.4623%209.55603%2023.7364%209.61657C22.294%209.7372%2020.8486%209.81635%2019.4004%209.85401C18.6625%209.87374%2017.9247%209.88294%2017.1872%209.88159C14.2517%209.87926%2011.3188%209.70768%208.40283%209.36769C8.08714%209.33002%207.77145%209.28966%207.45577%209.24863C7.70056%209.28024%207.27786%209.22441%207.19225%209.2123C6.9916%209.18405%206.79095%209.15468%206.5903%209.12419C5.91679%209.02262%205.24729%208.8975%204.57511%208.78786C3.76249%208.65333%202.98531%208.72059%202.25026%209.12419C1.6469%209.45624%201.15857%209.96544%200.850401%2010.5838C0.533376%2011.243%200.439071%2011.9608%200.297279%2012.6691C0.155487%2013.3774%20-0.0652275%2014.1395%200.0183763%2014.8666C0.198292%2016.4359%201.28915%2017.7113%202.85823%2017.9965C4.33434%2018.2655%205.81847%2018.4835%207.30662%2018.6691C13.1524%2019.3892%2019.0582%2019.4753%2024.9223%2018.9261C25.3998%2018.8812%2025.8767%2018.8324%2026.3529%2018.7794C26.5016%2018.763%2026.6521%2018.7802%2026.7934%2018.8299C26.9347%2018.8795%2027.0631%2018.9603%2027.1693%2019.0663C27.2755%2019.1724%2027.3567%2019.3009%2027.4071%2019.4426C27.4575%2019.5843%2027.4757%2019.7356%2027.4605%2019.8853L27.312%2021.3369C27.0128%2024.2701%2026.7136%2027.2031%2026.4144%2030.1358C26.1023%2033.2157%2025.7882%2036.2953%2025.472%2039.3747C25.3829%2040.242%2025.2937%2041.109%2025.2045%2041.9758C25.1189%2042.8294%2025.1069%2043.7099%2024.9457%2044.5534C24.6915%2045.8799%2023.7986%2046.6945%2022.4957%2046.9925C21.3021%2047.2657%2020.0827%2047.4091%2018.8586%2047.4203C17.5016%2047.4277%2016.1452%2047.3672%2014.7881%2047.3746C13.3395%2047.3826%2011.5651%2047.2481%2010.4468%2046.1638C9.46426%2045.2113%209.32849%2043.72%209.19472%2042.4306C9.01637%2040.7234%208.83957%2039.0164%208.66434%2037.3097L7.68116%2027.8192L7.0451%2021.6786C7.0344%2021.577%207.0237%2021.4768%207.01367%2021.3745C6.93742%2020.642%206.42176%2019.925%205.60913%2019.962C4.91354%2019.9929%204.12299%2020.5875%204.20458%2021.3745L4.67611%2025.927L5.65126%2035.3442C5.92905%2038.0191%206.20617%2040.6944%206.48262%2043.3703C6.53613%2043.8828%206.58628%2044.3967%206.64247%2044.9093C6.94812%2047.7102%209.075%2049.2196%2011.7089%2049.6448C13.2472%2049.8936%2014.8229%2049.9448%2016.384%2049.9703C18.3851%2050.0026%2020.4063%2050.08%2022.3747%2049.7154C25.2915%2049.1773%2027.4799%2047.2185%2027.7922%2044.1801C27.8814%2043.303%2027.9706%2042.4256%2028.0597%2041.548C28.3563%2038.6458%2028.6523%2035.7433%2028.9479%2032.8406L29.9151%2023.3562L30.3585%2019.0095C30.3806%2018.794%2030.4711%2018.5913%2030.6166%2018.4315C30.7621%2018.2717%2030.9549%2018.1633%2031.1665%2018.1223C32.0005%2017.9588%2032.7977%2017.6796%2033.391%2017.0413C34.3354%2016.0249%2034.5233%2014.6998%2034.1896%2013.3639ZM2.81542%2014.3016C2.82813%2014.2955%202.80472%2014.4052%202.79469%2014.4563C2.79268%2014.3789%202.7967%2014.3103%202.81542%2014.3016ZM2.89635%2014.9312C2.90304%2014.9265%202.9231%2014.9534%202.94384%2014.9857C2.9124%2014.9561%202.89234%2014.9339%202.89568%2014.9312H2.89635ZM2.97594%2015.0368C3.0047%2015.0859%203.02009%2015.1168%202.97594%2015.0368V15.0368ZM3.13579%2015.1673H3.1398C3.1398%2015.172%203.14716%2015.1767%203.14984%2015.1814C3.1454%2015.1762%203.14048%2015.1715%203.13513%2015.1673H3.13579ZM31.1277%2014.9722C30.828%2015.2588%2030.3766%2015.392%2029.9305%2015.4586C24.9276%2016.2052%2019.8519%2016.5832%2014.7942%2016.4164C11.1745%2016.292%207.59287%2015.8877%204.00928%2015.3785C3.65815%2015.3287%203.27758%2015.2642%203.03614%2015.0038C2.58133%2014.5128%202.80472%2013.524%202.9231%2012.9307C3.03145%2012.3872%203.23879%2011.6628%203.88154%2011.5854C4.88478%2011.467%206.04989%2011.8928%207.04243%2012.0442C8.2374%2012.2276%209.43684%2012.3744%2010.6407%2012.4848C15.7787%2012.9556%2021.0029%2012.8823%2026.1181%2012.1935C27.0505%2012.0675%2027.9795%2011.9211%2028.9051%2011.7543C29.7298%2011.6056%2030.6441%2011.3264%2031.1424%2012.1854C31.4841%2012.7706%2031.5296%2013.5536%2031.4768%2014.2148C31.4605%2014.5029%2031.3354%2014.7739%2031.127%2014.9722H31.1277Z'%20fill='%23FFDD00'/%3e%3c/svg%3e";
function ua(e) {
  const t = String(e.summary || e.capabilities || "").trim(), n = e.hubEvent === "nifti-send" ? "nifti-send" : "dicom-send", i = {
    id: String(e.id || "").trim(),
    product: String(e.product || "").trim(),
    title: String(e.title || "").trim(),
    location: String(e.location || "").trim(),
    summary: t,
    capabilities: t,
    hubEvent: n
  }, r = String(e.githubUrl || "").trim();
  r && (i.githubUrl = r);
  const o = String(e.websiteUrl || "").trim();
  o && (i.websiteUrl = o);
  const s = String(e.buyMeACoffeeUrl || "").trim();
  s && (i.buyMeACoffeeUrl = s);
  const a = String(e.iconUrl || "").trim();
  a && (i.iconUrl = a);
  const l = String(e.cite || "").trim();
  l && (i.cite = l);
  const h = String(e.citeUrl || "").trim();
  h && (i.citeUrl = h);
  const c = String(e.license || "").trim();
  c && (i.license = c);
  const u = String(e.licenseUrl || "").trim();
  u && (i.licenseUrl = u);
  const d = String(e.licenseRestriction || "").trim();
  return d && (i.licenseRestriction = d), e.localCapable === true && (i.localCapable = true), i;
}
function la(e) {
  return (e == null ? void 0 : e.enabled) !== false;
}
var z = Object.freeze(
  [Wo, es, ds, Es, ks, zs, aa].filter(la).map(ua)
);
var fa = Object.freeze(
  z.filter((e) => e.localCapable === true)
);
function da(e) {
  return !!(e && e.localCapable === true);
}
function P(e) {
  return String(e ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function ba(e) {
  if (!e || typeof e != "object") return "";
  const t = P(e.title), n = P(e.summary || e.capabilities), i = String(e.iconUrl || "").trim(), r = ['<div class="hub-inference-info-card">'];
  i ? r.push(
    `<div class="hub-inference-info-heading"><img class="hub-inference-info-icon" src="${P(i)}" alt="" width="28" height="28" decoding="async" /><div class="hub-inference-info-title">${t}</div></div>`
  ) : r.push(`<div class="hub-inference-info-title">${t}</div>`), n && r.push(`<p class="hub-inference-info-summary">${n}</p>`);
  const o = String(e.cite || "").trim(), s = String(e.citeUrl || "").trim();
  if (o) {
    const f = P(o);
    s ? r.push(
      `<p class="hub-inference-info-citation"><span class="hub-inference-info-citation-label">Citation:</span> <a class="hub-inference-info-citation-link" href="${P(s)}" target="_blank" rel="noopener noreferrer">${f}</a></p>`
    ) : r.push(
      `<p class="hub-inference-info-citation"><span class="hub-inference-info-citation-label">Citation:</span> ${f}</p>`
    );
  }
  const a = String(e.license || "").trim(), l = String(e.licenseUrl || "").trim(), h = String(e.licenseRestriction || "").trim();
  if (a || h) {
    const f = ['<p class="hub-inference-info-license">'];
    if (a) {
      const m = P(a), g = l ? `<a class="hub-inference-info-license-link" href="${P(l)}" target="_blank" rel="noopener noreferrer">${m}</a>` : m;
      f.push(
        `<span class="hub-inference-info-license-label">License:</span> ${g}`
      );
    }
    a && h && f.push(" "), h && f.push(
      `<span class="hub-inference-info-restriction">${P(h)}</span>`
    ), f.push("</p>"), r.push(f.join(""));
  }
  r.push('<div class="hub-inference-info-links">');
  const c = String(e.githubUrl || "").trim();
  c && r.push(
    `<a class="hub-inference-info-link hub-inference-info-github" href="${P(c)}" target="_blank" rel="noopener noreferrer"><svg class="hub-inference-info-github-icon" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>GitHub</a>`
  );
  const u = String(e.websiteUrl || "").trim();
  u && r.push(
    `<a class="hub-inference-info-link" href="${P(u)}" target="_blank" rel="noopener noreferrer">Website</a>`
  ), s && r.push(
    `<a class="hub-inference-info-link" href="${P(s)}" target="_blank" rel="noopener noreferrer" title="${P(o || "Citation")}">Cite</a>`
  );
  const d = String(e.buyMeACoffeeUrl || "").trim();
  return d && r.push(
    `<a class="hub-inference-info-link hub-inference-info-coffee" href="${P(d)}" target="_blank" rel="noopener noreferrer" title="Buy Me a Coffee"><img class="hub-inference-info-coffee-icon" src="${ca}" alt="" width="12" height="16" decoding="async" />Buy me a coffee</a>`
  ), r.push("</div></div>"), r.join("");
}
function ha(e, t) {
  if (!e || typeof e != "object")
    return "";
  const n = [
    String(e.title || "").trim(),
    String(e.summary || e.capabilities || "").trim(),
    `Location: ${String(e.location || "").trim()}`
  ].filter((a) => a && !a.endsWith(": "));
  t && n.push(`Status: ${t}`);
  const i = String(e.cite || "").trim();
  i && n.push(`Citation: ${i}`);
  const r = String(e.license || "").trim();
  r && n.push(`License: ${r}`);
  const o = String(e.licenseRestriction || "").trim();
  o && n.push(o);
  const s = String(e.githubUrl || "").trim();
  return s && n.push(s), n.join(`
`);
}
function or(e) {
  const t = String(e || "").trim().toUpperCase().replace(/-/g, "_");
  if (!t)
    return null;
  const n = t.replace(/_/g, "");
  return z.find((i) => {
    const r = String(i.product || "").trim().toUpperCase().replace(/-/g, "_");
    return r === t || r.replace(/_/g, "") === n;
  }) || null;
}
function ma(e, t = false) {
  return t ? "Checking\u2026" : e ? e.error ? "Error" : e.online ? `Online${e.job ? ` \xB7 job ${e.job}` : ""}` : "Offline" : "Unknown";
}
function ga(e, t = z) {
  const n = Array.isArray(e) ? [...e] : [], i = (o) => {
    const s = String((o == null ? void 0 : o.id) || "").trim(), a = t.findIndex((l) => l.id === s);
    return a < 0 ? Number.MAX_SAFE_INTEGER : a;
  }, r = (o) => {
    var s;
    return (s = o == null ? void 0 : o.probe) != null && s.online ? 0 : o != null && o.probing || !(o != null && o.probe) ? 1 : 2;
  };
  return n.sort((o, s) => {
    const a = r(o) - r(s);
    return a !== 0 ? a : i(o.server) - i(s.server);
  });
}
function pa(e, t) {
  return String((e == null ? void 0 : e.id) || "").trim() === "totalseg" && (t != null && t.totalSegmentator) ? { totalSegmentator: t.totalSegmentator } : {};
}
function sr(e) {
  return String(e || "").trim().toUpperCase();
}
var ar = Object.freeze({
  ira: (e) => e === "SLICERLIVE-IRA" || e.startsWith("SLICERLIVE-IRA"),
  ohif: (e) => e === "OHIF" || e.startsWith("OHIF"),
  volview: (e) => e === "VOLVIEW" || e.startsWith("VOLVIEW"),
  slicer: (e) => e === "3DSLICER-ID" || e === "SLICER-HUB" || e.startsWith("3DSLICER") || e.startsWith("SLICER-HUB"),
  /** hub-mirror Four-Up LiveScene stream client (not desktop Image Display). */
  hubMirror: (e) => e === "SLICERLIVE-HUB-MIRROR" || e.startsWith("SLICERLIVE-HUB-MIRROR"),
  mhub: (e) => e === "MHUB" || e.startsWith("MHUB") || e === "MHUB_AI" || e.startsWith("MHUB_AI") || e === "MHUBAI" || e.startsWith("MHUBAI"),
  lung: (e) => e === "LUNGSCREENING" || e.startsWith("LUNGSCREENING"),
  totalseg: (e) => e === "TOTALSEG" || e.startsWith("TOTALSEG") || e === "TOTAL_SEGMENTATOR" || e.startsWith("TOTAL_SEGMENTATOR") || e === "TOTAL-SEGMENTATOR" || e.startsWith("TOTAL-SEGMENTATOR"),
  neuro: (e) => e === "NEURO_SEG" || e.startsWith("NEURO_SEG") || e === "NEUROSEG" || e.startsWith("NEUROSEG"),
  dental: (e) => e === "DENTAL_SEG" || e.startsWith("DENTAL_SEG") || e === "DENTALSEG" || e.startsWith("DENTALSEG") || e === "DENTALSEGMENTATOR" || e.startsWith("DENTALSEGMENTATOR"),
  txrv: (e) => e === "TORCHXRAYVISION" || e.startsWith("TORCHXRAYVISION") || e === "TXRV" || e.startsWith("TXRV") || e === "TORCH_XRAY_VISION" || e.startsWith("TORCH_XRAY_VISION") || e === "TORCH-XRAY-VISION" || e.startsWith("TORCH-XRAY-VISION"),
  flexray: (e) => e === "FLEXRAY" || e.startsWith("FLEXRAY") || e === "FLEX_RAY" || e.startsWith("FLEX_RAY"),
  reporting: (e) => e === "RPT" || e.startsWith("RPT-") || e === "CAST-RPT" || e.startsWith("CAST-RPT"),
  classroom: (e) => e === "CLASSROOM" || e.startsWith("CLASSROOM-") || e === "CAST-CLASSROOM" || e.startsWith("CAST-CLASSROOM")
});
function ya(e, t) {
  const n = ar[t];
  return n ? n(sr(e)) : false;
}
function cr(e, t, n) {
  if (!e || typeof e != "object")
    return { online: false, items: [], raw: n, ...t ? { productName: t } : {} };
  const i = (
    /** @type {{ items?: unknown }} */
    e
  ), r = Array.isArray(i.items) ? i.items : [], o = Bt(r, "job");
  return {
    online: jt(e),
    items: r,
    ...o ? { job: o } : {},
    raw: n,
    ...t ? { productName: t } : {}
  };
}
function ur(e) {
  if (!e || typeof e != "object")
    return { online: false, items: [], error: "Empty STATUS result" };
  const t = (
    /** @type {{ ok?: boolean, status?: number, data?: unknown }} */
    e
  );
  if (t.ok === false)
    return {
      online: false,
      items: [],
      error: `HTTP ${t.status ?? "?"}`,
      raw: t.data
    };
  const { responses: n } = Be(t.data);
  for (const i of n) {
    const r = i.data;
    if (!r || typeof r != "object")
      continue;
    const o = rt(i) || void 0, s = cr(r, o, t.data);
    if (s.online || Array.isArray(
      /** @type {{ items?: unknown }} */
      r.items
    ))
      return s;
  }
  return { online: false, items: [], raw: t.data };
}
async function qt(e, t) {
  const n = String((t == null ? void 0 : t.subscriberName) || "").trim();
  if (!(e != null && e.request))
    return { ok: false, status: 0, data: { error: "No Hub client" } };
  if (!n)
    return { ok: false, status: 0, data: { error: "No subscriber name" } };
  const i = String((t == null ? void 0 : t.topic) || "").trim(), r = String((t == null ? void 0 : t.subscriberProductName) || "").trim(), o = String((t == null ? void 0 : t.subscriberActor) || "").trim(), s = String((t == null ? void 0 : t.targetActor) || "*").trim() || "*", a = String((t == null ? void 0 : t.targetProductName) || "").trim();
  return e.request({
    "subscriber.name": n,
    ...r ? { "subscriber.product.name": r } : {},
    ...o ? { "subscriber.actor": o } : {},
    "target.actor": s,
    ...a ? { "target.product.name": a } : {},
    event: {
      "hub.event": Ie("STATUS"),
      ...i ? { "hub.topic": i } : {},
      context: { dataType: "STATUS" }
    }
  });
}
async function Sa(e, t) {
  const n = String((t == null ? void 0 : t.subscriberName) || "").trim();
  if (!(e != null && e.request))
    return { ok: false, status: 0, data: { error: "No Hub client" } };
  if (!n)
    return { ok: false, status: 0, data: { error: "No subscriber name" } };
  const i = String((t == null ? void 0 : t.topic) || "").trim(), r = String((t == null ? void 0 : t.subscriberProductName) || "").trim(), o = String((t == null ? void 0 : t.subscriberActor) || "").trim(), s = String((t == null ? void 0 : t.targetActor) || "ID").trim() || "ID", a = String((t == null ? void 0 : t.targetProductName) || "").trim();
  return e.request({
    "subscriber.name": n,
    ...r ? { "subscriber.product.name": r } : {},
    ...o ? { "subscriber.actor": o } : {},
    "target.actor": s,
    ...a ? { "target.product.name": a } : {},
    event: {
      "hub.event": Ie("LIVESCENE"),
      ...i ? { "hub.topic": i } : {},
      context: { dataType: "LIVESCENE" }
    }
  });
}
async function Ca(e, t) {
  const n = String((t == null ? void 0 : t.subscriberName) || "").trim(), i = String((t == null ? void 0 : t.studyInstanceUID) || "").trim();
  if (!(e != null && e.request))
    return { ok: false, status: 0, data: { error: "No Hub client" } };
  if (!n)
    return { ok: false, status: 0, data: { error: "No subscriber name" } };
  if (!i)
    return { ok: false, status: 0, data: { error: "No studyInstanceUID" } };
  const r = String((t == null ? void 0 : t.topic) || "").trim(), o = String((t == null ? void 0 : t.seriesInstanceUID) || "").trim(), s = String((t == null ? void 0 : t.subscriberProductName) || "").trim(), a = String((t == null ? void 0 : t.subscriberActor) || "").trim(), l = String((t == null ? void 0 : t.targetActor) || "ID").trim() || "ID", h = String((t == null ? void 0 : t.targetProductName) || "").trim();
  return e.request({
    "subscriber.name": n,
    ...s ? { "subscriber.product.name": s } : {},
    ...a ? { "subscriber.actor": a } : {},
    "target.actor": l,
    ...h ? { "target.product.name": h } : {},
    event: {
      "hub.event": Ie("DICOM"),
      ...r ? { "hub.topic": r } : {},
      context: {
        dataType: "DICOM",
        studyInstanceUID: i,
        ...o ? { seriesInstanceUID: o } : {}
      }
    }
  });
}
async function Ea(e, t) {
  const n = String((t == null ? void 0 : t.targetProductName) || "").trim(), i = String((t == null ? void 0 : t.subscriberName) || "").trim();
  if (!(e != null && e.request))
    return { online: false, items: [], error: "No Hub client" };
  if (!i)
    return { online: false, items: [], error: "No subscriber name" };
  if (!n)
    return { online: false, items: [], error: "No target product" };
  try {
    const r = await qt(e, {
      subscriberName: i,
      subscriberProductName: t == null ? void 0 : t.subscriberProductName,
      subscriberActor: t == null ? void 0 : t.subscriberActor,
      topic: t == null ? void 0 : t.topic,
      targetProductName: n
    });
    return ur(r);
  } catch (r) {
    return {
      online: false,
      items: [],
      error: r instanceof Error ? r.message : String(r)
    };
  }
}
function wa(e, t) {
  if (!e) return false;
  const n = or(e);
  return n && n.id === t.id ? true : t.id === "mhub" ? nt(e) : t.id === "totalseg" ? Ze(e) : t.id === "lung" ? Ke(e) : t.id === "neuro" ? Qe(e) : t.id === "dental" ? et(e) : t.id === "txrv" ? tt(e) : t.id === "flexray" ? Dt(e) : false;
}
function lr(e) {
  if (!e || typeof e != "object")
    return z.map((i) => ({
      server: i,
      probe: { online: false, items: [], error: "Empty STATUS result" }
    }));
  const t = (
    /** @type {{ ok?: boolean, status?: number, data?: unknown }} */
    e
  );
  if (t.ok === false) {
    const i = `HTTP ${t.status ?? "?"}`;
    return z.map((r) => ({
      server: r,
      probe: { online: false, items: [], error: i, raw: t.data }
    }));
  }
  const { responses: n } = Be(t.data);
  return z.map((i) => {
    for (const r of n) {
      const o = rt(r);
      if (!wa(o, i)) continue;
      const s = r.data;
      return {
        server: i,
        probe: cr(
          s,
          o || i.product,
          t.data
        )
      };
    }
    return {
      server: i,
      probe: { online: false, items: [], raw: t.data }
    };
  });
}
async function Ta(e, t) {
  const n = String((t == null ? void 0 : t.subscriberName) || "").trim();
  if (!(e != null && e.request) || !n)
    return z.map((i) => ({
      server: i,
      probe: {
        online: false,
        items: [],
        error: e != null && e.request ? "No subscriber name" : "No Hub client"
      }
    }));
  try {
    const i = await qt(e, {
      subscriberName: n,
      subscriberProductName: t == null ? void 0 : t.subscriberProductName,
      subscriberActor: t == null ? void 0 : t.subscriberActor,
      topic: t == null ? void 0 : t.topic,
      targetActor: "*"
    });
    return lr(i);
  } catch (i) {
    const r = i instanceof Error ? i.message : String(i);
    return z.map((o) => ({
      server: o,
      probe: { online: false, items: [], error: r }
    }));
  }
}
async function Na(e, t) {
  const n = String((t == null ? void 0 : t.targetProductName) || "").trim(), i = String((t == null ? void 0 : t.hubEvent) || "").trim(), r = String((t == null ? void 0 : t.subscriberName) || "").trim(), o = String((t == null ? void 0 : t.topic) || "").trim(), s = Array.isArray(t == null ? void 0 : t.files) ? t.files : [];
  if (!(e != null && e.publish))
    return { ok: false, detail: "No Hub client" };
  if (!r)
    return { ok: false, detail: "No subscriber name" };
  if (!o)
    return { ok: false, detail: "No Hub topic" };
  if (!n)
    return { ok: false, detail: "No target product" };
  if (!i)
    return { ok: false, detail: "No hub.event" };
  if (!s.length)
    return { ok: false, detail: "No volume files" };
  const a = String((t == null ? void 0 : t.subscriberProductName) || "").trim(), l = String((t == null ? void 0 : t.subscriberActor) || "").trim(), h = t != null && t.contextExtras && typeof t.contextExtras == "object" ? t.contextExtras : {};
  try {
    const c = await e.publish({
      "subscriber.name": r,
      ...a ? { "subscriber.product.name": a } : {},
      ...l ? { "subscriber.actor": l } : {},
      "target.product.name": n,
      event: {
        "hub.topic": o,
        "hub.event": i,
        context: {
          files: s.map((d) => ({
            url: String(d.url || "").trim(),
            fileName: String(d.fileName || "volume").trim() || "volume"
          })),
          ...h
        }
      }
    }), u = !!(c && /** @type {{ ok?: boolean }} */
    c.ok !== false);
    return {
      ok: u,
      detail: u ? "published" : `publish status ${/** @type {{ status?: number }} */
      (c == null ? void 0 : c.status) ?? "?"}`,
      response: c
    };
  } catch (c) {
    return {
      ok: false,
      detail: c instanceof Error ? c.message : String(c)
    };
  }
}
var La = 1e4;
var ln = 5e3;
var fn = {
  config: {
    hub: {},
    session: {},
    productName: void 0,
    productVersion: void 0,
    callbackUrl: void 0,
    autoStart: false,
    autoReconnect: false,
    preserveSessionTopicFromToken: false
  },
  hub: null,
  session: null,
  reconnectInterval: null,
  onMessageCallback: null,
  onConnectionStateChangeCallback: null
};
function va(e, t) {
  t.classHierarchy.push("HubClient");
  function n(c, u) {
    t.onConnectionStateChangeCallback && t.onConnectionStateChangeCallback(c, u);
  }
  function i() {
    const c = t.session.productName || t.config.productName || Et;
    return Hr(c);
  }
  function r() {
    const c = t.hub.authorization_endpoint;
    if (typeof c == "string" && c.trim())
      return c.trim();
    try {
      return `${new URL(t.hub.token_endpoint).origin}/oauth/authorize`;
    } catch {
      return "";
    }
  }
  function o() {
    console.debug("HubClient: websocket is closed."), t.hub.resubscribeRequested = true, n("disconnected");
  }
  const s = Ci(() => ({
    hubEndpoint: t.hub.hub_endpoint,
    accessToken: t.hub.token
  }));
  function a(c) {
    try {
      const u = JSON.parse(c);
      if (u["hub.mode"])
        return;
      const d = u.event;
      if (!d || d["hub.event"] === "heartbeat" || u.id === t.hub.lastPublishedMessageID)
        return;
      t.onMessageCallback && t.onMessageCallback(u);
    } catch (u) {
      console.warn("HubClient: websocket processing error:", u);
    }
  }
  async function l() {
    t.hub.resubscribeRequested && t.hub.subscribed && t.config.autoReconnect ? (console.debug("HubClient: Try to resubscribe"), t.hub.resubscribeRequested = false, await e.subscribe() !== 202 && (t.hub.resubscribeRequested = true)) : !t.hub.subscribed && t.hub.resubscribeRequested && (t.hub.resubscribeRequested = false);
  }
  e.onMessage = (c) => {
    t.onMessageCallback = c;
  }, e.onConnectionStateChange = (c) => {
    t.onConnectionStateChangeCallback = c;
  }, e.delete = Ur(() => {
    t.reconnectInterval && (clearInterval(t.reconnectInterval), t.reconnectInterval = null), e.unsubscribe();
  }, e.delete), e.getHubConfig = () => {
    const c = t.hub;
    return {
      name: c.name,
      friendlyName: c.friendlyName,
      version: c.version,
      hub_endpoint: c.hub_endpoint,
      authorization_endpoint: c.authorization_endpoint,
      token_endpoint: c.token_endpoint,
      client_id: c.client_id,
      client_secret: c.client_secret
    };
  }, e.getSessionConfig = () => {
    const c = t.session;
    return {
      subscriberName: c.subscriberName,
      productName: c.productName,
      productVersion: c.productVersion,
      actors: c.actors,
      topic: c.topic,
      events: c.events,
      lease: c.lease,
      userName: c.userName,
      defaultTargetActor: c.defaultTargetActor
    };
  }, e.getConnectionState = () => {
    const c = t.hub;
    return {
      token: c.token,
      subscribed: c.subscribed,
      resubscribeRequested: c.resubscribeRequested,
      websocket: c.websocket,
      lastPublishedMessageID: c.lastPublishedMessageID
    };
  }, e.setTopic = (c) => {
    console.debug("HubClient: setting topic to", c), t.session.topic = c;
  }, e.setToken = (c) => {
    t.hub.token = c;
  }, e.setSubscriberName = (c) => {
    t.session.subscriberName = c;
  }, e.setUserName = (c) => {
    t.session.userName = c || "";
  }, e.authenticate = async () => {
    const c = r();
    if (!c)
      throw new Error(
        "HubClient.authenticate: no authorization_endpoint or token_endpoint configured."
      );
    try {
      const g = new URL(c);
      console.debug(
        "HubClient: Authorizing at:",
        `${g.origin}${g.pathname}`
      );
    } catch {
      console.debug("HubClient: Authorizing at hub");
    }
    const u = t.session.productName || t.config.productName || "UNKNOWN", d = new URLSearchParams();
    t.hub.lastIdToken ? d.append("id_token", t.hub.lastIdToken) : t.session.userName && d.append("user_name", t.session.userName), d.append("client_product_name", u), t.session.topic && d.append("topic", t.session.topic);
    let f;
    try {
      f = await fetch(c, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: d
      });
    } catch (g) {
      const p = g instanceof Error ? g.message : String(g);
      throw console.error("HubClient: Exception during authenticate:", p), g;
    }
    if (f.status !== 200) {
      const g = await f.text().catch(() => "");
      throw console.error(
        "HubClient: Authenticate failed. Status:",
        f.status,
        g
      ), new Error(
        `HubClient.authenticate failed (HTTP ${f.status})`
      );
    }
    const m = await f.json();
    return typeof m.user_name == "string" && m.user_name && (t.session.userName = m.user_name), {
      user_name: m.user_name || "",
      code: m.code || "",
      expires_in: typeof m.expires_in == "number" ? m.expires_in : void 0
    };
  }, e.getToken = async (c) => {
    if (typeof c != "string" || !c)
      return console.error(
        "HubClient.getToken: code is required (call authenticate() first)."
      ), false;
    try {
      const f = new URL(t.hub.token_endpoint);
      console.debug(
        "HubClient: Exchanging code at:",
        `${f.origin}${f.pathname}`
      );
    } catch {
      console.debug("HubClient: Exchanging code at hub");
    }
    const u = t.session.productName || t.config.productName || "UNKNOWN", d = new URLSearchParams();
    d.append("grant_type", "authorization_code"), d.append("code", c), d.append("client_id", t.hub.client_id || ""), d.append("client_secret", t.hub.client_secret || ""), d.append("client_product_name", u);
    try {
      const f = await fetch(t.hub.token_endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: d
      });
      if (f.status === 200) {
        const m = await f.json();
        return typeof m.access_token == "string" && m.access_token && (t.hub.token = m.access_token), typeof m.id_token == "string" && m.id_token && (t.hub.lastIdToken = m.id_token), m.topic && typeof m.topic == "string" && (t.config.preserveSessionTopicFromToken || e.setTopic(m.topic), t.config.autoStart && e.subscribe()), !!t.hub.token;
      }
      return await f.text(), console.error(
        "HubClient: Error getting token. Status:",
        f.status
      ), false;
    } catch (f) {
      const m = f instanceof Error ? f.message : String(f);
      return console.error("HubClient: Exception getting token:", m), false;
    }
  }, e.subscribe = async (c = false) => {
    const u = t.session.topic && t.session.topic.trim();
    if (!u)
      return console.warn(
        "HubClient: Error. subscription not sent. No topic defined."
      ), "error: topic not defined";
    if ((!t.session.subscriberName || !String(t.session.subscriberName).trim()) && (t.session.subscriberName = wt(
      t.session.productName || t.config.productName || "UNKNOWN"
    )), !t.hub.token)
      return console.warn(
        "HubClient: Error. subscription not sent. No token available."
      ), "error: no token";
    const d = tn(t.config.callbackUrl), f = nn(
      "subscribe",
      {
        ...t.session,
        topic: u,
        subscriberName: t.session.subscriberName
      },
      d,
      t.config.productVersion
    ), m = {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        Authorization: `Bearer ${t.hub.token}`
      },
      body: f,
      signal: typeof AbortSignal < "u" && AbortSignal.timeout ? AbortSignal.timeout(ln) : void 0
    };
    try {
      n("connecting");
      const g = await fetch(t.hub.hub_endpoint, m);
      if (g.status === 202) {
        t.hub.resubscribeRequested = false;
        let p;
        try {
          p = await g.json();
        } catch (w) {
          t.hub.subscribed = false;
          const N = w instanceof Error ? w.message : String(w);
          return console.error(
            "HubClient: Subscribe 202 body was not valid JSON:",
            N
          ), n("error"), 0;
        }
        const S = p["hub.channel.endpoint"];
        if (!S || typeof S != "string")
          return t.hub.subscribed = false, console.error(
            "HubClient: Subscribe 202 missing hub.channel.endpoint"
          ), n("error"), 0;
        let A = S;
        try {
          const w = new URL(t.hub.hub_endpoint), N = new URL(S), E = w.protocol === "https:" ? "wss:" : "ws:";
          A = S.replace(
            N.origin,
            `${E}//${w.host}`
          );
        } catch {
        }
        if (t.hub.websocket) {
          try {
            t.hub.websocket.removeEventListener("close", o), t.hub.websocket.close();
          } catch {
          }
          t.hub.websocket = null;
        }
        let U;
        try {
          U = new WebSocket(A);
        } catch (w) {
          t.hub.subscribed = false;
          const N = w instanceof Error ? w.message : String(w);
          return console.error(
            "HubClient: Failed to open WebSocket after subscribe:",
            N
          ), n("error"), 0;
        }
        return t.hub.subscribed = true, t.hub.websocket = U, t.hub.websocket.onopen = function() {
          this.send(
            JSON.stringify({
              "hub.channel.endpoint": A
            })
          ), n("connected");
        }, t.hub.websocket.addEventListener("message", (w) => {
          typeof w.data == "string" && a(w.data);
        }), t.hub.websocket.addEventListener("close", o), t.hub.websocket.onerror = function() {
          console.warn("HubClient: Error reported on websocket"), n("error");
        }, g.status;
      }
      if (g.status === 401) {
        if (console.warn(
          "HubClient: Subscription response 401 - Token refresh needed."
        ), c)
          return g.status;
        try {
          const { code: p } = await e.authenticate();
          if (p && await e.getToken(p))
            return await e.subscribe(true);
        } catch (p) {
          const S = p instanceof Error ? p.message : String(p);
          console.error(
            "HubClient: Token refresh after 401 failed:",
            S
          );
        }
      } else
        console.error(
          "HubClient: Subscription rejected by hub. Status:",
          g.status
        );
      return g.status;
    } catch (g) {
      t.hub.subscribed = false;
      const p = g instanceof Error ? g.message : String(g);
      return console.error("HubClient: Exception subscribing to the hub:", p), 0;
    }
  }, e.unsubscribe = async () => {
    t.hub.subscribed = false, t.hub.resubscribeRequested = false;
    const c = tn(t.config.callbackUrl), u = nn(
      "unsubscribe",
      t.session,
      c,
      t.config.productVersion
    );
    try {
      (await fetch(t.hub.hub_endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
          Authorization: `Bearer ${t.hub.token}`
        },
        body: u,
        signal: typeof AbortSignal < "u" && AbortSignal.timeout ? AbortSignal.timeout(ln) : void 0
      })).status === 202 && console.debug(
        "HubClient: Unsubscribe successfully from hub",
        t.hub.name
      );
    } catch (d) {
      const f = d instanceof Error ? d.message : String(d);
      console.warn("HubClient: Error unsubscribing from the hub.", f);
    }
    t.hub.websocket && (t.hub.websocket.close(), t.hub.websocket = null), n("disconnected");
  };
  function h(c, u = t.hub) {
    const d = { ...c, timestamp: (/* @__PURE__ */ new Date()).toJSON() };
    d.id = Re(i()), u.lastPublishedMessageID = d.id;
    const f = t.session.subscriberName && t.session.subscriberName.trim(), m = t.session.productName && t.session.productName.trim();
    if (f && d["subscriber.name"] === void 0 && (d["subscriber.name"] = f), m && d["subscriber.product.name"] === void 0 && (d["subscriber.product.name"] = m), d.event && !d.event["hub.topic"] && (d.event["hub.topic"] = t.session.topic), d["target.actor"] === void 0 && t.session.defaultTargetActor && !String(
      d["target.subscriber.name"] || ""
    ).trim()) {
      const p = Oe(
        t.session.defaultTargetActor
      );
      p && (d["target.actor"] = p);
    }
    return d;
  }
  e.fetchPayload = (c) => s.fetchPayload(c), e.fetchAllPayloads = (c) => s.fetchAllPayloads(c), e.hasPendingPayload = (c) => s.hasPendingPayload(c), e.publishMultipart = async (c, u, d = t.hub) => {
    let f = h(c, d);
    const m = await Fe(u);
    f = ct(f);
    let g = await en(f);
    if (!g.length) {
      const S = (f.event && typeof f.event["hub.event"] == "string" ? f.event["hub.event"] : "") === "dicom-send" ? "dicom-send.dcm" : "nifti-send.nii.gz";
      f.event.context = {
        files: [
          {
            data: m,
            fileName: Rr(f, S),
            mimeType: "application/octet-stream",
            byteLength: m.byteLength
          }
        ]
      }, g = [m];
    }
    return e.publishBinaryBatch(f, g, d);
  }, e.publishBinaryBatch = async (c, u, d = t.hub) => {
    const f = ct(
      h(c, d)
    );
    return Br({
      msg: f,
      fileBytesList: u,
      hub: d,
      messageIdPrefix: i
    });
  }, e.publish = async (c, u = t.hub) => {
    let d = h(c, u);
    if (d = ct(d), Mr(d)) {
      const f = await en(d);
      if (f.length)
        return e.publishBinaryBatch(d, f, u);
    }
    try {
      return await fetch(u.hub_endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${u.token}`
        },
        body: JSON.stringify(d)
      });
    } catch (f) {
      const m = f instanceof Error ? f.message : String(f);
      return console.debug("HubClient:", m), null;
    }
  }, e.publishNiftiMultipart = async (c, u, d) => e.publishBinaryBatch(c, [u], d), e.request = async (c = {}) => {
    const u = String(c["subscriber.name"] || "").trim();
    if (!u)
      throw new Error('HubClient.request: "subscriber.name" is required.');
    const d = t.hub, f = d.token && d.token.trim();
    if (!f)
      throw new Error(
        "HubClient.request: token is required (call authenticate() then getToken(code) first)."
      );
    const m = (d.hub_endpoint || "").replace(/\/+$/, ""), g = {
      "subscriber.name": u,
      id: c.id && String(c.id).trim() || Re(i()),
      timestamp: c.timestamp && String(c.timestamp).trim() || (/* @__PURE__ */ new Date()).toJSON()
    };
    if (c.event && typeof c.event == "object") {
      g.event = { ...c.event };
      const H = t.session.topic && String(t.session.topic).trim();
      H && !g.event["hub.topic"] && (g.event["hub.topic"] = H);
    } else
      throw new Error(
        'HubClient.request: "event" with hub.event is required.'
      );
    const p = g.event["hub.event"];
    if (!p || !String(p).trim())
      throw new Error(
        'HubClient.request: event["hub.event"] must be a *-request event name.'
      );
    c["subscriber.actor"] && String(c["subscriber.actor"]).trim() && (g["subscriber.actor"] = String(c["subscriber.actor"]).trim());
    const S = Object.prototype.hasOwnProperty.call(
      c,
      "target.actor"
    );
    let A = S ? Oe(c["target.actor"]) : void 0;
    A === void 0 && !S && t.session.defaultTargetActor && (A = Oe(t.session.defaultTargetActor)), A !== void 0 && (g["target.actor"] = A);
    const U = c["target.product.name"] !== void 0 ? c["target.product.name"] : c.targetProductName, w = Nn(U);
    w !== void 0 && (g["target.product.name"] = w);
    const N = await fetch(m, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${f}`
      },
      body: JSON.stringify(g)
    });
    let E;
    if ((N.headers.get("content-type") || "").includes("application/json"))
      try {
        E = await N.json();
      } catch {
        E = "";
      }
    else
      E = await N.text();
    return { ok: N.ok, status: N.status, data: E };
  }, e.sendCastRequestResponse = (c, u, d, f) => {
    if (!t.hub.websocket || typeof WebSocket > "u" || t.hub.websocket.readyState !== WebSocket.OPEN)
      return;
    let m = "";
    if (typeof u == "string" ? m = u.trim() : u != null && (m = String(u).trim()), !m) {
      console.error(
        "HubClient.sendCastRequestResponse requires a non-empty dataType."
      );
      return;
    }
    const g = pn(m), p = {
      timestamp: (/* @__PURE__ */ new Date()).toJSON(),
      id: Re(i()),
      "subscriber.name": t.session.subscriberName || void 0,
      "subscriber.product.name": t.session.productName || void 0,
      event: {
        "hub.topic": f || t.session.topic,
        "hub.event": g,
        context: {
          id: c,
          dataType: m,
          data: d
        }
      }
    };
    Array.isArray(t.session.actors) && t.session.actors.length > 0 && (p.actor = t.session.actors[0]), t.hub.websocket.send(JSON.stringify(p));
  }, t.config.autoReconnect && (t.reconnectInterval = setInterval(
    l,
    La
  ));
}
function fr(e, t, n = {}) {
  Object.assign(t, fn, n), t.config = { ...fn.config, ...n }, t.session = {
    ...fi(),
    ...t.config.session || {}
  }, (!t.session.subscriberName || !String(t.session.subscriberName).trim()) && (t.session.subscriberName = wt(
    t.session.productName || t.config.productName || "UNKNOWN"
  )), t.hub = {
    ...li(),
    ...t.config.hub || {},
    ...di()
  }, Ar(e, t), Ir(e, t, ["config"]), va(e, t);
}
var Aa = xr(fr, "HubClient");
var Ia = {
  newInstance: Aa,
  extend: fr,
  applyCastPublishEnvelopeFields: To,
  batchContextFiles: Xn,
  BINARY_PLACEHOLDER: Me,
  buildDicomwebImagingStudyOpenContext: Ji,
  buildDicomUrlImagingStudyOpenContext: Xi,
  buildFilesImagingStudyOpenContext: zn,
  buildIdcImagingStudyOpenContext: Zi,
  buildLocalDicomImagingStudyOpenContext: Yi,
  buildNiftiUrlImagingStudyOpenContext: Ki,
  castMessageHasPendingFilePayloads: Kn,
  buildCastConferenceView: xt,
  castConferenceSelfStatusLabel: $t,
  castConferencePlaceStatusLabel: _n,
  castConferenceLeaderPlace: Ht,
  castConferenceParticipantRowStatus: Rn,
  resolveHubConferenceHeaderActions: Mn,
  applyCastConferenceHeaderActions: Pi,
  renderCastConferencePresence: Ri,
  openHubConferenceParticipantsPopup: Mi,
  closeCastConferenceParticipantsPopup: W,
  isHubConferenceParticipantsPopupOpen: Oi,
  HUB_CONFERENCE_EXIT_ACK_MS: An,
  HUB_CONFERENCE_POLL_MS: Hi,
  HUB_CONFERENCE_POPUP_SIZE: Ti,
  HUB_REPORTING_WINDOW_NAME: Ni,
  HUB_CLASSROOM_WINDOW_NAME: Li,
  HUB_WORKLIST_WINDOW_NAME: vi,
  clearHubDialogNearAnchor: kt,
  placeHubDialogNearAnchor: Pn,
  HUB_CONFERENCE_TITLE_PRESETS: In,
  HUB_CONFERENCE_STRINGS_EN: bt,
  HUB_DEFAULT_SUBSCRIBER_ACTOR: Ye,
  createHubConferenceDialog: Di,
  confirmHubConferenceTakeover: qi,
  HUB_IMAGE_DISPLAY_ACTOR: ht,
  createHubConferenceInviteController: ji,
  createHubConferenceInviteDialog: Dn,
  interpolateCastConferenceString: ye,
  mergeCastConferenceStrings: ee,
  parseConferenceInviteFromMessage: Fn,
  shouldShowConferenceInvite: Bi,
  showHubInfoToast: Wi,
  HUB_DICOMWEB_ROOT: Ot,
  HUB_ENVELOPE_ANY: Q,
  HUB_IDENTIFIER_DICOM_UID: $e,
  HUB_IDENTIFIER_IDC: Ge,
  HUB_IDENTIFIER_IDC_SOURCE_BUCKET: Xe,
  HUB_IDENTIFIER_NIFTI_FILENAME: Rt,
  HUB_IDENTIFIER_NIFTI_URL: _t,
  HUB_IDENTIFIER_OHIF_MODE: qe,
  HUB_IDENTIFIER_VOLVIEW_SAMPLE_ID: je,
  HUB_IDENTIFIER_WORKLIST_SAMPLE_ID: fe,
  HUB_IMAGING_STUDY_OPEN_PROFILE: He,
  HUB_OPEN_MODE: de,
  HUB_OPEN_MODE_DICOMWEB: Te,
  HUB_OPEN_MODE_DICOM_URL: We,
  HUB_OPEN_MODE_FILES: ce,
  HUB_OPEN_MODE_IDC: ze,
  HUB_OPEN_MODE_LOCAL_DICOM: Ve,
  HUB_CONFERENCE_ICON_CLASS: Tn,
  HUB_CONFERENCE_ICON_SVG: ai,
  HUB_RADIO_ICON_CLASS: Sn,
  HUB_RADIO_ICON_SVG: Cn,
  HUB_RADIO_SLASH_CLASS: wn,
  HUB_RADIO_STATUS_ICON_CLASS: En,
  castRadioStatusMarkup: ci,
  castRadioStatusVisual: ui,
  collatedResponsesFromRequestResult: jr,
  conferenceHostTopic: X,
  conferenceSceneLeaderTopic: vt,
  countEventContextObjects: Ft,
  createHubConference: Hn,
  joinHubConference: $n,
  dataTypeFromEventName: oi,
  decodeBase64ToArrayBuffer: Gn,
  DEFAULT_CAST_PUBLISH_ENVELOPE_FIELDS: Eo,
  deleteHubConference: Ut,
  DICOMSR_UPDATE: ei,
  DICOM_SEND: Jr,
  ensureCastSubscribeEvents: si,
  ensureHubSubscribeEvents: yn,
  extractDicomSeriesUid: oe,
  extractDicomStudyUid: se,
  extractDicomwebRoot: mt,
  extractFilePayloadsForEvent: mo,
  extractIdcSeriesUid: gt,
  extractIdcSourceBucket: pt,
  extractIdentifierValue: j,
  extractImagingStudyFiles: ue,
  extractNiftiDownloadUrl: Ne,
  extractNiftiFilename: Mt,
  extractOhifMode: Bn,
  extractOpenMode: qn,
  extractStudyContextItem: le,
  extractStudyUIDFromResource: eo,
  extractVolviewSampleId: jn,
  fetchHubConferences: we,
  fetchHubConferenceTopics: xn,
  filePayloadToArrayBuffer: Yn,
  filePayloadToFile: go,
  findActiveHubConference: Ee,
  generateSubscriberName: wt,
  getActorKeyword: Vn,
  getHubEventLower: ke,
  getInboundTargetActorKeyword: to,
  HEARTBEAT: ni,
  httpUrlFromHubEndpoint: Ue,
  IDC_CLAUDE_SEND: Kr,
  IMAGINGSTUDY_CLOSE: zr,
  IMAGINGSTUDY_OPEN: Wr,
  NIFTI_SEND: Yr,
  PNG_SEND: Zr,
  SCENE_UPDATE: Vr,
  STATUS_UPDATE: dt,
  SUBSCRIPTION_REMOVED: ti,
  ANNOTATION_DELETE: Xr,
  ANNOTATION_UPDATE: Gr,
  CONFERENCE_CHAT: Qr,
  isHubConferenceHost: At,
  isHubConferenceParticipant: It,
  isHubConferenceSceneLeader: Un,
  isHubEndpointInCloud: gn,
  isCastBinaryEvent: $r,
  isHubBinaryEvent: ve,
  isRequestEvent: ri,
  isResponseEvent: ii,
  isRunningInCloud: mn,
  isStatusPayloadOnline: jt,
  isStatusRequestDataType: ko,
  isTotalSegmentatorProduct: Ze,
  isLungScreeningProduct: Ke,
  isNeuroSegProduct: Qe,
  isDentalSegProduct: et,
  isTorchXrayVisionProduct: tt,
  isFlexrayProduct: Dt,
  isMhubProduct: nt,
  isInferenceProduct: vo,
  lungScreeningAvailableFromStatusResponses: Io,
  neuroSegAvailableFromStatusResponses: Uo,
  dentalSegAvailableFromStatusResponses: xo,
  torchXrayVisionAvailableFromStatusResponses: Ho,
  mhubAvailableFromStatusResponses: $o,
  normalizeProductToken: Ct,
  HUB_INFERENCE_SERVERS: z,
  LOCAL_AI_SERVERS: fa,
  buildInferenceRunContextExtras: pa,
  findInferenceServerByProduct: or,
  inferenceProbeAvailabilityLabel: ma,
  inferenceServerInfoText: ha,
  isLocalCapableInferenceServer: da,
  orderInferenceServerProbeRows: ga,
  renderInferenceServerInfoCardHtml: ba,
  HUB_PRODUCT_MATCHERS: ar,
  matchesHubProduct: ya,
  normalizeHubProductName: sr,
  parseProductStatusProbe: ur,
  publishCastUrlSend: Na,
  requestHubDicom: Ca,
  requestHubLiveScene: Sa,
  requestHubProductStatus: Ea,
  requestHubStatus: qt,
  mapInferenceServersFromStatusResult: lr,
  probeCastInferenceServers: Ta,
  cloneContextArray: no,
  messageEventContext: ro,
  messageSubscriberName: io,
  messageProductName: oo,
  messageActor: so,
  parseStatusUpdateMessage: ao,
  normalizeConferenceParticipants: Lt,
  normalizeDataType: Nt,
  normalizeImagingStudyContext: Pt,
  normalizeOptionalEnvelopeField: Le,
  normalizeStudyUID: yt,
  openCastHubPopup: xi,
  castReportingPopupPlacement: Ai,
  hubReportingPopupFeatures: Ii,
  placeHubPopupWindow: Ui,
  parseCollatedRequestResult: Be,
  productNameFromStatusResponseItem: rt,
  REQUEST_SUFFIX: Se,
  requestEventFor: Ie,
  resolveHubConferenceClientUrl: wi,
  resolveHubConferenceState: ki,
  resolveHubConferenceView: kn,
  resolveHubFileMessage: No,
  resolveHubAdminUrl: Ei,
  resolveHubPublishEnvelopeFields: wo,
  resolveImagingStudyOpenPlan: Qi,
  resolveTargetActorForWire: Oe,
  resolveTargetProductNameForWire: Nn,
  responseEventFor: pn,
  RESPONSE_SUFFIX: Ce,
  sanitizeCastMessageForDisplay: Co,
  selectFirstMatchingHubKey: qr,
  statusItemValue: Bt,
  stringifyForLog: St,
  summarizeInboundCastMessage: So,
  summarizeOutboundCastPublish: yo,
  TOTAL_SEGMENTATOR_PRODUCT_ALIASES: Qn,
  LUNG_SCREENING_PRODUCT_ALIASES: er,
  NEURO_SEG_PRODUCT_ALIASES: tr,
  DENTAL_SEG_PRODUCT_ALIASES: nr,
  TORCHXRAYVISION_PRODUCT_ALIASES: rr,
  MHUB_PRODUCT_ALIASES: ir,
  totalSegmentatorAvailableFromStatusResponses: Ao,
  transferHubConferenceLead: $i
};

// render/cast-transport.ts
var LIVESYNC_CONTEXT_KEY = "livesync";
function livesyncContextFromWire(wire) {
  return [{ key: LIVESYNC_CONTEXT_KEY, resource: wire }];
}
function wireFromSceneUpdateContext(context) {
  if (Array.isArray(context)) {
    for (const item of context) {
      if (!item || typeof item !== "object") continue;
      const row = item;
      if (String(row.key || "").toLowerCase() !== LIVESYNC_CONTEXT_KEY) continue;
      if (row.resource != null) return row.resource;
    }
    return null;
  }
  if (context && typeof context === "object") {
    const obj = context;
    if (obj[LIVESYNC_CONTEXT_KEY] != null) return obj[LIVESYNC_CONTEXT_KEY];
    if (obj.livesync != null) return obj.livesync;
  }
  return null;
}
var CastTransport = class {
  constructor(cast) {
    this.cast = cast;
  }
  onMessage;
  onOpen;
  onClose;
  open = false;
  closed = false;
  get isOpen() {
    return this.open && this.cast.isConnected();
  }
  connect() {
    this.closed = false;
    if (this.cast.isConnected()) {
      this.open = true;
      this.onOpen?.();
      return;
    }
    this.open = false;
  }
  /** Notify LiveSync that the Cast bind socket is ready (call after subscribe succeeds). */
  notifyOpen() {
    if (this.closed) return;
    this.open = true;
    this.onOpen?.();
  }
  /** Notify LiveSync that Cast disconnected. */
  notifyClose() {
    const wasOpen = this.open;
    this.open = false;
    if (wasOpen) this.onClose?.();
  }
  /** Deliver one inbound LiveSync wire message from a Cast scene-update. */
  deliver(wire) {
    if (this.closed || !this.open) return;
    this.onMessage?.(wire);
  }
  send(msg) {
    if (!this.isOpen) return;
    void Promise.resolve(this.cast.publishSceneUpdate(msg)).catch((err) => {
      console.warn("[cast-transport] scene-update publish failed", err);
    });
  }
  close() {
    this.closed = true;
    this.open = false;
  }
};

// render/demos/hub-mirror/hub-mirror-cast.ts
var LOG = "[hub-mirror-cast]";
var HUB_CREDENTIALS = {
  client_id: "client_id_3d_Slicer",
  client_secret: "client_secret_3d_Slicer"
};
var LOCAL_HUB = {
  name: "local",
  version: "1.0",
  hub_endpoint: "http://127.0.0.1:2018/api/hub",
  authorization_endpoint: "http://127.0.0.1:2018/oauth/authorize",
  token_endpoint: "http://127.0.0.1:2018/oauth/token",
  ...HUB_CREDENTIALS
};
var ID_ACTOR = "ID";
var PRODUCT_NAME = "SLICERLIVE-HUB-MIRROR";
var HUB_BLOBS_EVENT = "HubBlobs";
function isServedFromSlicerHubMount(location2 = globalThis.location) {
  const path = String(location2?.pathname || "");
  if (!path.startsWith("/hub-mirror")) return false;
  if (mn(location2)) return true;
  const port = location2.port || (location2.protocol === "https:" ? "443" : "80");
  return port === "2018";
}
function shouldUseSameOriginHub(location2 = globalThis.location) {
  return isServedFromSlicerHubMount(location2) || mn(location2);
}
function resolveHubConfig(location2 = globalThis.location) {
  if (!shouldUseSameOriginHub(location2)) {
    return LOCAL_HUB;
  }
  const origin = location2.origin;
  return {
    name: "hub",
    version: "1.0",
    hub_endpoint: `${origin}/api/hub`,
    authorization_endpoint: `${origin}/oauth/authorize`,
    token_endpoint: `${origin}/oauth/token`,
    ...HUB_CREDENTIALS
  };
}
function installHubBlobFetch(hashToBytes) {
  setBlobFetch(async (url) => {
    const u = String(url || "");
    let hash = "";
    try {
      const path = new URL(u, "http://local/").pathname;
      const base = path.split("/").pop() || "";
      if (base.startsWith("sha256-")) {
        hash = base.replace(/\.bin$/i, "");
      }
    } catch {
    }
    if (!hash && u.startsWith("sha256-")) hash = u.replace(/\.bin$/i, "");
    if (!hash) {
      const m = u.match(/(sha256-[a-fA-F0-9]+)/);
      if (m) hash = m[1];
    }
    const bytes = hash ? hashToBytes.get(hash) : void 0;
    if (bytes) {
      return new Response(bytes, {
        status: 200,
        headers: { "Content-Type": "application/octet-stream" }
      });
    }
    const synthetic = u.includes("hub-mirror.local") || hash && hash.startsWith("sha256-");
    if (synthetic) {
      console.warn(
        `${LOG} missing blob cache entry for`,
        hash || u,
        `(cached=${hashToBytes.size})`
      );
      return new Response(null, { status: 404, statusText: "Blob not in hub cache" });
    }
    return fetch(url);
  });
}
function contentHashFromFileName(fileName) {
  let name = String(fileName || "").trim();
  if (name.toLowerCase().endsWith(".bin")) {
    name = name.slice(0, -4);
  }
  return name;
}
function ingestBlobFiles(hashToBytes, files) {
  let n = 0;
  for (const entry of files) {
    const hash = contentHashFromFileName(String(entry.fileName || ""));
    if (!hash.startsWith("sha256-")) continue;
    const data = entry.data;
    let bytes = null;
    if (data instanceof ArrayBuffer) bytes = new Uint8Array(data);
    else if (data instanceof Uint8Array) bytes = data;
    if (!bytes || !bytes.byteLength) continue;
    hashToBytes.set(hash, bytes);
    n++;
  }
  return n;
}
function isHubBlobsWire(wire) {
  if (!wire || typeof wire !== "object") return false;
  const ev = String(wire.event || "").trim();
  return ev === HUB_BLOBS_EVENT;
}
async function sleep(ms2) {
  await new Promise((r) => globalThis.setTimeout(r, ms2));
}
async function fetchHubBlobFilesWithRetry(client, files, opts = {}) {
  const attempts = opts.attempts ?? 8;
  const baseDelayMs = opts.baseDelayMs ?? 400;
  const pending = files.filter((f) => {
    const ids = f.payloadIds;
    return Array.isArray(ids) && ids.length > 0 && f.data == null;
  });
  if (!pending.length) return files;
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try {
      const synthetic = {
        event: {
          "hub.event": "ImagingStudy-open",
          context: { files: pending.map((f) => ({ ...f })) }
        }
      };
      const resolved = await No(
        client,
        synthetic
      );
      const event = resolved.event || synthetic.event;
      return Xn(event);
    } catch (err) {
      lastErr = err;
      const delay = baseDelayMs * Math.pow(2, Math.min(i, 4));
      console.warn(
        `${LOG} payload fetch attempt ${i + 1}/${attempts} failed; retry in ${delay}ms`,
        err
      );
      await sleep(delay);
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr || "payload fetch failed"));
}
async function connectHubMirrorCast() {
  const params = new URLSearchParams(globalThis.location.search);
  let topic = params.get("topic")?.trim() || sessionStorage.getItem("hub-mirror.cast.topic")?.trim() || "";
  let token = params.get("id-token")?.trim() || sessionStorage.getItem("hub-mirror.cast.token")?.trim() || "";
  if (!topic || !token) {
    throw new Error(
      "hub-mirror requires ?topic= and ?id-token= (open from worklist 3D Slicer web mirror button)"
    );
  }
  try {
    sessionStorage.setItem("hub-mirror.cast.topic", topic);
    sessionStorage.setItem("hub-mirror.cast.token", token);
  } catch {
  }
  try {
    const url = new URL(globalThis.location.href);
    url.searchParams.delete("id-token");
    globalThis.history.replaceState({}, "", url.toString());
  } catch {
  }
  const hub = resolveHubConfig();
  const subscriberName = wt("HubMirror");
  const hashToBytes = /* @__PURE__ */ new Map();
  installHubBlobFetch(hashToBytes);
  let resolveReady;
  let readySettled = false;
  const ready = new Promise((r) => {
    resolveReady = () => {
      if (readySettled) return;
      readySettled = true;
      r();
    };
  });
  const client = Ia.newInstance({
    hub,
    session: {
      subscriberName,
      productName: PRODUCT_NAME,
      productVersion: "0.1",
      actors: [ID_ACTOR],
      topic: topic || void 0,
      // Scene control + HubBlobs payload refs only — no imagingstudy-* subscription.
      events: [Vr],
      lease: 7200,
      defaultTargetActor: "ID"
    },
    callbackUrl: `${globalThis.location.origin}/hubCallback`,
    preserveSessionTopicFromToken: Boolean(topic),
    autoReconnect: true
  });
  let castWsConnected = false;
  const transport = new CastTransport({
    isConnected: () => castWsConnected,
    publishSceneUpdate: async (wire) => {
      await client.publish({
        event: {
          "hub.topic": topic,
          "hub.event": Vr,
          context: livesyncContextFromWire(wire)
        },
        actor: ID_ACTOR,
        "target.actor": "ID",
        "target.product.name": "3DSLICER-ID"
      });
    }
  });
  let blobsReady = false;
  let ingestInFlight = false;
  const pendingWire = [];
  const flushPending = () => {
    blobsReady = true;
    for (const w of pendingWire.splice(0)) transport.deliver(w);
  };
  globalThis.setTimeout(() => {
    if (!blobsReady && !ingestInFlight) {
      console.info(`${LOG} blob wait timeout \u2014 releasing livesync buffer`);
      flushPending();
    }
  }, 6e4);
  client.onMessage(async (message) => {
    const eventName = ke(message?.event);
    const from = io(message);
    if (from && from === subscriberName) return;
    if (eventName !== Vr) return;
    const eventObj = message.event || {};
    const rawContext = eventObj.context;
    const contextFiles = Xn(eventObj);
    const wire = wireFromSceneUpdateContext(
      rawContext != null ? rawContext : ro(message)
    );
    const hubBlobs = isHubBlobsWire(wire) || contextFiles.some((f) => {
      const ids = f.payloadIds;
      return Array.isArray(ids) && ids.length > 0;
    });
    if (hubBlobs) {
      ingestInFlight = true;
      try {
        const filesMeta = contextFiles.length > 0 ? contextFiles : Array.isArray(wire?.files) ? wire.files : [];
        console.info(
          `${LOG} HubBlobs received \u2014 fetching ${filesMeta.length} payload(s)\u2026`
        );
        if (!filesMeta.length) {
          flushPending();
          return;
        }
        const files = await fetchHubBlobFilesWithRetry(client, filesMeta);
        const n = ingestBlobFiles(hashToBytes, files);
        console.info(`${LOG} HubBlobs cached`, n);
        flushPending();
      } catch (err) {
        console.error(`${LOG} HubBlobs resolve failed`, err);
        flushPending();
      } finally {
        ingestInFlight = false;
      }
      return;
    }
    if (wire == null) return;
    const wireEvent = String(wire.event || "");
    if (wireEvent === "Snapshot") {
      const nodes = wire.scene?.nodes;
      console.info(
        `${LOG} Snapshot received nodes=`,
        nodes ? Object.keys(nodes).length : 0,
        blobsReady ? "(deliver)" : "(buffer until HubBlobs)"
      );
    }
    if (!blobsReady) {
      pendingWire.push(wire);
      return;
    }
    transport.deliver(wire);
  });
  client.onConnectionStateChange((wsState) => {
    console.info(`${LOG} ws state=`, wsState);
    if (wsState === "connected") {
      castWsConnected = true;
      transport.notifyOpen();
      resolveReady();
    } else if (wsState === "error" || wsState === "disconnected") {
      castWsConnected = false;
      transport.notifyClose();
    }
  });
  client.setToken(token);
  if (topic) client.setTopic(topic);
  const subStatus = await client.subscribe();
  console.info(`${LOG} subscribe status=`, subStatus);
  return {
    transport,
    client,
    topic,
    subscriberName,
    ready,
    close: () => {
      transport.close();
      void client.unsubscribe?.();
    }
  };
}
var HUB_MIRROR_BLOB_BASE = "http://hub-mirror.local/mrson/";

// render/demos/hub-mirror/hub-mirror-browser.ts
var status = (m) => {
  const e = document.getElementById("status-text");
  if (e) e.textContent = m;
};
var el = (id) => document.getElementById(id);
var CELLS = ["red", "yellow", "green", "threeD"];
var SLICE_CELLS = ["red", "yellow", "green"];
async function main() {
  if (!navigator.gpu) {
    status("WebGPU not available");
    return;
  }
  const p = new URLSearchParams(location.search);
  status("connecting to Slicer hub\u2026");
  let cast;
  try {
    cast = await connectHubMirrorCast();
    await Promise.race([
      cast.ready,
      new Promise(
        (_, rej) => setTimeout(() => rej(new Error("Cast connect timeout")), 2e4)
      )
    ]);
  } catch (err) {
    status(err instanceof Error ? err.message : String(err));
    return;
  }
  const httpBase = HUB_MIRROR_BLOB_BASE;
  const gpu = await initDevice();
  const preferred = navigator.gpu.getPreferredCanvasFormat();
  const srgb = preferred + "-srgb";
  const cv = {}, cx = {};
  for (const c of CELLS) {
    cv[c] = el("c-" + c);
    cx[c] = cv[c].getContext("webgpu");
    cx[c].configure({ device: gpu.device, format: preferred, viewFormats: [srgb], alphaMode: "opaque" });
  }
  const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
  const resizeAll = () => {
    for (const c of CELLS) {
      cv[c].width = Math.max(1, Math.round(cv[c].clientWidth * dpr));
      cv[c].height = Math.max(1, Math.round(cv[c].clientHeight * dpr));
    }
  };
  resizeAll();
  const camera = VtkCamera.slicerDefault();
  let scene = null;
  const fields3d = /* @__PURE__ */ new Map();
  let volumeField = null;
  let volumeShown3D = false;
  let clip = null;
  let inReplay = false;
  let followCamera = true;
  let scrubToSlicer = true;
  const slice = new SliceRenderer(gpu, srgb);
  let volumeReady = false;
  let segOverlay = null;
  let segFill = 0.5;
  let segOutline = 1;
  const planes = {};
  const sliceBranched = {};
  const CELL_ORIENT = { red: "axial", green: "coronal", yellow: "sagittal" };
  const visible = new Set(CELLS);
  const clearCanvas = (c) => {
    const enc = gpu.device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: cx[c].getCurrentTexture().createView({ format: srgb }), clearValue: { r: 0.02, g: 0.024, b: 0.04, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    pass.end();
    gpu.device.queue.submit([enc.finish()]);
  };
  const a3d = mountAdaptive3d({
    scene: () => scene,
    view: () => cx.threeD.getCurrentTexture().createView({ format: srgb }),
    size: () => ({ w: visible.has("threeD") ? cv.threeD.width : 0, h: cv.threeD.height }),
    setCamera: (s, w, h) => s.setCamera(camera.position, camera.focalPoint, camera.viewUp, camera.viewAngle, w, h),
    gpu,
    movingScaleCap: 0.4,
    // heavy segmentation DVR: ~0.4x res while moving -> ~60fps interactive (measured 16ms @0.33)
    target: 8
    // converge AA in ~0.8s after motion stops (not ~2.8s)
  });
  const renderSlice = (c) => {
    if (c === "threeD" || !visible.has(c)) return;
    if (!volumeReady) {
      clearCanvas(c);
      return;
    }
    const pl = planes[c];
    if (!pl) {
      clearCanvas(c);
      return;
    }
    const [lo2, hi2] = volumeField.aabb();
    const axis = pl.orient === "axial" ? 2 : pl.orient === "coronal" ? 1 : 0;
    const off01 = Math.max(0, Math.min(1, (pl.posMm - lo2[axis]) / Math.max(hi2[axis] - lo2[axis], 1e-6)));
    if (!sliceBranched[c]) {
      if (pl.centerRAS && pl.fovX && pl.fovY) slice.setMirrorFrame(pl.orient, pl.centerRAS, pl.fovX, pl.fovY);
      else slice.resetView(pl.orient);
    }
    slice.setPlane(pl.orient, off01);
    slice.renderToView(cx[c].getCurrentTexture().createView({ format: srgb }), cv[c].width, cv[c].height);
  };
  const renderSlices = () => {
    for (const c of SLICE_CELLS) renderSlice(c);
  };
  const rebuild3d = () => {
    const fs2 = [...fields3d.values()];
    if (volumeShown3D && volumeField) fs2.unshift(volumeField);
    if (fs2.length === 0) {
      scene = null;
      clearCanvas("threeD");
      return;
    }
    if (!scene) scene = new SceneRenderer(gpu, srgb);
    scene.build(fs2);
    if (clip) scene.setClipBox(clip.lo, clip.hi);
    a3d.draw();
  };
  const LAYOUTS = {
    fourUp: ["red", "yellow", "green", "threeD"],
    conventional: ["red", "yellow", "green", "threeD"],
    conventionalWidescreen: ["red", "yellow", "green", "threeD"],
    fourByThree: ["red", "yellow", "green", "threeD"],
    oneUp3D: ["threeD"],
    dual3D: ["threeD"],
    oneUpRed: ["red"],
    oneUpYellow: ["yellow"],
    oneUpGreen: ["green"]
  };
  const applyLayout = (name) => {
    const cells = LAYOUTS[name] ?? LAYOUTS.fourUp;
    visible.clear();
    for (const c of cells) visible.add(c);
    const grid = document.getElementById("grid");
    grid.style.gridTemplateColumns = cells.length === 1 ? "1fr" : "1fr 1fr";
    grid.style.gridTemplateRows = cells.length === 1 ? "1fr" : "1fr 1fr";
    for (const c of CELLS) document.getElementById("cell-" + c).classList.toggle("hidden", !visible.has(c));
    resizeAll();
    renderSlices();
    a3d.draw();
  };
  const view = {
    setField(k, f) {
      fields3d.set(k, f);
      rebuild3d();
    },
    removeField(k) {
      if (fields3d.delete(k)) rebuild3d();
    },
    // A field changed IN PLACE (markup point moved, colours, etc.): re-pack the material uniforms
    // (sphere/segment positions live there) so the change reaches the GPU — the render's flush()
    // then uploads it. Without this, redraw re-renders STALE uniforms and the glyph never moves.
    redraw() {
      scene?.syncUniforms();
      a3d.draw();
    },
    setCamera(c) {
      if (!inReplay && !followCamera) return;
      camera.position = c.position;
      camera.focalPoint = c.focalPoint;
      camera.viewUp = c.viewUp;
      if (c.viewAngle) camera.viewAngle = c.viewAngle;
      a3d.draw();
    },
    setClipBox(lo2, hi2) {
      clip = lo2 ? { lo: lo2, hi: hi2 } : null;
      if (scene) {
        if (clip) scene.setClipBox(clip.lo, clip.hi);
        else scene.setClipPlanes([]);
      }
      a3d.draw();
    },
    setVolumeField(f, wl) {
      volumeField = f;
      if (f) {
        const [lo2, hi2] = f.aabb();
        slice.setVolume(f.patientToTexture(), lo2, hi2);
        slice.setTextures(f.volumeTexture(), segOverlay ?? void 0);
        if (wl) slice.setWindowLevel(wl.win, wl.lev);
        slice.setOverlayOpacity(segOverlay ? segFill : 0);
        slice.setOutlineOpacity(segOverlay ? segOutline : 0);
        volumeReady = true;
        renderSlices();
      } else {
        volumeReady = false;
        for (const c of SLICE_CELLS) clearCanvas(c);
      }
      rebuild3d();
    },
    showVolume3D(show) {
      volumeShown3D = show;
      rebuild3d();
    },
    setSlicePlane(cell, pl) {
      cell = cell.toLowerCase();
      if (!(cell in CELL_ORIENT)) return;
      planes[cell] = pl;
      renderSlice(cell);
    },
    // SliceDM keys by Slicer layoutName (Red/Green/Yellow); this demo has the fixed trio
    setLayout(name) {
      applyLayout(name);
    },
    setSegmentationOverlay(tex, fillOpacity, outlineOpacity) {
      segOverlay = tex;
      segFill = fillOpacity;
      segOutline = outlineOpacity;
      if (volumeField) {
        slice.setTextures(volumeField.volumeTexture(), tex ?? void 0);
        slice.setOverlayOpacity(tex ? fillOpacity : 0);
        slice.setOutlineOpacity(tex ? outlineOpacity : 0);
      }
      renderSlices();
    }
  };
  addEventListener("resize", () => {
    resizeAll();
    renderSlices();
    a3d.draw();
  });
  const seged = location.pathname.includes("seged") || p.has("seged");
  const segManager = seged ? new SegEditDisplayableManager(gpu.device, { onEdit: (k) => status("seged: applied " + k) }) : new SegmentationDisplayableManager(gpu.device, 1.5);
  const markupsDM = new MarkupsDisplayableManager();
  const live = new LiveScene(httpBase, [
    new LayoutDisplayableManager(),
    new CameraDisplayableManager(),
    new VolumeRenderingDisplayableManager(gpu.device),
    new SliceDisplayableManager(),
    segManager,
    markupsDM,
    new RoiCropDisplayableManager()
  ]);
  live.view = view;
  const sync = new LiveSync(live, cast.transport);
  status("hub-mirror connected \u2014 waiting for LiveScene\u2026");
  const nodeVisible = (type) => {
    const n = live.find(type);
    return !!(n && n.visible !== false);
  };
  const setNodeVisible = (type, on2) => {
    const n = live.find(type);
    if (n) live.write({ op: "patch", id: n.id, path: "#/visible", value: on2 });
  };
  const controls = [
    {
      label: "Volume rendering",
      disabled: () => !live.find("volumeRenderingDisplay"),
      get: () => nodeVisible("volumeRenderingDisplay"),
      set: (on2) => setNodeVisible("volumeRenderingDisplay", on2)
    },
    {
      label: "Segmentation",
      disabled: () => !live.find("segmentation"),
      get: () => nodeVisible("segmentation"),
      set: (on2) => setNodeVisible("segmentation", on2)
    }
  ];
  const chrome = installChrome({ controls, anchor: cv.threeD });
  live.subscribe((c) => {
    if (c.type === "volumeRenderingDisplay" || c.type === "segmentation") chrome.refresh();
    if (c.kind === "upsert" || c.kind === "reset") {
      status(`hub-mirror: ${live.nodes.size} node(s)`);
    }
  });
  Object.assign(globalThis, { __live: live, __sync: sync, __camState: () => camera.state() });
  if (seged) {
    Object.assign(globalThis, {
      __seged: {
        fields: () => [...fields3d.keys()],
        vr3d: () => volumeShown3D,
        cam: () => camera.state(),
        rebuild: () => {
          rebuild3d();
          return [...fields3d.keys()];
        },
        redraw: () => a3d.draw(),
        palette: () => segManager.diag?.()
      }
    });
  }
  let drag = null;
  const HIT_PX = 16;
  const evPx = (e) => ({ sx: e.offsetX * dpr, sy: e.offsetY * dpr });
  const pick = (sx, sy) => {
    let best = null, bestD = HIT_PX * dpr;
    for (const hd of markupsDM.handles()) {
      const pr = camera.worldToDisplay(hd.ras, cv.threeD.width, cv.threeD.height);
      if (pr.depth <= 0) continue;
      const d = Math.hypot(pr.x - sx, pr.y - sy);
      if (d < bestD) {
        bestD = d;
        best = { id: hd.id, index: hd.index, depth: pr.depth };
      }
    }
    return best;
  };
  const opFor = (sx, sy) => {
    const ras = camera.displayToWorldAtDepth(sx, sy, drag.depth, cv.threeD.width, cv.threeD.height);
    return { ras, op: { op: "cmd", id: drag.id, cmd: "setControlPoint", args: { index: drag.index, position: ras } } };
  };
  cv.threeD.addEventListener("pointerdown", (e) => {
    if (inReplay || !visible.has("threeD")) return;
    const { sx, sy } = evPx(e);
    const h = pick(sx, sy);
    if (h) {
      drag = h;
      markupsDM.touch(h.id, h.index);
      cv.threeD.setPointerCapture(e.pointerId);
      cv.threeD.style.cursor = "grabbing";
      e.preventDefault();
    }
  });
  cv.threeD.addEventListener("pointermove", (e) => {
    if (inReplay) return;
    const { sx, sy } = evPx(e);
    if (!drag) {
      cv.threeD.style.cursor = pick(sx, sy) ? "grab" : "default";
      return;
    }
    const { ras, op } = opFor(sx, sy);
    markupsDM.moveLocal(drag.id, drag.index, ras, live);
    markupsDM.touch(drag.id, drag.index);
    sync.sendOps([op]);
  });
  const endDrag = (e) => {
    if (!drag) return;
    const { id, index } = drag;
    const { sx, sy } = evPx(e);
    sync.sendOps([opFor(sx, sy).op]);
    sync.flush();
    markupsDM.touch(id, index);
    try {
      cv.threeD.releasePointerCapture(e.pointerId);
    } catch {
    }
    cv.threeD.style.cursor = "default";
    drag = null;
  };
  cv.threeD.addEventListener("pointerup", endDrag);
  cv.threeD.addEventListener("pointercancel", endDrag);
  const retryBtn = document.getElementById("status-retry");
  const statusBar = document.getElementById("status");
  retryBtn?.addEventListener("click", () => sync.reconnectNow());
  let countdown;
  const stopCountdown = () => {
    if (countdown !== void 0) {
      clearInterval(countdown);
      countdown = void 0;
    }
  };
  const renderStatus = (s) => {
    stopCountdown();
    if (s.state === "connected") {
      status("mirroring Slicer");
      statusBar?.classList.remove("down");
      if (retryBtn) retryBtn.hidden = true;
    } else if (s.state === "connecting") {
      status(s.attempt > 0 ? "reconnecting\u2026" : "connecting to Slicer live channel\u2026");
      statusBar?.classList.toggle("down", s.attempt > 0);
      if (retryBtn) retryBtn.hidden = true;
    } else {
      statusBar?.classList.add("down");
      if (retryBtn) retryBtn.hidden = false;
      const tick = () => {
        const secs = Math.max(0, Math.ceil((s.nextRetryAt - Date.now()) / 1e3));
        status(`connection lost \u2014 reconnecting in ${secs}s`);
      };
      tick();
      countdown = setInterval(tick, 500);
    }
  };
  sync.onStatus = renderStatus;
  const STROKE_WINDOW_MS = 2500;
  let strokeField = null;
  function updateStrokeOverlay(t) {
    const active = src?.strokesInWindow?.(t, STROKE_WINDOW_MS) ?? [];
    const segs = [];
    for (const s of active) {
      const pts = s.edit.points ?? [];
      const a = Math.max(0.05, 1 - (t - s.t) / STROKE_WINDOW_MS);
      const col = s.edit.mode === "remove" ? [1, 0.35, 0.35, a] : [0.45, 1, 0.55, a];
      for (let i = 0; i + 1 < pts.length; i++) segs.push({ a: pts[i], b: pts[i + 1], radius: 2.5, color: col });
    }
    if (!strokeField) {
      strokeField = new CapsuleField(segs, { screenSpace: true, ghost: true });
      view.setField("strokeOverlay", strokeField);
    } else {
      strokeField.setSegments(segs);
      view.redraw();
    }
  }
  function clearStrokeOverlay() {
    if (strokeField) {
      view.removeField("strokeOverlay");
      strokeField = null;
    }
  }
  const tl = document.getElementById("timeline");
  const scrub = document.getElementById("tl-scrub");
  const timeLbl = document.getElementById("tl-time");
  const preview = document.getElementById("tl-preview");
  const playBtn = document.getElementById("tl-play");
  const liveBtn = document.getElementById("tl-live");
  const markBtn = document.getElementById("tl-mark");
  const liveHttpBase = httpBase;
  let src = null;
  let displayed = null;
  let restoring = false, pendingT = null;
  let playTimer, playAnchorWall = 0;
  const GAP_MS = 1e3;
  let playSched = [];
  let playTotal = 0;
  function buildSchedule(startT) {
    const [lo2, hi2] = src.span();
    const s0 = Math.max(lo2, Math.min(hi2, startT));
    const times = [s0, ...src.frameTimes().filter((t) => t > s0 && t <= hi2)];
    if (times[times.length - 1] < hi2) times.push(hi2);
    playSched = [];
    let cum = 0;
    for (let i = 0; i + 1 < times.length; i++) {
      const dur = Math.min(times[i + 1] - times[i], GAP_MS);
      playSched.push({ recStart: times[i], recEnd: times[i + 1], playStart: cum, playDur: dur });
      cum += dur;
    }
    playTotal = cum;
  }
  const warp = (E) => {
    for (const s of playSched) if (E < s.playStart + s.playDur) return s.recStart + (s.playDur > 0 ? (E - s.playStart) / s.playDur : 1) * (s.recEnd - s.recStart);
    return src.span()[1];
  };
  let t0 = 0;
  const tAt = (v) => {
    if (!src) return 0;
    const [a, b] = src.span();
    return a + v / 1e3 * Math.max(0, b - a);
  };
  const fmt = (t) => {
    const s = Math.max(0, (t - t0) / 1e3);
    return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  };
  const stopPlay = () => {
    if (playTimer !== void 0) {
      clearInterval(playTimer);
      playTimer = void 0;
      playBtn.classList.remove("on");
      playBtn.textContent = "\u25B6";
    }
  };
  let branched = false;
  async function restore(t) {
    if (!src) return;
    if (restoring) {
      pendingT = t;
      return;
    }
    restoring = true;
    for (const c of SLICE_CELLS) sliceBranched[c] = false;
    const target = src.seek(t);
    await live.applySnapshot(target, displayed ?? /* @__PURE__ */ new Map(), { force: (n) => n.type === "camera" || n.type === "view" });
    displayed = target;
    if (scrubToSlicer) {
      const ops = [];
      for (const n of target.values()) {
        if (n.type === "camera") {
          ops.push({ op: "patch", id: n.id, path: "#/position", value: n.position });
          ops.push({ op: "patch", id: n.id, path: "#/focalPoint", value: n.focalPoint });
          ops.push({ op: "patch", id: n.id, path: "#/viewUp", value: n.viewUp });
        } else if (n.type === "view" && n.kind === "slice" && typeof n.offset === "number") {
          ops.push({ op: "patch", id: n.id, path: "#/offset", value: n.offset });
        }
      }
      if (ops.length) sync.sendOps(ops);
    }
    updateStrokeOverlay(t);
    if (branched) {
      branched = false;
      tl.classList.remove("branched");
    }
    restoring = false;
    if (pendingT !== null) {
      const n = pendingT;
      pendingT = null;
      restore(n);
    }
  }
  function setLiveUI() {
    tl.classList.remove("replay");
    scrub.disabled = true;
    scrub.value = "1000";
    playBtn.disabled = true;
    markBtn.disabled = true;
    preview.style.display = "none";
    updateLiveBtn();
  }
  function updateLiveBtn() {
    liveBtn.textContent = "Live";
    liveBtn.disabled = followCamera;
    liveBtn.classList.toggle("on", followCamera);
    timeLbl.textContent = followCamera ? "\u25CF recording" : "\u2387 looking around \u2014 Live to resync";
  }
  function resyncLive() {
    followCamera = true;
    const cam = [...live.nodes.values()].find((n) => n.type === "camera");
    if (cam) view.setCamera({ position: cam.position, focalPoint: cam.focalPoint, viewUp: cam.viewUp, viewAngle: cam.viewAngle });
    updateLiveBtn();
  }
  function enterReplay(recording) {
    stopPlay();
    src = recording;
    inReplay = true;
    live.httpBase = recording.base;
    live.setLive(false);
    displayed = /* @__PURE__ */ new Map();
    [t0] = recording.span();
    tl.classList.add("replay");
    scrub.disabled = false;
    playBtn.disabled = false;
    liveBtn.disabled = false;
    liveBtn.textContent = "Live";
    liveBtn.classList.remove("on");
    markBtn.disabled = true;
    Object.assign(globalThis, { __recording: recording });
    status(`replay: ${recording.session.id} \u2014 scrub / drag to explore`);
    attachSliceInteraction();
    scrub.value = "0";
    restore(recording.span()[0]);
  }
  async function goLive() {
    stopPlay();
    detachSliceInteraction();
    clearStrokeOverlay();
    for (const c of SLICE_CELLS) sliceBranched[c] = false;
    inReplay = false;
    branched = false;
    followCamera = true;
    tl.classList.remove("branched");
    live.httpBase = liveHttpBase;
    if (displayed !== null) {
      await live.applySnapshot(live.nodes, displayed);
      displayed = null;
    }
    live.setLive(true);
    src = null;
    setLiveUI();
    status("mirroring Slicer");
  }
  scrub.addEventListener("input", () => {
    if (!src) return;
    stopPlay();
    const th = src.nearestThumb(tAt(Number(scrub.value)));
    if (th) {
      preview.src = th.url;
      preview.style.display = "block";
      const frac = Number(scrub.value) / 1e3, x = 12 + frac * (window.innerWidth - 24 - 200);
      preview.style.left = `${Math.max(6, Math.min(window.innerWidth - 206, x))}px`;
    }
    timeLbl.textContent = fmt(tAt(Number(scrub.value)));
  });
  scrub.addEventListener("change", () => {
    if (!src) return;
    preview.style.display = "none";
    restore(tAt(Number(scrub.value)));
  });
  playBtn.addEventListener("click", () => {
    if (!src) return;
    if (playTimer !== void 0) {
      stopPlay();
      return;
    }
    const startT = Number(scrub.value) >= 999 ? src.span()[0] : tAt(Number(scrub.value));
    buildSchedule(startT);
    playAnchorWall = Date.now();
    playBtn.classList.add("on");
    playBtn.textContent = "\u23F8";
    restore(startT);
    playTimer = setInterval(() => {
      if (!src) {
        stopPlay();
        return;
      }
      const [lo2, hi2] = src.span();
      const E = Date.now() - playAnchorWall;
      if (E >= playTotal) {
        stopPlay();
        scrub.value = "1000";
        restore(hi2);
        return;
      }
      const t = warp(E);
      scrub.value = String(Math.round((t - lo2) / Math.max(1, hi2 - lo2) * 1e3));
      timeLbl.textContent = fmt(t);
      restore(t);
    }, 120);
  });
  liveBtn.addEventListener("click", () => {
    if (src) goLive();
    else resyncLive();
  });
  const branch = () => {
    if (inReplay) {
      stopPlay();
      if (!branched) {
        branched = true;
        tl.classList.add("branched");
      }
      timeLbl.textContent = "\u2387 branched \u2014 Play to resume";
    } else if (followCamera) {
      followCamera = false;
      updateLiveBtn();
    }
  };
  const cam3d = new CameraInteractor(camera, () => a3d.draw());
  const xy3d = (e) => {
    const r = cv.threeD.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  cv.threeD.addEventListener("contextmenu", (e) => e.preventDefault());
  cv.threeD.addEventListener("pointerdown", (e) => {
    if (drag) return;
    const { x, y: y2 } = xy3d(e);
    cam3d.start(e.button, x, y2, cv.threeD.clientHeight, { shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey, alt: e.altKey });
    cv.threeD.setPointerCapture(e.pointerId);
    branch();
  });
  cv.threeD.addEventListener("pointermove", (e) => {
    if (cam3d.action === "none") return;
    const { x, y: y2 } = xy3d(e);
    cam3d.move(x, y2, cv.threeD.clientWidth, cv.threeD.clientHeight);
  });
  const end3d = (e) => {
    if (cam3d.action !== "none") {
      cam3d.end();
      try {
        cv.threeD.releasePointerCapture(e.pointerId);
      } catch {
      }
    }
  };
  cv.threeD.addEventListener("pointerup", end3d);
  cv.threeD.addEventListener("pointercancel", end3d);
  cv.threeD.addEventListener("wheel", (e) => {
    e.preventDefault();
    cam3d.wheel(e.deltaY < 0);
    branch();
  }, { passive: false });
  let sliceCtl = [];
  function attachSliceInteraction() {
    detachSliceInteraction();
    for (const c of SLICE_CELLS) {
      sliceCtl.push(attachSliceControls(cv[c], {
        orient: CELL_ORIENT[c],
        getSlice: () => slice,
        step: (fwd) => {
          const pl = planes[c];
          if (!pl || !volumeField) return;
          const [lo2, hi2] = volumeField.aabb();
          const axis = pl.orient === "axial" ? 2 : pl.orient === "coronal" ? 1 : 0;
          pl.posMm = Math.max(lo2[axis], Math.min(hi2[axis], pl.posMm + (hi2[axis] - lo2[axis]) * 0.02 * (fwd ? -1 : 1)));
        },
        redraw: () => {
          sliceBranched[c] = true;
          branch();
          renderSlice(c);
        }
      }));
    }
  }
  function detachSliceInteraction() {
    for (const s of sliceCtl) s.detach();
    sliceCtl = [];
  }
  const recName = p.get("rec");
  if (recName) {
    status(`loading recording ${recName}\u2026`);
    const recording = await Recording.load(new URL(`rec/${recName}/`, httpBase).href);
    enterReplay(recording);
    sync.connect();
    return;
  }
  setLiveUI();
  async function loadLatestRecording() {
    for (let i = 0; i < 15; i++) {
      try {
        const list = await (await fetch(new URL("recs", liveHttpBase).href)).json();
        const recs = list.recordings || [];
        const fresh = recs.filter((r) => r.hasContent && r.endedAt && Date.now() - r.endedAt < 2e4).sort((a, b) => b.endedAt - a.endedAt);
        if (fresh.length) {
          const r = await Recording.load(new URL(`rec/${fresh[0].name}/`, liveHttpBase).href);
          if (r.session.frames.length) return r;
        }
      } catch {
      }
      await new Promise((r) => setTimeout(r, 300));
    }
    return null;
  }
  let loadingRec = false;
  live.subscribe(async (c) => {
    if (c.kind !== "reset" || loadingRec) return;
    loadingRec = true;
    status("finalizing recording\u2026");
    const recording = await loadLatestRecording();
    loadingRec = false;
    if (recording) enterReplay(recording);
    else if (!src) status("mirroring Slicer");
  });
  await sync.connect();
}
main();
