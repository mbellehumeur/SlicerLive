// render/device.ts
async function initDevice() {
  const gpu = navigator.gpu;
  if (!gpu) throw new Error("WebGPU not available (need Chrome/Edge/Safari or Deno --unstable-webgpu)");
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const want = ["float32-filterable", "timestamp-query", "shader-f16"].filter((f) => adapter.features.has(f));
  const lim = adapter.limits;
  const requiredLimits = {};
  const raise = (k2) => {
    const v2 = lim[k2];
    if (typeof v2 === "number") requiredLimits[k2] = v2;
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
      for (let k2 = 0; k2 < 4; k2++) s += a[k2 * 4 + r] * b[c * 4 + k2];
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
function perspectiveZOTile(fovy, viewW, viewH, x, y, w, h, near, far) {
  const t = near * Math.tan(fovy / 2), b = -t;
  const r = t * (viewW / viewH), l = -r;
  const l2 = l + (r - l) * x / viewW, r2 = l + (r - l) * (x + w) / viewW;
  const t2 = t - (t - b) * y / viewH, b2 = t - (t - b) * (y + h) / viewH;
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
  const M2 = transpose4(ijkToRAS);
  const A = new Float32Array(16);
  for (let a = 0; a < 3; a++) {
    A[a * 4 + a] = dims[a];
    A[12 + a] = -0.5;
  }
  A[15] = 1;
  return multiply(M2, A);
}
function volumeAABBFromIjkToRAS(ijkToRAS, dims) {
  const t2r = texToRASFromIjkToRAS(ijkToRAS, dims);
  const lo = [Infinity, Infinity, Infinity], hi2 = [-Infinity, -Infinity, -Infinity];
  for (let c = 0; c < 8; c++) {
    const u = c & 1, v2 = c >> 1 & 1, w = c >> 2 & 1;
    for (let r = 0; r < 3; r++) {
      const p = t2r[r] * u + t2r[4 + r] * v2 + t2r[8 + r] * w + t2r[12 + r];
      if (p < lo[r]) lo[r] = p;
      if (p > hi2[r]) hi2[r] = p;
    }
  }
  return [lo, hi2];
}
function applyMat4(m, p) {
  const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
  const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
  const z2 = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15] || 1;
  return [x / w, y / w, z2 / w];
}
function spacingFromIjkToRAS(ijkToRAS) {
  const col = (c) => Math.hypot(ijkToRAS[c], ijkToRAS[4 + c], ijkToRAS[8 + c]);
  return [col(0), col(1), col(2)];
}

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
function transformedAABB(m, lo, hi2) {
  const mn2 = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < 8; i++) {
    const c = applyMat4(m, [i & 1 ? hi2[0] : lo[0], i & 2 ? hi2[1] : lo[1], i & 4 ? hi2[2] : lo[2]]);
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
  setClim(lo, hi2) {
    this.clim = [lo, hi2];
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
    const [lo, hi2] = this.origBox ?? this.box;
    return [(lo[0] + hi2[0]) / 2, (lo[1] + hi2[1]) / 2, (lo[2] + hi2[2]) / 2];
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
  bindEntries(_s, base) {
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
  bindEntries(_s, base) {
    const e = [{ binding: base, resource: this.tex.createView() }];
    if (this.attrTex) e.push({ binding: base + 1, resource: this.attrTex.createView() });
    return e;
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
  const ds = new DecompressionStream("deflate");
  return await new Response(new Response(buf).body.pipeThrough(ds)).arrayBuffer();
}
var blobFetch = (url) => fetch(url);
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
  const cross3 = [];
  const rest = [];
  for (let kk = 0; kk < ncz; kk++) {
    for (let jj = 0; jj < ncy; jj++) {
      for (let ii2 = 0; ii2 < ncx; ii2++) {
        const job = [kk, jj, ii2];
        if (kk === midK) axial.push(job);
        else if (jj === midJ || ii2 === midI) cross3.push(job);
        else rest.push(job);
      }
    }
  }
  axial.sort(byCenter);
  cross3.sort(byCenter);
  rest.sort(byCenter);
  return [...axial, ...cross3, ...rest];
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
  let lo = Infinity, hi2 = -Infinity;
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
            const v2 = chunk[src + xx];
            out[dst + xx] = v2;
            if (v2 < lo) lo = v2;
            if (v2 > hi2) hi2 = v2;
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
  return { data: out, dtype: z2.dtype, dims, range: [lo, hi2] };
}

// render/mrson.ts
var TYPE_TO_CLASS = {
  image: "vtkMRMLScalarVolumeNode",
  mesh: "vtkMRMLModelNode",
  segmentation: "vtkMRMLSegmentationNode",
  markup: "vtkMRMLMarkupsFiducialNode",
  transform: "vtkMRMLLinearTransformNode",
  camera: "vtkMRMLCameraNode",
  view: "vtkMRMLViewNode",
  transferFunction: "vtkMRMLVolumePropertyNode",
  scalarVolumeDisplay: "vtkMRMLScalarVolumeDisplayNode",
  volumeRenderingDisplay: "vtkMRMLGPURayCastVolumeRenderingDisplayNode",
  modelDisplay: "vtkMRMLModelDisplayNode",
  markupDisplay: "vtkMRMLMarkupsDisplayNode"
};
function isMrsonScene(raw) {
  const r = raw;
  if (!r || typeof r !== "object") return false;
  if (r.mrson !== void 0) return true;
  return !!r.nodes && Object.values(r.nodes).some((n) => typeof n?.type === "string");
}
var colorRows = (a) => Array.isArray(a) ? a.map((s) => [s.value, s.rgba[0], s.rgba[1], s.rgba[2]]) : [];
var opacityRows = (a) => Array.isArray(a) ? a.map((s) => [s.value, s.opacity]) : [];
function adaptMrsonScene(scene) {
  const nodes = scene.nodes ?? {};
  const out = {};
  for (const [id, n] of Object.entries(nodes)) {
    const cls = n.source?.mrmlClass ?? TYPE_TO_CLASS[n.type] ?? n.type;
    const refs = { ...n.refs ?? {} };
    const attrs = {};
    switch (n.type) {
      case "image":
        attrs.zarr = n.zarr;
        attrs.ijkToRAS = n.ijkToRAS;
        attrs.dims = n.dims;
        attrs.comps = n.comps;
        break;
      case "transferFunction":
        attrs.color = colorRows(n.colorStops);
        attrs.scalarOpacity = opacityRows(n.scalarOpacity);
        attrs.gradientOpacity = opacityRows(n.gradientOpacity);
        attrs.shade = n.shade;
        break;
      case "scalarVolumeDisplay":
        attrs.window = n.window;
        attrs.level = n.level;
        attrs.color = n.color;
        attrs.visibility = n.visible ? 1 : 0;
        break;
      case "volumeRenderingDisplay":
        if (n.refs?.transferFunction) refs.volumeProperty = n.refs.transferFunction;
        break;
      case "markup": {
        attrs.controlPoints = n.controlPoints;
        const dc = (n.refs?.display ?? []).map((d) => nodes[d]?.color).find(Boolean);
        if (dc) attrs.color = dc.slice(0, 3);
        break;
      }
      case "camera":
        attrs.position = n.position;
        attrs.focalPoint = n.focalPoint;
        attrs.viewUp = n.viewUp;
        attrs.viewAngle = n.viewAngle;
        attrs.parallelScale = n.parallelScale;
        break;
      default:
        break;
    }
    out[id] = { id, class: cls, name: n.name, refs, attrs, blobs: [] };
  }
  return { blobBase: scene.blobBase, nodes: out };
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
      return Array.from({ length: comps }, (_2, c) => a[1 + c] + u * (b[1 + c] - a[1 + c]));
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
function lutFromWindowLevel() {
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const g = Math.round(t * 255);
    lut[i * 4 + 0] = lut[i * 4 + 1] = lut[i * 4 + 2] = g;
    lut[i * 4 + 3] = Math.round(Math.max(0, Math.min(1, (t - 0.15) / 0.85)) * 200);
  }
  return lut;
}
function parseMarkups(nodes) {
  const out = [];
  for (const n of Object.values(nodes)) {
    if (!/Markups.*Node$/.test(n.class)) continue;
    const cps = n.attrs?.controlPoints ?? n.attrs?.markups;
    if (!Array.isArray(cps)) continue;
    const color = n.attrs?.color ?? [1, 0.85, 0.2];
    cps.forEach((cp, i) => {
      const c = cp;
      const p = c.position ?? cp;
      if (!Array.isArray(p) || p.length < 3) return;
      out.push({ ras: [p[0], p[1], p[2]], label: c.label ?? `${n.name ?? "F"}-${i + 1}`, color });
    });
  }
  return out;
}
function scaledIjkToRAS(base, factor) {
  const m = base.slice();
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) m[r * 4 + c] *= factor;
  }
  return m;
}
function levelFromRecord(L, fallbackIjk) {
  const level = Number(L.level ?? 0);
  const z2 = {
    dir: L.dir,
    dataset: String(L.dataset ?? level),
    shape: L.shape,
    chunks: L.chunks,
    chunkGrid: L.chunkGrid,
    dtype: String(L.dtype ?? "<f4"),
    bytes: L.bytes,
    chunkHashes: L.chunkHashes
  };
  const ijk = Array.isArray(L.ijkToRAS) ? L.ijkToRAS.slice() : scaledIjkToRAS(fallbackIjk, 2 ** level);
  return { level, z: z2, ijkToRAS: ijk };
}
async function resolveSceneLevels(parsed) {
  const attrs = parsed.vol.attrs ?? {};
  if (Array.isArray(attrs.zarrLevels) && attrs.zarrLevels.length > 0) {
    return attrs.zarrLevels.map((L) => levelFromRecord(L, parsed.ijkToRAS)).sort((a, b) => a.level - b.level);
  }
  try {
    const r = await fetch(parsed.blobBase + "meta.json");
    if (r.ok) {
      const meta = await r.json();
      if (Array.isArray(meta.levels) && meta.levels.length > 0) {
        return meta.levels.map((L) => levelFromRecord(L, parsed.ijkToRAS)).sort((a, b) => a.level - b.level);
      }
    }
  } catch {
  }
  return [{ level: 0, z: parsed.z, ijkToRAS: parsed.ijkToRAS.slice() }];
}
async function parseSceneJson(sceneSource, opts = {}) {
  const pageBase = globalThis.location?.href ?? "file:///";
  let raw;
  let sceneAbs;
  if (typeof sceneSource === "string") {
    raw = await (await fetch(sceneSource)).json();
    sceneAbs = new URL(sceneSource, pageBase).href;
  } else {
    raw = sceneSource;
    const hint = typeof sceneSource.url === "string" ? sceneSource.url.trim() : "";
    const blobHint = String(sceneSource.blobBase || "").trim();
    if (/^https?:\/\//i.test(blobHint) || blobHint.startsWith("blob:")) {
      sceneAbs = hint || blobHint || pageBase;
    } else {
      sceneAbs = hint || pageBase;
    }
  }
  const adapted = isMrsonScene(raw) ? adaptMrsonScene(raw) : raw;
  const wrapper = adapted.nodes ? adapted : { nodes: adapted };
  const nodes = wrapper.nodes;
  const blobBase = new URL(wrapper.blobBase ?? "./blobs/", sceneAbs).href;
  const vol = Object.values(nodes).find((n) => n.class === "vtkMRMLScalarVolumeNode" && n.attrs?.zarr);
  if (!vol) throw new Error("no zarr ScalarVolumeNode in scene");
  const z2 = vol.attrs.zarr;
  let ijkToRAS = vol.attrs.ijkToRAS;
  if (!ijkToRAS) throw new Error("volume node has no ijkToRAS");
  if (opts.extraTranslationRAS) {
    const t = opts.extraTranslationRAS;
    ijkToRAS = ijkToRAS.slice();
    ijkToRAS[3] += t[0];
    ijkToRAS[7] += t[1];
    ijkToRAS[11] += t[2];
  }
  let vp;
  for (const dispId of vol.refs?.display ?? []) {
    const disp = nodes[dispId];
    for (const vpId of disp?.refs?.volumeProperty ?? []) {
      if (nodes[vpId]?.class === "vtkMRMLVolumePropertyNode") vp = nodes[vpId];
    }
  }
  const disp0 = nodes[(vol.refs?.display ?? [])[0]]?.attrs ?? {};
  const winFallback = disp0.window ?? 1400;
  const levFallback = disp0.level ?? 300;
  let lut, clim, shade;
  if (vp?.attrs?.color && vp?.attrs?.scalarOpacity) {
    const colorTF = vp.attrs.color, opacityTF = vp.attrs.scalarOpacity;
    const lo = colorTF[0][0], hi2 = colorTF[colorTF.length - 1][0];
    clim = [lo, hi2];
    lut = lutFromTransferFunctions(colorTF, opacityTF, clim);
    shade = vp.attrs.shade ? [0.25, 0.75, 0.5, 24] : [1, 0, 0, 1];
  } else {
    clim = [levFallback - winFallback / 2, levFallback + winFallback / 2];
    lut = lutFromWindowLevel();
    shade = [0.25, 0.75, 0.5, 24];
  }
  const win = disp0.window ?? winFallback;
  const lev = disp0.level ?? levFallback;
  return {
    nodes,
    blobBase,
    vol,
    z: z2,
    ijkToRAS,
    lut,
    clim,
    shade,
    win,
    lev,
    markups: parseMarkups(nodes)
  };
}
function sceneVolumeFromParts(field, voxels, dims, parsed, range, ijkToRAS = parsed.ijkToRAS) {
  const [lo, hi2] = field.aabb();
  const center = [(lo[0] + hi2[0]) / 2, (lo[1] + hi2[1]) / 2, (lo[2] + hi2[2]) / 2];
  const radius = Math.hypot(hi2[0] - lo[0], hi2[1] - lo[1], hi2[2] - lo[2]) / 2;
  return {
    field,
    voxels,
    dims,
    ijkToRAS,
    name: parsed.vol.name ?? "volume",
    range,
    center,
    radius,
    win: parsed.win,
    lev: parsed.lev,
    markups: parsed.markups,
    vr: { lut: parsed.lut, clim: parsed.clim, shade: parsed.shade }
  };
}
async function loadLevelSlabProgressive(dev, parsed, lev, handlers, opts) {
  const [nz, ny, nx] = lev.z.shape;
  const dims = [nx, ny, nz];
  const voxels = new Float32Array(nx * ny * nz);
  voxels.fill(parsed.lev - parsed.win / 2 - 1);
  const field = new ImageField(dev, voxels, dims, [1, 1, 1], parsed.lut, {
    clim: parsed.clim,
    ijkToRAS: lev.ijkToRAS,
    shade: parsed.shade
  });
  const sv = sceneVolumeFromParts(field, voxels, dims, parsed, [0, 0], lev.ijkToRAS);
  if (opts.notifyReady) handlers.onReady?.(sv);
  const zv = await fetchZarrVolume(parsed.blobBase, lev.z, {
    onBytes: handlers.onBytes,
    into: voxels,
    onChunk: (info) => {
      field.updateScalarsRegion(
        voxels,
        info.xStart,
        info.yStart,
        info.zStart,
        info.xCount,
        info.yCount,
        info.zCount
      );
      if (opts.notifySlabs) handlers.onSlab?.(info.zStart, info.zCount, sv);
    }
  });
  sv.range = zv.range;
  return sv;
}
async function loadSceneVolumeFieldProgressive(dev, sceneSource, handlers = {}, opts = {}) {
  const parsed = await parseSceneJson(sceneSource, opts);
  const levels = await resolveSceneLevels(parsed);
  const byCoarse = [...levels].sort((a, b) => b.level - a.level);
  const coarse = byCoarse[0];
  let current = await loadLevelSlabProgressive(dev, parsed, coarse, handlers, {
    notifyReady: true,
    notifySlabs: true
  });
  for (let i = 1; i < byCoarse.length; i++) {
    if (opts.signal?.aborted) break;
    const lev = byCoarse[i];
    const pending = await loadLevelSlabProgressive(dev, parsed, lev, handlers, {
      notifyReady: false,
      notifySlabs: false
    });
    if (opts.signal?.aborted) {
      pending.field.destroy();
      break;
    }
    current = pending;
    handlers.onUpgrade?.(current);
  }
  return current;
}

// render/slice-renderer.ts
var DEFAULT_FORMAT = "rgba8unorm-srgb";
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
    outA = clamp(ov.a * u.params.w * edge, 0.0, 1.0);
  }
  col = mix(col, ov.rgb, max(fillA, outA));
  return vec4<f32>(srgb2physical(col), 1.0);
}
`
);
var BASES = {
  axial: { uDir: [-1, 0, 0], vDir: [0, 1, 0], nDir: [0, 0, 1] },
  coronal: { uDir: [-1, 0, 0], vDir: [0, 0, 1], nDir: [0, 1, 0] },
  sagittal: { uDir: [0, -1, 0], vDir: [0, 0, 1], nDir: [1, 0, 0] }
};
var dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function ijkAxisForRasAxis(ijkToRAS, rasAxis) {
  let best = 0, bestMag = -1;
  for (let c = 0; c < 3; c++) {
    const mag = Math.abs(ijkToRAS[rasAxis * 4 + c]);
    if (mag > bestMag) {
      bestMag = mag;
      best = c;
    }
  }
  return best;
}
function slicerDefaultOffset01(orient, dims, ijkToRAS, rasLo, rasHi) {
  const b = BASES[orient];
  const nAbs = b.nDir.map(Math.abs);
  const n = nAbs[0] >= nAbs[1] && nAbs[0] >= nAbs[2] ? 0 : nAbs[1] >= nAbs[2] ? 1 : 2;
  const a = ijkAxisForRasAxis(ijkToRAS, n);
  const m = Math.floor((dims[a] - 1) / 2);
  const ijk = [(dims[0] - 1) / 2, (dims[1] - 1) / 2, (dims[2] - 1) / 2];
  ijk[a] = m;
  const ras = ijkToRAS[n * 4 + 0] * ijk[0] + ijkToRAS[n * 4 + 1] * ijk[1] + ijkToRAS[n * 4 + 2] * ijk[2] + ijkToRAS[n * 4 + 3];
  const span = rasHi[n] - rasLo[n];
  return span === 0 ? 0.5 : (ras - rasLo[n]) / span;
}
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
    sagittal: { panU: 0, panV: 0, zoom: 1 }
  };
  cX = [0, 0, 0];
  // in-plane centre of the LAST rendered frame (for viewToTex picking)
  // Optional per-orientation basis override (reslice along a volume's own axes). null = the
  // anatomical preset.
  basisOverride = {};
  constructor(gpu, format = DEFAULT_FORMAT) {
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
    const { lo, hi: hi2 } = this.extentAlong(n);
    return Math.max(0, Math.min(1, (dot3(ras, n) - lo) / Math.max(hi2 - lo, 1e-6)));
  }
  basisOf(orient) {
    return this.basisOverride[orient] ?? BASES[orient];
  }
  /** Extent of the volume's RAS bounding box projected onto a direction — the generalisation
   *  of "rasHi[axis] - rasLo[axis]" to an oblique axis. Reduces to exactly that for the
   *  anatomical bases, since projecting an axis-aligned box on its own axis is the axis span. */
  extentAlong(d) {
    let lo = Infinity, hi2 = -Infinity;
    for (let i = 0; i < 8; i++) {
      const c = [
        i & 1 ? this.rasHi[0] : this.rasLo[0],
        i & 2 ? this.rasHi[1] : this.rasLo[1],
        i & 4 ? this.rasHi[2] : this.rasLo[2]
      ];
      const t = dot3(c, d);
      if (t < lo) lo = t;
      if (t > hi2) hi2 = t;
    }
    return { lo, hi: hi2 };
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
    const u = this.extentAlong(b.uDir), v2 = this.extentAlong(b.vDir);
    return Math.max(u.hi - u.lo, v2.hi - v2.lo);
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
    const u = this.extentAlong(b.uDir), v2 = this.extentAlong(b.vDir);
    const uExt = u.hi - u.lo, vExt = v2.hi - v2.lo;
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
    const vs = this.viewState[orient];
    const { uS0, vS0 } = this.fitUV(orient, aspectWH);
    const uS = uS0 / vs.zoom, vS = vS0 / vs.zoom;
    const c = [(this.rasLo[0] + this.rasHi[0]) / 2, (this.rasLo[1] + this.rasHi[1]) / 2, (this.rasLo[2] + this.rasHi[2]) / 2];
    const nx = this.extentAlong(b.nDir);
    const want = nx.lo + Math.max(0, Math.min(1, offset01)) * (nx.hi - nx.lo);
    const have = dot3(c, b.nDir);
    c[0] += b.nDir[0] * (want - have);
    c[1] += b.nDir[1] * (want - have);
    c[2] += b.nDir[2] * (want - have);
    c[0] += b.uDir[0] * vs.panU + b.vDir[0] * vs.panV;
    c[1] += b.uDir[1] * vs.panU + b.vDir[1] * vs.panV;
    c[2] += b.uDir[2] * vs.panU + b.vDir[2] * vs.panV;
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
    const vs = this.viewState[orient];
    const { uS0, vS0 } = this.fitUV(orient, w / h);
    const z2 = Math.max(0.2, Math.min(50, vs.zoom * factor));
    vs.panU += (pu - 0.5) * (uS0 / vs.zoom - uS0 / z2);
    vs.panV += (0.5 - pv) * (vS0 / vs.zoom - vS0 / z2);
    vs.zoom = z2;
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
  setViewState(vs) {
    for (const k2 of Object.keys(vs)) {
      const v2 = vs[k2];
      if (v2 && Number.isFinite(v2.zoom) && v2.zoom > 0) this.viewState[k2] = { ...v2 };
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
    const st2 = this.viewState[orient];
    const { uS0, vS0 } = this.fitUV(orient, aspectWH);
    const fovX = uS0 / st2.zoom, fovY = vS0 / st2.zoom;
    const volC = [(this.rasLo[0] + this.rasHi[0]) / 2, (this.rasLo[1] + this.rasHi[1]) / 2, (this.rasLo[2] + this.rasHi[2]) / 2];
    const centerRAS = [
      volC[0] + b.uDir[0] * st2.panU + b.vDir[0] * st2.panV,
      volC[1] + b.uDir[1] * st2.panU + b.vDir[1] * st2.panV,
      volC[2] + b.uDir[2] * st2.panU + b.vDir[2] * st2.panV
    ];
    return { centerRAS, fovX, fovY };
  }
  /** Map a view (u,v) in [0,1] (y down) to normalized texture coords for the current
   *  plane — for click picking. Returns the tex coord; the caller converts to IJK via
   *  ijk = tex*dims - 0.5. Anisotropy/rotation are handled by the same p2t the shader uses. */
  viewToTex(u, v2) {
    const b = this.basisOf(this.orient);
    const uS = this.uSpanMm || this.viewSpanMm();
    const vS = this.vSpanMm || this.viewSpanMm();
    const c = this.cX;
    const ras = [
      c[0] + b.uDir[0] * (u - 0.5) * uS + b.vDir[0] * (0.5 - v2) * vS,
      c[1] + b.uDir[1] * (u - 0.5) * uS + b.vDir[1] * (0.5 - v2) * vS,
      c[2] + b.uDir[2] * (u - 0.5) * uS + b.vDir[2] * (0.5 - v2) * vS
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
    const v2 = 0.5 - (d[0] * b.vDir[0] + d[1] * b.vDir[1] + d[2] * b.vDir[2]) / vS;
    return { u, v: v2, distMm: dot3(d, b.nDir) };
  }
  /** Map a view (u,v in [0,1], y down) on a plane back to a RAS point ON that plane —
   *  the exact inverse of rasToView (same pan/zoom/aspect). Used to drag a 2D markup:
   *  the point lands on the current slice (its out-of-plane coord becomes the plane offset). */
  viewToRas(orient, offset01, u, v2, aspectWH) {
    const { b, c, uS, vS } = this.frameFor(orient, offset01, aspectWH);
    const du = (u - 0.5) * uS, dv = (0.5 - v2) * vS;
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
    const enc2 = this.dev.createCommandEncoder();
    const pass = enc2.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind);
    pass.draw(3);
    pass.end();
    this.dev.queue.submit([enc2.finish()]);
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
    const enc2 = this.dev.createCommandEncoder();
    const pass = enc2.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(this.blitPipeline);
    pass.setBindGroup(0, this.blitBind);
    pass.draw(3);
    pass.end();
    this.dev.queue.submit([enc2.finish()]);
  }
  async renderToRGBA(w, h) {
    const target = this.dev.createTexture({ size: [w, h], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    this.drawInto(target.createView(), w, h);
    const bpr = Math.ceil(w * 4 / 256) * 256;
    const buf = this.dev.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc2 = this.dev.createCommandEncoder();
    enc2.copyTextureToBuffer({ texture: target }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: h }, [w, h]);
    this.dev.queue.submit([enc2.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) out.set(padded.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
    buf.unmap();
    target.destroy();
    buf.destroy();
    return out;
  }
};

// render/slice-interactor.ts
var NORMAL = {
  axial: { axis: 2, sign: 1 },
  // sliceToRAS col2 = +S
  coronal: { axis: 1, sign: 1 },
  // sliceToRAS col2 = +A
  sagittal: { axis: 0, sign: -1 }
  // sliceToRAS col2 = -R
};
function ijkAxisForRasAxis2(ijkToRAS, rasAxis) {
  let best = 0, bestMag = -1;
  for (let c = 0; c < 3; c++) {
    const mag = Math.abs(ijkToRAS[rasAxis * 4 + c]);
    if (mag > bestMag) {
      bestMag = mag;
      best = c;
    }
  }
  return best;
}
function sliceSpacingFor(orient, ijkToRAS) {
  const n = NORMAL[orient].axis;
  const a = ijkAxisForRasAxis2(ijkToRAS, n);
  return Math.hypot(ijkToRAS[a], ijkToRAS[4 + a], ijkToRAS[8 + a]);
}
function sliceBoundsFor(orient, rasLo, rasHi) {
  const { axis, sign } = NORMAL[orient];
  return sign > 0 ? [rasLo[axis], rasHi[axis]] : [-rasHi[axis], -rasLo[axis]];
}
function offset01ToMm(orient, offset01, rasLo, rasHi) {
  const { axis, sign } = NORMAL[orient];
  return sign * (rasLo[axis] + offset01 * (rasHi[axis] - rasLo[axis]));
}
function mmToOffset01(orient, mm, rasLo, rasHi) {
  const { axis, sign } = NORMAL[orient];
  const ras = sign * mm;
  const span = rasHi[axis] - rasLo[axis];
  return span === 0 ? 0.5 : (ras - rasLo[axis]) / span;
}
var SliceInteractor = class {
  constructor(geom) {
    this.geom = geom;
  }
  setGeometry(g) {
    this.geom = g;
  }
  spacing(orient) {
    return sliceSpacingFor(orient, this.geom.ijkToRAS);
  }
  bounds(orient) {
    return sliceBoundsFor(orient, this.geom.rasLo, this.geom.rasHi);
  }
  /** vtkMRMLSliceIntersectionWidget::MoveSlice — returns the NEW offset01, or the
   *  unchanged one if the step would leave the slice bounds (Slicer rejects, not clamps). */
  moveSlice(orient, offset01, deltaMm) {
    const { rasLo, rasHi } = this.geom;
    const cur = offset01ToMm(orient, offset01, rasLo, rasHi);
    const next = cur + deltaMm;
    const [lo, hi2] = this.bounds(orient);
    if (next < lo || next > hi2) return offset01;
    return mmToOffset01(orient, next, rasLo, rasHi);
  }
  incrementSlice(orient, offset01) {
    return this.moveSlice(orient, offset01, this.spacing(orient));
  }
  decrementSlice(orient, offset01) {
    return this.moveSlice(orient, offset01, -this.spacing(orient));
  }
  /** Map a wheel event to a step. Returns the new offset01. */
  wheel(orient, offset01, forward) {
    return forward ? this.incrementSlice(orient, offset01) : this.decrementSlice(orient, offset01);
  }
  /** Slicer's slice-view keyboard bindings. Returns the new offset01 (unchanged if the
   *  key isn't a stepping key). `key` is a DOM KeyboardEvent.key value. */
  key(orient, offset01, key) {
    switch (key) {
      case "f":
      case "F":
      case "ArrowRight":
      case "ArrowUp":
        return this.incrementSlice(orient, offset01);
      case "b":
      case "B":
      case "ArrowLeft":
      case "ArrowDown":
        return this.decrementSlice(orient, offset01);
      default:
        return offset01;
    }
  }
  /** True if this key is one Slicer's slice view consumes for stepping. */
  static isStepKey(key) {
    return ["f", "F", "b", "B", "ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown"].includes(key);
  }
};

// render/scene-renderer.ts
var DEFAULT_FORMAT2 = "rgba8unorm-srgb";
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
  meshPass(enc2, w, h) {
    const t = this.meshTargets(w, h);
    const pass = enc2.beginRenderPass({
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
  constructor(gpu, format = DEFAULT_FORMAT2) {
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
    const enc2 = this.dev.createCommandEncoder();
    const mb = this.meshPass(enc2, renderW, renderH);
    const tp = enc2.beginRenderPass({ colorAttachments: [{ view: this.lowView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.pipeline);
    tp.setBindGroup(0, this.bind);
    tp.setBindGroup(1, mb);
    tp.draw(3);
    tp.end();
    const sp = enc2.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    sp.setPipeline(this.superresPipeline);
    sp.setBindGroup(0, this.superresBind);
    sp.draw(3);
    sp.end();
    this.dev.queue.submit([enc2.finish()]);
  }
  /** Encode trace (producer) + resolve (reconstructor) into `enc`, output to `outView`. */
  encodeFrame(enc2, outView) {
    const mb = this.meshPass(enc2, this.traceW, this.traceH);
    const tp = enc2.beginRenderPass({ colorAttachments: [{ view: this.traceView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.pipeline);
    tp.setBindGroup(0, this.bind);
    tp.setBindGroup(1, mb);
    tp.draw(3);
    tp.end();
    const rp = enc2.beginRenderPass({ colorAttachments: [{ view: outView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
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
    for (let k2 = 0; k2 < 2; k2++) {
      this.accumTex[k2]?.destroy();
      this.accumTex[k2] = this.dev.createTexture({ size: [width, height], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      this.accumView[k2] = this.accumTex[k2].createView();
    }
    for (let k2 = 0; k2 < 2; k2++) {
      this.accumBind[k2] = this.dev.createBindGroup({
        layout: this.accumPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.traceView },
          { binding: 1, resource: this.accumView[k2] },
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
    const enc2 = this.dev.createCommandEncoder();
    const mb = this.meshPass(enc2, width, height);
    const tp = enc2.beginRenderPass({ colorAttachments: [{ view: this.traceView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.pipeline);
    tp.setBindGroup(0, this.bind);
    tp.setBindGroup(1, mb);
    tp.draw(3);
    tp.end();
    const ap = enc2.beginRenderPass({ colorAttachments: [
      { view: this.accumView[next], loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } },
      { view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }
    ] });
    ap.setPipeline(this.accumPipeline);
    ap.setBindGroup(0, this.accumBind[prev]);
    ap.draw(3);
    ap.end();
    this.dev.queue.submit([enc2.finish()]);
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
    const wf = (v2) => (Number.isFinite(v2) ? v2 : 0).toFixed(6);
    const boxSkipWGSL = (p) => {
      const [lo, hi2] = p.field.aabb();
      return `
fn skip_${p.field.kind}${p.slot}(wp : vec3<f32>) -> f32 {
  let q = max(vec3<f32>(${wf(lo[0])}, ${wf(lo[1])}, ${wf(lo[2])}) - wp,
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
  setClipBox(lo, hi2) {
    this.setClipPlanes([
      [1, 0, 0, -lo[0]],
      [-1, 0, 0, hi2[0]],
      // keep lo.x <= x <= hi.x
      [0, 1, 0, -lo[1]],
      [0, -1, 0, hi2[1]],
      [0, 0, 1, -lo[2]],
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
  async pick(u, v2) {
    if (!this.pickPipeline || !this.pickBind || !this.placed.length) return null;
    return this.serialise(async () => {
      this.mat[this.pickOff] = u * 2 - 1;
      this.mat[this.pickOff + 1] = 1 - v2 * 2;
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
  serialise(fn) {
    const next = this.pickChain.then(fn, fn);
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
    const enc2 = this.dev.createCommandEncoder();
    const pass = enc2.beginRenderPass({ colorAttachments: [{ view: this.pickTarget.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    pass.setPipeline(this.pickPipeline);
    pass.setBindGroup(0, this.pickBind);
    pass.draw(3);
    pass.end();
    enc2.copyTextureToBuffer({ texture: this.pickTarget }, { buffer: this.pickReadBuf, bytesPerRow: 256, rowsPerImage: 1 }, [1, 1]);
    this.dev.queue.submit([enc2.finish()]);
    await this.pickReadBuf.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(this.pickReadBuf.getMappedRange().slice(0, 16));
    this.pickReadBuf.unmap();
    return r[3] > 0.5 ? [r[0], r[1], r[2]] : null;
  }
  renderToView(view, width, height) {
    this.ensureTrace(width, height);
    this.flush();
    this.dev.queue.writeBuffer(this.resolveBgBuf, 0, this.mat.subarray(12, 16));
    const enc2 = this.dev.createCommandEncoder();
    this.encodeFrame(enc2, view);
    this.dev.queue.submit([enc2.finish()]);
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
    const qs = this.dev.createQuerySet({ type: "timestamp", count: 2 });
    const resolve = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const samples = [];
    for (let i = 0; i < iters; i++) {
      const enc2 = this.dev.createCommandEncoder();
      const mb = this.meshPass(enc2, width, height);
      const pass = enc2.beginRenderPass({
        colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
        timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bind);
      pass.setBindGroup(1, mb);
      pass.draw(3);
      pass.end();
      enc2.resolveQuerySet(qs, 0, 2, resolve, 0);
      enc2.copyBufferToBuffer(resolve, 0, read, 0, 16);
      this.dev.queue.submit([enc2.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(read.getMappedRange());
      const ms = Number(t[1] - t[0]) / 1e6;
      read.unmap();
      if (ms > 0 && Number.isFinite(ms)) samples.push(ms);
    }
    target.destroy();
    qs.destroy();
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
    const enc2 = this.dev.createCommandEncoder();
    this.encodeFrame(enc2, target.createView());
    const bpr = Math.ceil(width * 4 / 256) * 256;
    const buf = this.dev.createBuffer({ size: bpr * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc2.copyTextureToBuffer({ texture: target }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: height }, [width, height]);
    this.dev.queue.submit([enc2.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) out.set(padded.subarray(y * bpr, y * bpr + width * 4), y * width * 4);
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
    const enc2 = this.dev.createCommandEncoder();
    this.meshPass(enc2, width, height);
    const smt = this.meshTargets(width, height);
    const streamMb = this.dev.createBindGroup({ layout: this.streamPipeline.getBindGroupLayout(1), entries: [{ binding: 0, resource: smt.col.createView() }, { binding: 1, resource: smt.depth.createView() }] });
    const tp = enc2.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.streamPipeline);
    tp.setBindGroup(0, this.streamBind);
    tp.setBindGroup(1, streamMb);
    tp.draw(3);
    tp.end();
    const bpr = Math.ceil(width * 4 / 256) * 256;
    const buf = this.dev.createBuffer({ size: bpr * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc2.copyTextureToBuffer({ texture: target }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: height }, [width, height]);
    this.dev.queue.submit([enc2.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) out.set(padded.subarray(y * bpr, y * bpr + width * 4), y * width * 4);
    buf.unmap();
    target.destroy();
    buf.destroy();
    return out;
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
function bakeColorizeRGBA(dev, labelmap, dims, palette, sigmaVoxels = 1.5) {
  const [dx, dy, dz] = dims;
  const storageUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
  const labelTex = dev.createTexture({ size: dims, dimension: "3d", format: "r8uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  dev.queue.writeTexture({ texture: labelTex }, labelmap, { bytesPerRow: dx, rowsPerImage: dy }, dims);
  const texA = dev.createTexture({ size: dims, dimension: "3d", format: "rgba16float", usage: storageUsage });
  const texB = dev.createTexture({ size: dims, dimension: "3d", format: "rgba16float", usage: storageUsage });
  const palBuf = dev.createBuffer({ size: 256 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const palData = new Float32Array(256 * 4);
  palData.set(palette.subarray(0, Math.min(palette.length, 256 * 4)));
  dev.queue.writeBuffer(palBuf, 0, palData);
  const dimsBuf = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(dimsBuf, 0, new Uint32Array([dx, dy, dz, 0]));
  const gx = Math.ceil(dx / 4), gy = Math.ceil(dy / 4), gz = Math.ceil(dz / 4);
  const initPipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: INIT_WGSL }), entryPoint: "main" } });
  const initBind = dev.createBindGroup({ layout: initPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: labelTex.createView() },
    { binding: 1, resource: texA.createView() },
    { binding: 2, resource: { buffer: palBuf } },
    { binding: 3, resource: { buffer: dimsBuf } }
  ] });
  const enc2 = dev.createCommandEncoder();
  {
    const p = enc2.beginComputePass();
    p.setPipeline(initPipe);
    p.setBindGroup(0, initBind);
    p.dispatchWorkgroups(gx, gy, gz);
    p.end();
  }
  if (sigmaVoxels <= 0) {
    dev.queue.submit([enc2.finish()]);
    labelTex.destroy();
    texB.destroy();
    return texA;
  }
  const { radius, w } = gaussHalfKernel(sigmaVoxels);
  const blurPipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: BLUR_WGSL }), entryPoint: "main" } });
  const passes = [[texA, texB, 0], [texB, texA, 1], [texA, texB, 2]];
  for (const [src, dst, axis] of passes) {
    const ub = dev.createBuffer({ size: 16 + 16 + 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(ub, 0, new Uint32Array([dx, dy, dz, 0, axis, radius, 0, 0]));
    dev.queue.writeBuffer(ub, 32, w);
    const b = dev.createBindGroup({ layout: blurPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: src.createView() },
      { binding: 1, resource: dst.createView() },
      { binding: 2, resource: { buffer: ub } }
    ] });
    const p = enc2.beginComputePass();
    p.setPipeline(blurPipe);
    p.setBindGroup(0, b);
    p.dispatchWorkgroups(gx, gy, gz);
    p.end();
  }
  dev.queue.submit([enc2.finish()]);
  labelTex.destroy();
  texA.destroy();
  return texB;
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
    const enc2 = dev.createCommandEncoder();
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
      const p = enc2.beginComputePass();
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
        const p = enc2.beginComputePass();
        p.setPipeline(this.blurPipe);
        p.setBindGroup(0, b);
        p.dispatchWorkgroups(gx, gy, gz);
        p.end();
      }
    }
    dev.queue.submit([enc2.finish()]);
  }
  destroy() {
    if (this.ownsLabel) this.labelTex.destroy();
    this.scratch?.destroy();
    this.palBuf.destroy();
    this.dimsBuf.destroy();
  }
};

// render/fiducial-field.ts
var MAX = 64;
var FiducialField = class {
  kind = "fid";
  bindingCount = 0;
  // procedural — all state lives in the uniform block
  spheres = new Float32Array(MAX * 4);
  // (cx,cy,cz,radius)
  colors = new Float32Array(MAX * 4);
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
    this.n = Math.min(list.length, MAX);
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
    return 12 + MAX * 4 * 2;
  }
  // params(4)+params2(4)+light(4) + spheres + colors
  sampleStep() {
    return 1;
  }
  aabb() {
    if (this.n === 0) return [[-1, -1, -1], [1, 1, 1]];
    const lo = [Infinity, Infinity, Infinity], hi2 = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < this.n; i++) {
      const r = this.screen ? 0 : this.spheres[i * 4 + 3];
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], this.spheres[i * 4 + a] - r);
        hi2[a] = Math.max(hi2[a], this.spheres[i * 4 + a] + r);
      }
    }
    if (this.screen) {
      const diag = Math.hypot(hi2[0] - lo[0], hi2[1] - lo[1], hi2[2] - lo[2]);
      const m = Math.max(40, diag * 0.15);
      for (let a = 0; a < 3; a++) {
        lo[a] -= m;
        hi2[a] += m;
      }
    }
    return [lo, hi2];
  }
  structMembers(s) {
    return [
      `  fid${s}_params : vec4<f32>,`,
      // n_spheres, visible, shininess, k_ambient
      `  fid${s}_params2 : vec4<f32>,`,
      // k_diffuse, k_specular, max_radius, _
      `  fid${s}_light : vec4<f32>,`,
      // light_color.rgb, _
      `  fid${s}_spheres : array<vec4<f32>, ${MAX}>,`,
      `  fid${s}_colors : array<vec4<f32>, ${MAX}>,`
    ].join("\n");
  }
  declareBindings(_s, _base) {
    return "";
  }
  bindEntries(_s, _base) {
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
    out.set(this.colors, off + 12 + MAX * 4);
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

// render/demos/roi-widget.ts
function createRoiWidget(lo, hi2, opts = {}) {
  const MIN_HALF = opts.minHalfMm ?? 5;
  const cov = opts.coverage ?? 0.35;
  const center = [(lo[0] + hi2[0]) / 2, (lo[1] + hi2[1]) / 2, (lo[2] + hi2[2]) / 2];
  const half = [(hi2[0] - lo[0]) * cov, (hi2[1] - lo[1]) * cov, (hi2[2] - lo[2]) * cov];
  const hR = Math.max(3, Math.hypot(hi2[0] - lo[0], hi2[1] - lo[1], hi2[2] - lo[2]) * 0.012);
  const bar = Math.max(1.2, hR * 0.35);
  const box = new RoiBoxField(center, half, { color: [1, 0.85, 0.25], barHalfMm: bar });
  const handles = new FiducialField([], { shininess: 60, kSpecular: 0.4, clippable: false, screenSpace: true, ghost: true });
  let hover = null;
  const metas = [];
  for (let axis = 0; axis < 3; axis++) for (const sign of [-1, 1]) metas.push({ kind: "face", axis, sign });
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) metas.push({ kind: "corner", s: [sx, sy, sz] });
  metas.push({ kind: "center" });
  const worldOf = (m) => {
    if (m.kind === "center") return [...center];
    if (m.kind === "face") {
      const w = [...center];
      w[m.axis] += m.sign * half[m.axis];
      return w;
    }
    return [center[0] + m.s[0] * half[0], center[1] + m.s[1] * half[1], center[2] + m.s[2] * half[2]];
  };
  const refreshHandles = () => {
    const pins = metas.map((m, i) => {
      const on2 = i === hover;
      const base = m.kind === "center" ? [0.4, 1, 0.5] : [0.35, 0.8, 1];
      return { center: worldOf(m), radius: on2 ? 13 : 8, color: on2 ? [1, 0.9, 0.3, 1] : [base[0], base[1], base[2], 0.5] };
    });
    handles.setSpheres(pins);
  };
  refreshHandles();
  const moveFace = (axis, sign, box0, deltaAxis) => {
    const opp = box0.center[axis] - sign * box0.half[axis];
    let face = box0.center[axis] + sign * box0.half[axis] + deltaAxis;
    face = sign > 0 ? Math.max(face, opp + 2 * MIN_HALF) : Math.min(face, opp - 2 * MIN_HALF);
    return [(face + opp) / 2, Math.abs(face - opp) / 2];
  };
  return {
    box,
    handles,
    center,
    half,
    lo: () => [center[0] - half[0], center[1] - half[1], center[2] - half[2]],
    hi: () => [center[0] + half[0], center[1] + half[1], center[2] + half[2]],
    handleList: () => metas.map((m, i) => ({
      id: i,
      world: worldOf(m),
      data: m,
      cursor: m.kind === "center" ? "move" : "grab"
    })),
    applyDrag(meta, box0, delta) {
      if (meta.kind === "center") {
        for (let a = 0; a < 3; a++) center[a] = box0.center[a] + delta[a];
      } else if (meta.kind === "face") {
        const [c, h] = moveFace(meta.axis, meta.sign, box0, delta[meta.axis]);
        center[meta.axis] = c;
        half[meta.axis] = h;
      } else {
        for (let a = 0; a < 3; a++) {
          const [c, h] = moveFace(a, meta.s[a], box0, delta[a]);
          center[a] = c;
          half[a] = h;
        }
      }
      box.setBox(center, half);
      refreshHandles();
    },
    setHover(i) {
      hover = i;
      refreshHandles();
    },
    snapshot: () => ({ center: [...center], half: [...half] }),
    setBox(c, h) {
      for (let a = 0; a < 3; a++) {
        center[a] = c[a];
        half[a] = h[a];
      }
      box.setBox(center, half);
      refreshHandles();
    }
  };
}

// algorithms/geom.ts
function labelmapHasInternalBoundary(lab, dims) {
  const [nx, ny, nz] = dims;
  const at = (x, y, z2) => lab[(z2 * ny + y) * nx + x];
  for (let z2 = 0; z2 < nz; z2++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const v2 = at(x, y, z2);
    if (v2 === 0) continue;
    if (x + 1 < nx) {
      const n = at(x + 1, y, z2);
      if (n !== 0 && n !== v2) return true;
    }
    if (y + 1 < ny) {
      const n = at(x, y + 1, z2);
      if (n !== 0 && n !== v2) return true;
    }
    if (z2 + 1 < nz) {
      const n = at(x, y, z2 + 1);
      if (n !== 0 && n !== v2) return true;
    }
  }
  return false;
}
function resampleIsotropic(lab, dims, ijkToRAS, maxDim, maxVoxels = Infinity) {
  const sp = spacingFromIjkToRAS2(ijkToRAS);
  const ext = [dims[0] * sp[0], dims[1] * sp[1], dims[2] * sp[2]];
  let vox = Math.max(...ext) / maxDim;
  const count = (v2) => Math.max(1, Math.round(ext[0] / v2)) * Math.max(1, Math.round(ext[1] / v2)) * Math.max(1, Math.round(ext[2] / v2));
  let coarsened = false;
  if (count(vox) > maxVoxels) {
    vox = Math.cbrt(ext[0] * ext[1] * ext[2] / maxVoxels);
    coarsened = true;
  }
  const cd = [Math.max(1, Math.round(ext[0] / vox)), Math.max(1, Math.round(ext[1] / vox)), Math.max(1, Math.round(ext[2] / vox))];
  const [nx, ny, nz] = dims, [cx, cy, cz] = cd;
  const out = new Uint32Array(cx * cy * cz);
  if (cx === nx && cy === ny && cz === nz) {
    for (let i = 0; i < out.length; i++) out[i] = lab[i];
    return { lab: out, dims, ijkToRAS, vox, coarsened };
  }
  for (let z2 = 0; z2 < cz; z2++) {
    const sz = Math.min(nz - 1, Math.floor((z2 + 0.5) * nz / cz));
    for (let y = 0; y < cy; y++) {
      const sy = Math.min(ny - 1, Math.floor((y + 0.5) * ny / cy));
      for (let x = 0; x < cx; x++) {
        const sx = Math.min(nx - 1, Math.floor((x + 0.5) * nx / cx));
        out[(z2 * cy + y) * cx + x] = lab[(sz * ny + sy) * nx + sx];
      }
    }
  }
  const r = [nx / cx, ny / cy, nz / cz], m = ijkToRAS.slice();
  for (let row = 0; row < 3; row++) {
    m[row * 4] *= r[0];
    m[row * 4 + 1] *= r[1];
    m[row * 4 + 2] *= r[2];
  }
  return { lab: out, dims: cd, ijkToRAS: m, vox, coarsened };
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
    const u322 = data instanceof Uint32Array ? data : Uint32Array.from(data);
    this.device.queue.writeTexture({ texture: this.labelTex }, u322, { bytesPerRow: dx * 4, rowsPerImage: dy }, [dx, dy, dz]);
    this.markDirty();
  }
  /** Region-limited labelmap write (`lo`/`size` in label-grid ijk; data x-fastest, tightly packed).
   *  Deliberately NO dirty notification: the caller pairs it with a region-limited rebake
   *  (SegmentationLogic.rebakeShellRegion) — an onDirty full rebake would defeat the point. */
  writeLabelRegion(data, lo, size) {
    this.device.queue.writeTexture(
      { texture: this.labelTex, origin: lo },
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
    const enc2 = this.device.createCommandEncoder();
    enc2.copyTextureToBuffer({ texture: this.labelTex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: dy }, [dx, dy, dz]);
    this.device.queue.submit([enc2.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint32Array(buf.getMappedRange());
    const out = new Uint32Array(dx * dy * dz);
    for (let z2 = 0; z2 < dz; z2++) for (let y = 0; y < dy; y++) {
      const src = (z2 * dy + y) * rowU32, dst = (z2 * dy + y) * dx;
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

// render/sdf-bake.ts
var INIT_WGSL2 = (
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
var FINAL_WGSL = (
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
    const mod = (code2) => dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: code2 }), entryPoint: "main" } });
    this.initPipe = mod(INIT_WGSL2);
    this.jfaPipe = mod(JFA_WGSL);
    this.finalPipe = mod(FINAL_WGSL);
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
    const enc2 = this.dev.createCommandEncoder();
    enc2.copyTextureToBuffer({ texture: this.sdfTex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: dy }, [dx, dy, dz]);
    this.dev.queue.submit([enc2.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const u162 = new Uint16Array(buf.getMappedRange());
    const h2f = (h) => {
      const s = h & 32768 ? -1 : 1, e = (h & 31744) >> 10, f = h & 1023;
      if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
      if (e === 31) return f ? NaN : s * Infinity;
      return s * Math.pow(2, e - 15) * (1 + f / 1024);
    };
    const out = new Float32Array(dx * dy * dz);
    for (let z2 = 0; z2 < dz; z2++) for (let y = 0; y < dy; y++) for (let x = 0; x < dx; x++) {
      out[(z2 * dy + y) * dx + x] = h2f(u162[(z2 * dy + y) * rowU16 + x * 4 + 3]);
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
    const lo = [Math.max(0, regionIjk.lo[0]), Math.max(0, regionIjk.lo[1]), Math.max(0, regionIjk.lo[2])];
    const hi2 = [Math.min(dx, regionIjk.hi[0]), Math.min(dy, regionIjk.hi[1]), Math.min(dz, regionIjk.hi[2])];
    const [rx, ry, rz] = [hi2[0] - lo[0], hi2[1] - lo[1], hi2[2] - lo[2]];
    if (rx <= 0 || ry <= 0 || rz <= 0) return;
    const g = [Math.ceil(rx / 4), Math.ceil(ry / 4), Math.ceil(rz / 4)];
    const region = { lo, hi: hi2 };
    let src = this.lastSeed;
    let enc2 = dev.createCommandEncoder();
    enc2.copyTextureToTexture({ texture: this.seed[src] }, { texture: this.seed[src ^ 1] }, this.dims);
    dev.queue.submit([enc2.finish()]);
    this.writeUni(0, lo);
    enc2 = dev.createCommandEncoder();
    {
      const b = dev.createBindGroup({ layout: this.initPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.labelTex.createView() },
        { binding: 1, resource: this.seed[src].createView() },
        { binding: 2, resource: { buffer: this.uni } }
      ] });
      const p2 = enc2.beginComputePass();
      p2.setPipeline(this.initPipe);
      p2.setBindGroup(0, b);
      p2.dispatchWorkgroups(g[0], g[1], g[2]);
      p2.end();
    }
    dev.queue.submit([enc2.finish()]);
    const maxDim = Math.max(rx, ry, rz);
    const steps = [];
    for (let s = 1 << Math.floor(Math.log2(Math.max(2, maxDim - 1))); s >= 1; s >>= 1) steps.push(s);
    steps.push(2, 1);
    for (const step of steps) {
      this.writeUni(step, lo);
      const dst = src ^ 1;
      enc2 = dev.createCommandEncoder();
      const b = dev.createBindGroup({ layout: this.jfaPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.seed[src].createView() },
        { binding: 1, resource: this.seed[dst].createView() },
        { binding: 2, resource: { buffer: this.uni } }
      ] });
      const p2 = enc2.beginComputePass();
      p2.setPipeline(this.jfaPipe);
      p2.setBindGroup(0, b);
      p2.dispatchWorkgroups(g[0], g[1], g[2]);
      p2.end();
      dev.queue.submit([enc2.finish()]);
      src = dst;
    }
    this.lastSeed = src;
    this.writeUni(0, lo);
    enc2 = dev.createCommandEncoder();
    const bf = dev.createBindGroup({ layout: this.finalPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.seed[src].createView() },
      { binding: 1, resource: this.labelTex.createView() },
      { binding: 2, resource: this.sdfTex.createView() },
      { binding: 3, resource: { buffer: this.uni } },
      { binding: 4, resource: { buffer: this.palBuf } },
      { binding: 5, resource: this.attrTex.createView() },
      { binding: 6, resource: { buffer: this.modeBuf } }
    ] });
    const p = enc2.beginComputePass();
    p.setPipeline(this.finalPipe);
    p.setBindGroup(0, bf);
    p.dispatchWorkgroups(g[0], g[1], g[2]);
    p.end();
    dev.queue.submit([enc2.finish()]);
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
    const enc2 = dev.createCommandEncoder();
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
    const p = enc2.beginComputePass();
    p.setPipeline(this.finalPipe);
    p.setBindGroup(0, bf);
    p.dispatchWorkgroups(gx, gy, gz);
    p.end();
    dev.queue.submit([enc2.finish()]);
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
    let enc2 = dev.createCommandEncoder();
    {
      const b = dev.createBindGroup({ layout: this.initPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.labelTex.createView() },
        { binding: 1, resource: this.seed[0].createView() },
        { binding: 2, resource: { buffer: this.uni } }
      ] });
      const p2 = enc2.beginComputePass();
      p2.setPipeline(this.initPipe);
      p2.setBindGroup(0, b);
      p2.dispatchWorkgroups(gx, gy, gz);
      p2.end();
    }
    dev.queue.submit([enc2.finish()]);
    let src = 0;
    for (const step of [...this.steps, ...extraSteps]) {
      this.writeUni(step);
      const dst = src ^ 1;
      enc2 = dev.createCommandEncoder();
      const b = dev.createBindGroup({ layout: this.jfaPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.seed[src].createView() },
        { binding: 1, resource: this.seed[dst].createView() },
        { binding: 2, resource: { buffer: this.uni } }
      ] });
      const p2 = enc2.beginComputePass();
      p2.setPipeline(this.jfaPipe);
      p2.setBindGroup(0, b);
      p2.dispatchWorkgroups(gx, gy, gz);
      p2.end();
      dev.queue.submit([enc2.finish()]);
      src = dst;
    }
    this.lastSeed = src;
    enc2 = dev.createCommandEncoder();
    const bf = dev.createBindGroup({ layout: this.finalPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.seed[src].createView() },
      { binding: 1, resource: this.labelTex.createView() },
      { binding: 2, resource: this.sdfTex.createView() },
      { binding: 3, resource: { buffer: this.uni } },
      { binding: 4, resource: { buffer: this.palBuf } },
      { binding: 5, resource: this.attrTex.createView() },
      { binding: 6, resource: { buffer: this.modeBuf } }
    ] });
    const p = enc2.beginComputePass();
    p.setPipeline(this.finalPipe);
    p.setBindGroup(0, bf);
    p.dispatchWorkgroups(gx, gy, gz);
    p.end();
    dev.queue.submit([enc2.finish()]);
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
    const enc2 = dev.createCommandEncoder();
    let passIdx = 0;
    for (const [srcT, dstT, axis] of passes) {
      const expand = region ? (2 - passIdx) * radius : 0;
      const lo = region ? [Math.max(0, region.lo[0] - expand), Math.max(0, region.lo[1] - expand), Math.max(0, region.lo[2] - expand)] : [0, 0, 0];
      const hi2 = region ? [Math.min(dx, region.hi[0] + expand), Math.min(dy, region.hi[1] + expand), Math.min(dz, region.hi[2] + expand)] : [dx, dy, dz];
      const ab = new ArrayBuffer(112);
      const u322 = new Uint32Array(ab), f32 = new Float32Array(ab), i32 = new Int32Array(ab);
      u322[0] = dx;
      u322[1] = dy;
      u322[2] = dz;
      u322[4] = axis;
      u322[5] = radius;
      f32.set(w, 8);
      i32[24] = lo[0];
      i32[25] = lo[1];
      i32[26] = lo[2];
      const ub = dev.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(ub, 0, ab);
      const b = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: srcT.createView() },
        { binding: 1, resource: dstT.createView() },
        { binding: 2, resource: { buffer: ub } }
      ] });
      const bp = enc2.beginComputePass();
      bp.setPipeline(pipe);
      bp.setBindGroup(0, b);
      bp.dispatchWorkgroups(Math.ceil((hi2[0] - lo[0]) / 4), Math.ceil((hi2[1] - lo[1]) / 4), Math.ceil((hi2[2] - lo[2]) / 4));
      bp.end();
      passIdx++;
    }
    if (region) {
      const sz = [region.hi[0] - region.lo[0], region.hi[1] - region.lo[1], region.hi[2] - region.lo[2]];
      enc2.copyTextureToTexture({ texture: scratch, origin: region.lo }, { texture: tex, origin: region.lo }, sz);
    } else {
      enc2.copyTextureToTexture({ texture: scratch }, { texture: tex }, this.dims);
    }
    dev.queue.submit([enc2.finish()]);
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
  regionToIjk(regionRAS, M2 = 8) {
    const inv = invertAffine(this.sdf.sdfIjkToRAS());
    const lo = [Infinity, Infinity, Infinity];
    const hi2 = [-Infinity, -Infinity, -Infinity];
    for (const x of [regionRAS.lo[0], regionRAS.hi[0]]) for (const y of [regionRAS.lo[1], regionRAS.hi[1]]) for (const z2 of [regionRAS.lo[2], regionRAS.hi[2]]) {
      const i = inv[0] * x + inv[1] * y + inv[2] * z2 + inv[3];
      const j2 = inv[4] * x + inv[5] * y + inv[6] * z2 + inv[7];
      const k2 = inv[8] * x + inv[9] * y + inv[10] * z2 + inv[11];
      lo[0] = Math.min(lo[0], i);
      lo[1] = Math.min(lo[1], j2);
      lo[2] = Math.min(lo[2], k2);
      hi2[0] = Math.max(hi2[0], i);
      hi2[1] = Math.max(hi2[1], j2);
      hi2[2] = Math.max(hi2[2], k2);
    }
    return {
      lo: [Math.floor(lo[0]) - M2, Math.floor(lo[1]) - M2, Math.floor(lo[2]) - M2],
      hi: [Math.ceil(hi2[0]) + M2, Math.ceil(hi2[1]) + M2, Math.ceil(hi2[2]) + M2]
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

// render/demos/segroulette-scene.ts
var SDF_MAX_DIM = 256;
function modalityLUT(modality, maxAlpha = 0.42) {
  const lut = new Uint8Array(256 * 4);
  const m = (modality ?? "CT").toUpperCase();
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let r, g, b, a;
    if (m === "PET" || m === "PT") {
      r = Math.min(1, t * 3);
      g = Math.min(1, Math.max(0, t * 3 - 1));
      b = Math.min(1, Math.max(0, t * 3 - 2));
      a = Math.max(0, (t - 0.25) / 0.75) * 0.9;
    } else {
      r = g = b = t;
      let aa = Math.max(0, (t - 0.42) / 0.58);
      aa *= aa;
      a = Math.min(maxAlpha, aa);
    }
    lut[i * 4] = Math.round(r * 255);
    lut[i * 4 + 1] = Math.round(g * 255);
    lut[i * 4 + 2] = Math.round(b * 255);
    lut[i * 4 + 3] = Math.round(a * 255);
  }
  return lut;
}
function buildSegrouletteScene(gpu, format, ct, seg, opts = {}) {
  const dev = gpu.device;
  const dims = ct.dims;
  const data = ct.vol instanceof Float32Array ? ct.vol : Float32Array.from(ct.vol);
  const clim = [ct.lev - ct.win / 2, ct.lev + ct.win / 2];
  const grayLut = modalityLUT(ct.modality);
  const grayClim = [clim[0], clim[1]];
  const grayShade = [0.25, 0.7, 0.45, 20];
  let baseVolLut = grayLut;
  let baseClim = [clim[0], clim[1]];
  let baseShade = [grayShade[0], grayShade[1], grayShade[2], grayShade[3]];
  let volShift = 0;
  const scaledVolLut = (o) => {
    const l = baseVolLut.slice();
    for (let i = 0; i < 256; i++) l[i * 4 + 3] = Math.round(l[i * 4 + 3] * o);
    return l;
  };
  let volumeField = opts.volumeField ?? new ImageField(dev, data, dims, [1, 1, 1], baseVolLut, {
    clim,
    ijkToRAS: ct.ijkToRAS,
    shade: baseShade
  });
  if (opts.volumeField) {
    volumeField.setLUT(baseVolLut);
    volumeField.setClim(baseClim[0], baseClim[1]);
    volumeField.setShade(baseShade);
  }
  const segments = [];
  const palette = new Float32Array(256 * 4);
  let activeSeg = seg;
  let volDims = [dims[0], dims[1], dims[2]];
  const fillSegmentsFrom = (labmap) => {
    segments.length = 0;
    palette.fill(0);
    const total = volDims[0] * volDims[1] * volDims[2];
    for (const [num2, r, g, b] of labmap.colors) {
      if (num2 === 0 || r === 0 && g === 0 && b === 0) continue;
      let n = 0;
      for (let i = 0; i < labmap.lab.length; i++) if (labmap.lab[i] === num2) n++;
      if (!n || n > total * 0.6) continue;
      if (num2 < 256) {
        palette[num2 * 4] = r;
        palette[num2 * 4 + 1] = g;
        palette[num2 * 4 + 2] = b;
        palette[num2 * 4 + 3] = 1;
      }
      segments.push({ num: num2, name: labmap.names[num2] ?? `Segment ${num2}`, color: [r, g, b], voxels: n });
    }
  };
  if (activeSeg) fillSegmentsFrom(activeSeg);
  const segOpacity = /* @__PURE__ */ new Map();
  const opacityOf = (num2) => segOpacity.get(num2) ?? 1;
  const visPalette = () => {
    const p = palette.slice();
    for (const s of segments) if (s.num < 256) p[s.num * 4 + 3] = opacityOf(s.num);
    return p;
  };
  let colorTex = activeSeg ? bakeColorizeRGBA(dev, activeSeg.lab, dims, visPalette(), 1.5) : void 0;
  let mode = "volume";
  let segLogic;
  let editable;
  const bakeSegSdf = (labmap) => {
    if (segments.length === 0) return;
    const cap = resampleIsotropic(labmap.lab, volDims, ct.ijkToRAS, opts.sdfMaxDim ?? SDF_MAX_DIM, opts.sdfMaxVoxels);
    if (cap.coarsened) console.warn(`SEGRoulette: SDF grid coarsened to fit the device memory budget (${cap.dims.join("\xD7")}, ${cap.vox.toFixed(2)}mm) \u2014 a remote render server would allow full resolution.`);
    editable = new EditableSegmentation(dev, cap.dims, { ijkToRAS: cap.ijkToRAS });
    const boundaryMode = labelmapHasInternalBoundary(cap.lab, cap.dims) ? "all" : "outer";
    segLogic = new SegmentationLogic(dev, editable, { renderMode: "sdf", opacity: 1, boundaryMode, refineDelayMs: opts.refineDelayMs });
    for (const s of segments) {
      segLogic.setLabelColor(s.num, s.color);
      segLogic.setLabelOpacity(s.num, opacityOf(s.num));
    }
    editable.loadLabelmap(cap.lab);
    segLogic.refineNow();
    mode = "sdf";
  };
  if (activeSeg && segments.length > 0) bakeSegSdf(activeSeg);
  let hasSeg = !!segLogic;
  const scene = new SceneRenderer(gpu, format);
  const [rasLo, rasHi] = volumeField.aabb();
  const roi = createRoiWidget(rasLo, rasHi, { coverage: 0.35 });
  let volumeOpacity = 1, segLayerOpacity = 1;
  let showVolume = true, showSeg = hasSeg;
  let roiEnabled = false, roiVisible = false;
  const currentSegFields = () => segLogic ? [segLogic.field()] : [];
  let extraFields = [];
  const rebuild = () => {
    const f = [];
    if (showVolume) f.push(volumeField);
    if (showSeg) f.push(...currentSegFields());
    if (roiVisible) {
      f.push(roi.box, roi.handles);
    }
    f.push(...extraFields);
    scene.build(f);
    scene.setBackground(0.05, 0.06, 0.09);
    if (roiEnabled) scene.setClipBox(roi.lo(), roi.hi());
    else scene.clearClip();
  };
  rebuild();
  const rebakeColorized = () => {
    if (!activeSeg) return;
    const nt2 = bakeColorizeRGBA(dev, activeSeg.lab, volDims, visPalette(), 1.5);
    const old = colorTex;
    colorTex = nt2;
    slice.setTextures(volumeField.volumeTexture(), colorTex);
    old?.destroy();
  };
  const clearSegGpu = () => {
    try {
      segLogic?.destroy();
    } catch {
    }
    try {
      editable?.destroy();
    } catch {
    }
    segLogic = void 0;
    editable = void 0;
    segments.length = 0;
    palette.fill(0);
    segOpacity.clear();
    activeSeg = void 0;
    out.hasSeg = false;
    showSeg = false;
    mode = "volume";
    const old = colorTex;
    colorTex = void 0;
    if (old) {
      try {
        slice.setTextures(volumeField.volumeTexture(), void 0);
      } catch {
      }
      try {
        old.destroy();
      } catch {
      }
    }
    slice.setOverlayOpacity(0);
  };
  const mountSegLabelmap = (labmap) => {
    activeSeg = labmap;
    fillSegmentsFrom(labmap);
    if (segments.length === 0) {
      activeSeg = void 0;
      out.hasSeg = false;
      showSeg = false;
      rebuild();
      return;
    }
    bakeSegSdf(labmap);
    if (!segLogic) return;
    out.hasSeg = true;
    out.mode = mode;
    showSeg = segLayerOpacity > 1e-3;
    rebakeColorized();
    slice.setOverlayOpacity(0.5);
    rebuild();
  };
  const slice = new SliceRenderer(gpu, format);
  slice.setVolume(volumeField.patientToTexture(), rasLo, rasHi);
  slice.setTextures(volumeField.volumeTexture(), colorTex);
  slice.setWindowLevel(ct.win, ct.lev);
  slice.setOverlayOpacity(activeSeg ? 0.5 : 0);
  const center = [(rasLo[0] + rasHi[0]) / 2, (rasLo[1] + rasHi[1]) / 2, (rasLo[2] + rasHi[2]) / 2];
  const radius = Math.hypot(rasHi[0] - rasLo[0], rasHi[1] - rasLo[1], rasHi[2] - rasLo[2]) / 2;
  let out;
  out = {
    scene,
    slice,
    center,
    radius,
    rasLo,
    rasHi,
    ijkToRAS: ct.ijkToRAS,
    dims,
    win: ct.win,
    lev: ct.lev,
    segments,
    mode,
    hasSeg,
    roi,
    setLayers(sv, ss) {
      this.setVolumeOpacity(sv ? 1 : 0);
      this.setSegOpacity(ss ? 1 : 0);
    },
    setVolumeOpacity(o) {
      volumeOpacity = Math.max(0, Math.min(1, o));
      const was = showVolume;
      showVolume = volumeOpacity > 1e-3;
      volumeField.setLUT(scaledVolLut(volumeOpacity));
      if (was !== showVolume) rebuild();
    },
    volumeOpacity: () => volumeOpacity,
    setVolumePreset(bake) {
      if (bake) {
        baseVolLut = bake.lut;
        baseClim = [bake.clim[0], bake.clim[1]];
        baseShade = bake.shade;
      } else {
        baseVolLut = grayLut;
        baseClim = [grayClim[0], grayClim[1]];
        baseShade = grayShade;
      }
      volumeField.setLUT(scaledVolLut(volumeOpacity));
      volumeField.setClim(baseClim[0] + volShift, baseClim[1] + volShift);
      volumeField.setShade(baseShade);
      scene.syncUniforms();
    },
    setVolumeShift(hu) {
      volShift = hu;
      volumeField.setClim(baseClim[0] + volShift, baseClim[1] + volShift);
      scene.syncUniforms();
    },
    volumeShift: () => volShift,
    setSegOpacity(o) {
      segLayerOpacity = Math.max(0, Math.min(1, o));
      const was = showSeg;
      showSeg = out.hasSeg && segLayerOpacity > 1e-3;
      segLogic?.setGlobalOpacity(segLayerOpacity);
      if (was !== showSeg) rebuild();
      else scene.syncUniforms();
    },
    segOpacity: () => segLayerOpacity,
    setSegmentOpacity(num2, opacity) {
      const o = Math.max(0, Math.min(1, opacity));
      if (o >= 1) segOpacity.delete(num2);
      else segOpacity.set(num2, o);
      rebakeColorized();
      segLogic?.setLabelOpacity(num2, o);
      segLogic?.refineNow();
    },
    setSegmentOpacities(entries) {
      for (const [num2, opacity] of entries) {
        const o = Math.max(0, Math.min(1, opacity));
        if (o >= 1) segOpacity.delete(num2);
        else segOpacity.set(num2, o);
        segLogic?.setLabelOpacity(num2, o);
      }
      rebakeColorized();
      segLogic?.refineNow();
    },
    segmentOpacity: (num2) => opacityOf(num2),
    setSegmentVisible(num2, visible) {
      this.setSegmentOpacity(num2, visible ? 1 : 0);
    },
    isSegmentVisible: (num2) => opacityOf(num2) > 0,
    setRoiEnabled(on2) {
      roiEnabled = on2;
      rebuild();
    },
    setRoiVisible(on2) {
      roiVisible = on2;
      rebuild();
    },
    roiEnabled: () => roiEnabled,
    roiVisible: () => roiVisible,
    setExtraFields(fields) {
      extraFields = fields;
      rebuild();
    },
    patchVolumeScalars(data2, zStart, zCount) {
      volumeField.updateScalars(data2, zStart, zCount);
    },
    updateGeometry(m) {
      volumeField.setIjkToRAS(m);
      const [lo, hi2] = volumeField.aabb();
      rasLo[0] = lo[0];
      rasLo[1] = lo[1];
      rasLo[2] = lo[2];
      rasHi[0] = hi2[0];
      rasHi[1] = hi2[1];
      rasHi[2] = hi2[2];
      out.ijkToRAS = m;
      out.center = [(lo[0] + hi2[0]) / 2, (lo[1] + hi2[1]) / 2, (lo[2] + hi2[2]) / 2];
      out.radius = Math.hypot(hi2[0] - lo[0], hi2[1] - lo[1], hi2[2] - lo[2]) / 2;
      slice.setVolume(volumeField.patientToTexture(), rasLo, rasHi);
      scene.syncUniforms();
    },
    swapVolumeField(field, nextDims, ijkToRAS) {
      const old = volumeField;
      volumeField = field;
      volDims = [nextDims[0], nextDims[1], nextDims[2]];
      out.dims = volDims;
      out.ijkToRAS = ijkToRAS;
      volumeField.setLUT(scaledVolLut(volumeOpacity));
      volumeField.setClim(baseClim[0] + volShift, baseClim[1] + volShift);
      volumeField.setShade(baseShade);
      const [lo, hi2] = volumeField.aabb();
      rasLo[0] = lo[0];
      rasLo[1] = lo[1];
      rasLo[2] = lo[2];
      rasHi[0] = hi2[0];
      rasHi[1] = hi2[1];
      rasHi[2] = hi2[2];
      out.center = [(lo[0] + hi2[0]) / 2, (lo[1] + hi2[1]) / 2, (lo[2] + hi2[2]) / 2];
      out.radius = Math.hypot(hi2[0] - lo[0], hi2[1] - lo[1], hi2[2] - lo[2]) / 2;
      slice.setVolume(volumeField.patientToTexture(), rasLo, rasHi);
      slice.setTextures(volumeField.volumeTexture(), colorTex);
      rebuild();
      old.destroy();
    },
    attachSegmentation(labmap) {
      if (out.hasSeg || segLogic) return;
      mountSegLabelmap(labmap);
    },
    replaceSegmentation(labmap) {
      clearSegGpu();
      mountSegLabelmap(labmap);
    },
    labelmap: () => activeSeg,
    reclip() {
      if (roiEnabled) scene.setClipBox(roi.lo(), roi.hi());
      else scene.clearClip();
      scene.syncUniforms();
    },
    destroy() {
      clearSegGpu();
    }
  };
  return out;
}

// render/demos/crosshair.ts
function createCrosshair(visible = true) {
  const listeners = /* @__PURE__ */ new Set();
  const notify = () => {
    for (const cb of listeners) cb();
  };
  const st2 = {
    ras: null,
    visible,
    set(ras) {
      this.ras = ras;
      notify();
    },
    toggle(on2) {
      this.visible = on2 ?? !this.visible;
      notify();
    },
    onChange(cb) {
      listeners.add(cb);
      return () => listeners.delete(cb);
    }
  };
  return st2;
}
function drawCross(ctx, x, y, opts = {}) {
  const size = opts.size ?? 11, gap = opts.gap ?? 3;
  ctx.save();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = opts.color ?? "rgba(120,220,255,0.95)";
  ctx.shadowColor = "rgba(0,0,0,0.8)";
  ctx.shadowBlur = 2;
  ctx.beginPath();
  ctx.moveTo(x - size, y);
  ctx.lineTo(x - gap, y);
  ctx.moveTo(x + gap, y);
  ctx.lineTo(x + size, y);
  ctx.moveTo(x, y - size);
  ctx.lineTo(x, y - gap);
  ctx.moveTo(x, y + gap);
  ctx.lineTo(x, y + size);
  ctx.stroke();
  ctx.restore();
}
function rasToScreen3D(cam, ras, w, h) {
  const vp = multiply(perspectiveZO(cam.viewAngle * Math.PI / 180, w / h, 1, 1e5), lookAt(cam.position, cam.focalPoint, cam.viewUp));
  const cw = vp[3] * ras[0] + vp[7] * ras[1] + vp[11] * ras[2] + vp[15];
  if (cw <= 0) return null;
  return {
    x: (vp[0] * ras[0] + vp[4] * ras[1] + vp[8] * ras[2] + vp[12]) / cw * 0.5 + 0.5,
    y: 1 - ((vp[1] * ras[0] + vp[5] * ras[1] + vp[9] * ras[2] + vp[13]) / cw * 0.5 + 0.5)
  };
}
var uvOf = (canvas, e) => {
  const r = canvas.getBoundingClientRect();
  return { u: (e.clientX - r.left) / r.width, v: (e.clientY - r.top) / r.height, aspect: r.width / r.height };
};
var isShiftHover = (e) => e.shiftKey && e.buttons === 0;
function mountCrosshair(cfg) {
  const state = createCrosshair(cfg.visible ?? true);
  const slices = ["axial", "coronal", "sagittal"];
  const all = [...slices, "threeD"];
  const ctx = {};
  for (const cell of all) {
    const o = document.createElement("canvas");
    o.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;border-radius:6px;background:transparent;";
    cfg.cells[cell].parentElement.appendChild(o);
    ctx[cell] = { c: o, g: o.getContext("2d") };
  }
  const redraw = () => {
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    for (const cell of all) {
      const { c, g } = ctx[cell];
      const w = cfg.cells[cell].clientWidth, h = cfg.cells[cell].clientHeight;
      if (!w || !h) continue;
      if (c.width !== Math.floor(w * dpr)) {
        c.width = Math.floor(w * dpr);
        c.height = Math.floor(h * dpr);
      }
      g.setTransform(c.width / w, 0, 0, c.height / h, 0, 0);
      g.clearRect(0, 0, w, h);
      if (!state.visible || !state.ras) continue;
      if (cell === "threeD") {
        const s = rasToScreen3D(cfg.getCamera(), state.ras, w, h);
        if (s) drawCross(g, s.x * w, s.y * h);
      } else {
        const pr2 = cfg.getSlice().rasToView(cell, cfg.getOffset(cell), state.ras, w / h);
        if (pr2.u >= 0 && pr2.u <= 1 && pr2.v >= 0 && pr2.v <= 1) drawCross(g, pr2.u * w, pr2.v * h);
      }
    }
  };
  state.onChange(redraw);
  let inFlight = false, queued = null;
  const pick3d = async (u, v2) => {
    inFlight = true;
    const ras = await cfg.getScene().pick(u, v2);
    inFlight = false;
    if (ras) {
      state.set(ras);
      cfg.onJump(ras);
    }
    if (queued) {
      const q2 = queued;
      queued = null;
      pick3d(q2.u, q2.v);
    }
  };
  cfg.cells.threeD.addEventListener("pointermove", (e) => {
    if (!isShiftHover(e)) return;
    const { u, v: v2 } = uvOf(cfg.cells.threeD, e);
    if (inFlight) queued = { u, v: v2 };
    else pick3d(u, v2);
  });
  for (const cell of slices) {
    cfg.cells[cell].addEventListener("pointermove", (e) => {
      if (!isShiftHover(e)) return;
      const { u, v: v2, aspect } = uvOf(cfg.cells[cell], e);
      const ras = cfg.getSlice().viewToRas(cell, cfg.getOffset(cell), u, v2, aspect);
      state.set(ras);
      cfg.onJump(ras);
    });
  }
  return { state, redraw };
}

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
function rotateAboutAxis(v2, axis, deg) {
  const k2 = normalize(axis);
  const t = deg * Math.PI / 180;
  const c = Math.cos(t), s = Math.sin(t);
  const kv = cross(k2, v2);
  const kd = dot(k2, v2);
  return [
    v2[0] * c + kv[0] * s + k2[0] * kd * (1 - c),
    v2[1] * c + kv[1] * s + k2[1] * kd * (1 - c),
    v2[2] * c + kv[2] * s + k2[2] * kd * (1 - c)
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
  translate(v2) {
    this.position = add(this.position, v2);
    this.focalPoint = add(this.focalPoint, v2);
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
  displayToWorldAtDepth(x, y, depth, w, h) {
    const { right, up } = this.basis();
    const dop = this.directionOfProjection;
    const halfH = Math.max(1e-6, depth) * Math.tan(this.viewAngle * Math.PI / 360);
    const aspect = w / h;
    const ndcx = x / w * 2 - 1;
    const ndcy = 1 - y / h * 2;
    const offset2 = add(scale(right, ndcx * halfH * aspect), scale(up, ndcy * halfH));
    return add(add(this.position, scale(dop, depth)), offset2);
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
    const [x, y] = _CameraInteractor.toDisplay(cssX, cssY, height);
    const dx = x - this.prev[0];
    const dy = y - this.prev[1];
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
        this.spin(x, y, this.prev[0], this.prev[1], width, height);
        break;
    }
    this.prev = [x, y];
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
  spin(x, y, px, py, width, height) {
    const cx = width / 2, cy = height / 2;
    const newAngle = Math.atan2(y - cy, x - cx) * 180 / Math.PI;
    const oldAngle = Math.atan2(py - cy, px - cx) * 180 / Math.PI;
    this.roll(newAngle - oldAngle);
  }
  /** vtkCamera::Roll — rotate viewUp about the direction of projection. */
  roll(deg) {
    const cam = this.camera;
    const axis = cam.directionOfProjection;
    const t = deg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
    const v2 = cam.viewUp;
    const k2 = axis;
    const kv = [k2[1] * v2[2] - k2[2] * v2[1], k2[2] * v2[0] - k2[0] * v2[2], k2[0] * v2[1] - k2[1] * v2[0]];
    const kd = k2[0] * v2[0] + k2[1] * v2[1] + k2[2] * v2[2];
    cam.viewUp = [
      v2[0] * c + kv[0] * s + k2[0] * kd * (1 - c),
      v2[1] * c + kv[1] * s + k2[1] * kd * (1 - c),
      v2[2] * c + kv[2] * s + k2[2] * kd * (1 - c)
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

// render/demos/camera-control.ts
function attachCameraControls(canvas, camera, opts = {}) {
  const interactor = new CameraInteractor(camera, opts.onChange);
  const local = (e) => {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  canvas.style.touchAction = "none";
  const docEl = (canvas.ownerDocument ?? document).documentElement;
  if (docEl) docEl.style.overscrollBehavior = "none";
  canvas.addEventListener("touchmove", (e) => e.preventDefault(), { passive: false });
  const pointers = /* @__PURE__ */ new Map();
  let pinch = null;
  let triple = null;
  const centroid = () => {
    let mx = 0, my = 0;
    for (const p of pointers.values()) {
      mx += p.x;
      my += p.y;
    }
    const n = pointers.size || 1;
    return { mx: mx / n, my: my / n };
  };
  const pinchState = () => {
    const [a, b] = [...pointers.values()];
    return { dist: Math.hypot(b.x - a.x, b.y - a.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
  };
  const on2 = () => opts.enabled?.() ?? true;
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    if (!on2()) return;
    const { x, y } = local(e);
    pointers.set(e.pointerId, { x, y });
    canvas.setPointerCapture(e.pointerId);
    if (pointers.size === 1) {
      interactor.start(e.button, x, y, canvas.clientHeight, { shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey, alt: e.altKey });
      opts.onLog?.("cameraStart", { action: interactor.action, x, y, button: e.button, shift: e.shiftKey, ctrl: e.ctrlKey, alt: e.altKey });
    } else if (pointers.size === 2) {
      interactor.end();
      pinch = pinchState();
    } else if (pointers.size === 3) {
      pinch = null;
      const c = centroid();
      triple = { mx: c.mx, my: c.my };
      opts.onVolumeDragStart?.();
    }
  });
  const endPointer = (e) => {
    if (!pointers.delete(e.pointerId)) return;
    if (!on2()) {
      interactor.end();
      pinch = null;
      return;
    }
    canvas.releasePointerCapture?.(e.pointerId);
    if (pointers.size < 2) pinch = null;
    if (pointers.size < 3 && triple) {
      triple = null;
      opts.onVolumeDragEnd?.();
    }
    if (pointers.size === 1) {
      const p = [...pointers.values()][0];
      interactor.start(0, p.x, p.y, canvas.clientHeight, { shift: false, ctrl: false, alt: false });
    } else if (pointers.size === 0) {
      interactor.end();
    }
  };
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);
  canvas.addEventListener("pointermove", (e) => {
    if (!on2()) return;
    if (!pointers.has(e.pointerId)) return;
    const { x, y } = local(e);
    pointers.set(e.pointerId, { x, y });
    if (pointers.size >= 3) {
      const c = centroid();
      if (triple) opts.onVolumeDrag?.(c.mx - triple.mx, c.my - triple.my);
      return;
    }
    if (pointers.size === 2) {
      const p = pinchState();
      if (pinch) {
        if (p.dist > 0 && pinch.dist > 0) camera.dolly(p.dist / pinch.dist);
        camera.panByDisplayDelta(p.mx - pinch.mx, pinch.my - p.my, canvas.clientWidth, canvas.clientHeight);
        opts.onChange?.();
      }
      pinch = p;
    } else if (interactor.action !== "none") {
      interactor.move(x, y, canvas.clientWidth, canvas.clientHeight);
    }
  });
  canvas.addEventListener("wheel", (e) => {
    if (!on2()) return;
    e.preventDefault();
    interactor.wheel(e.deltaY < 0);
    opts.onLog?.("cameraWheel", { deltaY: e.deltaY, distance: camera.distance });
  }, { passive: false });
  return interactor;
}
function framedCamera(center, radius, distMul = 2.6) {
  return new VtkCamera(
    [center[0], center[1] + radius * distMul, center[2]],
    [...center],
    [0, 0, 1],
    30
  );
}

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
      const { u, v: v2, w, h: hh } = uv(e);
      cfg.getSlice().zoomAbout(cfg.orient, Math.exp(-e.deltaY * 15e-4), u, v2, w, hh);
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
      const { u: u2, v: v3 } = uv(e);
      view = { mode: wantZoom ? "zoom" : "pan", x: e.clientX, y: e.clientY, pu: u2, pv: v3 };
      canvas.style.cursor = wantZoom ? "ns-resize" : "grabbing";
      canvas.setPointerCapture(e.pointerId);
      return;
    }
    if (e.button !== 0) return;
    e.preventDefault();
    const { u, v: v2, w, h: hh } = uv(e);
    const mode = cfg.leftMode?.() ?? (cfg.wl?.enabled() ? "wl" : "scroll");
    if (h.onLeftGrab?.(u, v2, w, hh)) {
      grabbed = { moved: 0 };
    } else if (mode === "wl" && cfg.wl) {
      const [win, lev] = cfg.wl.get();
      wlDrag = { x: e.clientX, y: e.clientY, win, lev };
      canvas.style.cursor = "crosshair";
    } else if (mode === "zoom" || mode === "pan") {
      view = { mode, x: e.clientX, y: e.clientY, pu: u, pv: v2 };
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
      const { u, v: v2, w, h: hh } = uv(e);
      h.onLeftDrag?.(u, v2, w, hh);
      return;
    }
    if (wlDrag && cfg.wl) {
      const [lo, hi2] = cfg.wl.range();
      const r = canvas.getBoundingClientRect();
      const gain = (hi2 - lo) / Math.max(1, Math.min(r.width, r.height));
      let win = wlDrag.win + gain * (e.clientX - wlDrag.x);
      if (win < 0) win = 0;
      let lev = wlDrag.lev + gain * (wlDrag.y - e.clientY);
      if (lev < lo - win / 2) lev = lo - win / 2;
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
      const { u, v: v2, w, h: hh } = uv(e);
      h.onHover(u, v2, w, hh);
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

// render/demos/view-grid.ts
function attachViewGrid(grid, cells, onResize) {
  let maxed = null;
  const cellDiv = (cell) => grid.querySelector(`.cell[data-cell="${cell}"]`);
  return {
    toggleMax(cell) {
      maxed = maxed === cell ? null : cell;
      for (const n of cells) cellDiv(n).classList.toggle("max", n === maxed);
      grid.classList.toggle("has-max", maxed !== null);
      requestAnimationFrame(onResize);
    },
    isMax(cell) {
      return maxed === cell;
    },
    maxCell: () => maxed
  };
}
function attachDoubleClick(canvas, onDbl) {
  let last = 0, lx = 0, ly = 0;
  canvas.addEventListener("pointerdown", (e) => {
    const dbl = e.timeStamp - last < 350 && Math.hypot(e.clientX - lx, e.clientY - ly) < 6;
    last = dbl ? 0 : e.timeStamp;
    lx = e.clientX;
    ly = e.clientY;
    if (dbl) {
      e.preventDefault();
      onDbl();
    }
  });
}

// render/demos/widget-control.ts
var sub2 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
var dot2 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function camMatrices(cam, w, h) {
  const view = lookAt(cam.position, cam.focalPoint, cam.viewUp);
  const proj = perspectiveZO(cam.viewAngle * Math.PI / 180, w / h, 1, 1e5);
  const vp = multiply(proj, view);
  return { vp, invVp: invert(vp) };
}
function worldToClip(vp, p) {
  return [
    vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12],
    vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13],
    vp[2] * p[0] + vp[6] * p[1] + vp[10] * p[2] + vp[14],
    vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15]
  ];
}
function attachWidgetControls(canvas, camera, opts) {
  const cursorCss = (e) => {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top, rw: r.width, rh: r.height };
  };
  const project = (vp, world, rw, rh) => {
    const c = worldToClip(vp, world);
    if (c[3] <= 0) return null;
    const ndcx = c[0] / c[3], ndcy = c[1] / c[3];
    return { x: (ndcx * 0.5 + 0.5) * rw, y: (1 - (ndcy * 0.5 + 0.5)) * rh };
  };
  const unprojectToPlane = (invVp, px, py, rw, rh, planePt) => {
    const ndcx = px / rw * 2 - 1, ndcy = 1 - py / rh * 2;
    const near = applyMat4(invVp, [ndcx, ndcy, 0]);
    const far = applyMat4(invVp, [ndcx, ndcy, 1]);
    const ro = near, rd = sub2(far, near);
    const n = sub2(camera.position, camera.focalPoint);
    const denom = dot2(rd, n);
    if (Math.abs(denom) < 1e-9) return [...planePt];
    const t = dot2(sub2(planePt, ro), n) / denom;
    return [ro[0] + rd[0] * t, ro[1] + rd[1] * t, ro[2] + rd[2] * t];
  };
  const pick = (e) => {
    const { x, y, rw, rh } = cursorCss(e);
    const { w, h } = opts.getSize();
    const { vp } = camMatrices(camera, w, h);
    const touch = e.pointerType === "touch";
    let best = null, bestD = Infinity;
    for (const hnd of opts.getHandles()) {
      const s = project(vp, hnd.world, rw, rh);
      if (!s) continue;
      const r = (hnd.pickPx ?? 16) * (touch ? 2.75 : 1);
      const d = Math.hypot(s.x - x, s.y - y);
      if (d < r && d < bestD) {
        bestD = d;
        best = hnd;
      }
    }
    return best;
  };
  let grabbed = null, hovered = null, grabbedId = -1;
  const release = (pointerId) => {
    if (!grabbed) return;
    const g = grabbed;
    grabbed = null;
    grabbedId = -1;
    try {
      canvas.releasePointerCapture(pointerId);
    } catch {
    }
    window.removeEventListener("pointermove", onMove, true);
    window.removeEventListener("pointerup", onUp, true);
    window.removeEventListener("pointercancel", onCancel, true);
    canvas.style.cursor = "";
    opts.onDragEnd?.(g);
  };
  const onDown = (e) => {
    if (e.button !== 0) return;
    if (grabbed) {
      release(e.pointerId);
      return;
    }
    if (e.isPrimary === false) return;
    const h = pick(e);
    if (!h) return;
    e.stopPropagation();
    e.preventDefault();
    grabbed = h;
    grabbedId = e.pointerId;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = h.cursor ? h.cursor : "grabbing";
    opts.onDragStart?.(h);
    window.addEventListener("pointermove", onMove, true);
    window.addEventListener("pointerup", onUp, true);
    window.addEventListener("pointercancel", onCancel, true);
  };
  const onMove = (e) => {
    if (!grabbed || e.pointerId !== grabbedId) return;
    e.stopPropagation();
    const { x, y, rw, rh } = cursorCss(e);
    const { w, h } = opts.getSize();
    const { invVp } = camMatrices(camera, w, h);
    const world = unprojectToPlane(invVp, x, y, rw, rh, grabbed.world);
    opts.onDrag(grabbed, world);
    opts.onChange?.();
  };
  const onUp = (e) => {
    if (!grabbed || e.pointerId !== grabbedId) return;
    e.stopPropagation();
    release(e.pointerId);
  };
  const onCancel = (e) => {
    if (grabbed && e.pointerId === grabbedId) release(e.pointerId);
  };
  const onHoverMove = (e) => {
    if (grabbed) return;
    const h = pick(e);
    if (h !== hovered) {
      hovered = h;
      canvas.style.cursor = h ? h.cursor ?? "grab" : "";
      opts.onHover?.(h);
      opts.onChange?.();
    }
  };
  canvas.addEventListener("pointerdown", onDown, true);
  canvas.addEventListener("pointermove", onHoverMove);
  return {
    detach() {
      canvas.removeEventListener("pointerdown", onDown, true);
      canvas.removeEventListener("pointermove", onHoverMove);
      window.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onUp, true);
      window.removeEventListener("pointercancel", onCancel, true);
    }
  };
}

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
function glass(el, extra = "") {
  el.style.cssText += ";background:linear-gradient(135deg,rgba(58,64,88,.55),rgba(20,24,38,.66));backdrop-filter:blur(20px) saturate(1.6);-webkit-backdrop-filter:blur(20px) saturate(1.6);border:1px solid rgba(255,255,255,.2);box-shadow:0 18px 50px rgba(0,0,0,.55);" + extra;
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
function openChromeHelp(host = document.body, sheets = DEFAULT_HELP) {
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
    const rows2 = sec.rows.map(([k2, d]) => `<div style="font:600 12px ui-monospace,Menlo,monospace;color:#fff5d6;white-space:nowrap">${k2}</div><div style="color:rgba(232,238,255,.85)">${d}</div>`).join("");
    panel.innerHTML += `<div style="margin-top:14px;padding:12px 14px;border-radius:10px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08)"><div style="font:700 11px -apple-system,system-ui,sans-serif;letter-spacing:1.1px;text-transform:uppercase;color:#9fe9ff;margin-bottom:9px">${sec.title}</div><div style="display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;align-items:baseline">${rows2}</div></div>`;
  }
  panel.innerHTML += `<div style="margin-top:16px;font-size:12px;color:rgba(232,238,255,.55)">Press <b style="color:#fff5d6">esc</b> or click outside to dismiss.</div>`;
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
  const afterPaint = (fn) => requestAnimationFrame(() => requestAnimationFrame(fn));
  const paintTri = (box, level, color) => {
    const pct = Math.round(level * 100);
    const c = `rgb(${Math.round(color[0] * 255)},${Math.round(color[1] * 255)},${Math.round(color[2] * 255)})`;
    box.style.opacity = level < 0.02 ? "0.75" : "1";
    box.innerHTML = `<span style="position:absolute;left:0;top:0;bottom:0;width:${pct}%;background:${c};opacity:.9"></span><span style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font:700 10px -apple-system,system-ui,sans-serif;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.75)">${pct}%</span>`;
  };
  const triNext = (v2) => v2 > 0.66 ? 0.5 : v2 > 0.04 ? 0 : 1;
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
  const rows2 = [];
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
        const fmt = c.slider.format ?? ((v2) => String(Math.round(v2)));
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
        rows2.push({ c, row, repaint: () => {
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
        rows2.push({ c, row, repaint: () => {
          pill.textContent = c.button.text();
        } });
        continue;
      }
      if (c.getOpacity && c.setOpacity) {
        const box = document.createElement("span");
        box.style.cssText = OPBOX_CSS;
        row.appendChild(box);
        const paint = attachOpacity(box, c.getOpacity, (o) => c.setOpacity(o), c.color ?? [0.62, 0.9, 1], () => opts.onChange?.());
        rows2.push({ c, row, repaint: paint });
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
        rows2.push({ c, row, sw });
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
    const S2 = opts.segments;
    segRows.length = 0;
    segHost.innerHTML = "";
    if (!S2) return;
    const list = S2.list();
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
      const paint = attachOpacity(box, () => S2.get(s.num), (o) => {
        if (!(S2.enabled && !S2.enabled())) S2.set(s.num, o);
      }, s.color, () => opts.onChange?.());
      wrap.appendChild(row);
      segRows.push({ num: s.num, box, color: s.color, paint });
    }
    segHost.appendChild(wrap);
    paintSegments();
  }
  function paintSegments() {
    const S2 = opts.segments;
    if (!S2) return;
    const dis = S2.enabled ? !S2.enabled() : false;
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
    for (const { c, el } of selEls) {
      const v2 = c.get();
      if (el.value !== v2) el.value = v2;
    }
    for (const { c, row, sw, repaint } of rows2) {
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

// render/sdf-text.ts
function layoutText(font, text, opts = {}) {
  const pxSize = opts.pxSize ?? font.sizePx;
  const s = pxSize / font.sizePx;
  const maxW = opts.maxWidthPx ?? Infinity;
  const lineH = (font.lineHeight + (opts.lineGap ?? 0)) * s;
  const space = (font.glyphs.get(" ")?.advance ?? font.sizePx * 0.3) * s;
  const quads = [];
  let maxLineW = 0, lineIdx = 0;
  const paragraphs = text.split("\n");
  for (const para of paragraphs) {
    const words = para.split(" ").filter((w, _i, a) => w.length > 0 || a.length === 1);
    let penX = 0;
    const baseline = () => lineIdx * lineH + font.ascent * s;
    const wordWidth = (w) => {
      let x = 0;
      for (const ch of w) x += (font.glyphs.get(ch)?.advance ?? space / s) * s;
      return x;
    };
    const emit = (w) => {
      for (const ch of w) {
        const g = font.glyphs.get(ch);
        if (g) {
          if (g.aw > 0 && g.ah > 0) quads.push({
            x: penX + g.offX * s,
            y: baseline() + g.offY * s,
            w: g.aw * s,
            h: g.ah * s,
            u0: g.ax / font.atlasW,
            v0: g.ay / font.atlasH,
            u1: (g.ax + g.aw) / font.atlasW,
            v1: (g.ay + g.ah) / font.atlasH
          });
          penX += g.advance * s;
        } else penX += space;
      }
    };
    for (let wi2 = 0; wi2 < words.length; wi2++) {
      const w = words[wi2];
      const need = (penX > 0 ? space : 0) + wordWidth(w);
      if (penX > 0 && penX + need > maxW) {
        maxLineW = Math.max(maxLineW, penX);
        lineIdx++;
        penX = 0;
      } else if (penX > 0) penX += space;
      emit(w);
    }
    maxLineW = Math.max(maxLineW, penX);
    lineIdx++;
  }
  return { quads, width: maxLineW, height: lineIdx * lineH, lines: lineIdx };
}
var INF = 1e20;
function edt1d(f, n, d, v2, z2) {
  let k2 = 0;
  v2[0] = 0;
  z2[0] = -INF;
  z2[1] = INF;
  for (let q2 = 1; q2 < n; q2++) {
    let s = (f[q2] + q2 * q2 - (f[v2[k2]] + v2[k2] * v2[k2])) / (2 * q2 - 2 * v2[k2]);
    while (s <= z2[k2]) {
      k2--;
      s = (f[q2] + q2 * q2 - (f[v2[k2]] + v2[k2] * v2[k2])) / (2 * q2 - 2 * v2[k2]);
    }
    k2++;
    v2[k2] = q2;
    z2[k2] = s;
    z2[k2 + 1] = INF;
  }
  k2 = 0;
  for (let q2 = 0; q2 < n; q2++) {
    while (z2[k2 + 1] < q2) k2++;
    const dv = q2 - v2[k2];
    d[q2] = dv * dv + f[v2[k2]];
  }
}
function edt2d(mask, w, h) {
  const g = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) g[i] = mask[i] ? 0 : INF;
  const maxn = Math.max(w, h);
  const f = new Float64Array(maxn), d = new Float64Array(maxn), z2 = new Float64Array(maxn + 1);
  const v2 = new Int32Array(maxn);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = g[y * w + x];
    edt1d(f, h, d, v2, z2);
    for (let y = 0; y < h; y++) g[y * w + x] = d[y];
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) f[x] = g[y * w + x];
    edt1d(f, w, d, v2, z2);
    for (let x = 0; x < w; x++) g[y * w + x] = d[x];
  }
  return g;
}
function sdfFromMask(alpha, w, h, spread) {
  const inside = new Uint8Array(w * h), outside = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const on2 = alpha[i] >= 128 ? 1 : 0;
    inside[i] = on2;
    outside[i] = on2 ? 0 : 1;
  }
  const dOut = edt2d(inside, w, h);
  const dIn = edt2d(outside, w, h);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const signed = inside[i] ? Math.sqrt(dIn[i]) : -Math.sqrt(dOut[i]);
    out[i] = Math.max(0, Math.min(255, Math.round((0.5 + signed / (2 * spread)) * 255)));
  }
  return out;
}
function makeCanvas(w, h) {
  const g = globalThis;
  if (g.OffscreenCanvas) return new g.OffscreenCanvas(w, h);
  if (g.document) {
    const c = g.document.createElement("canvas");
    c.width = w;
    c.height = h;
    return c;
  }
  throw new Error("sdf-text: no canvas available (browser only)");
}
var DEFAULT_CHARS = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\xB0\xB1\xD7\xB5\u2013\u2014";
function buildFontAtlas(opts = {}) {
  const sizePx = opts.sizePx ?? 44;
  const spread = opts.spread ?? 6;
  const chars = [...opts.chars ?? DEFAULT_CHARS];
  const fontStr = `${opts.weight ?? "normal"} ${sizePx}px ${opts.fontFamily ?? "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"}`;
  const cell = Math.ceil(sizePx * 1.6) + spread * 2;
  const atlasW = opts.atlasW ?? 1024;
  const cols = Math.max(1, Math.floor(atlasW / cell));
  const rows2 = Math.ceil(chars.length / cols);
  const atlasH = rows2 * cell;
  const cv = makeCanvas(atlasW, atlasH);
  const ctx = cv.getContext("2d");
  if (!ctx) throw new Error("sdf-text: no 2d context");
  ctx.font = fontStr;
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "#fff";
  const glyphs = /* @__PURE__ */ new Map();
  let ascent = 0, descent = 0;
  const placed = [];
  chars.forEach((ch, i) => {
    const m = ctx.measureText(ch);
    const asc = m.actualBoundingBoxAscent ?? sizePx * 0.75, desc = m.actualBoundingBoxDescent ?? sizePx * 0.2;
    const left = m.actualBoundingBoxLeft ?? 0, right = m.actualBoundingBoxRight ?? m.width;
    ascent = Math.max(ascent, asc);
    descent = Math.max(descent, desc);
    const cx = i % cols * cell, cy = Math.floor(i / cols) * cell;
    const inkW = Math.max(0, left + right), inkH = Math.max(0, asc + desc);
    const drawX = cx + spread + left, drawY = cy + spread + asc;
    if (ch !== " " && inkW > 0 && inkH > 0) ctx.fillText(ch, drawX, drawY);
    placed.push({ ch, cx, cy, iw: inkW, ih: inkH, offX: -(spread + left), offY: -(spread + asc), advance: m.width });
  });
  const img = ctx.getImageData(0, 0, atlasW, atlasH).data;
  const alpha = new Uint8Array(atlasW * atlasH);
  for (let i = 0; i < atlasW * atlasH; i++) alpha[i] = img[i * 4 + 3];
  const data = new Uint8Array(atlasW * atlasH);
  data.fill(0);
  for (const p of placed) {
    const cw = Math.min(cell, atlasW - p.cx), chh = Math.min(cell, atlasH - p.cy);
    if (p.iw <= 0 || p.ih <= 0) {
      glyphs.set(p.ch, { advance: p.advance, ax: p.cx, ay: p.cy, aw: 0, ah: 0, offX: 0, offY: 0 });
      continue;
    }
    const sub5 = new Uint8Array(cw * chh);
    for (let y = 0; y < chh; y++) for (let x = 0; x < cw; x++) sub5[y * cw + x] = alpha[(p.cy + y) * atlasW + (p.cx + x)];
    const sdf = sdfFromMask(sub5, cw, chh, spread);
    for (let y = 0; y < chh; y++) for (let x = 0; x < cw; x++) data[(p.cy + y) * atlasW + (p.cx + x)] = sdf[y * cw + x];
    const aw = Math.ceil(p.iw) + spread * 2, ah = Math.ceil(p.ih) + spread * 2;
    glyphs.set(p.ch, { advance: p.advance, ax: p.cx, ay: p.cy, aw: Math.min(aw, cw), ah: Math.min(ah, chh), offX: p.offX, offY: p.offY });
  }
  return { sizePx, spread, atlasW, atlasH, data, glyphs, ascent, descent, lineHeight: ascent + descent };
}

// render/label-layout.ts
var DEFAULTS = {
  anchorSpring: 26,
  repulsion: 200,
  damping: 0.86,
  maxSpeed: 1600,
  margin: 8,
  gap: 10,
  standoff: 96,
  ringGap: 14,
  keepOutForce: 40
};
function seedCards(anchorsPx, sizes, standoff = DEFAULTS.standoff) {
  const GA = 2.399963229728653;
  return anchorsPx.map((a, i) => {
    const ang = i * GA;
    return { x: a.x + Math.cos(ang) * standoff, y: a.y + Math.sin(ang) * standoff, vx: 0, vy: 0, w: sizes[i].w, h: sizes[i].h };
  });
}
function layoutStep(cards, anchorsPx, viewport, dtSec, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const dt2 = Math.min(Math.max(dtSec, 0), 0.05);
  if (dt2 === 0) return;
  const n = cards.length;
  const fx = new Float64Array(n), fy = new Float64Array(n);
  const kos = o.keepOuts;
  const GA = 2.399963229728653;
  let gx = 0, gy = 0;
  if (kos && kos.length) {
    for (const k2 of kos) {
      gx += k2.x;
      gy += k2.y;
    }
    gx /= kos.length;
    gy /= kos.length;
  }
  let slotArc = null;
  let slotRX = 0, slotRY = 0;
  if (o.boundary && n > 8) {
    const bnd = o.boundary, bcx = (bnd.minX + bnd.maxX) / 2, bcy = (bnd.minY + bnd.maxY) / 2;
    slotRX = (bnd.maxX - bnd.minX) / 2 + o.ringGap;
    slotRY = (bnd.maxY - bnd.minY) / 2 + o.ringGap;
    const P2 = 2 * (2 * slotRX + 2 * slotRY);
    const order = Array.from({ length: n }, (_2, i) => i).sort((i, j2) => Math.atan2(anchorsPx[i].y - bcy, anchorsPx[i].x - bcx) - Math.atan2(anchorsPx[j2].y - bcy, anchorsPx[j2].x - bcx));
    const foot = order.map((idx) => Math.max(cards[idx].w, cards[idx].h) + o.gap);
    let total = 0;
    for (const f of foot) total += f;
    const extra = Math.max(0, P2 - total) / n;
    slotArc = new Float64Array(n);
    let acc = 0;
    for (let k2 = 0; k2 < n; k2++) {
      acc += (foot[k2] + extra) / 2;
      slotArc[order[k2]] = acc % P2;
      acc += (foot[k2] + extra) / 2;
    }
  }
  for (let i = 0; i < n; i++) {
    const a = anchorsPx[i], c = cards[i];
    const halfDiag = 0.5 * Math.hypot(c.w, c.h);
    const bnd = o.boundary;
    const own = kos && kos[i];
    let tx, ty;
    if (bnd) {
      const bcx = (bnd.minX + bnd.maxX) / 2, bcy = (bnd.minY + bnd.maxY) / 2;
      const bhx = (bnd.maxX - bnd.minX) / 2, bhy = (bnd.maxY - bnd.minY) / 2;
      if (slotArc) {
        const RX = slotRX, RY = slotRY, W2 = 2 * RX, H2 = 2 * RY;
        let s = slotArc[i];
        let rx, ry, nx, ny;
        if (s < W2) {
          rx = -RX + s;
          ry = -RY;
          nx = 0;
          ny = -1;
        } else if ((s -= W2) < H2) {
          rx = RX;
          ry = -RY + s;
          nx = 1;
          ny = 0;
        } else if ((s -= H2) < W2) {
          rx = RX - s;
          ry = RY;
          nx = 0;
          ny = 1;
        } else {
          s -= W2;
          rx = -RX;
          ry = RY - s;
          nx = -1;
          ny = 0;
        }
        tx = bcx + rx + nx * halfDiag;
        ty = bcy + ry + ny * halfDiag;
      } else {
        let dirx = a.x - bcx, diry = a.y - bcy;
        let dl = Math.hypot(dirx, diry);
        if (dl < 0.01) {
          dirx = Math.cos(i * GA);
          diry = Math.sin(i * GA);
          dl = 1;
        }
        dirx /= dl;
        diry /= dl;
        const tEdge = Math.min(bhx / Math.max(Math.abs(dirx), 1e-4), bhy / Math.max(Math.abs(diry), 1e-4));
        const Rr2 = tEdge + halfDiag + o.ringGap;
        tx = bcx + dirx * Rr2;
        ty = bcy + diry * Rr2;
      }
    } else if (own) {
      let dirx = own.x - gx, diry = own.y - gy;
      let dl = Math.hypot(dirx, diry);
      if (dl < 0.01) {
        dirx = Math.cos(i * GA);
        diry = Math.sin(i * GA);
        dl = 1;
      }
      dirx /= dl;
      diry /= dl;
      const R2 = own.radius + halfDiag + o.ringGap;
      tx = own.x + dirx * R2;
      ty = own.y + diry * R2;
    } else {
      const dx = c.x - a.x, dy = c.y - a.y, d = Math.hypot(dx, dy) || 1;
      tx = a.x + dx / d * o.standoff;
      ty = a.y + dy / d * o.standoff;
    }
    fx[i] += (tx - c.x) * o.anchorSpring;
    fy[i] += (ty - c.y) * o.anchorSpring;
    if (o.boundary) {
      const k2 = o.boundary;
      const bcx = (k2.minX + k2.maxX) / 2, bcy = (k2.minY + k2.maxY) / 2;
      const bhx = (k2.maxX - k2.minX) / 2 + halfDiag, bhy = (k2.maxY - k2.minY) / 2 + halfDiag;
      const penx = bhx - Math.abs(c.x - bcx), peny = bhy - Math.abs(c.y - bcy);
      if (penx > 0 && peny > 0) {
        if (penx < peny) fx[i] += Math.sign(c.x - bcx || 1) * penx * o.keepOutForce;
        else fy[i] += Math.sign(c.y - bcy || 1) * peny * o.keepOutForce;
      }
    }
    if (o.reserved) for (const r of o.reserved) {
      const rcx = r.x + r.w / 2, rcy = r.y + r.h / 2;
      const px = (c.w + r.w) / 2 + o.gap - Math.abs(c.x - rcx);
      const py = (c.h + r.h) / 2 + o.gap - Math.abs(c.y - rcy);
      if (px > 0 && py > 0) {
        const F2 = o.keepOutForce * 2.5;
        if (px < py) fx[i] += Math.sign(c.x - rcx || 1) * px * F2;
        else fy[i] += Math.sign(c.y - rcy || 1) * py * F2;
      }
    }
    if (kos) for (const k2 of kos) {
      let dx = c.x - k2.x, dy = c.y - k2.y;
      let d = Math.hypot(dx, dy);
      if (d < 1e-3) {
        dx = Math.cos(i * GA);
        dy = Math.sin(i * GA);
        d = 1;
      }
      const need = k2.radius + halfDiag;
      if (d < need) {
        const push = (need - d) * o.keepOutForce;
        fx[i] += dx / d * push;
        fy[i] += dy / d * push;
      }
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j2 = i + 1; j2 < n; j2++) {
      const a = cards[i], b = cards[j2];
      const ox = (a.w + b.w) / 2 + o.gap - Math.abs(a.x - b.x);
      const oy = (a.h + b.h) / 2 + o.gap - Math.abs(a.y - b.y);
      if (ox <= 0 || oy <= 0) continue;
      if (ox <= oy) {
        const s = Math.sign(a.x - b.x || i - j2 || 1), f = o.repulsion * ox;
        fx[i] += s * f;
        fx[j2] -= s * f;
      } else {
        const s = Math.sign(a.y - b.y || i - j2 || 1), f = o.repulsion * oy;
        fy[i] += s * f;
        fy[j2] -= s * f;
      }
    }
  }
  for (let i = 0; i < n; i++) {
    const c = cards[i];
    c.vx = (c.vx + fx[i] * dt2) * Math.pow(o.damping, dt2 / (1 / 60));
    c.vy = (c.vy + fy[i] * dt2) * Math.pow(o.damping, dt2 / (1 / 60));
    const sp = Math.hypot(c.vx, c.vy);
    if (sp > o.maxSpeed) {
      c.vx = c.vx / sp * o.maxSpeed;
      c.vy = c.vy / sp * o.maxSpeed;
    }
    c.x += c.vx * dt2;
    c.y += c.vy * dt2;
    const hw = c.w / 2 + o.margin, hh = c.h / 2 + o.margin;
    if (c.x < hw) {
      c.x = hw;
      if (c.vx < 0) c.vx = 0;
    } else if (c.x > viewport.w - hw) {
      c.x = viewport.w - hw;
      if (c.vx > 0) c.vx = 0;
    }
    if (c.y < hh) {
      c.y = hh;
      if (c.vy < 0) c.vy = 0;
    } else if (c.y > viewport.h - hh) {
      c.y = viewport.h - hh;
      if (c.vy > 0) c.vy = 0;
    }
  }
  for (let iter = 0; iter < 4; iter++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      for (let j2 = i + 1; j2 < n; j2++) {
        const a = cards[i], b = cards[j2];
        const ox = (a.w + b.w) / 2 + o.gap - Math.abs(a.x - b.x);
        const oy = (a.h + b.h) / 2 + o.gap - Math.abs(a.y - b.y);
        if (ox <= 0 || oy <= 0) continue;
        moved = true;
        if (ox <= oy) {
          const m = Math.sign(a.x - b.x || i - j2 || 1) * ox / 2;
          a.x += m;
          b.x -= m;
        } else {
          const m = Math.sign(a.y - b.y || i - j2 || 1) * oy / 2;
          a.y += m;
          b.y -= m;
        }
      }
    }
    if (o.reserved) for (let i = 0; i < n; i++) {
      const c = cards[i];
      for (const r of o.reserved) {
        const rcx = r.x + r.w / 2, rcy = r.y + r.h / 2;
        const px = (c.w + r.w) / 2 + o.gap - Math.abs(c.x - rcx);
        const py = (c.h + r.h) / 2 + o.gap - Math.abs(c.y - rcy);
        if (px > 0 && py > 0) {
          moved = true;
          if (px < py) c.x += Math.sign(c.x - rcx || 1) * px;
          else c.y += Math.sign(c.y - rcy || 1) * py;
        }
      }
    }
    for (let i = 0; i < n; i++) {
      const c = cards[i], hw = c.w / 2 + o.margin, hh = c.h / 2 + o.margin;
      c.x = Math.min(Math.max(c.x, hw), viewport.w - hw);
      c.y = Math.min(Math.max(c.y, hh), viewport.h - hh);
    }
    if (!moved) break;
  }
}

// render/label-cards.ts
var BUTTONS = [
  { label: "Isolate", part: "isolate" },
  { label: "Hide", part: "hide" },
  { label: "Reset opacities", part: "reset" }
];
var BLACK = [0, 0, 0, 1];
var GREY = [0.32, 0.32, 0.36, 1];
var SHAPE_WGSL = (
  /* wgsl */
  `
struct VP { size : vec4<f32> };
@group(0) @binding(0) var<uniform> u : VP;
struct VO { @builtin(position) pos : vec4<f32>, @location(0) p : vec2<f32>, @location(1) center : vec2<f32>,
            @location(2) half : vec2<f32>, @location(3) params : vec4<f32>, @location(4) fill : vec4<f32>, @location(5) border : vec4<f32> };
@vertex fn vs(@location(0) posPx : vec2<f32>, @location(1) center : vec2<f32>, @location(2) half : vec2<f32>,
              @location(3) params : vec4<f32>, @location(4) fill : vec4<f32>, @location(5) border : vec4<f32>) -> VO {
  var o : VO;
  o.pos = vec4<f32>(posPx.x / u.size.x * 2.0 - 1.0, 1.0 - posPx.y / u.size.y * 2.0, 0.0, 1.0);
  o.p = posPx; o.center = center; o.half = half; o.params = params; o.fill = fill; o.border = border;
  return o;
}
fn sdRoundBox(p : vec2<f32>, b : vec2<f32>, r : f32) -> f32 {
  let q = abs(p) - b + vec2<f32>(r);
  return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
}
@fragment fn fs(i : VO) -> @location(0) vec4<f32> {
  if (i.params.z > 0.5) { return vec4<f32>(i.fill.rgb * i.fill.a, i.fill.a); }   // plain (leader lines)
  let d = sdRoundBox(i.p - i.center, i.half, i.params.x);
  let cov = 1.0 - smoothstep(0.0, 1.0, d);
  if (cov <= 0.0) { discard; }
  let inner = 1.0 - smoothstep(0.0, 1.0, d + i.params.y);
  var col = mix(i.border, i.fill, inner);
  col.a = col.a * cov;
  return vec4<f32>(col.rgb * col.a, col.a);
}`
);
var TEXT_WGSL = (
  /* wgsl */
  `
struct VP { size : vec4<f32> };
@group(0) @binding(0) var<uniform> u : VP;
@group(0) @binding(1) var atlas : texture_2d<f32>;
@group(0) @binding(2) var samp : sampler;
struct TO { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32>, @location(1) color : vec4<f32> };
@vertex fn vs(@location(0) posPx : vec2<f32>, @location(1) uv : vec2<f32>, @location(2) color : vec4<f32>) -> TO {
  var o : TO;
  o.pos = vec4<f32>(posPx.x / u.size.x * 2.0 - 1.0, 1.0 - posPx.y / u.size.y * 2.0, 0.0, 1.0);
  o.uv = uv; o.color = color; return o;
}
@fragment fn fs(i : TO) -> @location(0) vec4<f32> {
  let d = textureSample(atlas, samp, i.uv).r;
  let aa = fwidth(d) + 1e-4;
  let a = smoothstep(0.5 - aa, 0.5 + aa, d) * i.color.a;
  if (a <= 0.001) { discard; }
  return vec4<f32>(i.color.rgb * a, a);
}`
);
var PREMUL_BLEND = {
  color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" }
};
var CardOverlay = class {
  dev;
  shapePipe;
  textPipe;
  vpBuf;
  shapeBind;
  textBind;
  atlasTex;
  font;
  st;
  dpr = 1;
  compact = false;
  // many segments → cards collapse to a mini card (swatch + name), full card on click
  shapeBuf;
  shapeCap = 0;
  textBuf;
  textCap = 0;
  cards = [];
  bodies = [];
  seeded = false;
  lastVisible = [];
  perf = { layoutMs: 0, buildMs: 0, submitMs: 0 };
  maxSpeed = 0;
  /** True when every card is at rest — lets the host stop rendering (dormant) until the next change. */
  settled() {
    return this.maxSpeed < 1.5;
  }
  constructor(gpu, format, font, style = {}) {
    this.dev = gpu.device;
    this.font = font;
    const sc = style.scale ?? 1;
    this.st = {
      titlePx: (style.titlePx ?? 15) * sc,
      bodyPx: (style.bodyPx ?? 12) * sc,
      padPx: (style.padPx ?? 10) * sc,
      gapPx: (style.gapPx ?? 5) * sc,
      maxTextPx: (style.maxTextPx ?? 240) * sc,
      radiusPx: (style.radiusPx ?? 6) * sc,
      borderPx: (style.borderPx ?? 1.25) * sc,
      glassRGBA: style.glassRGBA ?? [1, 1, 1, 0.82],
      borderRGBA: style.borderRGBA ?? [0, 0, 0, 0.92],
      leaderRGBA: style.leaderRGBA ?? [0.05, 0.05, 0.05, 0.85],
      leaderPx: (style.leaderPx ?? 1.5) * sc,
      buttonRGBA: style.buttonRGBA ?? [0.9, 0.91, 0.95, 1]
    };
    this.atlasTex = this.dev.createTexture({ size: [font.atlasW, font.atlasH], format: "r8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.dev.queue.writeTexture({ texture: this.atlasTex }, font.data, { bytesPerRow: font.atlasW, rowsPerImage: font.atlasH }, [font.atlasW, font.atlasH]);
    const sampler = this.dev.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
    this.vpBuf = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const shapeMod = this.dev.createShaderModule({ code: SHAPE_WGSL });
    this.shapePipe = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: shapeMod, entryPoint: "vs", buffers: [{ arrayStride: 72, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x2" },
        { shaderLocation: 1, offset: 8, format: "float32x2" },
        { shaderLocation: 2, offset: 16, format: "float32x2" },
        { shaderLocation: 3, offset: 24, format: "float32x4" },
        { shaderLocation: 4, offset: 40, format: "float32x4" },
        { shaderLocation: 5, offset: 56, format: "float32x4" }
      ] }] },
      fragment: { module: shapeMod, entryPoint: "fs", targets: [{ format, blend: PREMUL_BLEND }] },
      primitive: { topology: "triangle-list" }
    });
    const textMod = this.dev.createShaderModule({ code: TEXT_WGSL });
    this.textPipe = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: textMod, entryPoint: "vs", buffers: [{ arrayStride: 32, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x2" },
        { shaderLocation: 1, offset: 8, format: "float32x2" },
        { shaderLocation: 2, offset: 16, format: "float32x4" }
      ] }] },
      fragment: { module: textMod, entryPoint: "fs", targets: [{ format, blend: PREMUL_BLEND }] },
      primitive: { topology: "triangle-list" }
    });
    this.shapeBind = this.dev.createBindGroup({ layout: this.shapePipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.vpBuf } }] });
    this.textBind = this.dev.createBindGroup({ layout: this.textPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.vpBuf } }, { binding: 1, resource: this.atlasTex.createView() }, { binding: 2, resource: sampler }] });
  }
  setCards(specs, dpr = 1, compact = false) {
    this.dpr = dpr;
    this.compact = compact;
    const s = this.st, titlePx = s.titlePx * dpr, bodyPx = s.bodyPx * dpr, pad = s.padPx * dpr, gap = s.gapPx * dpr, maxT = s.maxTextPx * dpr;
    const sw = Math.round(titlePx), btnH = Math.round(bodyPx + 10 * dpr), btnGap = 4 * dpr, btnInnerPad = 8 * dpr;
    const miniPx = Math.max(9 * dpr, Math.round(titlePx * 0.72)), miniSw = Math.round(miniPx);
    const miniPad = Math.round(pad * 0.55), miniGap = Math.round(gap * 0.7);
    const num2 = (n) => n.toLocaleString("en-US");
    this.cards = specs.map((spec) => {
      const titleL = layoutText(this.font, spec.title, { pxSize: titlePx, maxWidthPx: maxT - sw - gap });
      const subL = spec.subtitle ? layoutText(this.font, spec.subtitle, { pxSize: bodyPx, maxWidthPx: maxT }) : null;
      const titleX = pad + (spec.swatch ? sw + gap : 0);
      const row1H = Math.max(titleL.height, spec.swatch ? sw : 0);
      const headBottom = pad + row1H + (subL ? gap + subL.height : 0);
      const headContentW = Math.max(titleX - pad + titleL.width, subL ? subL.width : 0);
      const head = {
        runs: [{ quads: offset(titleL.quads, titleX, pad), color: BLACK }],
        swatch: spec.swatch ? { x: pad, y: pad, w: sw, h: sw, color: [spec.swatch[0], spec.swatch[1], spec.swatch[2], 1] } : void 0
      };
      if (subL) head.runs.push({ quads: offset(subL.quads, pad, pad + row1H + gap), color: GREY });
      const cw = Math.ceil(headContentW + pad * 2), ch = Math.ceil(headBottom + pad);
      const nameL = layoutText(this.font, spec.title, { pxSize: miniPx, maxWidthPx: maxT });
      const mRow = Math.max(nameL.height, spec.swatch ? miniSw : 0);
      const mTitleX = miniPad + (spec.swatch ? miniSw + miniGap : 0);
      const miniHead = {
        runs: [{ quads: offset(nameL.quads, mTitleX, miniPad + (mRow - nameL.height) / 2), color: BLACK }],
        swatch: spec.swatch ? { x: miniPad, y: miniPad + (mRow - miniSw) / 2, w: miniSw, h: miniSw, color: [spec.swatch[0], spec.swatch[1], spec.swatch[2], 1] } : void 0
      };
      const mw = Math.ceil(mTitleX + nameL.width + miniPad), mh = Math.ceil(mRow + miniPad * 2);
      const lines = [];
      if (spec.stat) {
        lines.push(`${num2(spec.stat.voxels)} voxels`);
        lines.push(`${spec.stat.volumeCc.toFixed(1)} cc`);
        if (spec.stat.hu) lines.push(`${spec.stat.hu.mean.toFixed(0)} \xB1 ${spec.stat.hu.std.toFixed(0)} HU`);
      }
      const lineLs = lines.map((t) => layoutText(this.font, t, { pxSize: bodyPx, maxWidthPx: maxT }));
      const btnLabels = BUTTONS.map((b) => layoutText(this.font, b.label, { pxSize: bodyPx, maxWidthPx: maxT }));
      const contentW = Math.max(headContentW, ...lineLs.map((l) => l.width), ...btnLabels.map((l) => l.width + btnInnerPad * 2));
      const ew = Math.ceil(contentW + pad * 2), btnW = ew - pad * 2;
      const extra = { runs: [], buttons: [] };
      let y = headBottom + gap;
      for (const l of lineLs) {
        extra.runs.push({ quads: offset(l.quads, pad, y), color: BLACK });
        y += l.height + gap * 0.6;
      }
      y += gap * 0.4;
      for (let k2 = 0; k2 < BUTTONS.length; k2++) {
        const lab = btnLabels[k2], lx = pad + (btnW - lab.width) / 2, ly = y + (btnH - lab.height) / 2;
        extra.buttons.push({ x: pad, y, w: btnW, h: btnH, part: BUTTONS[k2].part, label: { quads: offset(lab.quads, lx, ly), color: BLACK } });
        y += btnH + btnGap;
      }
      const eh = Math.ceil(y - btnGap + pad);
      return { spec, cw, ch, ew, eh: Math.max(eh, ch), mw, mh, miniHead, head, extra, expanded: false };
    });
    this.seeded = false;
  }
  size(c) {
    if (c.expanded) return { w: c.ew, h: c.eh };
    if (this.compact) return { w: c.mw, h: c.mh };
    return { w: c.cw, h: c.ch };
  }
  hitTest(px, py) {
    for (let i = this.cards.length - 1; i >= 0; i--) {
      if (!this.lastVisible[i]) continue;
      const b = this.bodies[i], c = this.cards[i], sz = this.size(c);
      if (px < b.x - sz.w / 2 || px > b.x + sz.w / 2 || py < b.y - sz.h / 2 || py > b.y + sz.h / 2) continue;
      if (c.expanded) {
        const lx = px - (b.x - sz.w / 2), ly = py - (b.y - sz.h / 2);
        for (const bt2 of c.extra.buttons) if (lx >= bt2.x && lx <= bt2.x + bt2.w && ly >= bt2.y && ly <= bt2.y + bt2.h) return { index: i, action: bt2.part };
      }
      return { index: i, action: "toggle" };
    }
    return null;
  }
  toggle(index) {
    const c = this.cards[index];
    if (c) c.expanded = !c.expanded;
  }
  spec(index) {
    return this.cards[index]?.spec;
  }
  cardCenter(index) {
    const b = this.bodies[index];
    return b ? { x: b.x, y: b.y } : null;
  }
  buttonCenter(index, part) {
    const b = this.bodies[index], c = this.cards[index];
    if (!b || !c) return null;
    const bt2 = c.extra.buttons.find((x) => x.part === part);
    if (!bt2) return null;
    const sz = this.size(c);
    return { x: b.x - sz.w / 2 + bt2.x + bt2.w / 2, y: b.y - sz.h / 2 + bt2.y + bt2.h / 2 };
  }
  /** Run the force layout for one frame WITHOUT drawing (the CardField path renders via the ray-march).
   *  Updates card body positions + visibility. Returns true while still moving. */
  layout(camera, vp, dtSec, keepOuts, extra) {
    if (!this.cards.length) return false;
    const anchors = this.cards.map((c) => camera.worldToDisplay(c.spec.anchorRAS, vp.w, vp.h));
    this.lastVisible = anchors.map((a) => a.depth > 0);
    const anchorsPx = anchors.map((a) => ({ x: a.x, y: a.y }));
    const sizes = this.cards.map((c) => this.size(c));
    if (!this.seeded || this.bodies.length !== this.cards.length) {
      this.bodies = seedCards(anchorsPx, sizes);
      this.seeded = true;
    } else for (let i = 0; i < this.bodies.length; i++) {
      this.bodies[i].w = sizes[i].w;
      this.bodies[i].h = sizes[i].h;
    }
    layoutStep(this.bodies, anchorsPx, { w: vp.w, h: vp.h }, dtSec, { keepOuts, boundary: extra?.boundary, reserved: extra?.reserved, ringGap: extra?.ringGap });
    return !this.settled();
  }
  /** Card body (screen centre + size) for index — for the CardField billboard. */
  body(index) {
    const b = this.bodies[index];
    return b ? { x: b.x, y: b.y, w: b.w, h: b.h } : null;
  }
  visible(index) {
    return !!this.lastVisible[index];
  }
  count() {
    return this.cards.length;
  }
  expanded(index) {
    return !!this.cards[index]?.expanded;
  }
  render(view, camera, vp, dtSec, keepOuts, extra) {
    if (!this.cards.length) return;
    const shown = extra?.shown;
    const anchors = this.cards.map((c) => camera.worldToDisplay(c.spec.anchorRAS, vp.w, vp.h));
    this.lastVisible = anchors.map((a, i) => a.depth > 0 && (shown ? shown[i] : true));
    const anchorsPx = anchors.map((a) => ({ x: a.x, y: a.y }));
    const sizes = this.cards.map((c) => this.size(c));
    if (!this.seeded || this.bodies.length !== this.cards.length) {
      this.bodies = seedCards(anchorsPx, sizes);
      this.seeded = true;
    } else for (let i = 0; i < this.bodies.length; i++) {
      this.bodies[i].w = sizes[i].w;
      this.bodies[i].h = sizes[i].h;
    }
    const _t0 = performance.now();
    const active = [];
    for (let i = 0; i < this.cards.length; i++) if (this.lastVisible[i]) active.push(i);
    const ab = active.map((i) => this.bodies[i]), aa = active.map((i) => anchorsPx[i]);
    const ako = keepOuts ? active.map((i) => keepOuts[i]) : void 0;
    layoutStep(ab, aa, { w: vp.w, h: vp.h }, dtSec, { keepOuts: ako, boundary: extra?.boundary, reserved: extra?.reserved, ringGap: extra?.ringGap });
    this.perf.layoutMs = performance.now() - _t0;
    let ms = 0;
    for (const b of ab) ms = Math.max(ms, Math.hypot(b.vx, b.vy));
    this.maxSpeed = ms;
    const _t1 = performance.now();
    const shape = [], text = [];
    const rect = (cx, cy, hw, hh, radius, border, fill, brd) => {
      const x0 = cx - hw, x1 = cx + hw, y0 = cy - hh, y1 = cy + hh;
      for (const [px, py] of [[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]]) shape.push(px, py, cx, cy, hw, hh, radius, border, 0, 0, ...fill, ...brd);
    };
    const line = (ax, ay, bx, by, width, col) => {
      let dx = bx - ax, dy = by - ay;
      const L = Math.hypot(dx, dy) || 1;
      dx /= L;
      dy /= L;
      const nx = -dy * width / 2, ny = dx * width / 2;
      for (const [px, py] of [[ax + nx, ay + ny], [bx + nx, by + ny], [bx - nx, by - ny], [ax + nx, ay + ny], [bx - nx, by - ny], [ax - nx, ay - ny]]) shape.push(px, py, 0, 0, -1, -1, 0, 0, 1, 0, ...col, 0, 0, 0, 0);
    };
    const glyphs = (runs, ox, oy) => {
      for (const run of runs) for (const q2 of run.quads) {
        const x0 = ox + q2.x, x1 = ox + q2.x + q2.w, y0 = oy + q2.y, y1 = oy + q2.y + q2.h, co = run.color;
        for (const [px, py, uu, vv] of [[x0, y0, q2.u0, q2.v0], [x1, y0, q2.u1, q2.v0], [x1, y1, q2.u1, q2.v1], [x0, y0, q2.u0, q2.v0], [x1, y1, q2.u1, q2.v1], [x0, y1, q2.u0, q2.v1]]) text.push(px, py, uu, vv, ...co);
      }
    };
    for (let i = 0; i < this.cards.length; i++) {
      if (!this.lastVisible[i]) continue;
      const c = this.cards[i], b = this.bodies[i], a = anchorsPx[i], sz = this.size(c);
      const hw = sz.w / 2, hh = sz.h / 2, ox = b.x - hw, oy = b.y - hh;
      const edge = boxEdgeToward(b.x, b.y, hw, hh, a.x, a.y);
      line(edge.x, edge.y, a.x, a.y, this.st.leaderPx * this.dpr, this.st.leaderRGBA);
      if (this.compact && !c.expanded) {
        rect(b.x, b.y, hw, hh, this.st.radiusPx * this.dpr, this.st.borderPx * this.dpr, this.st.glassRGBA, this.st.borderRGBA);
        if (c.miniHead.swatch) {
          const w2 = c.miniHead.swatch;
          rect(ox + w2.x + w2.w / 2, oy + w2.y + w2.h / 2, w2.w / 2, w2.h / 2, 2 * this.dpr, 1.25 * this.dpr, w2.color, BLACK);
        }
        glyphs(c.miniHead.runs, ox, oy);
        continue;
      }
      rect(b.x, b.y, hw, hh, this.st.radiusPx * this.dpr, this.st.borderPx * this.dpr, this.st.glassRGBA, this.st.borderRGBA);
      if (c.head.swatch) {
        const w2 = c.head.swatch;
        rect(ox + w2.x + w2.w / 2, oy + w2.y + w2.h / 2, w2.w / 2, w2.h / 2, 2 * this.dpr, 1.5 * this.dpr, w2.color, BLACK);
      }
      glyphs(c.head.runs, ox, oy);
      if (c.expanded) {
        for (const bt2 of c.extra.buttons) rect(ox + bt2.x + bt2.w / 2, oy + bt2.y + bt2.h / 2, bt2.w / 2, bt2.h / 2, 4 * this.dpr, 1 * this.dpr, this.st.buttonRGBA, BLACK);
        glyphs(c.extra.runs, ox, oy);
        glyphs(c.extra.buttons.map((bt2) => bt2.label), ox, oy);
      }
    }
    this.perf.buildMs = performance.now() - _t1;
    const _t2 = performance.now();
    this.dev.queue.writeBuffer(this.vpBuf, 0, new Float32Array([vp.w, vp.h, vp.dpr, 0]));
    const shapeArr = new Float32Array(shape), textArr = new Float32Array(text);
    this.shapeBuf = this.ensure(this.shapeBuf, shapeArr.byteLength, "shape");
    if (shapeArr.length) this.dev.queue.writeBuffer(this.shapeBuf, 0, shapeArr);
    this.textBuf = this.ensure(this.textBuf, textArr.byteLength, "text");
    if (textArr.length) this.dev.queue.writeBuffer(this.textBuf, 0, textArr);
    const enc2 = this.dev.createCommandEncoder();
    const pass = enc2.beginRenderPass({ colorAttachments: [{ view, loadOp: "load", storeOp: "store" }] });
    if (shapeArr.length) {
      pass.setPipeline(this.shapePipe);
      pass.setBindGroup(0, this.shapeBind);
      pass.setVertexBuffer(0, this.shapeBuf);
      pass.draw(shapeArr.length / 18);
    }
    if (textArr.length) {
      pass.setPipeline(this.textPipe);
      pass.setBindGroup(0, this.textBind);
      pass.setVertexBuffer(0, this.textBuf);
      pass.draw(textArr.length / 8);
    }
    pass.end();
    this.dev.queue.submit([enc2.finish()]);
    this.perf.submitMs = performance.now() - _t2;
  }
  ensure(buf, bytes, tag) {
    const need = Math.max(64, bytes);
    if (buf && (tag === "shape" ? this.shapeCap : this.textCap) >= need) return buf;
    buf?.destroy();
    const nb = this.dev.createBuffer({ size: need, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    if (tag === "shape") this.shapeCap = need;
    else this.textCap = need;
    return nb;
  }
  /** Current card size (device px) for the given index. */
  cardSize(index) {
    const c = this.cards[index];
    return c ? this.size(c) : null;
  }
  /** Bake ONE card's INK (border + swatch + buttons + text; NO glass fill) into an RGBA texture in
   *  card-local coordinates, for use as a CardField's front-face content. Glass transparent (a=0). */
  bakeCard(index, ss = 2) {
    const c = this.cards[index];
    if (!c) return null;
    const sz = this.size(c), w = Math.max(2, Math.ceil(sz.w * ss)), h = Math.max(2, Math.ceil(sz.h * ss));
    const dpr = this.dpr;
    const tex = this.dev.createTexture({ size: [w, h], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    const shape = [], text = [];
    const rect = (cx, cy, hw, hh, radius, border, fill, brd) => {
      const x0 = cx - hw, x1 = cx + hw, y0 = cy - hh, y1 = cy + hh;
      for (const [px, py] of [[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]]) shape.push(px, py, cx, cy, hw, hh, radius, border, 0, 0, ...fill, ...brd);
    };
    const glyphs = (runs, ox, oy) => {
      for (const run of runs) for (const q2 of run.quads) {
        const x0 = ox + q2.x * ss, x1 = ox + (q2.x + q2.w) * ss, y0 = oy + q2.y * ss, y1 = oy + (q2.y + q2.h) * ss, co = run.color;
        for (const [px, py, uu, vv] of [[x0, y0, q2.u0, q2.v0], [x1, y0, q2.u1, q2.v0], [x1, y1, q2.u1, q2.v1], [x0, y0, q2.u0, q2.v0], [x1, y1, q2.u1, q2.v1], [x0, y1, q2.u0, q2.v1]]) text.push(px, py, uu, vv, ...co);
      }
    };
    const transparent = [0, 0, 0, 0];
    rect(sz.w / 2 * ss, sz.h / 2 * ss, sz.w / 2 * ss, sz.h / 2 * ss, this.st.radiusPx * dpr * ss, this.st.borderPx * dpr * ss, transparent, this.st.borderRGBA);
    if (c.head.swatch) {
      const w2 = c.head.swatch;
      rect((w2.x + w2.w / 2) * ss, (w2.y + w2.h / 2) * ss, w2.w / 2 * ss, w2.h / 2 * ss, 2 * dpr * ss, 1.5 * dpr * ss, w2.color, BLACK);
    }
    glyphs(c.head.runs, 0, 0);
    if (c.expanded) {
      for (const bt2 of c.extra.buttons) rect((bt2.x + bt2.w / 2) * ss, (bt2.y + bt2.h / 2) * ss, bt2.w / 2 * ss, bt2.h / 2 * ss, 4 * dpr * ss, 1 * dpr * ss, this.st.buttonRGBA, BLACK);
      glyphs(c.extra.runs, 0, 0);
      glyphs(c.extra.buttons.map((bt2) => bt2.label), 0, 0);
    }
    this.dev.queue.writeBuffer(this.vpBuf, 0, new Float32Array([w, h, dpr, 0]));
    const sArr = new Float32Array(shape), tArr = new Float32Array(text);
    const sBuf = this.dev.createBuffer({ size: Math.max(64, sArr.byteLength), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    const tBuf = this.dev.createBuffer({ size: Math.max(64, tArr.byteLength), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    if (sArr.length) this.dev.queue.writeBuffer(sBuf, 0, sArr);
    if (tArr.length) this.dev.queue.writeBuffer(tBuf, 0, tArr);
    const enc2 = this.dev.createCommandEncoder();
    const pass = enc2.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    if (sArr.length) {
      pass.setPipeline(this.shapePipe);
      pass.setBindGroup(0, this.shapeBind);
      pass.setVertexBuffer(0, sBuf);
      pass.draw(sArr.length / 18);
    }
    if (tArr.length) {
      pass.setPipeline(this.textPipe);
      pass.setBindGroup(0, this.textBind);
      pass.setVertexBuffer(0, tBuf);
      pass.draw(tArr.length / 8);
    }
    pass.end();
    this.dev.queue.submit([enc2.finish()]);
    sBuf.destroy();
    tBuf.destroy();
    return { tex, w, h };
  }
  dispose() {
    this.atlasTex.destroy();
    this.vpBuf.destroy();
    this.shapeBuf?.destroy();
    this.textBuf?.destroy();
  }
};
function offset(quads, dx, dy) {
  return quads.map((q2) => ({ ...q2, x: q2.x + dx, y: q2.y + dy }));
}
function boxEdgeToward(cx, cy, hw, hh, tx, ty) {
  const dx = tx - cx, dy = ty - cy;
  if (dx === 0 && dy === 0) return { x: cx, y: cy + hh };
  const sx = dx !== 0 ? hw / Math.abs(dx) : Infinity, sy = dy !== 0 ? hh / Math.abs(dy) : Infinity;
  const t = Math.min(sx, sy);
  return { x: cx + dx * t, y: cy + dy * t };
}

// render/segment-cards.ts
function bodyText(term, name) {
  if (!term || !term.type && !term.region) return void 0;
  const norm2 = (t) => (t ?? "").trim().toLowerCase();
  const parts = [];
  if (term.type?.meaning && norm2(term.type.meaning) !== norm2(name)) parts.push(term.type.meaning);
  if (term.region?.meaning && norm2(term.region.meaning) !== norm2(name) && norm2(term.region.meaning) !== norm2(term.type?.meaning)) parts.push(term.region.meaning);
  if (!parts.length && term.type) parts.push(`${term.type.scheme} ${term.type.value}`.trim());
  return parts.length ? parts.join(" \xB7 ") : void 0;
}
function build(ct, seg, segments, maxCards) {
  const [nx, ny, nz] = ct.dims, M2 = ct.ijkToRAS;
  const isCT = (ct.modality ?? "CT") === "CT";
  const wanted = new Set([...segments].sort((a, b) => b.voxels - a.voxels).slice(0, maxCards).map((s) => s.num));
  const acc = /* @__PURE__ */ new Map();
  for (let k2 = 0; k2 < nz; k2++) for (let j2 = 0; j2 < ny; j2++) for (let i = 0; i < nx; i++) {
    const idx = (k2 * ny + j2) * nx + i, v2 = seg.lab[idx];
    if (!v2 || !wanted.has(v2)) continue;
    let e = acc.get(v2);
    if (!e) {
      e = { n: 0, x: 0, y: 0, z: 0, s: 0, s2: 0, lo: [i, j2, k2], hi: [i, j2, k2] };
      acc.set(v2, e);
    }
    e.n++;
    e.x += i;
    e.y += j2;
    e.z += k2;
    if (i < e.lo[0]) e.lo[0] = i;
    if (j2 < e.lo[1]) e.lo[1] = j2;
    if (k2 < e.lo[2]) e.lo[2] = k2;
    if (i > e.hi[0]) e.hi[0] = i;
    if (j2 > e.hi[1]) e.hi[1] = j2;
    if (k2 > e.hi[2]) e.hi[2] = k2;
    if (isCT) {
      const hu = ct.vol[idx];
      e.s += hu;
      e.s2 += hu * hu;
    }
  }
  const voxMm3 = Math.abs(M2[0] * (M2[5] * M2[10] - M2[6] * M2[9]) - M2[1] * (M2[4] * M2[10] - M2[6] * M2[8]) + M2[2] * (M2[4] * M2[9] - M2[5] * M2[8]));
  const ras = (i, j2, k2) => [M2[0] * i + M2[1] * j2 + M2[2] * k2 + M2[3], M2[4] * i + M2[5] * j2 + M2[6] * k2 + M2[7], M2[8] * i + M2[9] * j2 + M2[10] * k2 + M2[11]];
  const term = seg.terminology ?? {};
  const specs = [], geom = [];
  for (const s of segments) {
    const e = acc.get(s.num);
    if (!e) continue;
    const c = ras(e.x / e.n, e.y / e.n, e.z / e.n);
    const hu = isCT ? { mean: e.s / e.n, std: Math.sqrt(Math.max(0, e.s2 / e.n - (e.s / e.n) ** 2)) } : void 0;
    specs.push({ anchorRAS: c, id: s.num, title: s.name, subtitle: bodyText(term[s.num], s.name), swatch: s.color, stat: { voxels: e.n, volumeCc: e.n * voxMm3 / 1e3, hu } });
    const [lx, ly, lz] = e.lo, [hx, hy, hz] = e.hi;
    geom.push({ corners: [[lx, ly, lz], [hx, ly, lz], [lx, hy, lz], [hx, hy, lz], [lx, ly, hz], [hx, ly, hz], [lx, hy, hz], [hx, hy, hz]].map(([i, j2, k2]) => ras(i, j2, k2)) });
  }
  return { specs, geom };
}
function mountSegmentCards(gpu, format, font, canvas, camera, hooks) {
  const overlay = new CardOverlay(gpu, format, font, hooks.style);
  const dpr = globalThis.devicePixelRatio || 1;
  let geom = [];
  let segs = [];
  let sceneBox = [];
  const px = (e) => {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * dpr, y: (e.clientY - r.top) * dpr };
  };
  let down = null;
  canvas.addEventListener("pointerdown", (e) => {
    const p = px(e);
    if (overlay.hitTest(p.x, p.y)) {
      down = { x: e.clientX, y: e.clientY, hit: true };
      e.stopPropagation();
      canvas.setPointerCapture(e.pointerId);
    } else down = { x: e.clientX, y: e.clientY, hit: false };
  }, true);
  canvas.addEventListener("pointerup", (e) => {
    if (down?.hit && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5) {
      const p = px(e), h = overlay.hitTest(p.x, p.y);
      if (h) {
        if (h.action === "toggle") overlay.toggle(h.index);
        else {
          const spec = overlay.spec(h.index);
          if (spec?.id != null) hooks.apply(spec.id, h.action, segs);
        }
        hooks.redraw();
      }
    }
    down = null;
  }, true);
  return {
    setScene(ct, seg, segments) {
      segs = segments;
      if (!ct || !seg) {
        geom = [];
        sceneBox = [];
        overlay.setCards([], dpr);
        return;
      }
      const compact = segments.length > (hooks.compactThreshold ?? 10);
      const cap = compact ? hooks.maxCardsCompact ?? 15 : hooks.maxCards ?? 12;
      const r = build(ct, seg, segments, cap);
      geom = r.geom;
      overlay.setCards(r.specs, dpr, compact);
      const [nx, ny, nz] = ct.dims, M2 = ct.ijkToRAS;
      const ras = (i, j2, k2) => [M2[0] * i + M2[1] * j2 + M2[2] * k2 + M2[3], M2[4] * i + M2[5] * j2 + M2[6] * k2 + M2[7], M2[8] * i + M2[9] * j2 + M2[10] * k2 + M2[11]];
      sceneBox = [[0, 0, 0], [nx, 0, 0], [0, ny, 0], [nx, ny, 0], [0, 0, nz], [nx, 0, nz], [0, ny, nz], [nx, ny, nz]].map(([i, j2, k2]) => ras(i, j2, k2));
    },
    draw(view, w, h, _dpr, dt2) {
      if (!geom.length) return false;
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const c of sceneBox) {
        const p = camera.worldToDisplay(c, w, h);
        if (p.depth <= 0) continue;
        minX = Math.min(minX, p.x);
        maxX = Math.max(maxX, p.x);
        minY = Math.min(minY, p.y);
        maxY = Math.max(maxY, p.y);
      }
      const reserved = hooks.reserved?.();
      const sv = hooks.segVisible;
      const shown = sv ? geom.map((_2, i) => {
        const id = overlay.spec(i)?.id;
        return id == null ? true : sv(id);
      }) : void 0;
      const extra = Number.isFinite(minX) ? { boundary: { minX: Math.max(minX, 0), minY: Math.max(minY, 0), maxX: Math.min(maxX, w), maxY: Math.min(maxY, h) }, reserved, shown, ringGap: 22 * dpr } : { reserved, shown };
      overlay.render(view, camera, { w, h, dpr }, dt2, void 0, extra);
      return !overlay.settled();
    },
    clear() {
      geom = [];
      sceneBox = [];
      overlay.setCards([], dpr);
    },
    count() {
      return geom.length;
    },
    bodyCss(i) {
      const b = overlay.body(i);
      return b ? { x: b.x / dpr, y: b.y / dpr } : null;
    }
  };
}

// render/vendor/idc_tools/s3.js
var idcS3 = (bucket) => "https://" + (bucket || "idc-open-data") + ".s3.us-east-1.amazonaws.com/";
async function fetchRetry(url, opts, tries = 6) {
  let err;
  for (let i = 0; i < tries; i++) {
    const ac = new AbortController();
    const to = setTimeout(() => ac.abort(), 2e4);
    try {
      const r = await fetch(url, { ...opts || {}, signal: ac.signal });
      if (!r.ok && r.status !== 206)
        throw new Error("HTTP " + r.status);
      return r;
    } catch (e) {
      err = e;
      if (i < tries - 1)
        await new Promise((res) => setTimeout(res, Math.min(4e3, 250 * 2 ** i) * (0.6 + 0.8 * Math.random())));
    } finally {
      clearTimeout(to);
    }
  }
  throw err;
}
async function s3ListKeys(prefix, bucket) {
  if (!prefix.endsWith("/"))
    prefix += "/";
  const base = idcS3(bucket);
  const keys = [];
  let token = null, more = true;
  while (more) {
    let url = `${base}?list-type=2&prefix=${encodeURIComponent(prefix)}`;
    if (token)
      url += `&continuation-token=${encodeURIComponent(token)}`;
    const xml = new DOMParser().parseFromString(await fetchRetry(url).then((r) => r.text()), "application/xml");
    for (const e of Array.from(xml.getElementsByTagName("Key"))) {
      const k2 = e.textContent;
      if (k2 && /\.dcm$/i.test(k2))
        keys.push(k2);
    }
    more = xml.getElementsByTagName("IsTruncated")[0]?.textContent === "true";
    token = more ? xml.getElementsByTagName("NextContinuationToken")[0]?.textContent ?? null : null;
  }
  return keys;
}
function ohifViewerURL(studyInstanceUID) {
  return studyInstanceUID ? `https://viewer.imaging.datacommons.cancer.gov/viewer/${studyInstanceUID}` : null;
}

// render/vendor/idc_tools/loader.js
var _worker = null;
function resolveWorkerURL(opts) {
  if (opts?.workerUrl)
    return opts.workerUrl;
  return new URL("./idc-worker.js", import.meta.url);
}
function runWorker(ctKeys, segKeys, ctBucket, segBucket, modality, handlers, opts) {
  if (_worker) {
    try {
      _worker.terminate();
    } catch {
    }
    _worker = null;
  }
  return new Promise((resolve, reject) => {
    const w = new Worker(resolveWorkerURL(opts));
    _worker = w;
    let ct, seg;
    let chain = Promise.resolve();
    w.onmessage = (e) => {
      const m = e.data;
      switch (m.t) {
        case "ctinfo":
          handlers.onSliceCount?.(m.count);
          break;
        case "thumb":
          handlers.onThumb?.(m.n, m.w, m.h, m.rgba);
          break;
        case "seg":
          handlers.onSegName?.(m.name);
          break;
        case "progress":
          handlers.onProgress?.({ frac: m.frac, msg: m.msg });
          break;
        case "ctready":
          try {
            handlers.onCTReady?.({
              dims: m.dims,
              ijkToRAS: m.ijkToRAS,
              win: m.win,
              lev: m.lev,
              dtype: m.dtype || "float32",
              modality
            });
          } catch (err) {
            console.error("[idc_tools] onCTReady", err);
          }
          break;
        case "ctgeom":
          try {
            handlers.onCTGeom?.({ ijkToRAS: m.ijkToRAS, win: m.win, lev: m.lev });
          } catch (err) {
            console.error("[idc_tools] onCTGeom", err);
          }
          break;
        case "ctslab": {
          const data = new Float32Array(m.vol);
          try {
            handlers.onCTSlab?.({
              zStart: m.zStart,
              zCount: m.zCount,
              data,
              dims: m.dims
            });
          } catch (err) {
            console.error("[idc_tools] onCTSlab", err);
          }
          break;
        }
        case "ct": {
          const Ctor = m.dtype === "float32" ? Float32Array : Int16Array;
          ct = { vol: new Ctor(m.vol), dims: m.dims, ijkToRAS: m.ijkToRAS, win: m.win, lev: m.lev, dtype: m.dtype, modality };
          chain = chain.then(() => handlers.onCT?.(ct)).catch((err) => console.error("[idc_tools] onCT", err));
          break;
        }
        case "labelmap": {
          seg = { lab: new Uint8Array(m.lab), colors: m.colors, names: m.names, terminology: m.terminology };
          chain = chain.then(() => handlers.onLabelmap?.(seg)).catch((err) => console.error("[idc_tools] onLabelmap", err));
          break;
        }
        case "error":
          w.terminate();
          if (_worker === w)
            _worker = null;
          reject(new Error(m.error));
          break;
        case "alldone":
          w.terminate();
          if (_worker === w)
            _worker = null;
          chain.then(() => resolve({ ct, seg }));
          break;
      }
    };
    w.onerror = (e) => {
      w.terminate();
      if (_worker === w)
        _worker = null;
      reject(new Error("idc_tools worker: " + (e.message || e)));
    };
    w.postMessage({ ctKeys, segKeys, ctBucket, segBucket, modality });
  });
}
async function loadSeries(entry, handlers = {}, opts) {
  const modality = { CT: "CT", MR: "MR", PT: "PET" }[entry.m] || entry.m;
  const ctKeys = await s3ListKeys(entry.c, entry.cb);
  if (!ctKeys.length)
    throw new Error("idc_tools: no DICOM under CT prefix " + entry.c);
  const segKeys = entry.s ? await s3ListKeys(entry.s, entry.sb) : [];
  const r = await runWorker(ctKeys, segKeys, entry.cb, entry.sb, modality, handlers, opts);
  r.entry = entry;
  return r;
}

// render/demos/ira/ira.ts
var PLANE_COLOR = {
  axial: "#f05a5a",
  sagittal: "#f0d24a",
  coronal: "#5ad07a"
};
function displayClass() {
  const sw = globalThis.screen?.width ?? globalThis.innerWidth;
  const sh = globalThis.screen?.height ?? globalThis.innerHeight;
  const devicePx = sw * (globalThis.devicePixelRatio || 1);
  if (Math.min(sw, sh) < 700) return "phone";
  if (devicePx >= 3400) return "large";
  return "desktop";
}
function ensureIraStyle() {
  if (document.getElementById("ira-style")) return;
  const st2 = document.createElement("style");
  st2.id = "ira-style";
  st2.textContent = STYLE;
  document.head.appendChild(st2);
}
function syncIraToolbarToggle(tb, toggleBtn) {
  const collapsed = tb.classList.contains("collapsed");
  toggleBtn.innerHTML = ICONS[collapsed ? "toolsExpand" : "toolsCollapse"] ?? "?";
  toggleBtn.title = collapsed ? "Show tools" : "Hide tools";
  toggleBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
  toggleBtn.setAttribute("aria-label", collapsed ? "Show tools" : "Hide tools");
}
function ensureIraToolbarShell(bar) {
  ensureIraStyle();
  let tb = document.getElementById("ira-toolbar");
  if (tb) return tb;
  tb = document.createElement("div");
  tb.id = "ira-toolbar";
  tb.classList.add("collapsed");
  bar.after(tb);
  const toggleBtn = document.createElement("button");
  toggleBtn.id = "ira-toolbar-toggle";
  toggleBtn.type = "button";
  toggleBtn.setAttribute("aria-expanded", "false");
  toggleBtn.setAttribute("aria-controls", "ira-toolbar-body");
  syncIraToolbarToggle(tb, toggleBtn);
  toggleBtn.addEventListener("click", () => {
    tb.classList.toggle("collapsed");
    syncIraToolbarToggle(tb, toggleBtn);
    globalThis.dispatchEvent(new Event("resize"));
  });
  const tbBody = document.createElement("div");
  tbBody.id = "ira-toolbar-body";
  tb.append(toggleBtn, tbBody);
  return tb;
}
function resetIraToolbarShell() {
  const tb = document.getElementById("ira-toolbar");
  const body = document.getElementById("ira-toolbar-body");
  const toggle = document.getElementById("ira-toolbar-toggle");
  if (body) body.replaceChildren();
  if (tb) {
    tb.classList.add("collapsed");
    if (toggle) syncIraToolbarToggle(tb, toggle);
  }
}
var I = (body) => `<svg viewBox="0 0 24 24" width="25" height="25" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;
var ICONS = {
  // Action tools
  patient: I(`<circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 5-6 8-6s6.5 2 8 6"/>`),
  prevStudy: I(`<path d="M3 7h6l2-2h10v14H3z"/><path d="M14 11l-4 3 4 3" fill="currentColor"/>`),
  nextStudy: I(`<path d="M3 7h6l2-2h10v14H3z"/><path d="M11 11l4 3-4 3" fill="currentColor"/>`),
  prevFS: I(`<path d="M12 6l-7 6 7 6M19 6l-7 6 7 6"/>`),
  // «
  nextFS: I(`<path d="M5 6l7 6-7 6M12 6l7 6-7 6"/>`),
  // »
  prevFrame: I(`<path d="M15 5l-8 7 8 7"/>`),
  // ‹
  nextFrame: I(`<path d="M9 5l8 7-8 7"/>`),
  // ›
  layoutIn: I(
    `<rect x="3" y="4" width="14" height="14"/><path d="M10 4v14M3 11h14"/><path d="M19 15l2 3h-4z" fill="currentColor"/>`
  ),
  layoutMulti: I(
    `<rect x="2" y="3" width="8" height="7"/><rect x="12" y="3" width="8" height="7"/><rect x="2" y="12" width="8" height="7"/><path d="M18 16l2 3h-4z" fill="currentColor"/>`
  ),
  invert: I(`<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 010 16z" fill="currentColor"/>`),
  // IEC 5411
  localizer: I(
    `<rect x="3" y="4" width="18" height="16"/><circle cx="12" cy="10" r="2.5"/><path d="M8 14c1-1.5 7-1.5 8 0"/><path d="M3 12h3M8 12h3M13 12h3M18 12h3" stroke-dasharray="2 2"/>`
  ),
  link: I(
    `<rect x="2" y="9" width="6" height="6" rx="3"/><rect x="9" y="9" width="6" height="6" rx="3"/><rect x="16" y="9" width="6" height="6" rx="3"/>`
  ),
  wlMode: I(
    `<circle cx="7" cy="10" r="4.5"/><path d="M7 5.5a4.5 4.5 0 010 9z" fill="currentColor"/><circle cx="17" cy="10" r="1.2" fill="currentColor"/><ellipse cx="17" cy="10" rx="4" ry="1.6"/><path d="M3 19h18"/>`
  ),
  // IEC 5435 + Bohr atom + line
  annotation: I(
    `<path d="M3 4h6M3 7h4M15 4h6M17 7h4M3 17h4M3 20h6M17 17h4M15 20h6"/>`
  ),
  print: I(
    `<rect x="6" y="3" width="12" height="5"/><rect x="3" y="8" width="18" height="8" rx="1"/><rect x="6" y="14" width="12" height="7"/>`
  ),
  reset: I(`<path d="M4 9V4h5"/><path d="M4 4l6 6"/><rect x="8" y="10" width="12" height="10"/>`),
  // IEC 5495
  report: I(`<path d="M4 4h16M4 7h16M4 10h10M4 15h16M4 18h12"/>`),
  rotate: I(`<path d="M19 12a7 7 0 11-2-4.9"/><path d="M17 3v4h4" fill="currentColor"/>`),
  // IEC 5772 one-way
  flip: I(`<path d="M12 3v18" stroke-dasharray="3 2"/><path d="M9 7L4 12l5 5zM15 7l5 5-5 5z" fill="currentColor"/>`),
  // IEC 5408
  cine: I(
    `<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 8h18M6 5v3M10 5v3M14 5v3M18 5v3"/><path d="M10 11l5 3.5-5 3.5z" fill="currentColor"/>`
  ),
  // IEC 1123
  presets: I(
    `<circle cx="10" cy="10" r="5.5"/><path d="M10 4.5a5.5 5.5 0 010 11z" fill="currentColor"/><path d="M17 15l2.5 4h-5z" fill="currentColor"/>`
  ),
  // IEC 5435 + dropdown
  help: I(`<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 114 2c-.9.7-1.5 1.2-1.5 2.3"/><circle cx="12" cy="17" r=".6" fill="currentColor"/>`),
  // IEC 5289
  advanced: I(`<path d="M2 9l10-4 10 4-10 4z" fill="currentColor"/><path d="M6 11v5c2 2.5 10 2.5 12 0v-5"/>`),
  // mortarboard
  // Modal tools
  zoom: I(`<circle cx="10" cy="10" r="6"/><path d="M15 15l6 6"/>`),
  pan: I(
    `<path d="M8 11V5.5a1.4 1.4 0 012.8 0V10m0-5.5V4a1.4 1.4 0 012.8 0v6m0-4.5a1.4 1.4 0 012.8 0V12m0-3a1.4 1.4 0 012.8 0v5c0 4-2.5 7-6.5 7S8.5 19 7 16l-2.4-4.2a1.3 1.3 0 012.2-1.3L8 12.5z"/>`
  ),
  window: I(`<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 010 16z" fill="currentColor"/><path d="M12 4v16" stroke-width="1"/>`),
  // IEC 5435
  scroll: I(
    `<rect x="7" y="3" width="12" height="9"/><rect x="5" y="7" width="12" height="9" fill="#05060a"/><rect x="3" y="11" width="12" height="9" fill="#05060a"/>`
  ),
  // stack of frames
  selectvp: I(`<path d="M5 3l7 16 2.2-6.2L20 10.5z" fill="currentColor"/>`),
  // up-left arrow
  distance: I(`<path d="M4 20L20 4"/><path d="M6 14l2 2M9 11l2 2M12 8l2 2M15 5l2 2"/>`),
  // IEC 5658 ruler
  angle: I(`<path d="M4 20h16L4 6z"/><path d="M10 20a8 8 0 00-2.5-5.5"/><path d="M9 12.5l-1.5 2M10 20l-2.4-.6" stroke-width="1.2"/>`),
  crosshair: I(
    `<circle cx="12" cy="12" r="8"/><path d="M12 4v5M12 15v5M4 12h5M15 12h5" stroke-dasharray="3 2"/>`
  ),
  // Cine controls
  play: I(`<path d="M7 5l12 7-12 7z" fill="currentColor"/>`),
  stop: I(`<rect x="6" y="6" width="12" height="12" fill="currentColor"/>`),
  pause: I(`<rect x="6" y="5" width="4" height="14" fill="currentColor"/><rect x="14" y="5" width="4" height="14" fill="currentColor"/>`),
  toStart: I(`<path d="M6 5v14M19 5l-10 7 10 7z" fill="currentColor"/>`),
  stepFwd: I(`<path d="M5 5l10 7-10 7z" fill="currentColor"/><rect x="17" y="5" width="2.5" height="14" fill="currentColor"/>`),
  // IEC 5471
  toEnd: I(`<path d="M18 5v14M5 5l10 7-10 7z" fill="currentColor"/>`),
  thumbs: I(`<rect x="3" y="3" width="7" height="7"/><rect x="3" y="13" width="7" height="7"/><path d="M13 5h8M13 9h6M13 15h8M13 19h6"/>`),
  // IDC actions
  share: I(`<circle cx="18" cy="5" r="2.6"/><circle cx="6" cy="12" r="2.6"/><circle cx="18" cy="19" r="2.6"/><path d="M8.3 10.8l7.4-4.3M8.3 13.2l7.4 4.3"/>`),
  download: I(`<path d="M12 3v11m0 0l-4-4m4 4l4-4"/><path d="M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2"/>`),
  // 3D-print mesh (STL): cube + download arrow
  stl: I(
    `<path d="M7 9l5-3 5 3v6l-5 3-5-3z"/><path d="M7 9l5 3 5-3M12 12v6"/><path d="M18 3v4m0 0l-2-2m2 2l2-2"/>`
  ),
  toolsCollapse: I(`<path d="M6 14l6-6 6 6"/>`),
  toolsExpand: I(`<path d="M6 10l6 6 6-6"/>`)
};
var STYLE = `
#ira-toolbar { display:flex; align-items:flex-start; gap:4px; padding:3px 8px; background:#0b0e16;
  border-bottom:1px solid #1b2740; }
#ira-toolbar.collapsed { padding:2px 8px; }
#ira-toolbar-toggle { flex:0 0 auto; }
#ira-toolbar-body { display:flex; align-items:center; gap:2px; flex-wrap:wrap; flex:1 1 auto; min-width:0; }
#ira-toolbar.collapsed #ira-toolbar-body { display:none; }
#ira-toolbar .sep { width:1px; height:28px; background:#1b2740; margin:0 6px; }
#ira-toolbar button { display:inline-flex; align-items:center; justify-content:center;
  width:38px; height:36px; color:#9fb3d0; background:none; border:1px solid transparent;
  border-radius:5px; cursor:pointer; padding:0; }
#ira-toolbar button:hover { background:#1b2740; color:#d6e2f2; }
#ira-toolbar button.active { background:#2b6cb0; color:#fff; border-color:#33507e; }
#ira-toolbar button:disabled { opacity:.3; cursor:default; background:none; }
#ira-cine-box { display:flex; align-items:center; gap:2px; margin-left:4px; padding:2px 6px;
  background:#11141d; border:1px solid #33507e; border-radius:5px; }
#ira-cine-box input { width:44px; font:600 11px -apple-system,system-ui,sans-serif; color:#d6e2f2;
  background:#0b0e16; border:1px solid #33507e; border-radius:3px; padding:1px 4px; }
#ira-strip { width:158px; overflow-y:auto; background:#0b0e16; border-right:1px solid #1b2740;
  flex:0 0 auto; }
#ira-strip .head { font:700 10px -apple-system,system-ui,sans-serif; letter-spacing:.6px;
  text-transform:uppercase; color:#5a6b85; padding:6px 8px 4px; }
#ira-strip .item { margin:0 6px 8px; border:1px solid #1b2740; border-radius:5px; cursor:pointer;
  overflow:hidden; background:#05060a; }
#ira-strip .item:hover { border-color:#33507e; }
#ira-strip .item.current { border-color:#2b6cb0; box-shadow:0 0 0 1px #2b6cb0; }
#ira-strip img { width:100%; aspect-ratio:1; object-fit:contain; background:#000; display:block; }
#ira-strip .cap { font:10px -apple-system,system-ui,sans-serif; color:#9fb3d0; padding:3px 6px;
  line-height:1.35; }
.ira-cell-selected { outline:1px solid #2b6cb0; outline-offset:-1px; }
.srv-grid.ann-none .srv-orient, .srv-grid.ann-none .srv-slice-readout,
.srv-grid.ann-min .srv-slice-readout { display:none; }
`;
function mountIra(cfg) {
  ensureIraStyle();
  let tool = "scroll";
  let selectedPlane = cfg.planes[0];
  let localizerOn = false;
  let inverted = false;
  let clamped = /^(NM|PT)$/i.test(cfg.modality);
  let annState = 0;
  let crossRas = null;
  const measurements = [];
  let pending = null;
  let layout = "fourUp";
  const overlays = /* @__PURE__ */ new Map();
  for (const p of cfg.planes) {
    const cell = cfg.cellOf(p);
    if (!cell) continue;
    cell.querySelectorAll("canvas.ira-overlay").forEach((el) => el.remove());
    const c = document.createElement("canvas");
    c.className = "ira-overlay";
    c.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;";
    cell.appendChild(c);
    overlays.set(p, c);
  }
  const resizeOverlays = () => {
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    for (const [, c] of overlays) {
      c.width = Math.max(1, Math.floor(c.clientWidth * dpr));
      c.height = Math.max(1, Math.floor(c.clientHeight * dpr));
    }
  };
  const rasOfClick = (p, u, v2, w, h) => cfg.slice().viewToRas(p, cfg.off01(p), u, v2, w / h);
  const toPx = (p, ras, w, h) => {
    const r = cfg.slice().rasToView(p, cfg.off01(p), ras, w / h);
    return { x: r.u * w, y: r.v * h };
  };
  const drawOverlay = (p) => {
    const c = overlays.get(p);
    if (!c) return;
    const ctx = c.getContext("2d");
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = c.width / dpr, h = c.height / dpr;
    ctx.clearRect(0, 0, w, h);
    if (!w || !h) return;
    if (localizerOn) {
      const nAxis = { sagittal: 0, coronal: 1, axial: 2 };
      for (const q2 of cfg.planes) {
        if (q2 === p) continue;
        const free = [0, 1, 2].find((a2) => a2 !== nAxis[p] && a2 !== nAxis[q2]);
        const mk = (t) => {
          const r = [0, 0, 0];
          r[nAxis[p]] = cfg.offsetMm(p);
          r[nAxis[q2]] = cfg.offsetMm(q2);
          r[free] = t;
          return r;
        };
        const a = toPx(p, mk(cfg.rasLo[free]), w, h);
        const b = toPx(p, mk(cfg.rasHi[free]), w, h);
        ctx.strokeStyle = PLANE_COLOR[q2];
        ctx.globalAlpha = 0.75;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
    }
    if (crossRas) {
      const pt2 = toPx(p, crossRas, w, h);
      if (pt2.x >= 0 && pt2.x <= w && pt2.y >= 0 && pt2.y <= h) {
        drawCross(ctx, pt2.x, pt2.y, { color: "#6cc4ff", size: 14, gap: 5 });
      }
    }
    const near = (m) => m.plane === p && Math.abs(m.mm - cfg.offsetMm(p)) <= cfg.spacing(p) / 2;
    ctx.font = "700 15px -apple-system,system-ui,sans-serif";
    ctx.lineJoin = "round";
    ctx.textBaseline = "alphabetic";
    const label = (text, x, y) => {
      ctx.lineWidth = 4;
      ctx.strokeStyle = "rgba(6,10,16,.92)";
      ctx.strokeText(text, x, y);
      ctx.fillStyle = "#8dffc4";
      ctx.fillText(text, x, y);
    };
    for (const m of [...measurements.filter(near), ...pending && pending.plane === p ? [pending] : []]) {
      const px = m.pts.map((r) => toPx(p, r, w, h));
      ctx.strokeStyle = "#6ce0a8";
      ctx.fillStyle = "#6ce0a8";
      ctx.lineWidth = 2;
      for (let i = 0; i + 1 < px.length; i += 2) {
        ctx.beginPath();
        ctx.moveTo(px[i].x, px[i].y);
        ctx.lineTo(px[i + 1].x, px[i + 1].y);
        ctx.stroke();
      }
      for (const q2 of px) {
        ctx.beginPath();
        ctx.arc(q2.x, q2.y, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
      if (m.kind === "distance" && m.pts.length === 2) {
        const d = Math.hypot(
          m.pts[0][0] - m.pts[1][0],
          m.pts[0][1] - m.pts[1][1],
          m.pts[0][2] - m.pts[1][2]
        );
        label(`${d.toFixed(1)} mm`, (px[0].x + px[1].x) / 2 + 8, (px[0].y + px[1].y) / 2 - 8);
      } else if (m.kind === "angle" && m.pts.length === 4) {
        const v1 = [m.pts[1][0] - m.pts[0][0], m.pts[1][1] - m.pts[0][1], m.pts[1][2] - m.pts[0][2]];
        const v2 = [m.pts[3][0] - m.pts[2][0], m.pts[3][1] - m.pts[2][1], m.pts[3][2] - m.pts[2][2]];
        const dot5 = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
        const deg = Math.acos(
          Math.min(1, Math.abs(dot5) / (Math.hypot(...v1) * Math.hypot(...v2) || 1))
        ) * 180 / Math.PI;
        label(`${deg.toFixed(1)}\xB0`, px[3].x + 8, px[3].y - 8);
      }
    }
  };
  const redrawOverlays = () => {
    for (const p of cfg.planes) drawOverlay(p);
  };
  const markSelected = () => {
    for (const p of cfg.planes) {
      cfg.cellOf(p)?.classList.toggle("ira-cell-selected", p === selectedPlane);
    }
  };
  for (const p of cfg.planes) {
    cfg.cellOf(p)?.addEventListener("pointerdown", () => {
      if (selectedPlane !== p) {
        selectedPlane = p;
        markSelected();
      }
    });
  }
  markSelected();
  ensureIraToolbarShell(cfg.bar);
  const tbBody = document.getElementById("ira-toolbar-body");
  tbBody.replaceChildren();
  const buttons = /* @__PURE__ */ new Map();
  const btn = (id, icon, tip, onClick, opts = {}) => {
    const b = document.createElement("button");
    b.id = "ira-" + id;
    b.innerHTML = ICONS[icon] ?? "?";
    b.title = opts.disabled && opts.disabledTip ? opts.disabledTip : tip;
    b.disabled = !!opts.disabled;
    if (onClick) b.addEventListener("click", onClick);
    tbBody.appendChild(b);
    buttons.set(id, b);
    return b;
  };
  const sep = () => {
    const s = document.createElement("div");
    s.className = "sep";
    tbBody.appendChild(s);
  };
  const stepSelected = (forward) => {
    cfg.step(selectedPlane, forward);
    cfg.redraw(selectedPlane);
  };
  btn("patient", "patient", "Select Patient \u2014 back to the patient/study/series browser", cfg.close);
  btn("prev-study", "prevStudy", "Previous Study", () => cfg.nav?.prevStudy?.(), {
    disabled: !cfg.nav?.prevStudy,
    disabledTip: "Previous Study \u2014 none earlier"
  });
  btn("next-study", "nextStudy", "Next Study", () => cfg.nav?.nextStudy?.(), {
    disabled: !cfg.nav?.nextStudy,
    disabledTip: "Next Study \u2014 none later"
  });
  btn("prev-fs", "prevFS", "Previous FrameSet (series) \u2014 Page Up", () => cfg.nav?.prevSeries?.(), {
    disabled: !cfg.nav?.prevSeries,
    disabledTip: "Previous FrameSet \u2014 none"
  });
  btn("next-fs", "nextFS", "Next FrameSet (series) \u2014 Page Down", () => cfg.nav?.nextSeries?.(), {
    disabled: !cfg.nav?.nextSeries,
    disabledTip: "Next FrameSet \u2014 none"
  });
  btn("prev-frame", "prevFrame", "Previous Frame \u2014 Left/Up arrow", () => stepSelected(false));
  btn("next-frame", "nextFrame", "Next Frame \u2014 Right/Down arrow", () => stepSelected(true));
  sep();
  const layoutSel = document.createElement("select");
  layoutSel.id = "ira-layout";
  layoutSel.title = "Layout Multiple Viewports \u2014 hanging protocol";
  layoutSel.style.cssText = "font:600 11px -apple-system,system-ui,sans-serif;color:#d6e2f2;background:#11141d;border:1px solid #33507e;border-radius:4px;padding:2px 4px;cursor:pointer;";
  for (const [v2, label] of [
    ["fourUp", "Four-Up (MPR + 3D)"],
    ["twoUp", "2 side-by-side"],
    ["single", "Single viewport"],
    ["browser", "Browser (patient/study/series)"]
  ]) layoutSel.add(new Option(label, v2));
  layoutSel.addEventListener("change", () => {
    if (layoutSel.value === "browser") cfg.close();
    else setLayout(layoutSel.value);
  });
  tbBody.appendChild(layoutSel);
  btn("layout-in", "layoutIn", "Layout Within Viewport", void 0, {
    disabled: true,
    disabledTip: "Layout Within Viewport \u2014 tiled stack layout not yet available (MPR planes shown instead)"
  });
  sep();
  const CURSORS = {
    zoom: "ns-resize",
    pan: "grab",
    wl: "crosshair",
    distance: "crosshair",
    angle: "crosshair",
    crosshair: "crosshair",
    select: "pointer",
    scroll: "default"
  };
  const setTool = (t) => {
    tool = t;
    pending = null;
    for (const [id, tl] of [
      ["tool-scroll", "scroll"],
      ["tool-window", "wl"],
      ["tool-zoom", "zoom"],
      ["tool-pan", "pan"],
      ["tool-select", "select"],
      ["tool-distance", "distance"],
      ["tool-angle", "angle"],
      ["tool-crosshair", "crosshair"]
    ]) buttons.get(id)?.classList.toggle("active", tool === tl);
    for (const p of cfg.planes) cfg.canvases[p].style.cursor = CURSORS[tool] ?? "default";
    cfg.onToolChange?.(tool);
    redrawOverlays();
  };
  btn("tool-scroll", "scroll", "Scroll (stack) \u2014 S", () => setTool("scroll"));
  btn("tool-window", "window", "Window (adjust W/L by dragging) \u2014 W", () => setTool("wl"));
  btn("tool-zoom", "zoom", "Zoom (drag up = out, down = in) \u2014 Z", () => setTool("zoom"));
  btn("tool-pan", "pan", "Pan (translate) \u2014 T", () => setTool("pan"));
  btn("tool-select", "selectvp", "Select Viewport \u2014 V", () => setTool("select"));
  btn("tool-distance", "distance", "Distance measurement (click start, click end) \u2014 D", () => setTool("distance"));
  btn("tool-angle", "angle", "Angle measurement (two segments, four clicks \u2014 Cobb) \u2014 A", () => setTool("angle"));
  btn("tool-crosshair", "crosshair", "Cross-hair (click to localize in all viewports) \u2014 J", () => setTool("crosshair"));
  sep();
  let presetPop = null;
  let presetOrigWL = null;
  let presetEsc = null;
  const closePresetPop = (revertWL) => {
    if (revertWL) cfg.wl.set(revertWL[0], revertWL[1]);
    presetPop?.remove();
    presetPop = null;
    document.removeEventListener("pointerdown", onPresetOutside, true);
    if (presetEsc) document.removeEventListener("keydown", presetEsc, true);
    presetEsc = null;
  };
  const onPresetOutside = (e) => {
    if (presetPop && !presetPop.contains(e.target) && e.target?.id !== "ira-presets") {
      closePresetPop(presetOrigWL ?? void 0);
    }
  };
  const openPresetPop = () => {
    if (presetPop) {
      closePresetPop(presetOrigWL ?? void 0);
      return;
    }
    presetOrigWL = cfg.wl.get();
    const orig = presetOrigWL;
    const anchor = buttons.get("presets").getBoundingClientRect();
    presetPop = document.createElement("div");
    presetPop.id = "ira-preset-pop";
    presetPop.style.cssText = `position:fixed;top:${Math.round(anchor.bottom + 4)}px;left:${Math.round(anchor.left)}px;z-index:1200;min-width:180px;background:#11141d;border:1px solid #33507e;border-radius:8px;padding:5px;box-shadow:0 10px 30px rgba(0,0,0,.6);font:12px -apple-system,system-ui,sans-serif;color:#d6e2f2;`;
    const head = document.createElement("div");
    head.textContent = "Window presets \u2014 hover to preview";
    head.style.cssText = "font-size:10px;color:#5a6b85;padding:3px 8px 5px;text-transform:uppercase;letter-spacing:.5px;";
    presetPop.appendChild(head);
    for (const opt of [...cfg.presetsEl.options].filter((o) => o.value !== "current")) {
      const item = document.createElement("div");
      item.className = "ira-preset-item";
      item.dataset.value = opt.value;
      item.textContent = opt.textContent;
      item.style.cssText = "padding:6px 10px;border-radius:5px;cursor:pointer;white-space:nowrap;";
      item.addEventListener("pointerenter", () => {
        for (const el of presetPop.querySelectorAll(".ira-preset-item")) {
          el.style.background = "transparent";
        }
        item.style.background = "#2b6cb0";
        cfg.presetsEl.value = opt.value;
        cfg.presetsEl.dispatchEvent(new Event("change"));
      });
      item.addEventListener("click", () => closePresetPop());
      presetPop.appendChild(item);
    }
    presetEsc = (e) => {
      if (e.key === "Escape") closePresetPop(orig);
    };
    document.body.appendChild(presetPop);
    setTimeout(() => {
      document.addEventListener("pointerdown", onPresetOutside, true);
      document.addEventListener("keydown", presetEsc, true);
    }, 0);
  };
  btn("presets", "presets", "Window Presets \u2014 hover to preview on the volume, click to keep (CT: Soft Tissue / Lung / Bone / Brain + Auto)", openPresetPop);
  btn("wl-clamp", "wlMode", "Window mode \u2014 Center/Width vs Clamped (lower limit fixed at 0; default for NM/PET)", () => {
    clamped = !clamped;
    buttons.get("wl-clamp").classList.toggle("active", clamped);
    if (clamped) {
      const [w] = cfg.wl.get();
      cfg.wl.set(w, w / 2);
    }
  });
  if (clamped) {
    const [w] = cfg.wl.get();
    cfg.wl.set(w, w / 2);
  }
  buttons.get("wl-clamp").classList.toggle("active", clamped);
  btn("invert", "invert", "Invert grayscale", () => {
    inverted = !inverted;
    buttons.get("invert").classList.toggle("active", inverted);
    for (const p of cfg.planes) cfg.canvases[p].style.filter = inverted ? "invert(1)" : "";
  });
  sep();
  btn("localizer", "localizer", "Localizer lines \u2014 O", () => {
    localizerOn = !localizerOn;
    buttons.get("localizer").classList.toggle("active", localizerOn);
    redrawOverlays();
  });
  btn("linksync", "link", "Link/unlink synchronized scrolling \u2014 L/U", void 0, {
    disabled: true,
    disabledTip: "Link/unlink synchronization \u2014 MPR planes of one series are always linked; enabled with compare viewports"
  });
  btn("annotation", "annotation", "Annotation \u2014 cycle full / minimal / off \u2014 I", () => {
    annState = (annState + 1) % 3;
    cfg.grid.classList.toggle("ann-min", annState === 1);
    cfg.grid.classList.toggle("ann-none", annState === 2);
    buttons.get("annotation").classList.toggle("active", annState !== 0);
  });
  sep();
  let cineBox = null;
  let cineTimer = null;
  const cineStop = () => {
    if (cineTimer !== null) clearInterval(cineTimer);
    cineTimer = null;
  };
  const cineClose = () => {
    cineStop();
    cineBox?.remove();
    cineBox = null;
    buttons.get("cine").classList.remove("active");
  };
  const cinePlay = (fps) => {
    cineStop();
    cineTimer = setInterval(() => {
      const before = cfg.off01(selectedPlane);
      cfg.step(selectedPlane, true);
      if (cfg.off01(selectedPlane) === before) cfg.setOff01(selectedPlane, 0);
      cfg.redraw(selectedPlane);
    }, 1e3 / Math.max(1, fps));
  };
  const cineToggle = () => {
    if (cineBox) {
      cineClose();
      return;
    }
    buttons.get("cine").classList.add("active");
    cineBox = document.createElement("div");
    cineBox.id = "ira-cine-box";
    const fps = document.createElement("input");
    fps.type = "number";
    fps.value = "10";
    fps.title = "Frame rate (frames/second)";
    const cbtn = (icon, tip, fn) => {
      const b = document.createElement("button");
      b.innerHTML = ICONS[icon];
      b.title = tip;
      b.addEventListener("click", fn);
      cineBox.appendChild(b);
      return b;
    };
    cbtn("play", "Cine Play (loop)", () => cinePlay(Number(fps.value) || 10));
    cbtn("pause", "Cine Pause", cineStop);
    cbtn("stop", "Cine Stop (back to first frame)", () => {
      cineStop();
      cfg.setOff01(selectedPlane, 0);
      cfg.redraw(selectedPlane);
    });
    cbtn("toStart", "Go to first frame", () => {
      cfg.setOff01(selectedPlane, 0);
      cfg.redraw(selectedPlane);
    });
    cbtn("stepFwd", "Step one frame", () => stepSelected(true));
    cbtn("toEnd", "Go to last frame", () => {
      cfg.setOff01(selectedPlane, 1);
      cfg.redraw(selectedPlane);
    });
    cineBox.appendChild(fps);
    const close = document.createElement("button");
    close.textContent = "\xD7";
    close.title = "Close cine";
    close.addEventListener("click", cineClose);
    cineBox.appendChild(close);
    tbBody.appendChild(cineBox);
    fps.addEventListener("change", () => {
      if (cineTimer !== null) cinePlay(Number(fps.value) || 10);
    });
    cinePlay(10);
  };
  btn("cine", "cine", "Cine (play through frames) \u2014 C", cineToggle);
  sep();
  const rotatable = /^(DX|CR)$/i.test(cfg.modality);
  btn("rotate", "rotate", "Rotate 90\xB0 clockwise", void 0, {
    disabled: !rotatable,
    disabledTip: "Rotate 90\xB0 \u2014 only active for DX/CR projection images (safety: IRA 4.16.4.2.2.5.2)"
  });
  btn("flip", "flip", "Flip horizontally", void 0, {
    disabled: !rotatable,
    disabledTip: "Flip \u2014 only active for DX/CR projection images (safety: IRA 4.16.4.2.2.5.2)"
  });
  sep();
  const reset = () => {
    cfg.resetViews();
    cfg.wl.set(...cfg.wl.auto);
    measurements.length = 0;
    pending = null;
    if (inverted) buttons.get("invert").click();
    cfg.redrawAll();
    redrawOverlays();
  };
  btn("reset", "reset", "Display Reset (zoom/pan/window/measurements; keeps scroll position) \u2014 Esc", reset);
  btn("print", "print", "Print selected viewport \u2014 P", () => printViewport());
  btn("report", "report", "Show Report \u2014 R", void 0, {
    disabled: true,
    disabledTip: "Show Report \u2014 no reports available for this patient"
  });
  btn("help", "help", "Help \u2014 tool guide and keyboard shortcuts \u2014 H / F1", () => showHelp());
  btn("advanced", "advanced", "Advanced Mode \u2014 toggle IRA basic (single stack) vs MPR Four-Up + 3D \u2014 F10", () => {
    setLayout(layout === "single" ? "fourUp" : "single");
  });
  if (cfg.extraTools?.length) {
    sep();
    for (const t of cfg.extraTools) btn(t.id, t.icon, t.title, t.run);
  }
  const setLayout = (l) => {
    layout = l;
    const show = {};
    const names = [...cfg.planes, "threeD"];
    if (l === "single") for (const n of names) show[n] = n === selectedPlane;
    else if (l === "twoUp") for (const n of names) show[n] = n === "axial" || n === "sagittal";
    else for (const n of names) show[n] = true;
    for (const n of names) {
      const cell = cfg.cellOf(n);
      if (cell) cell.style.display = show[n] ? "" : "none";
    }
    cfg.grid.style.gridTemplateColumns = l === "single" ? "1fr" : "1fr 1fr";
    cfg.grid.style.gridTemplateRows = l === "fourUp" ? "1fr 1fr" : "1fr";
    layoutSel.value = l;
    buttons.get("advanced").classList.toggle("active", l === "fourUp");
    globalThis.dispatchEvent(new Event("resize"));
  };
  let strip = null;
  if (cfg.strip?.length) {
    strip = document.createElement("div");
    strip.id = "ira-strip";
    strip.innerHTML = `<div class="head">Series</div>`;
    for (const it2 of cfg.strip) {
      const item = document.createElement("div");
      item.className = "item" + (it2.current ? " current" : "");
      item.title = it2.lines.join(" \xB7 ") + " \u2014 double-click to load";
      const img = document.createElement("img");
      img.alt = "";
      it2.thumb().then((b) => {
        if (b) img.src = URL.createObjectURL(b);
      }).catch(() => {
      });
      const cap = document.createElement("div");
      cap.className = "cap";
      cap.textContent = it2.lines.join(" \xB7 ");
      item.append(img, cap);
      item.addEventListener("dblclick", () => it2.open());
      strip.appendChild(item);
    }
    const row = document.createElement("div");
    row.id = "ira-row";
    row.style.cssText = "flex:1;display:flex;min-height:0;";
    cfg.grid.parentElement.insertBefore(row, cfg.grid);
    row.append(strip, cfg.grid);
  }
  const stripBtn = btn("thumbs", "thumbs", "Show/hide the series thumbnail strip", () => {
    if (!strip) return;
    strip.style.display = strip.style.display === "none" ? "" : "none";
    stripBtn.classList.toggle("active", strip.style.display !== "none");
    globalThis.dispatchEvent(new Event("resize"));
  });
  if (!strip) {
    stripBtn.disabled = true;
    stripBtn.title = "Series thumbnail strip \u2014 no sibling series available";
  } else stripBtn.classList.add("active");
  const dc = displayClass();
  if (dc === "phone") {
    if (strip) strip.style.display = "none";
    stripBtn.classList.remove("active");
    setLayout("single");
  } else {
    setLayout(cfg.initialLayout ?? "fourUp");
  }
  const printViewport = () => {
    const p = selectedPlane;
    cfg.redraw(p);
    requestAnimationFrame(() => {
      setTimeout(() => {
        const src = cfg.canvases[p];
        const ov = overlays.get(p);
        const out = document.createElement("canvas");
        out.width = src.width;
        out.height = src.height;
        const ctx = out.getContext("2d");
        try {
          ctx.drawImage(src, 0, 0);
          if (inverted) {
            ctx.globalCompositeOperation = "difference";
            ctx.fillStyle = "#fff";
            ctx.fillRect(0, 0, out.width, out.height);
            ctx.globalCompositeOperation = "source-over";
          }
          if (ov) ctx.drawImage(ov, 0, 0, out.width, out.height);
        } catch {
        }
        const [win, lev] = cfg.wl.get();
        const w = globalThis.open("", "_blank", "width=900,height=700");
        if (!w) return;
        w.document.write(
          `<title>SlicerRad print \u2014 ${p}</title><p style="font:12px sans-serif">For Investigational Use Only \u2014 Not for Diagnostic Use \xB7 ${p} \xB7 W ${Math.round(win)} / L ${Math.round(lev)}</p><img style="max-width:100%" src="${out.toDataURL("image/png")}">`
        );
        w.document.close();
        w.focus();
        setTimeout(() => w.print(), 200);
      }, 60);
    });
  };
  const showHelp = () => {
    if (document.getElementById("ira-help-overlay")) return;
    const h = document.createElement("div");
    h.id = "ira-help-overlay";
    h.style.cssText = "position:fixed;inset:0;z-index:2000;background:rgba(5,6,10,.85);display:flex;align-items:center;justify-content:center;";
    h.innerHTML = `
      <div style="background:#11141d;border:1px solid #33507e;border-radius:10px;padding:20px 26px;
        max-width:560px;max-height:80vh;overflow:auto;font:13px -apple-system,system-ui,sans-serif;color:#d6e2f2;">
        <h3 style="margin:0 0 10px;color:#fff;">SlicerRad reader \u2014 IHE Basic Image Review tools</h3>
        <table style="border-collapse:collapse;width:100%;">
        ${[
      ["S", "Scroll through slices (default; wheel always scrolls)"],
      ["W", "Window \u2014 drag to adjust window/level"],
      ["Z / T", "Zoom / Pan (translate)"],
      ["V", "Select viewport (click also selects)"],
      ["D / A", "Distance (2 clicks) / Angle (4 clicks, Cobb) measurements"],
      ["J", "Cross-hair \u2014 click localizes in all viewports"],
      ["O", "Localizer lines on/off"],
      ["I", "Annotations \u2014 full / minimal / off"],
      ["C", "Cine \u2014 plays immediately; controls appear in the toolbar"],
      ["\u2190 \u2191 / \u2192 \u2193", "Previous / next frame"],
      ["PgUp / PgDn", "Previous / next series (FrameSet)"],
      ["Esc", "Display Reset \u2014 zoom/pan/window/measurements (keeps scroll position)"],
      ["P", "Print selected viewport"],
      ["F10", "Advanced Mode \u2014 Four-Up MPR + 3D vs single-stack"],
      ["H / F1", "This help"]
    ].map(
      ([k2, d]) => `<tr><td style="padding:3px 14px 3px 0;color:#9fd0b3;white-space:nowrap;font-weight:600;">${k2}</td><td style="padding:3px 0;color:#9fb3d0;">${d}</td></tr>`
    ).join("")}
        </table>
        <p style="color:#9fb3d0;margin:14px 0 4px;">This reader implements the IHE Radiology <b>Basic Image Review (IRA)</b> profile \u2014
        its tool set, icons and shortcuts follow the supplement (Rev 1.3 \xA74.16.4.2.2.5.13).</p>
        <p style="margin:0 0 4px;"><a id="ira-docs-link" href="https://profiles.ihe.net/RAD/" target="_blank" rel="noopener"
          style="color:#6cc4ff;">IHE Radiology profiles (Basic Image Review) \u2192</a></p>
        <p style="margin:0;"><a href="https://www.ihe.net/uploadedFiles/Documents/Radiology/IHE_RAD_Suppl_BIR_Rev1.3_TI_2016_09-09.pdf" target="_blank" rel="noopener"
          style="color:#6cc4ff;">IRA Technical Framework Supplement (PDF) \u2192</a></p>
        <button id="ira-help-close" style="margin-top:14px;font:600 12px -apple-system,system-ui,sans-serif;
          color:#d6e2f2;background:#1b2740;border:1px solid #33507e;border-radius:4px;padding:5px 14px;cursor:pointer;">Close</button>
      </div>`;
    document.body.appendChild(h);
    h.querySelector("#ira-help-close").addEventListener("click", () => h.remove());
    h.addEventListener("click", (e) => {
      if (e.target === h) h.remove();
    });
  };
  const onKey = (e) => {
    if (!document.body.contains(cfg.overlay)) return;
    const t = e.target;
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(t?.tagName ?? "")) return;
    const k2 = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    const map = {
      s: () => setTool("scroll"),
      w: () => setTool("wl"),
      z: () => setTool("zoom"),
      t: () => setTool("pan"),
      v: () => setTool("select"),
      d: () => setTool("distance"),
      a: () => setTool("angle"),
      j: () => setTool("crosshair"),
      o: () => buttons.get("localizer").click(),
      i: () => buttons.get("annotation").click(),
      c: () => cineToggle(),
      p: () => printViewport(),
      h: () => showHelp(),
      F1: () => showHelp(),
      F10: () => buttons.get("advanced").click(),
      Escape: () => reset(),
      ArrowLeft: () => stepSelected(false),
      ArrowUp: () => stepSelected(false),
      ArrowRight: () => stepSelected(true),
      ArrowDown: () => stepSelected(true),
      PageUp: () => cfg.nav?.prevSeries?.(),
      PageDown: () => cfg.nav?.nextSeries?.()
    };
    const fn = map[k2];
    if (fn) {
      e.preventDefault();
      fn();
    }
  };
  document.addEventListener("keydown", onKey);
  const hooks = (p) => ({
    onLeftGrab: (u, v2, w, h) => {
      if (tool === "select") return true;
      if (tool === "crosshair") {
        crossRas = rasOfClick(p, u, v2, w, h);
        cfg.jumpAll(crossRas);
        redrawOverlays();
        return true;
      }
      if (tool === "distance" || tool === "angle") {
        const ras = rasOfClick(p, u, v2, w, h);
        if (!pending || pending.plane !== p) {
          pending = { kind: tool, plane: p, mm: cfg.offsetMm(p), pts: [ras] };
        } else {
          pending.pts.push(ras);
          const needed = tool === "distance" ? 2 : 4;
          if (pending.pts.length >= needed) {
            measurements.push(pending);
            pending = null;
          }
        }
        drawOverlay(p);
        return true;
      }
      return false;
    }
  });
  resizeOverlays();
  return {
    tool: () => tool,
    setTool,
    measureCount: () => measurements.length,
    leftMode: () => tool === "wl" || tool === "zoom" || tool === "pan" ? tool : "scroll",
    hooks,
    drawOverlay,
    resize: () => {
      resizeOverlays();
      redrawOverlays();
    },
    selected: () => selectedPlane,
    setLayout,
    reset,
    detachKeys: () => {
      document.removeEventListener("keydown", onKey);
      cineStop();
    }
  };
}

// render/demos/idc-share.ts
function studyShareURL(src) {
  const u = new URL(globalThis.location.origin + globalThis.location.pathname);
  if (src.st) u.searchParams.set("StudyInstanceUIDs", src.st);
  u.searchParams.set("series", src.c);
  if (src.cb && src.cb !== "idc-open-data") u.searchParams.set("bucket", src.cb);
  if (src.s) u.searchParams.set("seg", src.s);
  if (src.sb && src.sb !== "idc-open-data") u.searchParams.set("segBucket", src.sb);
  if (src.m && src.m !== "CT") u.searchParams.set("modality", src.m);
  const pid = (src.sd.split("\xB7")[0] || "").trim();
  if (pid && pid !== "IDC") u.searchParams.set("patient", pid);
  if (src.col && src.col !== "IDC") u.searchParams.set("collection", src.col);
  return u.toString();
}
var OVERLAY_CSS = "position:fixed;inset:0;z-index:2000;background:rgba(5,6,10,.85);display:flex;align-items:center;justify-content:center;";
var CARD_CSS = "background:#11141d;border:1px solid #33507e;border-radius:10px;padding:20px 24px;width:600px;max-width:92vw;font:13px -apple-system,system-ui,sans-serif;color:#d6e2f2;";
function shareStudy(src) {
  if (document.getElementById("idc-share")) return;
  const here = studyShareURL(src);
  const portal = src.st && ohifViewerURL(src.st) || "";
  const overlay = document.createElement("div");
  overlay.id = "idc-share";
  overlay.style.cssText = OVERLAY_CSS;
  const esc = (s) => s.replace(/"/g, "&quot;");
  const row = (label, url, hint) => `
    <div style="margin:0 0 14px;">
      <div style="color:#9fd0b3;font-weight:600;margin-bottom:4px;">${label}</div>
      <div style="display:flex;gap:6px;">
        <input readonly value="${esc(url)}" style="flex:1;font:12px ui-monospace,monospace;color:#d6e2f2;
          background:#0b0e16;border:1px solid #33507e;border-radius:4px;padding:6px 8px;">
        <button data-copy="${esc(url)}" style="font:600 12px -apple-system,system-ui,sans-serif;color:#d6e2f2;
          background:#1b2740;border:1px solid #33507e;border-radius:4px;padding:6px 12px;cursor:pointer;">Copy</button>
      </div>
      <div style="color:#5a6b85;font-size:11px;margin-top:3px;">${hint}</div>
    </div>`;
  overlay.innerHTML = `
    <div style="${CARD_CSS}">
      <h3 style="margin:0 0 4px;color:#fff;">Share study</h3>
      <p style="margin:0 0 16px;color:#9fb3d0;">${src.sd}${src.st ? " \xB7 " + src.st : ""}</p>
      ${row("Open in SlicerLive (this viewer)", here, "Reopens this study in the SlicerLive BIR reader (OHIF-style + direct-S3 params).")}
      ${portal ? row("Open in IDC OHIF portal", portal, "The canonical IDC viewer link for the same study.") : ""}
      <div style="text-align:right;margin-top:6px;">
        <button id="idc-share-close" style="font:600 12px -apple-system,system-ui,sans-serif;color:#9fb3d0;
          background:none;border:none;cursor:pointer;padding:6px 4px;">Close</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);
  const close = () => overlay.remove();
  overlay.addEventListener("click", (e) => e.target === overlay && close());
  overlay.querySelector("#idc-share-close").addEventListener("click", close);
  for (const b of overlay.querySelectorAll("button[data-copy]")) {
    b.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(b.dataset.copy);
        b.textContent = "Copied \u2713";
        setTimeout(() => b.textContent = "Copy", 1400);
      } catch {
        b.textContent = "Copy failed";
      }
    });
  }
}
var DL_LANES = Math.max(2, Math.min(6, (navigator.hardwareConcurrency || 4) - 1));
var safeName = (s) => s.replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 80) || "study";
async function downloadStudy(src, seriesList, onProgress, opts) {
  let root;
  if (opts?.root) root = opts.root;
  else {
    const picker = globalThis.showDirectoryPicker;
    if (!picker) throw new Error("File System Access API unavailable \u2014 cannot pick a download folder");
    root = await picker({ mode: "readwrite", id: "slicerlive-idc-download" });
  }
  onProgress(0, 0, 0, "listing series\u2026");
  const studyDir = await root.getDirectoryHandle(
    safeName(`${src.col}_${(src.st || src.c).slice(-12)}`),
    { create: true }
  );
  const tasks = [];
  let total = 0, sn2 = 0;
  for (const s of seriesList) {
    onProgress(0, total, 0, `listing ${s.modality} ${s.seriesDescription || s.prefix.slice(0, 8) + "\u2026"}\u2026`);
    let keys = await s3ListKeys(s.prefix, s.bucket);
    if (!keys.length) continue;
    if (opts?.maxFilesPerSeries) keys = keys.slice(0, opts.maxFilesPerSeries);
    total += keys.length;
    const dir = await studyDir.getDirectoryHandle(
      safeName(`${s.seriesNumber || ++sn2}_${s.modality}_${s.prefix.slice(0, 8)}`),
      { create: true }
    );
    tasks.push({ dir, base: idcS3(s.bucket), keys });
  }
  if (!total) throw new Error("no DICOM objects found to download");
  onProgress(0, total, 0, `downloading ${total} files (${DL_LANES} at a time)\u2026`);
  let done = 0, bytes = 0;
  const errors = [];
  for (const task of tasks) {
    let next = 0;
    const lane = async () => {
      while (next < task.keys.length) {
        const key = task.keys[next++];
        try {
          const resp = await fetchRetry(task.base + key, {});
          const buf = await resp.arrayBuffer();
          const fname = safeName(key.split("/").pop() || `obj-${done}.dcm`);
          const fh = await task.dir.getFileHandle(fname, { create: true });
          const w = await fh.createWritable();
          await w.write(buf);
          await w.close();
          bytes += buf.byteLength;
        } catch (e) {
          errors.push(`${key}: ${e.message}`);
        }
        done++;
        if (done % 5 === 0 || done === total) onProgress(done, total, bytes, `${done}/${total} files`);
      }
    };
    await Promise.all(Array.from({ length: DL_LANES }, lane));
  }
  const mb = (bytes / 1e6).toFixed(0);
  onProgress(
    done,
    total,
    bytes,
    `saved ${done - errors.length}/${total} files (${mb} MB) to \u201C${root.name}/${studyDir.name}\u201D` + (errors.length ? ` \u2014 ${errors.length} failed` : "")
  );
  if (errors.length) console.warn("IDC download errors:", errors.slice(0, 10));
}
async function downloadStudyWithDialog(src, listAll) {
  if (document.getElementById("idc-download")) return;
  const overlay = document.createElement("div");
  overlay.id = "idc-download";
  overlay.style.cssText = OVERLAY_CSS;
  overlay.innerHTML = `
    <div style="${CARD_CSS.replace("width:600px", "width:560px")}">
      <h3 style="margin:0 0 4px;color:#fff;">Downloading study</h3>
      <p style="margin:0 0 6px;color:#9fb3d0;">${src.sd}${src.st ? " \xB7 " + src.st : ""}</p>
      <p style="margin:0 0 14px;color:#5a6b85;font-size:11px;">${src.lic || "NCI Imaging Data Commons \u2014 open data"}</p>
      <div style="height:8px;background:#0b0e16;border:1px solid #1b2740;border-radius:5px;overflow:hidden;">
        <div id="idc-dl-bar" style="height:100%;width:0%;background:#2b6cb0;transition:width .2s;"></div>
      </div>
      <p id="idc-dl-note" style="margin:10px 0 0;color:#9fb3d0;font-variant-numeric:tabular-nums;">starting\u2026</p>
      <div style="text-align:right;margin-top:12px;">
        <button id="idc-dl-close" style="font:600 12px -apple-system,system-ui,sans-serif;color:#9fb3d0;
          background:none;border:none;cursor:pointer;padding:6px 4px;">Close</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);
  const bar = overlay.querySelector("#idc-dl-bar");
  const note = overlay.querySelector("#idc-dl-note");
  const closeBtn = overlay.querySelector("#idc-dl-close");
  closeBtn.addEventListener("click", () => overlay.remove());
  try {
    const loaded = [{ prefix: src.c, bucket: src.cb, modality: src.m }];
    if (src.s) loaded.push({ prefix: src.s, bucket: src.sb || "idc-open-data", modality: "SEG" });
    const seriesList = listAll ? await listAll().catch(() => loaded) : loaded;
    await downloadStudy(src, seriesList.length ? seriesList : loaded, (done, total, bytes, msg) => {
      bar.style.width = (total ? 100 * done / total : 0).toFixed(1) + "%";
      note.textContent = total ? `${done.toLocaleString("en-US")} / ${total.toLocaleString("en-US")} files \xB7 ${(bytes / 1e6).toFixed(0)} MB \u2014 ${msg}` : msg;
    });
    bar.style.width = "100%";
    closeBtn.textContent = "Done";
  } catch (err) {
    note.textContent = "download failed: " + err.message;
    note.style.color = "#ff6b74";
    closeBtn.textContent = "Close";
  }
}
globalThis.__slicerLiveIdc = {
  studyShareURL,
  shareStudy,
  downloadStudy,
  downloadStudyWithDialog
};

// render/demos/segment-stl.ts
var OVERLAY_CSS2 = "position:fixed;inset:0;z-index:2000;background:rgba(5,6,10,.85);display:flex;align-items:center;justify-content:center;";
var CARD_CSS2 = "background:#11141d;border:1px solid #33507e;border-radius:10px;padding:20px 24px;width:420px;max-width:92vw;max-height:80vh;overflow:auto;font:13px -apple-system,system-ui,sans-serif;color:#d6e2f2;";
function safeStlFileName(name) {
  const s = name.replace(/[^\w.\-]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 80);
  return (s || "segment") + ".stl";
}
function downloadStl(buffer, filename) {
  const blob = new Blob([buffer], { type: "model/stl" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
function pickSegment(segments) {
  return new Promise((resolve) => {
    if (document.getElementById("seg-stl-pick")) {
      resolve(void 0);
      return;
    }
    const overlay = document.createElement("div");
    overlay.id = "seg-stl-pick";
    overlay.style.cssText = OVERLAY_CSS2;
    const rows2 = segments.map((s) => {
      const [r, g, b] = s.color.map((c) => Math.round(c * 255));
      const swatch = `display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:8px;background:rgb(${r},${g},${b});vertical-align:middle;`;
      return `<button type="button" data-num="${s.num}" style="display:block;width:100%;text-align:left;
          font:600 13px -apple-system,system-ui,sans-serif;color:#d6e2f2;background:#0b0e16;
          border:1px solid #1b2740;border-radius:6px;padding:10px 12px;margin:0 0 6px;cursor:pointer;">
          <span style="${swatch}"></span>${escapeHtml(s.name)}
          <span style="color:#5a6b85;font-weight:500;margin-left:6px;">#${s.num}</span>
        </button>`;
    }).join("");
    overlay.innerHTML = `
      <div style="${CARD_CSS2}">
        <h3 style="margin:0 0 4px;color:#fff;">Save segment as STL</h3>
        <p style="margin:0 0 14px;color:#9fb3d0;">Binary STL in LPS (3D-print convention). Pick a segment:</p>
        <div id="seg-stl-list">${rows2}</div>
        <div style="text-align:right;margin-top:10px;">
          <button type="button" id="seg-stl-cancel" style="font:600 12px -apple-system,system-ui,sans-serif;
            color:#9fb3d0;background:none;border:none;cursor:pointer;padding:6px 4px;">Cancel</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    const done = (seg) => {
      overlay.remove();
      resolve(seg);
    };
    overlay.querySelector("#seg-stl-cancel").addEventListener("click", () => done());
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) done();
    });
    overlay.querySelectorAll("#seg-stl-list button").forEach((btn) => {
      btn.addEventListener("click", () => {
        const num2 = Number(btn.dataset.num);
        done(segments.find((s) => s.num === num2));
      });
      btn.addEventListener("mouseenter", () => {
        btn.style.borderColor = "#2b6cb0";
        btn.style.background = "#151a28";
      });
      btn.addEventListener("mouseleave", () => {
        btn.style.borderColor = "#1b2740";
        btn.style.background = "#0b0e16";
      });
    });
  });
}
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function segmentToStl(lab, dims, ijkToRAS, segmentNum) {
  const [nx, ny, nz] = dims;
  if (lab.length !== nx * ny * nz) {
    throw new Error(`labelmap size ${lab.length} != dims ${nx}\xD7${ny}\xD7${nz}`);
  }
  let i0 = nx, j0 = ny, k0 = nz, i1 = -1, j1 = -1, k1 = -1;
  for (let k2 = 0; k2 < nz; k2++) {
    for (let j2 = 0; j2 < ny; j2++) {
      const base = (k2 * ny + j2) * nx;
      for (let i = 0; i < nx; i++) {
        if (lab[base + i] !== segmentNum) continue;
        if (i < i0) i0 = i;
        if (j2 < j0) j0 = j2;
        if (k2 < k0) k0 = k2;
        if (i > i1) i1 = i;
        if (j2 > j1) j1 = j2;
        if (k2 > k1) k1 = k2;
      }
    }
  }
  if (i1 < 0) throw new Error(`segment ${segmentNum} is empty`);
  const x0 = i0 - 1, y0 = j0 - 1, z0 = k0 - 1;
  const x1 = i1 + 1, y1 = j1 + 1, z1 = k1 + 1;
  const sx = x1 - x0 + 1, sy = y1 - y0 + 1, sz = z1 - z0 + 1;
  const mask = new Uint8Array(sx * sy * sz);
  for (let k2 = 0; k2 < sz; k2++) {
    const gk = k2 + z0;
    for (let j2 = 0; j2 < sy; j2++) {
      const gj = j2 + y0;
      for (let i = 0; i < sx; i++) {
        const gi2 = i + x0;
        if (gi2 < 0 || gj < 0 || gk < 0 || gi2 >= nx || gj >= ny || gk >= nz) continue;
        if (lab[(gk * ny + gj) * nx + gi2] === segmentNum) mask[(k2 * sy + j2) * sx + i] = 1;
      }
    }
  }
  const tris = marchingCubes(mask, sx, sy, sz, x0, y0, z0);
  if (!tris.length) throw new Error(`segment ${segmentNum}: no surface triangles`);
  const M2 = ijkToRAS;
  const lps = new Array(tris.length);
  for (let t = 0; t < tris.length; t += 9) {
    for (let v2 = 0; v2 < 3; v2++) {
      const i = tris[t + v2 * 3], j2 = tris[t + v2 * 3 + 1], k2 = tris[t + v2 * 3 + 2];
      const rx = M2[0] * i + M2[1] * j2 + M2[2] * k2 + M2[3];
      const ry = M2[4] * i + M2[5] * j2 + M2[6] * k2 + M2[7];
      const rz = M2[8] * i + M2[9] * j2 + M2[10] * k2 + M2[11];
      lps[t + v2 * 3] = -rx;
      lps[t + v2 * 3 + 1] = -ry;
      lps[t + v2 * 3 + 2] = rz;
    }
  }
  return writeBinaryStl(lps, "SlicerLive IRA. SPACE=LPS");
}
function writeBinaryStl(flatTris, headerText) {
  const nTri = flatTris.length / 9 | 0;
  const buf = new ArrayBuffer(84 + nTri * 50);
  const u8 = new Uint8Array(buf);
  const dv = new DataView(buf);
  const hdr = new TextEncoder().encode(headerText.slice(0, 79));
  u8.set(hdr, 0);
  dv.setUint32(80, nTri, true);
  let off = 84;
  for (let t = 0; t < nTri; t++) {
    const b = t * 9;
    const ax = flatTris[b], ay = flatTris[b + 1], az = flatTris[b + 2];
    const bx = flatTris[b + 3], by = flatTris[b + 4], bz = flatTris[b + 5];
    const cx = flatTris[b + 6], cy = flatTris[b + 7], cz = flatTris[b + 8];
    let nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay);
    let ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
    let nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len;
    ny /= len;
    nz /= len;
    dv.setFloat32(off, nx, true);
    dv.setFloat32(off + 4, ny, true);
    dv.setFloat32(off + 8, nz, true);
    dv.setFloat32(off + 12, ax, true);
    dv.setFloat32(off + 16, ay, true);
    dv.setFloat32(off + 20, az, true);
    dv.setFloat32(off + 24, bx, true);
    dv.setFloat32(off + 28, by, true);
    dv.setFloat32(off + 32, bz, true);
    dv.setFloat32(off + 36, cx, true);
    dv.setFloat32(off + 40, cy, true);
    dv.setFloat32(off + 44, cz, true);
    dv.setUint16(off + 48, 0, true);
    off += 50;
  }
  return buf;
}
var EDGE_TABLE = new Uint16Array([
  0,
  265,
  515,
  778,
  1030,
  1295,
  1541,
  1804,
  2060,
  2309,
  2575,
  2822,
  3082,
  3331,
  3593,
  3840,
  400,
  153,
  915,
  666,
  1430,
  1183,
  1941,
  1692,
  2460,
  2197,
  2975,
  2710,
  3482,
  3219,
  3993,
  3728,
  560,
  825,
  51,
  314,
  1590,
  1855,
  1077,
  1340,
  2620,
  2869,
  2111,
  2358,
  3642,
  3891,
  3129,
  3376,
  928,
  681,
  419,
  170,
  1958,
  1711,
  1445,
  1196,
  2988,
  2725,
  2479,
  2214,
  4010,
  3747,
  3497,
  3232,
  1120,
  1385,
  1635,
  1898,
  102,
  367,
  613,
  876,
  3180,
  3429,
  3695,
  3942,
  2154,
  2403,
  2665,
  2912,
  1520,
  1273,
  2035,
  1786,
  502,
  255,
  1013,
  764,
  3580,
  3317,
  4095,
  3830,
  2554,
  2291,
  3065,
  2800,
  1616,
  1881,
  1107,
  1370,
  598,
  863,
  85,
  348,
  3676,
  3925,
  3167,
  3414,
  2650,
  2899,
  2137,
  2384,
  1984,
  1737,
  1475,
  1226,
  966,
  719,
  453,
  204,
  4044,
  3781,
  3535,
  3270,
  3018,
  2755,
  2505,
  2240,
  2240,
  2505,
  2755,
  3018,
  3270,
  3535,
  3781,
  4044,
  204,
  453,
  719,
  966,
  1226,
  1475,
  1737,
  1984,
  2384,
  2137,
  2899,
  2650,
  3414,
  3167,
  3925,
  3676,
  348,
  85,
  863,
  598,
  1370,
  1107,
  1881,
  1616,
  2800,
  3065,
  2291,
  2554,
  3830,
  4095,
  3317,
  3580,
  764,
  1013,
  255,
  502,
  1786,
  2035,
  1273,
  1520,
  2912,
  2665,
  2403,
  2154,
  3942,
  3695,
  3429,
  3180,
  876,
  613,
  367,
  102,
  1898,
  1635,
  1385,
  1120,
  3232,
  3497,
  3747,
  4010,
  2214,
  2479,
  2725,
  2988,
  1196,
  1445,
  1711,
  1958,
  170,
  419,
  681,
  928,
  3376,
  3129,
  3891,
  3642,
  2358,
  2111,
  2869,
  2620,
  1340,
  1077,
  1855,
  1590,
  314,
  51,
  825,
  560,
  3728,
  3993,
  3219,
  3482,
  2710,
  2975,
  2197,
  2460,
  1692,
  1941,
  1183,
  1430,
  666,
  915,
  153,
  400,
  3840,
  3593,
  3331,
  3082,
  2822,
  2575,
  2309,
  2060,
  1804,
  1541,
  1295,
  1030,
  778,
  515,
  265,
  0
]);
var TRI_TABLE = (() => {
  const raw = [
    [],
    [0, 8, 3],
    [0, 1, 9],
    [1, 8, 3, 9, 8, 1],
    [1, 2, 10],
    [0, 8, 3, 1, 2, 10],
    [9, 2, 10, 0, 2, 9],
    [2, 8, 3, 2, 10, 8, 10, 9, 8],
    [3, 11, 2],
    [0, 11, 2, 8, 11, 0],
    [1, 9, 0, 2, 3, 11],
    [1, 11, 2, 1, 9, 11, 9, 8, 11],
    [3, 10, 1, 11, 10, 3],
    [0, 10, 1, 0, 8, 10, 8, 11, 10],
    [3, 9, 0, 3, 11, 9, 11, 10, 9],
    [9, 8, 10, 10, 8, 11],
    [4, 7, 8],
    [4, 3, 0, 7, 3, 4],
    [0, 1, 9, 8, 4, 7],
    [4, 1, 9, 4, 7, 1, 7, 3, 1],
    [1, 2, 10, 8, 4, 7],
    [3, 4, 7, 3, 0, 4, 1, 2, 10],
    [9, 2, 10, 9, 0, 2, 8, 4, 7],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4],
    [8, 4, 7, 3, 11, 2],
    [11, 4, 7, 11, 2, 4, 2, 0, 4],
    [9, 0, 1, 8, 4, 7, 2, 3, 11],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3],
    [4, 7, 11, 4, 11, 9, 9, 11, 10],
    [9, 5, 4],
    [9, 5, 4, 0, 8, 3],
    [0, 5, 4, 1, 5, 0],
    [8, 5, 4, 8, 3, 5, 3, 1, 5],
    [1, 2, 10, 9, 5, 4],
    [3, 0, 8, 1, 2, 10, 4, 9, 5],
    [5, 2, 10, 5, 4, 2, 4, 0, 2],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8],
    [9, 5, 4, 2, 3, 11],
    [0, 11, 2, 0, 8, 11, 4, 9, 5],
    [0, 5, 4, 0, 1, 5, 2, 3, 11],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5],
    [10, 3, 11, 10, 1, 3, 9, 5, 4],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3],
    [5, 4, 8, 5, 8, 10, 10, 8, 11],
    [9, 7, 8, 5, 7, 9],
    [9, 3, 0, 9, 5, 3, 5, 7, 3],
    [0, 7, 8, 0, 1, 7, 1, 5, 7],
    [1, 5, 3, 3, 5, 7],
    [9, 7, 8, 9, 5, 7, 10, 1, 2],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2],
    [2, 10, 5, 2, 5, 3, 3, 5, 7],
    [7, 9, 5, 7, 8, 9, 3, 11, 2],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7],
    [11, 2, 1, 11, 1, 7, 7, 1, 5],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0],
    [11, 10, 5, 7, 11, 5],
    [10, 6, 5],
    [0, 8, 3, 5, 10, 6],
    [9, 0, 1, 5, 10, 6],
    [1, 8, 3, 1, 9, 8, 5, 10, 6],
    [1, 6, 5, 2, 6, 1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8],
    [9, 6, 5, 9, 0, 6, 0, 2, 6],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8],
    [2, 3, 11, 10, 6, 5],
    [11, 0, 8, 11, 2, 0, 10, 6, 5],
    [0, 1, 9, 2, 3, 11, 5, 10, 6],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11],
    [6, 3, 11, 6, 5, 3, 5, 1, 3],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9],
    [6, 5, 9, 6, 9, 11, 11, 9, 8],
    [5, 10, 6, 4, 7, 8],
    [4, 3, 0, 4, 7, 3, 6, 5, 10],
    [1, 9, 0, 5, 10, 6, 8, 4, 7],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4],
    [6, 1, 2, 6, 5, 1, 4, 7, 8],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9],
    [3, 11, 2, 7, 8, 4, 10, 6, 5],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9],
    [10, 4, 9, 6, 4, 10],
    [4, 10, 6, 4, 9, 10, 0, 8, 3],
    [10, 0, 1, 10, 6, 0, 6, 4, 0],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10],
    [1, 4, 9, 1, 2, 4, 2, 6, 4],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4],
    [0, 2, 4, 4, 2, 6],
    [8, 3, 2, 8, 2, 4, 4, 2, 6],
    [10, 4, 9, 10, 6, 4, 11, 2, 3],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4],
    [6, 4, 8, 11, 6, 8],
    [7, 10, 6, 7, 8, 10, 8, 9, 10],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0],
    [10, 6, 7, 10, 7, 1, 1, 7, 3],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9],
    [7, 8, 0, 7, 0, 6, 6, 0, 2],
    [7, 3, 2, 6, 7, 2],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6],
    [0, 9, 1, 11, 6, 7],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0],
    [7, 11, 6],
    [7, 6, 11],
    [3, 0, 8, 11, 7, 6],
    [0, 1, 9, 11, 7, 6],
    [8, 1, 9, 8, 3, 1, 11, 7, 6],
    [10, 1, 2, 6, 11, 7],
    [1, 2, 10, 3, 0, 8, 6, 11, 7],
    [2, 9, 0, 2, 10, 9, 6, 11, 7],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8],
    [7, 2, 3, 6, 2, 7],
    [7, 0, 8, 7, 6, 0, 6, 2, 0],
    [2, 7, 6, 2, 3, 7, 0, 1, 9],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6],
    [10, 7, 6, 10, 1, 7, 1, 3, 7],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7],
    [7, 6, 10, 7, 10, 8, 8, 10, 9],
    [6, 8, 4, 11, 8, 6],
    [3, 6, 11, 3, 0, 6, 0, 4, 6],
    [8, 6, 11, 8, 4, 6, 9, 0, 1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6],
    [6, 8, 4, 6, 11, 8, 2, 10, 1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3],
    [8, 2, 3, 8, 4, 2, 4, 6, 2],
    [0, 4, 2, 4, 6, 2],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8],
    [1, 9, 4, 1, 4, 2, 2, 4, 6],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3],
    [10, 9, 4, 6, 10, 4],
    [4, 9, 5, 7, 6, 11],
    [0, 8, 3, 4, 9, 5, 11, 7, 6],
    [5, 0, 1, 5, 4, 0, 7, 6, 11],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5],
    [9, 5, 4, 10, 1, 2, 7, 6, 11],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6],
    [7, 2, 3, 7, 6, 2, 5, 4, 9],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10],
    [6, 9, 5, 6, 11, 9, 11, 8, 9],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11],
    [6, 11, 3, 6, 3, 5, 5, 3, 1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2],
    [9, 5, 6, 9, 6, 0, 0, 6, 2],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8],
    [1, 5, 6, 2, 1, 6],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0],
    [0, 3, 8, 5, 6, 10],
    [10, 5, 6],
    [11, 5, 10, 7, 5, 11],
    [11, 5, 10, 11, 7, 5, 8, 3, 0],
    [5, 11, 7, 5, 10, 11, 1, 9, 0],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2],
    [2, 5, 10, 2, 3, 5, 3, 7, 5],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2],
    [1, 3, 5, 3, 7, 5],
    [0, 8, 7, 0, 7, 1, 1, 7, 5],
    [9, 0, 3, 9, 3, 5, 5, 3, 7],
    [9, 8, 7, 5, 9, 7],
    [5, 8, 4, 5, 10, 8, 10, 11, 8],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5],
    [9, 4, 5, 2, 11, 3],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4],
    [5, 10, 2, 5, 2, 4, 4, 2, 0],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2],
    [8, 4, 5, 8, 5, 3, 3, 5, 1],
    [0, 4, 5, 1, 0, 5],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5],
    [9, 4, 5],
    [4, 11, 7, 4, 9, 11, 9, 10, 11],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3],
    [11, 7, 4, 11, 4, 2, 2, 4, 0],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10],
    [1, 10, 2, 8, 7, 4],
    [4, 9, 1, 4, 1, 7, 7, 1, 3],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1],
    [4, 0, 3, 7, 4, 3],
    [4, 8, 7],
    [9, 10, 8, 10, 11, 8],
    [3, 0, 9, 3, 9, 11, 11, 9, 10],
    [0, 1, 10, 0, 10, 8, 8, 10, 11],
    [3, 1, 10, 11, 3, 10],
    [1, 2, 11, 1, 11, 9, 9, 11, 8],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9],
    [0, 2, 11, 8, 0, 11],
    [3, 2, 11],
    [2, 3, 8, 2, 8, 10, 10, 8, 9],
    [9, 10, 2, 0, 9, 2],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8],
    [1, 10, 2],
    [1, 3, 8, 9, 1, 8],
    [0, 9, 1],
    [0, 3, 8],
    []
  ];
  return raw.map((row) => Int8Array.from(row));
})();
var CORNER = [
  [0, 0, 0],
  [1, 0, 0],
  [1, 1, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [1, 1, 1],
  [0, 1, 1]
];
var EDGES = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 0],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 4],
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7]
];
function marchingCubes(mask, sx, sy, sz, ox, oy, oz) {
  const out = [];
  const vert = (i, j2, k2) => mask[(k2 * sy + j2) * sx + i] ? 1 : 0;
  const edgePos = new Float32Array(12 * 3);
  for (let k2 = 0; k2 < sz - 1; k2++) {
    for (let j2 = 0; j2 < sy - 1; j2++) {
      for (let i = 0; i < sx - 1; i++) {
        let cube = 0;
        for (let c = 0; c < 8; c++) {
          const [di2, dj, dk] = CORNER[c];
          if (vert(i + di2, j2 + dj, k2 + dk)) cube |= 1 << c;
        }
        const edges = EDGE_TABLE[cube];
        if (!edges) continue;
        for (let e = 0; e < 12; e++) {
          if (!(edges & 1 << e)) continue;
          const [a, b] = EDGES[e];
          const [ai2, aj, ak] = CORNER[a];
          const [bi2, bj, bk] = CORNER[b];
          edgePos[e * 3] = i + (ai2 + bi2) * 0.5;
          edgePos[e * 3 + 1] = j2 + (aj + bj) * 0.5;
          edgePos[e * 3 + 2] = k2 + (ak + bk) * 0.5;
        }
        const tri = TRI_TABLE[cube];
        for (let t = 0; t + 2 < tri.length; t += 3) {
          for (let v2 = 0; v2 < 3; v2++) {
            const e = tri[t + v2];
            out.push(edgePos[e * 3] + ox, edgePos[e * 3 + 1] + oy, edgePos[e * 3 + 2] + oz);
          }
        }
      }
    }
  }
  return out;
}

// render/ct-vr-presets.ts
var CT_VR_PRESETS = [
  {
    name: "CT-AAA",
    label: "CT AAA (angio)",
    shade: true,
    light: [0.1, 0.9, 0.2, 10],
    colorTF: [[-3024, 0, 0, 0], [143.556, 0.6157, 0.3569, 0.1843], [166.222, 0.8824, 0.6039, 0.2902], [214.389, 1, 1, 1], [419.736, 1, 0.937, 0.9545], [3071, 0.8275, 0.6588, 1]],
    opacityTF: [[-3024, 0], [143.556, 0], [166.222, 0.6863], [214.389, 0.6961], [419.736, 0.8333], [3071, 0.8039]]
  },
  {
    name: "CT-Bone",
    label: "CT Bone",
    shade: true,
    light: [0.1, 0.9, 0.2, 10],
    colorTF: [[-3024, 0, 0, 0], [-16.446, 0.7294, 0.2549, 0.302], [641.385, 0.9059, 0.8157, 0.5529], [3071, 1, 1, 1]],
    opacityTF: [[-3024, 0], [-16.446, 0], [641.385, 0.7157], [3071, 0.7059]]
  },
  {
    name: "CT-Bones",
    label: "CT Bones",
    shade: true,
    light: [0.2, 1, 0, 1],
    colorTF: [[-1e3, 0.3, 0.3, 1], [-488, 0.3, 1, 0.3], [463.28, 1, 0, 0], [659.15, 1, 0.9125, 0.0375], [953, 1, 0.3, 0.3]],
    opacityTF: [[-1e3, 0], [152.19, 0], [278.93, 0.1905], [952, 0.2]]
  },
  {
    name: "CT-Cardiac",
    label: "CT Cardiac",
    shade: true,
    light: [0.1, 0.9, 0.2, 10],
    colorTF: [[-3024, 0, 0, 0], [-77.688, 0.549, 0.251, 0.149], [94.952, 0.8824, 0.6039, 0.2902], [179.052, 1, 0.937, 0.9545], [260.439, 0.6157, 0, 0], [3071, 0.8275, 0.6588, 1]],
    opacityTF: [[-3024, 0], [-77.688, 0], [94.952, 0.2857], [179.052, 0.5536], [260.439, 0.8482], [3071, 0.875]]
  },
  {
    name: "CT-Chest-Contrast-Enhanced",
    label: "CT Chest (contrast)",
    shade: true,
    light: [0.1, 0.9, 0.2, 10],
    colorTF: [[-3024, 0, 0, 0], [67.011, 0.549, 0.251, 0.149], [251.105, 0.8824, 0.6039, 0.2902], [439.291, 1, 0.937, 0.9545], [3071, 0.8275, 0.6588, 1]],
    opacityTF: [[-3024, 0], [67.011, 0], [251.105, 0.4464], [439.291, 0.625], [3071, 0.6161]]
  },
  {
    name: "CT-Coronary-Arteries",
    label: "CT Coronary",
    shade: false,
    light: [0.2, 1, 0, 1],
    colorTF: [[-2048, 0, 0, 0], [136.47, 0, 0, 0], [159.215, 0.1598, 0.1598, 0.1598], [318.43, 0.7647, 0.7647, 0.7647], [478.693, 1, 1, 1], [3661, 1, 1, 1]],
    opacityTF: [[-2048, 0], [136.47, 0], [159.215, 0.2589], [318.43, 0.5714], [478.693, 0.7768], [3661, 1]]
  },
  {
    name: "CT-Fat",
    label: "CT Fat",
    shade: false,
    light: [0.2, 1, 0, 1],
    colorTF: [[-1e3, 0.3, 0.3, 1], [-497.5, 0.3, 1, 0.3], [-99, 0, 0, 1], [-76.946, 0, 1, 0], [-65.481, 0.8354, 0.8889, 0.0165], [83.89, 1, 0, 0], [463.28, 1, 0, 0], [659.15, 1, 0.9125, 0.0375], [2952, 1, 0.3003, 0.2999]],
    opacityTF: [[-1e3, 0], [-100, 0], [-99, 0.15], [-60, 0.15], [-59, 0], [101.2, 0], [952, 0]]
  },
  {
    name: "CT-Lung",
    label: "CT Lung",
    shade: true,
    light: [0.2, 1, 0, 1],
    colorTF: [[-1e3, 0.3, 0.3, 1], [-600, 0, 0, 1], [-530, 0.1347, 0.7817, 0.0725], [-460, 0.9292, 1, 0.1095], [-400, 0.8889, 0.2549, 0.024], [2952, 1, 0.3, 0.3]],
    opacityTF: [[-1e3, 0], [-600, 0], [-599, 0.15], [-400, 0.15], [-399, 0], [2952, 0]]
  },
  {
    name: "CT-MIP",
    label: "CT MIP",
    shade: true,
    light: [0.1, 0.9, 0.2, 10],
    colorTF: [[-3024, 0, 0, 0], [-637.62, 1, 1, 1], [700, 1, 1, 1], [3071, 1, 1, 1]],
    opacityTF: [[-3024, 0], [-637.62, 0], [700, 1], [3071, 1]]
  },
  {
    name: "CT-Muscle",
    label: "CT Muscle",
    shade: true,
    light: [0.1, 0.9, 0.2, 10],
    colorTF: [[-3024, 0, 0, 0], [-155.407, 0.549, 0.251, 0.149], [217.641, 0.8824, 0.6039, 0.2902], [419.736, 1, 0.937, 0.9545], [3071, 0.8275, 0.6588, 1]],
    opacityTF: [[-3024, 0], [-155.407, 0], [217.641, 0.6765], [419.736, 0.8333], [3071, 0.8039]]
  },
  {
    name: "CT-Pulmonary-Arteries",
    label: "CT Pulmonary",
    shade: true,
    light: [0.2, 1, 0, 1],
    colorTF: [[-2048, 0, 0, 0], [-568.625, 0, 0, 0], [-364.081, 0.3961, 0.302, 0.1804], [-244.813, 0.6118, 0.3529, 0.0706], [18.277, 0.8431, 0.0157, 0.1569], [447.798, 0.7529, 0.7529, 0.7529], [3592.73, 1, 1, 1]],
    opacityTF: [[-2048, 0], [-568.625, 0], [-364.081, 0.0714], [-244.813, 0.4018], [18.277, 0.6071], [447.798, 0.8304], [3592.73, 0.8393]]
  },
  {
    name: "CT-Soft-Tissue",
    label: "CT Soft Tissue",
    shade: false,
    light: [0.2, 1, 0, 1],
    colorTF: [[-2048, 0, 0, 0], [-167.01, 0, 0, 0], [-160, 0.0556, 0.0556, 0.0556], [240, 1, 1, 1], [3661, 1, 1, 1]],
    opacityTF: [[-2048, 0], [-167.01, 0], [-160, 1], [240, 1], [3661, 1]]
  }
];
function presetLUT(p) {
  const lo = p.colorTF[0][0];
  const hi2 = p.colorTF[p.colorTF.length - 1][0];
  const clim = [lo, hi2];
  const lut = lutFromTransferFunctions(p.colorTF, p.opacityTF, clim);
  const shade = p.shade ? [p.light[0], p.light[1], p.light[2], p.light[3]] : [1, 0, 0, 1];
  return { lut, clim, shade };
}

// render/demos/vr-preset-menu.ts
function openVrPresetMenu(opts) {
  if (document.getElementById("vr-preset-menu")) return;
  const overlay = document.createElement("div");
  overlay.id = "vr-preset-menu";
  overlay.style.cssText = "position:fixed;inset:0;z-index:2100;background:rgba(5,6,10,.82);display:flex;align-items:center;justify-content:center;";
  const card = document.createElement("div");
  card.style.cssText = "background:#11141d;border:1px solid #33507e;border-radius:12px;padding:18px 20px;max-width:min(720px,94vw);max-height:88vh;overflow-y:auto;font:13px -apple-system,system-ui,sans-serif;color:#d6e2f2;";
  card.innerHTML = `<div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:12px;"><h3 style="margin:0;color:#fff;font-size:16px;">Volume rendering preset</h3><span style="color:#5a6b85;font-size:11px;">Slicer CT presets \xB7 thumbnails rendered from this study</span></div>`;
  const grid = document.createElement("div");
  grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px;";
  card.appendChild(grid);
  const close = () => overlay.remove();
  for (const it2 of opts.items) {
    const cell = document.createElement("button");
    const sel = it2.name === opts.current;
    cell.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:6px;padding:8px 6px 7px;cursor:pointer;background:" + (sel ? "rgba(84,198,240,.18)" : "rgba(255,255,255,.04)") + ";border:1px solid " + (sel ? "#54c6f0" : "rgba(255,255,255,.12)") + ";border-radius:10px;color:#eaf0ff;";
    it2.canvas.style.cssText = "width:110px;height:110px;border-radius:7px;background:#05060a;display:block;";
    const lab = document.createElement("div");
    lab.textContent = it2.label;
    lab.style.cssText = "font:600 11.5px -apple-system,system-ui,sans-serif;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:110px;" + (sel ? "color:#9fe9ff;" : "");
    cell.appendChild(it2.canvas);
    cell.appendChild(lab);
    cell.onclick = () => {
      opts.onPick(it2.name);
      close();
    };
    grid.appendChild(cell);
  }
  overlay.appendChild(card);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });
  const esc = (e) => {
    if (e.key === "Escape") {
      close();
      document.removeEventListener("keydown", esc, true);
    }
  };
  document.addEventListener("keydown", esc, true);
  document.body.appendChild(overlay);
}

// ../../cast-interface/cast-js-client/dist/cast-client.js
function hn(e, t) {
  t.classHierarchy || (t.classHierarchy = []), e.getClassName = () => t.classHierarchy[t.classHierarchy.length - 1] || "CastClient", e.isA = (n) => t.classHierarchy.includes(n), typeof e.delete != "function" && (e.delete = () => {
  });
}
function pn(e, t, n) {
  n.forEach((r) => {
    const i = `get${r.charAt(0).toUpperCase()}${r.slice(1)}`;
    e[i] = () => t[r];
  });
}
function mn(e, t) {
  return () => {
    e(), typeof t == "function" && t();
  };
}
function yn(e, t) {
  return function(r = {}) {
    const i = {};
    return e(i, { classHierarchy: [t] }, r), i;
  };
}
var Fe = "CAST";
var ut = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
function Tt(e, t = Fe) {
  return String(e || t).trim().replace(/[^a-zA-Z0-9]+/g, "-").replace(/^-+|-+$/g, "") || t;
}
function gn(e) {
  return `${Tt(e)}-`;
}
function ie(e) {
  return typeof crypto < "u" && crypto.randomUUID ? e + crypto.randomUUID().replace(/-/g, "").slice(0, 16) : e + Math.random().toString(36).substring(2, 18);
}
function De(e = Fe) {
  const t = Tt(e);
  let n = "";
  for (let r = 0; r < 6; r++) {
    const i = Math.floor(Math.random() * ut.length);
    n += ut[i];
  }
  return `${t}-${n}`;
}
var lt = ["dicom", "nifti", "jpg", "png", "nrrd"];
function Me(e) {
  if (typeof e != "string") return false;
  const t = e.trim().toLowerCase();
  if (!t) return false;
  for (let n = 0; n < lt.length; n++) {
    const r = lt[n];
    if (t === r || t.startsWith(`${r}-`) || t.startsWith(`${r}_`))
      return true;
  }
  return false;
}
function Sn(e) {
  return typeof ArrayBuffer < "u" && ArrayBuffer.isView(e);
}
async function ae(e) {
  if (e instanceof ArrayBuffer)
    return e;
  if (Sn(e)) {
    const { buffer: t, byteOffset: n, byteLength: r } = e;
    return t.slice(n, n + r);
  }
  if (typeof Blob < "u" && e instanceof Blob && typeof e.arrayBuffer == "function")
    return e.arrayBuffer();
  throw new Error(
    "CastClient: binary resource.data must be ArrayBuffer, TypedArray, DataView, Blob, or File"
  );
}
function j(e) {
  const t = e && e.context;
  if (!t || typeof t != "object" || Array.isArray(t))
    return [];
  const n = t.files;
  return Array.isArray(n) ? n.filter((r) => r && typeof r == "object") : [];
}
function je(e) {
  if (!e || typeof e != "object" || e.data != null)
    return { payloadIds: [], chunkByteLengths: [] };
  const t = e.payloadIds;
  if (Array.isArray(t) && t.length) {
    const r = t.map((s) => typeof s == "string" ? s.trim() : "").filter(Boolean), i = Array.isArray(e.chunkByteLengths) ? e.chunkByteLengths.filter((s) => typeof s == "number" && s >= 0) : [];
    if (i.length === r.length)
      return { payloadIds: r, chunkByteLengths: i };
    if (r.length === 1) {
      const s = typeof e.byteLength == "number" && e.byteLength >= 0 ? e.byteLength : null;
      return {
        payloadIds: r,
        chunkByteLengths: s != null ? [s] : []
      };
    }
  }
  const n = typeof e.payloadId == "string" ? e.payloadId.trim() : "";
  if (n) {
    const r = typeof e.byteLength == "number" && e.byteLength >= 0 ? e.byteLength : null;
    return {
      payloadIds: [n],
      chunkByteLengths: r != null ? [r] : []
    };
  }
  return { payloadIds: [], chunkByteLengths: [] };
}
function Cn(e) {
  const t = j(e);
  for (let n = 0; n < t.length; n++) {
    const r = t[n];
    if (r.data == null) {
      const i = je(r);
      if (i.payloadIds.length)
        return {
          kind: "files",
          index: n,
          chunkIndex: 0,
          payloadId: i.payloadIds[0],
          expectedChunkBytes: i.chunkByteLengths[0] ?? null
        };
    }
  }
  return null;
}
function Nt(e) {
  return Cn(e);
}
function Ie(e) {
  return !!Nt(e);
}
function _e(e, t) {
  return (typeof e == "string" ? e.trim().toLowerCase() : "").startsWith("nifti") ? t === 0 ? "nifti-send.nii.gz" : `nifti-send-${t + 1}.nii.gz` : t === 0 ? "dicom-send.dcm" : `dicom-send-${t + 1}.dcm`;
}
function ke(e) {
  return (typeof e == "string" ? e.trim().toLowerCase() : "").startsWith("nifti") ? "application/octet-stream" : "application/dicom";
}
function Ae(e) {
  const t = e && e.event;
  if (!t || !Me(t["hub.event"]) || j(t).length)
    return e;
  const n = t["hub.event"], r = t.context;
  let i = [];
  Array.isArray(r) ? i = r : r != null && (i = [r]);
  const s = [];
  for (let c = 0; c < i.length; c++) {
    const a = i[c];
    if (a && typeof a == "object") {
      const d = a.resource;
      if (d && typeof d == "object" && d.data != null) {
        const f = { ...d };
        (typeof f.fileName != "string" || !f.fileName.trim()) && (f.fileName = _e(n, c)), (typeof f.mimeType != "string" || !f.mimeType.trim()) && (f.mimeType = ke(n)), s.push(f);
      }
    }
  }
  return s.length ? {
    ...e,
    event: {
      ...t,
      context: {
        files: s
      }
    }
  } : e;
}
function wn(e, t) {
  const n = e && e.event, r = n && n.context;
  let i = [];
  Array.isArray(r) ? i = r : r != null && (i = [r]);
  for (let s = 0; s < i.length; s++) {
    const c = i[s];
    if (c && typeof c == "object") {
      const a = c.resource;
      if (a && typeof a == "object") {
        const d = typeof a.fileName == "string" ? a.fileName.trim() : "";
        if (d)
          return d;
      }
    }
  }
  return t;
}
function ue(e) {
  if (!e || typeof e != "object")
    return false;
  const t = typeof e.url == "string" ? e.url.trim() : "";
  if (!t)
    return false;
  const n = t.toLowerCase();
  return n.startsWith("http://") || n.startsWith("https://");
}
function En(e) {
  const t = [];
  for (let n = 0; n < e.length; n += 1)
    ue(e[n]) || t.push(n);
  return t;
}
function Tn(e) {
  const t = e && e.event;
  if (!t || !Me(t["hub.event"]))
    return false;
  const n = j(t);
  return n.length ? n.some(
    (r) => r.data != null && !ue(r)
  ) : false;
}
async function Nn(e, t = "", n = 0) {
  if (!e || typeof e != "object")
    throw new Error(
      "CastClient: binary batch publish files[] entries must be objects"
    );
  const r = ue(e);
  let i = typeof e.byteLength == "number" && e.byteLength >= 0 ? e.byteLength : null;
  if ("data" in e && e.data != null) {
    if (typeof e.data == "string")
      throw new Error(
        "CastClient: binary batch publish string payloads are not supported; pass binary input instead"
      );
    i = (await ae(e.data)).byteLength;
  }
  if (i == null && r) {
    const c = { ...e };
    return delete c.data, delete c.binaryTransfer, delete c.payloadId, delete c.payloadIds, delete c.chunkByteLengths, delete c.expiresAt, (typeof c.fileName != "string" || !c.fileName.trim()) && (c.fileName = _e(t, n)), (typeof c.mimeType != "string" || !c.mimeType.trim()) && (c.mimeType = ke(t)), c;
  }
  if (i == null)
    throw new Error(
      "CastClient: binary batch publish requires files[].data or files[].byteLength"
    );
  const s = { ...e };
  return delete s.data, delete s.binaryTransfer, r || delete s.url, delete s.payloadId, delete s.payloadIds, delete s.chunkByteLengths, delete s.expiresAt, s.byteLength = i, (typeof s.fileName != "string" || !s.fileName.trim()) && (s.fileName = _e(t, n)), (typeof s.mimeType != "string" || !s.mimeType.trim()) && (s.mimeType = ke(t)), s;
}
async function An(e) {
  if (!e.event || typeof e.event != "object")
    throw new Error("CastClient: binary batch publish requires event object");
  const t = e.event;
  if (!Me(t["hub.event"]))
    return e;
  const n = t.context;
  if (!n || typeof n != "object" || Array.isArray(n))
    throw new Error(
      "CastClient: binary batch publish requires event.context object"
    );
  const r = n.files;
  if (!Array.isArray(r) || !r.length)
    throw new Error(
      "CastClient: binary batch publish requires non-empty event.context.files[]"
    );
  const i = t["hub.event"], s = await Promise.all(
    r.map(
      (c, a) => Nn(c, i, a)
    )
  );
  return {
    ...e,
    event: {
      ...t,
      context: {
        ...n,
        files: s
      }
    }
  };
}
async function ft(e) {
  const t = e && e.event;
  if (!t)
    return [];
  const r = j(t).filter(
    (i) => i && typeof i == "object" && i.data != null && !ue(i)
  );
  return Promise.all(r.map((i) => ae(i.data)));
}
function vn(e, t, n) {
  const r = new TextEncoder(), i = r.encode(`\r
`), s = [], c = (a) => {
    s.push(r.encode(a));
  };
  c(
    `--${e}\r
Content-Type: application/dicom+json\r
\r
${t}\r
`
  );
  for (let a = 0; a < n.length; a++) {
    const d = n[a], f = d && d.mimeType && String(d.mimeType).trim() || "application/octet-stream";
    c(`--${e}\r
Content-Type: ${f}\r
\r
`), s.push(new Uint8Array(d.buffer)), s.push(i);
  }
  return c(`--${e}--\r
`), new Blob(s);
}
async function Ln({
  msg: e,
  fileBytesList: t,
  hub: n,
  messageIdPrefix: r
}) {
  const i = await An(e), s = await Promise.all(
    (t || []).map((b) => ae(b))
  ), c = j(i.event), a = En(c);
  if (s.length !== a.length)
    throw new Error(
      `CastClient: binary batch publish expected ${a.length} file part(s), got ${s.length}`
    );
  const d = s.map((b, u) => {
    const l = a[u];
    return {
      buffer: b,
      mimeType: c[l] && c[l].mimeType || "application/octet-stream"
    };
  }), f = `cast-batch-${ie(r())}`, o = vn(
    f,
    JSON.stringify(i),
    d
  );
  try {
    return await fetch(n.hub_endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${n.token}`,
        "Content-Type": `multipart/related; boundary="${f}"; type="application/dicom"`
      },
      body: o
    });
  } catch (b) {
    const u = b instanceof Error ? b.message : String(b);
    return console.debug("CastClient:", u), null;
  }
}
function le(e) {
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
  ), r = Array.isArray(n.responses) ? n.responses.filter((c) => c && typeof c == "object") : [], i = Array.isArray(n.expected) ? n.expected.map((c) => String(c)) : [], s = Array.isArray(n.missing) ? n.missing.map((c) => String(c)) : [];
  return {
    responses: r,
    expected: i,
    missing: s,
    timedOut: !!n.timedOut,
    ok: !!n.ok,
    id: n.id != null ? String(n.id) : null,
    actor: n.actor != null ? String(n.actor) : null,
    productName: n.productName != null ? String(n.productName) : null
  };
}
function In(e) {
  return le(e).responses;
}
function At(e, t) {
  if (t === "file:")
    return true;
  const n = String(e || "").toLowerCase();
  return !!(n === "localhost" || n === "127.0.0.1" || n === "[::1]" || /^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(n) || /^192\.168\.\d{1,3}\.\d{1,3}$/.test(n) || /^172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}$/.test(n));
}
function vt(e = typeof window < "u" ? window.location : null) {
  return e ? !At(e.hostname, e.protocol) : false;
}
function Lt(e) {
  try {
    const t = new URL(e);
    return !At(t.hostname, t.protocol);
  } catch {
    return false;
  }
}
function _n(e, t, n = vt()) {
  const r = t.find((s) => {
    var a;
    const c = (a = e[s]) == null ? void 0 : a.hubEndpoint;
    return c && Lt(c) === n;
  });
  if (r)
    return r;
  const i = n ? "cloud" : "local";
  return e[i] ? i : t[0];
}
var G = "-request";
var W = "-response";
function qe(e) {
  return typeof e != "string" ? "" : e.trim().toLowerCase();
}
function ze(e) {
  const t = qe(e);
  return t ? `${t}${G}` : "";
}
function It(e) {
  const t = qe(e);
  return t ? `${t}${W}` : "";
}
function kn(e) {
  return typeof e != "string" ? false : e.endsWith(G);
}
function Un(e) {
  return typeof e != "string" ? false : e.endsWith(W);
}
function $n(e) {
  return typeof e != "string" ? "" : e.endsWith(G) ? e.slice(0, -G.length) : e.endsWith(W) ? e.slice(0, -W.length) : "";
}
function Rn(e) {
  if (!(e != null && e.length))
    return ["*"];
  if (e.some((i) => String(i).trim() === "*"))
    return ["*"];
  const t = ze("STATUS"), n = new Set(
    e.map((i) => String(i).trim().toLowerCase()).filter(Boolean)
  ), r = [];
  return t && !n.has(t.toLowerCase()) && (r.push(t), n.add(t.toLowerCase())), n.has("status-update") || r.push("status-update"), r.length ? [...e, ...r] : [...e];
}
var _t = "cast-radio-icon";
var kt = `<svg class="${_t}" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9"/><path d="M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5"/><circle cx="12" cy="12" r="2"/><path d="M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5"/><path d="M19.1 4.9C23 8.8 23 15.1 19.1 19"/></svg>`;
var Ut = "cast-radio-status-icon";
var $t = "cast-radio-slash";
function On(e = {}) {
  const t = e.slashId || "castRadioSlash";
  return `<span class="${Ut}">${kt}<span id="${t}" class="${$t}" aria-hidden="true" hidden></span></span>`;
}
function xn(e, t = {}) {
  const n = !!t.conferenceActive;
  return e === "connected" ? {
    tone: "connected",
    showSlash: false,
    pulse: false,
    conferencePulse: n
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
function Pn() {
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
function Bn() {
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
function se(e) {
  if (e == null)
    return;
  const t = String(e).trim();
  if (!(!t || t === "*"))
    return t;
}
function Rt(e) {
  if (e == null)
    return;
  const t = String(e).trim();
  if (!(!t || t === "*"))
    return t;
}
function Fn() {
  return {
    token: "",
    lastIdToken: "",
    lastPublishedMessageID: "",
    subscribed: false,
    resubscribeRequested: false,
    websocket: null
  };
}
function Dn() {
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
function dt(e) {
  return e || (typeof window < "u" ? `${window.location.origin}/castCallback` : "");
}
function bt(e, t, n, r = "") {
  const i = new URLSearchParams();
  i.append("hub.mode", e), i.append("hub.channel.type", "websocket"), i.append("hub.callback", n), i.append("hub.events", (t.events || []).toString()), i.append("hub.topic", t.topic || ""), i.append("hub.lease", String(t.lease || 999)), i.append("subscriber.name", t.subscriberName || ""), i.append("subscriber.product.name", t.productName || ""), i.append(
    "subscriber.product.version",
    t.productVersion || r || ""
  );
  const s = (t.actors || []).map((a) => a.trim()).filter(Boolean);
  s.length && i.append("subscriber.actors", JSON.stringify(s));
  const c = Dn();
  return c && i.append("subscriber.client_info", JSON.stringify(c)), i;
}
var ve = 25;
function Mn(e) {
  try {
    const t = new URL(e);
    return t.hostname.toLowerCase() === "localhost" && (t.hostname = "127.0.0.1"), t.toString();
  } catch {
    return e;
  }
}
function jn(e) {
  if (!(e instanceof Error))
    return false;
  const t = e.name || "";
  return t === "TypeError" || t === "NetworkError";
}
function qn(e, t) {
  if (!e)
    return "";
  try {
    const n = `/api/hub/payloads/${encodeURIComponent(e)}`;
    return Mn(new URL(n, t).toString());
  } catch (n) {
    return console.warn("CastClient: invalid payloadId url", e, n), "";
  }
}
function zn(e) {
  delete e.binaryTransfer, delete e.url, delete e.payloadId, delete e.payloadIds, delete e.chunkByteLengths, delete e.expiresAt;
}
function Hn(e, t) {
  const n = e.reduce((s, c) => s + c.byteLength, 0);
  if (typeof t == "number" && t >= 0 && n !== t)
    throw new Error(
      `CastClient: payload size mismatch expected=${t} received=${n}`
    );
  const r = new Uint8Array(n);
  let i = 0;
  for (let s = 0; s < e.length; s++)
    r.set(new Uint8Array(e[s]), i), i += e[s].byteLength;
  return r.buffer;
}
function ht(e, t, n) {
  const r = e && e.event, i = r && r.context, s = i && i.files, c = s && s[t];
  return !c || typeof c != "object" ? false : (zn(c), c.data = n, c.byteLength = n.byteLength, true);
}
async function Gn(e, t, { hubEndpoint: n, accessToken: r }) {
  const i = qn(e, n);
  if (!i)
    throw new Error("CastClient.fetchPayload: missing or invalid payloadId");
  const s = {};
  r && (s.Authorization = `Bearer ${r}`);
  async function c() {
    const a = await fetch(i, {
      method: "GET",
      credentials: "omit",
      headers: s
    });
    if (!a.ok)
      throw new Error(
        `CastClient.fetchPayload: GET ${a.status} ${a.statusText}`
      );
    const d = await a.arrayBuffer();
    if (typeof t == "number" && t >= 0 && d.byteLength !== t)
      throw new Error(
        `CastClient.fetchPayload: chunk size mismatch payloadId=${e.slice(
          0,
          8
        )} expected=${t} received=${d.byteLength}`
      );
    return d;
  }
  try {
    return await c();
  } catch (a) {
    if (jn(a))
      return c();
    throw a;
  }
}
async function pt(e, t) {
  const n = je(e);
  if (!n.payloadIds.length)
    throw new Error("CastClient.fetchPayload: no payloadIds on file entry");
  const r = [];
  for (let i = 0; i < n.payloadIds.length; i += ve) {
    const s = n.payloadIds.slice(
      i,
      i + ve
    ), c = n.chunkByteLengths.slice(
      i,
      i + ve
    ), a = await Promise.all(
      s.map(
        (d, f) => Gn(d, c[f] ?? null, t)
      )
    );
    r.push(...a);
  }
  return Hn(
    r,
    typeof e.byteLength == "number" ? e.byteLength : null
  );
}
function mt(e) {
  return JSON.parse(JSON.stringify(e));
}
function Wn(e) {
  const t = () => e();
  return {
    hasPendingPayload(n) {
      const r = n && n.event;
      return !!(r && Ie(r));
    },
    async fetchPayload(n) {
      const r = n && n.event;
      if (!r || !Ie(r))
        return n;
      const i = Nt(r);
      if (!i)
        return n;
      const s = typeof performance < "u" && performance.now ? performance.now() : Date.now(), c = r.context.files[i.index], a = await pt(c, t()), d = mt(n);
      if (!ht(d, i.index, a))
        throw new Error("CastClient.fetchPayload: no payload slot on message");
      const o = (typeof performance < "u" && performance.now ? performance.now() : Date.now()) - s;
      return console.debug(
        `CastClient: payload fetched fileIndex=${i.index} bytes=${a.byteLength} elapsed=${(o / 1e3).toFixed(2)}s`
      ), d;
    },
    async fetchAllPayloads(n) {
      const r = n && n.event, i = r ? j(r) : [], s = [];
      for (let a = 0; a < i.length; a++)
        je(i[a]).payloadIds.length && i[a].data == null && s.push(a);
      if (!s.length)
        return n;
      const c = mt(n);
      for (let a = 0; a < s.length; a++) {
        const d = s[a], f = c.event.context.files[d], o = await pt(f, t());
        if (!ht(c, d, o))
          throw new Error("CastClient.fetchAllPayloads: attach failed");
      }
      return c;
    }
  };
}
function Z(e) {
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
function Vn(e) {
  const t = Z(e);
  if (!t)
    return "";
  const n = t.href.endsWith("/") ? t.href : `${t.href}/`;
  return new URL("admin", n).href;
}
function Jn(e, t = {}) {
  var d, f, o, b;
  const n = Z(e);
  if (!n)
    return "";
  const r = new URL("/api/hub/conference-client", n.origin), i = (d = t.subscriberName) == null ? void 0 : d.trim(), s = (f = t.topic) == null ? void 0 : f.trim(), c = ((o = t.theme) == null ? void 0 : o.trim()) || "volview", a = ((b = t.mode) == null ? void 0 : b.trim().toLowerCase()) === "light" ? "light" : "dark";
  return i && r.searchParams.set("subscriberName", i), s && r.searchParams.set("topic", s), r.searchParams.set("theme", c), r.searchParams.set("mode", a), r.href;
}
var Kn = { width: 336, height: 288 };
function Yn(e, t, n = { width: 800, height: 600 }) {
  if (!e || typeof window > "u")
    return;
  const r = n.width, i = n.height, s = Math.max(0, Math.floor((window.screen.width - r) / 2)), c = Math.max(0, Math.floor((window.screen.height - i) / 2)), a = [
    "popup",
    `width=${r}`,
    `height=${i}`,
    `left=${s}`,
    `top=${c}`,
    "noopener",
    "noreferrer"
  ].join(",");
  window.open(e, t, a);
}
var Xn = 3e4;
var Ot = 1200;
var xt = [
  "Test conference",
  "US annotations",
  "Tumor Board",
  "Case discussion",
  "Pedicle screw"
];
function He(e, t) {
  const n = Z(e);
  return n ? new URL(t, n.origin).href : null;
}
function Ge(e) {
  if (!Array.isArray(e))
    return [];
  const t = /* @__PURE__ */ new Set();
  return e.map((n) => String(n ?? "").trim()).filter((n) => !n || t.has(n) ? false : (t.add(n), true));
}
function Q(e) {
  return String((e == null ? void 0 : e.hostTopic) ?? (e == null ? void 0 : e.user) ?? "").trim();
}
function We(e, t) {
  const n = Q(t), r = String(e ?? "").trim();
  return !r || !n ? false : r === n || r.toLowerCase() === n.toLowerCase();
}
function Ve(e, t, n) {
  const r = String(e ?? "").trim(), i = String(t ?? "").trim(), s = Q(n), c = Array.isArray(n == null ? void 0 : n.topics) ? n.topics.map((a) => String(a).trim()).filter(Boolean) : [];
  return !!(r && (r === s || c.includes(r)) || i && i === s);
}
function V(e, t, n) {
  return n.find(
    (r) => Ve(e, t, r)
  ) || null;
}
async function Pt(e) {
  const t = He(e, "/api/hub/conference-topics");
  if (!t)
    return [];
  try {
    const n = await fetch(t);
    if (!n.ok)
      return [];
    const r = await n.json();
    return Array.isArray(r) ? r.map((i) => String(i).trim()).filter((i) => i && i !== "*") : [];
  } catch {
    return [];
  }
}
async function J(e) {
  const t = Z(e);
  if (!t)
    return [];
  const n = new URL("/api/hub/conference", t.origin).href;
  try {
    const r = await fetch(n);
    if (!r.ok)
      return [];
    const i = await r.json();
    return Array.isArray(i) ? i : [];
  } catch {
    return [];
  }
}
async function Bt(e, t, n, r, i) {
  const s = He(e, "/api/hub/conference");
  if (!s)
    throw new Error("Invalid hub endpoint");
  const c = {
    hostTopic: String(t).trim(),
    title: String(n).trim(),
    topics: r
  }, a = String(i ?? "").trim();
  a && (c.hostUserName = a);
  const d = await fetch(s, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(c)
  });
  if (!d.ok) {
    const f = await d.text();
    throw new Error(f || `HTTP ${d.status}`);
  }
}
async function Je(e, t, n) {
  const r = He(e, "/api/hub/conference");
  if (!r)
    throw new Error("Invalid hub endpoint");
  const i = { hostTopic: String(t).trim() }, s = n == null ? void 0 : n.trim();
  s && (i.leaveTopic = s);
  const c = await fetch(r, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(i)
  });
  if (!c.ok) {
    const a = await c.text();
    throw new Error(a || `HTTP ${c.status}`);
  }
}
async function Zn(e, t, n) {
  const r = await J(e), i = V(t, n, r);
  return {
    active: !!i,
    title: String((i == null ? void 0 : i.title) ?? "").trim(),
    participants: Ge(i == null ? void 0 : i.participants)
  };
}
function Ke(e, t) {
  if (!t)
    return null;
  const n = String((e == null ? void 0 : e.topic) ?? "").trim(), r = String((e == null ? void 0 : e.subscriberName) ?? "").trim(), i = String((e == null ? void 0 : e.userName) ?? "").trim(), s = Q(t);
  if (!s || !Ve(n, r, t))
    return null;
  const c = Array.isArray(t.topics) ? t.topics.map((l) => String(l).trim()).filter(Boolean) : [], a = Ge(t.participants), d = [], f = /* @__PURE__ */ new Set();
  for (const l of [n, s, ...c])
    !l || f.has(l) || (f.add(l), d.push(l));
  const o = We(n, t) ? "leading" : "following", b = s, u = d.map((l) => {
    const p = !!n && (l === n || l.toLowerCase() === n.toLowerCase()), h = l === b || l.toLowerCase() === b.toLowerCase(), g = h && o === "following";
    let C = "";
    return p ? C = o === "leading" ? "You \xB7 leading" : "You" : g ? C = "Following" : o === "leading" && (C = "Following you"), {
      placeId: l,
      label: p && i ? i : l,
      isSelf: p,
      isLeading: h,
      isFollowing: g,
      statusLabel: C
    };
  });
  return {
    title: String(t.title ?? "").trim(),
    places: u,
    selfPlaceId: n,
    leadingPlaceId: b,
    selfRole: o,
    hostTopic: s,
    attendeeTopics: c,
    participants: a
  };
}
async function Ft(e, t, n, r) {
  const i = await J(e), s = V(t, n, i);
  return Ke({ topic: t, subscriberName: n, userName: r }, s);
}
function S(e, t) {
  return `${e}-${t}`;
}
function B(e) {
  return String(e ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function Qn(e) {
  const t = typeof (e == null ? void 0 : e.getSession) == "function" ? e.getSession : () => ({}), n = typeof (e == null ? void 0 : e.onConferenceChange) == "function" ? e.onConferenceChange : null, r = String((e == null ? void 0 : e.classPrefix) || "cast-conference").trim(), i = (e == null ? void 0 : e.root) || (typeof document < "u" ? document.body : null);
  if (!i)
    throw new Error("createCastConferenceDialog requires a DOM root");
  let s = null, c = false;
  const a = document.createElement("div");
  a.className = S(r, "overlay"), a.hidden = true, a.innerHTML = `
    <div class="${S(r, "dialog")}" role="dialog" aria-modal="true" aria-labelledby="${S(r, "title")}" tabindex="-1">
      <div class="${S(r, "header")}">
        <h2 id="${S(r, "title")}" class="${S(r, "title")}">Conferencing</h2>
        <button type="button" class="${S(r, "close")}" data-cast-conference-close aria-label="Close">\xD7</button>
      </div>
      <div class="${S(r, "body")}">
        <p class="${S(r, "status")}" data-cast-conference-status role="alert" hidden></p>
        <div data-cast-conference-create>
          <div class="${S(r, "field")}">
            <label class="${S(r, "label")}" for="${S(r, "title-select")}">Conference title</label>
            <select id="${S(r, "title-select")}" class="${S(r, "select")}" data-cast-conference-title-select>
              <option value="">Select a conference title</option>
              ${xt.map(
    (m) => `<option value="${B(m)}"${m === "Test conference" ? " selected" : ""}>${B(m)}</option>`
  ).join("")}
              <option value="other">Other\u2026</option>
            </select>
          </div>
          <div class="${S(r, "field")}" data-cast-conference-custom-group hidden>
            <label class="${S(r, "label")}" for="${S(r, "custom-title")}">Custom title</label>
            <input id="${S(r, "custom-title")}" class="${S(r, "input")}" type="text" placeholder="Enter conference title" data-cast-conference-custom-title />
          </div>
          <div class="${S(r, "field")}">
            <span class="${S(r, "label")}">Users</span>
            <div class="${S(r, "topics")}" data-cast-conference-topics>
              <p class="${S(r, "muted")}">Loading users\u2026</p>
            </div>
          </div>
          <button type="button" class="${S(r, "primary")}" data-cast-conference-create-btn>Create conference</button>
        </div>
        <div data-cast-conference-manage hidden>
          <div class="${S(r, "label")}">Manage conference</div>
          <div class="${S(r, "info")}" data-cast-conference-info></div>
          <button type="button" class="${S(r, "primary")}" data-cast-conference-exit-btn>Leave conference</button>
        </div>
      </div>
      <div class="${S(r, "actions")}">
        <button type="button" class="${S(r, "secondary")}" data-cast-conference-close>Close</button>
      </div>
    </div>
  `, i.appendChild(a);
  const d = a.querySelector(`.${S(r, "dialog")}`), f = a.querySelector("[data-cast-conference-status]"), o = a.querySelector("[data-cast-conference-create]"), b = a.querySelector("[data-cast-conference-manage]"), u = a.querySelector("[data-cast-conference-title-select]"), l = a.querySelector("[data-cast-conference-custom-group]"), p = a.querySelector("[data-cast-conference-custom-title]"), h = a.querySelector("[data-cast-conference-topics]"), g = a.querySelector("[data-cast-conference-info]"), C = a.querySelector("[data-cast-conference-create-btn]"), y = a.querySelector("[data-cast-conference-exit-btn]");
  function E(m, I2) {
    if (!f)
      return;
    const T = String(I2 || "").trim();
    if (f.classList.remove(
      S(r, "status-success"),
      S(r, "status-error")
    ), !T) {
      f.hidden = true, f.textContent = "";
      return;
    }
    f.hidden = false, f.textContent = T, m === "success" ? f.classList.add(S(r, "status-success")) : m === "error" && f.classList.add(S(r, "status-error"));
  }
  function w(m) {
    s = m, n && n(m);
  }
  function L() {
    const m = t() || {};
    return {
      hubEndpoint: String(m.hubEndpoint ?? "").trim(),
      topic: String(m.topic ?? "").trim(),
      subscriberName: String(m.subscriberName ?? "").trim(),
      userName: String(m.userName ?? "").trim(),
      connected: !!m.connected
    };
  }
  function U() {
    const m = String((u == null ? void 0 : u.value) || "").trim();
    return m === "other" ? String((p == null ? void 0 : p.value) || "").trim() : m;
  }
  function ct() {
    return h ? Array.from(
      h.querySelectorAll(
        'input[type="checkbox"][data-cast-conference-topic]:checked'
      )
    ).map((m) => String(m.value || "").trim()).filter(Boolean) : [];
  }
  function x(m, I2) {
    if (h) {
      if (h.replaceChildren(), !m.length) {
        const T = document.createElement("p");
        T.className = S(r, "muted"), T.textContent = "No users available", h.appendChild(T);
        return;
      }
      for (const T of m) {
        const N = document.createElement("label");
        N.className = S(r, "topic-row");
        const A = document.createElement("input");
        A.type = "checkbox", A.dataset.castConferenceTopic = "1", A.value = T, A.checked = true;
        const $2 = document.createElement("span");
        $2.textContent = T, N.appendChild(A), N.appendChild($2), h.appendChild(N);
      }
    }
  }
  function at(m) {
    if (o && (o.hidden = !!m), b && (b.hidden = !m), m && g) {
      const I2 = m.places.length ? `<div class="${S(r, "places")}">${m.places.map((T) => {
        const N = [S(r, "place")];
        T.isSelf && N.push(S(r, "place-self")), T.isLeading && N.push(S(r, "place-leading")), T.isFollowing && N.push(S(r, "place-following"));
        const A = T.statusLabel ? `<span class="${S(r, "place-status")}">${B(
          T.statusLabel
        )}</span>` : "";
        return `<span class="${N.join(
          " "
        )}" title="${B(T.placeId)}"><span class="${S(
          r,
          "place-name"
        )}">${B(T.label)}</span>${A}</span>`;
      }).join("")}</div>` : `<div class="${S(r, "muted")}">No places</div>`;
      g.innerHTML = `<div><strong>Title:</strong> ${B(
        m.title || "N/A"
      )}</div><div class="${S(r, "places-label")}">Participants</div>${I2}`;
    }
    y && (y.textContent = (m == null ? void 0 : m.selfRole) === "leading" ? "End conference" : "Leave conference");
  }
  async function re() {
    const m = L();
    if (!m.hubEndpoint || !m.connected)
      return x([], m.topic), at(null), w(null), null;
    const [I2, T] = await Promise.all([
      Pt(m.hubEndpoint),
      J(m.hubEndpoint)
    ]), N = V(
      m.topic,
      m.subscriberName,
      T
    ), A = Ke(m, N);
    return N || x(I2, m.topic), at(A), w(A), A;
  }
  async function ln() {
    if (c)
      return;
    const m = L(), I2 = U(), T = ct();
    if (!m.connected) {
      E("error", "Connect to the Cast hub first.");
      return;
    }
    if (!m.topic) {
      E("error", "Cast topic is required to host a conference.");
      return;
    }
    if (!I2) {
      E("error", "Conference title is required.");
      return;
    }
    if (!T.length) {
      E("error", "Select at least one attendee topic.");
      return;
    }
    c = true, C && (C.disabled = true, C.textContent = "Creating\u2026"), E("", "");
    try {
      await Bt(
        m.hubEndpoint,
        m.topic,
        I2,
        T,
        m.userName
      ), await re(), P2();
    } catch (N) {
      E(
        "error",
        N instanceof Error ? N.message : "Failed to create conference."
      );
    } finally {
      c = false, C && (C.disabled = false, C.textContent = "Create conference");
    }
  }
  async function fn() {
    if (c)
      return;
    const m = L(), I2 = await J(m.hubEndpoint), T = V(
      m.topic,
      m.subscriberName,
      I2
    );
    if (!T) {
      await re();
      return;
    }
    const N = Q(T), A = We(m.topic, T);
    c = true, y && (y.disabled = true, y.textContent = A ? "Ending\u2026" : "Leaving\u2026"), E("", "");
    try {
      await Je(
        m.hubEndpoint,
        N,
        A ? void 0 : m.topic
      ), w(null), E(
        "success",
        A ? "Conference ended." : "Left conference."
      ), await new Promise(($2) => {
        setTimeout($2, Ot);
      }), P2();
    } catch ($2) {
      E(
        "error",
        $2 instanceof Error ? $2.message : "Failed to update conference."
      );
    } finally {
      c = false, y && (y.disabled = false, y.textContent = A ? "End conference" : "Leave conference");
    }
  }
  function dn() {
    E("", ""), u && (u.value = "Test conference"), p && (p.value = ""), l && (l.hidden = true), a.hidden = false, d instanceof HTMLElement && d.focus({ preventScroll: true }), re().catch(() => {
    });
  }
  function P2() {
    a.hidden = true, E("", "");
  }
  function bn() {
    P2(), a.remove();
  }
  return u == null || u.addEventListener("change", () => {
    l && (l.hidden = String(u.value || "") !== "other");
  }), C == null || C.addEventListener("click", () => {
    ln().catch(() => {
    });
  }), y == null || y.addEventListener("click", () => {
    fn().catch(() => {
    });
  }), a.querySelectorAll("[data-cast-conference-close]").forEach((m) => {
    m.addEventListener("click", () => P2());
  }), a.addEventListener("click", (m) => {
    m.target === a && P2();
  }), {
    open: dn,
    close: P2,
    destroy: bn,
    refresh: re,
    getView: () => s,
    get element() {
      return a;
    }
  };
}
var Ue = "ID";
function er(e = {}) {
  const t = Array.isArray(e.selfActors) ? e.selfActors.map((s) => String(s || "").trim()).filter(Boolean) : [], n = Array.isArray(e.connectedActors) ? e.connectedActors.map((s) => String(s || "").trim()).filter(Boolean) : [];
  return t.some(
    (s) => s === Ue || s.toUpperCase() === "ID"
  ) ? true : !n.some(
    (s) => s === Ue || s.toUpperCase() === "ID"
  );
}
function Dt(e) {
  const t = e && typeof e == "object" ? (
    /** @type {{ event?: Record<string, unknown> }} */
    e.event
  ) : null;
  if (!t || typeof t != "object")
    return null;
  const n = String(t["hub.event"] || "").trim().toLowerCase();
  if (n !== "conference-start" && n !== "conference-end")
    return null;
  const r = t.context && typeof t.context == "object" ? (
    /** @type {Record<string, unknown>} */
    t.context
  ) : {}, i = String(
    r.hostTopic || t["hub.topic"] || ""
  ).trim(), s = String(r.title || "").trim(), c = String(r.hostUserName || "").trim();
  return {
    hubEvent: n,
    title: s,
    hostTopic: i,
    hostUserName: c,
    hostLabel: c || i || "host"
  };
}
function v(e, t) {
  return `${e}-${t}`;
}
function Mt(e = {}) {
  const t = typeof e.onDecision == "function" ? e.onDecision : null, n = String(
    e.classPrefix || "cast-conference-invite"
  ).trim(), r = e.root || (typeof document < "u" ? document.body : null);
  if (!r)
    throw new Error("createCastConferenceInviteDialog requires a DOM root");
  let i = null, s = false;
  const c = document.createElement("div");
  c.className = v(n, "overlay"), c.hidden = true, c.innerHTML = `
    <div class="${v(n, "dialog")}" role="dialog" aria-modal="true" aria-labelledby="${v(n, "title")}" tabindex="-1">
      <div class="${v(n, "header")}">
        <h2 id="${v(n, "title")}" class="${v(n, "title")}">Conference invitation</h2>
        <button type="button" class="${v(n, "close")}" data-cast-invite-close aria-label="Close">\xD7</button>
      </div>
      <div class="${v(n, "body")}">
        <p class="${v(n, "lead")}" data-cast-invite-lead></p>
        <p class="${v(n, "status")}" data-cast-invite-status role="alert" hidden></p>
        <div class="${v(n, "actions")}">
          <button type="button" class="${v(n, "primary")}" data-cast-invite-follow>Join and follow</button>
          <button type="button" class="${v(n, "secondary")}" data-cast-invite-join>Join but do not follow</button>
          <button type="button" class="${v(n, "secondary")}" data-cast-invite-decline>Do not join</button>
        </div>
      </div>
    </div>
  `, r.appendChild(c);
  const a = c.querySelector(`.${v(n, "dialog")}`), d = c.querySelector("[data-cast-invite-lead]"), f = c.querySelector("[data-cast-invite-status]"), o = c.querySelector("[data-cast-invite-follow]"), b = c.querySelector("[data-cast-invite-join]"), u = c.querySelector("[data-cast-invite-decline]");
  function l(y, E) {
    if (!(f instanceof HTMLElement)) return;
    const w = String(E || "").trim();
    f.hidden = !w, f.textContent = w, f.classList.toggle(v(n, "status-error"), y === "error");
  }
  function p(y) {
    s = y;
    for (const E of [o, b, u])
      E instanceof HTMLButtonElement && (E.disabled = y);
  }
  function h() {
    c.hidden = true, i = null, l("", ""), p(false);
  }
  function g(y) {
    const E = String((y == null ? void 0 : y.hostLabel) || (y == null ? void 0 : y.hostUserName) || (y == null ? void 0 : y.hostTopic) || "host").trim() || "host", w = String((y == null ? void 0 : y.title) || "").trim();
    i = {
      hubEvent: "conference-start",
      title: w,
      hostTopic: String((y == null ? void 0 : y.hostTopic) || "").trim(),
      hostUserName: String((y == null ? void 0 : y.hostUserName) || "").trim(),
      hostLabel: E
    }, d instanceof HTMLElement && (d.textContent = w ? `${w} \u2014 join and follow ${E}?` : `Join conference and follow ${E}?`), o instanceof HTMLElement && (o.textContent = `Join and follow ${E}`), l("", ""), c.hidden = false, a instanceof HTMLElement && a.focus({ preventScroll: true });
  }
  async function C(y) {
    if (s || !i) return;
    const E = i;
    p(true);
    try {
      t && await t(y, E), h();
    } catch (w) {
      l(
        "error",
        w instanceof Error ? w.message : "Failed to update conference."
      ), p(false);
    }
  }
  return o == null || o.addEventListener("click", () => {
    C("join-follow");
  }), b == null || b.addEventListener("click", () => {
    C("join");
  }), u == null || u.addEventListener("click", () => {
    C("decline");
  }), c.querySelectorAll("[data-cast-invite-close]").forEach((y) => {
    y.addEventListener("click", () => h());
  }), c.addEventListener("click", (y) => {
    y.target === c && h();
  }), {
    open: g,
    close: h,
    destroy() {
      h(), c.remove();
    },
    isOpen() {
      return !c.hidden;
    },
    getInvite() {
      return i;
    },
    element: c
  };
}
function tr(e) {
  const t = typeof (e == null ? void 0 : e.getSession) == "function" ? e.getSession : () => ({}), n = typeof (e == null ? void 0 : e.shouldShow) == "function" ? e.shouldShow : () => true, r = typeof (e == null ? void 0 : e.onConferenceChange) == "function" ? e.onConferenceChange : null, i = typeof (e == null ? void 0 : e.onFollowChange) == "function" ? e.onFollowChange : null, s = (e == null ? void 0 : e.dialog) || Mt({
    root: e == null ? void 0 : e.root,
    classPrefix: e == null ? void 0 : e.classPrefix,
    onDecision: async (f, o) => {
      await a(f, o);
    }
  });
  async function c() {
    const f = t(), o = await Ft(
      f.hubEndpoint || "",
      f.topic || "",
      f.subscriberName || "",
      f.userName || ""
    );
    return r == null || r(o), o;
  }
  async function a(f, o) {
    const b = t(), u = String((o == null ? void 0 : o.hostTopic) || "").trim();
    if (!u)
      throw new Error("Missing conference host topic.");
    if (f === "decline") {
      await Je(
        b.hubEndpoint || "",
        u,
        b.topic || ""
      ), i == null || i(false), r == null || r(null);
      return;
    }
    const l = f === "join-follow";
    i == null || i(l, l ? o : void 0), await c();
  }
  function d(f) {
    const o = Dt(f);
    if (!o) return false;
    const b = t(), u = String(b.topic || "").trim();
    if (o.hubEvent === "conference-end") {
      const l = (
        /** @type {{ context?: Record<string, unknown> }} */
        f.event || {}
      ), p = l.context && typeof l.context == "object" ? l.context : {}, h = String(p.leaveTopic || "").trim();
      return !h || h === u ? (s.close(), i == null || i(false), r == null || r(null)) : c(), true;
    }
    return u && o.hostTopic && (u === o.hostTopic || u.toLowerCase() === o.hostTopic.toLowerCase()) ? (i == null || i(true), c(), true) : n() ? (s.open(o), c(), true) : (c(), true);
  }
  return {
    handleHubMessage: d,
    open: (f) => s.open(f),
    close: () => s.close(),
    destroy: () => s.destroy(),
    refresh: c,
    dialog: s
  };
}
var fe = "http://fhircast.hl7.org/StructureDefinition/fhircast-imaging-study-open";
var de = "urn:dicom:uid";
var Ye = "urn:cast:nifti-url";
var Xe = "urn:cast:nifti-filename";
var be = "urn:cast:volview-sample-id";
var ee = "urn:cast:worklist-sample-id";
var he = "urn:cast:ohif-mode";
var te = "urn:cast:open-mode";
var K = "dicomweb";
var pe = "dicom-url";
var F = "files";
var me = "idc";
var ye = "idc";
var ge = "idc-source-bucket";
var Ze = "urn:cast:dicomweb-root";
function k(e) {
  return typeof e != "string" ? "" : e.trim().replace(/^urn:oid:/i, "");
}
function yt(e) {
  return typeof e == "string" ? e.trim().toLowerCase() : "";
}
function M(e, t) {
  if (!Array.isArray(e))
    return null;
  const n = String(t || "").trim().toLowerCase(), r = e.find(
    (i) => i && typeof i == "object" && typeof i.key == "string" && i.key.trim().toLowerCase() === n
  );
  return !(r != null && r.resource) || typeof r.resource != "object" ? null : r.resource;
}
function _(e, t) {
  const n = M(e, "study");
  if (!n)
    return "";
  const r = Array.isArray(n.identifier) ? n.identifier : [], i = yt(t), s = r.find(
    (c) => yt(c == null ? void 0 : c.system) === i
  );
  return typeof (s == null ? void 0 : s.value) == "string" ? s.value.trim() : "";
}
function H(e) {
  const t = M(e, "study");
  return t ? k(t.uid) || k(_(e, de)) : "";
}
function z(e) {
  const t = M(e, "series");
  if (t)
    return k(t.uid);
  const n = M(e, "study"), i = (Array.isArray(n == null ? void 0 : n.series) ? n.series : [])[0];
  return !i || typeof i != "object" ? "" : k(i.uid);
}
function $e(e) {
  const t = _(e, Ze);
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
function Y(e) {
  const t = _(e, Ye);
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
function Qe(e) {
  return _(e, Xe);
}
function jt(e) {
  return _(e, he);
}
function qt(e) {
  return _(e, be) || _(e, ee);
}
function nr(e) {
  return typeof e.url == "string" ? e.url.trim() : typeof e.uri == "string" ? e.uri.trim() : "";
}
function rr(e) {
  return typeof e.fileName == "string" ? e.fileName.trim() : typeof e.filename == "string" ? e.filename.trim() : "";
}
var ir = /* @__PURE__ */ new Set(["http:", "https:", "s3:", "gs:"]);
function Se(e) {
  if (!e || typeof e != "object")
    return null;
  const t = nr(e);
  if (!t)
    return null;
  try {
    const c = new URL(t);
    if (!ir.has(c.protocol))
      return null;
  } catch {
    return null;
  }
  const n = rr(e), r = typeof e.mimeType == "string" ? e.mimeType.trim() : "", i = typeof e.role == "string" ? e.role.trim() : "", s = typeof e.label == "string" ? e.label.trim() : "";
  return {
    url: t,
    fileName: n,
    mimeType: r,
    role: i,
    label: s
  };
}
function D(e) {
  const t = M(e, "files"), r = (Array.isArray(t == null ? void 0 : t.files) ? t.files : []).map((s) => Se(s)).filter((s) => s !== null);
  if (r.length > 0)
    return r;
  const i = Y(e);
  return i ? [
    {
      url: i,
      fileName: Qe(e),
      mimeType: "",
      role: "",
      label: ""
    }
  ] : [];
}
function Re(e) {
  return k(_(e, ye));
}
function Oe(e) {
  return _(
    e,
    ge
  ).toLowerCase() === "gcs" ? "gcs" : "aws";
}
function zt(e) {
  const t = _(e, te);
  return t === K || t === pe || t === F || t === me ? t : D(e).length > 0 || Y(e) ? F : H(e) && !Y(e) ? K : "";
}
function Ht({
  id: e,
  files: t,
  patientReference: n,
  openMode: r,
  includeLegacyNiftiIdentifiers: i = false
}) {
  const s = String(e || "").trim() || "study", c = (Array.isArray(t) ? t : []).map((o) => Se(o)).filter((o) => o !== null), a = [
    { system: te, value: r },
    {
      system: ee,
      value: s
    },
    {
      system: be,
      value: s
    }
  ];
  if (i && c.length === 1) {
    const [o] = c;
    a.push({
      system: Ye,
      value: o.url
    }), o.fileName && a.push({
      system: Xe,
      value: o.fileName
    });
  }
  const d = {
    resourceType: "ImagingStudy",
    id: s,
    meta: {
      profile: [fe]
    },
    identifier: a,
    status: "available"
  };
  n && (d.subject = { reference: String(n).trim() });
  const f = [
    {
      key: "study",
      resource: d
    }
  ];
  return c.length > 0 && f.push({
    key: "files",
    resource: {
      files: c.map((o) => ({
        url: o.url,
        fileName: o.fileName || void 0,
        mimeType: o.mimeType || void 0,
        role: o.role || void 0,
        label: o.label || void 0
      }))
    }
  }), f;
}
function Gt({
  id: e,
  files: t,
  patientReference: n,
  includeLegacyNiftiIdentifiers: r = true
}) {
  return Ht({
    id: e,
    files: t,
    patientReference: n,
    openMode: F,
    includeLegacyNiftiIdentifiers: r
  });
}
function sr({
  id: e,
  files: t,
  patientReference: n
}) {
  return Ht({
    id: e,
    files: t,
    patientReference: n,
    openMode: pe,
    includeLegacyNiftiIdentifiers: false
  });
}
function or({
  id: e,
  studyInstanceUID: t,
  seriesInstanceUID: n,
  dicomwebRoot: r,
  patientReference: i,
  ohifMode: s,
  files: c,
  sourceBucket: a
}) {
  const d = String(e || "").trim() || "study", f = k(t), o = k(n), b = String(r || "").trim().replace(/\/$/, ""), u = [
    { system: te, value: K },
    { system: de, value: f },
    {
      system: ee,
      value: d
    }
  ];
  b && u.push({ system: Ze, value: b });
  const l = String(s || "").trim();
  l && u.push({
    system: he,
    value: l
  });
  const p = (Array.isArray(c) ? c : []).map((C) => Se(C)).filter((C) => C !== null);
  if (p.length > 0) {
    const C = String(a || "aws").trim().toLowerCase() === "gcs" ? "gcs" : "aws";
    u.push({
      system: ge,
      value: C
    }), o && u.push({ system: ye, value: o });
  }
  const h = {
    resourceType: "ImagingStudy",
    id: d,
    uid: f,
    meta: {
      profile: [fe]
    },
    identifier: u,
    status: "available"
  };
  i && (h.subject = { reference: String(i).trim() });
  const g = [
    {
      key: "study",
      resource: h
    }
  ];
  return o && g.push({
    key: "series",
    resource: {
      resourceType: "ImagingStudy",
      uid: o
    }
  }), p.length > 0 && g.push({
    key: "files",
    resource: {
      files: p.map((C) => ({
        url: C.url,
        fileName: C.fileName || void 0,
        mimeType: C.mimeType || void 0,
        role: C.role || void 0,
        label: C.label || void 0
      }))
    }
  }), g;
}
function cr({
  id: e,
  studyInstanceUID: t,
  seriesInstanceUID: n,
  sourceBucket: r,
  files: i,
  patientReference: s,
  ohifMode: c
}) {
  const a = String(e || "").trim() || "study", d = k(t), f = k(n), o = String(r || "aws").trim().toLowerCase() === "gcs" ? "gcs" : "aws", b = (Array.isArray(i) ? i : []).map((g) => Se(g)).filter((g) => g !== null), u = [
    { system: te, value: me },
    { system: de, value: d },
    {
      system: ee,
      value: a
    },
    {
      system: be,
      value: a
    },
    { system: ge, value: o }
  ];
  f && u.push({ system: ye, value: f });
  const l = String(c || "").trim();
  l && u.push({
    system: he,
    value: l
  });
  const p = {
    resourceType: "ImagingStudy",
    id: a,
    uid: d,
    meta: {
      profile: [fe]
    },
    identifier: u,
    status: "available"
  };
  s && (p.subject = { reference: String(s).trim() });
  const h = [
    {
      key: "study",
      resource: p
    }
  ];
  return f && h.push({
    key: "series",
    resource: {
      resourceType: "ImagingStudy",
      uid: f
    }
  }), b.length > 0 && h.push({
    key: "files",
    resource: {
      files: b.map((g) => ({
        url: g.url,
        fileName: g.fileName || void 0,
        mimeType: g.mimeType || void 0,
        role: g.role || void 0,
        label: g.label || void 0
      }))
    }
  }), h;
}
function ar({
  id: e,
  url: t,
  filename: n,
  patientReference: r
}) {
  return Gt({
    id: e,
    files: [{ url: t, fileName: n }],
    patientReference: r,
    includeLegacyNiftiIdentifiers: true
  });
}
function et(e) {
  if (Array.isArray(e) || !e || typeof e != "object")
    return e;
  const t = Object.entries(e).reduce((n, [r, i]) => (i && typeof i == "object" && !Array.isArray(i) && n.push({ key: r, resource: i }), n), []);
  return t.length ? t : e;
}
function gt(e, t, n) {
  const r = et(e), i = D(r);
  return i.length === 0 ? null : { mode: "files", studyId: t, files: i, ohifMode: n };
}
function ur(e) {
  const t = et(e), n = qt(t) || "study", r = zt(t), i = jt(t) || void 0;
  if (r === K) {
    const d = H(t);
    if (!d)
      return null;
    const f = {
      mode: "dicomweb",
      studyId: n,
      studyInstanceUID: d,
      seriesInstanceUID: z(t) || void 0,
      dicomwebRoot: $e(t) || void 0,
      ohifMode: i
    }, o = D(t);
    return o.length > 0 && (f.idcFallback = {
      studyInstanceUID: d,
      seriesInstanceUID: Re(t) || z(t) || void 0,
      sourceBucket: Oe(t),
      files: o
    }), f;
  }
  if (r === F) {
    const d = gt(t, n, i);
    if (d)
      return d;
  }
  if (r === pe) {
    const d = D(t);
    return d.length ? { mode: "dicom-url", studyId: n, files: d, ohifMode: i } : null;
  }
  if (r === me) {
    const d = H(t);
    if (!d)
      return null;
    const f = D(t);
    return f.length ? {
      mode: "idc",
      studyId: n,
      studyInstanceUID: d,
      seriesInstanceUID: Re(t) || z(t) || void 0,
      sourceBucket: Oe(t),
      files: f,
      ohifMode: i
    } : null;
  }
  const s = gt(t, n, i);
  if (s)
    return s;
  const c = Y(t);
  if (c) {
    const d = Qe(t);
    return {
      mode: "files",
      studyId: n,
      files: [{ url: c, fileName: d || "volume.nii.gz", label: n }],
      ohifMode: i
    };
  }
  const a = H(t);
  return a ? {
    mode: "dicomweb",
    studyId: n,
    studyInstanceUID: a,
    seriesInstanceUID: z(t) || void 0,
    dicomwebRoot: $e(t) || void 0,
    ohifMode: i
  } : null;
}
function xe(e) {
  return typeof e != "string" ? "" : e.trim().replace(/^urn:oid:/i, "");
}
function lr(e) {
  if (!e || typeof e != "object")
    return "";
  const t = xe(e.uid);
  if (t)
    return t;
  const r = (Array.isArray(e.identifier) ? e.identifier : []).find(
    (i) => typeof (i == null ? void 0 : i.system) == "string" && i.system.toLowerCase() === "urn:dicom:uid"
  );
  return xe(r == null ? void 0 : r.value);
}
function ne(e) {
  const t = e == null ? void 0 : e["hub.event"];
  return typeof t == "string" ? t.trim().toLowerCase() : "";
}
function Wt(e) {
  return typeof e == "string" ? e.trim().toUpperCase() : !e || typeof e != "object" ? "" : (typeof e.keyword == "string" && e.keyword || typeof e.id == "string" && e.id || typeof e.key == "string" && e.key || "").trim().toUpperCase();
}
function fr(e) {
  const t = e == null ? void 0 : e["target.actor"];
  return t == null ? "" : Wt(t);
}
function dr(e) {
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
function br(e) {
  const t = e == null ? void 0 : e.event;
  return t && typeof t == "object" && Array.isArray(t.context) ? t.context : [];
}
function hr(e) {
  return String(
    (e == null ? void 0 : e["subscriber.name"]) || (e == null ? void 0 : e.subscriber) || ""
  ).trim() || null;
}
function pr(e) {
  return String((e == null ? void 0 : e["subscriber.product.name"]) || "").trim();
}
function mr(e) {
  const t = String((e == null ? void 0 : e["subscriber.actor"]) || "").trim();
  if (t)
    return t;
  const n = e == null ? void 0 : e["subscriber.actors"];
  if (Array.isArray(n))
    for (const r of n) {
      const i = String(r || "").trim();
      if (i)
        return i;
    }
  return "";
}
function yr(e) {
  const t = e == null ? void 0 : e.event;
  if (ne(t) !== "status-update")
    return null;
  const n = t != null && t.context && typeof t.context == "object" && !Array.isArray(t.context) ? t.context : {};
  return { message: String(n.message ?? "").trim(), level: String(n.level ?? "info") };
}
function gr(e) {
  const t = atob(e), n = new Uint8Array(t.length);
  for (let r = 0; r < t.length; r += 1)
    n[r] = t.charCodeAt(r);
  return n;
}
function Vt(e) {
  try {
    const t = gr(e), n = new ArrayBuffer(t.byteLength);
    return new Uint8Array(n).set(t), n;
  } catch {
    return null;
  }
}
function Jt(e) {
  const t = e == null ? void 0 : e.context;
  if (!t || typeof t != "object" || Array.isArray(t))
    return [];
  const n = t.files;
  return Array.isArray(n) ? n.filter((r) => !!(r && typeof r == "object")) : [];
}
function Sr(e) {
  const t = e == null ? void 0 : e.context;
  return Array.isArray(t) ? t : t != null ? [t] : [];
}
function St(e, t, n) {
  return e instanceof ArrayBuffer ? { arrayBuffer: e, fileName: t, mimeType: n } : typeof e == "string" && e ? { fileName: t, data: e, mimeType: n } : null;
}
function Kt(e) {
  return e === "nifti-send" ? "application/vnd.unknown.nifti-1" : "application/dicom";
}
function Cr(e, t) {
  return e === "nifti-send" ? `cast-nifti-send-${t + 1}.nii.gz` : `cast-dicom-send-${t + 1}.dcm`;
}
function wr(e, t) {
  return e === "nifti-send" ? `cast-nifti-send-${t + 1}.nii.gz` : "dicom-sr.dcm";
}
function Er(e, t) {
  return typeof e.mimeType == "string" && e.mimeType.trim() ? e.mimeType.trim() : Kt(t);
}
function Tr(e, t) {
  return typeof e.mimeType == "string" && e.mimeType.trim() ? e.mimeType.trim() : typeof e.contentType == "string" && e.contentType.trim() ? e.contentType.trim() : Kt(t);
}
function Nr(e, t) {
  const n = e == null ? void 0 : e.event;
  if (ne(n) !== t.toLowerCase())
    return [];
  const r = Jt(n);
  if (r.length) {
    const s = r.map((c, a) => {
      const d = Er(c, t), f = typeof c.fileName == "string" && c.fileName.trim() ? c.fileName.trim() : Cr(t, a);
      return St(c.data, f, d);
    }).filter(Boolean);
    if (s.length)
      return s;
  }
  return Sr(n).map((s, c) => {
    if (!s || typeof s != "object")
      return null;
    const a = s.resource;
    if (!a || typeof a != "object")
      return null;
    const d = Tr(a, t), f = typeof a.fileName == "string" && a.fileName.trim() ? a.fileName.trim() : wr(t, c);
    return St(a.data, f, d);
  }).filter(Boolean);
}
function Yt(e) {
  return (e == null ? void 0 : e.arrayBuffer) instanceof ArrayBuffer ? e.arrayBuffer : typeof (e == null ? void 0 : e.data) == "string" ? Vt(e.data) : null;
}
function Ar(e, t, n) {
  const r = Yt(e);
  if (!r)
    return null;
  const i = (e == null ? void 0 : e.fileName) || t, s = (e == null ? void 0 : e.mimeType) || n;
  return new File([r], i, { type: s });
}
var Ct = 4e3;
var vr = 480;
var oe = "[binary/redacted]";
function ce(e, t) {
  return t > 8 ? "[max-depth]" : e instanceof ArrayBuffer || ArrayBuffer.isView(e) ? oe : typeof e == "string" ? e.length > vr ? `${e.slice(0, 160)}\u2026 (${e.length} chars)` : e : Array.isArray(e) ? e.map((n) => ce(n, t + 1)) : e && typeof e == "object" ? Object.entries(e).reduce((n, [r, i]) => (r.toLowerCase() === "data" && typeof i == "string" && i.length > 200 ? n[r] = oe : n[r] = ce(i, t + 1), n), {}) : e;
}
function Pe(e) {
  try {
    const t = JSON.stringify(ce(e, 0));
    return t.length > Ct ? `${t.slice(0, Ct)}\u2026` : t;
  } catch {
    return "[unserializable]";
  }
}
function tt(e) {
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
function Xt(e, t) {
  if (!e)
    return t === 1 ? "1 object" : `${t} objects`;
  if (t === 0)
    return e;
  const n = e === "dicom-send" || e === "nifti-send" ? "image" : "object", r = t === 1 ? n : `${n}s`;
  return `${e} (${t} ${r})`;
}
function Lr(e) {
  const t = e != null && e.event && typeof e.event == "object" ? e.event : void 0, r = ne(t) || "publish", i = tt(t);
  return { label: Xt(r, i), detail: "" };
}
function Ir(e) {
  if (!e || typeof e != "object")
    return { label: "raw", detail: Pe(e) };
  const t = e.event, r = ne(t) || (typeof (t == null ? void 0 : t.event) == "string" ? String(t.event) : "message"), i = tt(t);
  return { label: Xt(r, i), detail: Pe(e) };
}
function _r(e) {
  return ce(e, 0);
}
var R = "*";
var Ce = "ID";
var kr = {
  subscriberName: R,
  subscriberActor: Ce,
  targetActor: R,
  targetProductName: R
};
function X(e) {
  return String(e ?? "").trim() || R;
}
function Ur(e, t) {
  return {
    subscriberName: (e.subscriberName ?? "").trim() || t.subscriberName || R,
    subscriberActor: (e.subscriberActor ?? "").trim() || Ce,
    targetActor: X(e.targetActor),
    targetProductName: X(e.targetProductName)
  };
}
function $r(e, t) {
  e["subscriber.name"] = t.subscriberName.trim() || R, e["subscriber.actor"] = t.subscriberActor.trim() || Ce, e["target.actor"] = X(t.targetActor), e["target.product.name"] = X(
    t.targetProductName
  );
}
function Zt(e) {
  const t = e == null ? void 0 : e.event;
  return !!(t && Ie(t));
}
async function Rr(e, t) {
  var i, s;
  if (!(((i = e == null ? void 0 : e.hasPendingPayload) == null ? void 0 : i.call(e, t)) || Zt(t)))
    return t;
  let r = t;
  if (e != null && e.fetchAllPayloads)
    r = await e.fetchAllPayloads(t);
  else if (e != null && e.fetchPayload)
    for (; (s = e.hasPendingPayload) != null && s.call(e, r); )
      r = await e.fetchPayload(r);
  return r;
}
var Qt = [
  "TOTALSEG",
  "TOTAL_SEGMENTATOR",
  "TOTAL-SEGMENTATOR"
];
var en = [
  "LUNGSCREENING",
  "LUNG_SCREENING"
];
var tn = [
  "NEURO_SEG",
  "NEUROSEG",
  "NEURO-SEG"
];
function Be(e) {
  return String(e || "").trim().toUpperCase().replace(/-/g, "_");
}
function nt(e, t) {
  const n = Be(e);
  if (!n)
    return false;
  const r = n.replace(/_/g, "");
  return t.some((i) => {
    const s = Be(i);
    return s === n || s.replace(/_/g, "") === r;
  });
}
function we(e) {
  return nt(e, Qt);
}
function Ee(e) {
  return nt(e, en);
}
function Te(e) {
  return nt(e, tn);
}
function Or(e) {
  return we(e) || Ee(e) || Te(e);
}
function rt(e, t) {
  if (!Array.isArray(e))
    return;
  const n = String(t).trim().toLowerCase(), r = e.find((s) => !s || typeof s != "object" ? false : String(s.key ?? "").trim().toLowerCase() === n);
  if (!r)
    return;
  const i = r.value;
  return typeof i == "string" ? i.trim() : void 0;
}
function it(e) {
  if (!e || typeof e != "object" || e.source !== "status")
    return false;
  const t = rt(e.items, "availability");
  return t ? t.toLowerCase() === "online" : Array.isArray(e.items) && e.items.length > 0;
}
function Ne(e) {
  const t = String((e == null ? void 0 : e.productName) ?? "").trim();
  if (t)
    return t;
  const n = e == null ? void 0 : e.data;
  if (n && typeof n == "object") {
    const r = String(n.product ?? "").trim();
    if (r)
      return r;
  }
  return "";
}
function st(e, t) {
  return Array.isArray(e) ? e.some((n) => {
    const r = Ne(n);
    return t(r) && it(n.data);
  }) : false;
}
function xr(e) {
  return st(
    e,
    we
  );
}
function Pr(e) {
  return st(e, Ee);
}
function Br(e) {
  return st(e, Te);
}
function Fr(e) {
  return typeof e != "string" ? false : e.trim().toUpperCase() === "STATUS";
}
var Dr = "lung";
var Mr = "LUNGSCREENING";
var jr = "Lung Screening";
var qr = "Toronto, Canada";
var zr = "CT lung screening (dicom-send / nifti-send). Returns a screening result line.";
var Hr = "https://github.com/cast-interface/slicer-cast-extension/blob/main/CastInterface/cast_resource_servers/products/lung_screening.py";
var Gr = "dicom-send";
var Wr = {
  id: Dr,
  product: Mr,
  title: jr,
  location: qr,
  summary: zr,
  githubUrl: Hr,
  hubEvent: Gr
};
var Vr = "totalseg";
var Jr = "TOTALSEG";
var Kr = "Total Segmentator";
var Yr = "Frankfurt, Germany";
var Xr = "Whole-body CT/MR multi-organ segmentation (dicom-send / nifti-send jobs). Returns DICOM SEG via dicom-send.";
var Zr = "https://github.com/wasserth/TotalSegmentator";
var Qr = "https://totalsegmentator.com/";
var ei = "https://buymeacoffee.com/totalsegmentator";
var ti = "dicom-send";
var ni = {
  id: Vr,
  product: Jr,
  title: Kr,
  location: Yr,
  summary: Xr,
  githubUrl: Zr,
  websiteUrl: Qr,
  buyMeACoffeeUrl: ei,
  hubEvent: ti
};
var ri = "neuro";
var ii = "NEURO_SEG";
var si = "Subcortical segmentation";
var oi = "Montreal, Canada";
var ci = "Neuro / subcortical segmentation (nifti-send). Prefers a NIfTI volume input.";
var ai = "https://github.com/cast-interface/slicer-cast-extension/blob/main/CastInterface/cast_resource_servers/products/neuro_seg.py";
var ui = "nifti-send";
var li = {
  id: ri,
  product: ii,
  title: si,
  location: oi,
  summary: ci,
  githubUrl: ai,
  hubEvent: ui
};
function Le(e) {
  const t = String(e.summary || e.capabilities || "").trim(), n = e.hubEvent === "nifti-send" ? "nifti-send" : "dicom-send", r = {
    id: String(e.id || "").trim(),
    product: String(e.product || "").trim(),
    title: String(e.title || "").trim(),
    location: String(e.location || "").trim(),
    summary: t,
    capabilities: t,
    hubEvent: n
  }, i = String(e.githubUrl || "").trim();
  i && (r.githubUrl = i);
  const s = String(e.websiteUrl || "").trim();
  s && (r.websiteUrl = s);
  const c = String(e.buyMeACoffeeUrl || "").trim();
  return c && (r.buyMeACoffeeUrl = c), r;
}
var O = Object.freeze([
  Le(Wr),
  Le(ni),
  Le(li)
]);
function q(e) {
  return String(e ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function fi(e) {
  if (!e || typeof e != "object") return "";
  const t = q(e.title), n = q(e.summary || e.capabilities), r = [
    '<div class="cast-inference-info-card">',
    `<div class="cast-inference-info-title">${t}</div>`
  ];
  n && r.push(`<p class="cast-inference-info-summary">${n}</p>`), r.push('<div class="cast-inference-info-links">');
  const i = String(e.githubUrl || "").trim();
  i && r.push(
    `<a class="cast-inference-info-link cast-inference-info-github" href="${q(i)}" target="_blank" rel="noopener noreferrer"><svg class="cast-inference-info-github-icon" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>GitHub</a>`
  );
  const s = String(e.websiteUrl || "").trim();
  s && r.push(
    `<a class="cast-inference-info-link" href="${q(s)}" target="_blank" rel="noopener noreferrer">Website</a>`
  );
  const c = String(e.buyMeACoffeeUrl || "").trim();
  return c && r.push(
    `<a class="cast-inference-info-coffee" href="${q(c)}" target="_blank" rel="noopener noreferrer" title="Buy Me a Coffee"><span aria-hidden="true">\u2615</span> Buy me a coffee</a>`
  ), r.push("</div></div>"), r.join("");
}
function di(e, t) {
  if (!e || typeof e != "object")
    return "";
  const n = [
    String(e.title || "").trim(),
    String(e.summary || e.capabilities || "").trim(),
    `Location: ${String(e.location || "").trim()}`
  ].filter((i) => i && !i.endsWith(": "));
  t && n.push(`Status: ${t}`);
  const r = String(e.githubUrl || "").trim();
  return r && n.push(r), n.join(`
`);
}
function nn(e) {
  const t = String(e || "").trim().toUpperCase().replace(/-/g, "_");
  if (!t)
    return null;
  const n = t.replace(/_/g, "");
  return O.find((r) => {
    const i = String(r.product || "").trim().toUpperCase().replace(/-/g, "_");
    return i === t || i.replace(/_/g, "") === n;
  }) || null;
}
function rn(e) {
  return String(e || "").trim().toUpperCase();
}
var sn = Object.freeze({
  ira: (e) => e === "SLICERLIVE-IRA" || e.startsWith("SLICERLIVE-IRA"),
  ohif: (e) => e === "OHIF" || e.startsWith("OHIF"),
  volview: (e) => e === "VOLVIEW" || e.startsWith("VOLVIEW"),
  slicer: (e) => e === "3DSLICER-ID" || e === "SLICER-HUB" || e.startsWith("3DSLICER") || e.startsWith("SLICER-HUB"),
  lung: (e) => e === "LUNGSCREENING" || e.startsWith("LUNGSCREENING"),
  totalseg: (e) => e === "TOTALSEG" || e.startsWith("TOTALSEG") || e === "TOTAL_SEGMENTATOR" || e.startsWith("TOTAL_SEGMENTATOR") || e === "TOTAL-SEGMENTATOR" || e.startsWith("TOTAL-SEGMENTATOR"),
  neuro: (e) => e === "NEURO_SEG" || e.startsWith("NEURO_SEG") || e === "NEUROSEG" || e.startsWith("NEUROSEG"),
  reporting: (e) => e === "CAST-RPT" || e.startsWith("CAST-RPT")
});
function bi(e, t) {
  const n = sn[t];
  return n ? n(rn(e)) : false;
}
function on(e, t, n) {
  if (!e || typeof e != "object")
    return { online: false, items: [], raw: n, ...t ? { productName: t } : {} };
  const r = (
    /** @type {{ items?: unknown }} */
    e
  ), i = Array.isArray(r.items) ? r.items : [], s = rt(i, "job");
  return {
    online: it(e),
    items: i,
    ...s ? { job: s } : {},
    raw: n,
    ...t ? { productName: t } : {}
  };
}
function cn(e) {
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
  const { responses: n } = le(t.data);
  for (const r of n) {
    const i = r.data;
    if (!i || typeof i != "object")
      continue;
    const s = Ne(r) || void 0, c = on(i, s, t.data);
    if (c.online || Array.isArray(
      /** @type {{ items?: unknown }} */
      i.items
    ))
      return c;
  }
  return { online: false, items: [], raw: t.data };
}
async function ot(e, t) {
  const n = String((t == null ? void 0 : t.subscriberName) || "").trim();
  if (!(e != null && e.request))
    return { ok: false, status: 0, data: { error: "No Cast client" } };
  if (!n)
    return { ok: false, status: 0, data: { error: "No subscriber name" } };
  const r = String((t == null ? void 0 : t.topic) || "").trim(), i = String((t == null ? void 0 : t.subscriberProductName) || "").trim(), s = String((t == null ? void 0 : t.subscriberActor) || "").trim(), c = String((t == null ? void 0 : t.targetActor) || "*").trim() || "*", a = String((t == null ? void 0 : t.targetProductName) || "").trim();
  return e.request({
    "subscriber.name": n,
    ...i ? { "subscriber.product.name": i } : {},
    ...s ? { "subscriber.actor": s } : {},
    "target.actor": c,
    ...a ? { "target.product.name": a } : {},
    event: {
      "hub.event": ze("STATUS"),
      ...r ? { "hub.topic": r } : {},
      context: { dataType: "STATUS" }
    }
  });
}
async function hi(e, t) {
  const n = String((t == null ? void 0 : t.targetProductName) || "").trim(), r = String((t == null ? void 0 : t.subscriberName) || "").trim();
  if (!(e != null && e.request))
    return { online: false, items: [], error: "No Cast client" };
  if (!r)
    return { online: false, items: [], error: "No subscriber name" };
  if (!n)
    return { online: false, items: [], error: "No target product" };
  try {
    const i = await ot(e, {
      subscriberName: r,
      subscriberProductName: t == null ? void 0 : t.subscriberProductName,
      subscriberActor: t == null ? void 0 : t.subscriberActor,
      topic: t == null ? void 0 : t.topic,
      targetProductName: n
    });
    return cn(i);
  } catch (i) {
    return {
      online: false,
      items: [],
      error: i instanceof Error ? i.message : String(i)
    };
  }
}
function pi(e, t) {
  if (!e) return false;
  const n = nn(e);
  return n && n.id === t.id ? true : t.id === "totalseg" ? we(e) : t.id === "lung" ? Ee(e) : t.id === "neuro" ? Te(e) : false;
}
function an(e) {
  if (!e || typeof e != "object")
    return O.map((r) => ({
      server: r,
      probe: { online: false, items: [], error: "Empty STATUS result" }
    }));
  const t = (
    /** @type {{ ok?: boolean, status?: number, data?: unknown }} */
    e
  );
  if (t.ok === false) {
    const r = `HTTP ${t.status ?? "?"}`;
    return O.map((i) => ({
      server: i,
      probe: { online: false, items: [], error: r, raw: t.data }
    }));
  }
  const { responses: n } = le(t.data);
  return O.map((r) => {
    for (const i of n) {
      const s = Ne(i);
      if (!pi(s, r)) continue;
      const c = i.data;
      return {
        server: r,
        probe: on(
          c,
          s || r.product,
          t.data
        )
      };
    }
    return {
      server: r,
      probe: { online: false, items: [], raw: t.data }
    };
  });
}
async function mi(e, t) {
  const n = String((t == null ? void 0 : t.subscriberName) || "").trim();
  if (!(e != null && e.request) || !n)
    return O.map((r) => ({
      server: r,
      probe: {
        online: false,
        items: [],
        error: e != null && e.request ? "No subscriber name" : "No Cast client"
      }
    }));
  try {
    const r = await ot(e, {
      subscriberName: n,
      subscriberProductName: t == null ? void 0 : t.subscriberProductName,
      subscriberActor: t == null ? void 0 : t.subscriberActor,
      topic: t == null ? void 0 : t.topic,
      targetActor: "*"
    });
    return an(r);
  } catch (r) {
    const i = r instanceof Error ? r.message : String(r);
    return O.map((s) => ({
      server: s,
      probe: { online: false, items: [], error: i }
    }));
  }
}
async function yi(e, t) {
  const n = String((t == null ? void 0 : t.targetProductName) || "").trim(), r = String((t == null ? void 0 : t.hubEvent) || "").trim(), i = String((t == null ? void 0 : t.subscriberName) || "").trim(), s = String((t == null ? void 0 : t.topic) || "").trim(), c = Array.isArray(t == null ? void 0 : t.files) ? t.files : [];
  if (!(e != null && e.publish))
    return { ok: false, detail: "No Cast client" };
  if (!i)
    return { ok: false, detail: "No subscriber name" };
  if (!s)
    return { ok: false, detail: "No Cast topic" };
  if (!n)
    return { ok: false, detail: "No target product" };
  if (!r)
    return { ok: false, detail: "No hub.event" };
  if (!c.length)
    return { ok: false, detail: "No volume files" };
  const a = String((t == null ? void 0 : t.subscriberProductName) || "").trim(), d = String((t == null ? void 0 : t.subscriberActor) || "").trim(), f = t != null && t.contextExtras && typeof t.contextExtras == "object" ? t.contextExtras : {};
  try {
    const o = await e.publish({
      "subscriber.name": i,
      ...a ? { "subscriber.product.name": a } : {},
      ...d ? { "subscriber.actor": d } : {},
      "target.product.name": n,
      event: {
        "hub.topic": s,
        "hub.event": r,
        context: {
          files: c.map((u) => ({
            url: String(u.url || "").trim(),
            fileName: String(u.fileName || "volume").trim() || "volume"
          })),
          ...f
        }
      }
    }), b = !!(o && /** @type {{ ok?: boolean }} */
    o.ok !== false);
    return {
      ok: b,
      detail: b ? "published" : `publish status ${/** @type {{ status?: number }} */
      (o == null ? void 0 : o.status) ?? "?"}`,
      response: o
    };
  } catch (o) {
    return {
      ok: false,
      detail: o instanceof Error ? o.message : String(o)
    };
  }
}
var gi = 1e4;
var wt = 5e3;
var Et = {
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
function Si(e, t) {
  t.classHierarchy.push("vtkCastClient");
  function n(o, b) {
    t.onConnectionStateChangeCallback && t.onConnectionStateChangeCallback(o, b);
  }
  function r() {
    const o = t.session.productName || t.config.productName || Fe;
    return gn(o);
  }
  function i() {
    const o = t.hub.authorization_endpoint;
    if (typeof o == "string" && o.trim())
      return o.trim();
    try {
      return `${new URL(t.hub.token_endpoint).origin}/oauth/authorize`;
    } catch {
      return "";
    }
  }
  function s() {
    console.debug("CastClient: websocket is closed."), t.hub.resubscribeRequested = true, n("disconnected");
  }
  const c = Wn(() => ({
    hubEndpoint: t.hub.hub_endpoint,
    accessToken: t.hub.token
  }));
  function a(o) {
    try {
      const b = JSON.parse(o);
      if (b["hub.mode"])
        return;
      const u = b.event;
      if (!u || u["hub.event"] === "heartbeat" || b.id === t.hub.lastPublishedMessageID)
        return;
      t.onMessageCallback && t.onMessageCallback(b);
    } catch (b) {
      console.warn("CastClient: websocket processing error:", b);
    }
  }
  async function d() {
    t.hub.resubscribeRequested && t.hub.subscribed && t.config.autoReconnect ? (console.debug("CastClient: Try to resubscribe"), t.hub.resubscribeRequested = false, await e.subscribe() !== 202 && (t.hub.resubscribeRequested = true)) : !t.hub.subscribed && t.hub.resubscribeRequested && (t.hub.resubscribeRequested = false);
  }
  e.onMessage = (o) => {
    t.onMessageCallback = o;
  }, e.onConnectionStateChange = (o) => {
    t.onConnectionStateChangeCallback = o;
  }, e.delete = mn(() => {
    t.reconnectInterval && (clearInterval(t.reconnectInterval), t.reconnectInterval = null), e.unsubscribe();
  }, e.delete), e.getHubConfig = () => {
    const o = t.hub;
    return {
      name: o.name,
      friendlyName: o.friendlyName,
      version: o.version,
      hub_endpoint: o.hub_endpoint,
      authorization_endpoint: o.authorization_endpoint,
      token_endpoint: o.token_endpoint,
      client_id: o.client_id,
      client_secret: o.client_secret
    };
  }, e.getSessionConfig = () => {
    const o = t.session;
    return {
      subscriberName: o.subscriberName,
      productName: o.productName,
      productVersion: o.productVersion,
      actors: o.actors,
      topic: o.topic,
      events: o.events,
      lease: o.lease,
      userName: o.userName,
      defaultTargetActor: o.defaultTargetActor
    };
  }, e.getConnectionState = () => {
    const o = t.hub;
    return {
      token: o.token,
      subscribed: o.subscribed,
      resubscribeRequested: o.resubscribeRequested,
      websocket: o.websocket,
      lastPublishedMessageID: o.lastPublishedMessageID
    };
  }, e.setTopic = (o) => {
    console.debug("CastClient: setting topic to", o), t.session.topic = o;
  }, e.setToken = (o) => {
    t.hub.token = o;
  }, e.setSubscriberName = (o) => {
    t.session.subscriberName = o;
  }, e.setUserName = (o) => {
    t.session.userName = o || "";
  }, e.authenticate = async () => {
    const o = i();
    if (!o)
      throw new Error(
        "CastClient.authenticate: no authorization_endpoint or token_endpoint configured."
      );
    try {
      const h = new URL(o);
      console.debug(
        "CastClient: Authorizing at:",
        `${h.origin}${h.pathname}`
      );
    } catch {
      console.debug("CastClient: Authorizing at hub");
    }
    const b = t.session.productName || t.config.productName || "CAST", u = new URLSearchParams();
    t.hub.lastIdToken ? u.append("id_token", t.hub.lastIdToken) : t.session.userName && u.append("user_name", t.session.userName), u.append("client_product_name", b), t.session.topic && u.append("topic", t.session.topic);
    let l;
    try {
      l = await fetch(o, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: u
      });
    } catch (h) {
      const g = h instanceof Error ? h.message : String(h);
      throw console.error("CastClient: Exception during authenticate:", g), h;
    }
    if (l.status !== 200) {
      const h = await l.text().catch(() => "");
      throw console.error(
        "CastClient: Authenticate failed. Status:",
        l.status,
        h
      ), new Error(
        `CastClient.authenticate failed (HTTP ${l.status})`
      );
    }
    const p = await l.json();
    return typeof p.user_name == "string" && p.user_name && (t.session.userName = p.user_name), {
      user_name: p.user_name || "",
      code: p.code || "",
      expires_in: typeof p.expires_in == "number" ? p.expires_in : void 0
    };
  }, e.getToken = async (o) => {
    if (typeof o != "string" || !o)
      return console.error(
        "CastClient.getToken: code is required (call authenticate() first)."
      ), false;
    try {
      const l = new URL(t.hub.token_endpoint);
      console.debug(
        "CastClient: Exchanging code at:",
        `${l.origin}${l.pathname}`
      );
    } catch {
      console.debug("CastClient: Exchanging code at hub");
    }
    const b = t.session.productName || t.config.productName || "CAST", u = new URLSearchParams();
    u.append("grant_type", "authorization_code"), u.append("code", o), u.append("client_id", t.hub.client_id || ""), u.append("client_secret", t.hub.client_secret || ""), u.append("client_product_name", b);
    try {
      const l = await fetch(t.hub.token_endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: u
      });
      if (l.status === 200) {
        const p = await l.json();
        return typeof p.access_token == "string" && p.access_token && (t.hub.token = p.access_token), typeof p.id_token == "string" && p.id_token && (t.hub.lastIdToken = p.id_token), p.topic && typeof p.topic == "string" && (t.config.preserveSessionTopicFromToken || e.setTopic(p.topic), t.config.autoStart && e.subscribe()), !!t.hub.token;
      }
      return await l.text(), console.error(
        "CastClient: Error getting token. Status:",
        l.status
      ), false;
    } catch (l) {
      const p = l instanceof Error ? l.message : String(l);
      return console.error("CastClient: Exception getting token:", p), false;
    }
  }, e.subscribe = async (o = false) => {
    const b = t.session.topic && t.session.topic.trim();
    if (!b)
      return console.warn(
        "CastClient: Error. subscription not sent. No topic defined."
      ), "error: topic not defined";
    if ((!t.session.subscriberName || !String(t.session.subscriberName).trim()) && (t.session.subscriberName = De(
      t.session.productName || t.config.productName || "CAST"
    )), !t.hub.token)
      return console.warn(
        "CastClient: Error. subscription not sent. No token available."
      ), "error: no token";
    const u = dt(t.config.callbackUrl), l = bt(
      "subscribe",
      {
        ...t.session,
        topic: b,
        subscriberName: t.session.subscriberName
      },
      u,
      t.config.productVersion
    ), p = {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        Authorization: `Bearer ${t.hub.token}`
      },
      body: l,
      signal: typeof AbortSignal < "u" && AbortSignal.timeout ? AbortSignal.timeout(wt) : void 0
    };
    try {
      n("connecting");
      const h = await fetch(t.hub.hub_endpoint, p);
      if (h.status === 202) {
        t.hub.resubscribeRequested = false;
        let g;
        try {
          g = await h.json();
        } catch (w) {
          t.hub.subscribed = false;
          const L = w instanceof Error ? w.message : String(w);
          return console.error(
            "CastClient: Subscribe 202 body was not valid JSON:",
            L
          ), n("error"), 0;
        }
        const C = g["hub.channel.endpoint"];
        if (!C || typeof C != "string")
          return t.hub.subscribed = false, console.error(
            "CastClient: Subscribe 202 missing hub.channel.endpoint"
          ), n("error"), 0;
        let y = C;
        try {
          const w = new URL(t.hub.hub_endpoint), L = new URL(C), U = w.protocol === "https:" ? "wss:" : "ws:";
          y = C.replace(
            L.origin,
            `${U}//${w.host}`
          );
        } catch {
        }
        if (t.hub.websocket) {
          try {
            t.hub.websocket.removeEventListener("close", s), t.hub.websocket.close();
          } catch {
          }
          t.hub.websocket = null;
        }
        let E;
        try {
          E = new WebSocket(y);
        } catch (w) {
          t.hub.subscribed = false;
          const L = w instanceof Error ? w.message : String(w);
          return console.error(
            "CastClient: Failed to open WebSocket after subscribe:",
            L
          ), n("error"), 0;
        }
        return t.hub.subscribed = true, t.hub.websocket = E, t.hub.websocket.onopen = function() {
          this.send(
            JSON.stringify({
              "hub.channel.endpoint": y
            })
          ), n("connected");
        }, t.hub.websocket.addEventListener("message", (w) => {
          typeof w.data == "string" && a(w.data);
        }), t.hub.websocket.addEventListener("close", s), t.hub.websocket.onerror = function() {
          console.warn("CastClient: Error reported on websocket"), n("error");
        }, h.status;
      }
      if (h.status === 401) {
        if (console.warn(
          "CastClient: Subscription response 401 - Token refresh needed."
        ), o)
          return h.status;
        try {
          const { code: g } = await e.authenticate();
          if (g && await e.getToken(g))
            return await e.subscribe(true);
        } catch (g) {
          const C = g instanceof Error ? g.message : String(g);
          console.error(
            "CastClient: Token refresh after 401 failed:",
            C
          );
        }
      } else
        console.error(
          "CastClient: Subscription rejected by hub. Status:",
          h.status
        );
      return h.status;
    } catch (h) {
      t.hub.subscribed = false;
      const g = h instanceof Error ? h.message : String(h);
      return console.error("CastClient: Exception subscribing to the hub:", g), 0;
    }
  }, e.unsubscribe = async () => {
    t.hub.subscribed = false, t.hub.resubscribeRequested = false;
    const o = dt(t.config.callbackUrl), b = bt(
      "unsubscribe",
      t.session,
      o,
      t.config.productVersion
    );
    try {
      (await fetch(t.hub.hub_endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
          Authorization: `Bearer ${t.hub.token}`
        },
        body: b,
        signal: typeof AbortSignal < "u" && AbortSignal.timeout ? AbortSignal.timeout(wt) : void 0
      })).status === 202 && console.debug(
        "CastClient: Unsubscribe successfully from hub",
        t.hub.name
      );
    } catch (u) {
      const l = u instanceof Error ? u.message : String(u);
      console.warn("CastClient: Error unsubscribing from the hub.", l);
    }
    t.hub.websocket && (t.hub.websocket.close(), t.hub.websocket = null), n("disconnected");
  };
  function f(o, b = t.hub) {
    const u = { ...o, timestamp: (/* @__PURE__ */ new Date()).toJSON() };
    u.id = ie(r()), b.lastPublishedMessageID = u.id;
    const l = t.session.subscriberName && t.session.subscriberName.trim(), p = t.session.productName && t.session.productName.trim();
    if (l && u["subscriber.name"] === void 0 && (u["subscriber.name"] = l), p && u["subscriber.product.name"] === void 0 && (u["subscriber.product.name"] = p), u.event && !u.event["hub.topic"] && (u.event["hub.topic"] = t.session.topic), u["target.actor"] === void 0 && t.session.defaultTargetActor && !String(
      u["target.subscriber.name"] || ""
    ).trim()) {
      const g = se(
        t.session.defaultTargetActor
      );
      g && (u["target.actor"] = g);
    }
    return u;
  }
  e.fetchPayload = (o) => c.fetchPayload(o), e.fetchAllPayloads = (o) => c.fetchAllPayloads(o), e.hasPendingPayload = (o) => c.hasPendingPayload(o), e.publishMultipart = async (o, b, u = t.hub) => {
    let l = f(o, u);
    const p = await ae(b);
    l = Ae(l);
    let h = await ft(l);
    if (!h.length) {
      const C = (l.event && typeof l.event["hub.event"] == "string" ? l.event["hub.event"] : "") === "dicom-send" ? "dicom-send.dcm" : "nifti-send.nii.gz";
      l.event.context = {
        files: [
          {
            data: p,
            fileName: wn(l, C),
            mimeType: "application/octet-stream",
            byteLength: p.byteLength
          }
        ]
      }, h = [p];
    }
    return e.publishBinaryBatch(l, h, u);
  }, e.publishBinaryBatch = async (o, b, u = t.hub) => {
    const l = Ae(
      f(o, u)
    );
    return Ln({
      msg: l,
      fileBytesList: b,
      hub: u,
      messageIdPrefix: r
    });
  }, e.publish = async (o, b = t.hub) => {
    let u = f(o, b);
    if (u = Ae(u), Tn(u)) {
      const l = await ft(u);
      if (l.length)
        return e.publishBinaryBatch(u, l, b);
    }
    try {
      return await fetch(b.hub_endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${b.token}`
        },
        body: JSON.stringify(u)
      });
    } catch (l) {
      const p = l instanceof Error ? l.message : String(l);
      return console.debug("CastClient:", p), null;
    }
  }, e.publishNiftiMultipart = async (o, b, u) => e.publishBinaryBatch(o, [b], u), e.request = async (o = {}) => {
    const b = String(o["subscriber.name"] || "").trim();
    if (!b)
      throw new Error('CastClient.request: "subscriber.name" is required.');
    const u = t.hub, l = u.token && u.token.trim();
    if (!l)
      throw new Error(
        "CastClient.request: token is required (call authenticate() then getToken(code) first)."
      );
    const p = `${(u.hub_endpoint || "").replace(
      /\/+$/,
      ""
    )}/request`, h = {
      "subscriber.name": b,
      id: o.id && String(o.id).trim() || ie(r()),
      timestamp: o.timestamp && String(o.timestamp).trim() || (/* @__PURE__ */ new Date()).toJSON()
    };
    if (o.event && typeof o.event == "object") {
      h.event = { ...o.event };
      const x = t.session.topic && String(t.session.topic).trim();
      x && !h.event["hub.topic"] && (h.event["hub.topic"] = x);
    } else
      throw new Error(
        'CastClient.request: "event" with hub.event is required.'
      );
    const g = h.event["hub.event"];
    if (!g || !String(g).trim())
      throw new Error(
        'CastClient.request: event["hub.event"] must be a *-request event name.'
      );
    o["subscriber.actor"] && String(o["subscriber.actor"]).trim() && (h["subscriber.actor"] = String(o["subscriber.actor"]).trim());
    const C = Object.prototype.hasOwnProperty.call(
      o,
      "target.actor"
    );
    let y = C ? se(o["target.actor"]) : void 0;
    y === void 0 && !C && t.session.defaultTargetActor && (y = se(t.session.defaultTargetActor)), y !== void 0 && (h["target.actor"] = y);
    const E = o["target.product.name"] !== void 0 ? o["target.product.name"] : o.targetProductName, w = Rt(E);
    w !== void 0 && (h["target.product.name"] = w);
    const L = await fetch(p, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${l}`
      },
      body: JSON.stringify(h)
    });
    let U;
    if ((L.headers.get("content-type") || "").includes("application/json"))
      try {
        U = await L.json();
      } catch {
        U = "";
      }
    else
      U = await L.text();
    return { ok: L.ok, status: L.status, data: U };
  }, e.sendCastRequestResponse = (o, b, u, l) => {
    if (!t.hub.websocket || typeof WebSocket > "u" || t.hub.websocket.readyState !== WebSocket.OPEN)
      return;
    let p = "";
    if (typeof b == "string" ? p = b.trim() : b != null && (p = String(b).trim()), !p) {
      console.error(
        "CastClient.sendCastRequestResponse requires a non-empty dataType."
      );
      return;
    }
    const h = It(p), g = {
      timestamp: (/* @__PURE__ */ new Date()).toJSON(),
      id: ie(r()),
      "subscriber.name": t.session.subscriberName || void 0,
      "subscriber.product.name": t.session.productName || void 0,
      event: {
        "hub.topic": l || t.session.topic,
        "hub.event": h,
        context: {
          id: o,
          dataType: p,
          data: u
        }
      }
    };
    Array.isArray(t.session.actors) && t.session.actors.length > 0 && (g.actor = t.session.actors[0]), t.hub.websocket.send(JSON.stringify(g));
  }, t.config.autoReconnect && (t.reconnectInterval = setInterval(
    d,
    gi
  ));
}
function un(e, t, n = {}) {
  Object.assign(t, Et, n), t.config = { ...Et.config, ...n }, t.session = {
    ...Bn(),
    ...t.config.session || {}
  }, (!t.session.subscriberName || !String(t.session.subscriberName).trim()) && (t.session.subscriberName = De(
    t.session.productName || t.config.productName || "CAST"
  )), t.hub = {
    ...Pn(),
    ...t.config.hub || {},
    ...Fn()
  }, hn(e, t), pn(e, t, ["config"]), Si(e, t);
}
var Ci = yn(un, "vtkCastClient");
var wi = {
  newInstance: Ci,
  extend: un,
  applyCastPublishEnvelopeFields: $r,
  batchContextFiles: Jt,
  BINARY_PLACEHOLDER: oe,
  buildDicomwebImagingStudyOpenContext: or,
  buildDicomUrlImagingStudyOpenContext: sr,
  buildFilesImagingStudyOpenContext: Gt,
  buildIdcImagingStudyOpenContext: cr,
  buildNiftiUrlImagingStudyOpenContext: ar,
  castMessageHasPendingFilePayloads: Zt,
  buildCastConferenceView: Ke,
  CAST_CONFERENCE_EXIT_ACK_MS: Ot,
  CAST_CONFERENCE_POLL_MS: Xn,
  CAST_CONFERENCE_POPUP_SIZE: Kn,
  CAST_CONFERENCE_TITLE_PRESETS: xt,
  CAST_DEFAULT_SUBSCRIBER_ACTOR: Ce,
  createCastConferenceDialog: Qn,
  CAST_IMAGE_DISPLAY_ACTOR: Ue,
  createCastConferenceInviteController: tr,
  createCastConferenceInviteDialog: Mt,
  parseConferenceInviteFromMessage: Dt,
  shouldShowConferenceInvite: er,
  CAST_DICOMWEB_ROOT: Ze,
  CAST_ENVELOPE_ANY: R,
  CAST_IDENTIFIER_DICOM_UID: de,
  CAST_IDENTIFIER_IDC: ye,
  CAST_IDENTIFIER_IDC_SOURCE_BUCKET: ge,
  CAST_IDENTIFIER_NIFTI_FILENAME: Xe,
  CAST_IDENTIFIER_NIFTI_URL: Ye,
  CAST_IDENTIFIER_OHIF_MODE: he,
  CAST_IDENTIFIER_VOLVIEW_SAMPLE_ID: be,
  CAST_IDENTIFIER_WORKLIST_SAMPLE_ID: ee,
  CAST_IMAGING_STUDY_OPEN_PROFILE: fe,
  CAST_OPEN_MODE: te,
  CAST_OPEN_MODE_DICOMWEB: K,
  CAST_OPEN_MODE_DICOM_URL: pe,
  CAST_OPEN_MODE_FILES: F,
  CAST_OPEN_MODE_IDC: me,
  CAST_RADIO_ICON_CLASS: _t,
  CAST_RADIO_ICON_SVG: kt,
  CAST_RADIO_SLASH_CLASS: $t,
  CAST_RADIO_STATUS_ICON_CLASS: Ut,
  castRadioStatusMarkup: On,
  castRadioStatusVisual: xn,
  collatedResponsesFromRequestResult: In,
  conferenceHostTopic: Q,
  countEventContextObjects: tt,
  createCastConference: Bt,
  dataTypeFromEventName: $n,
  decodeBase64ToArrayBuffer: Vt,
  DEFAULT_CAST_PUBLISH_ENVELOPE_FIELDS: kr,
  deleteCastConference: Je,
  ensureCastSubscribeEvents: Rn,
  extractDicomSeriesUid: z,
  extractDicomStudyUid: H,
  extractDicomwebRoot: $e,
  extractFilePayloadsForEvent: Nr,
  extractIdcSeriesUid: Re,
  extractIdcSourceBucket: Oe,
  extractIdentifierValue: _,
  extractImagingStudyFiles: D,
  extractNiftiDownloadUrl: Y,
  extractNiftiFilename: Qe,
  extractOhifMode: jt,
  extractOpenMode: zt,
  extractStudyContextItem: M,
  extractStudyUIDFromResource: lr,
  extractVolviewSampleId: qt,
  fetchCastConferences: J,
  fetchCastConferenceTopics: Pt,
  filePayloadToArrayBuffer: Yt,
  filePayloadToFile: Ar,
  findActiveCastConference: V,
  generateSubscriberName: De,
  getActorKeyword: Wt,
  getHubEventLower: ne,
  getInboundTargetActorKeyword: fr,
  httpUrlFromHubEndpoint: Z,
  isCastConferenceHost: We,
  isCastConferenceParticipant: Ve,
  isHubEndpointInCloud: Lt,
  isRequestEvent: kn,
  isResponseEvent: Un,
  isRunningInCloud: vt,
  isStatusPayloadOnline: it,
  isStatusRequestDataType: Fr,
  isTotalSegmentatorProduct: we,
  isLungScreeningProduct: Ee,
  isNeuroSegProduct: Te,
  isInferenceProduct: Or,
  lungScreeningAvailableFromStatusResponses: Pr,
  neuroSegAvailableFromStatusResponses: Br,
  normalizeProductToken: Be,
  CAST_INFERENCE_SERVERS: O,
  findInferenceServerByProduct: nn,
  inferenceServerInfoText: di,
  renderInferenceServerInfoCardHtml: fi,
  CAST_PRODUCT_MATCHERS: sn,
  matchesCastProduct: bi,
  normalizeCastProductName: rn,
  parseProductStatusProbe: cn,
  publishCastUrlSend: yi,
  requestCastProductStatus: hi,
  requestCastStatus: ot,
  mapInferenceServersFromStatusResult: an,
  probeCastInferenceServers: mi,
  cloneContextArray: dr,
  messageEventContext: br,
  messageSubscriberName: hr,
  messageProductName: pr,
  messageActor: mr,
  parseStatusUpdateMessage: yr,
  normalizeConferenceParticipants: Ge,
  normalizeDataType: qe,
  normalizeImagingStudyContext: et,
  normalizeOptionalEnvelopeField: X,
  normalizeStudyUID: xe,
  openCastHubPopup: Yn,
  parseCollatedRequestResult: le,
  productNameFromStatusResponseItem: Ne,
  REQUEST_SUFFIX: G,
  requestEventFor: ze,
  resolveCastConferenceClientUrl: Jn,
  resolveCastConferenceState: Zn,
  resolveCastConferenceView: Ft,
  resolveCastFileMessage: Rr,
  resolveCastHubAdminUrl: Vn,
  resolveCastPublishEnvelopeFields: Ur,
  resolveImagingStudyOpenPlan: ur,
  resolveTargetActorForWire: se,
  resolveTargetProductNameForWire: Rt,
  responseEventFor: It,
  RESPONSE_SUFFIX: W,
  sanitizeCastMessageForDisplay: _r,
  selectFirstMatchingHubKey: _n,
  statusItemValue: rt,
  stringifyForLog: Pe,
  summarizeInboundCastMessage: Ir,
  summarizeOutboundCastPublish: Lr,
  TOTAL_SEGMENTATOR_PRODUCT_ALIASES: Qt,
  LUNG_SCREENING_PRODUCT_ALIASES: en,
  NEURO_SEG_PRODUCT_ALIASES: tn,
  totalSegmentatorAvailableFromStatusResponses: xr
};

// render/demos/ira/ira-cast.ts
var LOG = "[ira-cast]";
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
function isHostedOnCastHub(location2 = globalThis.location) {
  return String(location2?.pathname || "").startsWith("/slicerlive");
}
function shouldUseSameOriginHub(location2 = globalThis.location) {
  return isHostedOnCastHub(location2) || vt(location2);
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
var EMPTY_CAST_CONTEXT = { "context.type": "", context: [] };
var ID_ACTOR = "ID";
var WORKLIST_ACTOR = "WORKLIST_CLIENT";
var SS_TOPIC = "ira.cast.topic";
var SS_TOKEN = "ira.cast.token";
var STUDY_URL_KEYS = [
  "series",
  "bucket",
  "seg",
  "segBucket",
  "patient",
  "modality",
  "collection",
  "scene",
  "StudyInstanceUIDs",
  "SeriesInstanceUIDs",
  "studyUID",
  "seriesUID",
  "requestType"
];
var client = null;
var topic = "";
var subscriberName = "";
var userName = "";
var connection = "idle";
var connectionDetail = "Not connected";
var conference = null;
var conferenceFollowHost = false;
var lastImagingStudyOpenContext = [];
var lastLiveScene = null;
var statusRequestState = "idle";
var onStatus = null;
var onImagingStudyOpen = null;
var onImagingStudyClose = null;
var onLiveSceneChange = null;
function setIraCastStatusListener(fn) {
  onStatus = fn;
}
function setIraImagingStudyOpenHandler(fn) {
  onImagingStudyOpen = fn;
}
function setIraImagingStudyCloseHandler(fn) {
  onImagingStudyClose = fn;
}
function persistCastCredentials(nextTopic, token) {
  try {
    if (nextTopic) sessionStorage.setItem(SS_TOPIC, nextTopic);
    if (token) sessionStorage.setItem(SS_TOKEN, token);
  } catch {
  }
}
function readPersistedCredentials() {
  try {
    return {
      topic: sessionStorage.getItem(SS_TOPIC)?.trim() || "",
      token: sessionStorage.getItem(SS_TOKEN)?.trim() || ""
    };
  } catch {
    return { topic: "", token: "" };
  }
}
function iraUrlHasStudyParams() {
  const p = new URLSearchParams(globalThis.location.search);
  return Boolean(
    p.get("series") || p.get("StudyInstanceUIDs") || p.get("studyUID") || p.get("scene")
  );
}
function syncIraStudyUrl(studyParams = {}) {
  if (!globalThis.history?.replaceState) return;
  const url = new URL(globalThis.location.href);
  for (const key of STUDY_URL_KEYS) url.searchParams.delete(key);
  const persisted = readPersistedCredentials();
  const nextTopic = topic || persisted.topic;
  const nextToken = persisted.token || String(client?.getHubConfig?.()?.token || "").trim();
  if (nextTopic) url.searchParams.set("topic", nextTopic);
  if (nextToken) url.searchParams.set("id-token", nextToken);
  for (const [key, value] of Object.entries(studyParams)) {
    const v2 = String(value || "").trim();
    if (v2) url.searchParams.set(key, v2);
  }
  globalThis.history.replaceState(null, "", url.toString());
}
function getIraCastSessionInfo() {
  return {
    topic,
    subscriberName: subscriberName || client?.getSessionConfig?.()?.subscriberName || "",
    userName: userName || client?.getSessionConfig?.()?.userName || "",
    connection,
    connectionDetail,
    hubEndpoint: client?.getHubConfig?.()?.hub_endpoint || resolveHubConfig().hub_endpoint,
    conference
  };
}
function setIraConferenceView(view) {
  conference = view;
  if (!view) {
    conferenceFollowHost = false;
  }
  onStatus?.(connection, connectionDetail);
}
function setIraConferenceFollowHost(followHost) {
  conferenceFollowHost = Boolean(followHost);
}
var conferenceHubHandler = null;
function setIraConferenceHubHandler(fn) {
  conferenceHubHandler = fn;
}
async function syncIraConferenceView() {
  if (connection !== "connected") {
    setIraConferenceView(null);
    return;
  }
  try {
    const hubEndpoint = client?.getHubConfig?.()?.hub_endpoint || resolveHubConfig().hub_endpoint;
    const view = await Ft(
      hubEndpoint,
      topic,
      subscriberName,
      userName
    );
    setIraConferenceView(view);
  } catch {
  }
}
function handleConferenceHubEvent(message) {
  if (conferenceHubHandler?.(message)) {
    return true;
  }
  const event = message?.event;
  const hubEvent = ne(event);
  if (hubEvent === "conference-start") {
    void syncIraConferenceView();
    return true;
  }
  if (hubEvent === "conference-end") {
    const context = event?.context && typeof event.context === "object" ? event.context : {};
    const leaveTopic = String(context.leaveTopic ?? "").trim();
    if (!leaveTopic || leaveTopic === topic) {
      setIraConferenceView(null);
    } else {
      void syncIraConferenceView();
    }
    return true;
  }
  return false;
}
function getIraHubAdminUrl() {
  const endpoint = client?.getHubConfig?.()?.hub_endpoint || resolveHubConfig().hub_endpoint;
  return Vn(endpoint) || endpoint;
}
function setIraImagingStudyContext(context) {
  lastImagingStudyOpenContext = Array.isArray(context) ? context : [];
}
function getIraImagingStudyOpenContext() {
  return dr(lastImagingStudyOpenContext);
}
var suppressInboundImagingStudy = 0;
function beginSuppressInboundImagingStudy(ms = 1500) {
  suppressInboundImagingStudy += 1;
  globalThis.setTimeout(() => {
    suppressInboundImagingStudy = Math.max(0, suppressInboundImagingStudy - 1);
  }, ms);
}
function buildIraPublishPayload(hubEvent, context) {
  return {
    event: {
      "hub.topic": topic,
      "hub.event": hubEvent,
      context
    },
    actor: ID_ACTOR,
    "target.actor": "*",
    "target.product.name": "*"
  };
}
async function publishIraImagingStudyTransition(nextContext) {
  const next = Array.isArray(nextContext) ? dr(nextContext) : [];
  const previous = dr(lastImagingStudyOpenContext);
  if (!client || connection !== "connected" || !topic) {
    lastImagingStudyOpenContext = next;
    lastLiveScene = null;
    onLiveSceneChange?.();
    console.warn(`${LOG} ImagingStudy transition local-only (not connected)`);
    return { closed: false, opened: false };
  }
  beginSuppressInboundImagingStudy();
  let closed = false;
  let opened = false;
  try {
    if (previous.length) {
      try {
        const res = await client.publish(
          buildIraPublishPayload("ImagingStudy-close", previous)
        );
        closed = Boolean(res?.ok);
        if (!closed) {
          console.warn(
            `${LOG} ImagingStudy-close failed`,
            res ? `HTTP ${res.status}` : "No response"
          );
        } else {
          console.info(`${LOG} published ImagingStudy-close`);
        }
      } catch (err) {
        console.error(`${LOG} ImagingStudy-close exception`, err);
      }
    }
    try {
      const res = await client.publish(
        buildIraPublishPayload("ImagingStudy-open", next)
      );
      opened = Boolean(res?.ok);
      if (!opened) {
        console.warn(
          `${LOG} ImagingStudy-open failed`,
          res ? `HTTP ${res.status}` : "No response"
        );
      } else {
        console.info(`${LOG} published ImagingStudy-open`);
      }
    } catch (err) {
      console.error(`${LOG} ImagingStudy-open exception`, err);
    }
  } finally {
    lastImagingStudyOpenContext = next;
    lastLiveScene = null;
    onLiveSceneChange?.();
  }
  return { closed, opened };
}
function getIraCastClient() {
  return client;
}
var onStatusUpdate = null;
function setIraStatusUpdateHandler(fn) {
  onStatusUpdate = fn;
}
var onNiftiSend = null;
function setIraNiftiSendHandler(fn) {
  onNiftiSend = fn;
}
var onDicomSend = null;
function setIraDicomSendHandler(fn) {
  onDicomSend = fn;
}
async function publishInferenceJob(opts) {
  if (!client || connection !== "connected") {
    return { ok: false, detail: "Not connected" };
  }
  const subscriber = subscriberName || client.getSessionConfig()?.subscriberName?.trim() || "";
  if (!subscriber) return { ok: false, detail: "No subscriber name" };
  if (!topic) return { ok: false, detail: "No Cast topic" };
  if (!opts.files.length) return { ok: false, detail: "No volume files" };
  return yi(client, {
    targetProductName: opts.product,
    hubEvent: opts.hubEvent,
    files: opts.files,
    subscriberName: subscriber,
    subscriberProductName: "SLICERLIVE-IRA",
    subscriberActor: ID_ACTOR,
    topic,
    contextExtras: opts.contextExtras
  });
}
function getIraLiveScene() {
  return lastLiveScene;
}
function setIraLiveScene(scene) {
  lastLiveScene = scene;
  onLiveSceneChange?.();
}
function setIraLiveSceneChangeListener(fn) {
  onLiveSceneChange = fn;
}
function buildStatusResponseData() {
  const context = dr(lastImagingStudyOpenContext);
  if (lastLiveScene) {
    const withoutScene = context.filter((item) => {
      if (!item || typeof item !== "object") return true;
      return item.key !== "scene";
    });
    withoutScene.push({ key: "scene", resource: lastLiveScene });
    return { "context.type": "ImagingStudy", context: withoutScene };
  }
  if (context.length) {
    return { "context.type": "ImagingStudy", context };
  }
  return EMPTY_CAST_CONTEXT;
}
function setConnection(next, detail) {
  const wasConnected = connection === "connected";
  connection = next;
  connectionDetail = detail;
  if (wasConnected && next !== "connected") {
    conference = null;
  }
  onStatus?.(next, detail);
  if (next === "connected") {
    void syncIraConferenceView();
  }
}
function stripCastParamsFromUrl() {
  if (!globalThis.history?.replaceState) return;
  const url = new URL(globalThis.location.href);
  if (!url.searchParams.has("topic") && !url.searchParams.has("id-token")) {
    return;
  }
  url.searchParams.delete("topic");
  url.searchParams.delete("id-token");
  globalThis.history.replaceState(null, "", url.toString());
}
function extractActorKeywords(actor) {
  if (actor == null) return [];
  if (typeof actor === "string") {
    const keyword2 = actor.trim();
    return keyword2 ? [keyword2] : [];
  }
  if (Array.isArray(actor)) {
    return actor.flatMap((entry) => extractActorKeywords(entry));
  }
  if (typeof actor === "object") {
    const obj = actor;
    const candidate = obj.id || obj.key || obj.name;
    if (candidate == null) return [];
    const keyword2 = String(candidate).trim();
    return keyword2 ? [keyword2] : [];
  }
  const keyword = String(actor).trim();
  return keyword ? [keyword] : [];
}
function handleStatusRequest(message) {
  const event = message?.event;
  if (!event || !kn(ne(event))) return false;
  const context = event.context && typeof event.context === "object" ? event.context : {};
  const correlationId = context.id;
  if (typeof correlationId !== "string" || !correlationId) return false;
  const requestedTargets = extractActorKeywords(message["target.actor"]);
  if (requestedTargets.length > 0 && !requestedTargets.includes("*") && !requestedTargets.includes(ID_ACTOR)) {
    return false;
  }
  if (context.dataType !== "STATUS") return false;
  const responseData = buildStatusResponseData();
  client?.sendCastRequestResponse(
    correlationId,
    "STATUS",
    responseData,
    event["hub.topic"]
  );
  return true;
}
function handleStatusUpdate(message) {
  const update = yr(message);
  if (!update) return false;
  if (update.message) onStatusUpdate?.(update.message, update.level);
  return true;
}
function isNiftiSendMessage(message) {
  const event = message?.event;
  return ne(event) === "nifti-send";
}
function isDicomSendMessage(message) {
  const event = message?.event;
  return ne(event) === "dicom-send";
}
async function handleResolvedFileSend(message, kind, handler) {
  if (!client || !handler) return;
  try {
    const resolved = await Rr(client, message);
    const event = resolved?.event ?? message.event;
    if (!event || typeof event !== "object") {
      console.warn(`${LOG} ${kind}: missing event after resolve`);
      return;
    }
    const files = Jt(event);
    const withData = files.filter((f) => {
      const data = f.data;
      return data instanceof ArrayBuffer || data instanceof Uint8Array || typeof data === "string";
    });
    if (!withData.length) {
      console.warn(`${LOG} ${kind}: no file payloads after resolve`);
      return;
    }
    console.info(
      `${LOG} ${kind}: ${withData.length} file(s) ready`,
      withData.map((f) => f.fileName || "?")
    );
    await handler(event);
  } catch (err) {
    console.error(`${LOG} ${kind} handler failed`, err);
  }
}
async function handleNiftiSendMessage(message) {
  await handleResolvedFileSend(message, "nifti-send", onNiftiSend);
}
async function handleDicomSendMessage(message) {
  await handleResolvedFileSend(message, "dicom-send", onDicomSend);
}
function handleImagingStudyMessage(message) {
  const event = message?.event;
  const name = ne(event);
  if (name === "imagingstudy-close") {
    if (suppressInboundImagingStudy > 0) {
      console.info(`${LOG} ImagingStudy-close ignored (local publish echo)`);
      return true;
    }
    lastImagingStudyOpenContext = [];
    lastLiveScene = null;
    onLiveSceneChange?.();
    console.info(`${LOG} ImagingStudy-close`);
    void Promise.resolve(onImagingStudyClose?.()).catch(
      (err) => console.error(`${LOG} ImagingStudy-close handler failed`, err)
    );
    return true;
  }
  if (name !== "imagingstudy-open") return false;
  if (suppressInboundImagingStudy > 0) {
    const context2 = br(message);
    lastImagingStudyOpenContext = dr(context2);
    console.info(`${LOG} ImagingStudy-open ignored (local publish echo)`);
    return true;
  }
  if (conference && !conferenceFollowHost) {
    console.info(`${LOG} ImagingStudy-open ignored (not following host)`);
    return true;
  }
  const context = br(message);
  lastImagingStudyOpenContext = dr(context);
  console.info(`${LOG} ImagingStudy-open context items=`, context.length);
  void Promise.resolve(onImagingStudyOpen?.(lastImagingStudyOpenContext)).catch(
    (err) => console.error(`${LOG} ImagingStudy-open handler failed`, err)
  );
  return true;
}
async function requestStatusFromWorklist(opts) {
  if (!client) {
    console.warn(`${LOG} STATUS skip: no client`);
    return;
  }
  const forceApply = Boolean(opts?.forceApply);
  const requestTopic = String(opts?.topic || topic || "").trim();
  if (!forceApply && statusRequestState !== "idle") return;
  const skipApplyStudy = !forceApply && (iraUrlHasStudyParams() || lastImagingStudyOpenContext.length > 0);
  if (skipApplyStudy) {
    console.warn(
      `${LOG} STATUS presence only (already have study URL or context)`
    );
  }
  const subscriber = subscriberName || client.getSessionConfig()?.subscriberName?.trim() || "";
  if (!subscriber) {
    console.warn(`${LOG} STATUS skip: no subscriber name yet`);
    return;
  }
  if (!forceApply) statusRequestState = "pending";
  const statusDataType = "STATUS";
  try {
    const message = {
      "subscriber.name": subscriber,
      "subscriber.product.name": "SLICERLIVE-IRA",
      "subscriber.actor": ID_ACTOR,
      "target.actor": WORKLIST_ACTOR,
      event: {
        "hub.event": ze(statusDataType),
        ...requestTopic ? { "hub.topic": requestTopic } : {},
        context: { dataType: statusDataType }
      }
    };
    console.warn(`${LOG} STATUS request \u2192 ${WORKLIST_ACTOR}`, {
      subscriber,
      topic: requestTopic,
      presenceOnly: skipApplyStudy,
      forceApply
    });
    const result = await client.request(message);
    console.warn(`${LOG} STATUS response`, {
      httpOk: result.ok,
      status: result.status,
      data: result.data
    });
    if (!result.ok) {
      if (!forceApply) statusRequestState = "idle";
      return;
    }
    if (!forceApply && (skipApplyStudy || lastImagingStudyOpenContext.length || iraUrlHasStudyParams())) {
      statusRequestState = "done";
      return;
    }
    const envelope = result.data && typeof result.data === "object" ? result.data : {};
    const responses = Array.isArray(envelope.responses) ? envelope.responses : [];
    let chosen;
    for (const item of responses) {
      const itemData = item?.data;
      if (!itemData || typeof itemData !== "object" || itemData["context.type"] !== "ImagingStudy") {
        continue;
      }
      const actor = String(item?.actor ?? "").trim().toUpperCase();
      if (actor === WORKLIST_ACTOR || chosen === void 0) {
        chosen = itemData;
        if (actor === WORKLIST_ACTOR) break;
      }
    }
    if (!chosen) {
      console.warn(
        `${LOG} STATUS: no ImagingStudy in ${responses.length} response(s)`
      );
      if (!forceApply) statusRequestState = "idle";
      return;
    }
    if (!forceApply && (lastImagingStudyOpenContext.length || iraUrlHasStudyParams())) {
      statusRequestState = "done";
      return;
    }
    const context = dr(chosen.context);
    if (!context.length) {
      console.warn(`${LOG} STATUS: ImagingStudy context empty`);
      if (!forceApply) statusRequestState = "idle";
      return;
    }
    lastImagingStudyOpenContext = context;
    if (!forceApply) statusRequestState = "done";
    console.warn(
      `${LOG} ImagingStudy from STATUS context items=`,
      context.length,
      forceApply ? "(follow host)" : ""
    );
    void Promise.resolve(onImagingStudyOpen?.(lastImagingStudyOpenContext)).catch(
      (err) => console.error(`${LOG} STATUS ImagingStudy-open handler failed`, err)
    );
  } catch (err) {
    if (!forceApply) statusRequestState = "idle";
    console.error(`${LOG} failed to request STATUS`, err);
  }
}
async function requestIraStatusFromHostWorklist(hostTopic) {
  const topicOverride = String(hostTopic || "").trim();
  if (!topicOverride) {
    console.warn(`${LOG} host STATUS skip: no host topic`);
    return;
  }
  await requestStatusFromWorklist({ topic: topicOverride, forceApply: true });
}
function scheduleStatusRequestFromWorklist(delayMs) {
  globalThis.setTimeout(() => {
    void requestStatusFromWorklist();
  }, delayMs);
}
function applyIraCastRadioVisual(statusEl) {
  const visual = xn(connection, {
    conferenceActive: Boolean(conference)
  });
  statusEl.classList.remove(
    "ira-cast-idle",
    "ira-cast-connecting",
    "ira-cast-connected",
    "ira-cast-disconnected",
    "ira-cast-error",
    "ira-cast-pulse",
    "ira-cast-conference-pulse"
  );
  statusEl.classList.add(`ira-cast-${visual.tone}`);
  statusEl.classList.toggle("ira-cast-pulse", visual.pulse);
  statusEl.classList.toggle("ira-cast-conference-pulse", visual.conferencePulse);
  statusEl.title = conference?.title ? `${connectionDetail} \xB7 ${conference.title}` : connectionDetail;
  const slash = statusEl.querySelector(".cast-radio-slash");
  if (slash) slash.hidden = !visual.showSlash;
}
async function connectIraCast() {
  const params = new URLSearchParams(globalThis.location.search);
  topic = params.get("topic")?.trim() || "";
  let urlToken = params.get("id-token")?.trim() || "";
  if (!topic && !urlToken) {
    setConnection("idle", "Not connected");
    return;
  }
  stripCastParamsFromUrl();
  setConnection("connecting", "Connecting to Cast hub\u2026");
  subscriberName = De("IRA");
  const hub = resolveHubConfig();
  client = wi.newInstance({
    hub,
    session: {
      subscriberName,
      productName: "SLICERLIVE-IRA",
      productVersion: "0.1",
      actors: [ID_ACTOR],
      topic: topic || void 0,
      events: [
        "imagingstudy-open",
        "imagingstudy-close",
        "status-request",
        "subscription-removed",
        "status-update",
        "conference-start",
        "conference-end",
        "dicom-send",
        "nifti-send"
      ],
      lease: 7200,
      defaultTargetActor: "*"
    },
    callbackUrl: `${globalThis.location.origin}/castCallback`,
    preserveSessionTopicFromToken: Boolean(topic),
    autoReconnect: true
  });
  client.onMessage((message) => {
    if (handleStatusRequest(message)) return;
    if (handleConferenceHubEvent(message)) return;
    if (handleStatusUpdate(message)) return;
    if (isDicomSendMessage(message)) {
      void handleDicomSendMessage(message);
      return;
    }
    if (isNiftiSendMessage(message)) {
      void handleNiftiSendMessage(message);
      return;
    }
    handleImagingStudyMessage(message);
  });
  client.onConnectionStateChange((wsState) => {
    console.warn(`${LOG} ws state=`, wsState);
    if (wsState === "connecting") {
      setConnection("connecting", "Websocket connecting");
    } else if (wsState === "connected") {
      setConnection("connected", "Websocket connected");
      scheduleStatusRequestFromWorklist(150);
    } else if (wsState === "error") {
      statusRequestState = "idle";
      setConnection("error", "Websocket error");
    } else {
      statusRequestState = "idle";
      setConnection("disconnected", "Websocket disconnected");
    }
  });
  try {
    if (urlToken) {
      client.setToken(urlToken);
    } else {
      const { code: code2 } = await client.authenticate();
      if (!code2 || !await client.getToken(code2)) {
        setConnection("error", "Token failed");
        return;
      }
    }
    if (topic) client.setTopic(topic);
    const hubTokenEarly = String(
      client.getHubConfig?.()?.token || urlToken || ""
    ).trim();
    if (hubTokenEarly) urlToken = hubTokenEarly;
    persistCastCredentials(topic, urlToken);
    const status2 = await client.subscribe();
    const session = client.getSessionConfig();
    if (session.subscriberName) subscriberName = session.subscriberName;
    if (session.topic) topic = session.topic;
    userName = String(session.userName || "").trim() || topic;
    const hubToken = String(client.getHubConfig?.()?.token || urlToken || "").trim();
    if (hubToken) urlToken = hubToken;
    persistCastCredentials(topic, urlToken);
    if (status2 === 202) {
      console.warn(
        `${LOG} subscribed topic=`,
        topic,
        "subscriber=",
        subscriberName
      );
      scheduleStatusRequestFromWorklist(400);
    } else {
      setConnection("disconnected", `Subscribe failed (${String(status2)})`);
    }
  } catch (err) {
    console.error(`${LOG} connect failed`, err);
    setConnection(
      "error",
      err instanceof Error ? err.message : String(err)
    );
  }
}

// logic/readers/nifti.ts
var DT = {
  2: { ctor: Uint8Array, bytes: 1, zarr: "|u1" },
  4: { ctor: Int16Array, bytes: 2, zarr: "<i2" },
  8: { ctor: Int32Array, bytes: 4, zarr: "<i4" },
  16: { ctor: Float32Array, bytes: 4, zarr: "<f4" },
  64: { ctor: Float64Array, bytes: 8, zarr: "<f8" },
  256: { ctor: Int8Array, bytes: 1, zarr: "|i1" },
  512: { ctor: Uint16Array, bytes: 2, zarr: "<u2" },
  768: { ctor: Uint32Array, bytes: 4, zarr: "<u4" }
};
function isGzip(bytes) {
  return bytes.length > 2 && bytes[0] === 31 && bytes[1] === 139;
}
async function gunzip(bytes) {
  const ds = new DecompressionStream("gzip");
  return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer());
}
function quaternToMat(b, c, d, qfac, dx, dy, dz) {
  let a = 1 - (b * b + c * c + d * d);
  if (a < 1e-7) {
    const l = Math.hypot(b, c, d);
    b /= l;
    c /= l;
    d /= l;
    a = 0;
  } else a = Math.sqrt(a);
  const R2 = [
    a * a + b * b - c * c - d * d,
    2 * b * c - 2 * a * d,
    2 * b * d + 2 * a * c,
    2 * b * c + 2 * a * d,
    a * a + c * c - b * b - d * d,
    2 * c * d - 2 * a * b,
    2 * b * d - 2 * a * c,
    2 * c * d + 2 * a * b,
    a * a + d * d - c * c - b * b
  ];
  const s = [dx, dy, dz * (qfac < 0 ? -1 : 1)];
  return R2.map((v2, i) => v2 * s[i % 3]);
}
async function parseNifti(input, name) {
  const bytes = isGzip(input) ? await gunzip(input) : input;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length < 348) throw new Error("not a NIfTI file (too short)");
  const sizeof = dv.getInt32(0, true);
  const isN2 = String.fromCharCode(bytes[4], bytes[5], bytes[6]).startsWith("n") && (sizeof === 540 || dv.getInt32(0, false) === 540);
  let little = true;
  let dims, datatype, pixdim, voxOffset, qformCode, sformCode;
  let quat, qoff, srow, sclSlope, sclInter;
  if (!isN2) {
    little = sizeof === 348;
    if (!little && dv.getInt32(0, false) !== 348) throw new Error("bad NIfTI-1 sizeof_hdr");
    const i16 = (o) => dv.getInt16(o, little), f32 = (o) => dv.getFloat32(o, little);
    dims = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => i16(40 + i * 2));
    datatype = i16(70);
    pixdim = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => f32(76 + i * 4));
    voxOffset = f32(108);
    sclSlope = f32(112);
    sclInter = f32(116);
    qformCode = i16(252);
    sformCode = i16(254);
    quat = [f32(256), f32(260), f32(264)];
    qoff = [f32(268), f32(272), f32(276)];
    srow = Array.from({ length: 12 }, (_2, i) => f32(280 + i * 4));
  } else {
    little = dv.getInt32(0, true) === 540;
    const i64 = (o) => Number(dv.getBigInt64(o, little)), f64 = (o) => dv.getFloat64(o, little), i32 = (o) => dv.getInt32(o, little);
    datatype = dv.getInt16(12, little);
    dims = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => i64(16 + i * 8));
    pixdim = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => f64(104 + i * 8));
    voxOffset = i64(168);
    sclSlope = f64(176);
    sclInter = f64(184);
    qformCode = i32(344);
    sformCode = i32(348);
    quat = [f64(352), f64(360), f64(368)];
    qoff = [f64(376), f64(384), f64(392)];
    srow = Array.from({ length: 12 }, (_2, i) => f64(400 + i * 8));
  }
  const nd = dims[0];
  if (nd < 3 && nd !== 2) throw new Error(`unsupported NIfTI ndim ${nd}`);
  const nx = dims[1], ny = dims[2], nz = nd >= 3 ? dims[3] : 1;
  const nt2 = nd >= 4 ? Math.max(1, dims[4]) : 1;
  if (nt2 !== 1) throw new Error(`4D NIfTI (${nt2} volumes) not supported yet \u2014 load one volume at a time`);
  const dt2 = DT[datatype];
  if (!dt2) throw new Error(`unsupported NIfTI datatype ${datatype}`);
  const n = nx * ny * nz;
  let data;
  const off = Math.round(voxOffset);
  if (little || dt2.bytes === 1) {
    const buf = bytes.buffer.slice(bytes.byteOffset + off, bytes.byteOffset + off + n * dt2.bytes);
    data = new dt2.ctor(buf);
  } else {
    const out = new dt2.ctor(new ArrayBuffer(n * dt2.bytes));
    const src = new DataView(bytes.buffer, bytes.byteOffset + off, n * dt2.bytes);
    for (let i = 0; i < n; i++) {
      const o = i * dt2.bytes;
      out[i] = dt2.bytes === 2 ? datatype === 512 ? src.getUint16(o, false) : src.getInt16(o, false) : dt2.bytes === 4 ? datatype === 16 ? src.getFloat32(o, false) : datatype === 768 ? src.getUint32(o, false) : src.getInt32(o, false) : src.getFloat64(o, false);
    }
    data = out;
  }
  const meta = { datatype, qformCode, sformCode, sclSlope, sclInter };
  let M2;
  if (sformCode > 0) M2 = [...srow.slice(0, 4), ...srow.slice(4, 8), ...srow.slice(8, 12), 0, 0, 0, 1];
  else if (qformCode > 0) {
    const R2 = quaternToMat(quat[0], quat[1], quat[2], pixdim[0] === 0 ? 1 : pixdim[0], pixdim[1], pixdim[2], pixdim[3]);
    M2 = [R2[0], R2[1], R2[2], qoff[0], R2[3], R2[4], R2[5], qoff[1], R2[6], R2[7], R2[8], qoff[2], 0, 0, 0, 1];
  } else M2 = [pixdim[1] || 1, 0, 0, 0, 0, pixdim[2] || 1, 0, 0, 0, 0, pixdim[3] || 1, 0, 0, 0, 0, 1];
  return { dims: [nx, ny, nz], ijkToRAS: M2, data, dtype: dt2.zarr, name, meta };
}

// render/demos/ira/ira-nifti-seg.ts
function fileBytes(entry) {
  const data = entry.data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (data instanceof Uint8Array) return data;
  if (typeof data === "string" && data) {
    try {
      const bin = atob(data.includes(",") ? data.split(",").pop() : data);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    } catch {
      return null;
    }
  }
  return null;
}
function colorForLabel(id) {
  const h = id * 0.618033988749895 % 1 * 6;
  const x = 1 - Math.abs(h % 2 - 1);
  let r = 0, g = 0, b = 0;
  if (h < 1) {
    r = 1;
    g = x;
  } else if (h < 2) {
    r = x;
    g = 1;
  } else if (h < 3) {
    g = 1;
    b = x;
  } else if (h < 4) {
    g = x;
    b = 1;
  } else if (h < 5) {
    r = x;
    b = 1;
  } else {
    r = 1;
    b = x;
  }
  return [0.25 + 0.7 * r, 0.25 + 0.7 * g, 0.25 + 0.7 * b];
}
function invAffineRowMajor(m) {
  const a = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const t = [m[3], m[7], m[11]];
  const det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) {
    throw new Error("NIfTI ijkToRAS is singular \u2014 cannot resample onto CT");
  }
  const id = 1 / det;
  const r = [
    (a[4] * a[8] - a[5] * a[7]) * id,
    (a[2] * a[7] - a[1] * a[8]) * id,
    (a[1] * a[5] - a[2] * a[4]) * id,
    (a[5] * a[6] - a[3] * a[8]) * id,
    (a[0] * a[8] - a[2] * a[6]) * id,
    (a[2] * a[3] - a[0] * a[5]) * id,
    (a[3] * a[7] - a[4] * a[6]) * id,
    (a[1] * a[6] - a[0] * a[7]) * id,
    (a[0] * a[4] - a[1] * a[3]) * id
  ];
  return [
    r[0],
    r[1],
    r[2],
    -(r[0] * t[0] + r[1] * t[1] + r[2] * t[2]),
    r[3],
    r[4],
    r[5],
    -(r[3] * t[0] + r[4] * t[1] + r[5] * t[2]),
    r[6],
    r[7],
    r[8],
    -(r[6] * t[0] + r[7] * t[1] + r[8] * t[2]),
    0,
    0,
    0,
    1
  ];
}
function applyRowMajor(m, i, j2, k2) {
  return [
    m[0] * i + m[1] * j2 + m[2] * k2 + m[3],
    m[4] * i + m[5] * j2 + m[6] * k2 + m[7],
    m[8] * i + m[9] * j2 + m[10] * k2 + m[11]
  ];
}
function resampleLabelmapToCt(src, srcDims, srcIjkToRas, ctDims, ctIjkToRas) {
  const [snx, sny, snz] = srcDims;
  const [nx, ny, nz] = ctDims;
  const rasToSrc = invAffineRowMajor(srcIjkToRas);
  const lab = new Uint8Array(nx * ny * nz);
  const present = /* @__PURE__ */ new Set();
  let clipped = 0;
  for (let k2 = 0; k2 < nz; k2++) {
    const plane = k2 * nx * ny;
    for (let j2 = 0; j2 < ny; j2++) {
      const row = plane + j2 * nx;
      for (let i = 0; i < nx; i++) {
        const [rx, ry, rz] = applyRowMajor(ctIjkToRas, i, j2, k2);
        const [si2, sj, sk] = applyRowMajor(rasToSrc, rx, ry, rz);
        const ii2 = Math.round(si2);
        const jj = Math.round(sj);
        const kk = Math.round(sk);
        if (ii2 < 0 || jj < 0 || kk < 0 || ii2 >= snx || jj >= sny || kk >= snz) {
          continue;
        }
        let v2 = Math.round(Number(src[kk * snx * sny + jj * snx + ii2]));
        if (!Number.isFinite(v2) || v2 <= 0) continue;
        if (v2 > 255) {
          clipped++;
          v2 = 255;
        }
        lab[row + i] = v2;
        present.add(v2);
      }
    }
  }
  return { lab, present, clipped };
}
function mergeSegLabelmaps(existing, incoming) {
  if (!existing?.lab?.length) {
    return { seg: incoming, added: Object.keys(incoming.names).length, skipped: 0 };
  }
  if (existing.lab.length !== incoming.lab.length) {
    throw new Error(
      `Cannot merge SEG: length ${existing.lab.length} vs ${incoming.lab.length}`
    );
  }
  const used = /* @__PURE__ */ new Set();
  for (const v2 of existing.lab) if (v2) used.add(v2);
  for (const [num2] of existing.colors) if (num2) used.add(num2);
  const freeIds = [];
  for (let id = 1; id <= 255; id++) {
    if (!used.has(id)) freeIds.push(id);
  }
  const incomingIds = [
    ...new Set(
      [
        ...incoming.colors.map(([num2]) => num2),
        ...Object.keys(incoming.names).map(Number)
      ].filter((n) => n > 0)
    )
  ].sort((a, b) => a - b);
  const remap = /* @__PURE__ */ new Map();
  let skipped = 0;
  for (const oldId of incomingIds) {
    const next = freeIds.shift();
    if (next == null) {
      skipped++;
      continue;
    }
    remap.set(oldId, next);
  }
  const lab = new Uint8Array(existing.lab);
  for (let i = 0; i < lab.length; i++) {
    if (lab[i]) continue;
    const src = incoming.lab[i];
    if (!src) continue;
    const dst = remap.get(src);
    if (dst) lab[i] = dst;
  }
  const colors = [...existing.colors];
  const names = { ...existing.names };
  const terminology = existing.terminology ? { ...existing.terminology } : {};
  for (const [oldId, newId] of remap) {
    const fromColors = incoming.colors.find(([num2]) => num2 === oldId);
    if (fromColors) {
      colors.push([newId, fromColors[1], fromColors[2], fromColors[3]]);
    } else {
      const [r, g, b] = colorForLabel(newId);
      colors.push([newId, r, g, b]);
    }
    names[newId] = incoming.names[oldId] || (fromColors ? `Segment ${oldId}` : `Segment ${newId}`);
    const term = incoming.terminology?.[oldId];
    if (term) terminology[newId] = term;
  }
  return {
    seg: {
      lab,
      colors,
      names,
      terminology: Object.keys(terminology).length ? terminology : void 0
    },
    added: remap.size,
    skipped
  };
}
async function segLabelmapFromNiftiSendEvent(event, ctDims, ctIjkToRas) {
  const files = Jt(event);
  let chosen = null;
  let bytes = null;
  for (const entry of files) {
    const b = fileBytes(entry);
    if (b?.length) {
      chosen = entry;
      bytes = b;
      break;
    }
  }
  if (!chosen || !bytes) {
    return { ok: false, error: "nifti-send had no downloadable file payloads" };
  }
  const fileName = typeof chosen.fileName === "string" && chosen.fileName.trim() || "segmentations.nii.gz";
  let vol;
  try {
    vol = await parseNifti(bytes, fileName);
  } catch (err) {
    return {
      ok: false,
      error: `NIfTI parse failed: ${err?.message ?? err}`
    };
  }
  const [snx, sny, snz] = vol.dims;
  const nSrc = snx * sny * snz;
  if (vol.data.length < nSrc) {
    return {
      ok: false,
      error: `NIfTI voxel buffer too short (${vol.data.length} < ${nSrc})`
    };
  }
  if (!ctIjkToRas || ctIjkToRas.length < 12) {
    return { ok: false, error: "Open volume has no ijkToRAS \u2014 cannot align NIfTI" };
  }
  if (!vol.ijkToRAS || vol.ijkToRAS.length < 12) {
    return { ok: false, error: "NIfTI has no ijkToRAS \u2014 cannot align to CT" };
  }
  let lab;
  let present;
  let clipped;
  try {
    ({ lab, present, clipped } = resampleLabelmapToCt(
      vol.data,
      vol.dims,
      vol.ijkToRAS,
      ctDims,
      ctIjkToRas
    ));
  } catch (err) {
    return {
      ok: false,
      error: `NIfTI\u2192CT resample failed: ${err?.message ?? err}`
    };
  }
  if (clipped) {
    console.warn(
      `[ira-nifti-seg] ${clipped} voxel(s) had label >255; clamped to 255`
    );
  }
  if (!present.size) {
    return { ok: false, error: "NIfTI labelmap is empty after RAS resample" };
  }
  const colors = [];
  const names = {};
  for (const id of [...present].sort((a, b) => a - b)) {
    const [r, g, b] = colorForLabel(id);
    colors.push([id, r, g, b]);
    names[id] = `Segment ${id}`;
  }
  return {
    ok: true,
    fileName,
    labelCount: present.size,
    seg: { lab, colors, names }
  };
}

// logic/readers/dicom-series.ts
var cross2 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
var dot4 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
var sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
var lps2ras = (v2) => [-v2[0], -v2[1], v2[2]];
var orientKey = (iop) => iop.map((v2) => Math.round(v2 * 1e3) / 1e3).join(",");
function groupSeries(instances) {
  const byUid = /* @__PURE__ */ new Map();
  for (const i of instances) {
    const k2 = i.seriesInstanceUID;
    if (!byUid.has(k2)) byUid.set(k2, []);
    byUid.get(k2).push(i);
  }
  const out = [];
  for (const [uid, list] of byUid) {
    const byOrient = /* @__PURE__ */ new Map();
    for (const i of list) {
      const k2 = orientKey(i.imageOrientationPatient);
      if (!byOrient.has(k2)) byOrient.set(k2, []);
      byOrient.get(k2).push(i);
    }
    let idx = 0;
    for (const g of byOrient.values()) out.push({ seriesInstanceUID: byOrient.size > 1 ? `${uid}#${idx++}` : uid, description: g[0].seriesDescription, modality: g[0].modality, instances: g });
  }
  return out;
}
function reconstructSeries(instances) {
  if (instances.length === 0) throw new Error("empty DICOM series");
  const iop = instances[0].imageOrientationPatient.map(Number);
  const rowDir = iop.slice(0, 3), colDir = iop.slice(3, 6), normal = cross2(rowDir, colDir);
  const slices = [...instances].sort((a, b) => dot4(a.imagePositionPatient, normal) - dot4(b.imagePositionPatient, normal));
  const s0 = slices[0], nz = slices.length, ny = s0.rows, nx = s0.columns;
  const ps = s0.pixelSpacing;
  const p0 = slices[0].imagePositionPatient, p1 = slices[nz - 1].imagePositionPatient;
  const sliceSpacing = nz > 1 ? dot4(sub3(p1, p0), normal) / (nz - 1) : s0.sliceThickness || 1;
  const c0 = lps2ras(rowDir.map((v2) => v2 * ps[1]));
  const c1 = lps2ras(colDir.map((v2) => v2 * ps[0]));
  const c2 = lps2ras(normal.map((v2) => v2 * sliceSpacing));
  const o = lps2ras(p0);
  const ijkToRAS = [c0[0], c1[0], c2[0], o[0], c0[1], c1[1], c2[1], o[1], c0[2], c1[2], c2[2], o[2], 0, 0, 0, 1];
  const data = new Float32Array(nx * ny * nz);
  for (let k2 = 0; k2 < nz; k2++) {
    const s = slices[k2], slope = s.rescaleSlope ?? 1, inter = s.rescaleIntercept ?? 0, off = k2 * nx * ny, px = s.pixels;
    if (slope === 1 && inter === 0) {
      for (let p = 0; p < nx * ny; p++) data[off + p] = px[p];
    } else {
      for (let p = 0; p < nx * ny; p++) data[off + p] = px[p] * slope + inter;
    }
  }
  const dtype = "<f4";
  const name = s0.seriesDescription || s0.modality || "DICOM";
  const meta = { seriesInstanceUID: s0.seriesInstanceUID, studyInstanceUID: s0.studyInstanceUID, modality: s0.modality, patientName: s0.patientName };
  if (typeof s0.windowCenter === "number" && typeof s0.windowWidth === "number") {
    meta.windowCenter = s0.windowCenter;
    meta.windowWidth = s0.windowWidth;
  }
  return { dims: [nx, ny, nz], ijkToRAS, data, dtype, name, meta };
}
var dcmjsPromise = null;
var DCMJS_MIRRORS = ["https://cdn.jsdelivr.net/npm/dcmjs@0.41.0/build/dcmjs.min.js", "https://unpkg.com/dcmjs@0.41.0/build/dcmjs.min.js"];
function loadDcmjs() {
  const w = globalThis;
  if (w.dcmjs) return Promise.resolve(w.dcmjs);
  if (!w.document) return Promise.reject(new Error("dcmjs is only available in a browser (no document); use reconstructSeries with parsed instances in Deno"));
  if (!dcmjsPromise) dcmjsPromise = new Promise((resolve, reject) => {
    let i = 0;
    const tryNext = () => {
      if (i >= DCMJS_MIRRORS.length) {
        reject(new Error("dcmjs: all CDN mirrors failed"));
        return;
      }
      const s = w.document.createElement("script");
      s.src = DCMJS_MIRRORS[i++];
      s.onload = () => w.dcmjs ? resolve(w.dcmjs) : tryNext();
      s.onerror = () => {
        s.remove();
        tryNext();
      };
      w.document.head.appendChild(s);
    };
    tryNext();
  });
  return dcmjsPromise;
}
var num = (v2, d = 0) => {
  const n = Number(Array.isArray(v2) ? v2[0] : v2);
  return Number.isFinite(n) ? n : d;
};
function pixelsFromPixelData(pd, signed, bitsAllocated, rows2, columns) {
  let buf;
  let byteOffset = 0;
  let byteLength;
  if (ArrayBuffer.isView(pd)) {
    buf = pd.buffer;
    byteOffset = pd.byteOffset;
    byteLength = pd.byteLength;
  } else {
    buf = pd;
    byteLength = pd.byteLength;
  }
  const samples = rows2 * columns;
  if (samples <= 0) return null;
  const bits = bitsAllocated === 8 ? 8 : 16;
  const need = samples * (bits / 8);
  if (byteLength < need) return null;
  if (bits === 8) {
    const u8 = new Uint8Array(buf, byteOffset, samples);
    if (signed) {
      const out2 = new Int16Array(samples);
      for (let i = 0; i < samples; i++) out2[i] = u8[i] << 24 >> 24;
      return out2;
    }
    const out = new Uint16Array(samples);
    out.set(u8);
    return out;
  }
  return signed ? new Int16Array(buf, byteOffset, samples) : new Uint16Array(buf, byteOffset, samples);
}
async function parseInstances(buffers) {
  const dcmjs = await loadDcmjs();
  const out = [];
  for (const buf of buffers) {
    let ds;
    try {
      ds = dcmjs.data.DicomMetaDictionary.naturalizeDataset(dcmjs.data.DicomMessage.readFile(buf).dict);
    } catch {
      continue;
    }
    if (!ds.PixelData || !ds.ImageOrientationPatient || !ds.ImagePositionPatient || !ds.PixelSpacing) continue;
    let pd = ds.PixelData;
    if (Array.isArray(pd)) pd = pd[0];
    if (!pd) continue;
    const signed = ds.PixelRepresentation === 1;
    const rows2 = num(ds.Rows);
    const columns = num(ds.Columns);
    const bitsAllocated = num(ds.BitsAllocated, 16);
    const pixels = pixelsFromPixelData(pd, signed, bitsAllocated, rows2, columns);
    if (!pixels) continue;
    out.push({
      seriesInstanceUID: String(ds.SeriesInstanceUID ?? "series"),
      sopInstanceUID: ds.SOPInstanceUID,
      rows: rows2,
      columns,
      pixelSpacing: ds.PixelSpacing.map(Number),
      imageOrientationPatient: ds.ImageOrientationPatient.map(Number),
      imagePositionPatient: ds.ImagePositionPatient.map(Number),
      sliceThickness: ds.SliceThickness != null ? num(ds.SliceThickness) : void 0,
      rescaleSlope: ds.RescaleSlope != null ? num(ds.RescaleSlope, 1) : 1,
      rescaleIntercept: ds.RescaleIntercept != null ? num(ds.RescaleIntercept) : 0,
      pixelRepresentation: signed ? 1 : 0,
      instanceNumber: ds.InstanceNumber != null ? num(ds.InstanceNumber) : void 0,
      windowCenter: ds.WindowCenter != null ? num(ds.WindowCenter) : void 0,
      windowWidth: ds.WindowWidth != null ? num(ds.WindowWidth) : void 0,
      modality: ds.Modality,
      pixels,
      patientName: typeof ds.PatientName === "object" ? ds.PatientName.Alphabetic : ds.PatientName,
      studyInstanceUID: ds.StudyInstanceUID,
      seriesDescription: ds.SeriesDescription
    });
  }
  return out;
}

// render/demos/ira/ira-dicom-seg.ts
function fileBytes2(entry) {
  const data = entry.data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (data instanceof Uint8Array) return data;
  if (typeof data === "string" && data) {
    try {
      const bin = atob(data.includes(",") ? data.split(",").pop() : data);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    } catch {
      return null;
    }
  }
  return null;
}
function sub4(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
var lps2ras2 = (v2) => [-v2[0], -v2[1], v2[2]];
function invAffine(m) {
  const a = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const t = [m[3], m[7], m[11]];
  const det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) {
    throw new Error("CT ijkToRAS is singular \u2014 cannot map DICOM SEG");
  }
  const id = 1 / det;
  const r = [
    (a[4] * a[8] - a[5] * a[7]) * id,
    (a[2] * a[7] - a[1] * a[8]) * id,
    (a[1] * a[5] - a[2] * a[4]) * id,
    (a[5] * a[6] - a[3] * a[8]) * id,
    (a[0] * a[8] - a[2] * a[6]) * id,
    (a[2] * a[3] - a[0] * a[5]) * id,
    (a[3] * a[7] - a[4] * a[6]) * id,
    (a[1] * a[6] - a[0] * a[7]) * id,
    (a[0] * a[4] - a[1] * a[3]) * id
  ];
  return [
    r[0],
    r[1],
    r[2],
    -(r[0] * t[0] + r[1] * t[1] + r[2] * t[2]),
    r[3],
    r[4],
    r[5],
    -(r[3] * t[0] + r[4] * t[1] + r[5] * t[2]),
    r[6],
    r[7],
    r[8],
    -(r[6] * t[0] + r[7] * t[1] + r[8] * t[2]),
    0,
    0,
    0,
    1
  ];
}
function asSeq(value) {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}
function code(seq) {
  const items = asSeq(seq);
  const c = items[0];
  if (!c || typeof c !== "object") return null;
  return {
    scheme: String(c.CodingSchemeDesignator || ""),
    value: String(c.CodeValue || ""),
    meaning: String(c.CodeMeaning || "")
  };
}
function pixelDataBits(ds) {
  let pd = ds.PixelData;
  if (Array.isArray(pd)) pd = pd[0];
  if (pd instanceof ArrayBuffer) return new Uint8Array(pd);
  if (pd instanceof Uint8Array) return pd;
  if (pd && typeof pd.buffer === "object") {
    const view = pd;
    return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  }
  throw new Error("DICOM SEG has no PixelData");
}
function buildLabelmap(ds, bits, ct, dcmjs) {
  const [nx, ny, nz] = ct.dims;
  const segCols = Number(ds.Columns) || nx;
  const segRows = Number(ds.Rows) || ny;
  const frameBytes = segCols * segRows >> 3;
  const lab = new Uint8Array(nx * ny * nz);
  const inv = invAffine(ct.ijkToRAS);
  const toIJK = (lps) => {
    const r = lps2ras2(lps);
    return [
      inv[0] * r[0] + inv[1] * r[1] + inv[2] * r[2] + inv[3],
      inv[4] * r[0] + inv[5] * r[1] + inv[6] * r[2] + inv[7],
      inv[8] * r[0] + inv[9] * r[1] + inv[10] * r[2] + inv[11]
    ];
  };
  const sharedSeq = asSeq(ds.SharedFunctionalGroupsSequence);
  const shared = sharedSeq[0] || {};
  const orientSeq = asSeq(shared.PlaneOrientationSequence);
  const orient0 = orientSeq[0] || {};
  const measuresSeq = asSeq(shared.PixelMeasuresSequence);
  const measures0 = measuresSeq[0] || {};
  const sIopRaw = orient0.ImageOrientationPatient ?? ct.iop ?? [1, 0, 0, 0, 1, 0];
  const sPsRaw = measures0.PixelSpacing ?? ct.ps ?? [1, 1];
  const sIop = (Array.isArray(sIopRaw) ? sIopRaw : [sIopRaw]).map(Number);
  const sPs = (Array.isArray(sPsRaw) ? sPsRaw : [sPsRaw]).map(Number);
  const colW = sIop.slice(0, 3).map((v2) => v2 * sPs[1]);
  const rowW = sIop.slice(3, 6).map((v2) => v2 * sPs[0]);
  const colors = [];
  const names = {};
  const terminology = {};
  for (const s of asSeq(ds.SegmentSequence)) {
    const cie = s.RecommendedDisplayCIELabValue;
    let rgb = [1, 1, 1];
    if (Array.isArray(cie) && cie.length >= 3) {
      try {
        rgb = dcmjs.data.Colors.dicomlab2RGB(cie.map(Number));
      } catch {
        rgb = colorForLabel(Number(s.SegmentNumber) || 1);
      }
    } else {
      rgb = colorForLabel(Number(s.SegmentNumber) || 1);
    }
    const num2 = Number(s.SegmentNumber);
    if (!Number.isFinite(num2) || num2 <= 0) continue;
    colors.push([num2, rgb[0], rgb[1], rgb[2]]);
    names[num2] = String(s.SegmentLabel || `Segment ${num2}`);
    const type = s.SegmentedPropertyTypeCodeSequence;
    const type0 = asSeq(type)[0];
    terminology[num2] = {
      category: code(s.SegmentedPropertyCategoryCodeSequence),
      type: code(type),
      typeModifier: code(type0?.SegmentedPropertyTypeModifierCodeSequence),
      region: code(s.AnatomicRegionSequence)
    };
  }
  const perFrame = asSeq(ds.PerFrameFunctionalGroupsSequence);
  const firstFg = perFrame[0] || {};
  const firstPos = asSeq(
    firstFg.PlanePositionSequence
  )[0];
  const ref = (Array.isArray(firstPos?.ImagePositionPatient) ? firstPos.ImagePositionPatient : [0, 0, 0]).map(Number);
  const o0 = toIJK(ref);
  const diCol = sub4(
    toIJK([ref[0] + colW[0], ref[1] + colW[1], ref[2] + colW[2]]),
    o0
  );
  const diRow = sub4(
    toIJK([ref[0] + rowW[0], ref[1] + rowW[1], ref[2] + rowW[2]]),
    o0
  );
  for (let f = 0; f < perFrame.length; f++) {
    const fg = perFrame[f];
    const segIdSeq = asSeq(fg.SegmentIdentificationSequence)[0];
    const segNum = Number(segIdSeq?.ReferencedSegmentNumber);
    const posSeq = asSeq(fg.PlanePositionSequence)[0];
    const ippRaw = posSeq?.ImagePositionPatient;
    if (!segNum || !Array.isArray(ippRaw)) continue;
    const ippLps = ippRaw.map(Number);
    const o = toIJK(ippLps);
    const fb = f * frameBytes;
    for (let row = 0; row < segRows; row++) {
      const bi2 = o[0] + row * diRow[0];
      const bj = o[1] + row * diRow[1];
      const bk = o[2] + row * diRow[2];
      const rb = row * segCols;
      for (let col = 0; col < segCols; col++) {
        const p = rb + col;
        if (!(bits[fb + (p >> 3)] >> (p & 7) & 1)) continue;
        const i = Math.round(bi2 + col * diCol[0]);
        const j2 = Math.round(bj + col * diCol[1]);
        const k2 = Math.round(bk + col * diCol[2]);
        if (i >= 0 && i < nx && j2 >= 0 && j2 < ny && k2 >= 0 && k2 < nz) {
          lab[k2 * nx * ny + j2 * nx + i] = segNum;
        }
      }
    }
  }
  return { lab, colors, names, terminology };
}
function isDicomSeg(ds) {
  const modality = String(ds.Modality || "").toUpperCase();
  if (modality === "SEG") return true;
  const sop = String(ds.SOPClassUID || "");
  return sop.includes("1.2.840.10008.5.1.4.1.1.66.4");
}
async function segLabelmapFromDicomSendEvent(event, ct) {
  const files = Jt(event);
  let chosen = null;
  let bytes = null;
  for (const entry of files) {
    const b = fileBytes2(entry);
    if (b?.length) {
      chosen = entry;
      bytes = b;
      break;
    }
  }
  if (!chosen || !bytes) {
    return { ok: false, error: "dicom-send had no downloadable file payloads" };
  }
  const fileName = typeof chosen.fileName === "string" && chosen.fileName.trim() || "segmentations.dcm";
  if (!ct.ijkToRAS || ct.ijkToRAS.length < 12) {
    return { ok: false, error: "Open volume has no ijkToRAS \u2014 cannot align DICOM SEG" };
  }
  let dcmjs;
  let ds;
  try {
    dcmjs = await loadDcmjs();
    const buf = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength ? bytes.buffer : bytes.slice().buffer;
    ds = dcmjs.data.DicomMetaDictionary.naturalizeDataset(
      dcmjs.data.DicomMessage.readFile(buf).dict
    );
  } catch (err) {
    return {
      ok: false,
      error: `DICOM parse failed: ${err?.message ?? err}`
    };
  }
  if (!isDicomSeg(ds)) {
    const mod = String(ds.Modality || "?");
    return {
      ok: false,
      error: `Expected DICOM SEG, got Modality=${mod} (RTSTRUCT is not supported)`
    };
  }
  let bits;
  try {
    bits = pixelDataBits(ds);
  } catch (err) {
    return { ok: false, error: err?.message ?? String(err) };
  }
  let seg;
  try {
    seg = buildLabelmap(ds, bits, ct, dcmjs);
  } catch (err) {
    return {
      ok: false,
      error: `DICOM SEG\u2192CT map failed: ${err?.message ?? err}`
    };
  }
  const present = /* @__PURE__ */ new Set();
  for (const v2 of seg.lab) if (v2) present.add(v2);
  if (!present.size) {
    return { ok: false, error: "DICOM SEG labelmap is empty after resample onto CT" };
  }
  return {
    ok: true,
    fileName,
    labelCount: present.size,
    seg
  };
}

// render/demos/ira/totalseg-tasks.ts
var TOTAL_SEG_GROUP_LABELS = {
  "total": "Total body",
  "body": "Body",
  "spine": "Spine",
  "lung": "Lung",
  "liver": "Liver",
  "kidney": "Kidney",
  "breast": "Breast",
  "abdomen": "Abdomen",
  "head": "Head",
  "headneck": "Head & neck",
  "brain": "Brain",
  "heart": "Heart",
  "bones": "Bones",
  "tissue": "Tissue",
  "face": "Face",
  "muscles": "Muscles"
};
var TOTAL_SEG_TASKS = [
  {
    id: "total",
    modality: "CT",
    group: "total",
    title: "total",
    licenseRequired: false,
    supportsFast: true,
    classes: ["spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas", "adrenal_gland_right", "adrenal_gland_left", "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right", "esophagus", "trachea", "thyroid_gland", "small_bowel", "duodenum", "colon", "urinary_bladder", "prostate", "kidney_cyst_left", "kidney_cyst_right", "sacrum", "vertebrae_S1", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1", "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", "vertebrae_T2", "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1", "heart", "aorta", "pulmonary_vein", "brachiocephalic_trunk", "subclavian_artery_right", "subclavian_artery_left", "common_carotid_artery_right", "common_carotid_artery_left", "brachiocephalic_vein_left", "brachiocephalic_vein_right", "atrial_appendage_left", "superior_vena_cava", "inferior_vena_cava", "portal_vein_and_splenic_vein", "iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right", "humerus_left", "humerus_right", "scapula_left", "scapula_right", "clavicula_left", "clavicula_right", "femur_left", "femur_right", "hip_left", "hip_right", "spinal_cord", "gluteus_maximus_left", "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right", "gluteus_minimus_left", "gluteus_minimus_right", "autochthon_left", "autochthon_right", "iliopsoas_left", "iliopsoas_right", "brain", "skull", "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5", "rib_left_6", "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12", "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6", "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12", "sternum", "costal_cartilages"]
  },
  {
    id: "total_mr",
    modality: "MR",
    group: "total",
    title: "total",
    licenseRequired: false,
    supportsFast: true,
    classes: ["spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas", "adrenal_gland_right", "adrenal_gland_left", "lung_left", "lung_right", "esophagus", "small_bowel", "duodenum", "colon", "urinary_bladder", "prostate", "sacrum", "vertebrae", "intervertebral_discs", "spinal_cord", "heart", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein", "iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right", "humerus_left", "humerus_right", "scapula_left", "scapula_right", "clavicula_left", "clavicula_right", "femur_left", "femur_right", "hip_left", "hip_right", "gluteus_maximus_left", "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right", "gluteus_minimus_left", "gluteus_minimus_right", "autochthon_left", "autochthon_right", "iliopsoas_left", "iliopsoas_right", "brain"]
  },
  {
    id: "body",
    modality: "CT",
    group: "body",
    title: "body",
    licenseRequired: false,
    supportsFast: true,
    classes: ["body_trunc", "body_extremities"]
  },
  {
    id: "body_mr",
    modality: "MR",
    group: "body",
    title: "body",
    licenseRequired: false,
    supportsFast: true,
    classes: ["body_trunc", "body_extremities"]
  },
  {
    id: "vertebrae_mr",
    modality: "MR",
    group: "spine",
    title: "vertebrae",
    licenseRequired: false,
    supportsFast: true,
    classes: ["sacrum", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1", "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", "vertebrae_T2", "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1"]
  },
  {
    id: "vertebrae_pp",
    modality: "CT",
    group: "spine",
    title: "vertebrae (postprocess)",
    licenseRequired: false,
    supportsFast: false,
    classes: ["vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7", "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5", "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5"]
  },
  {
    id: "vertebrae_pp_refined",
    modality: "CT",
    group: "spine",
    title: "vertebrae (refined)",
    licenseRequired: false,
    supportsFast: false,
    classes: ["vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7", "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5", "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5"]
  },
  {
    id: "vertebrae_body",
    modality: "CT",
    group: "spine",
    title: "vertebrae body",
    licenseRequired: false,
    supportsFast: false,
    classes: ["vertebrae_body", "intervertebral_discs"]
  },
  {
    id: "lung_nodules",
    modality: "CT",
    group: "lung",
    title: "nodules",
    licenseRequired: false,
    supportsFast: false,
    classes: ["lung", "lung_nodules"]
  },
  {
    id: "lung_vessels",
    modality: "CT",
    group: "lung",
    title: "vessels and airways",
    licenseRequired: false,
    supportsFast: true,
    classes: ["lung_airways", "lung_airways_wall", "lung_arteries", "lung_veins"]
  },
  {
    id: "liver_lesions",
    modality: "CT",
    group: "liver",
    title: "lesions",
    licenseRequired: false,
    supportsFast: false,
    classes: ["liver_lesions"]
  },
  {
    id: "liver_lesions_mr",
    modality: "MR",
    group: "liver",
    title: "lesions",
    licenseRequired: false,
    supportsFast: false,
    classes: ["liver_lesions"]
  },
  {
    id: "liver_segments",
    modality: "CT",
    group: "liver",
    title: "segments",
    licenseRequired: false,
    supportsFast: false,
    classes: ["liver_segment_1", "liver_segment_2", "liver_segment_3", "liver_segment_4", "liver_segment_5", "liver_segment_6", "liver_segment_7", "liver_segment_8"]
  },
  {
    id: "liver_segments_mr",
    modality: "MR",
    group: "liver",
    title: "segments",
    licenseRequired: false,
    supportsFast: false,
    classes: ["liver_segment_1", "liver_segment_2", "liver_segment_3", "liver_segment_4", "liver_segment_5", "liver_segment_6", "liver_segment_7", "liver_segment_8"]
  },
  {
    id: "liver_vessels",
    modality: "CT",
    group: "liver",
    title: "vessels",
    licenseRequired: false,
    supportsFast: false,
    classes: ["liver_vessels", "liver_tumor"]
  },
  {
    id: "kidney_cysts",
    modality: "CT",
    group: "kidney",
    title: "cysts",
    licenseRequired: false,
    supportsFast: false,
    classes: ["kidney_cyst_left", "kidney_cyst_right"]
  },
  {
    id: "breasts",
    modality: "CT",
    group: "breast",
    title: "breasts",
    licenseRequired: false,
    supportsFast: false,
    classes: ["breast"]
  },
  {
    id: "abdominal_muscles",
    modality: "CT",
    group: "abdomen",
    title: "abdominal muscles",
    licenseRequired: false,
    supportsFast: false,
    classes: ["pectoralis_major_right", "pectoralis_major_left", "rectus_abdominis_right", "rectus_abdominis_left", "serratus_anterior_right", "serratus_anterior_left", "latissimus_dorsi_right", "latissimus_dorsi_left", "trapezius_right", "trapezius_left", "external_oblique_right", "external_oblique_left", "internal_oblique_right", "internal_oblique_left", "erector_spinae_right", "erector_spinae_left", "transversospinalis_right", "transversospinalis_left", "psoas_major_right", "psoas_major_left", "quadratus_lumborum_right", "quadratus_lumborum_left"]
  },
  {
    id: "trunk_cavities",
    modality: "CT",
    group: "abdomen",
    title: "trunk cavities",
    licenseRequired: false,
    supportsFast: false,
    classes: ["abdominal_cavity", "thoracic_cavity", "pericardium", "mediastinum"]
  },
  {
    id: "craniofacial_structures",
    modality: "CT",
    group: "head",
    title: "craniofacial structures",
    licenseRequired: false,
    supportsFast: false,
    classes: ["mandible", "teeth_lower", "skull", "head", "sinus_maxillary", "sinus_frontal", "teeth_upper"]
  },
  {
    id: "head_glands_cavities",
    modality: "CT",
    group: "head",
    title: "glands and cavities",
    licenseRequired: false,
    supportsFast: false,
    classes: ["eye_left", "eye_right", "eye_lens_left", "eye_lens_right", "optic_nerve_left", "optic_nerve_right", "parotid_gland_left", "parotid_gland_right", "submandibular_gland_right", "submandibular_gland_left", "nasopharynx", "oropharynx", "hypopharynx", "nasal_cavity_right", "nasal_cavity_left", "auditory_canal_right", "auditory_canal_left", "soft_palate", "hard_palate"]
  },
  {
    id: "head_muscles",
    modality: "CT",
    group: "head",
    title: "muscles",
    licenseRequired: false,
    supportsFast: false,
    classes: ["masseter_right", "masseter_left", "temporalis_right", "temporalis_left", "lateral_pterygoid_right", "lateral_pterygoid_left", "medial_pterygoid_right", "medial_pterygoid_left", "tongue", "digastric_right", "digastric_left"]
  },
  {
    id: "oculomotor_muscles",
    modality: "CT",
    group: "head",
    title: "oculomotor muscles",
    licenseRequired: false,
    supportsFast: false,
    classes: ["skull", "eyeball_right", "lateral_rectus_muscle_right", "superior_oblique_muscle_right", "levator_palpebrae_superioris_right", "superior_rectus_muscle_right", "medial_rectus_muscle_left", "inferior_oblique_muscle_right", "inferior_rectus_muscle_right", "optic_nerve_left", "eyeball_left", "lateral_rectus_muscle_left", "superior_oblique_muscle_left", "levator_palpebrae_superioris_left", "superior_rectus_muscle_left", "medial_rectus_muscle_right", "inferior_oblique_muscle_left", "inferior_rectus_muscle_left", "optic_nerve_right"]
  },
  {
    id: "teeth",
    modality: "CT",
    group: "head",
    title: "teeth",
    licenseRequired: false,
    supportsFast: false,
    classes: ["lower_jawbone", "upper_jawbone", "left_inferior_alveolar_canal", "right_inferior_alveolar_canal", "left_maxillary_sinus", "right_maxillary_sinus", "pharynx", "bridge", "crown", "implant", "upper_right_central_incisor_fdi11", "upper_right_lateral_incisor_fdi12", "upper_right_canine_fdi13", "upper_right_first_premolar_fdi14", "upper_right_second_premolar_fdi15", "upper_right_first_molar_fdi16", "upper_right_second_molar_fdi17", "upper_right_third_molar_fdi18", "upper_left_central_incisor_fdi21", "upper_left_lateral_incisor_fdi22", "upper_left_canine_fdi23", "upper_left_first_premolar_fdi24", "upper_left_second_premolar_fdi25", "upper_left_first_molar_fdi26", "upper_left_second_molar_fdi27", "upper_left_third_molar_fdi28", "lower_left_central_incisor_fdi31", "lower_left_lateral_incisor_fdi32", "lower_left_canine_fdi33", "lower_left_first_premolar_fdi34", "lower_left_second_premolar_fdi35", "lower_left_first_molar_fdi36", "lower_left_second_molar_fdi37", "lower_left_third_molar_fdi38", "lower_right_central_incisor_fdi41", "lower_right_lateral_incisor_fdi42", "lower_right_canine_fdi43", "lower_right_first_premolar_fdi44", "lower_right_second_premolar_fdi45", "lower_right_first_molar_fdi46", "lower_right_second_molar_fdi47", "lower_right_third_molar_fdi48", "left_mandibular_incisive_canal_fdi103", "right_mandibular_incisive_canal_fdi104", "lingual_canal", "upper_right_central_incisor_pulp_fdi111", "upper_right_lateral_incisor_pulp_fdi112", "upper_right_canine_pulp_fdi113", "upper_right_first_premolar_pulp_fdi114", "upper_right_second_premolar_pulp_fdi115", "upper_right_first_molar_pulp_fdi116", "upper_right_second_molar_pulp_fdi117", "upper_right_third_molar_pulp_fdi118", "upper_left_central_incisor_pulp_fdi121", "upper_left_lateral_incisor_pulp_fdi122", "upper_left_canine_pulp_fdi123", "upper_left_first_premolar_pulp_fdi124", "upper_left_second_premolar_pulp_fdi125", "upper_left_first_molar_pulp_fdi126", "upper_left_second_molar_pulp_fdi127", "upper_left_third_molar_pulp_fdi128", "lower_left_central_incisor_pulp_fdi131", "lower_left_lateral_incisor_pulp_fdi132", "lower_left_canine_pulp_fdi133", "lower_left_first_premolar_pulp_fdi134", "lower_left_second_premolar_pulp_fdi135", "lower_left_first_molar_pulp_fdi136", "lower_left_second_molar_pulp_fdi137", "lower_left_third_molar_pulp_fdi138", "lower_right_central_incisor_pulp_fdi141", "lower_right_lateral_incisor_pulp_fdi142", "lower_right_canine_pulp_fdi143", "lower_right_first_premolar_pulp_fdi144", "lower_right_second_premolar_pulp_fdi145", "lower_right_first_molar_pulp_fdi146", "lower_right_second_molar_pulp_fdi147", "lower_right_third_molar_pulp_fdi148"]
  },
  {
    id: "headneck_bones_vessels",
    modality: "CT",
    group: "headneck",
    title: "bones and vessels",
    licenseRequired: false,
    supportsFast: false,
    classes: ["larynx_air", "thyroid_cartilage", "hyoid", "cricoid_cartilage", "zygomatic_arch_right", "zygomatic_arch_left", "styloid_process_right", "styloid_process_left", "internal_carotid_artery_right", "internal_carotid_artery_left", "internal_jugular_vein_right", "internal_jugular_vein_left"]
  },
  {
    id: "headneck_muscles",
    modality: "CT",
    group: "headneck",
    title: "muscles",
    licenseRequired: false,
    supportsFast: false,
    classes: ["sternocleidomastoid_right", "sternocleidomastoid_left", "superior_pharyngeal_constrictor", "middle_pharyngeal_constrictor", "inferior_pharyngeal_constrictor", "trapezius_right", "trapezius_left", "platysma_right", "platysma_left", "levator_scapulae_right", "levator_scapulae_left", "anterior_scalene_right", "anterior_scalene_left", "middle_scalene_right", "middle_scalene_left", "posterior_scalene_right", "posterior_scalene_left", "sterno_thyroid_right", "sterno_thyroid_left", "thyrohyoid_right", "thyrohyoid_left", "prevertebral_right", "prevertebral_left"]
  },
  {
    id: "brain_aneurysm",
    modality: "MR",
    group: "brain",
    title: "aneurysm (TOF MRI)",
    licenseRequired: false,
    supportsFast: false,
    classes: ["brain_aneurysm"]
  },
  {
    id: "cerebral_bleed",
    modality: "CT",
    group: "brain",
    title: "cerebral bleed",
    licenseRequired: false,
    supportsFast: false,
    classes: ["intracerebral_hemorrhage"]
  },
  {
    id: "brain_structures",
    modality: "CT",
    group: "brain",
    title: "structures",
    licenseRequired: true,
    supportsFast: false,
    classes: ["brainstem", "subarachnoid_space", "venous_sinuses", "septum_pellucidum", "cerebellum", "caudate_nucleus", "lentiform_nucleus", "insular_cortex", "internal_capsule", "ventricle", "central_sulcus", "frontal_lobe", "parietal_lobe", "occipital_lobe", "temporal_lobe", "thalamus"]
  },
  {
    id: "ventricle_parts",
    modality: "CT",
    group: "brain",
    title: "ventricle parts",
    licenseRequired: false,
    supportsFast: false,
    classes: ["ventricle_frontal_horn_left", "ventricle_occipital_horn_left", "ventricle_body_left", "ventricle_temporal_horn_left", "ventricle_trigone_left", "ventricle_frontal_horn_right", "ventricle_occipital_horn_right", "ventricle_body_right", "ventricle_temporal_horn_right", "ventricle_trigone_right", "third_ventricle", "fourth_ventricle"]
  },
  {
    id: "aortic_sinuses",
    modality: "CT",
    group: "heart",
    title: "aortic sinuses",
    licenseRequired: true,
    supportsFast: false,
    classes: ["left_ventricular_outflow_tract", "right_coronary_cusp", "left_coronary_cusp", "non_coronary_cusp"]
  },
  {
    id: "heartchambers_highres",
    modality: "CT",
    group: "heart",
    title: "chambers highres",
    licenseRequired: true,
    supportsFast: false,
    classes: ["heart_myocardium", "heart_atrium_left", "heart_ventricle_left", "heart_atrium_right", "heart_ventricle_right", "aorta", "pulmonary_artery"]
  },
  {
    id: "coronary_arteries",
    modality: "CT",
    group: "heart",
    title: "coronary arteries",
    licenseRequired: true,
    supportsFast: false,
    classes: ["coronary_arteries"]
  },
  {
    id: "pleural_pericard_effusion",
    modality: "CT",
    group: "heart",
    title: "pleural / pericardial effusion",
    licenseRequired: false,
    supportsFast: false,
    classes: ["lung_pleural", "pleural_effusion", "pericardial_effusion"]
  },
  {
    id: "appendicular_bones",
    modality: "CT",
    group: "bones",
    title: "appendicular bones",
    licenseRequired: true,
    supportsFast: false,
    classes: ["patella", "tibia", "fibula", "tarsal", "metatarsal", "phalanges_feet", "ulna", "radius", "carpal", "metacarpal", "phalanges_hand"]
  },
  {
    id: "appendicular_bones_mr",
    modality: "MR",
    group: "bones",
    title: "appendicular bones",
    licenseRequired: true,
    supportsFast: false,
    classes: ["patella", "tibia", "fibula", "tarsal", "metatarsal", "phalanges_feet", "ulna", "radius"]
  },
  {
    id: "hip_implant",
    modality: "CT",
    group: "bones",
    title: "hip implant",
    licenseRequired: false,
    supportsFast: false,
    classes: ["hip_implant"]
  },
  {
    id: "tissue_4_types",
    modality: "CT",
    group: "tissue",
    title: "tissue 4 types",
    licenseRequired: true,
    supportsFast: false,
    classes: ["subcutaneous_fat", "torso_fat", "skeletal_muscle", "intermuscular_fat"]
  },
  {
    id: "tissue_types",
    modality: "CT",
    group: "tissue",
    title: "tissue types",
    licenseRequired: true,
    supportsFast: false,
    classes: ["subcutaneous_fat", "torso_fat", "skeletal_muscle"]
  },
  {
    id: "tissue_types_mr",
    modality: "MR",
    group: "tissue",
    title: "tissue types",
    licenseRequired: true,
    supportsFast: false,
    classes: ["subcutaneous_fat", "torso_fat", "skeletal_muscle"]
  },
  {
    id: "face",
    modality: "CT",
    group: "face",
    title: "face",
    licenseRequired: true,
    supportsFast: false,
    classes: ["face"]
  },
  {
    id: "face_mr",
    modality: "MR",
    group: "face",
    title: "face",
    licenseRequired: true,
    supportsFast: false,
    classes: ["face"]
  },
  {
    id: "thigh_shoulder_muscles",
    modality: "CT",
    group: "muscles",
    title: "thigh and shoulder muscles",
    licenseRequired: true,
    supportsFast: false,
    classes: ["quadriceps_femoris_left", "quadriceps_femoris_right", "thigh_medial_compartment_left", "thigh_medial_compartment_right", "thigh_posterior_compartment_left", "thigh_posterior_compartment_right", "sartorius_left", "sartorius_right", "deltoid", "supraspinatus", "infraspinatus", "subscapularis", "coracobrachial", "trapezius", "pectoralis_minor", "serratus_anterior", "teres_major", "triceps_brachii"]
  },
  {
    id: "thigh_shoulder_muscles_mr",
    modality: "MR",
    group: "muscles",
    title: "thigh and shoulder muscles",
    licenseRequired: true,
    supportsFast: false,
    classes: ["quadriceps_femoris_left", "quadriceps_femoris_right", "thigh_medial_compartment_left", "thigh_medial_compartment_right", "thigh_posterior_compartment_left", "thigh_posterior_compartment_right", "sartorius_left", "sartorius_right", "deltoid", "supraspinatus", "infraspinatus", "subscapularis", "coracobrachial", "trapezius", "pectoralis_minor", "serratus_anterior", "teres_major", "triceps_brachii"]
  }
];
var TASK_BY_ID = new Map(TOTAL_SEG_TASKS.map((t) => [t.id, t]));
function getTotalSegTask(id) {
  return TASK_BY_ID.get(id);
}
function tasksForModality(modality) {
  return TOTAL_SEG_TASKS.filter((t) => t.modality === modality);
}
function groupsForModality(modality) {
  const seen = /* @__PURE__ */ new Set();
  const out = [];
  for (const t of tasksForModality(modality)) {
    if (seen.has(t.group)) continue;
    seen.add(t.group);
    out.push(t.group);
  }
  return out;
}
function tasksInGroup(group, modality) {
  return tasksForModality(modality).filter((t) => t.group === group);
}
function defaultTaskId(modality) {
  return modality === "MR" ? "total_mr" : "total";
}
function taskOptionLabel(task) {
  const lic = task.licenseRequired ? " [license required]" : "";
  return `${task.title}${lic}`;
}
function taskSupportsFast(taskId) {
  const task = getTotalSegTask(taskId);
  if (!task) return true;
  return task.supportsFast !== false;
}
function searchClasses(query, modality, limit = 40) {
  const q2 = query.trim().toLowerCase();
  if (q2.length < 2) return [];
  const hits = [];
  for (const task of tasksForModality(modality)) {
    for (const className of task.classes) {
      if (!className.toLowerCase().includes(q2)) continue;
      hits.push({ className, task });
      if (hits.length >= limit) return hits;
    }
  }
  return hits;
}
var TOTAL_SEG_TASK_OPTIONS = TOTAL_SEG_TASKS.map((t) => ({
  value: t.id,
  title: taskOptionLabel(t)
}));

// logic/window-level.ts
var isIntType = (a) => !(a instanceof Float32Array || a instanceof Float64Array);
function histogramPercentiles(data, loP = 1e-3, hiP = 0.999) {
  const n = data.length;
  if (n === 0) return [0, 1];
  const step = n > 4e6 ? Math.floor(n / 4e6) : 1;
  let mn2 = Infinity, mx = -Infinity;
  for (let i = 0; i < n; i += step) {
    const v2 = data[i];
    if (v2 < mn2) mn2 = v2;
    if (v2 > mx) mx = v2;
  }
  if (mx <= mn2) return [mn2, mn2];
  const intType = isIntType(data);
  const bins = intType ? Math.min(mx - mn2 + 1, 1 << 16) : 1e3;
  const scale2 = (bins - 1) / (mx - mn2);
  const hist = new Float64Array(bins);
  let total = 0;
  for (let i = 0; i < n; i += step) {
    hist[Math.round((data[i] - mn2) * scale2)]++;
    total++;
  }
  const binValue = (b) => mn2 + b / scale2;
  const at = (frac) => {
    const target = frac * total;
    let c = 0;
    for (let b = 0; b < bins; b++) {
      c += hist[b];
      if (c >= target) return binValue(b);
    }
    return mx;
  };
  return [at(loP), at(hiP)];
}
function autoWindowLevel(data) {
  const [lo, hi2] = histogramPercentiles(data, 1e-3, 0.999);
  return { window: Math.max(1e-6, hi2 - lo), level: (lo + hi2) / 2 };
}

// render/demos/ira/ira-scene.ts
var LIVESCENE_MIME = "application/vnd.slicerlive.scene+json";
var EMBEDDED_SCENE_PARAM = "embedded";
var EMBEDDED_SCENE_STORAGE_KEY = "ira.embeddedLiveScene";
function looksLikeSceneJson(f) {
  const role = String(f.role || "").toLowerCase();
  if (role === "scene") return true;
  const name = String(f.fileName || "").toLowerCase();
  const url = String(f.url || "").toLowerCase();
  if (name.endsWith(".json")) return true;
  try {
    const path = new URL(url).pathname.toLowerCase();
    return path.endsWith(".json");
  } catch {
    return url.includes(".json");
  }
}
function isLiveSceneDoc(value) {
  if (!value || typeof value !== "object") return false;
  const doc = value;
  const nodes = doc.nodes;
  const blobBase = String(doc.blobBase || "").trim();
  return Boolean(nodes && typeof nodes === "object" && blobBase);
}
function sceneDocFromOpenContext(context) {
  if (!Array.isArray(context)) return null;
  for (const item of context) {
    if (!item || typeof item !== "object") continue;
    const row = item;
    if (String(row.key || "").toLowerCase() !== "scene") continue;
    if (isLiveSceneDoc(row.resource)) {
      return { ...row.resource };
    }
  }
  return null;
}
function sceneUrlFromOpenContext(context) {
  const plan = ur(context);
  if (!plan || plan.mode !== "files") return null;
  const files = Array.isArray(plan.files) ? plan.files : [];
  const sceneFile = files.find((f) => String(f.role || "").toLowerCase() === "scene") || files.find(looksLikeSceneJson) || files[0];
  const url = String(sceneFile?.url || "").trim();
  if (!url) return null;
  if (!looksLikeSceneJson(sceneFile || {})) return null;
  return url;
}
function studyParamsFromSceneUrl(sceneUrl) {
  return { scene: String(sceneUrl || "").trim() };
}
function studyParamsFromSceneDoc(doc) {
  const url = String(doc.url || "").trim();
  if (url) return { scene: url };
  try {
    globalThis.sessionStorage?.setItem(
      EMBEDDED_SCENE_STORAGE_KEY,
      JSON.stringify(doc)
    );
  } catch (err) {
    console.warn("[ira] failed to stash embedded LiveScene", err);
  }
  return { scene: EMBEDDED_SCENE_PARAM };
}
function takeEmbeddedSceneFromStorage() {
  try {
    const raw = globalThis.sessionStorage?.getItem(EMBEDDED_SCENE_STORAGE_KEY);
    if (!raw) return null;
    globalThis.sessionStorage?.removeItem(EMBEDDED_SCENE_STORAGE_KEY);
    const data = JSON.parse(raw);
    return isLiveSceneDoc(data) ? data : null;
  } catch {
    return null;
  }
}
function imagingStudyContextFromSceneUrl(sceneUrl) {
  const url = String(sceneUrl || "").trim();
  if (!url) return [];
  let fileName = "scene.json";
  try {
    const path = new URL(url).pathname;
    const base = path.split("/").pop();
    if (base) fileName = base;
  } catch {
  }
  return Gt({
    id: `scene-${fileName.replace(/\.[^.]+$/, "") || "livescene"}`,
    files: [
      {
        url,
        fileName,
        role: "scene",
        label: "LiveScene",
        mimeType: LIVESCENE_MIME
      }
    ]
  });
}
function imagingStudyContextFromSceneDoc(doc, idHint) {
  const url = String(doc.url || "").trim();
  const filesCtx = url ? imagingStudyContextFromSceneUrl(url) : Gt({
    id: String(idHint || "livescene").trim() || "livescene",
    files: []
  });
  const withoutScene = filesCtx.filter((item) => {
    if (!item || typeof item !== "object") return true;
    return item.key !== "scene";
  });
  withoutScene.push({ key: "scene", resource: { ...doc } });
  return withoutScene;
}
async function fetchLiveSceneDoc(sceneUrl) {
  const url = String(sceneUrl || "").trim();
  if (!url) return null;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch LiveScene (${res.status})`);
  }
  const data = await res.json();
  const nodes = data.nodes && typeof data.nodes === "object" ? data.nodes : void 0;
  const doc = {
    ...data,
    url
  };
  if (typeof data.blobBase === "string") doc.blobBase = data.blobBase;
  if (typeof data.mrson === "string") doc.mrson = data.mrson;
  if (nodes) doc.nodes = nodes;
  return doc;
}
function imagingStudyContextFromLocalStudy(opts) {
  const studyUid = String(opts.studyInstanceUID || "").trim();
  const seriesUid = String(opts.seriesInstanceUID || "").trim();
  const id = String(opts.id || "").trim() || studyUid || seriesUid || `local-${Date.now()}`;
  const context = Gt({
    id,
    files: []
  });
  for (const item of context) {
    if (!item || typeof item !== "object") continue;
    if (item.key !== "study") continue;
    const resource = item.resource;
    if (!resource || typeof resource !== "object") continue;
    resource.description = String(opts.label || "").trim() || "Local study";
    const modality = String(opts.modality || "").trim().toUpperCase();
    if (modality) {
      resource.modality = [{ coding: [{ system: "http://dicom.nema.org/resources/ontology/DCM", code: modality }] }];
    }
    if (studyUid) {
      const identifiers = Array.isArray(resource.identifier) ? [...resource.identifier] : [];
      identifiers.push({
        system: "urn:dicom:uid",
        value: `urn:oid:${studyUid}`
      });
      resource.identifier = identifiers;
    }
    if (opts.anonymized) {
      resource.meta = {
        ...typeof resource.meta === "object" && resource.meta ? resource.meta : {},
        security: [
          {
            system: "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            code: "ANONYED",
            display: "anonymized"
          }
        ]
      };
    }
  }
  return context;
}
function modalityFromSceneName(name) {
  const n = String(name || "").toUpperCase();
  if (/\bMR\b/.test(n) || n.includes("MRHEAD") || n.startsWith("MR")) return "MR";
  return "CT";
}
function iraVolumeFromSceneVolume(sv) {
  return {
    vol: sv.voxels,
    dims: sv.dims,
    ijkToRAS: sv.ijkToRAS,
    win: sv.win,
    lev: sv.lev,
    dtype: "float32",
    modality: modalityFromSceneName(sv.name)
  };
}
function iraVolumeFromLocalVolume(vol) {
  const meta = vol.meta || {};
  const winMeta = Number(meta.windowWidth);
  const levMeta = Number(meta.windowCenter);
  let win;
  let lev;
  if (Number.isFinite(winMeta) && Number.isFinite(levMeta) && winMeta > 0) {
    win = winMeta;
    lev = levMeta;
  } else {
    const wl = autoWindowLevel(vol.data);
    win = wl.window;
    lev = wl.level;
  }
  const data = new Float32Array(vol.data.length);
  data.set(vol.data);
  let modality = String(meta.modality || "CT").trim().toUpperCase();
  if (modality === "PET") modality = "PT";
  if (modality !== "CT" && modality !== "MR" && modality !== "PT") modality = "CT";
  return {
    vol: data,
    dims: vol.dims,
    ijkToRAS: vol.ijkToRAS.slice(),
    win,
    lev,
    dtype: "float32",
    modality
  };
}
function vrPresetFromSceneVolume(sv) {
  if (!sv.vr) return null;
  return {
    lut: sv.vr.lut,
    clim: [sv.vr.clim[0], sv.vr.clim[1]],
    shade: [
      sv.vr.shade[0],
      sv.vr.shade[1],
      sv.vr.shade[2],
      sv.vr.shade[3]
    ]
  };
}

// render/demos/ira/totalseg-modality.ts
function codingModality(value) {
  if (!value) return "";
  if (typeof value === "string") return value.trim().toUpperCase();
  if (Array.isArray(value)) {
    for (const entry of value) {
      const m = codingModality(entry);
      if (m) return m;
    }
    return "";
  }
  if (typeof value === "object") {
    const obj = value;
    if (obj.coding != null) return codingModality(obj.coding);
    if (obj.code != null) return String(obj.code).trim().toUpperCase();
    if (obj.text != null) return String(obj.text).trim().toUpperCase();
  }
  return "";
}
function normalizeTotalSegModality(raw) {
  const m = raw.trim().toUpperCase();
  if (!m) return "";
  if (m === "CT" || m === "CTAC") return "CT";
  if (m === "MR" || m === "MRI") return "MR";
  if (m === "PT" || m === "PET" || m === "NM") return "other";
  return "other";
}
function modalityFromContext(context) {
  for (const item of context) {
    if (!item || typeof item !== "object") continue;
    const resource = item.resource;
    if (!resource || typeof resource !== "object") continue;
    const res = resource;
    const fromStudy = codingModality(res.modality);
    if (fromStudy) return fromStudy;
    if (Array.isArray(res.series)) {
      for (const series of res.series) {
        if (!series || typeof series !== "object") continue;
        const fromSeries = codingModality(
          series.modality
        );
        if (fromSeries) return fromSeries;
      }
    }
  }
  return "";
}
function modalityFromUrl() {
  try {
    const p = new URLSearchParams(globalThis.location.search);
    return String(p.get("modality") || "").trim().toUpperCase();
  } catch {
    return "";
  }
}
function modalityFromLiveScene() {
  const scene = getIraLiveScene();
  if (!scene?.nodes || typeof scene.nodes !== "object") return "";
  for (const node of Object.values(scene.nodes)) {
    if (!node || typeof node !== "object") continue;
    const name = String(node.name || "");
    if (!name) continue;
    return modalityFromSceneName(name);
  }
  return "";
}
function resolveOpenStudyTotalSegModality() {
  const ctx = getIraImagingStudyOpenContext();
  const fromCtx = normalizeTotalSegModality(modalityFromContext(ctx));
  if (fromCtx === "CT" || fromCtx === "MR") {
    return { modality: fromCtx, source: "ImagingStudy context" };
  }
  if (fromCtx === "other") {
    return {
      modality: null,
      source: "ImagingStudy context",
      reason: "Open study modality is not CT or MR"
    };
  }
  const fromUrl = normalizeTotalSegModality(modalityFromUrl());
  if (fromUrl === "CT" || fromUrl === "MR") {
    return { modality: fromUrl, source: "URL modality" };
  }
  if (fromUrl === "other") {
    return {
      modality: null,
      source: "URL modality",
      reason: "Open study modality is not CT or MR"
    };
  }
  const fromScene = normalizeTotalSegModality(modalityFromLiveScene());
  if (fromScene === "CT" || fromScene === "MR") {
    return { modality: fromScene, source: "scene volume name" };
  }
  return {
    modality: "CT",
    source: "default",
    assumed: true
  };
}

// render/demos/ira/ira-inference.ts
var INFERENCE_SERVERS = [...O];
var MAX_INFERENCE_DICOM_FILES = 300;
var selectedId = null;
var rows = [];
var installed = false;
var floatPos = null;
function $(id) {
  return document.getElementById(id);
}
function clampFloatPos(left, top, panel) {
  const margin = 8;
  const w = panel.offsetWidth || 480;
  const h = panel.offsetHeight || 320;
  const maxL = Math.max(margin, window.innerWidth - w - margin);
  const maxT = Math.max(margin, window.innerHeight - h - margin);
  return {
    left: Math.min(maxL, Math.max(margin, left)),
    top: Math.min(maxT, Math.max(margin, top))
  };
}
function applyFloatPos(host, panel) {
  if (!floatPos) return;
  const next = clampFloatPos(floatPos.left, floatPos.top, panel);
  floatPos = next;
  host.style.left = `${next.left}px`;
  host.style.top = `${next.top}px`;
  host.style.right = "auto";
}
function installInferenceWindowDrag() {
  const host = $("iraInferenceModal");
  const handle = host?.querySelector(
    "[data-ira-inference-drag]"
  );
  const panel = host?.querySelector(".ira-modal-inference");
  if (!host || !handle || !panel || handle.dataset.iraDragBound === "1") return;
  handle.dataset.iraDragBound = "1";
  let dragging = false;
  let startX = 0;
  let startY = 0;
  let originLeft = 0;
  let originTop = 0;
  const onMove = (ev) => {
    if (!dragging) return;
    const next = clampFloatPos(
      originLeft + (ev.clientX - startX),
      originTop + (ev.clientY - startY),
      panel
    );
    floatPos = next;
    host.style.left = `${next.left}px`;
    host.style.top = `${next.top}px`;
    host.style.right = "auto";
  };
  const onUp = (ev) => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("ira-dragging");
    try {
      handle.releasePointerCapture(ev.pointerId);
    } catch {
    }
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    window.removeEventListener("pointercancel", onUp);
  };
  handle.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    const t = ev.target;
    if (t?.closest("button, a, input, select, textarea, label")) return;
    dragging = true;
    handle.classList.add("ira-dragging");
    startX = ev.clientX;
    startY = ev.clientY;
    const rect = host.getBoundingClientRect();
    originLeft = rect.left;
    originTop = rect.top;
    host.style.left = `${originLeft}px`;
    host.style.top = `${originTop}px`;
    host.style.right = "auto";
    try {
      handle.setPointerCapture(ev.pointerId);
    } catch {
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    ev.preventDefault();
  });
}
function volumeFilesFromOpenStudy() {
  const ctx = getIraImagingStudyOpenContext();
  const files = D(ctx);
  const out = [];
  for (const f of files) {
    const role = String(f.role || "").toLowerCase();
    if (role === "seg") continue;
    const url = String(f.url || "").trim();
    if (!url) continue;
    const fileName = String(f.fileName || f.name || "volume").trim() || "volume";
    out.push({ url, fileName });
  }
  return out;
}
function parseIdcPrefixUrl(urlStr) {
  try {
    const u = new URL(urlStr);
    const host = u.hostname.toLowerCase();
    const m = host.match(/^([^.]+)\.s3[.-]/);
    if (!m) return null;
    const path = u.pathname.replace(/^\/+/, "").replace(/\/+$/, "");
    if (!path || path.includes("/")) return null;
    if (/\.(dcm|nii|gz|zip)$/i.test(path)) return null;
    return { bucket: m[1], prefix: path };
  } catch {
    return null;
  }
}
function looksLikePrefixEntry(file) {
  if (file.url.endsWith("/") || file.fileName.endsWith("/")) return true;
  return Boolean(parseIdcPrefixUrl(file.url));
}
async function expandVolumeFilesForSend(files) {
  const out = [];
  for (const file of files) {
    if (!looksLikePrefixEntry(file)) {
      out.push(file);
      continue;
    }
    const parsed = parseIdcPrefixUrl(file.url.replace(/\/?$/, "/"));
    const bucket = parsed?.bucket || (() => {
      try {
        const host = new URL(file.url).hostname.toLowerCase();
        return host.match(/^([^.]+)\.s3[.-]/)?.[1] || "idc-open-data";
      } catch {
        return "idc-open-data";
      }
    })();
    const prefix = parsed?.prefix || file.fileName.replace(/\/+$/, "") || (() => {
      try {
        return new URL(file.url).pathname.replace(/^\/+|\/+$/g, "").split("/")[0];
      } catch {
        return "";
      }
    })();
    if (!prefix) {
      throw new Error(`Cannot expand IDC prefix URL: ${file.url}`);
    }
    const keys = await s3ListKeys(prefix.endsWith("/") ? prefix : `${prefix}/`, bucket);
    if (!keys.length) {
      throw new Error(`No DICOM objects under S3 prefix ${bucket}/${prefix}/`);
    }
    if (keys.length > MAX_INFERENCE_DICOM_FILES) {
      throw new Error(
        `Series has ${keys.length} DICOM files (max ${MAX_INFERENCE_DICOM_FILES})`
      );
    }
    const base = idcS3(bucket);
    for (const key of keys) {
      const fileName = key.split("/").pop() || key;
      out.push({ url: `${base}${key}`, fileName });
    }
  }
  return out;
}
function availabilityLabel(probe, probing) {
  if (probing) return "Checking\u2026";
  if (!probe) return "Unknown";
  if (probe.error) return "Error";
  if (probe.online) {
    const job = probe.job ? ` \xB7 job ${probe.job}` : "";
    return `Online${job}`;
  }
  return "Offline";
}
function orderedRows() {
  const catalogIndex = (def) => INFERENCE_SERVERS.findIndex((s) => s.id === def.id);
  const sortKey = (row) => {
    if (row.probe?.online) return 0;
    if (row.probing || !row.probe) return 1;
    return 2;
  };
  return [...rows].sort((a, b) => {
    const d = sortKey(a) - sortKey(b);
    if (d !== 0) return d;
    return catalogIndex(a.def) - catalogIndex(b.def);
  });
}
var totalSegUiBound = false;
var totalSegModality = "CT";
var totalSegAssumed = false;
var totalSegUnsupportedReason = "";
function fillSelect(select, options, selected) {
  select.replaceChildren();
  for (const opt of options) {
    const el = document.createElement("option");
    el.value = opt.value;
    el.textContent = opt.label;
    if (opt.value === selected) el.selected = true;
    select.append(el);
  }
  if (!select.value && options[0]) select.value = options[0].value;
}
function syncTotalSegTaskSelect(preferredTaskId) {
  const groupSel = $("iraTotalSegGroup");
  const taskSel = $("iraTotalSegTask");
  if (!groupSel || !taskSel || !totalSegModality) {
    if (taskSel) taskSel.replaceChildren();
    return;
  }
  const group = groupSel.value || "total";
  const tasks = tasksInGroup(group, totalSegModality);
  const preferred = preferredTaskId && tasks.some((t) => t.id === preferredTaskId) ? preferredTaskId : tasks[0]?.id || "";
  fillSelect(
    taskSel,
    tasks.map((t) => ({ value: t.id, label: taskOptionLabel(t) })),
    preferred
  );
  const taskRow = taskSel.closest(".ira-totalseg-row");
  if (taskRow) taskRow.hidden = tasks.length <= 1;
}
function syncTotalSegGroupSelect(preferredTaskId) {
  const groupSel = $("iraTotalSegGroup");
  if (!groupSel || !totalSegModality) {
    if (groupSel) groupSel.replaceChildren();
    return;
  }
  const groups = groupsForModality(totalSegModality);
  const preferredTask = preferredTaskId ? getTotalSegTask(preferredTaskId) : void 0;
  const preferredGroup = preferredTask && preferredTask.modality === totalSegModality ? preferredTask.group : groups.includes("total") ? "total" : groups[0] || "";
  fillSelect(
    groupSel,
    groups.map((g) => ({
      value: g,
      label: TOTAL_SEG_GROUP_LABELS[g] || g
    })),
    preferredGroup
  );
  syncTotalSegTaskSelect(preferredTaskId || defaultTaskId(totalSegModality));
}
function clearTotalSegHits() {
  const hits = $("iraTotalSegHits");
  if (!hits) return;
  hits.replaceChildren();
  hits.hidden = true;
}
function selectTotalSegTask(task) {
  if (!totalSegModality || task.modality !== totalSegModality) return;
  syncTotalSegGroupSelect(task.id);
  clearTotalSegHits();
  const find = $("iraTotalSegFind");
  if (find) find.value = "";
}
function renderTotalSegHits(query) {
  const hitsEl = $("iraTotalSegHits");
  if (!hitsEl || !totalSegModality) {
    clearTotalSegHits();
    return;
  }
  const hits = searchClasses(query, totalSegModality);
  hitsEl.replaceChildren();
  if (!hits.length) {
    hitsEl.hidden = true;
    return;
  }
  for (const hit of hits) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ira-totalseg-hit";
    btn.setAttribute("role", "option");
    const cls = document.createElement("div");
    cls.className = "ira-totalseg-hit-class";
    cls.textContent = hit.className;
    const taskLine = document.createElement("div");
    taskLine.className = "ira-totalseg-hit-task";
    taskLine.textContent = `${TOTAL_SEG_GROUP_LABELS[hit.task.group] || hit.task.group} \xB7 ${taskOptionLabel(hit.task)} (${hit.task.id})`;
    btn.append(cls, taskLine);
    btn.addEventListener("click", () => selectTotalSegTask(hit.task));
    hitsEl.append(btn);
  }
  hitsEl.hidden = false;
}
function currentTotalSegTaskId() {
  const taskSel = $("iraTotalSegTask");
  const raw = String(taskSel?.value || "").trim();
  return raw || void 0;
}
function refreshTotalSegModalityUi(opts) {
  const prevModality = totalSegModality;
  const prevTask = currentTotalSegTaskId();
  const resolved = resolveOpenStudyTotalSegModality();
  totalSegUnsupportedReason = "";
  totalSegAssumed = false;
  if (resolved.modality) {
    totalSegModality = resolved.modality;
    totalSegAssumed = Boolean(resolved.assumed);
  } else {
    totalSegModality = null;
    totalSegUnsupportedReason = resolved.reason;
  }
  const modEl = $("iraTotalSegModality");
  if (modEl) {
    if (totalSegModality) {
      modEl.textContent = totalSegAssumed ? `${totalSegModality} (assumed)` : totalSegModality;
    } else {
      modEl.textContent = "unsupported";
    }
  }
  const modalityChanged = prevModality !== totalSegModality;
  const find = $("iraTotalSegFind");
  if (find) {
    find.disabled = !totalSegModality;
    if (modalityChanged || opts?.resetSelection) {
      find.value = "";
      clearTotalSegHits();
    }
  } else {
    clearTotalSegHits();
  }
  if (totalSegModality) {
    const keep = !opts?.resetSelection && prevTask && getTotalSegTask(prevTask)?.modality === totalSegModality ? prevTask : defaultTaskId(totalSegModality);
    syncTotalSegGroupSelect(keep);
  } else {
    const groupSel = $("iraTotalSegGroup");
    const taskSel = $("iraTotalSegTask");
    groupSel?.replaceChildren();
    taskSel?.replaceChildren();
  }
}
function ensureTotalSegUiBindings() {
  if (totalSegUiBound) return;
  totalSegUiBound = true;
  const groupSel = $("iraTotalSegGroup");
  const taskSel = $("iraTotalSegTask");
  const find = $("iraTotalSegFind");
  groupSel?.addEventListener("change", () => syncTotalSegTaskSelect());
  find?.addEventListener("input", () => {
    renderTotalSegHits(find.value);
  });
  find?.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") clearTotalSegHits();
  });
}
function syncTotalSegOptionsVisibility() {
  const panel = $("iraTotalSegOptions");
  if (!panel) return;
  const show = selectedId === "totalseg";
  panel.hidden = !show;
  if (show) {
    ensureTotalSegUiBindings();
    refreshTotalSegModalityUi({ resetSelection: false });
  }
}
function readTotalSegOptions() {
  const select = $("iraTotalSegTask");
  const fallback = totalSegModality ? defaultTaskId(totalSegModality) : "total";
  const raw = String(select?.value || fallback).trim();
  const known = getTotalSegTask(raw);
  const task = known ? raw : fallback;
  return {
    task,
    // Prefer --fast when TotalSeg allows it for this task (catalog supportsFast).
    fast: taskSupportsFast(task)
  };
}
function renderList() {
  const list = $("iraInferenceList");
  if (!list) return;
  list.replaceChildren();
  for (const row of orderedRows()) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ira-inference-row";
    if (row.def.id === selectedId) btn.classList.add("is-selected");
    btn.dataset.id = row.def.id;
    const title = document.createElement("div");
    title.className = "ira-inference-row-title";
    title.textContent = row.def.title;
    const desc = document.createElement("div");
    desc.className = "ira-inference-desc";
    desc.textContent = row.def.summary || row.def.capabilities || "";
    const meta = document.createElement("div");
    meta.className = "ira-inference-row-meta";
    const status2 = document.createElement("span");
    status2.className = "ira-inference-avail";
    if (row.probe?.online) status2.classList.add("is-online");
    else if (row.probe && !row.probing) status2.classList.add("is-offline");
    status2.textContent = availabilityLabel(row.probe, row.probing);
    meta.append(status2);
    const loc = document.createElement("span");
    loc.className = "ira-inference-location";
    loc.textContent = row.def.location;
    meta.append(loc);
    btn.append(title, desc, meta);
    btn.addEventListener("click", () => {
      selectedId = row.def.id;
      renderList();
      updateDetail();
    });
    list.append(btn);
  }
}
function updateDetail() {
  const detail = $("iraInferenceDetail");
  const runBtn = $("iraInferenceRunBtn");
  const note = $("iraInferenceNote");
  const row = rows.find((r) => r.def.id === selectedId);
  if (!detail || !runBtn) return;
  if (!row) {
    detail.className = "ira-inference-detail";
    detail.textContent = "Select an inference server.";
    runBtn.disabled = true;
    if (note) note.textContent = "";
    syncTotalSegOptionsVisibility();
    return;
  }
  const opLines = [];
  if (row.probe?.error) opLines.push(`Probe error: ${row.probe.error}`);
  if (row.probe?.items?.length) {
    for (const it2 of row.probe.items) {
      const k2 = String(it2.key ?? "").trim();
      const v2 = String(it2.value ?? "").trim();
      if (k2 && v2 && k2 !== "availability") opLines.push(`${k2}: ${v2}`);
    }
  }
  detail.className = "ira-inference-detail ira-inference-detail-card";
  detail.innerHTML = fi(row.def) + (opLines.length ? `<pre class="ira-inference-op-notes">${opLines.map(
    (l) => l.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  ).join("\n")}</pre>` : "");
  const info = getIraCastSessionInfo();
  const connected = info.connection === "connected";
  const files = volumeFilesFromOpenStudy();
  const online = Boolean(row.probe?.online);
  const needsNifti = row.def.hubEvent === "nifti-send";
  const hasNifti = files.some((f) => /\.nii(\.gz)?$/i.test(f.fileName) || /\.nii(\.gz)?(\?|$)/i.test(f.url));
  syncTotalSegOptionsVisibility();
  let disableReason = "";
  if (!connected) disableReason = "Connect to Cast (open from the worklist) to run inference.";
  else if (!online) disableReason = "Server is offline \u2014 refresh status or start the resource server.";
  else if (!files.length) disableReason = "Open a study first (ImagingStudy context with volume files).";
  else if (needsNifti && !hasNifti) {
    disableReason = "This server expects NIfTI; the open study only has DICOM/prefix URLs. Use Lung Screening or Total Segmentator, or send NIfTI.";
  } else if (row.def.id === "totalseg" && !totalSegModality) {
    disableReason = totalSegUnsupportedReason || "TotalSegmentator requires a CT or MR study.";
  }
  runBtn.disabled = Boolean(disableReason);
  if (note) {
    if (disableReason) {
      note.textContent = disableReason;
    } else if (row.def.id === "totalseg") {
      note.textContent = totalSegAssumed ? "Assuming CT \u2014 open an MR study for MR tasks." : "";
    } else {
      note.textContent = "Ready to run on the open study.";
    }
  }
}
function appendLog(line) {
  const pre = $("iraInferenceLog");
  if (!pre) return;
  const text = String(line || "").replace(/\s+$/, "");
  if (!text) return;
  const stamped = /^\[[^\]]+\]\s/.test(text) ? text : `[${(/* @__PURE__ */ new Date()).toLocaleTimeString()}] ${text}`;
  pre.textContent = `${pre.textContent || ""}${pre.textContent ? "\n" : ""}${stamped}`;
  pre.scrollTop = pre.scrollHeight;
}
var closeAfterJobTimer = null;
var keepDialogOpenAfterJob = false;
function scheduleCloseAfterJobFinished() {
  if (closeAfterJobTimer != null) window.clearTimeout(closeAfterJobTimer);
  closeAfterJobTimer = window.setTimeout(() => {
    closeAfterJobTimer = null;
    closeMedicalInferenceDialog();
  }, 700);
}
function onInferenceStatusUpdate(message, level) {
  appendLog(message);
  if (level === "error" || /No segmentation to save/i.test(message) || /Segmentation failed/i.test(message) || /no output produced/i.test(message) || /Failed to publish/i.test(message)) {
    keepDialogOpenAfterJob = true;
  }
  if (/\bJob finished\b/i.test(message)) {
    if (!keepDialogOpenAfterJob) {
      scheduleCloseAfterJobFinished();
    }
    keepDialogOpenAfterJob = false;
  }
}
function selectPreferredOnlineServer() {
  const online = orderedRows().filter((r) => r.probe?.online);
  if (!online.length) return;
  if (selectedId && online.some((r) => r.def.id === selectedId)) return;
  selectedId = online[0].def.id;
}
async function probeAll() {
  const info = getIraCastSessionInfo();
  const client2 = getIraCastClient();
  if (info.connection !== "connected" || !client2) {
    for (const row of rows) {
      row.probing = false;
      row.probe = { online: false, items: [], error: "Not connected to Cast" };
    }
    renderList();
    updateDetail();
    appendLog("Not connected \u2014 cannot probe inference servers.");
    return;
  }
  for (const row of rows) {
    row.probing = true;
    row.probe = null;
  }
  renderList();
  updateDetail();
  appendLog("Probing inference servers\u2026");
  try {
    const results = await mi(client2, {
      subscriberName: info.subscriberName,
      subscriberProductName: "SLICERLIVE-IRA",
      subscriberActor: "ID",
      topic: info.topic || void 0
    });
    const byId = new Map(results.map((r) => [r.server.id, r.probe]));
    for (const row of rows) {
      row.probe = byId.get(row.def.id) || {
        online: false,
        items: [],
        error: "No STATUS response"
      };
      row.probing = false;
    }
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    for (const row of rows) {
      row.probe = { online: false, items: [], error };
      row.probing = false;
    }
  }
  selectPreferredOnlineServer();
  renderList();
  updateDetail();
  const online = rows.filter((r) => r.probe?.online).map((r) => r.def.title);
  appendLog(
    online.length ? `Online: ${online.join(", ")}` : "No inference servers responded online."
  );
}
async function runSelected() {
  const row = rows.find((r) => r.def.id === selectedId);
  if (!row) return;
  const runBtn = $("iraInferenceRunBtn");
  if (runBtn) runBtn.disabled = true;
  keepDialogOpenAfterJob = false;
  if (closeAfterJobTimer != null) {
    window.clearTimeout(closeAfterJobTimer);
    closeAfterJobTimer = null;
  }
  appendLog(`Running ${row.def.title} (${row.def.product})\u2026`);
  try {
    const rawFiles = volumeFilesFromOpenStudy();
    appendLog(
      rawFiles.some(looksLikePrefixEntry) ? "Expanding IDC series prefix to DICOM instance URLs\u2026" : `Sending ${rawFiles.length} volume file(s)\u2026`
    );
    const files = await expandVolumeFilesForSend(rawFiles);
    appendLog(`Publishing ${files.length} file(s) as ${row.def.hubEvent}\u2026`);
    const contextExtras = row.def.id === "totalseg" ? readTotalSegOptions() : void 0;
    if (contextExtras) {
      appendLog(
        `TotalSeg options: task=${contextExtras.task}, fast=${contextExtras.fast}`
      );
    }
    const result = await publishInferenceJob({
      product: row.def.product,
      hubEvent: row.def.hubEvent,
      files,
      contextExtras
    });
    appendLog(result.ok ? `Published ${row.def.hubEvent} \u2192 ${row.def.product}` : `Publish failed: ${result.detail}`);
    if (result.ok) appendLog("Watch this log for status-update lines from the server.");
    else keepDialogOpenAfterJob = true;
  } catch (err) {
    keepDialogOpenAfterJob = true;
    appendLog(`Run error: ${err instanceof Error ? err.message : String(err)}`);
  }
  updateDetail();
}
function openMedicalInferenceDialog() {
  const modal = $("iraInferenceModal");
  if (!modal) {
    console.error("[ira] Medical Inference modal missing (#iraInferenceModal)");
    return;
  }
  rows = INFERENCE_SERVERS.map((def) => ({ def, probe: null, probing: false }));
  selectedId = null;
  const log = $("iraInferenceLog");
  if (log) log.textContent = "";
  installInferenceWindowDrag();
  modal.hidden = false;
  const panel = modal.querySelector(".ira-modal-inference");
  if (panel) applyFloatPos(modal, panel);
  try {
    ensureTotalSegUiBindings();
    renderList();
    updateDetail();
  } catch (err) {
    console.error("[ira] Medical Inference dialog render failed", err);
  }
  setIraStatusUpdateHandler(onInferenceStatusUpdate);
  void probeAll();
}
function closeMedicalInferenceDialog() {
  if (closeAfterJobTimer != null) {
    window.clearTimeout(closeAfterJobTimer);
    closeAfterJobTimer = null;
  }
  const modal = $("iraInferenceModal");
  if (modal) modal.hidden = true;
  setIraStatusUpdateHandler(null);
}
function installMedicalInferenceUi() {
  if (installed) return;
  installed = true;
  ensureTotalSegUiBindings();
  installInferenceWindowDrag();
  $("iraInferenceRefreshBtn")?.addEventListener("click", () => {
    void probeAll();
  });
  $("iraInferenceRunBtn")?.addEventListener("click", () => {
    void runSelected();
  });
}

// logic/readers/dicom-local.ts
function looksDicom(bytes, name) {
  if (bytes.length >= 132 && String.fromCharCode(bytes[128], bytes[129], bytes[130], bytes[131]) === "DICM") return true;
  return /\.dcm$/i.test(name) || /^[0-9.]+$/.test(name) || name.toLowerCase() === "dicomdir" === false && /(^|\/)[^./]+$/.test(name) && bytes.length > 256;
}
async function* walkDir(dir, prefix = "") {
  for await (const [name, handle] of dir.entries()) {
    const h = handle;
    if (h.kind === "directory") yield* walkDir(h, `${prefix}${name}/`);
    else {
      try {
        yield { name: `${prefix}${name}`, file: await h.getFile() };
      } catch {
      }
    }
  }
}
async function indexFiles(files, onProgress) {
  const buffers = [];
  let scanned = 0, dicom = 0;
  for (const f of files) {
    scanned++;
    const buf = await f.arrayBuffer();
    if (looksDicom(new Uint8Array(buf, 0, Math.min(200, buf.byteLength)), f.name)) {
      buffers.push(buf);
      dicom++;
    }
    if (scanned % 20 === 0) onProgress?.({ scanned, dicom, note: f.name });
  }
  onProgress?.({ scanned, dicom, note: "parsing headers\u2026" });
  return seriesEntries(buffers);
}
async function indexDirectory(dir, onProgress) {
  const files = [];
  for await (const { file } of walkDir(dir)) files.push(file);
  return indexFiles(files, onProgress);
}
async function seriesEntries(buffers) {
  const groups = groupSeries(await parseInstances(buffers));
  return groups.map((g) => {
    const e = {
      seriesInstanceUID: g.seriesInstanceUID,
      studyInstanceUID: g.instances[0].studyInstanceUID,
      patientName: g.instances[0].patientName,
      modality: g.modality,
      description: g.description,
      count: g.instances.length,
      buffers: []
    };
    e._instances = g.instances;
    return e;
  });
}
function loadEntry(entry) {
  const inst = entry._instances;
  if (inst && inst.length) return reconstructSeries(inst);
  throw new Error("series has no parsed instances (re-index the folder)");
}

// logic/readers/unzip-files.ts
var enc = new TextDecoder("utf-8");
function u16(view, o) {
  return view.getUint16(o, true);
}
function u32(view, o) {
  return view.getUint32(o, true);
}
async function inflateRaw(bytes) {
  const ds = new DecompressionStream("deflate-raw");
  return new Uint8Array(
    await new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer()
  );
}
function findCentralDirectory(view) {
  const bytes = view.byteLength;
  const start = Math.max(0, bytes - 22 - 65535);
  for (let i = bytes - 22; i >= start; i--) {
    if (u32(view, i) !== 101010256) continue;
    return { count: u16(view, i + 10), offset: u32(view, i + 16) };
  }
  throw new Error("ZIP end-of-central-directory not found");
}
async function unzipToFiles(buf) {
  const bytes = new Uint8Array(buf);
  const view = new DataView(buf);
  const { offset: cdStart, count } = findCentralDirectory(view);
  const files = [];
  let o = cdStart;
  for (let n = 0; n < count; n++) {
    if (o + 46 > bytes.length || u32(view, o) !== 33639248) {
      throw new Error("ZIP central directory truncated or corrupt");
    }
    const method = u16(view, o + 10);
    const compSize = u32(view, o + 20);
    const uncompSize = u32(view, o + 24);
    const nameLen = u16(view, o + 28);
    const extraLen = u16(view, o + 30);
    const commentLen = u16(view, o + 32);
    const localOff = u32(view, o + 42);
    const name = enc.decode(bytes.subarray(o + 46, o + 46 + nameLen));
    o += 46 + nameLen + extraLen + commentLen;
    if (!name || name.endsWith("/")) continue;
    if (localOff + 30 > bytes.length || u32(view, localOff) !== 67324752) {
      throw new Error(`ZIP local header missing for ${name}`);
    }
    const localNameLen = u16(view, localOff + 26);
    const localExtraLen = u16(view, localOff + 28);
    const dataStart = localOff + 30 + localNameLen + localExtraLen;
    const dataEnd = dataStart + compSize;
    if (dataEnd > bytes.length) throw new Error(`ZIP truncated (${name})`);
    const payload = bytes.subarray(dataStart, dataEnd);
    let raw;
    if (method === 0) {
      raw = payload.slice();
    } else if (method === 8) {
      raw = await inflateRaw(payload);
    } else {
      throw new Error(`ZIP compression method ${method} not supported (${name})`);
    }
    if (uncompSize && raw.byteLength !== uncompSize) {
      throw new Error(
        `ZIP size mismatch for ${name}: got ${raw.byteLength}, expected ${uncompSize}`
      );
    }
    const base = name.split("/").pop() || name;
    files.push(new File([raw], base));
  }
  if (!files.length) throw new Error("No files found in ZIP");
  return files;
}
async function unzipFileToFiles(file) {
  return unzipToFiles(await file.arrayBuffer());
}

// render/demos/ira/ira-header.ts
var ID_ACTOR2 = "ID";
var THEME_KEY = "pw46.ira.theme";
var THEME_SLICERLIVE = "slicerlive";
var THEME_OHIF = "ohif";
function getStoredTheme() {
  try {
    const v2 = localStorage.getItem(THEME_KEY)?.trim();
    if (v2 === THEME_OHIF || v2 === THEME_SLICERLIVE) return v2;
  } catch {
  }
  return THEME_SLICERLIVE;
}
function setStoredTheme(theme) {
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
  }
}
function applyTheme(theme) {
  const next = theme === THEME_OHIF ? THEME_OHIF : THEME_SLICERLIVE;
  document.documentElement.dataset.theme = next;
  setStoredTheme(next);
}
function fillSessionDl(dl) {
  const info = getIraCastSessionInfo();
  const rows2 = [
    ["Hub", info.hubEndpoint || "\u2014"],
    ["Topic", info.topic || "(none)"],
    ["Subscriber", info.subscriberName || "\u2014"],
    ["User", info.userName || "\u2014"],
    ["Status", info.connectionDetail]
  ];
  dl.replaceChildren();
  for (const [k2, v2] of rows2) {
    const dt2 = document.createElement("dt");
    dt2.textContent = k2;
    const dd = document.createElement("dd");
    dd.textContent = v2;
    dl.append(dt2, dd);
  }
}
function fillSceneViewPre(pre) {
  const scene = getIraLiveScene();
  if (!scene) {
    pre.textContent = "No LiveScene loaded";
    return;
  }
  try {
    pre.textContent = JSON.stringify(scene, null, 2);
  } catch {
    pre.textContent = String(scene);
  }
}
function refreshSceneViewButton() {
  const btn = document.getElementById(
    "iraSceneViewBtn"
  );
  if (!btn) return;
  const hasScene = Boolean(getIraLiveScene());
  btn.disabled = !hasScene;
  btn.title = hasScene ? "Scene View" : "No LiveScene loaded";
}
var localStudyPick = null;
var onMountLocalSeries;
function setLoadLocalStatus(msg, show = true) {
  const el = document.getElementById("iraLoadLocalStatus");
  if (!el) return;
  el.textContent = msg;
  el.hidden = !show || !msg;
}
function clearLoadLocalSeries() {
  const host = document.getElementById("iraLoadLocalSeries");
  if (!host) return;
  host.replaceChildren();
  host.hidden = true;
}
function setLocalStudySelection(pick) {
  localStudyPick = pick;
  const label = document.getElementById("iraLoadLocalSelection");
  const confirm = document.getElementById(
    "iraLoadLocalConfirmBtn"
  );
  if (label) {
    if (!pick) label.textContent = "No folder or ZIP selected.";
    else if (pick.kind === "folder") label.textContent = `Folder: ${pick.name}`;
    else label.textContent = `ZIP: ${pick.name}`;
  }
  if (confirm) confirm.disabled = !pick;
  clearLoadLocalSeries();
  setLoadLocalStatus("", false);
}
function openLoadLocalStudyDialog() {
  const modal = document.getElementById("iraLoadLocalModal");
  const anon = document.getElementById(
    "iraLoadLocalAnonymize"
  );
  if (anon) anon.checked = true;
  setLocalStudySelection(null);
  if (modal) {
    modal.removeAttribute("hidden");
    modal.hidden = false;
  }
}
function closeLoadLocalStudyDialog() {
  const modal = document.getElementById("iraLoadLocalModal");
  if (modal) {
    modal.setAttribute("hidden", "");
    modal.hidden = true;
  }
  setLocalStudySelection(null);
}
function seriesButtonLabel(entry, anonymized) {
  const modality = String(entry.modality || "?").trim() || "?";
  const desc = String(entry.description || "").trim();
  const patient = anonymized ? "Anonymous" : String(entry.patientName || "").trim() || "Unknown patient";
  const title = desc ? `${modality} \xB7 ${desc}` : `${modality} \xB7 ${entry.seriesInstanceUID.slice(-12)}`;
  const meta = `${patient} \xB7 ${entry.count} image${entry.count === 1 ? "" : "s"}`;
  return { title, meta };
}
async function indexLocalPick(pick, onProgress) {
  if (pick.kind === "folder") {
    return indexDirectory(pick.handle, (p) => {
      onProgress(
        `Scanning: ${p.dicom} DICOM / ${p.scanned} files${p.note ? ` \u2014 ${p.note}` : ""}`
      );
    });
  }
  onProgress("Unpacking ZIP\u2026");
  const files = await unzipFileToFiles(pick.file);
  return indexFiles(files, (p) => {
    onProgress(
      `Scanning: ${p.dicom} DICOM / ${p.scanned} files${p.note ? ` \u2014 ${p.note}` : ""}`
    );
  });
}
async function mountChosenSeries(entry, anonymized) {
  if (!onMountLocalSeries) {
    throw new Error("Local study loader is not ready yet");
  }
  setLoadLocalStatus("Reconstructing series\u2026");
  await onMountLocalSeries(entry, { anonymized });
  closeLoadLocalStudyDialog();
}
function renderSeriesPicker(entries, anonymized) {
  const host = document.getElementById("iraLoadLocalSeries");
  if (!host) return;
  host.replaceChildren();
  host.hidden = false;
  for (const entry of entries) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ira-load-local-series-btn";
    const { title, meta } = seriesButtonLabel(entry, anonymized);
    const titleEl = document.createElement("span");
    titleEl.textContent = title;
    const metaEl = document.createElement("span");
    metaEl.className = "ira-load-local-series-meta";
    metaEl.textContent = meta;
    btn.append(titleEl, metaEl);
    btn.title = `${title} \u2014 ${meta}`;
    btn.addEventListener("click", () => {
      host.querySelectorAll("button").forEach((b) => {
        b.disabled = true;
      });
      void mountChosenSeries(entry, anonymized).catch((err) => {
        console.error("[ira] local series mount failed", err);
        setLoadLocalStatus(
          `Error: ${err?.message ?? String(err)}`
        );
        host.querySelectorAll("button").forEach((b) => {
          b.disabled = false;
        });
      });
    });
    host.append(btn);
  }
  setLoadLocalStatus(`Select a series (${entries.length} found).`);
}
function installLoadLocalStudyUi() {
  const zipInput = document.getElementById(
    "iraLoadLocalZipInput"
  );
  document.getElementById("iraLoadLocalStudyBtn")?.addEventListener("click", (ev) => {
    ev.stopPropagation();
    openLoadLocalStudyDialog();
  });
  document.getElementById("iraLoadLocalCancelBtn")?.addEventListener("click", () => closeLoadLocalStudyDialog());
  document.getElementById("iraLoadLocalFolderBtn")?.addEventListener("click", async () => {
    const picker = globalThis.showDirectoryPicker;
    if (!picker) {
      const label = document.getElementById("iraLoadLocalSelection");
      if (label) {
        label.textContent = "Folder picker unavailable in this browser \u2014 choose a ZIP instead.";
      }
      return;
    }
    try {
      const handle = await picker({ id: "ira-local-dicom" });
      setLocalStudySelection({
        kind: "folder",
        name: handle.name || "Selected folder",
        handle
      });
    } catch (err) {
      if (err.name !== "AbortError") {
        console.error("[ira] folder pick failed", err);
      }
    }
  });
  document.getElementById("iraLoadLocalZipBtn")?.addEventListener("click", async () => {
    const picker = globalThis.showOpenFilePicker;
    if (picker) {
      try {
        const handles = await picker({
          multiple: false,
          types: [
            {
              description: "ZIP archive",
              accept: { "application/zip": [".zip"] }
            }
          ]
        });
        const file = await handles[0]?.getFile();
        if (file) {
          setLocalStudySelection({ kind: "zip", name: file.name, file });
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          console.error("[ira] zip pick failed", err);
        }
      }
      return;
    }
    zipInput?.click();
  });
  zipInput?.addEventListener("change", () => {
    const file = zipInput.files?.[0];
    if (file) setLocalStudySelection({ kind: "zip", name: file.name, file });
    zipInput.value = "";
  });
  document.getElementById("iraLoadLocalConfirmBtn")?.addEventListener("click", () => {
    const pick = localStudyPick;
    if (!pick) return;
    const anon = document.getElementById(
      "iraLoadLocalAnonymize"
    );
    const anonymized = Boolean(anon?.checked);
    const confirm = document.getElementById(
      "iraLoadLocalConfirmBtn"
    );
    const folderBtn = document.getElementById(
      "iraLoadLocalFolderBtn"
    );
    const zipBtn = document.getElementById(
      "iraLoadLocalZipBtn"
    );
    if (confirm) confirm.disabled = true;
    if (folderBtn) folderBtn.disabled = true;
    if (zipBtn) zipBtn.disabled = true;
    clearLoadLocalSeries();
    setLoadLocalStatus("Scanning\u2026");
    void indexLocalPick(pick, setLoadLocalStatus).then(async (entries) => {
      if (!entries.length) {
        setLoadLocalStatus("No DICOM image series found.");
        return;
      }
      if (entries.length === 1) {
        await mountChosenSeries(entries[0], anonymized);
        return;
      }
      renderSeriesPicker(entries, anonymized);
    }).catch((err) => {
      console.error("[ira] local study index failed", err);
      setLoadLocalStatus(
        `Error: ${err?.message ?? String(err)}`
      );
    }).finally(() => {
      if (confirm) confirm.disabled = !localStudyPick;
      if (folderBtn) folderBtn.disabled = false;
      if (zipBtn) zipBtn.disabled = false;
    });
  });
}
function iraChipStatus(place) {
  if (place.isSelf && place.isLeading) return "Host";
  if (place.isSelf) return "You";
  if (place.isFollowing) return "Following";
  return String(place.statusLabel || "").trim();
}
function renderConferenceChips(view) {
  const host = document.getElementById("iraConferenceChips");
  if (!host) return;
  host.replaceChildren();
  if (!view?.places?.length) {
    host.hidden = true;
    return;
  }
  host.hidden = false;
  for (const place of view.places) {
    const chip = document.createElement("span");
    chip.className = "ira-conference-chip";
    if (place.isSelf) chip.classList.add("ira-conference-chip-self");
    if (place.isLeading) chip.classList.add("ira-conference-chip-leading");
    if (place.isFollowing) chip.classList.add("ira-conference-chip-following");
    const name = document.createElement("span");
    name.className = "ira-conference-chip-name";
    name.textContent = place.label;
    const tip = [place.label, place.statusLabel, place.placeId].map((s) => String(s || "").trim()).filter(Boolean).join(" \u2014 ");
    name.title = tip;
    chip.title = tip;
    chip.append(name);
    const short = iraChipStatus(place);
    if (short) {
      const status2 = document.createElement("span");
      status2.className = "ira-conference-chip-status";
      status2.textContent = short;
      chip.append(status2);
    }
    host.append(chip);
  }
}
function refreshLeaveConferenceButton(view) {
  const btn = document.getElementById(
    "iraLeaveConferenceBtn"
  );
  if (!btn) return;
  const info = getIraCastSessionInfo();
  const active = Boolean(view?.places?.length);
  const connected = info.connection === "connected";
  if (!active) {
    btn.hidden = true;
    btn.disabled = true;
    btn.textContent = "Leave conference";
    btn.title = "Leave conference";
    btn.setAttribute("aria-label", "Leave conference");
    return;
  }
  const hostAction = view?.selfRole === "leading";
  const label = hostAction ? "End conference" : "Leave conference";
  btn.hidden = false;
  btn.disabled = !connected;
  btn.textContent = label;
  btn.title = label;
  btn.setAttribute("aria-label", label);
}
function refreshStartConferenceButton(view) {
  const btn = document.getElementById(
    "iraStartConferenceBtn"
  );
  if (!btn) return;
  const active = Boolean(view);
  const connected = getIraCastSessionInfo().connection === "connected";
  btn.hidden = active;
  btn.disabled = active || !connected;
  btn.title = connected ? "Start a conference" : "Connect to Cast hub to start a conference";
}
function refreshRadio() {
  const statusEl = document.getElementById("iraCastStatus");
  if (statusEl) applyIraCastRadioVisual(statusEl);
  const view = getIraCastSessionInfo().conference;
  renderConferenceChips(view);
  refreshStartConferenceButton(view);
  refreshLeaveConferenceButton(view);
}
function installIraHeader(opts = {}) {
  onMountLocalSeries = opts.onMountLocalSeries;
  applyTheme(getStoredTheme());
  const statusEl = document.getElementById("iraCastStatus");
  if (statusEl) {
    statusEl.innerHTML = On({ slashId: "iraCastRadioSlash" });
  }
  refreshRadio();
  setIraCastStatusListener(() => refreshRadio());
  refreshSceneViewButton();
  setIraLiveSceneChangeListener(() => refreshSceneViewButton());
  const castBtn = document.getElementById("iraCastStatusBtn");
  const castMenu = document.getElementById("iraCastStatusMenu");
  const helpBtn = document.getElementById("iraSettingsBtn");
  const sessionModal = document.getElementById("iraSessionModal");
  const sceneViewModal = document.getElementById("iraSceneViewModal");
  const aboutModal = document.getElementById("iraAboutModal");
  const loadLocalModal = document.getElementById("iraLoadLocalModal");
  const sessionDl = document.getElementById("iraSessionDl");
  const sceneViewPre = document.getElementById("iraSceneViewPre");
  installMedicalInferenceUi();
  installLoadLocalStudyUi();
  if (!castBtn || !castMenu) return;
  const closeMenus = () => {
    castMenu.hidden = true;
    castBtn.setAttribute("aria-expanded", "false");
  };
  let openConferenceDialog = () => {
    console.warn("[ira] conference dialog not available");
  };
  try {
    const conferenceDialog = Qn({
      classPrefix: "cast-conference",
      getSession: () => {
        const info = getIraCastSessionInfo();
        return {
          hubEndpoint: info.hubEndpoint,
          topic: info.topic,
          subscriberName: info.subscriberName,
          userName: info.userName,
          connected: info.connection === "connected"
        };
      },
      onConferenceChange: (view) => {
        setIraConferenceView(view);
      }
    });
    openConferenceDialog = () => conferenceDialog.open();
  } catch (err) {
    console.error("[ira] conference dialog init failed", err);
  }
  try {
    const inviteController = tr({
      classPrefix: "cast-conference-invite",
      getSession: () => {
        const info = getIraCastSessionInfo();
        return {
          hubEndpoint: info.hubEndpoint,
          topic: info.topic,
          subscriberName: info.subscriberName,
          userName: info.userName,
          connected: info.connection === "connected"
        };
      },
      shouldShow: () => er({
        selfActors: [ID_ACTOR2]
      }),
      onConferenceChange: (view) => {
        setIraConferenceView(view);
      },
      onFollowChange: (followHost, invite) => {
        setIraConferenceFollowHost(followHost);
        const hostTopic = String(invite?.hostTopic || "").trim();
        if (followHost && hostTopic) {
          void requestIraStatusFromHostWorklist(hostTopic);
        }
      }
    });
    setIraConferenceHubHandler(
      (message) => inviteController.handleHubMessage(message)
    );
  } catch (err) {
    console.error("[ira] conference invite init failed", err);
  }
  castBtn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    const open = castMenu.hidden;
    closeMenus();
    castMenu.hidden = !open;
    castBtn.setAttribute("aria-expanded", open ? "true" : "false");
  });
  helpBtn?.addEventListener("click", (ev) => {
    ev.stopPropagation();
    closeMenus();
    openChromeHelp();
  });
  document.addEventListener("click", () => closeMenus());
  castMenu.addEventListener("click", (ev) => ev.stopPropagation());
  document.getElementById("iraStartConferenceBtn")?.addEventListener("click", (ev) => {
    ev.stopPropagation();
    closeMenus();
    if (getIraCastSessionInfo().conference) return;
    openConferenceDialog();
  });
  document.getElementById("iraMedicalInferenceBtn")?.addEventListener("click", (ev) => {
    ev.stopPropagation();
    closeMenus();
    openMedicalInferenceDialog();
  });
  document.getElementById("iraLeaveConferenceBtn")?.addEventListener("click", (ev) => {
    ev.stopPropagation();
    closeMenus();
    const info = getIraCastSessionInfo();
    const view = info.conference;
    if (!view || info.connection !== "connected") return;
    const leaveBtn = document.getElementById(
      "iraLeaveConferenceBtn"
    );
    const hostAction = view.selfRole === "leading";
    if (leaveBtn) {
      leaveBtn.disabled = true;
      leaveBtn.textContent = hostAction ? "Ending\u2026" : "Leaving\u2026";
    }
    void Je(
      info.hubEndpoint,
      view.hostTopic,
      hostAction ? void 0 : info.topic
    ).then(() => {
      setIraConferenceView(null);
    }).catch((err) => {
      console.error("[ira] leave conference failed", err);
    }).finally(() => {
      refreshRadio();
    });
  });
  document.getElementById("iraSessionInfoBtn")?.addEventListener("click", () => {
    if (sessionDl) fillSessionDl(sessionDl);
    if (sessionModal) sessionModal.hidden = false;
    closeMenus();
  });
  document.getElementById("iraSceneViewBtn")?.addEventListener("click", () => {
    const btn = document.getElementById("iraSceneViewBtn");
    if (btn?.disabled) return;
    if (sceneViewPre) fillSceneViewPre(sceneViewPre);
    if (sceneViewModal) sceneViewModal.hidden = false;
    closeMenus();
  });
  document.getElementById("iraHubBtn")?.addEventListener("click", () => {
    window.open(getIraHubAdminUrl(), "_blank", "noopener,noreferrer");
    closeMenus();
  });
  document.querySelectorAll("[data-ira-close-modal]").forEach((el) => {
    el.addEventListener("click", () => {
      if (sessionModal) sessionModal.hidden = true;
      if (sceneViewModal) sceneViewModal.hidden = true;
      if (aboutModal) aboutModal.hidden = true;
      if (loadLocalModal) closeLoadLocalStudyDialog();
      closeMedicalInferenceDialog();
    });
  });
}

// render/demos/ira/ira-idc.ts
function bucketFromIdcUrl(urlStr) {
  try {
    const host = new URL(urlStr).hostname.toLowerCase();
    const m = host.match(/^([^.]+)\.s3[.-]/);
    if (m) return m[1];
    if (host.includes("idc-open-data")) return "idc-open-data";
  } catch {
  }
  return "idc-open-data";
}
function crdcPrefixFromIdcUrl(urlStr) {
  try {
    const path = new URL(urlStr).pathname.replace(/^\/+|\/+$/g, "");
    const first = path.split("/")[0]?.trim() || "";
    return first;
  } catch {
    return "";
  }
}
function isSegFile(f) {
  const role = String(f.role || "").toLowerCase();
  const label = String(f.label || "").toLowerCase();
  const name = String(f.fileName || "").toLowerCase();
  return role === "seg" || label === "seg" || name.includes("seg");
}
function sourceFromIdcOpenContext(context) {
  const plan = ur(context);
  if (!plan || plan.mode !== "idc") {
    return null;
  }
  const files = Array.isArray(plan.files) ? plan.files : [];
  if (!files.length) return null;
  const segFile = files.find(isSegFile);
  const volFile = files.find((f) => !isSegFile(f)) || files[0];
  const volUrl = String(volFile?.url || "").trim();
  if (!volUrl) return null;
  const c = crdcPrefixFromIdcUrl(volUrl);
  if (!c) return null;
  const cb = bucketFromIdcUrl(volUrl);
  let s;
  let sb;
  if (segFile?.url) {
    const segUrl = String(segFile.url).trim();
    s = crdcPrefixFromIdcUrl(segUrl) || void 0;
    sb = s ? bucketFromIdcUrl(segUrl) : void 0;
  }
  const st2 = String(plan.studyInstanceUID || "").trim();
  const id = String(plan.studyId || "IDC").trim();
  return {
    c,
    cb,
    s,
    sb: sb || (s ? cb : void 0),
    m: "CT",
    col: "IDC",
    st: st2,
    sd: `${id} \xB7 ${c.slice(0, 8)}\u2026`,
    lic: "NCI Imaging Data Commons"
  };
}
function studyParamsFromIdcSource(source) {
  const params = {
    series: source.c,
    bucket: source.cb || "idc-open-data"
  };
  if (source.s) {
    params.seg = source.s;
    params.segBucket = source.sb || source.cb || "idc-open-data";
  }
  const patient = String(source.sd || "").split("\xB7")[0]?.trim();
  if (patient) params.patient = patient;
  if (source.m) params.modality = source.m;
  if (source.col) params.collection = source.col;
  return params;
}

// render/demos/ira/ira-browser.ts
var P = new URLSearchParams(location.search);
var applyCastNiftiSeg = null;
var applyCastDicomSeg = null;
var castNiftiMountId = 0;
var IDC_INDEX_BASE = globalThis.__IDC_INDEX_BASE || P.get("indexBase") || "https://js2.jetstream-cloud.org:8001/swift/v1/idc-index/";
var GROUPS_URL = new URL("idc-rad-groups.json", IDC_INDEX_BASE).href;
var PARQUET_URL = new URL("idc-rad-slim.parquet", IDC_INDEX_BASE).href;
var HYPARQUET_ESM = "https://cdn.jsdelivr.net/npm/hyparquet@1.28.2/+esm";
var splitList = (v2) => (v2 ?? "").split(",").map((x) => x.trim()).filter(Boolean);
var _dirCache = null;
async function loadGroupDir(onStatus2) {
  if (_dirCache) return _dirCache;
  onStatus2("fetching the IDC index directory\u2026");
  let resp;
  try {
    const cache = globalThis.caches ? await globalThis.caches.open("idc-rad") : null;
    if (cache) resp = await cache.match(GROUPS_URL);
    if (!resp) {
      resp = await fetch(GROUPS_URL);
      if (resp.ok && cache) await cache.put(GROUPS_URL, resp.clone());
    }
  } catch {
    resp = await fetch(GROUPS_URL);
  }
  if (!resp || !resp.ok) throw new Error(`index directory not reachable at ${GROUPS_URL} (HTTP ${resp?.status ?? "?"})`);
  _dirCache = await resp.json();
  return _dirCache;
}
var SLIM_COLS = [
  "StudyInstanceUID",
  "SeriesInstanceUID",
  "crdc_series_uuid",
  "aws_bucket",
  "Modality",
  "instanceCount",
  "SeriesDescription",
  "PatientID",
  "collection_id",
  "license_short_name",
  "source_DOI"
];
async function readStudyRows(studyUID, onStatus2) {
  const dir = await loadGroupDir(onStatus2);
  const RGS = dir.rowGroupSize;
  const spans = [];
  for (let i = 0; i < dir.groups.length; i++) {
    const g = dir.groups[i];
    if (studyUID >= g.min && studyUID <= g.max) spans.push([i * RGS, Math.min((i + 1) * RGS, dir.total)]);
  }
  if (!spans.length) {
    throw new Error(
      `StudyInstanceUID not found in the slim IDC index (${dir.version}): ${studyUID}. It may be a non-radiology study, or from a newer index. The direct form always works: ?series=<crdc_series_uuid>&bucket=idc-open-data.`
    );
  }
  onStatus2(`range-reading the IDC index (${spans.length} group${spans.length > 1 ? "s" : ""})\u2026`);
  try {
    const hp = await import(HYPARQUET_ESM);
    const file = await hp.asyncBufferFromUrl({ url: PARQUET_URL });
    const metadata = await hp.parquetMetadataAsync(file);
    const parts = await Promise.all(
      spans.map(([rowStart, rowEnd]) => hp.parquetReadObjects({ file, metadata, columns: SLIM_COLS, rowStart, rowEnd }))
    );
    const rows2 = parts.flat().filter((r) => r.StudyInstanceUID === studyUID);
    if (!rows2.length) throw new Error(`StudyInstanceUID not found in the IDC index: ${studyUID}`);
    return rows2;
  } catch (e) {
    if (e.message.includes("not found")) throw e;
    throw new Error(
      `couldn't range-read the slim IDC index at ${PARQUET_URL} \u2014 check the CORS bucket (?indexBase=). Meanwhile the direct form works: ?series=<crdc_series_uuid>&bucket=idc-open-data. (${e.message})`
    );
  }
}
async function resolveSource(onStatus2) {
  const series = P.get("series");
  if (series) {
    return {
      c: series,
      cb: P.get("bucket") || "idc-open-data",
      s: P.get("seg") || void 0,
      sb: P.get("segBucket") || "idc-open-data",
      m: (P.get("modality") || "CT").toUpperCase(),
      col: P.get("collection") || "IDC",
      st: "",
      sd: `${P.get("patient") || "IDC"} \xB7 ${series.slice(0, 8)}\u2026`,
      lic: "NCI Imaging Data Commons"
    };
  }
  const studyUIDs = [...splitList(P.get("StudyInstanceUIDs")), ...splitList(P.get("studyUID"))];
  if (studyUIDs.length) {
    const want = [
      ...splitList(P.get("SeriesInstanceUIDs")),
      ...splitList(P.get("initialSeriesInstanceUID")),
      ...splitList(P.get("seriesUID"))
    ];
    return await resolveFromIndex(studyUIDs[0], want, onStatus2);
  }
  return null;
}
async function resolveFromIndex(studyUID, wantSeries, onStatus2) {
  const memoKey = `idc-rad:${studyUID}:${wantSeries.join(",")}`;
  try {
    const hit = localStorage.getItem(memoKey);
    if (hit) return JSON.parse(hit);
  } catch {
  }
  const inStudy = await readStudyRows(studyUID, onStatus2);
  const want = new Set(wantSeries);
  const IMG = /* @__PURE__ */ new Set(["CT", "MR", "PT", "PET", "NM"]);
  const imgs = inStudy.filter((r) => IMG.has(String(r.Modality).toUpperCase()));
  const segs = inStudy.filter((r) => String(r.Modality).toUpperCase() === "SEG");
  const chosen = imgs.find((r) => want.has(r.SeriesInstanceUID)) ?? imgs.slice().sort((a, b) => Number(b.instanceCount) - Number(a.instanceCount))[0];
  if (!chosen) throw new Error("no CT/MR/PET/NM image series found in this study");
  const seg = segs.find((r) => want.has(r.SeriesInstanceUID)) ?? segs[0];
  const mod = String(chosen.Modality).toUpperCase();
  const src = {
    c: String(chosen.crdc_series_uuid),
    cb: String(chosen.aws_bucket),
    s: seg ? String(seg.crdc_series_uuid) : void 0,
    sb: seg ? String(seg.aws_bucket) : void 0,
    m: mod === "PET" ? "PT" : mod,
    col: String(chosen.collection_id),
    st: studyUID,
    sd: `${chosen.PatientID} \xB7 ${chosen.SeriesDescription || mod}`,
    lic: `${chosen.license_short_name || "IDC"} \xB7 ${chosen.collection_id}${chosen.source_DOI ? " \xB7 doi:" + chosen.source_DOI : ""}`
  };
  try {
    localStorage.setItem(memoKey, JSON.stringify(src));
  } catch {
  }
  return src;
}
var status = (msg, err = false) => {
  const el = document.getElementById("status");
  if (el) {
    el.textContent = msg;
    el.style.color = err ? "#ff6b74" : "#9fb3d0";
  }
};
var cvEl = (id) => document.getElementById(id);
var PLANES = ["axial", "coronal", "sagittal"];
var SEG_FILL = 0.5;
var CT_WL_PRESETS = [
  { name: "Soft Tissue", win: 400, lev: 40 },
  { name: "Lung", win: 1500, lev: -600 },
  { name: "Bone", win: 1800, lev: 400 },
  { name: "Brain", win: 80, lev: 40 },
  { name: "Abdomen", win: 350, lev: 50 }
];
async function main() {
  let mountLocalSeries = async () => {
    throw new Error("IRA viewer is still starting \u2014 try again in a moment");
  };
  installIraHeader({
    onMountLocalSeries: (entry, opts) => mountLocalSeries(entry, opts)
  });
  void connectIraCast();
  if (!navigator.gpu) {
    status("WebGPU not available \u2014 try Chrome/Edge 113+ or Safari 18+.", true);
    return;
  }
  status("starting\u2026");
  const gpu = await initDevice();
  const preferred = navigator.gpu.getPreferredCanvasFormat();
  const srgb = preferred + "-srgb";
  const names = ["axial", "threeD", "coronal", "sagittal"];
  const cv = {}, cx = {};
  for (const n of names) {
    cv[n] = cvEl("c-" + n);
    cx[n] = cv[n].getContext("webgpu");
    cx[n].configure({ device: gpu.device, format: preferred, viewFormats: [srgb], alphaMode: "opaque" });
  }
  let studyMount = null;
  let studyAbort = null;
  let emptyCleanup = null;
  let studyLoadToken = 0;
  setIraNiftiSendHandler(async (event) => {
    if (!applyCastNiftiSeg) {
      status("nifti-send received but no study is open", true);
      return;
    }
    await applyCastNiftiSeg(event);
  });
  setIraDicomSendHandler(async (event) => {
    if (!applyCastDicomSeg) {
      status("dicom-send received but no study is open", true);
      return;
    }
    await applyCastDicomSeg(event);
  });
  const clearViewportDisplay = () => {
    for (const el of document.querySelectorAll(
      ".ann, .ira-overlay, #ira-strip, #ira-cine-box, #sl-badge, #sl-popup, #sl-help"
    )) {
      el.remove();
    }
    resetIraToolbarShell();
    for (const p of PLANES) {
      document.querySelector(`[data-cell="${p}"]`)?.classList.remove("ira-cell-selected");
      const readout = document.getElementById("sr-" + p);
      if (readout) readout.textContent = "";
    }
    const info = document.getElementById("info");
    if (info) info.textContent = "";
    for (const n of names) {
      try {
        const tex = cx[n].getCurrentTexture();
        const enc2 = gpu.device.createCommandEncoder();
        const pass = enc2.beginRenderPass({
          colorAttachments: [
            {
              view: tex.createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
              loadOp: "clear",
              storeOp: "store"
            }
          ]
        });
        pass.end();
        gpu.device.queue.submit([enc2.finish()]);
      } catch {
      }
    }
  };
  const disposeStudy = () => {
    studyLoadToken += 1;
    studyAbort?.abort();
    studyAbort = null;
    studyMount?.dispose();
    studyMount = null;
    clearViewportDisplay();
  };
  const showEmptyChrome = () => {
    const bar = document.getElementById("ira-bar");
    if (bar) ensureIraToolbarShell(bar);
    if (emptyCleanup) {
      clearViewportDisplay();
      status("No study loaded");
      return;
    }
    const resizeEmpty = () => {
      const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
      for (const n of names) {
        cv[n].width = Math.max(1, Math.round(cv[n].clientWidth * dpr));
        cv[n].height = Math.max(1, Math.round(cv[n].clientHeight * dpr));
      }
      clearViewportDisplay();
    };
    globalThis.addEventListener("resize", resizeEmpty);
    const ros = [];
    for (const n of names) {
      const ro = new ResizeObserver(() => resizeEmpty());
      ro.observe(cv[n]);
      ros.push(ro);
    }
    attachViewGrid(document.getElementById("grid"), [...names], resizeEmpty);
    attachDoubleClick(cv.threeD, () => {
    });
    resizeEmpty();
    emptyCleanup = () => {
      globalThis.removeEventListener("resize", resizeEmpty);
      for (const ro of ros) ro.disconnect();
      emptyCleanup = null;
    };
    status("No study loaded");
  };
  const loadStudy = async (input) => {
    emptyCleanup?.();
    emptyCleanup = null;
    disposeStudy();
    const token = studyLoadToken;
    studyAbort = new AbortController();
    const signal = studyAbort.signal;
    try {
      const mount = await mountStudyFromSource(
        input,
        gpu,
        preferred,
        srgb,
        names,
        cv,
        cx,
        signal
      );
      if (token !== studyLoadToken || signal.aborted) {
        mount.dispose();
        clearViewportDisplay();
        return;
      }
      studyMount = mount;
    } catch (err) {
      if (token !== studyLoadToken || signal.aborted || err instanceof DOMException && err.name === "AbortError") {
        clearViewportDisplay();
        return;
      }
      console.error("[ira] study load failed", err);
      status("error: " + (err?.message ?? err), true);
      showEmptyChrome();
    }
  };
  setIraImagingStudyOpenHandler(async (ctx) => {
    const sceneDoc = sceneDocFromOpenContext(ctx);
    if (sceneDoc) {
      status("Opening scene\u2026");
      syncIraStudyUrl(studyParamsFromSceneDoc(sceneDoc));
      await loadStudy({ sceneDoc });
      return;
    }
    const sceneUrl2 = sceneUrlFromOpenContext(ctx);
    if (sceneUrl2) {
      status("Opening scene\u2026");
      syncIraStudyUrl(studyParamsFromSceneUrl(sceneUrl2));
      await loadStudy({ sceneUrl: sceneUrl2 });
      return;
    }
    const idcSource = sourceFromIdcOpenContext(ctx);
    if (!idcSource) {
      status(
        "Cast ImagingStudy-open: unsupported context (need scene JSON or idc CRDC prefix URLs)",
        true
      );
      return;
    }
    status("Opening study\u2026");
    syncIraStudyUrl(studyParamsFromIdcSource(idcSource));
    await loadStudy(idcSource);
  });
  setIraImagingStudyCloseHandler(() => {
    setIraLiveScene(null);
    setIraImagingStudyContext([]);
    disposeStudy();
    syncIraStudyUrl({});
    showEmptyChrome();
    status("No study loaded");
  });
  mountLocalSeries = async (entry, { anonymized }) => {
    status("Reconstructing local series\u2026");
    const vol = loadEntry(entry);
    const localVolume = iraVolumeFromLocalVolume(vol);
    const modality = String(entry.modality || localVolume.modality || "CT").trim();
    const desc = String(entry.description || vol.name || "").trim();
    const patient = anonymized ? "Anonymous" : String(entry.patientName || "").trim() || "Local";
    const label = [patient, modality, desc || `${entry.count} images`].filter(Boolean).join(" \xB7 ");
    await loadStudy({ localVolume, label, anonymized });
    const info = document.getElementById("info");
    if (info) info.textContent = label;
    const openContext = imagingStudyContextFromLocalStudy({
      label,
      modality,
      studyInstanceUID: entry.studyInstanceUID,
      seriesInstanceUID: entry.seriesInstanceUID,
      anonymized
    });
    await publishIraImagingStudyTransition(openContext);
  };
  const setLoad = (m) => {
    status(m);
  };
  const sceneUrl = (P.get("scene") || "").trim();
  if (sceneUrl === EMBEDDED_SCENE_PARAM) {
    const embedded = takeEmbeddedSceneFromStorage();
    if (!embedded) {
      status("Embedded LiveScene missing \u2014 reopen from the worklist.", true);
      showEmptyChrome();
      return;
    }
    await loadStudy({ sceneDoc: embedded });
    return;
  }
  if (sceneUrl) {
    await loadStudy({ sceneUrl });
    return;
  }
  const source = await resolveSource(setLoad);
  if (!source) {
    showEmptyChrome();
    return;
  }
  await loadStudy(source);
}
async function mountStudyFromSource(input, gpu, preferred, srgb, names, cv, cx, signal) {
  let source;
  let ct;
  let seg;
  let sceneVr = null;
  let sceneStream = null;
  let sceneStreamDone = null;
  let sceneRedrawRaf = 0;
  let pendingLodUpgrade = null;
  let pendingIdcSeg;
  let idcStreamVol = null;
  let idcCtAssembled = false;
  let scHold = null;
  let applyIdcGeom = null;
  let sceneLodField;
  if ("localVolume" in input) {
    setIraLiveScene(null);
    setIraImagingStudyContext([]);
    syncIraStudyUrl({});
    ct = input.localVolume;
    const patientBit = input.anonymized ? "Anonymous" : (input.label.split("\xB7")[0] || "Local").trim();
    source = {
      c: "",
      cb: "",
      m: ct.modality || "CT",
      col: "Local",
      st: "",
      sd: input.anonymized ? `Anonymous \xB7 ${input.label}` : input.label || `${patientBit} \xB7 Local`,
      lic: "Local"
    };
    status(`loaded local series \xB7 ${ct.dims.join("\xD7")}`);
  } else if ("sceneDoc" in input || "sceneUrl" in input) {
    status("loading scene\u2026");
    let sceneDoc = null;
    if ("sceneDoc" in input) {
      sceneDoc = input.sceneDoc;
      setIraLiveScene(sceneDoc);
      setIraImagingStudyContext(imagingStudyContextFromSceneDoc(sceneDoc));
    } else {
      try {
        sceneDoc = await fetchLiveSceneDoc(input.sceneUrl);
        setIraLiveScene(sceneDoc);
      } catch (err) {
        console.warn("[ira] LiveScene document fetch failed", err);
        setIraLiveScene(null);
      }
      setIraImagingStudyContext(imagingStudyContextFromSceneUrl(input.sceneUrl));
    }
    const sceneSource = "sceneDoc" in input ? {
      blobBase: input.sceneDoc.blobBase,
      nodes: input.sceneDoc.nodes,
      url: input.sceneDoc.url
    } : input.sceneUrl;
    let mb = 0;
    let resolveReady;
    const ready = new Promise((r) => {
      resolveReady = r;
    });
    sceneStreamDone = loadSceneVolumeFieldProgressive(gpu.device, sceneSource, {
      onBytes: (n) => {
        if (signal?.aborted) return;
        mb += n;
        status(`streaming scene\u2026 ${(mb / 1e6).toFixed(1)} MB`);
      },
      onReady: (sv2) => {
        if (signal?.aborted) return;
        sceneLodField = sv2.field;
        resolveReady(sv2);
      },
      onSlab: (zStart, zCount, sv2) => {
        if (signal?.aborted) return;
        if (sceneStream) {
          sceneStream.patch(sv2.voxels, zStart, zCount);
          if (!sceneRedrawRaf) {
            sceneRedrawRaf = requestAnimationFrame(() => {
              sceneRedrawRaf = 0;
              sceneStream?.redraw();
            });
          }
        }
      },
      onUpgrade: (sv2) => {
        if (signal?.aborted) return;
        status(`refining\u2026 ${sv2.dims.join("\xD7")}`);
        sceneLodField = sv2.field;
        if (sceneStream?.upgrade) sceneStream.upgrade(sv2);
        else pendingLodUpgrade = sv2;
      }
    }, { signal });
    const sv = await ready;
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    ct = iraVolumeFromSceneVolume(sv);
    sceneVr = vrPresetFromSceneVolume(sv);
    source = {
      c: "",
      cb: "",
      m: ct.modality || "CT",
      col: "LiveScene",
      st: "",
      sd: `${sv.name} \xB7 LiveScene`,
      lic: "LiveScene"
    };
  } else {
    setIraLiveScene(null);
    source = input;
    status("contacting the NCI Imaging Data Commons\u2026");
    let resolveReady;
    const ready = new Promise((r) => {
      resolveReady = r;
    });
    const modality = { CT: "CT", MR: "MR", PT: "PET" }[source.m] || source.m;
    sceneStreamDone = loadSeries(source, {
      onProgress: (p) => {
        if (signal?.aborted) return;
        status(`${p.msg}${p.frac ? ` \u2014 ${Math.round(p.frac * 100)}%` : ""}`);
      },
      onCTReady: (meta) => {
        if (signal?.aborted) return;
        const [nx, ny, nz] = meta.dims;
        const empty = meta.lev - meta.win / 2 - 1;
        idcStreamVol = new Float32Array(nx * ny * nz);
        idcStreamVol.fill(empty);
        resolveReady({
          vol: idcStreamVol,
          dims: meta.dims,
          ijkToRAS: meta.ijkToRAS,
          win: meta.win,
          lev: meta.lev,
          dtype: "float32",
          modality: meta.modality || modality
        });
      },
      onCTGeom: (meta) => {
        if (signal?.aborted) return;
        if (ct) {
          ct.ijkToRAS = meta.ijkToRAS;
          if (Number.isFinite(meta.win)) ct.win = meta.win;
          if (Number.isFinite(meta.lev)) ct.lev = meta.lev;
        }
        applyIdcGeom?.(meta.ijkToRAS);
      },
      onCTSlab: (info) => {
        if (signal?.aborted) return;
        const host = idcStreamVol;
        if (!host) return;
        const [nx, ny] = info.dims;
        host.set(info.data, info.zStart * nx * ny);
        if (sceneStream) {
          sceneStream.patch(host, info.zStart, info.zCount);
          if (!sceneRedrawRaf) {
            sceneRedrawRaf = requestAnimationFrame(() => {
              sceneRedrawRaf = 0;
              sceneStream?.redraw();
            });
          }
        }
      },
      onCT: (full) => {
        if (signal?.aborted) return;
        if (idcStreamVol && full.vol instanceof Float32Array && full.vol !== idcStreamVol) {
          idcStreamVol.set(full.vol);
        }
        if (ct && Number.isFinite(full.win) && Number.isFinite(full.lev)) {
          ct.win = full.win;
          ct.lev = full.lev;
        }
        idcCtAssembled = true;
        if (sceneStream && idcStreamVol) {
          sceneStream.patch(idcStreamVol, 0, full.dims[2]);
          sceneStream.redraw();
        }
      },
      onLabelmap: (lab) => {
        if (signal?.aborted) return;
        pendingIdcSeg = lab;
      }
    }, {
      // Cache-bust the classic worker (browsers often keep idc-worker.js sticky across IRA reloads).
      workerUrl: new URL("./idc-worker.js?v=idc-fetch-11", import.meta.url)
    }).then((res) => {
      if (res.seg) pendingIdcSeg = res.seg;
      return res;
    });
    ct = await ready;
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    seg = void 0;
  }
  const sc = buildSegrouletteScene(gpu, srgb, ct, seg, {
    volumeField: sceneLodField
  });
  scHold = sc;
  const thisCastMountId = ++castNiftiMountId;
  if (sceneVr) sc.setVolumePreset(sceneVr);
  let vmin = Infinity, vmax = -Infinity;
  const sampleRange = () => {
    vmin = Infinity;
    vmax = -Infinity;
    const buf = ct.vol;
    const vstride = Math.max(1, Math.floor(buf.length / 2e6));
    for (let i = 0; i < buf.length; i += vstride) {
      const v2 = buf[i];
      if (v2 < vmin) vmin = v2;
      if (v2 > vmax) vmax = v2;
    }
  };
  sampleRange();
  const dataRange = () => [vmin, vmax];
  sc.setVolumeOpacity(seg ? 0.5 : 1);
  let sliceOutline = false;
  let roiEnabled = false, roiVisible = false, roiFirstEnable = true;
  let annOn = true;
  let labelsOn = true;
  const dpr3 = Math.min(2, globalThis.devicePixelRatio || 1);
  const cardFont = buildFontAtlas({
    sizePx: 44,
    pad: 6,
    fontFamily: 'Helvetica, "Helvetica Neue", Arial, sans-serif'
  });
  let cardDt = performance.now();
  let cardDrawPending = false;
  let studyDisposed = false;
  const annEls = {};
  const buildAnnotations = () => {
    for (const p of PLANES) {
      const cell = document.querySelector(`[data-cell="${p}"]`);
      if (!cell) continue;
      const mk = (cls, html) => {
        const e = document.createElement("div");
        e.className = "ann " + cls;
        e.innerHTML = html;
        e.style.cssText = "position:absolute;z-index:2;font:600 10px ui-monospace,Menlo,monospace;color:#cfe0f5;text-shadow:0 0 3px #000,0 0 3px #000;pointer-events:none;line-height:1.35;max-width:46%;" + (cls === "tl" ? "top:22px;left:7px;" : cls === "tr" ? "top:22px;right:7px;text-align:right;" : "bottom:22px;right:7px;text-align:right;");
        cell.appendChild(e);
        return e;
      };
      const pid = (source.sd.split("\xB7")[0] || "").trim() || source.col;
      mk("tl", `${pid}<br>${source.col} \xB7 ${ct.modality}`);
      mk("tr", `${ct.modality} \xB7 ${sc.dims[2]} slices${sc.hasSeg ? " \xB7 SEG" : ""}<br>${sc.dims[0]}\xD7${sc.dims[1]}`);
      annEls[p] = { br: mk("br", "") };
    }
    updateAnnWL();
  };
  const updateAnnWL = () => {
    for (const p of PLANES) {
      if (annEls[p]) annEls[p].br.innerHTML = `W ${Math.round(wl.win)} / L ${Math.round(wl.lev)}`;
    }
  };
  const applyAnn = () => {
    for (const e of document.querySelectorAll(".ann")) e.style.display = annOn ? "" : "none";
  };
  const off = {
    axial: slicerDefaultOffset01("axial", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi),
    coronal: slicerDefaultOffset01("coronal", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi),
    sagittal: slicerDefaultOffset01("sagittal", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi)
  };
  let sliceIx = new SliceInteractor({ ijkToRAS: sc.ijkToRAS, rasLo: sc.rasLo, rasHi: sc.rasHi });
  const camera = framedCamera(sc.center, sc.radius);
  const wl = { win: sc.win, lev: sc.lev };
  let ira = null;
  let xhair = null;
  const drawSlice = (p) => {
    if (studyDisposed) return;
    ira?.drawOverlay(p);
    if (!cv[p].width) return;
    sc.slice.setPlane(p, off[p]);
    sc.slice.renderToView(cx[p].getCurrentTexture().createView({ format: srgb }), cv[p].width, cv[p].height);
    updateReadout(p);
  };
  const draw3d = () => {
    if (studyDisposed) return;
    if (!cv.threeD.width) return;
    sc.scene.setCamera(camera.position, camera.focalPoint, camera.viewUp, camera.viewAngle, cv.threeD.width, cv.threeD.height);
    sc.scene.renderToView(cx.threeD.getCurrentTexture().createView({ format: srgb }), cv.threeD.width, cv.threeD.height);
    if (!labelsOn) return;
    const now = performance.now();
    const dt2 = (now - cardDt) / 1e3;
    cardDt = now;
    const moving = segCards.draw(
      cx.threeD.getCurrentTexture().createView({ format: srgb }),
      cv.threeD.width,
      cv.threeD.height,
      dpr3,
      dt2
    );
    if (moving && !cardDrawPending) {
      cardDrawPending = true;
      requestAnimationFrame(() => {
        cardDrawPending = false;
        draw3d();
      });
    }
  };
  const segCards = mountSegmentCards(gpu, srgb, cardFont, cv.threeD, camera, {
    apply: (id, action) => {
      if (action === "reset") sc.setSegmentOpacities(sc.segments.map((s) => [s.num, 1]));
      else if (action === "hide") sc.setSegmentOpacity(id, 0);
      else sc.setSegmentOpacities(sc.segments.map((s) => [s.num, s.num === id ? 1 : 0.4]));
      drawAll();
    },
    redraw: () => draw3d(),
    maxCards: 12,
    reserved: () => {
      const cr2 = cv.threeD.getBoundingClientRect();
      if (!cr2.width || !cr2.height) return [];
      const sx = cv.threeD.width / cr2.width;
      const sy = cv.threeD.height / cr2.height;
      const pad = 6;
      const out = [];
      for (const sel of ["#sl-badge", ".lab.threeD"]) {
        const el = document.querySelector(sel);
        if (!el) continue;
        const r = el.getBoundingClientRect();
        const x0 = Math.max(r.left, cr2.left);
        const y0 = Math.max(r.top, cr2.top);
        const x1 = Math.min(r.right, cr2.right);
        const y1 = Math.min(r.bottom, cr2.bottom);
        if (x1 <= x0 || y1 <= y0) continue;
        out.push({
          x: (x0 - cr2.left) * sx - pad,
          y: (y0 - cr2.top) * sy - pad,
          w: (x1 - x0) * sx + 2 * pad,
          h: (y1 - y0) * sy + 2 * pad
        });
      }
      return out;
    },
    segVisible: (num2) => sc.isSegmentVisible(num2)
  });
  segCards.setScene(ct, seg ?? null, sc.segments);
  const drawAll = () => {
    for (const p of PLANES) drawSlice(p);
    draw3d();
    xhair?.redraw();
  };
  const refreshSegChrome = (labmap) => {
    seg = labmap;
    sc.setVolumeOpacity(0.5);
    segCards.setScene(ct, seg, sc.segments);
    for (const p of PLANES) {
      const cell = document.querySelector(`[data-cell="${p}"]`);
      const tr2 = cell?.querySelector(".ann.tr");
      if (tr2) {
        tr2.innerHTML = `${ct.modality} \xB7 ${sc.dims[2]} slices${sc.hasSeg ? " \xB7 SEG" : ""}<br>${sc.dims[0]}\xD7${sc.dims[1]}`;
      }
    }
    drawAll();
  };
  applyCastNiftiSeg = async (event) => {
    if (studyDisposed || thisCastMountId !== castNiftiMountId) return;
    status("Loading Cast segmentation\u2026");
    const result = await segLabelmapFromNiftiSendEvent(event, sc.dims, sc.ijkToRAS);
    if (studyDisposed || thisCastMountId !== castNiftiMountId) return;
    if (!result.ok) {
      status(result.error, true);
      return;
    }
    let merged = result.seg;
    let added = result.labelCount;
    let skipped = 0;
    try {
      const m = mergeSegLabelmaps(sc.labelmap() ?? seg, result.seg);
      merged = m.seg;
      added = m.added;
      skipped = m.skipped;
    } catch (err) {
      status(`Merge failed: ${err?.message ?? err}`, true);
      return;
    }
    sc.replaceSegmentation(merged);
    if (!sc.hasSeg) {
      status("Cast segmentation produced no renderable segments", true);
      return;
    }
    refreshSegChrome(merged);
    const skipNote = skipped ? ` \xB7 ${skipped} skipped (label id limit)` : "";
    status(
      `Added ${added} TotalSeg segment${added === 1 ? "" : "s"} from ${result.fileName}${skipNote}`
    );
  };
  applyCastDicomSeg = async (event) => {
    if (studyDisposed || thisCastMountId !== castNiftiMountId) return;
    status("Loading Cast DICOM SEG\u2026");
    const result = await segLabelmapFromDicomSendEvent(event, {
      dims: sc.dims,
      ijkToRAS: sc.ijkToRAS
    });
    if (studyDisposed || thisCastMountId !== castNiftiMountId) return;
    if (!result.ok) {
      status(result.error, true);
      return;
    }
    let merged = result.seg;
    let added = result.labelCount;
    let skipped = 0;
    try {
      const m = mergeSegLabelmaps(sc.labelmap() ?? seg, result.seg);
      merged = m.seg;
      added = m.added;
      skipped = m.skipped;
    } catch (err) {
      status(`Merge failed: ${err?.message ?? err}`, true);
      return;
    }
    sc.replaceSegmentation(merged);
    if (!sc.hasSeg) {
      status("Cast DICOM SEG produced no renderable segments", true);
      return;
    }
    refreshSegChrome(merged);
    const skipNote = skipped ? ` \xB7 ${skipped} skipped (label id limit)` : "";
    status(
      `Added ${added} segment${added === 1 ? "" : "s"} from ${result.fileName}${skipNote}`
    );
  };
  if (sceneStreamDone) {
    const applyLodUpgrade = (sv) => {
      const oldCenter = [sc.center[0], sc.center[1], sc.center[2]];
      sc.swapVolumeField(sv.field, sv.dims, sv.ijkToRAS);
      ct.vol = sv.voxels;
      ct.dims = sv.dims;
      ct.ijkToRAS = sv.ijkToRAS;
      for (let i = 0; i < 3; i++) {
        const d = sc.center[i] - oldCenter[i];
        camera.position[i] += d;
        camera.focalPoint[i] += d;
      }
      sliceIx = new SliceInteractor({ ijkToRAS: sc.ijkToRAS, rasLo: sc.rasLo, rasHi: sc.rasHi });
      off.axial = slicerDefaultOffset01("axial", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi);
      off.coronal = slicerDefaultOffset01("coronal", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi);
      off.sagittal = slicerDefaultOffset01("sagittal", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi);
      sampleRange();
      drawAll();
    };
    sceneStream = {
      patch: (data, zStart, zCount) => sc.patchVolumeScalars(data, zStart, zCount),
      redraw: () => drawAll(),
      upgrade: applyLodUpgrade
    };
    const [nx, ny, nz] = ct.dims;
    if (ct.vol instanceof Float32Array && nz > 0) {
      sc.patchVolumeScalars(ct.vol, 0, nz);
    }
    if (pendingLodUpgrade) {
      const u = pendingLodUpgrade;
      pendingLodUpgrade = null;
      applyLodUpgrade(u);
    }
  }
  applyIdcGeom = (ijkToRAS) => {
    const old = [sc.center[0], sc.center[1], sc.center[2]];
    sc.updateGeometry(ijkToRAS);
    for (let i = 0; i < 3; i++) {
      const d = sc.center[i] - old[i];
      camera.position[i] += d;
      camera.focalPoint[i] += d;
    }
    sliceIx = new SliceInteractor({ ijkToRAS: sc.ijkToRAS, rasLo: sc.rasLo, rasHi: sc.rasHi });
    drawAll();
  };
  const updateReadout = (p) => {
    const el = document.getElementById("sr-" + p);
    if (!el) return;
    const [lo, hi2] = sliceIx.bounds(p);
    const sp = sliceIx.spacing(p);
    const mm = offset01ToMm(p, off[p], sc.rasLo, sc.rasHi);
    el.textContent = `${Math.round((mm - lo) / sp + 0.5)}/${Math.round((hi2 - lo) / sp)} \xB7 ${mm.toFixed(1)} mm`;
  };
  {
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    for (const n of names) {
      cv[n].width = Math.max(1, Math.round(cv[n].clientWidth * dpr));
      cv[n].height = Math.max(1, Math.round(cv[n].clientHeight * dpr));
    }
    drawAll();
  }
  if (idcStreamVol && sceneStreamDone && !idcCtAssembled) {
    status("streaming volume\u2026");
    while (!idcCtAssembled) {
      await Promise.race([
        new Promise((r) => setTimeout(r, 16)),
        sceneStreamDone.then(() => {
          idcCtAssembled = true;
        })
      ]);
      if (ct.vol instanceof Float32Array && ct.dims[2] > 0) {
        sc.patchVolumeScalars(ct.vol, 0, ct.dims[2]);
        drawAll();
      }
    }
    sampleRange();
  }
  const resize = () => {
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    for (const n of names) {
      cv[n].width = Math.max(1, Math.round(cv[n].clientWidth * dpr));
      cv[n].height = Math.max(1, Math.round(cv[n].clientHeight * dpr));
    }
    ira?.resize();
    drawAll();
  };
  globalThis.addEventListener("resize", resize);
  const resizeObservers = [];
  for (const n of names) {
    const ro = new ResizeObserver(() => resize());
    ro.observe(cv[n]);
    resizeObservers.push(ro);
  }
  const viewGrid = attachViewGrid(document.getElementById("grid"), [...names], resize);
  attachDoubleClick(cv.threeD, () => viewGrid.toggleMax("threeD"));
  const setWL = (win, lev) => {
    wl.win = win;
    wl.lev = lev;
    sc.slice.setWindowLevel(win, lev);
    for (const p of PLANES) drawSlice(p);
    xhair?.redraw();
    syncWl();
    updateAnnWL();
  };
  const presets = document.getElementById("wl-readout");
  const syncWl = () => {
    presets.innerHTML = "";
    presets.add(new Option(`W ${Math.round(wl.win)} / L ${Math.round(wl.lev)}`, "current", true, true));
    presets.add(new Option(`Auto (${Math.round(sc.win)}/${Math.round(sc.lev)})`, "auto"));
    for (const q2 of CT_WL_PRESETS) presets.add(new Option(`${q2.name} (${q2.win}/${q2.lev})`, q2.name));
  };
  presets.addEventListener("change", () => {
    if (presets.value === "auto") setWL(sc.win, sc.lev);
    else {
      const q2 = CT_WL_PRESETS.find((x) => x.name === presets.value);
      if (q2) setWL(q2.win, q2.lev);
    }
  });
  syncWl();
  const nAxisOf = { axial: 2, coronal: 1, sagittal: 0 };
  const jumpAll = (ras) => {
    for (const p of PLANES) {
      const a = nAxisOf[p];
      off[p] = Math.max(0, Math.min(1, (ras[a] - sc.rasLo[a]) / (sc.rasHi[a] - sc.rasLo[a])));
    }
    drawAll();
  };
  const sliceControls = [];
  const listAllStudySeries = source.st ? async () => {
    const rows2 = await readStudyRows(source.st, () => {
    });
    return rows2.map((r) => ({
      prefix: String(r.crdc_series_uuid),
      bucket: String(r.aws_bucket),
      modality: String(r.Modality).toUpperCase(),
      seriesUID: String(r.SeriesInstanceUID),
      seriesDescription: r.SeriesDescription ? String(r.SeriesDescription) : void 0
    }));
  } : void 0;
  ira = mountIra({
    overlay: document.getElementById("viewer"),
    bar: document.getElementById("ira-bar"),
    grid: document.getElementById("grid"),
    planes: PLANES,
    canvases: cv,
    cellOf: (name) => document.querySelector(`[data-cell="${name}"]`),
    slice: () => sc.slice,
    off01: (p) => off[p],
    setOff01: (p, v2) => off[p] = Math.max(0, Math.min(1, v2)),
    offsetMm: (p) => offset01ToMm(p, off[p], sc.rasLo, sc.rasHi),
    spacing: (p) => sliceIx.spacing(p),
    step: (p, fwd) => off[p] = sliceIx.wheel(p, off[p], fwd),
    redraw: (p) => {
      drawSlice(p);
      xhair?.redraw();
    },
    redrawAll: drawAll,
    rasLo: sc.rasLo,
    rasHi: sc.rasHi,
    wl: { get: () => [wl.win, wl.lev], set: setWL, auto: [sc.win, sc.lev] },
    presetsEl: presets,
    resetViews: () => sliceControls.forEach((c) => c.resetView()),
    close: () => status("This is the SlicerLive Basic Image Review demo \u2014 reload to restart."),
    jumpAll,
    modality: ct.modality,
    extraTools: [
      {
        id: "idc-share",
        icon: "share",
        title: "Share \u2014 copy a link that reopens this study here (or in the IDC OHIF portal)",
        run: () => shareStudy(source)
      },
      {
        id: "idc-download",
        icon: "download",
        title: "Download this study's DICOM to a local folder (streamed, parallel)",
        run: () => downloadStudyWithDialog(source, listAllStudySeries)
      },
      {
        id: "seg-stl",
        icon: "stl",
        title: "Save segment as STL for 3D printing (LPS)",
        run: () => {
          void (async () => {
            const labmap = sc.labelmap();
            if (!sc.hasSeg || !labmap || !sc.segments.length) {
              status("No segmentation loaded \u2014 open a study with SEG to export STL.");
              return;
            }
            const picked = await pickSegment(sc.segments);
            if (!picked) return;
            try {
              status(`Meshing ${picked.name}\u2026`);
              const buf = segmentToStl(labmap.lab, sc.dims, sc.ijkToRAS, picked.num);
              downloadStl(buf, safeStlFileName(picked.name));
              status(`Saved ${safeStlFileName(picked.name)}`);
            } catch (err) {
              status("STL export failed: " + (err?.message ?? err), true);
            }
          })();
        }
      }
    ]
  });
  globalThis.__iraDbg = {
    ready: () => !!ira,
    dims: () => sc.dims,
    tool: () => ira?.tool(),
    cellOrder: () => [...document.querySelectorAll("#grid .cell")].map((c) => c.dataset.cell),
    // test hooks
    setRoi: (en2, vis) => {
      roiEnabled = en2;
      roiVisible = vis;
      if (en2) roiFirstEnable = false;
      sc.setRoiEnabled(en2);
      sc.setRoiVisible(vis);
      draw3d();
    },
    roiState: () => ({ visible: sc.roiVisible(), handles: sc.roi.handleList().length }),
    setSegOpacity: (o) => {
      sc.setSegOpacity(o);
      sc.slice.setOverlayOpacity(o * SEG_FILL);
      drawAll();
    }
  };
  for (const p of PLANES) {
    const hooks = ira.hooks(p);
    sliceControls.push(attachSliceControls(cv[p], {
      orient: p,
      getSlice: () => sc.slice,
      step: (fwd) => off[p] = sliceIx.wheel(p, off[p], fwd),
      redraw: () => {
        drawSlice(p);
        xhair?.redraw();
      },
      wl: {
        enabled: () => ira.tool() === "wl",
        get: () => [wl.win, wl.lev],
        set: setWL,
        range: dataRange,
        reset: () => setWL(sc.win, sc.lev)
      },
      leftMode: () => ira.leftMode(),
      hooks: {
        onLeftGrab: hooks.onLeftGrab,
        onDoubleClick: () => {
          viewGrid.toggleMax(p);
          return true;
        }
      }
    }));
  }
  attachCameraControls(cv.threeD, camera, { onChange: () => {
    draw3d();
    xhair?.redraw();
  } });
  let roiBox0 = null;
  attachWidgetControls(cv.threeD, camera, {
    getHandles: () => sc.roiVisible() ? sc.roi.handleList().map((h) => ({ id: h.id, world: h.world, data: h.data, cursor: h.cursor })) : [],
    getSize: () => ({ w: cv.threeD.width, h: cv.threeD.height }),
    onDragStart: () => {
      roiBox0 = sc.roi.snapshot();
    },
    onDrag: (h, world) => {
      if (!roiBox0) return;
      const d = [world[0] - h.world[0], world[1] - h.world[1], world[2] - h.world[2]];
      sc.roi.applyDrag(h.data, roiBox0, d);
      sc.reclip();
      draw3d();
      xhair?.redraw();
    },
    onHover: (h) => {
      sc.roi.setHover(h ? h.id : null);
      sc.scene.syncUniforms();
      draw3d();
    },
    onChange: () => draw3d()
  });
  xhair = mountCrosshair({
    cells: { axial: cv.axial, coronal: cv.coronal, sagittal: cv.sagittal, threeD: cv.threeD },
    getScene: () => sc.scene,
    getSlice: () => sc.slice,
    getCamera: () => camera,
    getOffset: (o) => off[o],
    onJump: jumpAll
  });
  let vrPreset = null;
  let vrShift = 0;
  const shiftRange = Math.max(200, (vmax - vmin) / 2);
  const defaultVrBake = sceneVr;
  const bakeOf = (name) => name ? presetLUT(CT_VR_PRESETS.find((p) => p.name === name)) : defaultVrBake;
  const presetLabel = () => vrPreset ? CT_VR_PRESETS.find((p) => p.name === vrPreset)?.label ?? vrPreset : defaultVrBake ? "Scene TF" : "Default (W/L)";
  const applyVrPreset = (name) => {
    vrPreset = name;
    sc.setVolumePreset(bakeOf(name));
    if (sc.volumeOpacity() <= 0) sc.setVolumeOpacity(1);
    draw3d();
    xhair?.redraw();
  };
  const applyVrShift = (hu) => {
    vrShift = hu;
    sc.setVolumeShift(hu);
    draw3d();
    xhair?.redraw();
  };
  const renderPresetThumbnails = () => {
    const THUMB = 116;
    const savedOp = sc.volumeOpacity(), savedShift = sc.volumeShift(), savedSeg = sc.segOpacity();
    sc.setVolumeShift(0);
    if (savedSeg > 0) sc.setSegOpacity(0);
    if (savedOp < 1) sc.setVolumeOpacity(1);
    const entries = [
      { name: null, label: defaultVrBake ? "Scene TF" : "Default (W/L)" },
      ...CT_VR_PRESETS.map((p) => ({ name: p.name, label: p.label }))
    ];
    const items = [];
    for (const e of entries) {
      const c = document.createElement("canvas");
      c.width = THUMB;
      c.height = THUMB;
      const cxt = c.getContext("webgpu");
      cxt.configure({ device: gpu.device, format: preferred, viewFormats: [srgb], alphaMode: "opaque" });
      sc.setVolumePreset(bakeOf(e.name));
      sc.scene.setCamera(camera.position, camera.focalPoint, camera.viewUp, camera.viewAngle, THUMB, THUMB);
      sc.scene.renderToView(cxt.getCurrentTexture().createView({ format: srgb }), THUMB, THUMB);
      items.push({ name: e.name, label: e.label, canvas: c });
    }
    sc.setVolumePreset(bakeOf(vrPreset));
    sc.setVolumeShift(savedShift);
    if (savedSeg > 0) sc.setSegOpacity(savedSeg);
    if (savedOp < 1) sc.setVolumeOpacity(savedOp);
    draw3d();
    return items;
  };
  Object.assign(globalThis.__iraDbg, {
    presets: () => CT_VR_PRESETS.map((p) => p.name),
    setPreset: (name) => applyVrPreset(name),
    setShift: (hu) => applyVrShift(hu),
    vrState: () => ({ preset: vrPreset, shift: vrShift, shiftRange, dataRange: [vmin, vmax] }),
    thumbCount: () => renderPresetThumbnails().length
  });
  const controls = [
    {
      label: "Volume render",
      getOpacity: () => sc.volumeOpacity(),
      setOpacity: (o) => {
        sc.setVolumeOpacity(o);
        draw3d();
        xhair?.redraw();
      },
      color: [0.75, 0.78, 0.85]
    },
    // Slicer's VR "Shift" — a plain slider directly below the opacity control.
    {
      label: "Shift",
      slider: {
        min: -shiftRange,
        max: shiftRange,
        step: Math.max(1, Math.round(shiftRange / 200)),
        get: () => vrShift,
        set: (v2) => applyVrShift(v2),
        format: (v2) => (v2 > 0 ? "+" : "") + Math.round(v2)
      },
      disabled: () => sc.volumeOpacity() <= 0
    },
    // Volume-rendering preset — opens the on-the-fly thumbnail menu (Slicer/OHIF CT presets).
    {
      label: "Preset",
      button: { text: () => presetLabel(), run: () => openVrPresetMenu({ items: renderPresetThumbnails(), current: vrPreset, onPick: applyVrPreset }) }
    },
    {
      label: "Segmentation",
      getOpacity: () => sc.segOpacity(),
      setOpacity: (o) => {
        sc.setSegOpacity(o);
        sc.slice.setOverlayOpacity(o * SEG_FILL);
        drawAll();
      },
      disabled: () => !sc.hasSeg,
      color: [0.62, 0.9, 1]
    },
    {
      label: "Segment labels",
      get: () => labelsOn,
      set: (on2) => {
        labelsOn = on2;
        draw3d();
      },
      disabled: () => !sc.hasSeg
    },
    // Segmentation overlay: fill (default) vs outline on the MPR planes.
    {
      label: "Segmentation outline",
      get: () => sliceOutline,
      set: (on2) => {
        sliceOutline = on2;
        sc.slice.setOverlayOutline(on2);
        for (const p of PLANES) drawSlice(p);
        xhair?.redraw();
      },
      disabled: () => !sc.hasSeg
    },
    // 3D volume-render crop: enabling it reveals the ROI box the first time (Slicer behaviour).
    {
      label: "Crop volume (3D)",
      get: () => roiEnabled,
      set: (on2) => {
        roiEnabled = on2;
        if (on2 && roiFirstEnable) {
          roiVisible = true;
          roiFirstEnable = false;
        }
        sc.setRoiEnabled(roiEnabled);
        sc.setRoiVisible(roiVisible);
        draw3d();
        xhair?.redraw();
      },
      disabled: () => sc.volumeOpacity() <= 0
    },
    {
      label: "Show ROI box",
      get: () => roiVisible,
      set: (on2) => {
        roiVisible = on2;
        sc.setRoiVisible(on2);
        draw3d();
        xhair?.redraw();
      }
    },
    // DICOM corner annotations (IHE IRA) — the "official radiology look".
    {
      label: "Corner annotations",
      get: () => annOn,
      set: (on2) => {
        annOn = on2;
        applyAnn();
      }
    }
  ];
  const chrome = installChrome({
    help: false,
    controls,
    anchor: cv.threeD.parentElement ?? void 0,
    segments: {
      list: () => sc.segments.map((s) => ({ num: s.num, name: s.name, color: s.color })),
      get: (num2) => sc.segmentOpacity(num2),
      set: (num2, o) => {
        sc.setSegmentOpacity(num2, o);
        drawAll();
      },
      enabled: () => sc.hasSeg
    }
  });
  buildAnnotations();
  applyAnn();
  resize();
  if (sceneStreamDone) {
    status(idcStreamVol ? "streaming volume\u2026" : "streaming scene\u2026");
    await sceneStreamDone;
    if (signal?.aborted) {
      return {
        dispose: () => {
          studyDisposed = true;
          labelsOn = false;
          try {
            segCards.clear();
          } catch {
          }
          try {
            ira?.detachKeys();
          } catch {
          }
          try {
            chrome?.destroy();
          } catch {
          }
          try {
            sc.destroy();
          } catch {
          }
          resetIraToolbarShell();
          for (const el of document.querySelectorAll(".ann, .ira-overlay")) {
            el.remove();
          }
        }
      };
    }
    sampleRange();
    if (pendingIdcSeg) {
      seg = pendingIdcSeg;
      sc.attachSegmentation(pendingIdcSeg);
      refreshSegChrome(pendingIdcSeg);
    }
    if (idcStreamVol && Number.isFinite(ct.win) && Number.isFinite(ct.lev)) {
      setWL(ct.win, ct.lev);
    }
    drawAll();
  }
  status("");
  return {
    dispose: () => {
      studyDisposed = true;
      if (thisCastMountId === castNiftiMountId) {
        applyCastNiftiSeg = null;
        applyCastDicomSeg = null;
      }
      labelsOn = false;
      globalThis.removeEventListener("resize", resize);
      for (const ro of resizeObservers) ro.disconnect();
      try {
        segCards.clear();
      } catch {
      }
      try {
        ira?.detachKeys();
      } catch {
      }
      try {
        chrome.destroy();
      } catch {
      }
      try {
        sc.destroy();
      } catch {
      }
      try {
        sceneLodField?.destroy();
      } catch {
      }
      resetIraToolbarShell();
      for (const el of document.querySelectorAll(".ann, .ira-overlay")) {
        el.remove();
      }
      for (const p of PLANES) {
        document.querySelector(`[data-cell="${p}"]`)?.classList.remove("ira-cell-selected");
        const readout = document.getElementById("sr-" + p);
        if (readout) readout.textContent = "";
      }
      const info = document.getElementById("info");
      if (info) info.textContent = "";
      for (const n of names) {
        try {
          const tex = cx[n].getCurrentTexture();
          const enc2 = gpu.device.createCommandEncoder();
          const pass = enc2.beginRenderPass({
            colorAttachments: [
              {
                view: tex.createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: "clear",
                storeOp: "store"
              }
            ]
          });
          pass.end();
          gpu.device.queue.submit([enc2.finish()]);
        } catch {
        }
      }
    }
  };
}
main().catch((e) => status("error: " + (e?.message ?? e), true));
