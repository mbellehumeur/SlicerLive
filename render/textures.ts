// Small GPU texture helpers shared across fields/renderers.
import type { Vec3 } from "./mat4.ts";

/**
 * Upload an r32float volume via writeTexture in depth chunks so each staging
 * copy stays under device.limits.maxBufferSize (Chrome default ~256 MiB).
 */
export function writeScalarTexture3D(
  dev: GPUDevice,
  texture: GPUTexture,
  data: Float32Array,
  dims: Vec3,
): void {
  writeScalarTexture3DSlices(dev, texture, data, dims, 0, dims[2]);
}

/**
 * Upload a contiguous z-range of an r32float volume (z-major layout).
 * `data` is the full volume buffer; only [zStart, zStart+zCount) is written.
 */
export function writeScalarTexture3DSlices(
  dev: GPUDevice,
  texture: GPUTexture,
  data: Float32Array,
  dims: Vec3,
  zStart: number,
  zCount: number,
): void {
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

  let z = zStart;
  const zEnd = zStart + depth;
  while (z < zEnd) {
    const chunkDepth = Math.min(slicesPerChunk, zEnd - z);
    const offsetFloats = z * dx * dy;
    const chunk = data.subarray(offsetFloats, offsetFloats + chunkDepth * dx * dy);
    dev.queue.writeTexture(
      { texture, origin: [0, 0, z] },
      chunk,
      { bytesPerRow, rowsPerImage },
      [dx, dy, chunkDepth],
    );
    z += chunkDepth;
  }
}

/**
 * Upload an axis-aligned box from a full-volume Float32Array (z-major).
 * Used for progressive zarr chunk patches (MPR mid-planes before full z-slabs).
 */
export function writeScalarTexture3DRegion(
  dev: GPUDevice,
  texture: GPUTexture,
  data: Float32Array,
  dims: Vec3,
  xStart: number,
  yStart: number,
  zStart: number,
  xCount: number,
  yCount: number,
  zCount: number,
): void {
  const [dx, dy, dz] = dims;
  if (xCount <= 0 || yCount <= 0 || zCount <= 0) return;
  if (xStart < 0 || yStart < 0 || zStart < 0) return;
  if (xStart >= dx || yStart >= dy || zStart >= dz) return;
  const w = Math.min(xCount, dx - xStart);
  const h = Math.min(yCount, dy - yStart);
  const d = Math.min(zCount, dz - zStart);
  if (w <= 0 || h <= 0 || d <= 0) return;

  // Full-volume row pitch so we can point at (xStart,yStart,zStart) inside `data`.
  const bytesPerRow = dx * 4;
  const rowsPerImage = dy;
  const offsetFloats = ((zStart * dy + yStart) * dx + xStart);
  dev.queue.writeTexture(
    { texture, origin: [xStart, yStart, zStart] },
    data,
    { offset: offsetFloats * 4, bytesPerRow, rowsPerImage },
    [w, h, d],
  );
}

/** r32float 3D texture from a scalar volume (z-major: idx = (z*dy+y)*dx+x). */
export function createScalarTexture(dev: GPUDevice, data: Float32Array, dims: Vec3): GPUTexture {
  const [dx, dy, dz] = dims;
  const tex = dev.createTexture({
    size: [dx, dy, dz],
    dimension: "3d",
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  writeScalarTexture3D(dev, tex, data, dims);
  return tex;
}
