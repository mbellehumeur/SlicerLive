// Minimal OME-Zarr volume loader — a TS port of the viewer's fetchZarrVolume.
// Pulls all chunks in parallel, gunzips each with DecompressionStream("deflate")
// (native in both Deno and browsers — no bundled zlib), and assembles them into
// one typed array in C-order (z,y,x). The rotated IJK->RAS geometry is carried
// separately in the scene json (attrs.ijkToRAS), not in the zarr.

export interface ZarrDesc {
  dir?: string;                // legacy positional layout: "<id>.zarr"
  dataset?: string;            // legacy: "0"
  shape: [number, number, number];   // (nz, ny, nx)  C-order
  chunks: [number, number, number];  // (cz, cy, cx)
  chunkGrid: [number, number, number]; // (ncz, ncy, ncx)
  dtype: string;               // e.g. "<i2"
  bytes?: number;
  chunkHashes?: Record<string, string>; // "k.j.i" -> content hash; chunk fetched at blobBase + hash
}

type TypedArrayCtor =
  | Int8ArrayConstructor | Uint8ArrayConstructor | Int16ArrayConstructor | Uint16ArrayConstructor
  | Int32ArrayConstructor | Uint32ArrayConstructor | Float32ArrayConstructor | Float64ArrayConstructor;

type TypedArray =
  | Int8Array | Uint8Array | Int16Array | Uint16Array
  | Int32Array | Uint32Array | Float32Array | Float64Array;

const ZDT: Record<string, TypedArrayCtor> = {
  "<f4": Float32Array, "<f8": Float64Array, "<i4": Int32Array, "<u4": Uint32Array,
  "<i2": Int16Array, "<u2": Uint16Array, "|i1": Int8Array, "|u1": Uint8Array, "<i1": Int8Array, "<u1": Uint8Array,
};

export interface ZarrVolume {
  data: Float32Array;                 // scalars as f32, ready for an r32float 3D texture
  dims: [number, number, number];     // (nx, ny, nz) — i,j,k extents (texture upload order)
  range: [number, number];            // observed [min, max] scalar value
}

/** Like ZarrVolume but keeps the stored NATIVE dtype (e.g. Uint8Array for "<u1") — 4× less host
 *  RAM and GPU upload than expanding to f32, so a matching-format texture (r8unorm) is cheap. */
export interface ZarrVolumeNative {
  data: TypedArray;
  dtype: string;
  dims: [number, number, number];
  range: [number, number];
}

/** Fired when every xy chunk of a zarr `kk` slab has been written into the host buffer. */
export interface ZarrSlabInfo {
  zStart: number;
  zCount: number;
  data: TypedArray;
  dims: [number, number, number]; // [nx, ny, nz]
}

/** Fired after each chunk is assembled into the host buffer (MPR-priority progressive path). */
export interface ZarrChunkInfo {
  xStart: number;
  yStart: number;
  zStart: number;
  xCount: number;
  yCount: number;
  zCount: number;
  data: TypedArray;
  dims: [number, number, number];
}

export interface ZarrFetchOpts {
  onBytes?: (n: number) => void;
  /** Called once per completed z-slab (all ncy×ncx chunks for that kk). */
  onSlab?: (info: ZarrSlabInfo) => void;
  /** Called after each chunk write — prefer this for early MPR paints. */
  onChunk?: (info: ZarrChunkInfo) => void;
  concurrency?: number;
  /** Optional preallocated destination (length nz*ny*nx). Used by progressive scene loads. */
  into?: Float32Array;
}

async function inflateDeflate(buf: ArrayBuffer): Promise<ArrayBuffer> {
  const ds = new DecompressionStream("deflate");
  return await new Response(new Response(buf).body!.pipeThrough(ds)).arrayBuffer();
}

/** Pluggable blob fetch: a SessionStore installs one that serves content-addressed blobs from its
 *  blobs/ cache and tees network fetches into it. Default = plain fetch. */
export type BlobFetch = (url: string) => Promise<Response>;
let blobFetch: BlobFetch = (url) => fetch(url);
export function setBlobFetch(f: BlobFetch | null): void { blobFetch = f ?? ((url) => fetch(url)); }
export function getBlobFetch(): BlobFetch { return blobFetch; }

function normalizeFetchOpts(
  onBytesOrOpts?: ((n: number) => void) | ZarrFetchOpts,
  concurrency = 12,
): Required<Pick<ZarrFetchOpts, "concurrency">> & ZarrFetchOpts {
  if (onBytesOrOpts && typeof onBytesOrOpts === "object") {
    return {
      concurrency: onBytesOrOpts.concurrency ?? 12,
      onBytes: onBytesOrOpts.onBytes,
      onSlab: onBytesOrOpts.onSlab,
      onChunk: onBytesOrOpts.onChunk,
      into: onBytesOrOpts.into,
    };
  }
  return {
    concurrency,
    onBytes: onBytesOrOpts as ((n: number) => void) | undefined,
  };
}

/** Fetch + assemble a zarr volume into an f32 array (C-order z,y,x). `blobBase`
 *  is the URL prefix that `dir` is relative to. onBytes(n) reports each chunk's
 *  compressed size for a progress bar.
 *
 *  Third arg may be a legacy `onBytes` callback or a {@link ZarrFetchOpts} object
 *  (with optional `onSlab` for progressive GPU upload). */
export async function fetchZarrVolume(
  blobBase: string,
  z: ZarrDesc,
  onBytesOrOpts?: ((n: number) => void) | ZarrFetchOpts,
  concurrency = 12,
): Promise<ZarrVolume> {
  const opts = normalizeFetchOpts(onBytesOrOpts, concurrency);
  // Progressive path: assemble directly into Float32Array so onChunk/onSlab can feed GPU patches.
  if (opts.onSlab || opts.onChunk || opts.into) {
    return await fetchZarrVolumeF32(blobBase, z, opts);
  }
  const zv = await fetchZarrVolumeNative(blobBase, z, opts);
  const data = zv.data instanceof Float32Array ? zv.data : Float32Array.from(zv.data);
  return { data, dims: zv.dims, range: zv.range };
}

/** As fetchZarrVolume, but the assembled array keeps the stored dtype (no f32 expansion). */
export async function fetchZarrVolumeNative(
  blobBase: string,
  z: ZarrDesc,
  onBytesOrOpts?: ((n: number) => void) | ZarrFetchOpts,
  concurrency = 12,
): Promise<ZarrVolumeNative> {
  const opts = normalizeFetchOpts(onBytesOrOpts, concurrency);
  const Ctor = ZDT[z.dtype] ?? Int16Array;
  return await assembleZarr(blobBase, z, Ctor, opts);
}

/** Assemble straight into Float32Array (for progressive r32float uploads). */
async function fetchZarrVolumeF32(
  blobBase: string,
  z: ZarrDesc,
  opts: ZarrFetchOpts & { concurrency: number },
): Promise<ZarrVolume> {
  const zv = await assembleZarr(blobBase, z, Float32Array, opts);
  return { data: zv.data as Float32Array, dims: zv.dims, range: zv.range };
}

/**
 * Fetch order for MPR TTFF then 3D fill:
 *  1) mid-K slab (axial mid-plane) — all jj,ii at mid kk
 *  2) rest of mid-J / mid-I cross (coronal + sagittal strips)
 *  3) remaining chunks center-out
 */
export function orderChunkJobsMprFirst(
  ncz: number,
  ncy: number,
  ncx: number,
): [number, number, number][] {
  const midK = (ncz - 1) >> 1;
  const midJ = (ncy - 1) >> 1;
  const midI = (ncx - 1) >> 1;
  const dist = (kk: number, jj: number, ii: number) =>
    Math.abs(kk - midK) + Math.abs(jj - midJ) + Math.abs(ii - midI);
  const byCenter = (a: [number, number, number], b: [number, number, number]) =>
    dist(a[0], a[1], a[2]) - dist(b[0], b[1], b[2]) ||
    a[0] - b[0] || a[1] - b[1] || a[2] - b[2];

  const axial: [number, number, number][] = [];
  const cross: [number, number, number][] = [];
  const rest: [number, number, number][] = [];
  for (let kk = 0; kk < ncz; kk++) {
    for (let jj = 0; jj < ncy; jj++) {
      for (let ii = 0; ii < ncx; ii++) {
        const job: [number, number, number] = [kk, jj, ii];
        if (kk === midK) axial.push(job);
        else if (jj === midJ || ii === midI) cross.push(job);
        else rest.push(job);
      }
    }
  }
  axial.sort(byCenter);
  cross.sort(byCenter);
  rest.sort(byCenter);
  return [...axial, ...cross, ...rest];
}

async function assembleZarr(
  blobBase: string,
  z: ZarrDesc,
  Ctor: TypedArrayCtor,
  opts: ZarrFetchOpts & { concurrency: number },
): Promise<ZarrVolumeNative> {
  const [nz, ny, nx] = z.shape, [cz, cy, cx] = z.chunks, [ncz, ncy, ncx] = z.chunkGrid;
  const hashes = z.chunkHashes;
  const posBase = blobBase + z.dir + "/" + z.dataset + "/";
  const chunkUrl = (kk: number, jj: number, ii: number) =>
    hashes ? blobBase + hashes[kk + "." + jj + "." + ii] : posBase + kk + "." + jj + "." + ii;
  const expected = nz * ny * nx;
  let out: TypedArray;
  if (Ctor === Float32Array && opts.into) {
    if (opts.into.length !== expected) {
      throw new Error(`zarr into length ${opts.into.length} != ${expected}`);
    }
    out = opts.into;
  } else {
    out = new Ctor(expected);
  }
  let lo = Infinity, hi = -Infinity;
  const onBytes = opts.onBytes;
  const onSlab = opts.onSlab;
  const onChunk = opts.onChunk;
  const dims: [number, number, number] = [nx, ny, nz];

  // Per-kk remaining (jj,ii) chunk count; when it hits 0 the z-slab is complete.
  const remaining = new Int32Array(ncz);
  const perSlab = ncy * ncx;
  for (let kk = 0; kk < ncz; kk++) remaining[kk] = perSlab;

  const jobs = orderChunkJobsMprFirst(ncz, ncy, ncx);

  let idx = 0;
  const worker = async () => {
    while (idx < jobs.length) {
      const [kk, jj, ii] = jobs[idx++];
      const resp = await blobFetch(chunkUrl(kk, jj, ii));
      let gz: ArrayBuffer;
      if (resp.body && onBytes) {
        const parts: Uint8Array[] = [];
        const rd = resp.body.getReader();
        let total = 0;
        for (;;) {
          const { done, value } = await rd.read();
          if (done) break;
          parts.push(value);
          total += value.byteLength;
          onBytes(value.byteLength);
        }
        const all = new Uint8Array(total);
        let o = 0;
        for (const p of parts) { all.set(p, o); o += p.byteLength; }
        gz = all.buffer;
      } else {
        gz = await resp.arrayBuffer();
        onBytes?.(gz.byteLength);
      }
      // Source chunk uses the STORED dtype layout even when assembling into f32.
      const SrcCtor = ZDT[z.dtype] ?? Int16Array;
      const chunk = new SrcCtor(await inflateDeflate(gz));
      const z0 = kk * cz, y0 = jj * cy, x0 = ii * cx;
      const zw = Math.min(cz, nz - z0), yw = Math.min(cy, ny - y0), xw = Math.min(cx, nx - x0);
      for (let zz = 0; zz < zw; zz++) {
        for (let yy = 0; yy < yw; yy++) {
          const src = (zz * cy + yy) * cx;
          const dst = ((z0 + zz) * ny + (y0 + yy)) * nx + x0;
          for (let xx = 0; xx < xw; xx++) {
            const v = chunk[src + xx];
            out[dst + xx] = v;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
        }
      }
      onChunk?.({
        xStart: x0, yStart: y0, zStart: z0,
        xCount: xw, yCount: yw, zCount: zw,
        data: out, dims,
      });
      if (onSlab) {
        remaining[kk]--;
        if (remaining[kk] === 0) {
          onSlab({
            zStart: z0,
            zCount: zw,
            data: out,
            dims,
          });
        }
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(opts.concurrency, jobs.length) }, worker));

  return { data: out, dtype: z.dtype, dims, range: [lo, hi] };
}
