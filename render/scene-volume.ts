// Load a real SlicerLive scene volume into an ImageField for the WebGPU renderer.
// Parses the scene json ({blobBase,nodes}), fetches the scalar volume (zarr chunks),
// and builds a color/opacity LUT from the scene's VolumePropertyNode transfer
// functions (falling back to a window/level grayscale ramp). The rotated ijkToRAS
// geometry is handed straight to the ImageField — no resampling to an axis-aligned box.

import { ImageField } from "./fields.ts";
import { fetchZarrVolume, type ZarrDesc } from "./zarr.ts";
import { adaptMrsonScene, isMrsonScene } from "./mrson.ts";
import type { Vec3 } from "./mat4.ts";

interface Node {
  id: string;
  class: string;
  name?: string;
  refs?: Record<string, string[]>;
  attrs?: Record<string, unknown>;
  blobs?: Record<string, unknown>;
}
interface SceneWrapper { blobBase?: string; nodes?: Record<string, Node>; url?: string }

/** URL or inline LiveScene document (`blobBase` + `nodes`). */
export type SceneJsonSource = string | {
  blobBase?: string;
  nodes?: Record<string, unknown>;
  url?: string;
  [key: string]: unknown;
};

export type TF = number[][]; // color: [s,r,g,b][]; opacity: [s,a][]

/** Piecewise-linear interpolation of a transfer function (sorted by first column) at scalar s. */
export function interpTF(tf: TF, s: number, comps: number): number[] {
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

/** Build a 256-entry rgba8 LUT sampling color+opacity TFs across [lo,hi]. */
export function lutFromTransferFunctions(colorTF: TF, opacityTF: TF, clim: [number, number]): Uint8Array {
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const s = clim[0] + (i / 255) * (clim[1] - clim[0]);
    const [r, g, b] = interpTF(colorTF, s, 3);
    const [a] = interpTF(opacityTF, s, 1);
    lut[i * 4 + 0] = Math.round(Math.max(0, Math.min(1, r)) * 255);
    lut[i * 4 + 1] = Math.round(Math.max(0, Math.min(1, g)) * 255);
    lut[i * 4 + 2] = Math.round(Math.max(0, Math.min(1, b)) * 255);
    lut[i * 4 + 3] = Math.round(Math.max(0, Math.min(1, a)) * 255);
  }
  return lut;
}

/** Grayscale ramp with a linear opacity foot from window/level (fallback when no VolumeProperty). */
export function lutFromWindowLevel(): Uint8Array {
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const g = Math.round(t * 255);
    lut[i * 4 + 0] = lut[i * 4 + 1] = lut[i * 4 + 2] = g;
    lut[i * 4 + 3] = Math.round(Math.max(0, Math.min(1, (t - 0.15) / 0.85)) * 200); // soft foot
  }
  return lut;
}

/** A markup control point read from the scene (RAS world position). */
export interface Markup { ras: Vec3; label: string; color: [number, number, number] }

export interface SceneVolume {
  field: ImageField;
  voxels: Float32Array;         // raw scalar volume (HU for CT), (z,y,x) C-order — for the segmenter
  dims: [number, number, number];
  ijkToRAS: number[];
  name: string;
  range: [number, number];      // observed voxel [min,max]
  center: Vec3;                  // world center (for camera framing)
  radius: number;               // world bounding radius (for camera distance)
  win: number;                  // display window (for MPR grayscale)
  lev: number;                  // display level
  markups: Markup[];            // fiducial control points from the scene's Markups nodes (RAS)
  /** Scene VolumeProperty bake (LUT + clim + shade) for VR presets. */
  vr: { lut: Uint8Array; clim: [number, number]; shade: [number, number, number, number] };
}

/** Read all Markups-node control points from a scene's node dict, in RAS. Control points
 *  accept either `{position:[x,y,z], label}` or a bare `[x,y,z]`; colour from the node. */
function parseMarkups(nodes: Record<string, Node>): Markup[] {
  const out: Markup[] = [];
  for (const n of Object.values(nodes)) {
    if (!/Markups.*Node$/.test(n.class)) continue;
    const cps = (n.attrs?.controlPoints ?? n.attrs?.markups) as unknown[] | undefined;
    if (!Array.isArray(cps)) continue;
    const color = (n.attrs?.color as [number, number, number]) ?? [1, 0.85, 0.2];
    cps.forEach((cp, i) => {
      const c = cp as { position?: number[]; label?: string } & number[];
      const p = (c.position ?? cp) as number[];
      if (!Array.isArray(p) || p.length < 3) return;
      out.push({ ras: [p[0], p[1], p[2]], label: c.label ?? `${n.name ?? "F"}-${i + 1}`, color });
    });
  }
  return out;
}

/** Fetch a scene json + its scalar volume and return a renderable ImageField. */
export interface LoadOpts {
  /** Extra RAS translation folded into ijkToRAS — e.g. the Multi-Volume selftest offsets
   *  the second volume +200mm along R via a linear transform node. */
  extraTranslationRAS?: Vec3;
  /** Abort further pyramid upgrades (in-flight chunk fetches are not cancelled). */
  signal?: AbortSignal;
}

/** One multiscale zarr level (level 0 = full res; higher = coarser). */
export interface SceneZarrLevel {
  level: number;
  z: ZarrDesc;
  ijkToRAS: number[];
}

/** Scale ijk step vectors so a coarse grid spans the same RAS box (origin fixed). */
export function scaledIjkToRAS(base: number[], factor: number): number[] {
  const m = base.slice();
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) m[r * 4 + c] *= factor;
  }
  return m;
}

function levelFromRecord(L: Record<string, unknown>, fallbackIjk: number[]): SceneZarrLevel {
  const level = Number(L.level ?? 0);
  const z: ZarrDesc = {
    dir: L.dir as string | undefined,
    dataset: String(L.dataset ?? level),
    shape: L.shape as [number, number, number],
    chunks: L.chunks as [number, number, number],
    chunkGrid: L.chunkGrid as [number, number, number],
    dtype: String(L.dtype ?? "<f4"),
    bytes: L.bytes as number | undefined,
    chunkHashes: L.chunkHashes as Record<string, string> | undefined,
  };
  const ijk = Array.isArray(L.ijkToRAS)
    ? (L.ijkToRAS as number[]).slice()
    : scaledIjkToRAS(fallbackIjk, 2 ** level);
  return { level, z, ijkToRAS: ijk };
}

/**
 * Resolve pyramid levels: `attrs.zarrLevels`, else `blobBase/meta.json` levels,
 * else the single `attrs.zarr` (slab-progressive fallback).
 */
export async function resolveSceneLevels(parsed: {
  blobBase: string;
  vol: Node;
  z: ZarrDesc;
  ijkToRAS: number[];
}): Promise<SceneZarrLevel[]> {
  const attrs = parsed.vol.attrs ?? {};
  if (Array.isArray(attrs.zarrLevels) && (attrs.zarrLevels as unknown[]).length > 0) {
    return (attrs.zarrLevels as Record<string, unknown>[])
      .map((L) => levelFromRecord(L, parsed.ijkToRAS))
      .sort((a, b) => a.level - b.level);
  }
  try {
    const r = await fetch(parsed.blobBase + "meta.json");
    if (r.ok) {
      const meta = await r.json() as { levels?: Record<string, unknown>[] };
      if (Array.isArray(meta.levels) && meta.levels.length > 0) {
        return meta.levels
          .map((L) => levelFromRecord(L, parsed.ijkToRAS))
          .sort((a, b) => a.level - b.level);
      }
    }
  } catch {
    /* no sidecar meta */
  }
  return [{ level: 0, z: parsed.z, ijkToRAS: parsed.ijkToRAS.slice() }];
}

type ParsedScene = {
  nodes: Record<string, Node>;
  blobBase: string;
  vol: Node;
  z: ZarrDesc;
  ijkToRAS: number[];
  lut: Uint8Array;
  clim: [number, number];
  shade: [number, number, number, number];
  win: number;
  lev: number;
  markups: Markup[];
};

async function parseSceneJson(
  sceneSource: SceneJsonSource,
  opts: LoadOpts = {},
): Promise<ParsedScene> {
  const pageBase = (globalThis as { location?: { href?: string } }).location?.href ?? "file:///";
  let raw: SceneWrapper | Record<string, Node>;
  let sceneAbs: string;
  if (typeof sceneSource === "string") {
    raw = await (await fetch(sceneSource)).json() as SceneWrapper | Record<string, Node>;
    sceneAbs = new URL(sceneSource, pageBase).href;
  } else {
    raw = sceneSource as SceneWrapper;
    const hint = typeof sceneSource.url === "string" ? sceneSource.url.trim() : "";
    const blobHint = String(sceneSource.blobBase || "").trim();
    // Absolute blobBase does not need a document URL; relative needs a base.
    if (/^https?:\/\//i.test(blobHint) || blobHint.startsWith("blob:")) {
      sceneAbs = hint || blobHint || pageBase;
    } else {
      sceneAbs = hint || pageBase;
    }
  }
  const adapted = isMrsonScene(raw) ? adaptMrsonScene(raw) as unknown as SceneWrapper : raw;
  const wrapper = (adapted as SceneWrapper).nodes ? adapted as SceneWrapper : { nodes: adapted as Record<string, Node> };
  const nodes = wrapper.nodes!;
  const blobBase = new URL(wrapper.blobBase ?? "./blobs/", sceneAbs).href;

  const vol = Object.values(nodes).find((n) => n.class === "vtkMRMLScalarVolumeNode" && n.attrs?.zarr);
  if (!vol) throw new Error("no zarr ScalarVolumeNode in scene");
  const z = vol.attrs!.zarr as ZarrDesc;
  let ijkToRAS = vol.attrs!.ijkToRAS as number[];
  if (!ijkToRAS) throw new Error("volume node has no ijkToRAS");
  if (opts.extraTranslationRAS) {
    const t = opts.extraTranslationRAS;
    ijkToRAS = ijkToRAS.slice();
    ijkToRAS[3] += t[0]; ijkToRAS[7] += t[1]; ijkToRAS[11] += t[2];
  }

  let vp: Node | undefined;
  for (const dispId of vol.refs?.display ?? []) {
    const disp = nodes[dispId];
    for (const vpId of disp?.refs?.volumeProperty ?? []) {
      if (nodes[vpId]?.class === "vtkMRMLVolumePropertyNode") vp = nodes[vpId];
    }
  }

  const disp0 = nodes[(vol.refs?.display ?? [])[0]]?.attrs ?? {};
  // Prefer scene display W/L for clim when no VolumeProperty; observed range is unknown until stream ends.
  const winFallback = (disp0.window as number) ?? 1400;
  const levFallback = (disp0.level as number) ?? 300;

  let lut: Uint8Array, clim: [number, number], shade: [number, number, number, number];
  if (vp?.attrs?.color && vp?.attrs?.scalarOpacity) {
    const colorTF = vp.attrs.color as TF, opacityTF = vp.attrs.scalarOpacity as TF;
    const lo = colorTF[0][0], hi = colorTF[colorTF.length - 1][0];
    clim = [lo, hi];
    lut = lutFromTransferFunctions(colorTF, opacityTF, clim);
    shade = vp.attrs.shade ? [0.25, 0.75, 0.5, 24] : [1, 0, 0, 1];
  } else {
    clim = [levFallback - winFallback / 2, levFallback + winFallback / 2];
    lut = lutFromWindowLevel();
    shade = [0.25, 0.75, 0.5, 24];
  }

  const win = (disp0.window as number) ?? winFallback;
  const lev = (disp0.level as number) ?? levFallback;

  return {
    nodes, blobBase, vol, z, ijkToRAS, lut, clim, shade, win, lev,
    markups: parseMarkups(nodes),
  };
}

function sceneVolumeFromParts(
  field: ImageField,
  voxels: Float32Array,
  dims: [number, number, number],
  parsed: ParsedScene,
  range: [number, number],
  ijkToRAS: number[] = parsed.ijkToRAS,
): SceneVolume {
  const [lo, hi] = field.aabb();
  const center: Vec3 = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const radius = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
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
    vr: { lut: parsed.lut, clim: parsed.clim, shade: parsed.shade },
  };
}

/** Fetch a scene json + its scalar volume and return a renderable ImageField (waits for full zarr). */
export async function loadSceneVolumeField(
  dev: GPUDevice,
  sceneSource: SceneJsonSource,
  onBytes?: (n: number) => void,
  opts: LoadOpts = {},
): Promise<SceneVolume> {
  return loadSceneVolumeFieldProgressive(dev, sceneSource, { onBytes }, opts);
}

export interface ProgressiveLoadHandlers {
  onBytes?: (n: number) => void;
  /** Called as soon as the ImageField exists (zeros + scene LUT/geometry). */
  onReady?: (sv: SceneVolume) => void;
  /** After each completed z-slab has been uploaded to the *displayed* `sv.field`. */
  onSlab?: (zStart: number, zCount: number, sv: SceneVolume) => void;
  /**
   * Finer pyramid level fully slab-filled into a pending ImageField and ready to replace
   * the displayed volume (coarse→fine LOD upgrade).
   */
  onUpgrade?: (sv: SceneVolume) => void;
}

/** Slab-stream one zarr level into a new ImageField. Optionally notify onReady/onSlab. */
async function loadLevelSlabProgressive(
  dev: GPUDevice,
  parsed: ParsedScene,
  lev: SceneZarrLevel,
  handlers: ProgressiveLoadHandlers,
  opts: { notifyReady: boolean; notifySlabs: boolean },
): Promise<SceneVolume> {
  const [nz, ny, nx] = lev.z.shape;
  const dims: [number, number, number] = [nx, ny, nz];
  const voxels = new Float32Array(nx * ny * nz);
  // Below-window fill so unset slabs read as black in MPR (zeros window to mid-gray).
  voxels.fill(parsed.lev - parsed.win / 2 - 1);
  const field = new ImageField(dev, voxels, dims, [1, 1, 1], parsed.lut, {
    clim: parsed.clim,
    ijkToRAS: lev.ijkToRAS,
    shade: parsed.shade,
  });
  const sv = sceneVolumeFromParts(field, voxels, dims, parsed, [0, 0], lev.ijkToRAS);
  if (opts.notifyReady) handlers.onReady?.(sv);

  const zv = await fetchZarrVolume(parsed.blobBase, lev.z, {
    onBytes: handlers.onBytes,
    into: voxels,
    onChunk: (info) => {
      field.updateScalarsRegion(
        voxels,
        info.xStart, info.yStart, info.zStart,
        info.xCount, info.yCount, info.zCount,
      );
      // Reuse onSlab as "something painted" so IRA can coalesce rAF redraws.
      if (opts.notifySlabs) handlers.onSlab?.(info.zStart, info.zCount, sv);
    },
  });
  sv.range = zv.range;
  return sv;
}

/**
 * Stream a scene volume with optional coarse→fine pyramid LOD.
 *
 * - Single-scale (`attrs.zarr` only): allocate below-window fill, onReady, patch as z-slabs complete.
 * - Multiscale (`attrs.zarrLevels` or `blobBase/meta.json`): mount coarsest with slabs, then
 *   slab-fill each finer level into a *pending* ImageField and fire onUpgrade when that
 *   level is complete (caller swaps the displayed field).
 */
export async function loadSceneVolumeFieldProgressive(
  dev: GPUDevice,
  sceneSource: SceneJsonSource,
  handlers: ProgressiveLoadHandlers = {},
  opts: LoadOpts = {},
): Promise<SceneVolume> {
  const parsed = await parseSceneJson(sceneSource, opts);
  const levels = await resolveSceneLevels(parsed);

  // Coarse→fine: highest level index first.
  const byCoarse = [...levels].sort((a, b) => b.level - a.level);
  const coarse = byCoarse[0]!;
  let current = await loadLevelSlabProgressive(dev, parsed, coarse, handlers, {
    notifyReady: true,
    notifySlabs: true,
  });

  for (let i = 1; i < byCoarse.length; i++) {
    if (opts.signal?.aborted) break;
    const lev = byCoarse[i]!;
    // Pending fine field: slab-fill GPU texture off-screen, then upgrade.
    const pending = await loadLevelSlabProgressive(dev, parsed, lev, handlers, {
      notifyReady: false,
      notifySlabs: false,
    });
    if (opts.signal?.aborted) {
      pending.field.destroy();
      break;
    }
    // Hand ownership of `pending.field` to the viewer via onUpgrade; do not destroy the
    // previously displayed field here — swapVolumeField (or an unused twin) owns that.
    current = pending;
    handlers.onUpgrade?.(current);
  }

  return current;
}
