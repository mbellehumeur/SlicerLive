// Load a LiveScene `vtkMRMLSegmentationNode` labelmap into an IRA SegLabelmap.
// Fetches the gzipped content-addressed blob, then nearest-neighbor resamples onto
// the CT IJK grid (scene labelmaps are often a tight crop of the volume).

import type { SegLabelmap } from "./vendor/idc_tools/types.js";
import { resampleLabelmapToCt } from "./labelmap-resample.ts";
import type { SceneJsonSource } from "./scene-volume.ts";

interface Node {
  id: string;
  class: string;
  name?: string;
  attrs?: Record<string, unknown>;
  blobs?: Record<string, unknown>;
}

interface BlobMeta {
  hash?: string;
  dtype?: string;
  count?: number;
  comps?: number;
  size?: number;
}

const DTYPE: Record<string, new (buffer: ArrayBuffer) => ArrayLike<number>> = {
  float32: Float32Array,
  "<f4": Float32Array,
  float64: Float64Array,
  "<f8": Float64Array,
  uint8: Uint8Array,
  "|u1": Uint8Array,
  int16: Int16Array,
  "<i2": Int16Array,
  uint16: Uint16Array,
  "<u2": Uint16Array,
  int32: Int32Array,
  "<i4": Int32Array,
  uint32: Uint32Array,
  "<u4": Uint32Array,
};

async function gunzip(bytes: ArrayBuffer): Promise<ArrayBuffer> {
  const body = new Response(bytes).body;
  if (!body) throw new Error("empty labelmap response body");
  return new Response(body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}

function isGzip(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

async function resolveSceneWrapper(
  sceneSource: SceneJsonSource,
): Promise<{ nodes: Record<string, Node>; blobBase: string }> {
  const pageBase =
    (globalThis as { location?: { href?: string } }).location?.href ?? "file:///";
  let raw: { blobBase?: string; nodes?: Record<string, Node>; url?: string } | Record<string, Node>;
  let sceneAbs: string;
  if (typeof sceneSource === "string") {
    raw = (await (await fetch(sceneSource)).json()) as typeof raw;
    sceneAbs = new URL(sceneSource, pageBase).href;
  } else {
    raw = sceneSource as typeof raw;
    const hint = typeof sceneSource.url === "string" ? sceneSource.url.trim() : "";
    const blobHint = String(sceneSource.blobBase || "").trim();
    if (/^https?:\/\//i.test(blobHint) || blobHint.startsWith("blob:")) {
      sceneAbs = hint || blobHint || pageBase;
    } else {
      sceneAbs = hint || pageBase;
    }
  }
  const wrapper = (raw as { nodes?: Record<string, Node> }).nodes
    ? (raw as { blobBase?: string; nodes: Record<string, Node> })
    : { nodes: raw as Record<string, Node> };
  const nodes = wrapper.nodes;
  if (!nodes || typeof nodes !== "object") {
    throw new Error("LiveScene has no nodes");
  }
  const blobBase = new URL(wrapper.blobBase ?? "./blobs/", sceneAbs).href;
  return { nodes, blobBase };
}

function blobUrl(blobBase: string, hash: string): string {
  const h = String(hash || "").trim();
  if (/^https?:\/\//i.test(h) || h.startsWith("blob:")) return h;
  return new URL(h, blobBase).href;
}

function asDims3(v: unknown): [number, number, number] | null {
  if (!Array.isArray(v) || v.length < 3) return null;
  const a = [Number(v[0]), Number(v[1]), Number(v[2])];
  if (!a.every((n) => Number.isFinite(n) && n > 0)) return null;
  return [a[0]!, a[1]!, a[2]!];
}

function asMat16(v: unknown): number[] | null {
  if (!Array.isArray(v) || v.length < 12) return null;
  const m = v.map(Number);
  if (!m.every((n) => Number.isFinite(n))) return null;
  while (m.length < 16) m.push(m.length % 5 === 0 ? 1 : 0);
  return m.slice(0, 16);
}

function buildColorsAndNames(
  attrs: Record<string, unknown>,
  present: Set<number>,
): { colors: [number, number, number, number][]; names: Record<number, string> } {
  const colors: [number, number, number, number][] = [];
  const names: Record<number, string> = {};
  const rawColors = attrs.segmentColors;
  if (Array.isArray(rawColors)) {
    for (const row of rawColors) {
      if (!Array.isArray(row) || row.length < 4) continue;
      const num = Math.round(Number(row[0]));
      if (!num || !present.has(num)) continue;
      colors.push([num, Number(row[1]), Number(row[2]), Number(row[3])]);
    }
  }
  const segments = attrs.segments;
  if (Array.isArray(segments) && Array.isArray(rawColors)) {
    for (let i = 0; i < segments.length && i < rawColors.length; i++) {
      const row = rawColors[i];
      const seg = segments[i] as { name?: string } | null;
      if (!Array.isArray(row) || !seg) continue;
      const num = Math.round(Number(row[0]));
      if (!num || !present.has(num)) continue;
      const name = String(seg.name || "").trim();
      if (name) names[num] = name;
    }
  }
  for (const id of present) {
    if (!colors.some(([n]) => n === id)) {
      colors.push([id, 0.7, 0.7, 0.7]);
    }
    if (!names[id]) names[id] = `Segment ${id}`;
  }
  colors.sort((a, b) => a[0] - b[0]);
  return { colors, names };
}

export type SceneSegResult =
  | { ok: true; seg: SegLabelmap }
  | { ok: false; error: string };

/**
 * Load the first LiveScene segmentation labelmap and resample it onto the CT grid.
 * Returns `{ ok: false }` when the scene has no usable segmentation (not an error for
 * volume-only scenes).
 */
export async function loadSceneSegLabelmap(
  sceneSource: SceneJsonSource,
  ctDims: [number, number, number],
  ctIjkToRas: ArrayLike<number>,
  opts: { signal?: AbortSignal } = {},
): Promise<SceneSegResult> {
  let nodes: Record<string, Node>;
  let blobBase: string;
  try {
    ({ nodes, blobBase } = await resolveSceneWrapper(sceneSource));
  } catch (err) {
    return { ok: false, error: (err as Error)?.message ?? String(err) };
  }

  const segNode = Object.values(nodes).find(
    (n) =>
      n.class === "vtkMRMLSegmentationNode" &&
      n.blobs &&
      typeof n.blobs === "object" &&
      (n.blobs as Record<string, unknown>).labelmap,
  );
  if (!segNode) {
    return { ok: false, error: "no vtkMRMLSegmentationNode with labelmap blob" };
  }

  const attrs = (segNode.attrs ?? {}) as Record<string, unknown>;
  const meta = segNode.blobs!.labelmap as BlobMeta;
  const hash = String(meta.hash || "").trim();
  if (!hash) return { ok: false, error: "segmentation labelmap blob missing hash" };

  const dims = asDims3(attrs.labelmapDims);
  if (!dims) return { ok: false, error: "segmentation missing labelmapDims" };
  const srcIjk = asMat16(attrs.labelmapIjkToRAS);
  if (!srcIjk) return { ok: false, error: "segmentation missing labelmapIjkToRAS" };

  const url = blobUrl(blobBase, hash);
  const res = await fetch(url, { signal: opts.signal });
  if (!res.ok) {
    return { ok: false, error: `labelmap fetch failed (${res.status}) ${url}` };
  }
  let raw = new Uint8Array(await res.arrayBuffer());
  if (isGzip(raw)) {
    raw = new Uint8Array(await gunzip(raw.buffer));
  }

  const Ctor = DTYPE[String(meta.dtype || "float32").toLowerCase()] ?? Float32Array;
  const needed = dims[0] * dims[1] * dims[2] * (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  if (raw.byteLength < needed) {
    return {
      ok: false,
      error: `labelmap buffer too short (${raw.byteLength} < ${needed})`,
    };
  }
  const src = new Ctor(raw.buffer, raw.byteOffset, dims[0] * dims[1] * dims[2]);

  let lab: Uint8Array;
  let present: Set<number>;
  let clipped: number;
  try {
    ({ lab, present, clipped } = resampleLabelmapToCt(
      src,
      dims,
      srcIjk,
      ctDims,
      ctIjkToRas,
    ));
  } catch (err) {
    return {
      ok: false,
      error: `scene labelmap→CT resample failed: ${(err as Error)?.message ?? err}`,
    };
  }

  if (clipped) {
    console.warn(`[scene-seg] ${clipped} voxel(s) had label >255; clamped to 255`);
  }
  if (!present.size) {
    return { ok: false, error: "scene labelmap empty after RAS resample onto CT" };
  }

  const { colors, names } = buildColorsAndNames(attrs, present);
  return {
    ok: true,
    seg: {
      lab,
      colors,
      names,
      fileName: segNode.name || "LiveScene segmentation",
      modality: "SEG",
    },
  };
}
