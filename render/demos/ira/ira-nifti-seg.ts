/**
 * Convert a Cast nifti-send multilabel volume into an IRA SegLabelmap
 * resampled onto the open CT IJK grid (RAS nearest-neighbor).
 */
import { batchContextFiles } from "../../../../../SlicerHub-js-clients/packages/client/dist/hub-client.js";
import { parseNifti } from "../../../logic/readers/nifti.ts";
import { resampleLabelmapToCt } from "../../labelmap-resample.ts";
import type { SegLabelmap, SegmentTerminology } from "../../vendor/idc_tools/types.js";

export { resampleLabelmapToCt } from "../../labelmap-resample.ts";

function fileBytes(entry: Record<string, unknown>): Uint8Array | null {
  const data = entry.data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (data instanceof Uint8Array) return data;
  if (typeof data === "string" && data) {
    try {
      const bin = atob(data.includes(",") ? data.split(",").pop()! : data);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    } catch {
      return null;
    }
  }
  return null;
}

/** Stable saturated colour from label id (golden-angle hue). */
export function colorForLabel(id: number): [number, number, number] {
  const h = ((id * 0.618033988749895) % 1) * 6;
  const x = 1 - Math.abs((h % 2) - 1);
  let r = 0,
    g = 0,
    b = 0;
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

export type NiftiSegResult =
  | { ok: true; seg: SegLabelmap; fileName: string; labelCount: number }
  | { ok: false; error: string };

/**
 * Merge an incoming Cast labelmap into an existing SEG.
 * Existing non-zero voxels win on conflict; incoming labels are remapped to free ids (1..255).
 */
export function mergeSegLabelmaps(
  existing: SegLabelmap | undefined,
  incoming: SegLabelmap,
): { seg: SegLabelmap; added: number; skipped: number } {
  if (!existing?.lab?.length) {
    return { seg: incoming, added: Object.keys(incoming.names).length, skipped: 0 };
  }
  if (existing.lab.length !== incoming.lab.length) {
    throw new Error(
      `Cannot merge SEG: length ${existing.lab.length} vs ${incoming.lab.length}`,
    );
  }

  const used = new Set<number>();
  for (const v of existing.lab) if (v) used.add(v);
  for (const [num] of existing.colors) if (num) used.add(num);

  const freeIds: number[] = [];
  for (let id = 1; id <= 255; id++) {
    if (!used.has(id)) freeIds.push(id);
  }

  const incomingIds = [
    ...new Set(
      [
        ...incoming.colors.map(([num]) => num),
        ...Object.keys(incoming.names).map(Number),
      ].filter((n) => n > 0),
    ),
  ].sort((a, b) => a - b);

  const remap = new Map<number, number>();
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
    if (lab[i]) continue; // keep original segments
    const src = incoming.lab[i];
    if (!src) continue;
    const dst = remap.get(src);
    if (dst) lab[i] = dst;
  }

  const colors: [number, number, number, number][] = [...existing.colors];
  const names: Record<number, string> = { ...existing.names };
  const terminology: Record<number, SegmentTerminology> = existing.terminology
    ? { ...existing.terminology }
    : {};

  for (const [oldId, newId] of remap) {
    const fromColors = incoming.colors.find(([num]) => num === oldId);
    if (fromColors) {
      colors.push([newId, fromColors[1], fromColors[2], fromColors[3]]);
    } else {
      const [r, g, b] = colorForLabel(newId);
      colors.push([newId, r, g, b]);
    }
    names[newId] =
      incoming.names[oldId] ||
      (fromColors ? `Segment ${oldId}` : `Segment ${newId}`);
    const term = incoming.terminology?.[oldId];
    if (term) terminology[newId] = term;
  }

  return {
    seg: {
      lab,
      colors,
      names,
      terminology: Object.keys(terminology).length ? terminology : undefined,
      // Prefer existing DICOM identity when merging Cast overlays onto it.
      sopInstanceUID: existing.sopInstanceUID || incoming.sopInstanceUID,
      seriesInstanceUID: existing.seriesInstanceUID || incoming.seriesInstanceUID,
      sourceSeriesInstanceUID:
        existing.sourceSeriesInstanceUID || incoming.sourceSeriesInstanceUID,
      fileName: incoming.fileName || existing.fileName,
      modality: existing.modality || incoming.modality || "SEG",
    },
    added: remap.size,
    skipped,
  };
}

/**
 * Parse the first nifti-send file in ``event`` and resample onto the open CT grid.
 */
export async function segLabelmapFromNiftiSendEvent(
  event: Record<string, unknown>,
  ctDims: [number, number, number],
  ctIjkToRas: ArrayLike<number>,
): Promise<NiftiSegResult> {
  const files = batchContextFiles(event) as Record<string, unknown>[];
  let chosen: Record<string, unknown> | null = null;
  let bytes: Uint8Array | null = null;
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

  const fileName =
    (typeof chosen.fileName === "string" && chosen.fileName.trim()) ||
    "segmentations.nii.gz";

  let vol;
  try {
    vol = await parseNifti(bytes, fileName);
  } catch (err) {
    return {
      ok: false,
      error: `NIfTI parse failed: ${(err as Error)?.message ?? err}`,
    };
  }

  const [snx, sny, snz] = vol.dims;
  const nSrc = snx * sny * snz;
  if (vol.data.length < nSrc) {
    return {
      ok: false,
      error: `NIfTI voxel buffer too short (${vol.data.length} < ${nSrc})`,
    };
  }
  if (!ctIjkToRas || ctIjkToRas.length < 12) {
    return { ok: false, error: "Open volume has no ijkToRAS — cannot align NIfTI" };
  }
  if (!vol.ijkToRAS || vol.ijkToRAS.length < 12) {
    return { ok: false, error: "NIfTI has no ijkToRAS — cannot align to CT" };
  }

  let lab: Uint8Array;
  let present: Set<number>;
  let clipped: number;
  try {
    ({ lab, present, clipped } = resampleLabelmapToCt(
      vol.data,
      vol.dims,
      vol.ijkToRAS,
      ctDims,
      ctIjkToRas,
    ));
  } catch (err) {
    return {
      ok: false,
      error: `NIfTI→CT resample failed: ${(err as Error)?.message ?? err}`,
    };
  }

  if (clipped) {
    console.warn(
      `[ira-nifti-seg] ${clipped} voxel(s) had label >255; clamped to 255`,
    );
  }
  if (!present.size) {
    return { ok: false, error: "NIfTI labelmap is empty after RAS resample" };
  }

  const colors: [number, number, number, number][] = [];
  const names: Record<number, string> = {};
  for (const id of [...present].sort((a, b) => a - b)) {
    const [r, g, b] = colorForLabel(id);
    colors.push([id, r, g, b]);
    names[id] = `Segment ${id}`;
  }

  return {
    ok: true,
    fileName,
    labelCount: present.size,
    seg: { lab, colors, names, fileName, modality: "SEG" },
  };
}
