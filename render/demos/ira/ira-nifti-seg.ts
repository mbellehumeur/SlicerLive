/**
 * Convert a Cast nifti-send multilabel volume into an IRA SegLabelmap
 * resampled onto the open CT IJK grid (RAS nearest-neighbor).
 */
import { batchContextFiles } from "../../../../../cast-interface/cast-js-client/dist/cast-client.js";
import { parseNifti } from "../../../logic/readers/nifti.ts";
import type { SegLabelmap, SegmentTerminology } from "../../vendor/idc_tools/types.js";

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

/** Invert a row-major 4×4 affine (same layout as CTVolume / parseNifti ijkToRAS). */
function invAffineRowMajor(m: ArrayLike<number>): number[] {
  const a = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const t = [m[3], m[7], m[11]];
  const det =
    a[0]! * (a[4]! * a[8]! - a[5]! * a[7]!) -
    a[1]! * (a[3]! * a[8]! - a[5]! * a[6]!) +
    a[2]! * (a[3]! * a[7]! - a[4]! * a[6]!);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) {
    throw new Error("NIfTI ijkToRAS is singular — cannot resample onto CT");
  }
  const id = 1 / det;
  const r = [
    (a[4]! * a[8]! - a[5]! * a[7]!) * id,
    (a[2]! * a[7]! - a[1]! * a[8]!) * id,
    (a[1]! * a[5]! - a[2]! * a[4]!) * id,
    (a[5]! * a[6]! - a[3]! * a[8]!) * id,
    (a[0]! * a[8]! - a[2]! * a[6]!) * id,
    (a[2]! * a[3]! - a[0]! * a[5]!) * id,
    (a[3]! * a[7]! - a[4]! * a[6]!) * id,
    (a[1]! * a[6]! - a[0]! * a[7]!) * id,
    (a[0]! * a[4]! - a[1]! * a[3]!) * id,
  ];
  return [
    r[0]!,
    r[1]!,
    r[2]!,
    -(r[0]! * t[0]! + r[1]! * t[1]! + r[2]! * t[2]!),
    r[3]!,
    r[4]!,
    r[5]!,
    -(r[3]! * t[0]! + r[4]! * t[1]! + r[5]! * t[2]!),
    r[6]!,
    r[7]!,
    r[8]!,
    -(r[6]! * t[0]! + r[7]! * t[1]! + r[8]! * t[2]!),
    0,
    0,
    0,
    1,
  ];
}

function applyRowMajor(
  m: ArrayLike<number>,
  i: number,
  j: number,
  k: number,
): [number, number, number] {
  return [
    m[0]! * i + m[1]! * j + m[2]! * k + m[3]!,
    m[4]! * i + m[5]! * j + m[6]! * k + m[7]!,
    m[8]! * i + m[9]! * j + m[10]! * k + m[11]!,
  ];
}

/**
 * Nearest-neighbor resample of multilabel NIfTI onto the CT IJK grid via RAS.
 * Fixes TotalSeg / ITK axis flips vs IDC DICOM volumes that share dims but not index order.
 */
function resampleLabelmapToCt(
  src: ArrayLike<number>,
  srcDims: [number, number, number],
  srcIjkToRas: ArrayLike<number>,
  ctDims: [number, number, number],
  ctIjkToRas: ArrayLike<number>,
): { lab: Uint8Array; present: Set<number>; clipped: number } {
  const [snx, sny, snz] = srcDims;
  const [nx, ny, nz] = ctDims;
  const rasToSrc = invAffineRowMajor(srcIjkToRas);
  const lab = new Uint8Array(nx * ny * nz);
  const present = new Set<number>();
  let clipped = 0;

  for (let k = 0; k < nz; k++) {
    const plane = k * nx * ny;
    for (let j = 0; j < ny; j++) {
      const row = plane + j * nx;
      for (let i = 0; i < nx; i++) {
        const [rx, ry, rz] = applyRowMajor(ctIjkToRas, i, j, k);
        const [si, sj, sk] = applyRowMajor(rasToSrc, rx, ry, rz);
        const ii = Math.round(si);
        const jj = Math.round(sj);
        const kk = Math.round(sk);
        if (ii < 0 || jj < 0 || kk < 0 || ii >= snx || jj >= sny || kk >= snz) {
          continue;
        }
        let v = Math.round(Number(src[kk * snx * sny + jj * snx + ii]));
        if (!Number.isFinite(v) || v <= 0) continue;
        if (v > 255) {
          clipped++;
          v = 255;
        }
        lab[row + i] = v;
        present.add(v);
      }
    }
  }
  return { lab, present, clipped };
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
    seg: { lab, colors, names },
  };
}
