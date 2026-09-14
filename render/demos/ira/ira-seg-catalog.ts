/**
 * Cast STATUS segmentation catalog — metadata only (IRA keeps SEG voxels).
 * Always present on IRA status-response as context key ``segmentation``.
 */
import type { SegLabelmap, SegmentTerminology } from "../../vendor/idc_tools/types.js";

export type SegCatalogCoded = {
  scheme: string;
  value: string;
  meaning: string;
};

export type SegCatalogSegment = {
  number: number;
  label: string;
  /** RGB in 0..1 */
  color: [number, number, number];
  terminology?: {
    category: SegCatalogCoded | null;
    type: SegCatalogCoded | null;
    typeModifier: SegCatalogCoded | null;
    region: SegCatalogCoded | null;
  };
  voxelCount: number;
  volumeMm3: number;
};

export type SegCatalogInstance = {
  sopInstanceUID: string;
  seriesInstanceUID: string;
  sourceSeriesInstanceUID: string;
  fileName: string;
  modality: string;
  segments: SegCatalogSegment[];
};

export type SegCatalog = {
  segmentations: SegCatalogInstance[];
};

export const EMPTY_SEG_CATALOG: SegCatalog = { segmentations: [] };

/** Absolute voxel volume (mm³) from row-major IJK→RAS. */
export function voxelVolumeMm3(ijkToRAS: ArrayLike<number>): number {
  const a = Number(ijkToRAS[0]);
  const b = Number(ijkToRAS[1]);
  const c = Number(ijkToRAS[2]);
  const d = Number(ijkToRAS[4]);
  const e = Number(ijkToRAS[5]);
  const f = Number(ijkToRAS[6]);
  const g = Number(ijkToRAS[8]);
  const h = Number(ijkToRAS[9]);
  const i = Number(ijkToRAS[10]);
  const det =
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  return Number.isFinite(det) ? Math.abs(det) : 0;
}

function coded(
  c: { scheme: string; value: string; meaning: string } | null | undefined,
): SegCatalogCoded | null {
  if (!c) return null;
  return {
    scheme: String(c.scheme || ""),
    value: String(c.value || ""),
    meaning: String(c.meaning || ""),
  };
}

function termFor(
  terminology: SegmentTerminology | undefined,
): SegCatalogSegment["terminology"] | undefined {
  if (!terminology) return undefined;
  return {
    category: coded(terminology.category),
    type: coded(terminology.type),
    typeModifier: coded(terminology.typeModifier),
    region: coded(terminology.region),
  };
}

export type SegInstanceMeta = {
  sopInstanceUID?: string;
  seriesInstanceUID?: string;
  sourceSeriesInstanceUID?: string;
  fileName?: string;
  modality?: string;
};

/** Count voxels per segment number (nonzero labels only). */
export function countSegmentVoxels(lab: Uint8Array): Map<number, number> {
  const counts = new Map<number, number>();
  for (let i = 0; i < lab.length; i++) {
    const v = lab[i]!;
    if (!v) continue;
    counts.set(v, (counts.get(v) || 0) + 1);
  }
  return counts;
}

/**
 * Build a STATUS catalog from the currently displayed labelmap.
 * Merged overlays are reported as one segmentation instance (current display).
 */
export function buildSegCatalog(
  labmap: SegLabelmap | null | undefined,
  ijkToRAS: ArrayLike<number> | null | undefined,
  meta?: SegInstanceMeta | null,
): SegCatalog {
  if (!labmap?.lab?.length) return { segmentations: [] };

  const voxVol = ijkToRAS && ijkToRAS.length >= 12 ? voxelVolumeMm3(ijkToRAS) : 0;
  const counts = countSegmentVoxels(labmap.lab);
  const numbers = [
    ...new Set([
      ...labmap.colors.map(([n]) => n),
      ...Object.keys(labmap.names).map(Number),
      ...counts.keys(),
    ]),
  ]
    .filter((n) => n > 0)
    .sort((a, b) => a - b);

  const segments: SegCatalogSegment[] = [];
  for (const num of numbers) {
    const voxelCount = counts.get(num) || 0;
    const fromColors = labmap.colors.find(([n]) => n === num);
    const color: [number, number, number] = fromColors
      ? [fromColors[1], fromColors[2], fromColors[3]]
      : [1, 1, 1];
    segments.push({
      number: num,
      label: labmap.names[num] || `Segment ${num}`,
      color,
      terminology: termFor(labmap.terminology?.[num]),
      voxelCount,
      volumeMm3: voxelCount * voxVol,
    });
  }

  if (!segments.length) return { segmentations: [] };

  const instance: SegCatalogInstance = {
    sopInstanceUID: String(meta?.sopInstanceUID || labmap.sopInstanceUID || ""),
    seriesInstanceUID: String(
      meta?.seriesInstanceUID || labmap.seriesInstanceUID || "",
    ),
    sourceSeriesInstanceUID: String(
      meta?.sourceSeriesInstanceUID || labmap.sourceSeriesInstanceUID || "",
    ),
    fileName: String(meta?.fileName || labmap.fileName || ""),
    modality: String(meta?.modality || labmap.modality || "SEG"),
    segments,
  };

  return { segmentations: [instance] };
}

/** Extract ``segmentation`` resource from a STATUS / ImagingStudy context array. */
export function segCatalogFromContext(context: unknown): SegCatalog | null {
  if (!Array.isArray(context)) return null;
  for (const item of context) {
    if (!item || typeof item !== "object") continue;
    const row = item as { key?: unknown; resource?: unknown };
    if (String(row.key || "").toLowerCase() !== "segmentation") continue;
    const resource = row.resource;
    if (!resource || typeof resource !== "object") {
      return { segmentations: [] };
    }
    const segs = (resource as { segmentations?: unknown }).segmentations;
    if (!Array.isArray(segs)) return { segmentations: [] };
    return { segmentations: segs as SegCatalogInstance[] };
  }
  return null;
}
