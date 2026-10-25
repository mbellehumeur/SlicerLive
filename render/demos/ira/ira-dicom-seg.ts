/**
 * Convert a Cast dicom-send DICOM SEG into an IRA SegLabelmap
 * resampled onto the open CT IJK grid (same geometry path as IDC SEG).
 */
import { batchContextFiles } from "../../../../../SlicerHub-js-clients/packages/client/dist/hub-client.js";
import { loadDcmjs } from "../../../logic/readers/dicom-series.ts";
import type { SegLabelmap, SegmentTerminology } from "../../vendor/idc_tools/types.js";
import { colorForLabel } from "./ira-nifti-seg.ts";

export type DicomSegResult =
  | { ok: true; seg: SegLabelmap; fileName: string; labelCount: number }
  | { ok: false; error: string };

type CtGeom = {
  dims: [number, number, number];
  ijkToRAS: ArrayLike<number>;
  /** DICOM LPS ImageOrientationPatient; optional if SEG Shared FG has orientation. */
  iop?: number[];
  /** DICOM PixelSpacing [row, col]; optional if SEG Shared FG has spacing. */
  ps?: number[];
};

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

function sub(a: number[], b: number[]): number[] {
  return [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
}

const lps2ras = (v: number[]): [number, number, number] => [-v[0]!, -v[1]!, v[2]!];

function invAffine(m: ArrayLike<number>): number[] {
  const a = [m[0]!, m[1]!, m[2]!, m[4]!, m[5]!, m[6]!, m[8]!, m[9]!, m[10]!];
  const t = [m[3]!, m[7]!, m[11]!];
  const det =
    a[0]! * (a[4]! * a[8]! - a[5]! * a[7]!) -
    a[1]! * (a[3]! * a[8]! - a[5]! * a[6]!) +
    a[2]! * (a[3]! * a[7]! - a[4]! * a[6]!);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) {
    throw new Error("CT ijkToRAS is singular — cannot map DICOM SEG");
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

function asSeq(value: unknown): unknown[] {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

function code(seq: unknown): { scheme: string; value: string; meaning: string } | null {
  const items = asSeq(seq);
  const c = items[0] as Record<string, unknown> | undefined;
  if (!c || typeof c !== "object") return null;
  return {
    scheme: String(c.CodingSchemeDesignator || ""),
    value: String(c.CodeValue || ""),
    meaning: String(c.CodeMeaning || ""),
  };
}

function pixelDataBits(ds: Record<string, unknown>): Uint8Array {
  let pd: unknown = ds.PixelData;
  if (Array.isArray(pd)) pd = pd[0];
  if (pd instanceof ArrayBuffer) return new Uint8Array(pd);
  if (pd instanceof Uint8Array) return pd;
  if (pd && typeof (pd as ArrayBufferView).buffer === "object") {
    const view = pd as ArrayBufferView;
    return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  }
  throw new Error("DICOM SEG has no PixelData");
}

/**
 * Port of idc-worker ``buildLabelmap`` — map binary SEG frames onto CT IJK.
 */
function buildLabelmap(
  ds: Record<string, unknown>,
  bits: Uint8Array,
  ct: CtGeom,
  dcmjs: { data: { Colors: { dicomlab2RGB: (lab: number[]) => number[] } } },
): SegLabelmap {
  const [nx, ny, nz] = ct.dims;
  const segCols = Number(ds.Columns) || nx;
  const segRows = Number(ds.Rows) || ny;
  const frameBytes = (segCols * segRows) >> 3;
  const lab = new Uint8Array(nx * ny * nz);
  const inv = invAffine(ct.ijkToRAS);
  const toIJK = (lps: number[]) => {
    const r = lps2ras(lps);
    return [
      inv[0]! * r[0]! + inv[1]! * r[1]! + inv[2]! * r[2]! + inv[3]!,
      inv[4]! * r[0]! + inv[5]! * r[1]! + inv[6]! * r[2]! + inv[7]!,
      inv[8]! * r[0]! + inv[9]! * r[1]! + inv[10]! * r[2]! + inv[11]!,
    ];
  };

  const sharedSeq = asSeq(ds.SharedFunctionalGroupsSequence);
  const shared = (sharedSeq[0] as Record<string, unknown>) || {};
  const orientSeq = asSeq(shared.PlaneOrientationSequence);
  const orient0 = (orientSeq[0] as Record<string, unknown>) || {};
  const measuresSeq = asSeq(shared.PixelMeasuresSequence);
  const measures0 = (measuresSeq[0] as Record<string, unknown>) || {};

  const sIopRaw =
    orient0.ImageOrientationPatient ??
    ct.iop ??
    [1, 0, 0, 0, 1, 0];
  const sPsRaw = measures0.PixelSpacing ?? ct.ps ?? [1, 1];
  const sIop = (Array.isArray(sIopRaw) ? sIopRaw : [sIopRaw]).map(Number);
  const sPs = (Array.isArray(sPsRaw) ? sPsRaw : [sPsRaw]).map(Number);
  const colW = sIop.slice(0, 3).map((v) => v * sPs[1]!);
  const rowW = sIop.slice(3, 6).map((v) => v * sPs[0]!);

  const colors: [number, number, number, number][] = [];
  const names: Record<number, string> = {};
  const terminology: Record<number, SegmentTerminology> = {};

  for (const s of asSeq(ds.SegmentSequence) as Record<string, unknown>[]) {
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
    const num = Number(s.SegmentNumber);
    if (!Number.isFinite(num) || num <= 0) continue;
    colors.push([num, rgb[0]!, rgb[1]!, rgb[2]!]);
    names[num] = String(s.SegmentLabel || `Segment ${num}`);
    const type = s.SegmentedPropertyTypeCodeSequence;
    const type0 = asSeq(type)[0] as Record<string, unknown> | undefined;
    terminology[num] = {
      category: code(s.SegmentedPropertyCategoryCodeSequence),
      type: code(type),
      typeModifier: code(type0?.SegmentedPropertyTypeModifierCodeSequence),
      region: code(s.AnatomicRegionSequence),
    };
  }

  const perFrame = asSeq(ds.PerFrameFunctionalGroupsSequence) as Record<
    string,
    unknown
  >[];
  const firstFg = perFrame[0] || {};
  const firstPos = asSeq(
    (firstFg as { PlanePositionSequence?: unknown }).PlanePositionSequence,
  )[0] as { ImagePositionPatient?: unknown } | undefined;
  const ref = (
    Array.isArray(firstPos?.ImagePositionPatient)
      ? firstPos!.ImagePositionPatient!
      : [0, 0, 0]
  ).map(Number);
  const o0 = toIJK(ref);
  const diCol = sub(
    toIJK([ref[0]! + colW[0]!, ref[1]! + colW[1]!, ref[2]! + colW[2]!]),
    o0,
  );
  const diRow = sub(
    toIJK([ref[0]! + rowW[0]!, ref[1]! + rowW[1]!, ref[2]! + rowW[2]!]),
    o0,
  );

  for (let f = 0; f < perFrame.length; f++) {
    const fg = perFrame[f]!;
    const segIdSeq = asSeq(fg.SegmentIdentificationSequence)[0] as
      | { ReferencedSegmentNumber?: unknown }
      | undefined;
    const segNum = Number(segIdSeq?.ReferencedSegmentNumber);
    const posSeq = asSeq(fg.PlanePositionSequence)[0] as
      | { ImagePositionPatient?: unknown }
      | undefined;
    const ippRaw = posSeq?.ImagePositionPatient;
    if (!segNum || !Array.isArray(ippRaw)) continue;
    const ippLps = ippRaw.map(Number);
    const o = toIJK(ippLps);
    const fb = f * frameBytes;
    for (let row = 0; row < segRows; row++) {
      const bi = o[0]! + row * diRow[0]!;
      const bj = o[1]! + row * diRow[1]!;
      const bk = o[2]! + row * diRow[2]!;
      const rb = row * segCols;
      for (let col = 0; col < segCols; col++) {
        const p = rb + col;
        if (!((bits[fb + (p >> 3)]! >> (p & 7)) & 1)) continue;
        const i = Math.round(bi + col * diCol[0]!);
        const j = Math.round(bj + col * diCol[1]!);
        const k = Math.round(bk + col * diCol[2]!);
        if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz) {
          lab[k * nx * ny + j * nx + i] = segNum;
        }
      }
    }
  }

  return { lab, colors, names, terminology };
}

function isDicomSeg(ds: Record<string, unknown>): boolean {
  const modality = String(ds.Modality || "").toUpperCase();
  if (modality === "SEG") return true;
  const sop = String(ds.SOPClassUID || "");
  return sop.includes("1.2.840.10008.5.1.4.1.1.66.4");
}

/**
 * Parse the first dicom-send SEG file in ``event`` onto the open CT grid.
 */
export async function segLabelmapFromDicomSendEvent(
  event: Record<string, unknown>,
  ct: CtGeom,
): Promise<DicomSegResult> {
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
    return { ok: false, error: "dicom-send had no downloadable file payloads" };
  }

  const fileName =
    (typeof chosen.fileName === "string" && chosen.fileName.trim()) ||
    "segmentations.dcm";

  if (!ct.ijkToRAS || ct.ijkToRAS.length < 12) {
    return { ok: false, error: "Open volume has no ijkToRAS — cannot align DICOM SEG" };
  }

  let dcmjs: Awaited<ReturnType<typeof loadDcmjs>>;
  let ds: Record<string, unknown>;
  try {
    dcmjs = await loadDcmjs();
    const buf =
      bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
        ? bytes.buffer
        : bytes.slice().buffer;
    ds = dcmjs.data.DicomMetaDictionary.naturalizeDataset(
      dcmjs.data.DicomMessage.readFile(buf).dict,
    ) as Record<string, unknown>;
  } catch (err) {
    return {
      ok: false,
      error: `DICOM parse failed: ${(err as Error)?.message ?? err}`,
    };
  }

  if (!isDicomSeg(ds)) {
    const mod = String(ds.Modality || "?");
    return {
      ok: false,
      error: `Expected DICOM SEG, got Modality=${mod} (RTSTRUCT is not supported)`,
    };
  }

  let bits: Uint8Array;
  try {
    bits = pixelDataBits(ds);
  } catch (err) {
    return { ok: false, error: (err as Error)?.message ?? String(err) };
  }

  let seg: SegLabelmap;
  try {
    seg = buildLabelmap(ds, bits, ct, dcmjs);
  } catch (err) {
    return {
      ok: false,
      error: `DICOM SEG→CT map failed: ${(err as Error)?.message ?? err}`,
    };
  }

  const present = new Set<number>();
  for (const v of seg.lab) if (v) present.add(v);
  if (!present.size) {
    return { ok: false, error: "DICOM SEG labelmap is empty after resample onto CT" };
  }

  const referencedSeries = asSeq(ds.ReferencedSeriesSequence)[0] as
    | { SeriesInstanceUID?: unknown }
    | undefined;

  seg.sopInstanceUID = String(ds.SOPInstanceUID || "");
  seg.seriesInstanceUID = String(ds.SeriesInstanceUID || "");
  seg.sourceSeriesInstanceUID = String(
    referencedSeries?.SeriesInstanceUID || "",
  );
  seg.fileName = fileName;
  seg.modality = "SEG";

  return {
    ok: true,
    fileName,
    labelCount: present.size,
    seg,
  };
}
