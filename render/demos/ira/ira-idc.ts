/**
 * Map Cast ImagingStudy-open (open-mode idc, CRDC prefix URLs) → IRA Source for loadSeries.
 */
import { resolveImagingStudyOpenPlan } from "../../../../../cast-interface/cast-js-client/dist/cast-client.js";

export type IraIdcSource = {
  c: string;
  cb: string;
  s?: string;
  sb?: string;
  m: string;
  col: string;
  st: string;
  sd: string;
  lic: string;
};

type PlanFile = {
  url?: string;
  fileName?: string;
  role?: string;
  label?: string;
};

function bucketFromIdcUrl(urlStr: string): string {
  try {
    const host = new URL(urlStr).hostname.toLowerCase();
    // idc-open-data.s3.amazonaws.com → idc-open-data
    const m = host.match(/^([^.]+)\.s3[.-]/);
    if (m) return m[1];
    if (host.includes("idc-open-data")) return "idc-open-data";
  } catch {
    /* ignore */
  }
  return "idc-open-data";
}

/** First path segment = CRDC series UUID (prefix under the bucket). */
function crdcPrefixFromIdcUrl(urlStr: string): string {
  try {
    const path = new URL(urlStr).pathname.replace(/^\/+|\/+$/g, "");
    const first = path.split("/")[0]?.trim() || "";
    return first;
  } catch {
    return "";
  }
}

function isSegFile(f: PlanFile): boolean {
  const role = String(f.role || "").toLowerCase();
  const label = String(f.label || "").toLowerCase();
  const name = String(f.fileName || "").toLowerCase();
  return role === "seg" || label === "seg" || name.includes("seg");
}

/**
 * Convert ImagingStudy-open context array to IRA direct-S3 Source, or null if not idc/prefix.
 */
export function sourceFromIdcOpenContext(
  context: unknown[]
): IraIdcSource | null {
  const plan = resolveImagingStudyOpenPlan(context) as {
    mode?: string;
    studyInstanceUID?: string;
    studyId?: string;
    files?: PlanFile[];
  } | null;
  if (!plan || plan.mode !== "idc") {
    return null;
  }
  const files = Array.isArray(plan.files) ? plan.files : [];
  if (!files.length) return null;

  const segFile = files.find(isSegFile);
  const volFile =
    files.find((f) => !isSegFile(f)) || files[0];
  const volUrl = String(volFile?.url || "").trim();
  if (!volUrl) return null;
  const c = crdcPrefixFromIdcUrl(volUrl);
  if (!c) return null;
  const cb = bucketFromIdcUrl(volUrl);

  let s: string | undefined;
  let sb: string | undefined;
  if (segFile?.url) {
    const segUrl = String(segFile.url).trim();
    s = crdcPrefixFromIdcUrl(segUrl) || undefined;
    sb = s ? bucketFromIdcUrl(segUrl) : undefined;
  }

  const st = String(plan.studyInstanceUID || "").trim();
  const id = String(plan.studyId || "IDC").trim();
  return {
    c,
    cb,
    s,
    sb: sb || (s ? cb : undefined),
    m: "CT",
    col: "IDC",
    st,
    sd: `${id} · ${c.slice(0, 8)}…`,
    lic: "NCI Imaging Data Commons",
  };
}

/** Map IDC source → IRA URL query params for a full page load. */
export function studyParamsFromIdcSource(
  source: IraIdcSource
): Record<string, string> {
  const params: Record<string, string> = {
    series: source.c,
    bucket: source.cb || "idc-open-data",
  };
  if (source.s) {
    params.seg = source.s;
    params.segBucket = source.sb || source.cb || "idc-open-data";
  }
  const patient = String(source.sd || "")
    .split("·")[0]
    ?.trim();
  if (patient) params.patient = patient;
  if (source.m) params.modality = source.m;
  if (source.col) params.collection = source.col;
  return params;
}
