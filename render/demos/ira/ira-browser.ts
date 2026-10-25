// ira-browser.ts — SlicerLive "Integrated Reporting Applications" gallery demo.
//
// Loads a study when the URL supplies series/StudyInstanceUID (OHIF/IID/direct-S3) or
// ?scene=<SlicerLive scene JSON URL>. With no params the viewer stays empty — Cast/worklist
// opens studies later in-place (no page reload; shareable URL via history.replaceState).
//
// Bundle for the gallery (idc-worker.js must sit next to the output):
//   deno run -A npm:esbuild@0.21.5 render/demos/ira/ira-browser.ts --bundle --format=esm \
//     --outfile=live/webgpu/ira.js
//   cp render/vendor/idc_tools/idc-worker.js live/webgpu/idc-worker.js
//   cp render/demos/ira/ira.html live/webgpu/ira.html
// then add a gallery tile linking webgpu/ira.html.
import { initDevice } from "../../device.ts";
import { loadSceneVolumeFieldProgressive, type SceneVolume } from "../../scene-volume.ts";
import { loadSceneSegLabelmap } from "../../scene-seg.ts";
import type { ImageField } from "../../fields.ts";
import { slicerDefaultOffset01, type PlaneBasis } from "../../slice-renderer.ts";
import { offset01ToMm, SliceInteractor } from "../../slice-interactor.ts";
import { buildSegrouletteScene, type SegrouletteScene } from "../segroulette-scene.ts";
import { type Crosshair4up, mountCrosshair } from "../crosshair.ts";
import { attachCameraControls, framedCamera } from "../camera-control.ts";
import { attachSliceControls } from "../slice-control.ts";
import { attachDoubleClick, attachViewGrid, type ViewGrid } from "../view-grid.ts";
import { attachWidgetControls } from "../widget-control.ts";
import type { Box, HandleMeta } from "../roi-widget.ts";
import { installChrome, type Chrome, type VizControl } from "../sl-chrome.ts";
import {
  colorizeDefaultOpacityEntries,
  resolveColorizeGroups,
  type ColorizeGroupResolved,
} from "./colorize-groups.ts";
import { buildFontAtlas } from "../../sdf-text.ts";
import { mountSegmentCards } from "../../segment-cards.ts";
import {
  ensureSegmentLabelsTranslated,
  resolveStoredLocale,
  segmentDisplayNameSync,
} from "../../../../i18n/index.ts";
import { loadSeries } from "../../vendor/idc_tools/index.js";
import type { CTVolume, SegLabelmap } from "../../vendor/idc_tools/types.js";
import {
  type IraApi,
  type IraTool,
  ensureIraToolbarShell,
  ensureLoadStudyToolbarButton,
  mountIra,
  type IraLayout,
  type Plane,
  resetIraToolbarShell,
} from "./ira.ts";

/** Header vanishing-brush toggle — enabled only while a study toolbar is mounted. */
let vanishingBrushIra: IraApi | null = null;

function syncIraVanishingBrushBtn(tool: IraTool | null): void {
  const btn = document.getElementById("iraVanishingBrushBtn") as HTMLButtonElement | null;
  if (!btn) return;
  if (tool === null) {
    btn.disabled = true;
    btn.classList.remove("is-active");
    btn.setAttribute("aria-pressed", "false");
    return;
  }
  btn.disabled = false;
  const active = tool === "marker";
  btn.classList.toggle("is-active", active);
  btn.setAttribute("aria-pressed", active ? "true" : "false");
}

function wireIraVanishingBrushBtn(): void {
  const btn = document.getElementById("iraVanishingBrushBtn");
  if (!btn || btn.dataset.wired === "1") return;
  btn.dataset.wired = "1";
  btn.addEventListener("click", () => {
    if (!vanishingBrushIra) return;
    vanishingBrushIra.setTool(vanishingBrushIra.tool() === "marker" ? "scroll" : "marker");
  });
}

/** Snapshot push (png-send) — enabled while a study toolbar is mounted. */
function syncIraSnapshotBtn(enabled: boolean): void {
  const btn = document.getElementById("iraSnapshotBtn") as HTMLButtonElement | null;
  if (!btn) return;
  btn.disabled = !enabled;
}

function wireIraSnapshotBtn(): void {
  const btn = document.getElementById("iraSnapshotBtn");
  if (!btn || btn.dataset.wired === "1") return;
  btn.dataset.wired = "1";
  btn.addEventListener("click", () => {
    void (async () => {
      const result = await publishIraPngSend();
      if (result.ok) {
        status("Snapshot sent");
      } else {
        status(result.detail || "Snapshot send failed", true);
      }
    })();
  });
}
import {
  VanishingStrokeStore,
  partitionVanishingOps,
} from "./ira-vanishing.ts";
import { SlicePlaneField, type SlicePlaneOpts } from "../../slice-plane-field.ts";
import { downloadStudyWithDialog, type SeriesRef, shareStudy } from "../idc-share.ts";
import { downloadStl, pickSegment, safeStlFileName, segmentToStl } from "../segment-stl.ts";
import { CT_VR_PRESETS, presetLUT } from "../../ct-vr-presets.ts";
import { openVrPresetMenu, type VrPresetItem } from "../vr-preset-menu.ts";
import type { Vec3 } from "../../mat4.ts";
import {
  connectIraCast,
  getIraCastSessionInfo,
  getIraLiveScene,
  isIraConferenceHost,
  isSlicerIdSubscriberName,
  publishIraPngSend,
  publishIraSceneUpdate,
  requestIraHubDicom,
  resolveHubConfig,
  setIraConferenceHostHandler,
  setIraDicomSendHandler,
  setIraImagingStudyCloseHandler,
  setIraImagingStudyContext,
  setIraImagingStudyOpenHandler,
  setIraLiveScene,
  setIraNiftiSendHandler,
  setIraPngFullSizeCaptureHandler,
  setIraSceneUpdateHandler,
  setIraSegCatalog,
  setIraSegLocaleRefreshHandler,
  syncIraStudyUrl,
  publishIraImagingStudyTransition,
  type IraOpenPublisherMeta,
} from "./ira-cast.ts";

/** Consumed once by the next mountIra (Slicer Image Display opens → fourUp). */
let pendingIraInitialLayout: IraLayout | undefined;
import {
  applySceneUpdateOps,
  buildIraDisplaySceneUpdateOps,
  buildIraLiveSceneDoc,
  IRA_SLICE_LAYOUT_BY_PLANE,
  IRA_SLICE_ORIENTATION_BY_PLANE,
  opsFromSceneUpdateContext,
  sliceToRASFromBasis,
  type IraSliceFrame,
  type IraSlicePlaneName,
} from "./ira-live-scene-bridge.ts";
import { mergeSegLabelmaps, segLabelmapFromNiftiSendEvent } from "./ira-nifti-seg.ts";
import { segLabelmapFromDicomSendEvent } from "./ira-dicom-seg.ts";
import { buildSegCatalog, EMPTY_SEG_CATALOG } from "./ira-seg-catalog.ts";
import { installIraHeader, openLoadLocalStudyDialog } from "./ira-header.ts";
import {
  sourceFromIdcOpenContext,
  studyParamsFromIdcSource,
} from "./ira-idc.ts";
import {
  iraVolumeFromLocalVolume,
  iraVolumeFromSceneVolume,
  EMBEDDED_SCENE_PARAM,
  fetchLiveSceneDoc,
  imagingStudyContextFromLocalStudy,
  imagingStudyContextFromSceneDoc,
  imagingStudyContextFromSceneUrl,
  loadVolumeFromRemoteFile,
  sceneDocFromOpenContext,
  sceneUrlFromOpenContext,
  studyParamsFromSceneDoc,
  studyParamsFromSceneUrl,
  takeEmbeddedSceneFromStorage,
  volumeFileFromOpenContext,
  vrPresetFromSceneVolume,
} from "./ira-scene.ts";
import {
  indexFiles,
  loadEntry,
  type SeriesEntry,
} from "../../../logic/readers/dicom-local.ts";
import {
  batchContextFiles,
  resolveImagingStudyOpenPlan,
} from "../../../../../SlicerHub-js-clients/packages/client/dist/hub-client.js";

function dicomSendFileBytes(entry: Record<string, unknown>): ArrayBuffer | null {
  const data = entry.data;
  if (data instanceof ArrayBuffer) return data;
  if (data instanceof Uint8Array) {
    return data.byteOffset === 0 && data.byteLength === data.buffer.byteLength
      ? data.buffer
      : data.slice().buffer;
  }
  if (typeof data === "string" && data) {
    try {
      const bin = atob(data.includes(",") ? data.split(",").pop()! : data);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out.buffer;
    } catch {
      return null;
    }
  }
  return null;
}

/** Prefer the largest non-SEG image series from a dicom-send payload. */
async function pickSeriesEntryFromDicomSend(
  event: Record<string, unknown>,
): Promise<SeriesEntry | null> {
  const files = batchContextFiles(event) as Record<string, unknown>[];
  const fileObjs: File[] = [];
  for (const [i, entry] of files.entries()) {
    const buf = dicomSendFileBytes(entry);
    if (!buf) continue;
    const name =
      (typeof entry.fileName === "string" && entry.fileName.trim()) ||
      `dicom-send-${i + 1}.dcm`;
    fileObjs.push(new File([buf], name, { type: "application/dicom" }));
  }
  if (!fileObjs.length) return null;
  const entries = await indexFiles(fileObjs);
  if (!entries.length) return null;
  const imageEntries = entries.filter((e) => {
    const mod = String(e.modality || "").toUpperCase();
    return mod !== "SEG" && mod !== "SR" && mod !== "PR";
  });
  const candidates = imageEntries.length ? imageEntries : entries;
  candidates.sort((a, b) => b.count - a.count);
  return candidates[0] || null;
}
// The demo case: an arbitrary KiTS kidney-tumour study (IDC c4kc_kits · KiTS-00051) — the
// noncontrast CT (95 slices) WITH its DICOM SEG (kidney + tumour). Which case loads is
// resolved from the URL (see resolveSource) — a drop-in for OHIF on the IDC portal:
//   OHIF / IDC-portal form:  ?StudyInstanceUIDs=<uid>[&SeriesInstanceUIDs=<uid[,uid]>]
//   IHE IID form:            ?requestType=STUDY&studyUID=<uid>[&seriesUID=<uid>]
//   Fast direct-S3 form:     ?series=<crdc_uuid>&bucket=<b>[&seg=<uuid>&segBucket=<b>]
//   SlicerLive scene form:   ?scene=<https://…/Scene.json> or ?scene=embedded (sessionStorage)
//   (no params → empty; Cast/worklist opens a study later)
const P = new URLSearchParams(location.search);

/** Active study applies Cast dicom/nifti-send overlays; cleared on dispose. */
let applyCastNiftiSeg:
  | ((event: Record<string, unknown>) => Promise<void>)
  | null = null;
let applyCastDicomSeg:
  | ((event: Record<string, unknown>) => Promise<void>)
  | null = null;
let castNiftiMountId = 0;

interface Source {
  c: string; // CT/MR/PET series S3 prefix (crdc uuid)
  cb: string; // its bucket
  s?: string; // optional SEG series prefix
  sb?: string; // SEG bucket
  m: string; // modality
  col: string; // collection
  st: string; // StudyInstanceUID
  sd: string; // patient · series label
  lic: string; // license/attribution
}

const KITS_DEFAULT: Source = {
  c: "e3e86cde-da96-44b0-9e3b-b0b7bdd5a675",
  cb: "idc-open-data",
  s: "04a800eb-2f06-4e29-a10d-934a6f5c7d47",
  sb: "idc-open-data",
  m: "CT",
  col: "c4kc_kits",
  st: "1.3.6.1.4.1.14519.5.2.1.6919.4624.368281589441706814147998236429",
  sd: "KiTS-00051 · noncontrast abdomen + kidney/tumour SEG",
  lic: "CC BY 3.0 · IDC c4kc_kits · doi:10.7937/tcia.2019.ix49e8nx",
};

// OHIF/IID UID resolution uses a SLIM, radiology-only IDC index (built offline by
// build-idc-slim.ts) hosted in a CORS-enabled bucket. Two artifacts sit under IDC_INDEX_BASE:
//   idc-rad-groups.json  — tiny (~43 KB) directory: full min/max StudyInstanceUID per 2000-row
//                          group of the parquet, sorted by StudyInstanceUID.
//   idc-rad-slim.parquet — 617k radiology series (drops pathology/SR/etc.), sorted by study,
//                          columns trimmed to what resolution needs.
// A lookup fetches the sidecar ONCE (Cache Storage, then in-memory), binary-searches it for the
// 1–2 groups whose [min,max] span the study, and RANGE-reads only those groups (~0.6 MB) — the
// 51 MB parquet is never fully downloaded. Resolved studies are memoized in localStorage, so a
// repeat launch of the same study skips the index entirely ("hit the penalty once").
//
// The public full-index hosts (GCS mirror, GitHub release) send no CORS, so a cross-origin
// browser can't read them — hence the slim copy in a CORS bucket. Default base is same-origin
// ./idc-rad/ (drop the two files next to ira.html); point ?indexBase= / __IDC_INDEX_BASE at the
// js2 bucket for the deployed gallery.
const IDC_INDEX_BASE = ((globalThis as Record<string, unknown>).__IDC_INDEX_BASE as string) ||
  P.get("indexBase") || "https://js2.jetstream-cloud.org:8001/swift/v1/idc-index/";
const GROUPS_URL = new URL("idc-rad-groups.json", IDC_INDEX_BASE).href;
const PARQUET_URL = new URL("idc-rad-slim.parquet", IDC_INDEX_BASE).href;
const HYPARQUET_ESM = "https://cdn.jsdelivr.net/npm/hyparquet@1.28.2/+esm";
const splitList = (v: string | null): string[] => (v ?? "").split(",").map((x) => x.trim()).filter(Boolean);

interface GroupDir { version: string; rowGroupSize: number; total: number; groups: { min: string; max: string }[] }
let _dirCache: GroupDir | null = null;

/** Fetch the slim-index group directory once (Cache Storage → in-memory memo). */
async function loadGroupDir(onStatus: (m: string) => void): Promise<GroupDir> {
  if (_dirCache) return _dirCache;
  onStatus("fetching the IDC index directory…");
  let resp: Response | undefined;
  try {
    // deno-lint-ignore no-explicit-any
    const cache = (globalThis as any).caches ? await (globalThis as any).caches.open("idc-rad") : null;
    if (cache) resp = await cache.match(GROUPS_URL);
    if (!resp) {
      resp = await fetch(GROUPS_URL);
      if (resp.ok && cache) await cache.put(GROUPS_URL, resp.clone());
    }
  } catch {
    resp = await fetch(GROUPS_URL);
  }
  if (!resp || !resp.ok) throw new Error(`index directory not reachable at ${GROUPS_URL} (HTTP ${resp?.status ?? "?"})`);
  _dirCache = await resp.json() as GroupDir;
  return _dirCache;
}

const SLIM_COLS = [
  "StudyInstanceUID", "SeriesInstanceUID", "crdc_series_uuid", "aws_bucket", "Modality",
  "instanceCount", "SeriesDescription", "PatientID", "collection_id", "license_short_name", "source_DOI",
];

/** Range-read every series row of a study from the slim index: find the 1–2 sorted row groups
 *  whose [min,max] span the study and read only those (~0.6 MB), never the whole parquet. */
// deno-lint-ignore no-explicit-any
async function readStudyRows(studyUID: string, onStatus: (m: string) => void): Promise<any[]> {
  const dir = await loadGroupDir(onStatus);
  const RGS = dir.rowGroupSize;
  const spans: [number, number][] = [];
  for (let i = 0; i < dir.groups.length; i++) {
    const g = dir.groups[i];
    if (studyUID >= g.min && studyUID <= g.max) spans.push([i * RGS, Math.min((i + 1) * RGS, dir.total)]);
  }
  if (!spans.length) {
    throw new Error(
      `StudyInstanceUID not found in the slim IDC index (${dir.version}): ${studyUID}. ` +
        `It may be a non-radiology study, or from a newer index. The direct form always works: ` +
        `?series=<crdc_series_uuid>&bucket=idc-open-data.`,
    );
  }
  onStatus(`range-reading the IDC index (${spans.length} group${spans.length > 1 ? "s" : ""})…`);
  try {
    // deno-lint-ignore no-explicit-any
    const hp: any = await import(HYPARQUET_ESM);
    const file = await hp.asyncBufferFromUrl({ url: PARQUET_URL });
    const metadata = await hp.parquetMetadataAsync(file); // footer once, reused per group
    const parts = await Promise.all(
      spans.map(([rowStart, rowEnd]) => hp.parquetReadObjects({ file, metadata, columns: SLIM_COLS, rowStart, rowEnd })),
    );
    const rows = parts.flat().filter((r: { StudyInstanceUID: string }) => r.StudyInstanceUID === studyUID);
    if (!rows.length) throw new Error(`StudyInstanceUID not found in the IDC index: ${studyUID}`);
    return rows;
  } catch (e) {
    if ((e as Error).message.includes("not found")) throw e;
    throw new Error(
      `couldn't range-read the slim IDC index at ${PARQUET_URL} — check the CORS bucket (?indexBase=). ` +
        `Meanwhile the direct form works: ?series=<crdc_series_uuid>&bucket=idc-open-data. (${(e as Error).message})`,
    );
  }
}

/** Decide which series to load from the URL: fast direct-S3, OHIF/IID UID (index lookup), or none. */
async function resolveSource(onStatus: (m: string) => void): Promise<Source | null> {
  const series = P.get("series");
  if (series) { // direct S3 prefix — no index needed
    return {
      c: series, cb: P.get("bucket") || "idc-open-data",
      s: P.get("seg") || undefined, sb: P.get("segBucket") || "idc-open-data",
      m: (P.get("modality") || "CT").toUpperCase(), col: P.get("collection") || "IDC", st: "",
      sd: `${P.get("patient") || "IDC"} · ${series.slice(0, 8)}…`, lic: "NCI Imaging Data Commons",
    };
  }
  const studyUIDs = [...splitList(P.get("StudyInstanceUIDs")), ...splitList(P.get("studyUID"))];
  if (studyUIDs.length) {
    const want = [
      ...splitList(P.get("SeriesInstanceUIDs")),
      ...splitList(P.get("initialSeriesInstanceUID")),
      ...splitList(P.get("seriesUID")),
    ];
    return await resolveFromIndex(studyUIDs[0], want, onStatus);
  }
  // No URL study params — open empty (Cast / worklist will open a study later).
  return null;
}

/** Resolve a StudyInstanceUID (+ optional SeriesInstanceUIDs) to an S3 image series (+ SEG)
 *  via the SLIM idc-index — the OHIF/IDC-portal drop-in path. Reads only the 1–2 sorted row
 *  groups whose [min,max] span the study (~0.6 MB), never the whole parquet. */
async function resolveFromIndex(
  studyUID: string,
  wantSeries: string[],
  onStatus: (m: string) => void,
): Promise<Source> {
  const memoKey = `idc-rad:${studyUID}:${wantSeries.join(",")}`;
  try {
    const hit = localStorage.getItem(memoKey);
    if (hit) return JSON.parse(hit) as Source;
  } catch { /* private mode / no storage */ }

  const inStudy = await readStudyRows(studyUID, onStatus);
  const want = new Set(wantSeries);
  const IMG = new Set(["CT", "MR", "PT", "PET", "NM"]);
  const imgs = inStudy.filter((r) => IMG.has(String(r.Modality).toUpperCase()));
  const segs = inStudy.filter((r) => String(r.Modality).toUpperCase() === "SEG");
  // Prefer an explicitly-requested image series; else the largest (primary) one.
  const chosen = imgs.find((r) => want.has(r.SeriesInstanceUID)) ??
    imgs.slice().sort((a, b) => Number(b.instanceCount) - Number(a.instanceCount))[0];
  if (!chosen) throw new Error("no CT/MR/PET/NM image series found in this study");
  // If a SEG was explicitly requested, honour it; else pick one whose SEG references the chosen
  // image (best-effort: just take the first SEG in the study).
  const seg = segs.find((r) => want.has(r.SeriesInstanceUID)) ?? segs[0];
  const mod = String(chosen.Modality).toUpperCase();
  const src: Source = {
    c: String(chosen.crdc_series_uuid), cb: String(chosen.aws_bucket),
    s: seg ? String(seg.crdc_series_uuid) : undefined, sb: seg ? String(seg.aws_bucket) : undefined,
    m: mod === "PET" ? "PT" : mod, col: String(chosen.collection_id), st: studyUID,
    sd: `${chosen.PatientID} · ${chosen.SeriesDescription || mod}`,
    lic: `${chosen.license_short_name || "IDC"} · ${chosen.collection_id}${chosen.source_DOI ? " · doi:" + chosen.source_DOI : ""}`,
  };
  try { localStorage.setItem(memoKey, JSON.stringify(src)); } catch { /* ignore */ }
  return src;
}

const status = (msg: string, err = false) => {
  const el = document.getElementById("status");
  if (el) {
    el.textContent = msg;
    el.style.color = err ? "#ff6b74" : "#9fb3d0";
  }
};
const cvEl = (id: string) => document.getElementById(id) as HTMLCanvasElement;

const PLANES: Plane[] = ["axial", "coronal", "sagittal", "oblique"];
const SEG_FILL = 0.5; // 2D MPR segmentation-overlay alpha at 100% (buildSegrouletteScene default)
const CT_WL_PRESETS = [
  { name: "Soft Tissue", win: 400, lev: 40 },
  { name: "Lung", win: 1500, lev: -600 },
  { name: "Bone", win: 1800, lev: 400 },
  { name: "Brain", win: 80, lev: 40 },
  { name: "Abdomen", win: 350, lev: 50 },
];

async function main() {
  // Cast header (radio + settings). Open handler is registered before connect so
  // STATUS / ImagingStudy-open can queue until loadStudy exists (GPU ready).
  let mountLocalSeries: (
    entry: SeriesEntry,
    opts: { anonymized: boolean },
  ) => Promise<void> = async () => {
    throw new Error("IRA viewer is still starting — try again in a moment");
  };
  installIraHeader({
    onMountLocalSeries: (entry, opts) => mountLocalSeries(entry, opts),
  });

  type LoadStudyFn = (
    input:
      | Source
      | { sceneUrl: string }
      | { sceneDoc: import("./ira-cast.ts").IraLiveSceneDoc }
      | {
          localVolume: CTVolume;
          label: string;
          anonymized: boolean;
        },
  ) => Promise<void>;

  let resolveLoadStudy: ((fn: LoadStudyFn) => void) | null = null;
  const loadStudyWhenReady = new Promise<LoadStudyFn>((resolve) => {
    resolveLoadStudy = resolve;
  });
  let openApplyChain: Promise<void> = Promise.resolve();

  const applyImagingStudyOpen = async (
    ctx: unknown[],
    meta?: IraOpenPublisherMeta,
  ) => {
    if (isSlicerIdSubscriberName(meta?.publisherSubscriberName)) {
      pendingIraInitialLayout = "fourUp";
    }
    const loadStudy = await loadStudyWhenReady;
    const sceneDoc = sceneDocFromOpenContext(ctx);
    if (sceneDoc) {
      status("Opening scene…");
      syncIraStudyUrl(studyParamsFromSceneDoc(sceneDoc));
      await loadStudy({ sceneDoc });
      return;
    }
    const sceneUrl = sceneUrlFromOpenContext(ctx);
    if (sceneUrl) {
      status("Opening scene…");
      syncIraStudyUrl(studyParamsFromSceneUrl(sceneUrl));
      await loadStudy({ sceneUrl });
      return;
    }
    const idcSource = sourceFromIdcOpenContext(ctx);
    if (idcSource) {
      status("Opening study…");
      syncIraStudyUrl(studyParamsFromIdcSource(idcSource));
      await loadStudy(idcSource);
      return;
    }
    const volumeFile = volumeFileFromOpenContext(ctx);
    if (volumeFile) {
      status("Downloading volume…");
      const hubEndpoint = getIraCastSessionInfo().hubEndpoint;
      let hubOrigin = "http://127.0.0.1:2018";
      try {
        hubOrigin = new URL(hubEndpoint).origin;
      } catch {
        /* keep default */
      }
      const { localVolume, label } = await loadVolumeFromRemoteFile(
        volumeFile,
        hubOrigin
      );
      syncIraStudyUrl({ volume: volumeFile.fileName });
      await loadStudy({ localVolume, label, anonymized: false });
      // loadStudy(localVolume) clears ImagingStudy context; restore Hub open
      // files so Remote AI can publish nifti-send / dicom-send jobs.
      setIraImagingStudyContext(ctx);
      syncIraStudyUrl({ volume: volumeFile.fileName });
      const info = document.getElementById("info");
      if (info) info.textContent = label;
      return;
    }
    const plan = resolveImagingStudyOpenPlan(ctx) as {
      mode?: string;
      studyInstanceUID?: string;
      seriesInstanceUID?: string;
    } | null;
    if (plan?.mode === "local-dicom") {
      const studyInstanceUID = String(plan.studyInstanceUID || "").trim();
      if (!studyInstanceUID) {
        status("local-dicom open missing studyInstanceUID", true);
        return;
      }
      status("Requesting DICOM from Image Display…");
      const result = await requestIraHubDicom({
        studyInstanceUID,
        seriesInstanceUID: plan.seriesInstanceUID,
      });
      if (!result.ok) {
        status(`dicom-request failed: ${result.detail}`, true);
      }
      // Volume arrives via inbound dicom-send handler.
      return;
    }
    status(
      "ImagingStudy-open: unsupported context (need scene JSON, idc CRDC prefix URLs, volume files, or local-dicom)",
      true,
    );
  };

  setIraImagingStudyOpenHandler((ctx, meta) => {
    status("Opening study…");
    openApplyChain = openApplyChain
      .then(() => applyImagingStudyOpen(ctx, meta))
      .catch((err) => {
        console.error("[ira] ImagingStudy-open apply failed", err);
        status("error: " + ((err as Error)?.message ?? err), true);
      });
    return openApplyChain;
  });

  let closeStudyImpl: (() => void) | null = null;
  let pendingClose = false;
  setIraImagingStudyCloseHandler(() => {
    if (closeStudyImpl) {
      closeStudyImpl();
      return;
    }
    pendingClose = true;
  });

  void connectIraCast();

  if (!(navigator as unknown as { gpu?: unknown }).gpu) {
    status("WebGPU not available — try Chrome/Edge 113+ or Safari 18+.", true);
    return;
  }
  status("starting…");
  wireIraVanishingBrushBtn();
  wireIraSnapshotBtn();
  const gpu = await initDevice();
  const preferred = (navigator as unknown as { gpu: GPU }).gpu.getPreferredCanvasFormat();
  const srgb = (preferred + "-srgb") as GPUTextureFormat;

  const names = ["axial", "threeD", "coronal", "sagittal", "oblique"] as const;
  const cv: Record<string, HTMLCanvasElement> = {}, cx: Record<string, GPUCanvasContext> = {};
  for (const n of names) {
    cv[n] = cvEl("c-" + n);
    cx[n] = cv[n].getContext("webgpu") as GPUCanvasContext;
    cx[n].configure({ device: gpu.device, format: preferred, viewFormats: [srgb], alphaMode: "opaque" });
  }

  type StudyMount = { dispose: () => void };
  let studyMount: StudyMount | null = null;
  let studyAbort: AbortController | null = null;
  let emptyCleanup: (() => void) | null = null;
  let studyLoadToken = 0;

  setIraNiftiSendHandler(async (event) => {
    if (!applyCastNiftiSeg) {
      status("nifti-send received but no study is open", true);
      return;
    }
    await applyCastNiftiSeg(event);
  });
  setIraDicomSendHandler(async (event, meta) => {
    if (isSlicerIdSubscriberName(meta?.publisherSubscriberName)) {
      pendingIraInitialLayout = "fourUp";
    }
    const fileCount = batchContextFiles(event).length;
    // Multi-file sends are study pulls (local-dicom); single-file keeps SEG overlay path.
    const preferVolumeOpen = fileCount > 1 || !applyCastDicomSeg;
    if (!preferVolumeOpen && applyCastDicomSeg) {
      await applyCastDicomSeg(event);
      return;
    }
    try {
      status("Reconstructing DICOM series…");
      const entry = await pickSeriesEntryFromDicomSend(event);
      if (!entry) {
        if (applyCastDicomSeg) {
          await applyCastDicomSeg(event);
          return;
        }
        status("dicom-send: no image series to open", true);
        return;
      }
      const vol = loadEntry(entry);
      const localVolume = iraVolumeFromLocalVolume(vol);
      const modality = String(entry.modality || localVolume.modality || "CT").trim();
      const desc = String(entry.description || vol.name || "").trim();
      const patient = String(entry.patientName || "").trim() || "Local";
      const label = [patient, modality, desc || `${entry.count} images`]
        .filter(Boolean)
        .join(" · ");
      const loadStudy = await loadStudyWhenReady;
      await loadStudy({ localVolume, label, anonymized: false });
      const info = document.getElementById("info");
      if (info) info.textContent = label;
      status(`Opened ${label}`);
    } catch (err) {
      console.error("[ira] dicom-send volume open failed", err);
      status("error: " + ((err as Error)?.message ?? err), true);
    }
  });

  const clearViewportDisplay = () => {
    for (const el of document.querySelectorAll(
      ".ann, .ira-overlay, #ira-strip, #ira-cine-box, #sl-badge, #sl-popup, #sl-help",
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
        const enc = gpu.device.createCommandEncoder();
        const pass = enc.beginRenderPass({
          colorAttachments: [
            {
              view: tex.createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
              loadOp: "clear",
              storeOp: "store",
            },
          ],
        });
        pass.end();
        gpu.device.queue.submit([enc.finish()]);
      } catch {
        /* ignore */
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
    ensureLoadStudyToolbarButton(() => openLoadLocalStudyDialog());
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
    const ros: ResizeObserver[] = [];
    for (const n of names) {
      const ro = new ResizeObserver(() => resizeEmpty());
      ro.observe(cv[n]);
      ros.push(ro);
    }
    attachViewGrid(document.getElementById("grid")!, [...names], resizeEmpty);
    attachDoubleClick(cv.threeD, () => {});
    resizeEmpty();
    emptyCleanup = () => {
      globalThis.removeEventListener("resize", resizeEmpty);
      for (const ro of ros) ro.disconnect();
      emptyCleanup = null;
    };
    status("No study loaded");
  };

  const loadStudy: LoadStudyFn = async (input) => {
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
        signal,
      );
      if (token !== studyLoadToken || signal.aborted) {
        mount.dispose();
        clearViewportDisplay();
        return;
      }
      studyMount = mount;
    } catch (err) {
      if (
        token !== studyLoadToken ||
        signal.aborted ||
        (err instanceof DOMException && err.name === "AbortError")
      ) {
        clearViewportDisplay();
        return;
      }
      console.error("[ira] study load failed", err);
      status("error: " + ((err as Error)?.message ?? err), true);
      showEmptyChrome();
    }
  };

  // Unblock any STATUS / ImagingStudy-open that arrived during GPU startup.
  resolveLoadStudy?.(loadStudy);
  resolveLoadStudy = null;

  closeStudyImpl = () => {
    setIraLiveScene(null);
    setIraSegCatalog(EMPTY_SEG_CATALOG);
    setIraImagingStudyContext([]);
    disposeStudy();
    syncIraStudyUrl({});
    showEmptyChrome();
    status("No study loaded");
  };
  if (pendingClose) {
    pendingClose = false;
    closeStudyImpl();
  }

  mountLocalSeries = async (entry, { anonymized }) => {
    status("Reconstructing local series…");
    const vol = loadEntry(entry);
    const localVolume = iraVolumeFromLocalVolume(vol);
    const modality = String(entry.modality || localVolume.modality || "CT").trim();
    const desc = String(entry.description || vol.name || "").trim();
    const patient = anonymized
      ? "Anonymous"
      : String(entry.patientName || "").trim() || "Local";
    const label = [patient, modality, desc || `${entry.count} images`]
      .filter(Boolean)
      .join(" · ");
    await loadStudy({ localVolume, label, anonymized });
    const info = document.getElementById("info");
    if (info) info.textContent = label;

    const openContext = imagingStudyContextFromLocalStudy({
      label,
      modality,
      studyInstanceUID: entry.studyInstanceUID,
      seriesInstanceUID: entry.seriesInstanceUID,
      anonymized,
    });
    await publishIraImagingStudyTransition(openContext);
  };

  // Which case? Resolve the URL — OHIF/IID StudyInstanceUID (via the IDC index), direct S3
  // prefix, SlicerLive scene JSON, or empty (Cast/worklist opens a study later).
  const setLoad = (m: string) => {
    status(m);
  };
  const sceneUrl = (P.get("scene") || "").trim();
  if (sceneUrl === EMBEDDED_SCENE_PARAM) {
    const embedded = takeEmbeddedSceneFromStorage();
    if (!embedded) {
      status("Embedded LiveScene missing — reopen from the worklist.", true);
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

async function mountStudyFromSource(
  input:
    | Source
    | { sceneUrl: string }
    | { sceneDoc: import("./ira-cast.ts").IraLiveSceneDoc }
    | {
        localVolume: CTVolume;
        label: string;
        anonymized: boolean;
      },
  gpu: Awaited<ReturnType<typeof initDevice>>,
  preferred: GPUTextureFormat,
  srgb: GPUTextureFormat,
  names: readonly ["axial", "threeD", "coronal", "sagittal", "oblique"],
  cv: Record<string, HTMLCanvasElement>,
  cx: Record<string, GPUCanvasContext>,
  signal?: AbortSignal,
): Promise<{ dispose: () => void }> {
  let source: Source;
  let ct: CTVolume;
  let seg: SegLabelmap | undefined;
  let sceneVr: {
    lut: Uint8Array;
    clim: [number, number];
    shade: [number, number, number, number];
  } | null = null;
  let sceneStream: {
    patch: (data: Float32Array, zStart: number, zCount: number) => void;
    redraw: () => void;
    upgrade?: (sv: SceneVolume) => void;
  } | null = null;
  let sceneStreamDone: Promise<unknown> | null = null;
  let sceneRedrawRaf = 0;
  /** LOD upgrade that arrived before sceneStream was wired. */
  let pendingLodUpgrade: SceneVolume | null = null;
  /** Late SEG from IDC progressive path (attached after mount). */
  let pendingIdcSeg: SegLabelmap | undefined;
  /** LiveScene embedded segmentation fetch (resolved before attach). */
  let sceneSegDone: Promise<void> | null = null;
  /** Host volume filled by IDC slabs (same buffer passed to buildSegrouletteScene). */
  let idcStreamVol: Float32Array | null = null;
  /** Set when the worker posts the final full CT (assemble done; SEG may still be loading). */
  let idcCtAssembled = false;
  /** Scene/camera for mid-stream ctgeom refinements (set after mount). */
  let scHold: SegrouletteScene | null = null;
  let applyIdcGeom: ((ijkToRAS: number[]) => void) | null = null;
  let applyIdcFocusZ: ((z: number) => void) | null = null;
  let pendingIdcFocusZ: number | null = null;

  let sceneLodField: ImageField | undefined;
  if ("localVolume" in input) {
    setIraLiveScene(null);
    setIraImagingStudyContext([]);
    syncIraStudyUrl({});
    ct = input.localVolume;
    const patientBit = input.anonymized
      ? "Anonymous"
      : (input.label.split("·")[0] || "Local").trim();
    source = {
      c: "",
      cb: "",
      m: ct.modality || "CT",
      col: "Local",
      st: "",
      sd: input.anonymized
        ? `Anonymous · ${input.label}`
        : input.label || `${patientBit} · Local`,
      lic: "Local",
    };
    status(`loaded local series · ${ct.dims.join("×")}`);
  } else if ("sceneDoc" in input || "sceneUrl" in input) {
    status("loading scene…");
    let sceneDoc: import("./ira-cast.ts").IraLiveSceneDoc | null = null;
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
    const sceneSource =
      "sceneDoc" in input
        ? {
            blobBase: input.sceneDoc.blobBase,
            nodes: input.sceneDoc.nodes,
            url: input.sceneDoc.url,
          }
        : input.sceneUrl;
    let mb = 0;
    let resolveReady!: (sv: SceneVolume) => void;
    const ready = new Promise<SceneVolume>((r) => {
      resolveReady = r;
    });
    sceneStreamDone = loadSceneVolumeFieldProgressive(gpu.device, sceneSource, {
      onBytes: (n) => {
        if (signal?.aborted) return;
        mb += n;
        status(`streaming scene… ${(mb / 1e6).toFixed(1)} MB`);
      },
      onReady: (sv) => {
        if (signal?.aborted) return;
        sceneLodField = sv.field;
        resolveReady(sv);
      },
      onSlab: (zStart, zCount, sv) => {
        if (signal?.aborted) return;
        if (sceneStream) {
          sceneStream.patch(sv.voxels, zStart, zCount);
          if (!sceneRedrawRaf) {
            sceneRedrawRaf = requestAnimationFrame(() => {
              sceneRedrawRaf = 0;
              sceneStream?.redraw();
            });
          }
        }
      },
      onUpgrade: (sv) => {
        if (signal?.aborted) return;
        status(`refining… ${sv.dims.join("×")}`);
        sceneLodField = sv.field;
        if (sceneStream?.upgrade) sceneStream.upgrade(sv);
        else pendingLodUpgrade = sv;
      },
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
      sd: `${sv.name} · LiveScene`,
      lic: "LiveScene",
    };
    // LiveScene may embed a vtkMRMLSegmentationNode (e.g. TotalSegmentator-CT).
    // Fetch/resample in parallel with the remaining volume stream; attach after mount.
    sceneSegDone = loadSceneSegLabelmap(sceneSource, ct.dims, ct.ijkToRAS, { signal }).then(
      (res) => {
        if (signal?.aborted) return;
        if (!res.ok) {
          if (!/no vtkMRMLSegmentationNode/.test(res.error)) {
            console.warn("[ira] LiveScene segmentation:", res.error);
          }
          return;
        }
        pendingIdcSeg = res.seg;
      },
    );
  } else {
    setIraLiveScene(null);
    source = input;
    status("contacting the NCI Imaging Data Commons…");
    let resolveReady!: (v: CTVolume) => void;
    const ready = new Promise<CTVolume>((r) => {
      resolveReady = r;
    });
    const modality = ({ CT: "CT", MR: "MR", PT: "PET" } as Record<string, string>)[source.m] || source.m;
    sceneStreamDone = loadSeries(source, {
      onProgress: (p: { msg: string; frac?: number }) => {
        if (signal?.aborted) return;
        status(`${p.msg}${p.frac ? ` — ${Math.round(p.frac * 100)}%` : ""}`);
      },
      onCTReady: (meta) => {
        if (signal?.aborted) return;
        const [nx, ny, nz] = meta.dims;
        // Below-window fill so unset slices read as black (zeros window to mid-gray).
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
          modality: meta.modality || modality,
        });
      },
      onCTGeom: (meta) => {
        if (signal?.aborted) return;
        if (ct) {
          ct.ijkToRAS = meta.ijkToRAS;
          if (Number.isFinite(meta.win!)) ct.win = meta.win!;
          if (Number.isFinite(meta.lev!)) ct.lev = meta.lev!;
        }
        applyIdcGeom?.(meta.ijkToRAS);
      },
      onCTFocus: (z) => {
        if (signal?.aborted) return;
        if (!Number.isFinite(z)) return;
        if (applyIdcFocusZ) applyIdcFocusZ(z);
        else pendingIdcFocusZ = z;
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
        // Keep host buffer in sync if the final transfer landed a fresh ArrayBuffer.
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
      },
    }, {
      // Cache-bust the classic worker (browsers often keep idc-worker.js sticky across IRA reloads).
      workerUrl: new URL("./idc-worker.js?v=idc-fetch-13", import.meta.url),
    }).then((res) => {
      if (res.seg) pendingIdcSeg = res.seg;
      return res;
    });
    ct = await ready;
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    seg = undefined; // SEG attaches after mount when the labelmap is ready
  }

  const sc: SegrouletteScene = buildSegrouletteScene(gpu, srgb, ct, seg, {
    volumeField: sceneLodField,
  });
  // Filled after installChrome — Colorize / refreshSegChrome refresh the badge popup.
  let chromeApi: Chrome | null = null;
  /** Assigned once chrome helpers exist (below); refreshSegChrome may call them after mount. */
  let applyColorizeGroupDefaults: () => void = () => {};
  let reinstallChrome: () => void = () => {};
  scHold = sc;
  const thisCastMountId = ++castNiftiMountId;
  if (sceneVr) sc.setVolumePreset(sceneVr);
  // Sampled scalar range of the loaded volume: drives the W/L drag gain (Slicer's
  // vtkMRMLWindowLevelWidget scales by the scalar range) and the VR shift-slider span.
  // For progressive scenes this is recomputed after the stream finishes.
  let vmin = Infinity, vmax = -Infinity;
  const sampleRange = () => {
    vmin = Infinity;
    vmax = -Infinity;
    const buf = ct.vol;
    const vstride = Math.max(1, Math.floor(buf.length / 2_000_000));
    for (let i = 0; i < buf.length; i += vstride) {
      const v = buf[i];
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
    }
  };
  sampleRange();
  const dataRange = (): [number, number] => [vmin, vmax];
  sc.setVolumeOpacity(seg ? 0.5 : 1); // with SEG: translucent VR; scene-only: full VR

  // SlicerLive display state (toggled from the badge popup, below).
  let sliceOutline = false; // segmentation overlay: fill (default) vs outline
  let roiEnabled = false, roiVisible = false, roiFirstEnable = true; // 3D volume-render crop box
  let annOn = true; // DICOM corner annotations (IHE IRA §4.16.4.2.2.5.8 "official radiology look")
  let labelsOn = true; // 3D segment label cards, on by default

  const dpr3 = Math.min(2, globalThis.devicePixelRatio || 1);
  const cardFont = buildFontAtlas({
    sizePx: 44,
    pad: 6,
    fontFamily: 'Helvetica, "Helvetica Neue", Arial, sans-serif',
  });
  let cardDt = performance.now();
  let cardDrawPending = false;
  let studyDisposed = false;

  // ---- corner annotations (IHE IRA §4.16.4.2.2.5.8) ---------------------------------------
  // The profile mandates overlaid corner text: patient identity (TL), institution/study (TR),
  // series/technique (BR); the slice position/number sits bottom-left with the readout. IDC
  // public data is de-identified, so we show what's present (ID, collection, modality, series,
  // geometry, live W/L). Sourced from the loaded metadata — real deployments read full tags.
  const annEls: Record<string, { br: HTMLElement }> = {};
  const buildAnnotations = () => {
    for (const p of PLANES) {
      const cell = document.querySelector(`[data-cell="${p}"]`) as HTMLElement;
      if (!cell) continue;
      const mk = (cls: string, html: string) => {
        const e = document.createElement("div");
        e.className = "ann " + cls;
        e.innerHTML = html;
        e.style.cssText = "position:absolute;z-index:2;font:600 10px ui-monospace,Menlo,monospace;" +
          "color:#cfe0f5;text-shadow:0 0 3px #000,0 0 3px #000;pointer-events:none;line-height:1.35;max-width:46%;" +
          (cls === "tl" ? "top:22px;left:7px;" : cls === "tr" ? "top:22px;right:7px;text-align:right;" : "bottom:22px;right:7px;text-align:right;");
        cell.appendChild(e);
        return e;
      };
      const pid = (source.sd.split("·")[0] || "").trim() || source.col;
      mk("tl", `${pid}<br>${source.col} · ${ct.modality}`);
      mk("tr", `${ct.modality} · ${sc.dims[2]} slices${sc.hasSeg ? " · SEG" : ""}<br>${sc.dims[0]}×${sc.dims[1]}`);
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
    for (const e of document.querySelectorAll(".ann")) (e as HTMLElement).style.display = annOn ? "" : "none";
  };

  const off: Record<Plane, number> = {
    axial: slicerDefaultOffset01("axial", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi),
    coronal: slicerDefaultOffset01("coronal", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi),
    sagittal: slicerDefaultOffset01("sagittal", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi),
    oblique: slicerDefaultOffset01("oblique", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi),
  };
  let sliceIx = new SliceInteractor({ ijkToRAS: sc.ijkToRAS, rasLo: sc.rasLo, rasHi: sc.rasHi });
  const camera = framedCamera(sc.center, sc.radius);
  const wl = { win: sc.win, lev: sc.lev };
  let ira: IraApi | null = null;
  const vanishingStore = new VanishingStrokeStore({
    onChange: () => {
      if (!ira) return;
      for (const p of PLANES) ira.drawOverlay(p);
      ira.drawOverlay3d();
    },
  });
  let xhair: Crosshair4up | null = null;

  // Dental Drop-Slice: textured oblique plane in the 3D view (SlicePlaneField).
  let obliqueIn3D: SlicePlaneField | null = null;
  let dentalLayout = false;
  const dot3 = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross3 = (a: Vec3, b: Vec3): Vec3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const norm3 = (v: Vec3): Vec3 => {
    const L = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / L, v[1] / L, v[2] / L];
  };
  /** Rodrigues rotation of `v` about unit axis `k` by `ang` radians. */
  const rotAxis = (v: Vec3, k: Vec3, ang: number): Vec3 => {
    const c = Math.cos(ang), s = Math.sin(ang);
    const d = dot3(k, v);
    const x = cross3(k, v);
    return [
      v[0] * c + x[0] * s + k[0] * d * (1 - c),
      v[1] * c + x[1] * s + k[1] * d * (1 - c),
      v[2] * c + x[2] * s + k[2] * d * (1 - c),
    ];
  };
  /** Pitch (about u) + yaw (about v); re-orthonormalize so u×v convention matches BASES (v = u×n). */
  const tiltBasis = (b: PlaneBasis, dPitch: number, dYaw: number): PlaneBasis => {
    let u = b.uDir, v = b.vDir, n = b.nDir;
    n = rotAxis(n, u, dPitch);
    v = rotAxis(v, u, dPitch);
    n = rotAxis(n, v, dYaw);
    u = rotAxis(u, v, dYaw);
    n = norm3(n);
    u = norm3([u[0] - n[0] * dot3(u, n), u[1] - n[1] * dot3(u, n), u[2] - n[2] * dot3(u, n)]);
    v = cross3(u, n);
    return { uDir: u, vDir: v, nDir: n };
  };
  const extentAlong = (d: Vec3): { lo: number; hi: number } => {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < 8; i++) {
      const c: Vec3 = [
        i & 1 ? sc.rasHi[0] : sc.rasLo[0],
        i & 2 ? sc.rasHi[1] : sc.rasLo[1],
        i & 4 ? sc.rasHi[2] : sc.rasLo[2],
      ];
      const t = dot3(c, d);
      if (t < lo) lo = t;
      if (t > hi) hi = t;
    }
    return { lo, hi };
  };
  /** Current oblique plane centre in RAS (scrub position; ignores pan). */
  const obliqueCenterRAS = (): Vec3 => {
    const b = sc.slice.basisOf("oblique");
    const origin: Vec3 = [
      (sc.rasLo[0] + sc.rasHi[0]) / 2,
      (sc.rasLo[1] + sc.rasHi[1]) / 2,
      (sc.rasLo[2] + sc.rasHi[2]) / 2,
    ];
    const nx = extentAlong(b.nDir);
    const want = nx.lo + Math.max(0, Math.min(1, off.oblique)) * (nx.hi - nx.lo);
    const have = dot3(origin, b.nDir);
    origin[0] += b.nDir[0] * (want - have);
    origin[1] += b.nDir[1] * (want - have);
    origin[2] += b.nDir[2] * (want - have);
    return origin;
  };
  /** Left-drag on Oblique (Scroll tool): tilt the plane; keep the RAS centre fixed. */
  const tiltObliqueByPx = (dxPx: number, dyPx: number) => {
    const sens = 0.005; // rad / px
    const center = obliqueCenterRAS();
    const next = tiltBasis(sc.slice.basisOf("oblique"), dyPx * sens, -dxPx * sens);
    sc.slice.setBasis("oblique", next);
    off.oblique = sc.slice.offset01Along("oblique", center);
    drawSlice("oblique");
    xhair?.redraw();
    syncLiveScene();
  };
  const obliquePlaneOpts = (): SlicePlaneOpts => {
    const b = sc.slice.basisOf("oblique");
    const origin = obliqueCenterRAS();
    let hu = 0, hv = 0;
    for (const sx of [sc.rasLo[0], sc.rasHi[0]]) {
      for (const sy of [sc.rasLo[1], sc.rasHi[1]]) {
        for (const sz of [sc.rasLo[2], sc.rasHi[2]]) {
          const r: Vec3 = [sx - origin[0], sy - origin[1], sz - origin[2]];
          hu = Math.max(hu, Math.abs(dot3(r, b.uDir)));
          hv = Math.max(hv, Math.abs(dot3(r, b.vDir)));
        }
      }
    }
    const halfWin = wl.win / 2;
    return {
      p2t: sc.patientToTexture(),
      origin,
      normal: b.nDir,
      uAxis: b.uDir,
      vAxis: b.vDir,
      halfExtU: hu,
      halfExtV: hv,
      clim: [wl.lev - halfWin, wl.lev + halfWin],
      halfThick: Math.max(0.25, sliceIx.spacing("oblique") / 2),
      normScale: sc.normScale(),
    };
  };
  const ensureObliqueIn3D = () => {
    const opts = obliquePlaneOpts();
    if (!obliqueIn3D) obliqueIn3D = new SlicePlaneField(sc.volumeTexture(), opts);
    else obliqueIn3D.setPlane(opts);
    return obliqueIn3D;
  };
  const setDentalObliqueVisible = (on: boolean) => {
    dentalLayout = on;
    if (on) {
      sc.setExtraFields([ensureObliqueIn3D()]);
      draw3d();
    } else {
      sc.setExtraFields([]);
      draw3d();
    }
  };
  const refreshObliqueIn3D = () => {
    if (!dentalLayout || !obliqueIn3D) return;
    obliqueIn3D.setPlane(obliquePlaneOpts());
    sc.scene.syncUniforms();
    draw3d();
  };
  const invalidateObliqueIn3D = () => {
    // Volume texture swapped (LOD) — drop the field so the next show recreates against the new tex.
    if (dentalLayout) sc.setExtraFields([]);
    obliqueIn3D = null;
    if (dentalLayout) {
      sc.setExtraFields([ensureObliqueIn3D()]);
    }
  };

  // Declared before drawSlice — refreshSegLabelTranslations/drawAll can run early in mount.
  const updateReadout = (p: Plane) => {
    const el = document.getElementById("sr-" + p);
    if (!el) return;
    const [lo, hi] = sliceIx.bounds(p);
    const sp = sliceIx.spacing(p);
    const mm = offset01ToMm(p, off[p], sc.rasLo, sc.rasHi);
    el.textContent = `${Math.round((mm - lo) / sp + 0.5)}/${Math.round((hi - lo) / sp)} · ${mm.toFixed(1)} mm`;
  };

  const drawSlice = (p: Plane) => {
    if (studyDisposed) return;
    ira?.drawOverlay(p);
    if (!cv[p].width) return;
    sc.slice.setPlane(p, off[p]);
    sc.slice.renderToView(cx[p].getCurrentTexture().createView({ format: srgb }), cv[p].width, cv[p].height);
    updateReadout(p);
    if (p === "oblique") refreshObliqueIn3D();
  };
  const draw3d = () => {
    if (studyDisposed) return;
    if (!cv.threeD.width) return;
    sc.scene.setCamera(camera.position, camera.focalPoint, camera.viewUp, camera.viewAngle, cv.threeD.width, cv.threeD.height);
    sc.scene.renderToView(cx.threeD.getCurrentTexture().createView({ format: srgb }), cv.threeD.width, cv.threeD.height);
    ira?.drawOverlay3d();
    if (!labelsOn) return;
    const now = performance.now();
    const dt = (now - cardDt) / 1000;
    cardDt = now;
    const moving = segCards.draw(
      cx.threeD.getCurrentTexture().createView({ format: srgb }),
      cv.threeD.width,
      cv.threeD.height,
      dpr3,
      dt,
    );
    // Keep drawing while cards are still gliding into place.
    if (moving && !cardDrawPending) {
      cardDrawPending = true;
      requestAnimationFrame(() => {
        cardDrawPending = false;
        draw3d();
      });
    }
  };
  let syncLiveScene = () => {
    /* assigned below once sc helpers exist */
  };
  const collectSliceFrames = (): IraSliceFrame[] => {
    const slices: IraSliceFrame[] = [];
    for (const p of PLANES) {
      const plane = p as IraSlicePlaneName;
      const layoutName = IRA_SLICE_LAYOUT_BY_PLANE[plane];
      const aspect = cv[p].width / Math.max(1, cv[p].height);
      const center = sc.slice.viewToRas(p, off[p], 0.5, 0.5, aspect) as [
        number,
        number,
        number,
      ];
      const { fovX, fovY } = sc.slice.mirrorFrame(p, aspect);
      const slab = sliceIx.spacing(p);
      const b = sc.slice.basisOf(p);
      slices.push({
        layoutName,
        orientation: IRA_SLICE_ORIENTATION_BY_PLANE[plane],
        sliceToRAS: sliceToRASFromBasis(
          [b.uDir[0], b.uDir[1], b.uDir[2]],
          [b.vDir[0], b.vDir[1], b.vDir[2]],
          [b.nDir[0], b.nDir[1], b.nDir[2]],
          center,
        ),
        fieldOfView: [fovX, fovY, slab],
      });
    }
    return slices;
  };
  let drawAll = () => {
    for (const p of PLANES) drawSlice(p);
    draw3d();
    xhair?.redraw();
  };
  const segCards = mountSegmentCards(gpu, srgb, cardFont, cv.threeD, camera, {
    apply: (id, action) => {
      if (action === "reset") sc.setSegmentOpacities(sc.segments.map((s) => [s.num, 1]));
      else if (action === "hide") sc.setSegmentOpacity(id, 0);
      else sc.setSegmentOpacities(sc.segments.map((s) => [s.num, s.num === id ? 1 : 0.4]));
      syncLiveScene();
      drawAll();
    },
    redraw: () => draw3d(),
    maxCards: 12,
    reserved: () => {
      const cr = cv.threeD.getBoundingClientRect();
      if (!cr.width || !cr.height) return [];
      const sx = cv.threeD.width / cr.width;
      const sy = cv.threeD.height / cr.height;
      const pad = 6;
      const out: { x: number; y: number; w: number; h: number }[] = [];
      for (const sel of ["#sl-badge", ".lab.threeD"]) {
        const el = document.querySelector(sel) as HTMLElement | null;
        if (!el) continue;
        const r = el.getBoundingClientRect();
        const x0 = Math.max(r.left, cr.left);
        const y0 = Math.max(r.top, cr.top);
        const x1 = Math.min(r.right, cr.right);
        const y1 = Math.min(r.bottom, cr.bottom);
        if (x1 <= x0 || y1 <= y0) continue;
        out.push({
          x: (x0 - cr.left) * sx - pad,
          y: (y0 - cr.top) * sy - pad,
          w: (x1 - x0) * sx + 2 * pad,
          h: (y1 - y0) * sy + 2 * pad,
        });
      }
      return out;
    },
    segVisible: (num) => sc.isSegmentVisible(num),
  });

  /** Display-only names for cards/chrome; wire/scene keep English on sc.segments. */
  const segmentsForUi = () =>
    sc.segments.map((s) => ({
      ...s,
      englishName: s.name,
      name: segmentDisplayNameSync(s.name) || s.name,
    }));

  const hubOriginForTranslate = (): string => {
    const ep = resolveHubConfig().hub_endpoint || "";
    return String(ep).replace(/\/api\/hub\/?$/, "").replace(/\/$/, "");
  };

  const refreshSegLabelTranslations = (): void => {
    void (async () => {
      const applyUi = () => {
        segCards.setScene(ct, seg ?? null, segmentsForUi());
        chromeApi?.refresh();
        drawAll();
      };
      if (!sc.segments.length) {
        applyUi();
        return;
      }
      const locale = resolveStoredLocale();
      if (locale !== "en") {
        const origin = hubOriginForTranslate();
        if (origin) {
          try {
            await ensureSegmentLabelsTranslated(
              sc.segments.map((s) => ({ label: s.name })),
              { hubOrigin: origin, locale },
            );
          } catch {
            /* keep sync/cache or English */
          }
        }
      }
      applyUi();
    })();
  };

  setIraSegLocaleRefreshHandler(refreshSegLabelTranslations);
  segCards.setScene(ct, seg ?? null, segmentsForUi());
  refreshSegLabelTranslations();

  // Shared with VR chrome (declared early so syncLiveScene / scene-update can use it).
  let vrShift = 0;
  let vrPreset: string | null = null;
  /** Filled once the VR preset helpers are ready (below). */
  let applyVrPresetFn: (name: string | null) => void = (name) => {
    vrPreset = name;
  };
  /** Skip host publish while applying an inbound scene-update. */
  let applyingRemoteSceneUpdate = false;
  /** Double-click maximized cell; shared with viewGrid (assigned later). */
  let maximizedCell: string | null = null;
  let viewGrid: ViewGrid | null = null;
  let hostScenePublishRaf = 0;
  let hostScenePublishPending: ReturnType<typeof buildIraLiveSceneDoc> | null = null;
  let hostScenePublishPrev: ReturnType<typeof getIraLiveScene> = null;

  const scheduleHostScenePublish = (
    doc: ReturnType<typeof buildIraLiveSceneDoc>,
    prev: ReturnType<typeof getIraLiveScene>,
    opts?: { full?: boolean },
  ) => {
    if (applyingRemoteSceneUpdate || !isIraConferenceHost()) return;
    hostScenePublishPending = doc;
    if (opts?.full) hostScenePublishPrev = null;
    else if (!hostScenePublishPrev) hostScenePublishPrev = prev;
    if (hostScenePublishRaf) return;
    hostScenePublishRaf = requestAnimationFrame(() => {
      hostScenePublishRaf = 0;
      const next = hostScenePublishPending;
      const from = hostScenePublishPrev;
      hostScenePublishPending = null;
      hostScenePublishPrev = null;
      if (!next || applyingRemoteSceneUpdate || !isIraConferenceHost()) {
        console.info("[ira] scene-update publish skip: raf gate", {
          hasNext: Boolean(next),
          applyingRemoteSceneUpdate,
          host: isIraConferenceHost(),
          role: getIraCastSessionInfo().conference?.selfRole ?? null,
        });
        return;
      }
      const ops = buildIraDisplaySceneUpdateOps(next, from);
      if (!ops.length) {
        console.info("[ira] scene-update publish skip: empty ops diff");
        return;
      }
      console.info(
        "[ira] scene-update publishing ops=",
        ops.length,
        ops.map((o) => `${o.id}${(o as { path?: string }).path || ""}`),
      );
      void publishIraSceneUpdate(ops);
    });
  };

  syncLiveScene = () => {
    if (studyDisposed) return;
    const prev = getIraLiveScene();
    const doc = buildIraLiveSceneDoc({
      base: prev,
      dims: sc.dims,
      ijkToRAS: sc.ijkToRAS,
      modality: ct.modality,
      volumeName: source.sd || ct.modality || "Volume",
      volumeOpacity: sc.volumeOpacity(),
      volumeShift: sc.volumeShift(),
      volumeShiftRange: Math.max(200, (vmax - vmin) / 2),
      volumePreset: vrPreset,
      cropEnabled: roiEnabled,
      roiVisible: roiVisible,
      segments: sc.segments.map((s) => ({
        num: s.num,
        name: s.name,
        color: s.color,
        visible: sc.isSegmentVisible(s.num),
        opacity: sc.segmentOpacity(s.num),
      })),
      labelsVisible: labelsOn,
      opacity: sc.segOpacity(),
      hasSeg: sc.hasSeg,
      camera: {
        position: [camera.position[0], camera.position[1], camera.position[2]],
        focalPoint: [camera.focalPoint[0], camera.focalPoint[1], camera.focalPoint[2]],
        viewUp: [camera.viewUp[0], camera.viewUp[1], camera.viewUp[2]],
        viewAngle: camera.viewAngle,
      },
      slices: collectSliceFrames(),
      maximizedCell,
    });
    setIraLiveScene(doc);
    scheduleHostScenePublish(doc, prev);
  };
  syncLiveScene();
  if (seg) setIraSegCatalog(buildSegCatalog(seg, sc.ijkToRAS));
  else setIraSegCatalog(EMPTY_SEG_CATALOG);

  setIraConferenceHostHandler(() => {
    if (studyDisposed || thisCastMountId !== castNiftiMountId) return;
    // Full display snapshot so followers catch up when host starts.
    syncLiveScene();
    const doc = getIraLiveScene();
    if (doc) scheduleHostScenePublish(doc, null, { full: true });
  });
  // Study loaded while already hosting (conference created earlier).
  if (isIraConferenceHost()) {
    const doc = getIraLiveScene();
    if (doc) scheduleHostScenePublish(doc, null, { full: true });
  }

  setIraSceneUpdateHandler((context) => {
    if (studyDisposed || thisCastMountId !== castNiftiMountId) return;
    const allOps = opsFromSceneUpdateContext(context);
    const { vanishing, rest } = partitionVanishingOps(allOps);
    if (vanishing.length) {
      vanishingStore.applyOps(vanishing);
    }
    if (!rest.length) return;
    applyingRemoteSceneUpdate = true;
    try {
      // Refresh from live sc so patch targets exist even if STATUS scene was volume-only.
      const base = buildIraLiveSceneDoc({
        base: getIraLiveScene(),
        dims: sc.dims,
        ijkToRAS: sc.ijkToRAS,
        modality: ct.modality,
        volumeName: source.sd || ct.modality || "Volume",
        volumeOpacity: sc.volumeOpacity(),
        volumeShift: sc.volumeShift(),
        volumeShiftRange: Math.max(200, (vmax - vmin) / 2),
        volumePreset: vrPreset,
        cropEnabled: roiEnabled,
        roiVisible: roiVisible,
        segments: sc.segments.map((s) => ({
          num: s.num,
          name: s.name,
          color: s.color,
          visible: sc.isSegmentVisible(s.num),
          opacity: sc.segmentOpacity(s.num),
        })),
        labelsVisible: labelsOn,
        opacity: sc.segOpacity(),
        hasSeg: sc.hasSeg || sc.segments.length > 0,
        camera: {
          position: [camera.position[0], camera.position[1], camera.position[2]],
          focalPoint: [camera.focalPoint[0], camera.focalPoint[1], camera.focalPoint[2]],
          viewUp: [camera.viewUp[0], camera.viewUp[1], camera.viewUp[2]],
          viewAngle: camera.viewAngle,
        },
        slices: collectSliceFrames(),
        maximizedCell,
      });
      const { doc, result } = applySceneUpdateOps(base, rest);
      setIraLiveScene(doc);
      if (typeof result.labelsVisible === "boolean") {
        labelsOn = result.labelsVisible;
      }
      // Batch opacity + visibility — per-segment setters rebake the whole volume each call;
      // hide-all / full segments rewrite would hang the browser for N≈100 TotalSeg labels.
      const opacityEntries = [...result.segmentOpacity].filter(
        ([num, op]) => sc.segmentOpacity(num) !== op,
      );
      const visibilityEntries = [...result.segmentVisibility].filter(
        ([num, visible]) => sc.isSegmentVisible(num) !== visible,
      );
      if (opacityEntries.length) sc.setSegmentOpacities(opacityEntries);
      if (visibilityEntries.length) sc.setSegmentVisibilities(visibilityEntries);
      if (typeof result.opacity === "number" && Number.isFinite(result.opacity)) {
        sc.setSegOpacity(result.opacity);
        sc.slice.setOverlayOpacity(result.opacity * SEG_FILL);
      }
      if (typeof result.volumeOpacity === "number" && Number.isFinite(result.volumeOpacity)) {
        sc.setVolumeOpacity(result.volumeOpacity);
      }
      if (typeof result.volumeShift === "number" && Number.isFinite(result.volumeShift)) {
        vrShift = result.volumeShift;
        sc.setVolumeShift(result.volumeShift);
      }
      if (typeof result.volumePreset === "string") {
        applyVrPresetFn(result.volumePreset || null);
      }
      if (typeof result.cropEnabled === "boolean") {
        roiEnabled = result.cropEnabled;
        if (result.cropEnabled && roiFirstEnable) {
          roiVisible = true;
          roiFirstEnable = false;
        }
        sc.setRoiEnabled(roiEnabled);
        sc.setRoiVisible(roiVisible);
      }
      if (typeof result.roiVisible === "boolean") {
        roiVisible = result.roiVisible;
        sc.setRoiVisible(roiVisible);
      }
      if (result.camera) {
        camera.position = [...result.camera.position];
        camera.focalPoint = [...result.camera.focalPoint];
        camera.viewUp = [...result.camera.viewUp];
        camera.viewAngle = result.camera.viewAngle;
      }
      for (const intent of result.slices) {
        const p = intent.plane;
        if (p === "oblique") {
          sc.slice.setBasis("oblique", intent.basis);
        }
        off[p] = sc.slice.offset01Along(p, intent.centerRAS);
        sc.slice.setPlane(p, off[p]);
        const fov = intent.fieldOfView;
        if (fov && fov[0] > 0 && fov[1] > 0) {
          sc.slice.setMirrorFrame(p, intent.centerRAS, fov[0], fov[1]);
        }
      }
      if (result.maximizedCell !== undefined) {
        maximizedCell = result.maximizedCell;
        viewGrid?.setMax(result.maximizedCell);
      }
      drawAll();
      // Re-paint local vanishing ink after slice/camera changes (leader has no self-echo path).
      if (ira) {
        for (const p of PLANES) ira.drawOverlay(p);
        ira.drawOverlay3d();
      }
      syncLiveScene();
    } finally {
      applyingRemoteSceneUpdate = false;
    }
  });

  const refreshSegChrome = (labmap: SegLabelmap) => {
    seg = labmap;
    sc.setVolumeOpacity(0.5);
    segCards.setScene(ct, seg, segmentsForUi());
    refreshSegLabelTranslations();
    setIraSegCatalog(buildSegCatalog(labmap, sc.ijkToRAS));
    syncLiveScene();
    // Colorize still on → re-resolve groups against the new labelmap and rebuild chrome rows.
    if (sc.segDisplayMode() === "colorize") {
      applyColorizeGroupDefaults();
      reinstallChrome();
    } else {
      chromeApi?.refresh();
    }
    for (const p of PLANES) {
      const cell = document.querySelector(`[data-cell="${p}"]`) as HTMLElement | null;
      const tr = cell?.querySelector(".ann.tr") as HTMLElement | null;
      if (tr) {
        tr.innerHTML =
          `${ct.modality} · ${sc.dims[2]} slices${sc.hasSeg ? " · SEG" : ""}<br>${sc.dims[0]}×${sc.dims[1]}`;
      }
    }
    drawAll();
  };

  applyCastNiftiSeg = async (event) => {
    if (studyDisposed || thisCastMountId !== castNiftiMountId) return;
    status("Loading Cast segmentation…");
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
      status(`Merge failed: ${(err as Error)?.message ?? err}`, true);
      return;
    }
    sc.replaceSegmentation(merged);
    if (!sc.hasSeg) {
      status("Cast segmentation produced no renderable segments", true);
      return;
    }
    refreshSegChrome(merged);
    const skipNote = skipped ? ` · ${skipped} skipped (label id limit)` : "";
    status(
      `Added ${added} TotalSeg segment${added === 1 ? "" : "s"} from ${result.fileName}${skipNote}`,
    );
  };

  applyCastDicomSeg = async (event) => {
    if (studyDisposed || thisCastMountId !== castNiftiMountId) return;
    status("Loading Cast DICOM SEG…");
    const result = await segLabelmapFromDicomSendEvent(event, {
      dims: sc.dims,
      ijkToRAS: sc.ijkToRAS,
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
      status(`Merge failed: ${(err as Error)?.message ?? err}`, true);
      return;
    }
    sc.replaceSegmentation(merged);
    if (!sc.hasSeg) {
      status("Cast DICOM SEG produced no renderable segments", true);
      return;
    }
    refreshSegChrome(merged);
    const skipNote = skipped ? ` · ${skipped} skipped (label id limit)` : "";
    status(
      `Added ${added} segment${added === 1 ? "" : "s"} from ${result.fileName}${skipNote}`,
    );
  };

  // Progressive stream: patch + redraw while slabs arrive (zarr or IDC); LOD field swaps.
  if (sceneStreamDone) {
    const applyLodUpgrade = (sv: SceneVolume) => {
      const oldCenter = [sc.center[0], sc.center[1], sc.center[2]] as [number, number, number];
      sc.swapVolumeField(sv.field, sv.dims, sv.ijkToRAS, sv.voxels);
      ct.vol = sv.voxels;
      ct.dims = sv.dims;
      ct.ijkToRAS = sv.ijkToRAS;
      for (let i = 0; i < 3; i++) {
        const d = sc.center[i] - oldCenter[i];
        camera.position[i] += d;
        camera.focalPoint[i] += d;
      }
      sliceIx = new SliceInteractor({ ijkToRAS: sc.ijkToRAS, rasLo: sc.rasLo, rasHi: sc.rasHi });
      // Re-default slice offsets to mid-volume on the new grid.
      off.axial = slicerDefaultOffset01("axial", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi);
      off.coronal = slicerDefaultOffset01("coronal", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi);
      off.sagittal = slicerDefaultOffset01("sagittal", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi);
      off.oblique = slicerDefaultOffset01("oblique", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi);
      sampleRange();
      invalidateObliqueIn3D();
      drawAll();
    };
    sceneStream = {
      patch: (data, zStart, zCount) => sc.patchVolumeScalars(data, zStart, zCount),
      redraw: () => drawAll(),
      upgrade: applyLodUpgrade,
    };
    // Catch slabs that landed between onReady and this wiring (host buffer is ahead of GPU).
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
  // Mid-stream / final ctgeom: update patient matrix and shift the camera by the center
  // delta so refining the origin does not look like a pan.
  applyIdcGeom = (ijkToRAS) => {
    const old = [sc.center[0], sc.center[1], sc.center[2]] as [number, number, number];
    sc.updateGeometry(ijkToRAS);
    for (let i = 0; i < 3; i++) {
      const d = sc.center[i] - old[i];
      camera.position[i] += d;
      camera.focalPoint[i] += d;
    }
    sliceIx = new SliceInteractor({ ijkToRAS: sc.ijkToRAS, rasLo: sc.rasLo, rasHi: sc.rasHi });
    drawAll();
  };
  // Keep MPR on the worker's painted mid slice when remaps move its IJK index.
  applyIdcFocusZ = (z) => {
    const [nx, ny, nz] = sc.dims;
    const zz = Math.max(0, Math.min(nz - 1, Math.round(z)));
    const ijk = [(nx - 1) / 2, (ny - 1) / 2, zz];
    const M = sc.ijkToRAS;
    const ras = [
      M[0] * ijk[0] + M[1] * ijk[1] + M[2] * ijk[2] + M[3],
      M[4] * ijk[0] + M[5] * ijk[1] + M[6] * ijk[2] + M[7],
      M[8] * ijk[0] + M[9] * ijk[1] + M[10] * ijk[2] + M[11],
    ];
    for (const p of PLANES) {
      if (p === "oblique") {
        off[p] = sc.slice.offset01Along("oblique", ras as Vec3);
      } else {
        const axis = p === "axial" ? 2 : p === "coronal" ? 1 : 0;
        const span = sc.rasHi[axis] - sc.rasLo[axis];
        off[p] = span === 0 ? 0.5 : (ras[axis] - sc.rasLo[axis]) / span;
        off[p] = Math.max(0, Math.min(1, off[p]));
      }
    }
    drawAll();
  };
  if (pendingIdcFocusZ != null) {
    applyIdcFocusZ(pendingIdcFocusZ);
    pendingIdcFocusZ = null;
  }

  // Size canvases and paint immediately so progressive slabs are visible (don't wait for full chrome).
  {
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    for (const n of names) {
      cv[n].width = Math.max(1, Math.round(cv[n].clientWidth * dpr));
      cv[n].height = Math.max(1, Math.round(cv[n].clientHeight * dpr));
    }
    drawAll();
  }

  // IDC: pump the event loop until assemble finishes so ctslab messages can patch+paint
  // before the rest of the chrome mounts (that work is sync and would otherwise stall paints).
  if (idcStreamVol && sceneStreamDone && !idcCtAssembled) {
    status("streaming volume…");
    while (!idcCtAssembled) {
      await Promise.race([
        new Promise<void>((r) => setTimeout(r, 16)),
        sceneStreamDone.then(() => {
          idcCtAssembled = true;
        }),
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
  const resizeObservers: ResizeObserver[] = [];
  for (const n of names) {
    const ro = new ResizeObserver(() => resize());
    ro.observe(cv[n]);
    resizeObservers.push(ro);
  }

  viewGrid = attachViewGrid(document.getElementById("grid")!, [...names], resize);
  attachDoubleClick(cv.threeD, () => {
    viewGrid!.toggleMax("threeD");
    maximizedCell = viewGrid!.maxCell();
    syncLiveScene();
  });

  const setWL = (win: number, lev: number) => {
    wl.win = win;
    wl.lev = lev;
    sc.slice.setWindowLevel(win, lev);
    for (const p of PLANES) drawSlice(p);
    refreshObliqueIn3D();
    xhair?.redraw();
    syncWl();
    updateAnnWL(); // live W/L in the corner annotation
  };
  // Hidden preset <select> the IRA "Presets" button drives (mountIra wants a presetsEl).
  const presets = document.getElementById("wl-readout") as HTMLSelectElement;
  const syncWl = () => {
    presets.innerHTML = "";
    presets.add(new Option(`W ${Math.round(wl.win)} / L ${Math.round(wl.lev)}`, "current", true, true));
    presets.add(new Option(`Auto (${Math.round(sc.win)}/${Math.round(sc.lev)})`, "auto"));
    for (const q of CT_WL_PRESETS) presets.add(new Option(`${q.name} (${q.win}/${q.lev})`, q.name));
  };
  presets.addEventListener("change", () => {
    if (presets.value === "auto") setWL(sc.win, sc.lev);
    else {
      const q = CT_WL_PRESETS.find((x) => x.name === presets.value);
      if (q) setWL(q.win, q.lev);
    }
  });
  syncWl();

  const nAxisOf: Record<Exclude<Plane, "oblique">, 0 | 1 | 2> = { axial: 2, coronal: 1, sagittal: 0 };
  const jumpAll = (ras: Vec3) => {
    for (const p of PLANES) {
      if (p === "oblique") {
        off[p] = sc.slice.offset01Along("oblique", ras);
      } else {
        const a = nAxisOf[p];
        off[p] = Math.max(0, Math.min(1, (ras[a] - sc.rasLo[a]) / (sc.rasHi[a] - sc.rasLo[a])));
      }
    }
    drawAll();
    syncLiveScene();
  };

  // The IHE Integrated Reporting Applications reader chrome (shared demos/ira/ira.ts).
  const sliceControls: { resetView(): void }[] = [];
  // Whole-study download: re-read the study's every (radiology) series from the slim index. Only
  // available when we came in by StudyInstanceUID; the direct-S3 path downloads the loaded pair.
  const listAllStudySeries = source.st
    ? async (): Promise<SeriesRef[]> => {
      const rows = await readStudyRows(source.st, () => {});
      return rows.map((r) => ({
        prefix: String(r.crdc_series_uuid),
        bucket: String(r.aws_bucket),
        modality: String(r.Modality).toUpperCase(),
        seriesUID: String(r.SeriesInstanceUID),
        seriesDescription: r.SeriesDescription ? String(r.SeriesDescription) : undefined,
      }));
    }
    : undefined;

  const initialLayout = pendingIraInitialLayout;
  pendingIraInitialLayout = undefined;
  ira = mountIra({
    overlay: document.getElementById("viewer")!,
    bar: document.getElementById("ira-bar")!,
    grid: document.getElementById("grid")!,
    planes: PLANES,
    canvases: cv,
    cellOf: (name) => document.querySelector(`[data-cell="${name}"]`),
    slice: () => sc.slice,
    off01: (p) => off[p],
    setOff01: (p, v) => (off[p] = Math.max(0, Math.min(1, v))),
    offsetMm: (p) => offset01ToMm(p, off[p], sc.rasLo, sc.rasHi),
    spacing: (p) => sliceIx.spacing(p),
    step: (p, fwd) => (off[p] = sliceIx.wheel(p, off[p], fwd)),
    redraw: (p) => {
      if (p === "threeD") draw3d();
      else drawSlice(p);
      xhair?.redraw();
      syncLiveScene();
    },
    redrawAll: () => {
      drawAll();
      syncLiveScene();
    },
    initialLayout,
    onLayoutChange: (l: IraLayout) => setDentalObliqueVisible(l === "dental"),
    readViewportRgba: async (plane) => {
      const canvas = cv[plane];
      const w = canvas?.width || 0;
      const h = canvas?.height || 0;
      if (!w || !h) return null;
      let raw: Uint8Array;
      if (plane === "threeD") {
        sc.scene.setCamera(
          camera.position,
          camera.focalPoint,
          camera.viewUp,
          camera.viewAngle,
          w,
          h,
        );
        raw = await sc.scene.renderToRGBA(w, h);
      } else {
        sc.slice.setPlane(plane, off[plane]);
        raw = await sc.slice.renderToRGBA(w, h);
      }
      // Canvas preferred format is often bgra8unorm*; ImageData/PNG need RGBA.
      if (String(preferred).toLowerCase().startsWith("bgra")) {
        for (let i = 0; i + 3 < raw.length; i += 4) {
          const b = raw[i];
          raw[i] = raw[i + 2];
          raw[i + 2] = b;
        }
      }
      return { rgba: raw, width: w, height: h };
    },
    rasLo: sc.rasLo,
    rasHi: sc.rasHi,
    wl: { get: () => [wl.win, wl.lev], set: setWL, auto: [sc.win, sc.lev] },
    presetsEl: presets,
    resetViews: () => sliceControls.forEach((c) => c.resetView()),
    close: () => status("This is the SlicerLive Basic Image Review demo — reload to restart."),
    jumpAll,
    modality: ct.modality || "",
    vanishingStore,
    onVanishingStroke: (op) => {
      void publishIraSceneUpdate([op as Record<string, unknown>]);
    },
    vanishingAuthor: () => {
      const info = getIraCastSessionInfo();
      return info.userName || info.subscriberName || undefined;
    },
    vanishing3d: {
      canvas: () => cv.threeD,
      pick: (u, v) => sc.scene.pick(u, v),
      worldToDisplay: (ras, w, h) => camera.worldToDisplay(ras, w, h),
      displayToWorldAtDepth: (x, y, depth, w, h) =>
        camera.displayToWorldAtDepth(x, y, depth, w, h) as Vec3,
    },
    onToolChange: (tool) => syncIraVanishingBrushBtn(tool),
    extraTools: [
      {
        id: "idc-share",
        icon: "share",
        title: "Share — copy a link that reopens this study here (or in the IDC OHIF portal)",
        run: () => shareStudy(source),
      },
      {
        id: "idc-download",
        icon: "download",
        title: "Download this study's DICOM to a local folder (streamed, parallel)",
        run: () => downloadStudyWithDialog(source, listAllStudySeries),
      },
      {
        id: "seg-stl",
        icon: "stl",
        title: "Save segment as STL for 3D printing (LPS)",
        run: () => {
          void (async () => {
            const labmap = sc.labelmap();
            if (!sc.hasSeg || !labmap || !sc.segments.length) {
              status("No segmentation loaded — open a study with SEG to export STL.");
              return;
            }
            const picked = await pickSegment(segmentsForUi());
            if (!picked) return;
            try {
              status(`Meshing ${picked.name}…`);
              const buf = segmentToStl(labmap.lab, sc.dims, sc.ijkToRAS, picked.num);
              downloadStl(buf, safeStlFileName(picked.name));
              status(`Saved ${safeStlFileName(picked.name)}`);
            } catch (err) {
              status("STL export failed: " + ((err as Error)?.message ?? err), true);
            }
          })();
        },
      },
      {
        id: "load-local",
        icon: "loadStudy",
        title: "Load study — open a local DICOM folder or ZIP",
        run: () => openLoadLocalStudyDialog(),
      },
    ],
  });
  vanishingBrushIra = ira;
  syncIraVanishingBrushBtn(ira.tool());
  syncIraSnapshotBtn(true);
  setIraPngFullSizeCaptureHandler(() => ira?.capturePngFullSize() ?? null);
  (globalThis as Record<string, unknown>).__iraDbg = {
    ready: () => !!ira,
    dims: () => sc.dims,
    tool: () => ira?.tool(),
    cellOrder: () => [...document.querySelectorAll("#grid .cell")].map((c) => (c as HTMLElement).dataset.cell),
    // test hooks
    setRoi: (en: boolean, vis: boolean) => {
      roiEnabled = en;
      roiVisible = vis;
      if (en) roiFirstEnable = false;
      sc.setRoiEnabled(en);
      sc.setRoiVisible(vis);
      draw3d();
    },
    roiState: () => ({ visible: sc.roiVisible(), handles: sc.roi.handleList().length }),
    setSegOpacity: (o: number) => {
      sc.setSegOpacity(o);
      sc.slice.setOverlayOpacity(o * SEG_FILL);
      drawAll();
    },
  };

  for (const p of PLANES) {
    const hooks = ira.hooks(p);
    // Oblique + Scroll tool: left-drag tilts the plane (wheel still scrolls). Other tools unchanged.
    let obliqueTilt: { u: number; v: number; w: number; h: number } | null = null;
    const obliqueHooks = p === "oblique"
      ? {
          onLeftGrab: (u: number, v: number, w: number, h: number) => {
            if (hooks.onLeftGrab(u, v, w, h)) return true;
            if (ira!.leftMode() !== "scroll") return false;
            obliqueTilt = { u, v, w, h };
            cv.oblique.style.cursor = "move";
            return true;
          },
          onLeftDrag: (u: number, v: number, w: number, h: number) => {
            if (obliqueTilt) {
              const dx = (u - obliqueTilt.u) * obliqueTilt.w;
              const dy = (v - obliqueTilt.v) * obliqueTilt.h;
              obliqueTilt = { u, v, w, h };
              tiltObliqueByPx(dx, dy);
              return;
            }
            hooks.onLeftDrag?.(u, v, w, h);
          },
          onLeftDrop: (movedPx: number) => {
            if (obliqueTilt) {
              obliqueTilt = null;
              cv.oblique.style.cursor = "";
              return;
            }
            hooks.onLeftDrop?.(movedPx);
          },
          onDoubleClick: () => {
            viewGrid!.toggleMax(p);
            maximizedCell = viewGrid!.maxCell();
            syncLiveScene();
            return true;
          },
        }
      : {
          onLeftGrab: hooks.onLeftGrab,
          onLeftDrag: hooks.onLeftDrag,
          onLeftDrop: hooks.onLeftDrop,
          onDoubleClick: () => {
            viewGrid!.toggleMax(p);
            maximizedCell = viewGrid!.maxCell();
            syncLiveScene();
            return true;
          },
        };
    sliceControls.push(attachSliceControls(cv[p], {
      orient: p,
      getSlice: () => sc.slice,
      step: (fwd) => (off[p] = sliceIx.wheel(p, off[p], fwd)),
      redraw: () => {
        drawSlice(p);
        xhair?.redraw();
        syncLiveScene();
      },
      wl: {
        enabled: () => ira!.tool() === "wl",
        get: () => [wl.win, wl.lev],
        set: setWL,
        range: dataRange,
        reset: () => setWL(sc.win, sc.lev),
      },
      leftMode: () => ira!.leftMode(),
      hooks: obliqueHooks,
    }));
  }
  attachCameraControls(cv.threeD, camera, {
    onChange: () => {
      draw3d();
      xhair?.redraw();
      syncLiveScene();
    },
  });

  // 3D ROI-crop widget: grab a face/corner/centre handle to resize/move the crop box; the
  // cursor changes over a handle while the box is visible. Capture-phase, so grabbing a
  // handle doesn't reach the camera; empty space bubbles through to orbit.
  let roiBox0: Box | null = null;
  attachWidgetControls(cv.threeD, camera, {
    getHandles: () =>
      sc.roiVisible()
        ? sc.roi.handleList().map((h) => ({ id: h.id, world: h.world, data: h.data, cursor: h.cursor }))
        : [],
    getSize: () => ({ w: cv.threeD.width, h: cv.threeD.height }),
    onDragStart: () => {
      roiBox0 = sc.roi.snapshot();
    },
    onDrag: (h, world) => {
      if (!roiBox0) return;
      const d: Vec3 = [world[0] - h.world[0], world[1] - h.world[1], world[2] - h.world[2]];
      sc.roi.applyDrag(h.data as HandleMeta, roiBox0, d);
      sc.reclip(); // re-crop + upload box/handle/clip in one syncUniforms
      draw3d();
      xhair?.redraw();
    },
    onHover: (h) => {
      sc.roi.setHover(h ? h.id : null);
      sc.scene.syncUniforms();
      draw3d();
    },
    onChange: () => draw3d(),
  });
  // After ROI widget so handle grabs (preventDefault) beat Marker ink.
  ira.attachVanishing3dInput();

  xhair = mountCrosshair({
    cells: {
      axial: cv.axial,
      coronal: cv.coronal,
      sagittal: cv.sagittal,
      threeD: cv.threeD,
      oblique: cv.oblique,
    },
    getScene: () => sc.scene,
    getSlice: () => sc.slice,
    getCamera: () => camera,
    getOffset: (o) => off[o as Plane],
    onJump: jumpAll,
  });

  // ---- 3D volume-rendering preset + shift (Slicer/OHIF CT presets) ------------------------
  // vrPreset = active preset id (null = grayscale W/L VR). vrShift = Slicer's VR "Shift" (offset
  // the whole transfer function along the scalar axis). Shift range derives from the actual data
  // range (sampled), like Slicer's shift slider spanning the scalar range.
  const shiftRange = Math.max(200, (vmax - vmin) / 2);
  // Default bake: scene VolumeProperty when loaded from ?scene=, else grayscale W/L (null).
  const defaultVrBake = sceneVr;
  const bakeOf = (name: string | null) => {
    if (!name) return defaultVrBake;
    const p = CT_VR_PRESETS.find((x) => x.name === name);
    if (!p) {
      console.warn(`[ira] unknown VR preset: ${name}`);
      return defaultVrBake;
    }
    return presetLUT(p);
  };
  const presetLabel = () => vrPreset ? (CT_VR_PRESETS.find((p) => p.name === vrPreset)?.label ?? vrPreset) : (defaultVrBake ? "Scene TF" : "Default (W/L)");
  applyVrPresetFn = (name: string | null) => {
    vrPreset = name;
    sc.setVolumePreset(bakeOf(name));
    if (name && sc.volumeOpacity() <= 0) sc.setVolumeOpacity(1); // reveal the VR so the preset is visible
    draw3d();
    xhair?.redraw();
    syncLiveScene();
  };
  const applyVrPreset = (name: string | null) => applyVrPresetFn(name);
  const applyVrShift = (hu: number) => {
    vrShift = hu;
    sc.setVolumeShift(hu);
    draw3d();
    xhair?.redraw();
    syncLiveScene();
  };

  // On-the-fly thumbnails: render THIS volume at the CURRENT camera with each preset into a small
  // WebGPU canvas. Full opacity, no shift, segmentation hidden for a clean preview; live state is
  // restored afterward.
  const renderPresetThumbnails = (): VrPresetItem[] => {
    const THUMB = 116;
    const savedOp = sc.volumeOpacity(), savedShift = sc.volumeShift(), savedSeg = sc.segOpacity();
    sc.setVolumeShift(0);
    if (savedSeg > 0) sc.setSegOpacity(0);
    if (savedOp < 1) sc.setVolumeOpacity(1);
    const entries: { name: string | null; label: string }[] = [
      { name: null, label: defaultVrBake ? "Scene TF" : "Default (W/L)" },
      ...CT_VR_PRESETS.map((p) => ({ name: p.name, label: p.label })),
    ];
    const items: VrPresetItem[] = [];
    for (const e of entries) {
      const c = document.createElement("canvas");
      c.width = THUMB; c.height = THUMB;
      const cxt = c.getContext("webgpu") as GPUCanvasContext;
      cxt.configure({ device: gpu.device, format: preferred, viewFormats: [srgb], alphaMode: "opaque" });
      sc.setVolumePreset(bakeOf(e.name));
      sc.scene.setCamera(camera.position, camera.focalPoint, camera.viewUp, camera.viewAngle, THUMB, THUMB);
      sc.scene.renderToView(cxt.getCurrentTexture().createView({ format: srgb }), THUMB, THUMB);
      items.push({ name: e.name, label: e.label, canvas: c });
    }
    // restore live VR state
    sc.setVolumePreset(bakeOf(vrPreset));
    sc.setVolumeShift(savedShift);
    if (savedSeg > 0) sc.setSegOpacity(savedSeg);
    if (savedOp < 1) sc.setVolumeOpacity(savedOp);
    draw3d();
    return items;
  };
  // SlicerLive badge popup: 3D volume-rendering + segmentation controls (NON-IRA), anchored
  // to the 3D cell — the SlicerLive-native version of the reader's "Live" controls.
  /** Live organ-group opacities while Colorize is on (name → 0..1). */
  let colorizeGroupOp = new Map<string, number>();
  let resolvedColorizeGroups: ColorizeGroupResolved[] = [];

  applyColorizeGroupDefaults = () => {
    resolvedColorizeGroups = resolveColorizeGroups(sc.segments);
    colorizeGroupOp = new Map(resolvedColorizeGroups.map((g) => [g.name, g.opacity]));
    const entries = colorizeDefaultOpacityEntries(sc.segments);
    if (entries.length) sc.setSegmentOpacities(entries);
  };

  const clearColorizeGroupState = () => {
    colorizeGroupOp.clear();
    resolvedColorizeGroups = [];
  };

  const restoreSurfaceSegmentOpacities = () => {
    sc.setSegmentOpacities(sc.segments.map((s) => [s.num, 1]));
  };

  // Colorize forces MPR contours on (fill at group opacity + full-colour outline); this is the
  // user's outline choice to restore when Colorize turns off.
  let outlineBeforeColorize: boolean | null = null;

  const setColorizeMode = (on: boolean) => {
    if (on) {
      if (!sc.hasSeg) return;
      sc.setSegDisplayMode("colorize");
      applyColorizeGroupDefaults();
      if (outlineBeforeColorize === null) outlineBeforeColorize = sliceOutline;
      sliceOutline = true;
      sc.slice.setOverlayOutline(true);
    } else {
      sc.setSegDisplayMode("surface");
      restoreSurfaceSegmentOpacities();
      clearColorizeGroupState();
      if (outlineBeforeColorize !== null) {
        sliceOutline = outlineBeforeColorize;
        outlineBeforeColorize = null;
        sc.slice.setOverlayOutline(sliceOutline);
      }
    }
    syncLiveScene();
    drawAll();
    xhair?.redraw();
    reinstallChrome();
  };

  Object.assign((globalThis as Record<string, unknown>).__iraDbg as object, {
    presets: () => CT_VR_PRESETS.map((p) => p.name),
    setPreset: (name: string | null) => applyVrPreset(name),
    setShift: (hu: number) => applyVrShift(hu),
    vrState: () => ({ preset: vrPreset, shift: vrShift, shiftRange, dataRange: [vmin, vmax] }),
    thumbCount: () => renderPresetThumbnails().length,
    colorize: () => sc.segDisplayMode() === "colorize",
    setColorize: (on: boolean) => setColorizeMode(!!on),
  });

  const buildVizControls = (): VizControl[] => {
    const controls: VizControl[] = [
      {
        label: "Volume render",
        getOpacity: () => sc.volumeOpacity(),
        setOpacity: (o) => {
          sc.setVolumeOpacity(o);
          draw3d();
          xhair?.redraw();
          syncLiveScene();
        },
        color: [0.75, 0.78, 0.85],
      },
      // Slicer's VR "Shift" — a plain slider directly below the opacity control.
      {
        label: "Shift",
        slider: {
          min: -shiftRange,
          max: shiftRange,
          step: Math.max(1, Math.round(shiftRange / 200)),
          get: () => vrShift,
          set: (v) => applyVrShift(v),
          format: (v) => (v > 0 ? "+" : "") + Math.round(v),
        },
        disabled: () => sc.volumeOpacity() <= 0,
      },
      // Volume-rendering preset — opens the on-the-fly thumbnail menu (Slicer/OHIF CT presets).
      {
        label: "Preset",
        button: { text: () => presetLabel(), run: () => openVrPresetMenu({ items: renderPresetThumbnails(), current: vrPreset, onPick: applyVrPreset }) },
      },
      {
        label: "Segmentation",
        getOpacity: () => sc.segOpacity(),
        setOpacity: (o) => {
          sc.setSegOpacity(o); // 3D surface global opacity
          sc.slice.setOverlayOpacity(o * SEG_FILL); // AND the 2D MPR overlay (fill/outline)
          syncLiveScene();
          drawAll();
        },
        disabled: () => !sc.hasSeg,
        color: [0.62, 0.9, 1.0],
      },
      // Colorize: CT+labelmap fusion + organ-group opacities (off = SDF Surface).
      {
        label: "Colorize",
        get: () => sc.segDisplayMode() === "colorize",
        set: (on) => setColorizeMode(on),
        disabled: () => !sc.hasSeg,
      },
      {
        label: "Segment labels",
        get: () => labelsOn,
        set: (on) => {
          labelsOn = on;
          syncLiveScene();
          draw3d();
        },
        disabled: () => !sc.hasSeg,
      },
      // Segmentation overlay: fill (default) vs outline on the MPR planes.
      {
        label: "Segmentation outline",
        get: () => sliceOutline,
        set: (on) => {
          sliceOutline = on;
          sc.slice.setOverlayOutline(on);
          for (const p of PLANES) drawSlice(p);
          xhair?.redraw();
        },
        disabled: () => !sc.hasSeg,
      },
      // 3D volume-render crop: enabling it reveals the ROI box the first time (Slicer behaviour).
      {
        label: "Crop volume (3D)",
        get: () => roiEnabled,
        set: (on) => {
          roiEnabled = on;
          if (on && roiFirstEnable) {
            roiVisible = true;
            roiFirstEnable = false;
          }
          sc.setRoiEnabled(roiEnabled);
          sc.setRoiVisible(roiVisible);
          syncLiveScene();
          draw3d();
          xhair?.redraw();
        },
        disabled: () => sc.volumeOpacity() <= 0,
      },
      {
        label: "Show ROI box",
        get: () => roiVisible,
        set: (on) => {
          roiVisible = on;
          sc.setRoiVisible(on);
          syncLiveScene();
          draw3d();
          xhair?.redraw();
        },
      },
      // DICOM corner annotations (IHE IRA) — the "official radiology look".
      {
        label: "Corner annotations",
        get: () => annOn,
        set: (on) => {
          annOn = on;
          applyAnn();
        },
      },
    ];

    if (sc.segDisplayMode() === "colorize" && resolvedColorizeGroups.length) {
      for (const g of resolvedColorizeGroups) {
        const first = sc.segments.find((s) => s.num === g.segments[0]);
        controls.push({
          label: g.name,
          section: "Organ groups",
          getOpacity: () => colorizeGroupOp.get(g.name) ?? g.opacity,
          setOpacity: (o) => {
            colorizeGroupOp.set(g.name, o);
            sc.setSegmentOpacities(g.segments.map((n) => [n, o]));
            syncLiveScene();
            drawAll();
          },
          color: first?.color ?? [0.62, 0.9, 1.0],
        });
      }
    }
    return controls;
  };

  reinstallChrome = () => {
    try { chromeApi?.destroy(); } catch { /* ignore */ }
    chromeApi = installChrome({
      help: false,
      controls: buildVizControls(),
      anchor: cv.threeD.parentElement ?? undefined,
      segments: {
        list: () => segmentsForUi().map((s) => ({ num: s.num, name: s.name, color: s.color })),
        get: (num) => sc.segmentOpacity(num),
        set: (num, o) => {
          sc.setSegmentOpacity(num, o);
          syncLiveScene();
          drawAll();
        },
        enabled: () => sc.hasSeg,
      },
    });
  };
  reinstallChrome();
  const chrome = chromeApi!;

  buildAnnotations();
  applyAnn();
  resize();
  if (sceneStreamDone) {
    status(idcStreamVol ? "streaming volume…" : "streaming scene…");
    await sceneStreamDone;
    if (signal?.aborted) {
      return {
        dispose: () => {
          studyDisposed = true;
          vanishingStore.clear();
          if (vanishingBrushIra === ira) {
            vanishingBrushIra = null;
            syncIraVanishingBrushBtn(null);
            syncIraSnapshotBtn(false);
          }
          setIraSegCatalog(EMPTY_SEG_CATALOG);
          labelsOn = false;
          try {
            segCards.clear();
          } catch {
            /* ignore */
          }
          try {
            ira?.detachKeys();
          } catch {
            /* ignore */
          }
          try {
            chrome?.destroy();
          } catch {
            /* ignore */
          }
          try {
            sc.destroy();
          } catch {
            /* ignore */
          }
          resetIraToolbarShell();
          for (const el of document.querySelectorAll(".ann, .ira-overlay")) {
            el.remove();
          }
        },
      };
    }
    if (sceneSegDone) {
      status("loading segmentation…");
      await sceneSegDone;
      if (signal?.aborted) {
        return {
          dispose: () => {
            studyDisposed = true;
            vanishingStore.clear();
            if (vanishingBrushIra === ira) {
              vanishingBrushIra = null;
              syncIraVanishingBrushBtn(null);
              syncIraSnapshotBtn(false);
            }
            setIraSegCatalog(EMPTY_SEG_CATALOG);
            labelsOn = false;
            try {
              segCards.clear();
            } catch {
              /* ignore */
            }
            try {
              ira?.detachKeys();
            } catch {
              /* ignore */
            }
            try {
              chrome?.destroy();
            } catch {
              /* ignore */
            }
            try {
              sc.destroy();
            } catch {
              /* ignore */
            }
            resetIraToolbarShell();
            for (const el of document.querySelectorAll(".ann, .ira-overlay")) {
              el.remove();
            }
          },
        };
      }
    }
    sampleRange();
    if (pendingIdcSeg) {
      seg = pendingIdcSeg;
      sc.attachSegmentation(pendingIdcSeg);
      refreshSegChrome(pendingIdcSeg);
    }
    // Apply refined W/L from final CT (MR/PET) if the worker updated it after fill.
    if (idcStreamVol && Number.isFinite(ct.win) && Number.isFinite(ct.lev)) {
      setWL(ct.win, ct.lev);
    }
    drawAll();
  }
  status("");

  return {
    dispose: () => {
      studyDisposed = true;
      vanishingStore.clear();
      dentalLayout = false;
      obliqueIn3D = null;
      if (vanishingBrushIra === ira) {
        vanishingBrushIra = null;
        syncIraVanishingBrushBtn(null);
        syncIraSnapshotBtn(false);
      }
      try {
        sc.setExtraFields([]);
      } catch {
        /* ignore */
      }
      if (thisCastMountId === castNiftiMountId) {
        applyCastNiftiSeg = null;
        applyCastDicomSeg = null;
      }
      setIraSegCatalog(EMPTY_SEG_CATALOG);
      labelsOn = false;
      setIraSegLocaleRefreshHandler(null);
      setIraSceneUpdateHandler(null);
      setIraConferenceHostHandler(null);
      setIraPngFullSizeCaptureHandler(null);
      globalThis.removeEventListener("resize", resize);
      for (const ro of resizeObservers) ro.disconnect();
      try {
        segCards.clear();
      } catch {
        /* ignore */
      }
      try {
        ira?.detachKeys();
      } catch {
        /* ignore */
      }
      try {
        chrome.destroy();
      } catch {
        /* ignore */
      }
      try {
        sc.destroy();
      } catch {
        /* ignore */
      }
      try {
        sceneLodField?.destroy();
      } catch {
        /* ignore */
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
          const enc = gpu.device.createCommandEncoder();
          const pass = enc.beginRenderPass({
            colorAttachments: [
              {
                view: tex.createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: "clear",
                storeOp: "store",
              },
            ],
          });
          pass.end();
          gpu.device.queue.submit([enc.finish()]);
        } catch {
          /* ignore */
        }
      }
    },
  };
}

main().catch((e) => status("error: " + ((e as Error)?.message ?? e), true));
