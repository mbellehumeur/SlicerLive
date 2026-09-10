/**
 * Map Cast ImagingStudy-open (embedded LiveScene or scene JSON URLs) → IRA load.
 */
import {
  buildFilesImagingStudyOpenContext,
  resolveImagingStudyOpenPlan,
} from "../../../../../cast-interface/cast-js-client/dist/cast-client.js";
import type { SceneVolume } from "../../scene-volume.ts";
import type { CTVolume } from "../../vendor/idc_tools/types.js";
import type { Volume } from "../../../logic/readers/nifti.ts";
import { autoWindowLevel } from "../../../logic/window-level.ts";
import type { IraLiveSceneDoc } from "./ira-cast.ts";

const LIVESCENE_MIME = "application/vnd.slicerlive.scene+json";

/** Query value for ?scene= when the LiveScene body is in sessionStorage. */
export const EMBEDDED_SCENE_PARAM = "embedded";
const EMBEDDED_SCENE_STORAGE_KEY = "ira.embeddedLiveScene";

type PlanFile = {
  url?: string;
  fileName?: string;
  role?: string;
  label?: string;
  mimeType?: string;
};

function looksLikeSceneJson(f: PlanFile): boolean {
  const role = String(f.role || "").toLowerCase();
  if (role === "scene") return true;
  const name = String(f.fileName || "").toLowerCase();
  const url = String(f.url || "").toLowerCase();
  if (name.endsWith(".json")) return true;
  try {
    const path = new URL(url).pathname.toLowerCase();
    return path.endsWith(".json");
  } catch {
    return url.includes(".json");
  }
}

function isLiveSceneDoc(value: unknown): value is IraLiveSceneDoc {
  if (!value || typeof value !== "object") return false;
  const doc = value as IraLiveSceneDoc;
  const nodes = doc.nodes;
  const blobBase = String(doc.blobBase || "").trim();
  return Boolean(nodes && typeof nodes === "object" && blobBase);
}

/**
 * Prefer ImagingStudy-open `{ key: "scene", resource }` (embedded LiveScene).
 */
export function sceneDocFromOpenContext(
  context: unknown[]
): IraLiveSceneDoc | null {
  if (!Array.isArray(context)) return null;
  for (const item of context) {
    if (!item || typeof item !== "object") continue;
    const row = item as { key?: unknown; resource?: unknown };
    if (String(row.key || "").toLowerCase() !== "scene") continue;
    if (isLiveSceneDoc(row.resource)) {
      return { ...(row.resource as IraLiveSceneDoc) };
    }
  }
  return null;
}

/**
 * Convert ImagingStudy-open context to a LiveScene JSON URL, or null.
 * Expects open-mode `files` with at least one `.json` / role=scene entry.
 * Prefer {@link sceneDocFromOpenContext} when the scene body is embedded.
 */
export function sceneUrlFromOpenContext(context: unknown[]): string | null {
  const plan = resolveImagingStudyOpenPlan(context) as {
    mode?: string;
    files?: PlanFile[];
  } | null;
  if (!plan || plan.mode !== "files") return null;
  const files = Array.isArray(plan.files) ? plan.files : [];
  const sceneFile =
    files.find((f) => String(f.role || "").toLowerCase() === "scene") ||
    files.find(looksLikeSceneJson) ||
    files[0];
  const url = String(sceneFile?.url || "").trim();
  if (!url) return null;
  if (!looksLikeSceneJson(sceneFile || {})) return null;
  return url;
}

/** Map scene URL → IRA query params for a full page load. */
export function studyParamsFromSceneUrl(sceneUrl: string): Record<string, string> {
  return { scene: String(sceneUrl || "").trim() };
}

/** Prefer provenance URL for shareable links; else stash body for `?scene=embedded`. */
export function studyParamsFromSceneDoc(
  doc: IraLiveSceneDoc
): Record<string, string> {
  const url = String(doc.url || "").trim();
  if (url) return { scene: url };
  try {
    globalThis.sessionStorage?.setItem(
      EMBEDDED_SCENE_STORAGE_KEY,
      JSON.stringify(doc)
    );
  } catch (err) {
    console.warn("[ira] failed to stash embedded LiveScene", err);
  }
  return { scene: EMBEDDED_SCENE_PARAM };
}

/** Read stashed embedded LiveScene (for `?scene=embedded`). */
export function takeEmbeddedSceneFromStorage(): IraLiveSceneDoc | null {
  try {
    const raw = globalThis.sessionStorage?.getItem(EMBEDDED_SCENE_STORAGE_KEY);
    if (!raw) return null;
    globalThis.sessionStorage?.removeItem(EMBEDDED_SCENE_STORAGE_KEY);
    const data = JSON.parse(raw) as unknown;
    return isLiveSceneDoc(data) ? (data as IraLiveSceneDoc) : null;
  } catch {
    return null;
  }
}

/** Synthetic ImagingStudy-open context for a bare `?scene=` load (STATUS catch-up). */
export function imagingStudyContextFromSceneUrl(sceneUrl: string): unknown[] {
  const url = String(sceneUrl || "").trim();
  if (!url) return [];
  let fileName = "scene.json";
  try {
    const path = new URL(url).pathname;
    const base = path.split("/").pop();
    if (base) fileName = base;
  } catch {
    /* keep default */
  }
  return buildFilesImagingStudyOpenContext({
    id: `scene-${fileName.replace(/\.[^.]+$/, "") || "livescene"}`,
    files: [
      {
        url,
        fileName,
        role: "scene",
        label: "LiveScene",
        mimeType: LIVESCENE_MIME,
      },
    ],
  });
}

/** ImagingStudy-open context that includes an embedded LiveScene resource. */
export function imagingStudyContextFromSceneDoc(
  doc: IraLiveSceneDoc,
  idHint?: string
): unknown[] {
  const url = String(doc.url || "").trim();
  const filesCtx = url
    ? imagingStudyContextFromSceneUrl(url)
    : buildFilesImagingStudyOpenContext({
        id: String(idHint || "livescene").trim() || "livescene",
        files: [],
      });
  const withoutScene = filesCtx.filter((item) => {
    if (!item || typeof item !== "object") return true;
    return (item as { key?: string }).key !== "scene";
  });
  withoutScene.push({ key: "scene", resource: { ...doc } });
  return withoutScene;
}

/** Fetch LiveScene JSON (blobBase + nodes) for STATUS / Scene View. */
export async function fetchLiveSceneDoc(
  sceneUrl: string
): Promise<IraLiveSceneDoc | null> {
  const url = String(sceneUrl || "").trim();
  if (!url) return null;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch LiveScene (${res.status})`);
  }
  const data = (await res.json()) as Record<string, unknown>;
  const nodes =
    data.nodes && typeof data.nodes === "object"
      ? (data.nodes as Record<string, unknown>)
      : undefined;
  const doc: IraLiveSceneDoc = {
    ...data,
    url,
  };
  if (typeof data.blobBase === "string") doc.blobBase = data.blobBase;
  if (typeof data.mrson === "string") doc.mrson = data.mrson;
  if (nodes) doc.nodes = nodes;
  return doc;
}

/** Synthetic ImagingStudy-open context for a local folder/ZIP series (no remote files). */
export function imagingStudyContextFromLocalStudy(opts: {
  id?: string;
  label: string;
  modality?: string;
  studyInstanceUID?: string;
  seriesInstanceUID?: string;
  anonymized?: boolean;
}): unknown[] {
  const studyUid = String(opts.studyInstanceUID || "").trim();
  const seriesUid = String(opts.seriesInstanceUID || "").trim();
  const id =
    String(opts.id || "").trim() ||
    studyUid ||
    seriesUid ||
    `local-${Date.now()}`;
  const context = buildFilesImagingStudyOpenContext({
    id,
    files: [],
  });
  for (const item of context) {
    if (!item || typeof item !== "object") continue;
    if ((item as { key?: string }).key !== "study") continue;
    const resource = (item as { resource?: Record<string, unknown> }).resource;
    if (!resource || typeof resource !== "object") continue;
    resource.description = String(opts.label || "").trim() || "Local study";
    const modality = String(opts.modality || "").trim().toUpperCase();
    if (modality) {
      resource.modality = [{ coding: [{ system: "http://dicom.nema.org/resources/ontology/DCM", code: modality }] }];
    }
    if (studyUid) {
      const identifiers = Array.isArray(resource.identifier)
        ? [...(resource.identifier as unknown[])]
        : [];
      identifiers.push({
        system: "urn:dicom:uid",
        value: `urn:oid:${studyUid}`,
      });
      resource.identifier = identifiers;
    }
    if (opts.anonymized) {
      resource.meta = {
        ...(typeof resource.meta === "object" && resource.meta
          ? (resource.meta as object)
          : {}),
        security: [
          {
            system: "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            code: "ANONYED",
            display: "anonymized",
          },
        ],
      };
    }
  }
  return context;
}

/** Guess modality from a scene / volume name. */
export function modalityFromSceneName(name: string): "CT" | "MR" {
  const n = String(name || "").toUpperCase();
  if (/\bMR\b/.test(n) || n.includes("MRHEAD") || n.startsWith("MR")) return "MR";
  return "CT";
}

/** Adapt a loaded SlicerLive SceneVolume into the CTVolume shape IRA mounts. */
export function iraVolumeFromSceneVolume(sv: SceneVolume): CTVolume {
  return {
    vol: sv.voxels,
    dims: sv.dims,
    ijkToRAS: sv.ijkToRAS,
    win: sv.win,
    lev: sv.lev,
    dtype: "float32",
    modality: modalityFromSceneName(sv.name),
  };
}

/** Adapt a local DICOM/NIfTI Volume into the CTVolume shape IRA mounts. */
export function iraVolumeFromLocalVolume(vol: Volume): CTVolume {
  const meta = vol.meta || {};
  const winMeta = Number(meta.windowWidth);
  const levMeta = Number(meta.windowCenter);
  let win: number;
  let lev: number;
  if (Number.isFinite(winMeta) && Number.isFinite(levMeta) && winMeta > 0) {
    win = winMeta;
    lev = levMeta;
  } else {
    const wl = autoWindowLevel(vol.data);
    win = wl.window;
    lev = wl.level;
  }
  // Match IDC worker: always mount as float32 scalars.
  const data = new Float32Array(vol.data.length);
  data.set(vol.data as ArrayLike<number>);
  let modality = String(meta.modality || "CT").trim().toUpperCase();
  if (modality === "PET") modality = "PT";
  if (modality !== "CT" && modality !== "MR" && modality !== "PT") modality = "CT";
  return {
    vol: data,
    dims: vol.dims,
    ijkToRAS: vol.ijkToRAS.slice(),
    win,
    lev,
    dtype: "float32",
    modality,
  };
}

/** VR bake for setVolumePreset from scene transfer functions (when present). */
export function vrPresetFromSceneVolume(
  sv: SceneVolume
): { lut: Uint8Array; clim: [number, number]; shade: [number, number, number, number] } | null {
  if (!sv.vr) return null;
  return {
    lut: sv.vr.lut,
    clim: [sv.vr.clim[0], sv.vr.clim[1]],
    shade: [
      sv.vr.shade[0],
      sv.vr.shade[1],
      sv.vr.shade[2],
      sv.vr.shade[3],
    ],
  };
}
