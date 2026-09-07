/**
 * Map Cast ImagingStudy-open (open-mode files, LiveScene JSON URLs) → IRA ?scene=.
 */
import {
  buildFilesImagingStudyOpenContext,
  resolveImagingStudyOpenPlan,
} from "../../../../../cast-interface/cast-js-client/dist/cast-client.js";
import type { SceneVolume } from "../../scene-volume.ts";
import type { CTVolume } from "../../vendor/idc_tools/types.js";
import type { IraLiveSceneDoc } from "./ira-cast.ts";

const LIVESCENE_MIME = "application/vnd.slicerlive.scene+json";

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

/**
 * Convert ImagingStudy-open context to a LiveScene JSON URL, or null.
 * Expects open-mode `files` with at least one `.json` / role=scene entry.
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

/** Guess modality from a scene / volume name. */
export function modalityFromSceneName(name: string): "CT" | "MR" {
  const n = String(name || "").toUpperCase();
  if (/\bMR\b/.test(n) || n.includes("MRHEAD") || n.startsWith("MR")) return "MR";
  return "CT";
}

/** Adapt a loaded SlicerLive SceneVolume into the CTVolume shape IRA mounts. */
export function ctVolumeFromSceneVolume(sv: SceneVolume): CTVolume {
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
