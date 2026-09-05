/**
 * Map Cast ImagingStudy-open (open-mode files, SlicerLive scene JSON URLs) → IRA ?scene=.
 */
import { resolveImagingStudyOpenPlan } from "../../../../../cast-interface/cast-js-client/dist/cast-client.js";
import type { SceneVolume } from "../../scene-volume.ts";
import type { CTVolume } from "../../vendor/idc_tools/types.js";

type PlanFile = {
  url?: string;
  fileName?: string;
  role?: string;
  label?: string;
};

function looksLikeSceneJson(f: PlanFile): boolean {
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
 * Convert ImagingStudy-open context to a SlicerLive scene JSON URL, or null.
 * Expects open-mode `files` with at least one `.json` entry.
 */
export function sceneUrlFromOpenContext(context: unknown[]): string | null {
  const plan = resolveImagingStudyOpenPlan(context) as {
    mode?: string;
    files?: PlanFile[];
  } | null;
  if (!plan || plan.mode !== "files") return null;
  const files = Array.isArray(plan.files) ? plan.files : [];
  const sceneFile = files.find(looksLikeSceneJson) || files[0];
  const url = String(sceneFile?.url || "").trim();
  if (!url) return null;
  if (!looksLikeSceneJson(sceneFile || {})) return null;
  return url;
}

/** Map scene URL → IRA query params for a full page load. */
export function studyParamsFromSceneUrl(sceneUrl: string): Record<string, string> {
  return { scene: String(sceneUrl || "").trim() };
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
