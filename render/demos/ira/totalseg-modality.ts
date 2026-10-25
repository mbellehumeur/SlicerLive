/**
 * Resolve open-study modality for TotalSeg task filtering (CT vs MR).
 */
import {
  getIraImagingStudyOpenContext,
  getIraLiveScene,
} from "./ira-hub.ts";
import { modalityFromSceneName } from "./ira-scene.ts";
import { mhubContextHintsFromImagingStudy } from "./mhub-models.ts";
import type { TotalSegModality } from "./totalseg-tasks.ts";

export type OpenStudyTotalSegModality =
  | { modality: TotalSegModality; source: string; assumed?: boolean }
  | { modality: null; source: string; reason: string };

export type ImagingModality = "CT" | "MR" | "PT";

export type OpenStudyImagingModality =
  | { modality: ImagingModality; source: string }
  | { modality: null; source: string; reason?: string };

function codingModality(value: unknown): string {
  if (!value) return "";
  if (typeof value === "string") return value.trim().toUpperCase();
  if (Array.isArray(value)) {
    for (const entry of value) {
      const m = codingModality(entry);
      if (m) return m;
    }
    return "";
  }
  if (typeof value === "object") {
    const obj = value as {
      code?: unknown;
      coding?: unknown;
      text?: unknown;
    };
    if (obj.coding != null) return codingModality(obj.coding);
    if (obj.code != null) return String(obj.code).trim().toUpperCase();
    if (obj.text != null) return String(obj.text).trim().toUpperCase();
  }
  return "";
}

function normalizeTotalSegModality(raw: string): TotalSegModality | "other" | "" {
  const m = raw.trim().toUpperCase();
  if (!m) return "";
  if (m === "CT" || m === "CTAC") return "CT";
  if (m === "MR" || m === "MRI") return "MR";
  if (m === "PT" || m === "PET" || m === "NM") return "other";
  return "other";
}

/** Normalize to CT / MR / PT for MHub filtering (null when unknown/unsupported). */
function normalizeImagingModality(raw: string): ImagingModality | "other" | "" {
  const m = raw.trim().toUpperCase();
  if (!m) return "";
  if (m === "CT" || m === "CTAC") return "CT";
  if (m === "MR" || m === "MRI") return "MR";
  if (m === "PT" || m === "PET" || m === "NM") return "PT";
  return "other";
}

function modalityFromContext(context: unknown[]): string {
  // Prefer shared MHubSkill-style hints (study.modality + IDC volume file labels).
  const hints = mhubContextHintsFromImagingStudy(context);
  if (hints.modality) return hints.modality;
  for (const item of context) {
    if (!item || typeof item !== "object") continue;
    const resource = (item as { resource?: unknown }).resource;
    if (!resource || typeof resource !== "object") continue;
    const res = resource as { modality?: unknown; series?: unknown };
    const fromStudy = codingModality(res.modality);
    if (fromStudy) return fromStudy;
    if (Array.isArray(res.series)) {
      for (const series of res.series) {
        if (!series || typeof series !== "object") continue;
        const fromSeries = codingModality(
          (series as { modality?: unknown }).modality,
        );
        if (fromSeries) return fromSeries;
      }
    }
  }
  return "";
}

function modalityFromUrl(): string {
  try {
    const p = new URLSearchParams(globalThis.location.search);
    return String(p.get("modality") || "").trim().toUpperCase();
  } catch {
    return "";
  }
}

function modalityFromLiveScene(): string {
  const scene = getIraLiveScene();
  if (!scene?.nodes || typeof scene.nodes !== "object") return "";
  for (const node of Object.values(scene.nodes)) {
    if (!node || typeof node !== "object") continue;
    const name = String((node as { name?: unknown }).name || "");
    if (!name) continue;
    return modalityFromSceneName(name);
  }
  return "";
}

function rawOpenStudyModality(): { raw: string; source: string } {
  const ctx = getIraImagingStudyOpenContext();
  const fromCtx = modalityFromContext(ctx);
  if (fromCtx) return { raw: fromCtx, source: "ImagingStudy context" };
  const fromUrl = modalityFromUrl();
  if (fromUrl) return { raw: fromUrl, source: "URL modality" };
  const fromScene = modalityFromLiveScene();
  if (fromScene) return { raw: fromScene, source: "scene volume name" };
  return { raw: "", source: "none" };
}

/**
 * Best-effort CT / MR / PT for MHub model filtering.
 * Returns modality null when unknown (caller shows all models).
 */
export function resolveOpenStudyImagingModality(): OpenStudyImagingModality {
  const { raw, source } = rawOpenStudyModality();
  const normalized = normalizeImagingModality(raw);
  if (normalized === "CT" || normalized === "MR" || normalized === "PT") {
    return { modality: normalized, source };
  }
  if (normalized === "other") {
    return {
      modality: null,
      source,
      reason: `Open study modality ${raw} is not CT, MR, or PT`,
    };
  }
  return { modality: null, source: "none", reason: "Unknown study modality" };
}

/**
 * Best-effort CT/MR for TotalSeg. Falls back to CT with ``assumed`` when unknown.
 * Returns modality null only for clearly unsupported modalities (e.g. PET).
 */
export function resolveOpenStudyTotalSegModality(): OpenStudyTotalSegModality {
  const ctx = getIraImagingStudyOpenContext();
  const fromCtx = normalizeTotalSegModality(modalityFromContext(ctx));
  if (fromCtx === "CT" || fromCtx === "MR") {
    return { modality: fromCtx, source: "ImagingStudy context" };
  }
  if (fromCtx === "other") {
    return {
      modality: null,
      source: "ImagingStudy context",
      reason: "Open study modality is not CT or MR",
    };
  }

  const fromUrl = normalizeTotalSegModality(modalityFromUrl());
  if (fromUrl === "CT" || fromUrl === "MR") {
    return { modality: fromUrl, source: "URL modality" };
  }
  if (fromUrl === "other") {
    return {
      modality: null,
      source: "URL modality",
      reason: "Open study modality is not CT or MR",
    };
  }

  const fromScene = normalizeTotalSegModality(modalityFromLiveScene());
  if (fromScene === "CT" || fromScene === "MR") {
    return { modality: fromScene, source: "scene volume name" };
  }

  return {
    modality: "CT",
    source: "default",
    assumed: true,
  };
}
