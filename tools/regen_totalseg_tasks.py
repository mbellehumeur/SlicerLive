#!/usr/bin/env python3
"""Regenerate SlicerLive/render/demos/ira/totalseg-tasks.ts from totalseg_info --json.

Requires TotalSegmentator installed in the current Python:

    python -m totalsegmentator.bin.totalseg_info --json

Run from pw46 (or pass --out). Does not commit; review the diff before landing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

GROUP_MAP = {
    "total": "total",
    "total_mr": "total",
    "body": "body",
    "body_mr": "body",
    "vertebrae_mr": "spine",
    "vertebrae_body": "spine",
    "vertebrae_pp": "spine",
    "vertebrae_pp_refined": "spine",
    "lung_nodules": "lung",
    "lung_vessels": "lung",
    "liver_segments": "liver",
    "liver_segments_mr": "liver",
    "liver_vessels": "liver",
    "liver_lesions": "liver",
    "liver_lesions_mr": "liver",
    "kidney_cysts": "kidney",
    "breasts": "breast",
    "abdominal_muscles": "abdomen",
    "trunk_cavities": "abdomen",
    "head_glands_cavities": "head",
    "head_muscles": "head",
    "oculomotor_muscles": "head",
    "craniofacial_structures": "head",
    "teeth": "head",
    "headneck_bones_vessels": "headneck",
    "headneck_muscles": "headneck",
    "cerebral_bleed": "brain",
    "brain_aneurysm": "brain",
    "brain_structures": "brain",
    "ventricle_parts": "brain",
    "pleural_pericard_effusion": "heart",
    "coronary_arteries": "heart",
    "heartchambers_highres": "heart",
    "aortic_sinuses": "heart",
    "hip_implant": "bones",
    "appendicular_bones": "bones",
    "appendicular_bones_mr": "bones",
    "tissue_types": "tissue",
    "tissue_4_types": "tissue",
    "tissue_types_mr": "tissue",
    "face": "face",
    "face_mr": "face",
    "thigh_shoulder_muscles": "muscles",
    "thigh_shoulder_muscles_mr": "muscles",
}

GROUP_LABEL = {
    "total": "Total body",
    "body": "Body",
    "spine": "Spine",
    "lung": "Lung",
    "liver": "Liver",
    "kidney": "Kidney",
    "breast": "Breast",
    "abdomen": "Abdomen",
    "head": "Head",
    "headneck": "Head & neck",
    "brain": "Brain",
    "heart": "Heart",
    "bones": "Bones",
    "tissue": "Tissue",
    "face": "Face",
    "muscles": "Muscles",
}

TITLE = {
    "total": "total",
    "total_mr": "total",
    "body": "body",
    "body_mr": "body",
    "vertebrae_mr": "vertebrae",
    "vertebrae_body": "vertebrae body",
    "vertebrae_pp": "vertebrae (postprocess)",
    "vertebrae_pp_refined": "vertebrae (refined)",
    "lung_nodules": "nodules",
    "lung_vessels": "vessels and airways",
    "liver_segments": "segments",
    "liver_segments_mr": "segments",
    "liver_vessels": "vessels",
    "liver_lesions": "lesions",
    "liver_lesions_mr": "lesions",
    "kidney_cysts": "cysts",
    "breasts": "breasts",
    "abdominal_muscles": "abdominal muscles",
    "trunk_cavities": "trunk cavities",
    "head_glands_cavities": "glands and cavities",
    "head_muscles": "muscles",
    "oculomotor_muscles": "oculomotor muscles",
    "craniofacial_structures": "craniofacial structures",
    "teeth": "teeth",
    "headneck_bones_vessels": "bones and vessels",
    "headneck_muscles": "muscles",
    "cerebral_bleed": "cerebral bleed",
    "brain_aneurysm": "aneurysm (TOF MRI)",
    "brain_structures": "structures",
    "ventricle_parts": "ventricle parts",
    "pleural_pericard_effusion": "pleural / pericardial effusion",
    "coronary_arteries": "coronary arteries",
    "heartchambers_highres": "chambers highres",
    "aortic_sinuses": "aortic sinuses",
    "hip_implant": "hip implant",
    "appendicular_bones": "appendicular bones",
    "appendicular_bones_mr": "appendicular bones",
    "tissue_types": "tissue types",
    "tissue_4_types": "tissue 4 types",
    "tissue_types_mr": "tissue types",
    "face": "face",
    "face_mr": "face",
    "thigh_shoulder_muscles": "thigh and shoulder muscles",
    "thigh_shoulder_muscles_mr": "thigh and shoulder muscles",
}


def skip(tid: str) -> bool:
    return (
        "LEGACY" in tid
        or tid in ("test", "total_highres_test", "total_v1", "total_v3")
        or tid.endswith("_auxiliary")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]  # SlicerLive/
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "render" / "demos" / "ira" / "totalseg-tasks.ts",
    )
    args = parser.parse_args()

    raw = subprocess.check_output(
        [sys.executable, "-m", "totalsegmentator.bin.totalseg_info", "--json"],
        text=True,
        encoding="utf-8",
    )
    data = json.loads(raw)
    tasks = data["tasks"]

    disallow_fast: set[str] = set()
    try:
        from totalsegmentator.python_api import TASK_CONFIGS  # type: ignore

        for tid, cfg in TASK_CONFIGS.items():
            if isinstance(cfg, dict) and cfg.get("disallow_fast"):
                disallow_fast.add(str(tid))
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not read TASK_CONFIGS for supportsFast: {exc}")

    out = []
    ungrouped = []
    for tid, meta in tasks.items():
        if skip(tid):
            continue
        if tid not in GROUP_MAP:
            ungrouped.append(tid)
            continue
        classes = meta.get("classes") or {}
        if isinstance(classes, dict):
            names = [
                classes[k]
                for k in sorted(
                    classes,
                    key=lambda x: int(x) if str(x).isdigit() else str(x),
                )
            ]
        else:
            names = list(classes)
        mod = meta.get("modality") or (
            "MR" if tid.endswith("_mr") or tid == "brain_aneurysm" else "CT"
        )
        out.append(
            {
                "id": tid,
                "modality": mod,
                "licenseRequired": bool(meta.get("license_required")),
                "supportsFast": tid not in disallow_fast,
                "group": GROUP_MAP[tid],
                "title": TITLE.get(tid, tid.replace("_", " ")),
                "classes": names,
            }
        )
    gorder = list(GROUP_LABEL.keys())
    out.sort(
        key=lambda t: (
            gorder.index(t["group"]) if t["group"] in gorder else 99,
            t["title"],
            t["id"],
        )
    )

    lines = [
        "/**",
        " * TotalSegmentator task catalog for IRA (slim registry from `totalseg_info --json`).",
        f" * Regenerated from TotalSegmentator {data.get('totalsegmentator_version')}.",
        " * Refresh: python SlicerLive/tools/regen_totalseg_tasks.py",
        " */",
        "",
        'export type TotalSegModality = "CT" | "MR";',
        "",
        "export type TotalSegTask = {",
        "  id: string;",
        "  modality: TotalSegModality;",
        "  group: string;",
        "  title: string;",
        "  licenseRequired?: boolean;",
        "  /** False when TotalSeg raises if ``--fast`` is passed (disallow_fast). */",
        "  supportsFast?: boolean;",
        "  classes: readonly string[];",
        "};",
        "",
        "export type TotalSegClassHit = {",
        "  className: string;",
        "  task: TotalSegTask;",
        "};",
        "",
        "/** Group id → display label (browse UI order). */",
        "export const TOTAL_SEG_GROUP_LABELS: Readonly<Record<string, string>> = {",
    ]
    for k, v in GROUP_LABEL.items():
        lines.append(f"  {json.dumps(k)}: {json.dumps(v)},")
    lines.append("};")
    lines.append("")
    lines.append(
        f"/** Checked-in catalog (TotalSegmentator {data.get('totalsegmentator_version')}). */"
    )
    lines.append("export const TOTAL_SEG_TASKS: readonly TotalSegTask[] = [")
    for t in out:
        lic = "true" if t["licenseRequired"] else "false"
        supports_fast = "true" if t["supportsFast"] else "false"
        lines.append("  {")
        lines.append(f"    id: {json.dumps(t['id'])},")
        lines.append(f"    modality: {json.dumps(t['modality'])},")
        lines.append(f"    group: {json.dumps(t['group'])},")
        lines.append(f"    title: {json.dumps(t['title'])},")
        lines.append(f"    licenseRequired: {lic},")
        lines.append(f"    supportsFast: {supports_fast},")
        lines.append(f"    classes: {json.dumps(t['classes'])},")
        lines.append("  },")
    lines.append("];")
    lines.append(
        """
const TASK_BY_ID = new Map(TOTAL_SEG_TASKS.map((t) => [t.id, t]));

export function getTotalSegTask(id: string): TotalSegTask | undefined {
  return TASK_BY_ID.get(id);
}

export function tasksForModality(modality: TotalSegModality): TotalSegTask[] {
  return TOTAL_SEG_TASKS.filter((t) => t.modality === modality);
}

export function groupsForModality(modality: TotalSegModality): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const t of tasksForModality(modality)) {
    if (seen.has(t.group)) continue;
    seen.add(t.group);
    out.push(t.group);
  }
  return out;
}

export function tasksInGroup(
  group: string,
  modality: TotalSegModality,
): TotalSegTask[] {
  return tasksForModality(modality).filter((t) => t.group === group);
}

export function defaultTaskId(modality: TotalSegModality): string {
  return modality === "MR" ? "total_mr" : "total";
}

export function taskOptionLabel(task: TotalSegTask): string {
  const lic = task.licenseRequired ? " [license required]" : "";
  return `${task.title}${lic}`;
}

/** Whether TotalSeg accepts ``--fast`` for this task (default true if unknown). */
export function taskSupportsFast(taskId: string): boolean {
  const task = getTotalSegTask(taskId);
  if (!task) return true;
  return task.supportsFast !== false;
}

/** Case-insensitive substring search over class names for a modality (find-task style). */
export function searchClasses(
  query: string,
  modality: TotalSegModality,
  limit = 40,
): TotalSegClassHit[] {
  const q = query.trim().toLowerCase();
  if (q.length < 2) return [];
  const hits: TotalSegClassHit[] = [];
  for (const task of tasksForModality(modality)) {
    for (const className of task.classes) {
      if (!className.toLowerCase().includes(q)) continue;
      hits.push({ className, task });
      if (hits.length >= limit) return hits;
    }
  }
  return hits;
}

/** @deprecated Prefer TOTAL_SEG_TASKS + helpers; kept for any leftover imports. */
export type TotalSegTaskOption = { value: string; title: string };
export const TOTAL_SEG_TASK_OPTIONS: readonly TotalSegTaskOption[] =
  TOTAL_SEG_TASKS.map((t) => ({
    value: t.id,
    title: taskOptionLabel(t),
  }));
export const TOTAL_SEG_DEFAULT_TASK = "total";
"""
    )
    args.out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {args.out} ({len(out)} tasks)")
    if ungrouped:
        print("Ungrouped (skipped):", ", ".join(ungrouped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
