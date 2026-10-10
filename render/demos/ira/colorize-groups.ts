// Organ-group partition for IRA Colorize mode (name-based, not TotalSeg numeric IDs).
// Mirrors examples/colorize GROUP_DEFAULT_OPACITY + prep.py GROUPS intent so Cast/IDC
// labelmaps that remapped SegmentNumbers still get Lungs/Ribs/… defaults and sliders.

export const COLORIZE_GROUP_DEFAULT_OPACITY: Record<string, number> = {
  Organs: 0.9,
  Heart: 0.9,
  Vessels: 0.9,
  Intestines: 0.55,
  Vertebrae: 0.8,
  Lungs: 0.05,
  Ribs: 0.55,
  "Other bone": 0.5,
  Muscle: 0.05,
};

/** TotalSeg-style class names per organ group (CT `total` + common v1/v2 aliases). */
export const COLORIZE_GROUPS: readonly { name: string; classes: readonly string[] }[] = [
  {
    name: "Organs",
    classes: [
      "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "pancreas",
      "adrenal_gland_right", "adrenal_gland_left", "thyroid_gland", "urinary_bladder",
      "prostate", "kidney_cyst_left", "kidney_cyst_right", "brain",
    ],
  },
  {
    name: "Heart",
    classes: [
      "heart", "heart_myocardium", "heart_atrium_left", "heart_ventricle_left",
      "heart_atrium_right", "heart_ventricle_right", "atrial_appendage_left",
    ],
  },
  {
    name: "Vessels",
    classes: [
      "aorta", "inferior_vena_cava", "superior_vena_cava", "portal_vein_and_splenic_vein",
      "pulmonary_vein", "pulmonary_artery", "brachiocephalic_trunk",
      "brachiocephalic_vein_left", "brachiocephalic_vein_right",
      "subclavian_artery_right", "subclavian_artery_left",
      "common_carotid_artery_right", "common_carotid_artery_left",
      "iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right",
    ],
  },
  {
    name: "Lungs",
    classes: [
      "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
      "lung_middle_lobe_right", "lung_lower_lobe_right", "lung_left", "lung_right", "trachea",
    ],
  },
  {
    name: "Intestines",
    classes: ["stomach", "esophagus", "small_bowel", "duodenum", "colon"],
  },
  {
    name: "Vertebrae",
    classes: [
      "sacrum", "spinal_cord", "vertebrae", "vertebrae_S1",
      "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
      "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8",
      "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3",
      "vertebrae_T2", "vertebrae_T1",
      "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4", "vertebrae_C3",
      "vertebrae_C2", "vertebrae_C1",
    ],
  },
  {
    name: "Ribs",
    classes: [
      "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5", "rib_left_6",
      "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12",
      "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6",
      "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12",
    ],
  },
  {
    name: "Other bone",
    classes: [
      "humerus_left", "humerus_right", "scapula_left", "scapula_right",
      "clavicula_left", "clavicula_right", "femur_left", "femur_right",
      "hip_left", "hip_right", "sternum", "costal_cartilages", "skull",
    ],
  },
  {
    name: "Muscle",
    classes: [
      "gluteus_maximus_left", "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right",
      "gluteus_minimus_left", "gluteus_minimus_right", "autochthon_left", "autochthon_right",
      "iliopsoas_left", "iliopsoas_right",
    ],
  },
];

const CLASS_TO_GROUP = new Map<string, string>();
for (const g of COLORIZE_GROUPS) {
  for (const c of g.classes) CLASS_TO_GROUP.set(c, g.name);
}

/** Lowercase + spaces/hyphens → underscores (DICOM "Liver" / NIfTI `liver`). */
export function normalizeSegName(name: string): string {
  return name.trim().toLowerCase().replace(/[\s\-]+/g, "_").replace(/_+/g, "_");
}

/** Map a normalized class/label to a group name, or null if unknown. */
export function colorizeGroupForName(name: string): string | null {
  const n = normalizeSegName(name);
  if (!n) return null;
  const exact = CLASS_TO_GROUP.get(n);
  if (exact) return exact;
  // Prefix families (extra vertebrae / ribs / muscles not listed explicitly).
  if (n.startsWith("vertebrae_") || n === "sacrum" || n === "spinal_cord") return "Vertebrae";
  if (n.startsWith("rib_")) return "Ribs";
  if (n.startsWith("lung_")) return "Lungs";
  if (n.startsWith("gluteus_") || n.startsWith("autochthon_") || n.startsWith("iliopsoas_")) return "Muscle";
  if (n.startsWith("heart_")) return "Heart";
  return null;
}

export type ColorizeGroupResolved = {
  name: string;
  segments: number[];
  opacity: number;
};

/** Groups that have ≥1 matching segment in the current labelmap. */
export function resolveColorizeGroups(
  segments: readonly { num: number; name: string }[],
): ColorizeGroupResolved[] {
  const byGroup = new Map<string, number[]>();
  for (const s of segments) {
    const g = colorizeGroupForName(s.name);
    if (!g) continue;
    let list = byGroup.get(g);
    if (!list) { list = []; byGroup.set(g, list); }
    list.push(s.num);
  }
  const out: ColorizeGroupResolved[] = [];
  for (const g of COLORIZE_GROUPS) {
    const nums = byGroup.get(g.name);
    if (!nums?.length) continue;
    out.push({
      name: g.name,
      segments: nums,
      opacity: COLORIZE_GROUP_DEFAULT_OPACITY[g.name] ?? 1,
    });
  }
  return out;
}

/** Opacity entries to apply when entering Colorize (group defaults for matched segs). */
export function colorizeDefaultOpacityEntries(
  segments: readonly { num: number; name: string }[],
): [number, number][] {
  const groups = resolveColorizeGroups(segments);
  const entries: [number, number][] = [];
  for (const g of groups) {
    for (const n of g.segments) entries.push([n, g.opacity]);
  }
  return entries;
}
