/**
 * Slim TotalSegmentator task list for IRA (CLI --task ids).
 * Labels aligned with cast-ohif-extension / totalsegmentator.com.
 */
export type TotalSegTaskOption = { value: string; title: string };

/** Insertion order matches OHIF / Slicer catalog; ``total`` first (default). */
export const TOTAL_SEG_TASK_OPTIONS: readonly TotalSegTaskOption[] = [
  { value: "total", title: "total" },
  { value: "total_mr", title: "total (MR)" },
  { value: "vertebrae_mr", title: "vertebrae (MR)" },
  { value: "lung_nodules", title: "lung: nodules" },
  { value: "lung_vessels", title: "lung: vessels and airways" },
  { value: "kidney_cysts", title: "kidney: cysts" },
  { value: "breasts", title: "breasts" },
  { value: "liver_segments", title: "liver: segments" },
  { value: "liver_segments_mr", title: "liver: segments (MR)" },
  { value: "liver_vessels", title: "liver: vessels" },
  { value: "liver_lesions", title: "liver: lesions" },
  { value: "liver_lesions_mr", title: "liver: lesions (MR)" },
  { value: "abdominal_muscles", title: "abdominal muscles" },
  { value: "trunk_cavities", title: "trunk cavities" },
  { value: "body", title: "body" },
  { value: "body_mr", title: "body (MR)" },
  { value: "head_glands_cavities", title: "head: glands and cavities" },
  { value: "head_muscles", title: "head: muscles" },
  { value: "oculomotor_muscles", title: "head: oculomotor muscles" },
  { value: "craniofacial_structures", title: "head: craniofacial structures" },
  { value: "teeth", title: "head: teeth" },
  { value: "headneck_bones_vessels", title: "head and neck: bones and vessels" },
  { value: "headneck_muscles", title: "head and neck: muscles" },
  { value: "cerebral_bleed", title: "brain: cerebral bleed" },
  { value: "brain_aneurysm", title: "brain: aneurysm (TOF MRI)" },
  { value: "hip_implant", title: "hip implant" },
  { value: "pleural_pericard_effusion", title: "heart: pleural and pericardial effusion" },
  { value: "coronary_arteries", title: "heart: coronary arteries [license required]" },
  { value: "vertebrae_body", title: "vertebrae body [license required]" },
  { value: "appendicular_bones", title: "appendicular bones [license required]" },
  { value: "appendicular_bones_mr", title: "appendicular bones (MR) [license required]" },
  { value: "tissue_types", title: "tissue types [license required]" },
  { value: "tissue_4_types", title: "tissue 4 types [license required]" },
  { value: "tissue_types_mr", title: "tissue types (MR) [license required]" },
  { value: "heartchambers_highres", title: "heart: chambers highres [license required]" },
  { value: "face", title: "face [license required]" },
  { value: "face_mr", title: "face (MR) [license required]" },
  { value: "brain_structures", title: "brain: structures [license required]" },
  { value: "thigh_shoulder_muscles", title: "thigh and shoulder: muscles [license required]" },
  {
    value: "thigh_shoulder_muscles_mr",
    title: "thigh and shoulder: muscles (MR) [license required]",
  },
];

export const TOTAL_SEG_DEFAULT_TASK = "total";
