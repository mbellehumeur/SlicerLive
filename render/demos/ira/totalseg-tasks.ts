/**
 * TotalSegmentator task catalog for IRA (slim registry from `totalseg_info --json`).
 * Regenerated from TotalSegmentator 2.18.0.
 * Refresh: python SlicerLive/tools/regen_totalseg_tasks.py
 */

export type TotalSegModality = "CT" | "MR";

export type TotalSegTask = {
  id: string;
  modality: TotalSegModality;
  group: string;
  title: string;
  licenseRequired?: boolean;
  /** False when TotalSeg raises if ``--fast`` is passed (disallow_fast). */
  supportsFast?: boolean;
  classes: readonly string[];
};

export type TotalSegClassHit = {
  className: string;
  task: TotalSegTask;
};

/** Group id → display label (browse UI order). */
export const TOTAL_SEG_GROUP_LABELS: Readonly<Record<string, string>> = {
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
};

/** Checked-in catalog (TotalSegmentator 2.18.0). */
export const TOTAL_SEG_TASKS: readonly TotalSegTask[] = [
  {
    id: "total",
    modality: "CT",
    group: "total",
    title: "total",
    licenseRequired: false,
    supportsFast: true,
    classes: ["spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas", "adrenal_gland_right", "adrenal_gland_left", "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right", "esophagus", "trachea", "thyroid_gland", "small_bowel", "duodenum", "colon", "urinary_bladder", "prostate", "kidney_cyst_left", "kidney_cyst_right", "sacrum", "vertebrae_S1", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1", "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", "vertebrae_T2", "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1", "heart", "aorta", "pulmonary_vein", "brachiocephalic_trunk", "subclavian_artery_right", "subclavian_artery_left", "common_carotid_artery_right", "common_carotid_artery_left", "brachiocephalic_vein_left", "brachiocephalic_vein_right", "atrial_appendage_left", "superior_vena_cava", "inferior_vena_cava", "portal_vein_and_splenic_vein", "iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right", "humerus_left", "humerus_right", "scapula_left", "scapula_right", "clavicula_left", "clavicula_right", "femur_left", "femur_right", "hip_left", "hip_right", "spinal_cord", "gluteus_maximus_left", "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right", "gluteus_minimus_left", "gluteus_minimus_right", "autochthon_left", "autochthon_right", "iliopsoas_left", "iliopsoas_right", "brain", "skull", "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5", "rib_left_6", "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12", "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6", "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12", "sternum", "costal_cartilages"],
  },
  {
    id: "total_mr",
    modality: "MR",
    group: "total",
    title: "total",
    licenseRequired: false,
    supportsFast: true,
    classes: ["spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas", "adrenal_gland_right", "adrenal_gland_left", "lung_left", "lung_right", "esophagus", "small_bowel", "duodenum", "colon", "urinary_bladder", "prostate", "sacrum", "vertebrae", "intervertebral_discs", "spinal_cord", "heart", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein", "iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right", "humerus_left", "humerus_right", "scapula_left", "scapula_right", "clavicula_left", "clavicula_right", "femur_left", "femur_right", "hip_left", "hip_right", "gluteus_maximus_left", "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right", "gluteus_minimus_left", "gluteus_minimus_right", "autochthon_left", "autochthon_right", "iliopsoas_left", "iliopsoas_right", "brain"],
  },
  {
    id: "body",
    modality: "CT",
    group: "body",
    title: "body",
    licenseRequired: false,
    supportsFast: true,
    classes: ["body_trunc", "body_extremities"],
  },
  {
    id: "body_mr",
    modality: "MR",
    group: "body",
    title: "body",
    licenseRequired: false,
    supportsFast: true,
    classes: ["body_trunc", "body_extremities"],
  },
  {
    id: "vertebrae_mr",
    modality: "MR",
    group: "spine",
    title: "vertebrae",
    licenseRequired: false,
    supportsFast: true,
    classes: ["sacrum", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1", "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", "vertebrae_T2", "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1"],
  },
  {
    id: "vertebrae_pp",
    modality: "CT",
    group: "spine",
    title: "vertebrae (postprocess)",
    licenseRequired: false,
    supportsFast: false,
    classes: ["vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7", "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5", "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5"],
  },
  {
    id: "vertebrae_pp_refined",
    modality: "CT",
    group: "spine",
    title: "vertebrae (refined)",
    licenseRequired: false,
    supportsFast: false,
    classes: ["vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7", "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5", "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5"],
  },
  {
    id: "vertebrae_body",
    modality: "CT",
    group: "spine",
    title: "vertebrae body",
    licenseRequired: false,
    supportsFast: false,
    classes: ["vertebrae_body", "intervertebral_discs"],
  },
  {
    id: "lung_nodules",
    modality: "CT",
    group: "lung",
    title: "nodules",
    licenseRequired: false,
    supportsFast: false,
    classes: ["lung", "lung_nodules"],
  },
  {
    id: "lung_vessels",
    modality: "CT",
    group: "lung",
    title: "vessels and airways",
    licenseRequired: false,
    supportsFast: true,
    classes: ["lung_airways", "lung_airways_wall", "lung_arteries", "lung_veins"],
  },
  {
    id: "liver_lesions",
    modality: "CT",
    group: "liver",
    title: "lesions",
    licenseRequired: false,
    supportsFast: false,
    classes: ["liver_lesions"],
  },
  {
    id: "liver_lesions_mr",
    modality: "MR",
    group: "liver",
    title: "lesions",
    licenseRequired: false,
    supportsFast: false,
    classes: ["liver_lesions"],
  },
  {
    id: "liver_segments",
    modality: "CT",
    group: "liver",
    title: "segments",
    licenseRequired: false,
    supportsFast: false,
    classes: ["liver_segment_1", "liver_segment_2", "liver_segment_3", "liver_segment_4", "liver_segment_5", "liver_segment_6", "liver_segment_7", "liver_segment_8"],
  },
  {
    id: "liver_segments_mr",
    modality: "MR",
    group: "liver",
    title: "segments",
    licenseRequired: false,
    supportsFast: false,
    classes: ["liver_segment_1", "liver_segment_2", "liver_segment_3", "liver_segment_4", "liver_segment_5", "liver_segment_6", "liver_segment_7", "liver_segment_8"],
  },
  {
    id: "liver_vessels",
    modality: "CT",
    group: "liver",
    title: "vessels",
    licenseRequired: false,
    supportsFast: false,
    classes: ["liver_vessels", "liver_tumor"],
  },
  {
    id: "kidney_cysts",
    modality: "CT",
    group: "kidney",
    title: "cysts",
    licenseRequired: false,
    supportsFast: false,
    classes: ["kidney_cyst_left", "kidney_cyst_right"],
  },
  {
    id: "breasts",
    modality: "CT",
    group: "breast",
    title: "breasts",
    licenseRequired: false,
    supportsFast: false,
    classes: ["breast"],
  },
  {
    id: "abdominal_muscles",
    modality: "CT",
    group: "abdomen",
    title: "abdominal muscles",
    licenseRequired: false,
    supportsFast: false,
    classes: ["pectoralis_major_right", "pectoralis_major_left", "rectus_abdominis_right", "rectus_abdominis_left", "serratus_anterior_right", "serratus_anterior_left", "latissimus_dorsi_right", "latissimus_dorsi_left", "trapezius_right", "trapezius_left", "external_oblique_right", "external_oblique_left", "internal_oblique_right", "internal_oblique_left", "erector_spinae_right", "erector_spinae_left", "transversospinalis_right", "transversospinalis_left", "psoas_major_right", "psoas_major_left", "quadratus_lumborum_right", "quadratus_lumborum_left"],
  },
  {
    id: "trunk_cavities",
    modality: "CT",
    group: "abdomen",
    title: "trunk cavities",
    licenseRequired: false,
    supportsFast: false,
    classes: ["abdominal_cavity", "thoracic_cavity", "pericardium", "mediastinum"],
  },
  {
    id: "craniofacial_structures",
    modality: "CT",
    group: "head",
    title: "craniofacial structures",
    licenseRequired: false,
    supportsFast: false,
    classes: ["mandible", "teeth_lower", "skull", "head", "sinus_maxillary", "sinus_frontal", "teeth_upper"],
  },
  {
    id: "head_glands_cavities",
    modality: "CT",
    group: "head",
    title: "glands and cavities",
    licenseRequired: false,
    supportsFast: false,
    classes: ["eye_left", "eye_right", "eye_lens_left", "eye_lens_right", "optic_nerve_left", "optic_nerve_right", "parotid_gland_left", "parotid_gland_right", "submandibular_gland_right", "submandibular_gland_left", "nasopharynx", "oropharynx", "hypopharynx", "nasal_cavity_right", "nasal_cavity_left", "auditory_canal_right", "auditory_canal_left", "soft_palate", "hard_palate"],
  },
  {
    id: "head_muscles",
    modality: "CT",
    group: "head",
    title: "muscles",
    licenseRequired: false,
    supportsFast: false,
    classes: ["masseter_right", "masseter_left", "temporalis_right", "temporalis_left", "lateral_pterygoid_right", "lateral_pterygoid_left", "medial_pterygoid_right", "medial_pterygoid_left", "tongue", "digastric_right", "digastric_left"],
  },
  {
    id: "oculomotor_muscles",
    modality: "CT",
    group: "head",
    title: "oculomotor muscles",
    licenseRequired: false,
    supportsFast: false,
    classes: ["skull", "eyeball_right", "lateral_rectus_muscle_right", "superior_oblique_muscle_right", "levator_palpebrae_superioris_right", "superior_rectus_muscle_right", "medial_rectus_muscle_left", "inferior_oblique_muscle_right", "inferior_rectus_muscle_right", "optic_nerve_left", "eyeball_left", "lateral_rectus_muscle_left", "superior_oblique_muscle_left", "levator_palpebrae_superioris_left", "superior_rectus_muscle_left", "medial_rectus_muscle_right", "inferior_oblique_muscle_left", "inferior_rectus_muscle_left", "optic_nerve_right"],
  },
  {
    id: "teeth",
    modality: "CT",
    group: "head",
    title: "teeth",
    licenseRequired: false,
    supportsFast: false,
    classes: ["lower_jawbone", "upper_jawbone", "left_inferior_alveolar_canal", "right_inferior_alveolar_canal", "left_maxillary_sinus", "right_maxillary_sinus", "pharynx", "bridge", "crown", "implant", "upper_right_central_incisor_fdi11", "upper_right_lateral_incisor_fdi12", "upper_right_canine_fdi13", "upper_right_first_premolar_fdi14", "upper_right_second_premolar_fdi15", "upper_right_first_molar_fdi16", "upper_right_second_molar_fdi17", "upper_right_third_molar_fdi18", "upper_left_central_incisor_fdi21", "upper_left_lateral_incisor_fdi22", "upper_left_canine_fdi23", "upper_left_first_premolar_fdi24", "upper_left_second_premolar_fdi25", "upper_left_first_molar_fdi26", "upper_left_second_molar_fdi27", "upper_left_third_molar_fdi28", "lower_left_central_incisor_fdi31", "lower_left_lateral_incisor_fdi32", "lower_left_canine_fdi33", "lower_left_first_premolar_fdi34", "lower_left_second_premolar_fdi35", "lower_left_first_molar_fdi36", "lower_left_second_molar_fdi37", "lower_left_third_molar_fdi38", "lower_right_central_incisor_fdi41", "lower_right_lateral_incisor_fdi42", "lower_right_canine_fdi43", "lower_right_first_premolar_fdi44", "lower_right_second_premolar_fdi45", "lower_right_first_molar_fdi46", "lower_right_second_molar_fdi47", "lower_right_third_molar_fdi48", "left_mandibular_incisive_canal_fdi103", "right_mandibular_incisive_canal_fdi104", "lingual_canal", "upper_right_central_incisor_pulp_fdi111", "upper_right_lateral_incisor_pulp_fdi112", "upper_right_canine_pulp_fdi113", "upper_right_first_premolar_pulp_fdi114", "upper_right_second_premolar_pulp_fdi115", "upper_right_first_molar_pulp_fdi116", "upper_right_second_molar_pulp_fdi117", "upper_right_third_molar_pulp_fdi118", "upper_left_central_incisor_pulp_fdi121", "upper_left_lateral_incisor_pulp_fdi122", "upper_left_canine_pulp_fdi123", "upper_left_first_premolar_pulp_fdi124", "upper_left_second_premolar_pulp_fdi125", "upper_left_first_molar_pulp_fdi126", "upper_left_second_molar_pulp_fdi127", "upper_left_third_molar_pulp_fdi128", "lower_left_central_incisor_pulp_fdi131", "lower_left_lateral_incisor_pulp_fdi132", "lower_left_canine_pulp_fdi133", "lower_left_first_premolar_pulp_fdi134", "lower_left_second_premolar_pulp_fdi135", "lower_left_first_molar_pulp_fdi136", "lower_left_second_molar_pulp_fdi137", "lower_left_third_molar_pulp_fdi138", "lower_right_central_incisor_pulp_fdi141", "lower_right_lateral_incisor_pulp_fdi142", "lower_right_canine_pulp_fdi143", "lower_right_first_premolar_pulp_fdi144", "lower_right_second_premolar_pulp_fdi145", "lower_right_first_molar_pulp_fdi146", "lower_right_second_molar_pulp_fdi147", "lower_right_third_molar_pulp_fdi148"],
  },
  {
    id: "headneck_bones_vessels",
    modality: "CT",
    group: "headneck",
    title: "bones and vessels",
    licenseRequired: false,
    supportsFast: false,
    classes: ["larynx_air", "thyroid_cartilage", "hyoid", "cricoid_cartilage", "zygomatic_arch_right", "zygomatic_arch_left", "styloid_process_right", "styloid_process_left", "internal_carotid_artery_right", "internal_carotid_artery_left", "internal_jugular_vein_right", "internal_jugular_vein_left"],
  },
  {
    id: "headneck_muscles",
    modality: "CT",
    group: "headneck",
    title: "muscles",
    licenseRequired: false,
    supportsFast: false,
    classes: ["sternocleidomastoid_right", "sternocleidomastoid_left", "superior_pharyngeal_constrictor", "middle_pharyngeal_constrictor", "inferior_pharyngeal_constrictor", "trapezius_right", "trapezius_left", "platysma_right", "platysma_left", "levator_scapulae_right", "levator_scapulae_left", "anterior_scalene_right", "anterior_scalene_left", "middle_scalene_right", "middle_scalene_left", "posterior_scalene_right", "posterior_scalene_left", "sterno_thyroid_right", "sterno_thyroid_left", "thyrohyoid_right", "thyrohyoid_left", "prevertebral_right", "prevertebral_left"],
  },
  {
    id: "brain_aneurysm",
    modality: "MR",
    group: "brain",
    title: "aneurysm (TOF MRI)",
    licenseRequired: false,
    supportsFast: false,
    classes: ["brain_aneurysm"],
  },
  {
    id: "cerebral_bleed",
    modality: "CT",
    group: "brain",
    title: "cerebral bleed",
    licenseRequired: false,
    supportsFast: false,
    classes: ["intracerebral_hemorrhage"],
  },
  {
    id: "brain_structures",
    modality: "CT",
    group: "brain",
    title: "structures",
    licenseRequired: true,
    supportsFast: false,
    classes: ["brainstem", "subarachnoid_space", "venous_sinuses", "septum_pellucidum", "cerebellum", "caudate_nucleus", "lentiform_nucleus", "insular_cortex", "internal_capsule", "ventricle", "central_sulcus", "frontal_lobe", "parietal_lobe", "occipital_lobe", "temporal_lobe", "thalamus"],
  },
  {
    id: "ventricle_parts",
    modality: "CT",
    group: "brain",
    title: "ventricle parts",
    licenseRequired: false,
    supportsFast: false,
    classes: ["ventricle_frontal_horn_left", "ventricle_occipital_horn_left", "ventricle_body_left", "ventricle_temporal_horn_left", "ventricle_trigone_left", "ventricle_frontal_horn_right", "ventricle_occipital_horn_right", "ventricle_body_right", "ventricle_temporal_horn_right", "ventricle_trigone_right", "third_ventricle", "fourth_ventricle"],
  },
  {
    id: "aortic_sinuses",
    modality: "CT",
    group: "heart",
    title: "aortic sinuses",
    licenseRequired: true,
    supportsFast: false,
    classes: ["left_ventricular_outflow_tract", "right_coronary_cusp", "left_coronary_cusp", "non_coronary_cusp"],
  },
  {
    id: "heartchambers_highres",
    modality: "CT",
    group: "heart",
    title: "chambers highres",
    licenseRequired: true,
    supportsFast: false,
    classes: ["heart_myocardium", "heart_atrium_left", "heart_ventricle_left", "heart_atrium_right", "heart_ventricle_right", "aorta", "pulmonary_artery"],
  },
  {
    id: "coronary_arteries",
    modality: "CT",
    group: "heart",
    title: "coronary arteries",
    licenseRequired: true,
    supportsFast: false,
    classes: ["coronary_arteries"],
  },
  {
    id: "pleural_pericard_effusion",
    modality: "CT",
    group: "heart",
    title: "pleural / pericardial effusion",
    licenseRequired: false,
    supportsFast: false,
    classes: ["lung_pleural", "pleural_effusion", "pericardial_effusion"],
  },
  {
    id: "appendicular_bones",
    modality: "CT",
    group: "bones",
    title: "appendicular bones",
    licenseRequired: true,
    supportsFast: false,
    classes: ["patella", "tibia", "fibula", "tarsal", "metatarsal", "phalanges_feet", "ulna", "radius", "carpal", "metacarpal", "phalanges_hand"],
  },
  {
    id: "appendicular_bones_mr",
    modality: "MR",
    group: "bones",
    title: "appendicular bones",
    licenseRequired: true,
    supportsFast: false,
    classes: ["patella", "tibia", "fibula", "tarsal", "metatarsal", "phalanges_feet", "ulna", "radius"],
  },
  {
    id: "hip_implant",
    modality: "CT",
    group: "bones",
    title: "hip implant",
    licenseRequired: false,
    supportsFast: false,
    classes: ["hip_implant"],
  },
  {
    id: "tissue_4_types",
    modality: "CT",
    group: "tissue",
    title: "tissue 4 types",
    licenseRequired: true,
    supportsFast: false,
    classes: ["subcutaneous_fat", "torso_fat", "skeletal_muscle", "intermuscular_fat"],
  },
  {
    id: "tissue_types",
    modality: "CT",
    group: "tissue",
    title: "tissue types",
    licenseRequired: true,
    supportsFast: false,
    classes: ["subcutaneous_fat", "torso_fat", "skeletal_muscle"],
  },
  {
    id: "tissue_types_mr",
    modality: "MR",
    group: "tissue",
    title: "tissue types",
    licenseRequired: true,
    supportsFast: false,
    classes: ["subcutaneous_fat", "torso_fat", "skeletal_muscle"],
  },
  {
    id: "face",
    modality: "CT",
    group: "face",
    title: "face",
    licenseRequired: true,
    supportsFast: false,
    classes: ["face"],
  },
  {
    id: "face_mr",
    modality: "MR",
    group: "face",
    title: "face",
    licenseRequired: true,
    supportsFast: false,
    classes: ["face"],
  },
  {
    id: "thigh_shoulder_muscles",
    modality: "CT",
    group: "muscles",
    title: "thigh and shoulder muscles",
    licenseRequired: true,
    supportsFast: false,
    classes: ["quadriceps_femoris_left", "quadriceps_femoris_right", "thigh_medial_compartment_left", "thigh_medial_compartment_right", "thigh_posterior_compartment_left", "thigh_posterior_compartment_right", "sartorius_left", "sartorius_right", "deltoid", "supraspinatus", "infraspinatus", "subscapularis", "coracobrachial", "trapezius", "pectoralis_minor", "serratus_anterior", "teres_major", "triceps_brachii"],
  },
  {
    id: "thigh_shoulder_muscles_mr",
    modality: "MR",
    group: "muscles",
    title: "thigh and shoulder muscles",
    licenseRequired: true,
    supportsFast: false,
    classes: ["quadriceps_femoris_left", "quadriceps_femoris_right", "thigh_medial_compartment_left", "thigh_medial_compartment_right", "thigh_posterior_compartment_left", "thigh_posterior_compartment_right", "sartorius_left", "sartorius_right", "deltoid", "supraspinatus", "infraspinatus", "subscapularis", "coracobrachial", "trapezius", "pectoralis_minor", "serratus_anterior", "teres_major", "triceps_brachii"],
  },
];

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
