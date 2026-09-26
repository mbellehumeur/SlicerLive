/**
 * Slim MHub model catalog for IRA Medical Inference / worklist.
 * Source: products/mhub/MHubSkill/data/models_summary.json
 * (cache_date: 2025-01-29).
 * Enriched with GitHub / website / cite via:
 *   node cast_resource_servers/products/mhub/export_ira_catalog.mjs
 */

export type MhubModelDef = {
  name: string;
  label: string;
  description: string;
  modalities: string[];
  category: string;
  /** Canonical body parts from model inputs[].bodypartexamined */
  bodyParts: string[];
  /** SegDB segment IDs this model can output (skill find). */
  segments: string[];
  githubUrl?: string;
  websiteUrl?: string;
  cite?: string;
  citeUrl?: string;
};

/** @type {readonly MhubModelDef[]} */
export const MHUB_MODELS: readonly MhubModelDef[] = Object.freeze([
  {
    "name": "bamf_nnunet_ct_kidney",
    "label": "BAMF CT Kidney Seg (nnU-Net)",
    "description": "An nnU-Net based model to segment kidney from CT scans",
    "modalities": [
      "CT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Kidney"
    ],
    "segments": [
      "KIDNEY",
      "KIDNEY+NEOPLASM_MALIGNANT_PRIMARY",
      "KIDNEY+CYST"
    ],
    "websiteUrl": "https://mhub.ai/models/bamf_nnunet_ct_kidney",
    "githubUrl": "https://github.com/bamf-health/aimi-kidney-ct",
    "cite": "Murugesan, Gowtham Krishnan, Diana McCrumb, Mariam Aboian, Tej Verma, Rahul Soni, Fatima Memon, and Jeff Van Oss. The AIMI Initiative: AI-Generated Annotations for Imaging Data Commons Collections. arXiv preprint arXiv:2310.14897 (2023).",
    "citeUrl": "https://arxiv.org/abs/2310.14897"
  },
  {
    "name": "bamf_nnunet_ct_liver",
    "label": "BAMF CT Liver Seg (nnU-Net)",
    "description": "An nnU-Net based model to segment liver from CT scans",
    "modalities": [
      "CT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Liver"
    ],
    "segments": [
      "LIVER"
    ],
    "websiteUrl": "https://mhub.ai/models/bamf_nnunet_ct_liver",
    "githubUrl": "https://github.com/bamf-health/aimi-liver-ct",
    "cite": "Murugesan, Gowtham Krishnan, Diana McCrumb, Mariam Aboian, Tej Verma, Rahul Soni, Fatima Memon, and Jeff Van Oss. The AIMI Initiative: AI-Generated Annotations for Imaging Data Commons Collections. arXiv preprint arXiv:2310.14897 (2023).",
    "citeUrl": "https://arxiv.org/abs/2310.14897"
  },
  {
    "name": "bamf_nnunet_mr_liver",
    "label": "BAMF MR Liver Seg (nnU-Net)",
    "description": "An nnU-Net based model to segment liver from T1 weighted MRI scans",
    "modalities": [
      "MR"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Liver"
    ],
    "segments": [
      "LIVER"
    ],
    "websiteUrl": "https://mhub.ai/models/bamf_nnunet_mr_liver",
    "githubUrl": "https://github.com/bamf-health/aimi-liver-mr",
    "cite": "Murugesan, Gowtham Krishnan, Diana McCrumb, Mariam Aboian, Tej Verma, Rahul Soni, Fatima Memon, and Jeff Van Oss. The AIMI Initiative: AI-Generated Annotations for Imaging Data Commons Collections. arXiv preprint arXiv:2310.14897 (2023).",
    "citeUrl": "https://arxiv.org/abs/2310.14897"
  },
  {
    "name": "bamf_nnunet_mr_prostate",
    "label": "BAMF MR Prostate Seg (nnU-Net)",
    "description": "bamf_nnunet_mr_prostate model is a semantic segmentation model where an NNUnet model is finetuned for MR images",
    "modalities": [
      "MR"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Prostate"
    ],
    "segments": [
      "PROSTATE"
    ],
    "websiteUrl": "https://mhub.ai/models/bamf_nnunet_mr_prostate",
    "githubUrl": "https://github.com/bamf-health/aimi-prostate-mr",
    "cite": "Murugesan, Gowtham Krishnan, Diana McCrumb, Mariam Aboian, Tej Verma, Rahul Soni, Fatima Memon, and Jeff Van Oss. The AIMI Initiative: AI-Generated Annotations for Imaging Data Commons Collections. arXiv preprint arXiv:2310.14897 (2023).",
    "citeUrl": "https://arxiv.org/abs/2310.14897"
  },
  {
    "name": "bamf_pet_ct_breast_tumor",
    "label": "BAMF PET CT Breast Seg (nnU-Net)",
    "description": "This model used to detect FDG-avid lesions in breast from FDG PET/CT scans",
    "modalities": [
      "CT",
      "PT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Breast"
    ],
    "segments": [
      "BREAST+FDG_AVID_TUMOR"
    ],
    "websiteUrl": "https://mhub.ai/models/bamf_pet_ct_breast_tumor",
    "githubUrl": "https://github.com/bamf-health/aimi-breast-pet-ct",
    "cite": "Murugesan, Gowtham Krishnan, Diana McCrumb, Mariam Aboian, Tej Verma, Rahul Soni, Fatima Memon, and Jeff Van Oss. The AIMI Initiative: AI-Generated Annotations for Imaging Data Commons Collections. arXiv preprint arXiv:2310.14897 (2023).",
    "citeUrl": "https://arxiv.org/abs/2310.14897"
  },
  {
    "name": "casust",
    "label": "CaSuSt",
    "description": "A deep learning model for cardiac sub-structure delineation on planning CT scans. The model delineates the heart contours and seven cardiac substructures based on individually trained binary models.",
    "modalities": [
      "CT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Chest"
    ],
    "segments": [
      "HEART",
      "LEFT_VENTRICLE",
      "RIGHT_VENTRICLE",
      "LEFT_ATRIUM",
      "RIGHT_ATRIUM",
      "CORONARY_ARTERY_LAD",
      "CORONARY_ARTERY_CFLX",
      "CORONARY_ARTERY_RIGHT"
    ],
    "websiteUrl": "https://mhub.ai/models/casust",
    "githubUrl": "https://github.com/LennyN95/CaSuSt",
    "cite": "Nürnberg, L, Bontempi, D, De Ruysscher, D, et al. Deep learning segmentation of heart substructures in radiotherapy treatment planning. Physica Medica: European journal of medical physics, 2022",
    "citeUrl": "https://cris.maastrichtuniversity.nl/en/publications/deep-learning-segmentation-of-heart-substructures-in-radiotherapy"
  },
  {
    "name": "gc_node21_baseline",
    "label": "Chest Radiograph Nodule Locator (NODE21 Baseline)",
    "description": "This model detects the location of nodules in Chest radiographs, and generates bounding boxes around these nodules.",
    "modalities": [
      "CR"
    ],
    "category": "Prediction",
    "bodyParts": [
      "Chest"
    ],
    "segments": [],
    "websiteUrl": "https://mhub.ai/models/gc_node21_baseline",
    "githubUrl": "https://github.com/node21challenge/node21_detection_baseline",
    "cite": "E. Sogancioglu et al., Nodule detection and generation on chest X-rays: NODE21 Challenge, in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2024.3382042.",
    "citeUrl": "https://doi.org/10.1109/TMI.2024.3382042"
  },
  {
    "name": "nnunet_liver",
    "label": "CT Liver Seg (nnU-Net)",
    "description": "nnU-Net's liver segmentation model is an AI-based pipeline for the automated segmentation of the liver and liver cancer in portal venous phase CT scans.",
    "modalities": [
      "CT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Abdomen"
    ],
    "segments": [
      "LIVER",
      "LIVER+NEOPLASM_MALIGNANT_PRIMARY"
    ],
    "websiteUrl": "https://mhub.ai/models/nnunet_liver",
    "githubUrl": "https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1",
    "cite": "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.",
    "citeUrl": "https://www.nature.com/articles/s41592-020-01008-z"
  },
  {
    "name": "gc_grt123_lung_cancer",
    "label": "CT Lung cancer risk prediction",
    "description": "This algorithm analyzes non-contrast CT scans of the thorax and predicts the lung cancer risk. The model consists of two modules. The first one is a 3D region proposal network for nodule detection, which outputs all suspicious nodules for a subject. The second one selects the top five nodules based on the detection confidence, evaluates their cancer probabilities and combines them with a leaky noisy-or gate to obtain the probability of lung cancer for the subject. This model was the winner of the Data Science Bowl 2017 competition hosted on Kaggle.",
    "modalities": [
      "CT"
    ],
    "category": "Prediction",
    "bodyParts": [
      "Chest"
    ],
    "segments": [],
    "websiteUrl": "https://mhub.ai/models/gc_grt123_lung_cancer",
    "githubUrl": "https://github.com/DIAGNijmegen/bodyct-dsb2017-grt123",
    "cite": "F. Liao, M. Liang, Z. Li, X. Hu and S. Song, 'Evaluate the Malignancy of Pulmonary Nodules Using the 3D Deep Leaky Noisy-or Network', in IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 11, pp. 3484-3495, Nov. 2019, doi: 10.1109/TNNLS.2019.2892409.",
    "citeUrl": "https://doi.org/10.1109/TNNLS.2019.2892409"
  },
  {
    "name": "bamf_pet_ct_lung_tumor",
    "label": "FDG PET/CT Lung and Lung Tumor Annotation",
    "description": "An nnU-Net based model to segment Lung and FDG-avid lesions in the lung from FDG PET/CT scans",
    "modalities": [
      "CT",
      "PT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Lung"
    ],
    "segments": [
      "LUNG",
      "LUNG+FDG_AVID_TUMOR"
    ],
    "websiteUrl": "https://mhub.ai/models/bamf_pet_ct_lung_tumor",
    "githubUrl": "https://github.com/bamf-health/aimi-lung-pet-ct",
    "cite": "Murugesan, Gowtham Krishnan, Diana McCrumb, Mariam Aboian, Tej Verma, Rahul Soni, Fatima Memon, and Jeff Van Oss. The AIMI Initiative: AI-Generated Annotations for Imaging Data Commons Collections. arXiv preprint arXiv:2310.14897 (2023).",
    "citeUrl": "https://arxiv.org/abs/2310.14897"
  },
  {
    "name": "fmcib_radiomics",
    "label": "Foundation Model for Cancer Imaging Biomarkers",
    "description": "A foundation model for cancer imaging biomarker discovery trained through self-supervised learning using a dataset of 11,467 radiographic lesions. The model features can be used as a data-driven substitute for classical radiomic features",
    "modalities": [
      "CT",
      "SEG"
    ],
    "category": "Prediction",
    "bodyParts": [
      "WHOLEBODY"
    ],
    "segments": [],
    "websiteUrl": "https://mhub.ai/models/fmcib_radiomics",
    "githubUrl": "https://github.com/AIM-Harvard/foundation-cancer-image-biomarker",
    "cite": "Pai, S., Bontempi, D., Hadzic, I. et al. Foundation model for cancer imaging biomarkers. Nat Mach Intell (2024). https://doi.org/10.1038/s42256-024-00807-9",
    "citeUrl": "https://doi.org/10.1038/s42256-024-00807-9"
  },
  {
    "name": "lungmask",
    "label": "LungMask",
    "description": "LungMask is an AI-based pipeline for the automated lungs and pulmonary lobes segmentation in CT (robust to the presence of severe pathologies).",
    "modalities": [
      "CT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Chest"
    ],
    "segments": [
      "RIGHT_LUNG",
      "LEFT_LUNG",
      "LEFT_UPPER_LUNG_LOBE",
      "LEFT_LOWER_LUNG_LOBE",
      "RIGHT_UPPER_LUNG_LOBE",
      "RIGHT_MIDDLE_LUNG_LOBE",
      "RIGHT_LOWER_LUNG_LOBE"
    ],
    "websiteUrl": "https://mhub.ai/models/lungmask",
    "githubUrl": "https://github.com/JoHof/lungmask",
    "cite": "Hofmanninger, J., Prayer, F., Pan, J. et al. Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem. Eur Radiol Exp 4, 50 (2020). https://doi.org/10.1186/s41747-020-00173-2",
    "citeUrl": "https://doi.org/10.1186/s41747-020-00173-2"
  },
  {
    "name": "mrsegmentator",
    "label": "MRSegmentator",
    "description": "MRSegmentator is an AI-based pipeline for the segmentation of 40 anatomical structures in MR images (with and without contrast).",
    "modalities": [
      "MRI|CT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "WHOLEBODY"
    ],
    "segments": [
      "SPLEEN",
      "RIGHT_KIDNEY",
      "LEFT_KIDNEY",
      "GALLBLADDER",
      "LIVER",
      "STOMACH",
      "PANCREAS",
      "RIGHT_ADRENAL_GLAND",
      "LEFT_ADRENAL_GLAND",
      "LEFT_LUNG",
      "RIGHT_LUNG",
      "HEART",
      "AORTA",
      "INFERIOR_VENA_CAVA",
      "PORTAL_AND_SPLENIC_VEIN",
      "LEFT_ILIAC_ARTERY",
      "RIGHT_ILIAC_ARTERY",
      "ESOPHAGUS",
      "SMALL_INTESTINE",
      "DUODENUM",
      "COLON",
      "URINARY_BLADDER",
      "SPINE",
      "SACRUM",
      "LEFT_HIP",
      "RIGHT_HIP",
      "LEFT_FEMUR",
      "RIGHT_FEMUR",
      "LEFT_AUTOCHTHONOUS_BACK_MUSCLE",
      "RIGHT_AUTOCHTHONOUS_BACK_MUSCLE",
      "LEFT_ILIOPSOAS",
      "RIGHT_ILIOPSOAS",
      "LEFT_GLUTEUS_MAXIMUS",
      "RIGHT_GLUTEUS_MAXIMUS",
      "LEFT_GLUTEUS_MEDIUS",
      "RIGHT_GLUTEUS_MEDIUS",
      "LEFT_GLUTEUS_MINIMUS",
      "RIGHT_GLUTEUS_MINIMUS"
    ],
    "websiteUrl": "https://mhub.ai/models/mrsegmentator",
    "githubUrl": "https://github.com/hhaentze/MRSegmentator",
    "cite": "Hartmut Häntze, Lina Xu, Felix J. Dorfner, Leonhard Donle, Daniel Truhn, Hugo Aerts, Mathias Prokop, Bram van Ginneken, Alessa Hering, Lisa C. Adams, and Keno K. Bressem. MRSegmentator: Robust multi-modality segmentation of 40 classes in MRI and CT sequences. arXiv, 2024.",
    "citeUrl": "https://arxiv.org/pdf/2405.06463"
  },
  {
    "name": "nnunet_pancreas",
    "label": "Pancreas Seg (nnU-Net)",
    "description": "nnU-Net's pancreas segmentation model is an AI-based pipeline for the automated segmentation of the pancreas and pancreatic cancer in portal venous phase CT scans.",
    "modalities": [
      "CT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Abdomen"
    ],
    "segments": [
      "PANCREAS",
      "PANCREAS+NEOPLASM_MALIGNANT_PRIMARY"
    ],
    "websiteUrl": "https://mhub.ai/models/nnunet_pancreas",
    "githubUrl": "https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1",
    "cite": "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.",
    "citeUrl": "https://www.nature.com/articles/s41592-020-01008-z"
  },
  {
    "name": "gc_nnunet_pancreas",
    "label": "Pancreatic Ductal Adenocarcinoma Detection in CT",
    "description": "This algorithm produces a tumor likelihood heatmap for the presence of pancreatic ductal adenocarcinoma (PDAC) in an input venous-phase contrast-enhanced computed tomography scan (CECT). Additionally, the algorithm provides the segmentation of multiple surrounding anatomical structures such as the pancreatic duct, common bile duct, veins and arteries. The heatmap and segmentations are resampled to the same spatial resolution and physical dimensions as the input CECT image for easier visualisation.",
    "modalities": [
      "CT"
    ],
    "category": "Prediction",
    "bodyParts": [
      "Abdomen"
    ],
    "segments": [
      "VEIN",
      "ARTERY",
      "PANCREAS",
      "PANCREATIC_DUCT",
      "BILE_DUCT"
    ],
    "websiteUrl": "https://mhub.ai/models/gc_nnunet_pancreas",
    "githubUrl": "https://github.com/DIAGNijmegen/CE-CT_PDAC_AutomaticDetection_nnUnet",
    "cite": "Alves N, Schuurmans M, Litjens G, Bosma JS, Hermans J, Huisman H. Fully Automatic Deep Learning Framework for Pancreatic Ductal Adenocarcinoma Detection on Computed Tomography. Cancers (Basel). 2022 Jan 13;14(2):376. doi: 10.3390/cancers14020376. PMID: 35053538; PMCID: PMC8774174.",
    "citeUrl": "https://doi.org/10.3390/cancers14020376"
  },
  {
    "name": "gc_picai_baseline",
    "label": "PI-CAI challenge baseline",
    "description": "This algorithm predicts a detection map for the likelihood of clinically significant prostate cancer (csPCa) using biparametric MRI (bpMRI). The algorithm ensembles 5-fold cross-validation models that were trained on the PI-CAI: Public Training and Development Dataset v2.0. The detection map is at the same spatial resolution and physical dimensions as the input axial T2-weighted image. This model algorithm was used as a baseline for the PI-CAI challenge hosted on Grand Challenge.",
    "modalities": [
      "MR"
    ],
    "category": "Prediction",
    "bodyParts": [
      "Prostate"
    ],
    "segments": [],
    "websiteUrl": "https://mhub.ai/models/gc_picai_baseline",
    "githubUrl": "https://github.com/DIAGNijmegen/picai_nnunet_semi_supervised_gc_algorithm",
    "cite": "J. S. Bosma, A. Saha, M. Hosseinzadeh, I. Slootweg, M. de Rooij, and H. Huisman, \"Semisupervised Learning with Report-guided Pseudo Labels for Deep Learning–based Prostate Cancer Detection Using Biparametric MRI\", Radiology: Artificial Intelligence, 230031, 2023. DOI: 10.1148/ryai.230031",
    "citeUrl": "https://doi.org/10.1148/ryai.230031"
  },
  {
    "name": "platipy",
    "label": "Platipy",
    "description": "Platipy's cardiac substructures segmentation model is a hybrid pipeline (AI- and atlas-based) for the automated segmentation of 16 cardiac substructures in CT scans (with and without contrast).",
    "modalities": [
      "CT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Chest"
    ],
    "segments": [
      "HEART",
      "AORTA",
      "LEFT_ATRIUM",
      "LEFT_VENTRICLE",
      "RIGHT_ATRIUM",
      "RIGHT_VENTRICLE",
      "CORONARY_ARTERY_LAD",
      "CORONARY_ARTERY_CFLX",
      "CORONARY_ARTERY_RIGHT",
      "PULMONARY_ARTERY",
      "SINOTRIAL_NODE",
      "TRICUSPID_VALVE",
      "PULMONARY_VALVE",
      "MITRAL_VALVE",
      "SUPERIOR_VENA_CAVA",
      "AORTIC_VALVE",
      "ATRIOVENTRICULAR_NODE"
    ],
    "websiteUrl": "https://mhub.ai/models/platipy",
    "githubUrl": "https://github.com/pyplati/platipy",
    "cite": "Finnegan, R.N., Chin, V., Chlap, P. et al. Open-source, fully-automated hybrid cardiac substructure segmentation: development and optimisation. Phys Eng Sci Med 46, 377–393 (2023).",
    "citeUrl": "https://link.springer.com/article/10.1007/s13246-023-01231-w"
  },
  {
    "name": "nnunet_prostate_zonal_task05",
    "label": "Prostate Transitional and Peripheral Zone Seg (nnU-Net)",
    "description": "nnU-Net's zonal prostate segmentation model is a multi-modality input AI-based pipeline for the automated segmentation of the peripheral and transition zone of the prostate on MRI scans.",
    "modalities": [
      "MR"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Prostate"
    ],
    "segments": [
      "PROSTATE_PERIPHERAL_ZONE",
      "PROSTATE_TRANSITION_ZONE"
    ],
    "websiteUrl": "https://mhub.ai/models/nnunet_prostate_zonal_task05",
    "githubUrl": "https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1",
    "cite": "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.",
    "citeUrl": "https://www.nature.com/articles/s41592-020-01008-z"
  },
  {
    "name": "monai_prostate158",
    "label": "Prostate Transitional and Peripheral Zone Seg (Prostate158)",
    "description": "Prostate158 is a zonal prostate segmentation model, a multi-modality input AI-based pipeline for the automated segmentation of the peripheral and central gland of the prostate on MRI T2 axial scans.",
    "modalities": [
      "MR"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Prostate"
    ],
    "segments": [
      "PROSTATE_TRANSITION_ZONE",
      "PROSTATE_PERIPHERAL_ZONE"
    ],
    "websiteUrl": "https://mhub.ai/models/monai_prostate158",
    "githubUrl": "https://github.com/Project-MONAI/model-zoo/tree/dev/models/prostate_mri_anatomy",
    "cite": "Lisa C. Adams and Marcus R. Makowski and Günther Engel and Maximilian Rattunde and Felix Busch and Patrick Asbach and Stefan M. Niehues and Shankeeth Vinayahalingam and Bram {van Ginneken} and Geert Litjens and Keno K. Bressem, Prostate158 - An expert-annotated 3T MRI dataset and algorithm for prostate cancer detection",
    "citeUrl": "https://doi.org/10.1016/j.compbiomed.2022.105817"
  },
  {
    "name": "gc_lunglobes",
    "label": "Pulmonary Lobes Seg (RTSU-Net)",
    "description": "RTSU-Net is an AI-based pipeline for the automated pulmonary lobes segmentation in CT (robust to the presence of severe pathologies).",
    "modalities": [
      "CT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Chest"
    ],
    "segments": [
      "LEFT_UPPER_LUNG_LOBE",
      "LEFT_LOWER_LUNG_LOBE",
      "RIGHT_UPPER_LUNG_LOBE",
      "RIGHT_MIDDLE_LUNG_LOBE",
      "RIGHT_LOWER_LUNG_LOBE"
    ],
    "websiteUrl": "https://mhub.ai/models/gc_lunglobes",
    "githubUrl": "https://github.com/DIAGNijmegen/bodyct-pulmonary-lobe-segmentation",
    "cite": "W. Xie, C. Jacobs, J. -P. Charbonnier and B. van Ginneken, 'Relational Modeling for Robust and Efficient Pulmonary Lobe Segmentation in CT Scans,' in IEEE Transactions on Medical Imaging, vol. 39, no. 8, pp. 2664-2675, Aug. 2020, doi: 10.1109/TMI.2020.2995108.",
    "citeUrl": "https://doi.org/10.1109/TMI.2020.2995108"
  },
  {
    "name": "pyradiomics",
    "label": "PyRadiomics",
    "description": "Run PyRadiomics directly on DICOM files with MHub.",
    "modalities": [
      "CT",
      "SEG"
    ],
    "category": "Prediction",
    "bodyParts": [
      "WHOLEBODY"
    ],
    "segments": [],
    "websiteUrl": "https://mhub.ai/models/pyradiomics",
    "githubUrl": "https://github.com/AIM-Harvard/pyradiomics",
    "cite": "van Griethuysen JJM, Fedorov A, Parmar C, Hosny A, Aucoin N, Narayan V, Beets-Tan RGH, Fillion-Robin JC, Pieper S, Aerts HJWL. Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Res. 2017 Nov 1;77(21):e104-e107. doi: 10.1158/0008-5472.CAN-17-0339. PMID: 29092951; PMCID: PMC5672828.",
    "citeUrl": "https://doi.org/10.1158/0008-5472.CAN-17-0339"
  },
  {
    "name": "gc_autopet_fpr",
    "label": "Second place in AutoPET challenge: False Positive Reduction Network",
    "description": "This algorithm segments tumor lesions in whole-body FDG-PET/CT. To prevent over-segmenting tumor regions due to high-uptake organs or inflammations, the model uses a false positive reduction network. First a self-supervised pre-trained global segmentation module coarsely delineates the candidate tumor regions using a self-supervised pre-trained encoder. The candidate tumor regions are then refined by removing false positives via a local refinement module.",
    "modalities": [
      "PT",
      "CT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "WHOLEBODY"
    ],
    "segments": [
      "BACKGROUND",
      "NEOPLASM_MALIGNANT_PRIMARY"
    ],
    "websiteUrl": "https://mhub.ai/models/gc_autopet_fpr",
    "githubUrl": "https://github.com/YigePeng/AutoPET_False_Positive_Reduction",
    "cite": "Y. Peng, J. Kim, D. Feng and L. Bi, 'Automatic Tumor Segmentation via False Positive Reduction Network for Whole-Body Multi-Modal PET/CT Images', In arXiv:2209.07705, doi: 10.48550/arXiv:2209.07705.",
    "citeUrl": "https://arxiv.org/abs/2209.07705"
  },
  {
    "name": "msk_smit_lung_gtv",
    "label": "SMIT Self-supervised Lung GTV Segmentation",
    "description": "A Lung GTV segmentation model, fine-tuned from a foundation model pretrained with 10K CT scans",
    "modalities": [
      "CT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Chest"
    ],
    "segments": [
      "LUNG+NEOPLASM_MALIGNANT_PRIMARY"
    ],
    "websiteUrl": "https://mhub.ai/models/msk_smit_lung_gtv",
    "githubUrl": "https://github.com/The-Veeraraghavan-Lab/CTRobust_Transformers.git",
    "cite": "Jiang, Jue, and Harini Veeraraghavan. Self-supervised pretraining in the wild imparts image acquisition robustness to medical image transformers: an application to lung cancer segmentation. Proceedings of machine learning research 250 (2024): 708.",
    "citeUrl": "https://openreview.net/pdf?id=G9Te2IevNm"
  },
  {
    "name": "gc_spider_baseline",
    "label": "Spine Seg (SPIDER Baseline)",
    "description": "This model segments three anatomical structures in lumbar spine MRI: vertebrae, intervertebral discs (IVDs), and spinal canal. Each vertebrae and IVD is labeled according to standard naming convention.",
    "modalities": [
      "MR"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Spine"
    ],
    "segments": [
      "VERTEBRAE_L5",
      "VERTEBRAE_L4",
      "VERTEBRAE_L3",
      "VERTEBRAE_L2",
      "VERTEBRAE_L1",
      "VERTEBRAE_T12",
      "VERTEBRAE_T11",
      "VERTEBRAE_T10",
      "VERTEBRAE_T9",
      "VERTEBRAE_T8",
      "VERTEBRAE_T7",
      "VERTEBRAE_T6",
      "VERTEBRAE_T5",
      "VERTEBRAE_T4",
      "VERTEBRAE_T3",
      "VERTEBRAE_T2",
      "VERTEBRAE_T1",
      "VERTEBRAE_C7",
      "VERTEBRAE_C6",
      "VERTEBRAE_C5",
      "VERTEBRAE_C4",
      "VERTEBRAE_C3",
      "VERTEBRAE_C2",
      "VERTEBRAE_C1",
      "VERTEBRAE_DISK_L5S1",
      "VERTEBRAE_DISK_L4L5",
      "VERTEBRAE_DISK_L3L4",
      "VERTEBRAE_DISK_L2L3",
      "VERTEBRAE_DISK_L1L2",
      "VERTEBRAE_DISK_T12L1",
      "VERTEBRAE_DISK_T11T12",
      "VERTEBRAE_DISK_T10T11",
      "VERTEBRAE_DISK_T9T10",
      "VERTEBRAE_DISK_T8T9",
      "VERTEBRAE_DISK_T7T8",
      "VERTEBRAE_DISK_T6T7",
      "VERTEBRAE_DISK_T5T6",
      "VERTEBRAE_DISK_T4T5",
      "VERTEBRAE_DISK_T3T4",
      "VERTEBRAE_DISK_T2T3",
      "VERTEBRAE_DISK_T1T2",
      "VERTEBRAE_DISK_C7T1",
      "VERTEBRAE_DISK_C6C7",
      "VERTEBRAE_DISK_C5C6",
      "VERTEBRAE_DISK_C4C5",
      "VERTEBRAE_DISK_C3C4",
      "VERTEBRAE_DISK_C2C3",
      "SPINAL_CANAL"
    ],
    "websiteUrl": "https://mhub.ai/models/gc_spider_baseline",
    "githubUrl": "https://github.com/DIAGNijmegen/SPIDER-Baseline-IIS",
    "cite": "van der Graaf, J.W., van Hooff, M.L., Buckens, C.F.M. et al. Lumbar spine segmentation in MR images: a dataset and a public benchmark. Sci Data 11, 264 (2024). doi: 10.1038/s41597-024-03090-w",
    "citeUrl": "https://doi.org/10.1038/s41597-024-03090-w"
  },
  {
    "name": "gc_stoic_baseline",
    "label": "STOIC2021 baseline",
    "description": "This model predicts the probability and severity of COVID-19 in lung-CT scans. Severe COVID-19 is defined as patient intubation or death within one month from the acquisition of the CT scan.",
    "modalities": [
      "CT"
    ],
    "category": "Prediction",
    "bodyParts": [
      "Chest"
    ],
    "segments": [],
    "websiteUrl": "https://mhub.ai/models/gc_stoic_baseline",
    "githubUrl": "https://github.com/luukboulogne/stoic2021-baseline",
    "cite": "Luuk H. Boulogne, Julian Lorenz, Daniel Kienzle, Robin Schon, Katja Ludwig, Rainer Lienhart, Simon Jegou, Guang Li, Cong Chen, Qi Wang, Derik Shi, Mayug Maniparambil, Dominik Muller, Silvan Mertes, Niklas Schroter, Fabio Hellmann, Miriam Elia, Ine Dirks, Matias Nicolas Bossa, Abel Diaz Berenguer, Tanmoy Mukherjee, Jef Vandemeulebroucke, Hichem Sahli, Nikos Deligiannis, Panagiotis Gonidakis, Ngoc Dung Huynh, Imran Razzak, Reda Bouadjenek, Mario Verdicchio, Pasquale Borrelli, Marco Aiello, James A. Meakin, Alexander Lemm, Christoph Russ, Razvan Ionasec, Nikos Paragios, Bram van Ginneken, Marie-Pierre Revel Dubois, The STOIC2021 COVID-19 AI challenge: applying reusable training methodologies to private data, arXiv:2306.10484, 2023, https://arxiv.org/abs/2306.10484v2",
    "citeUrl": "https://arxiv.org/abs/2306.10484v2"
  },
  {
    "name": "nnunet_segthor",
    "label": "Thoracic OAR (nnU-Net)",
    "description": "nnU-Net's thoracic OAR segmentation model is an AI-based pipeline for the automated segmentation of the heart, the aorta, the esophagus and the trachea in CT scans (with and without contrast).",
    "modalities": [
      "CT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Chest"
    ],
    "segments": [
      "ESOPHAGUS",
      "HEART",
      "TRACHEA",
      "AORTA"
    ],
    "websiteUrl": "https://mhub.ai/models/nnunet_segthor",
    "githubUrl": "https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1",
    "cite": "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.",
    "citeUrl": "https://www.nature.com/articles/s41592-020-01008-z"
  },
  {
    "name": "gc_tiger_lb2",
    "label": "TIGER challenge winner: Team VUNO",
    "description": "This algorithm predicts the percentage of stromal area covered by tumour infiltrating lymphocytes (TIL) on H&E-stained whole-slide images of breast cancer histopathology. This algorithm first segments invasive tumor and tumor-associated stroma and subsequently detects lymphocytes and plasma cells, which are finally combined to estimate the relevant regions for TIL estimation. This model algorithm was the challenge winner of the TIGER challenge hosted on Grand Challenge.",
    "modalities": [
      "SM"
    ],
    "category": "Prediction",
    "bodyParts": [
      "Breast"
    ],
    "segments": [],
    "websiteUrl": "https://mhub.ai/models/gc_tiger_lb2",
    "githubUrl": "https://github.com/vuno/tiger_challenge",
    "citeUrl": "https://github.com/vuno/tiger_challenge/blob/720f8dfca4624792c8e57915c4222efec5a0c2d4/figure/method_description.pdf"
  },
  {
    "name": "gc_wsi_bgseg",
    "label": "Tissue-Background segmentation in histopathological whole-slide images",
    "description": "This algorithm segments the background and tissue in histopathological whole-slide images.",
    "modalities": [
      "SM"
    ],
    "category": "Prediction",
    "bodyParts": [
      "WHOLEBODY"
    ],
    "segments": [],
    "websiteUrl": "https://mhub.ai/models/gc_wsi_bgseg",
    "githubUrl": "https://github.com/DIAGNijmegen/pathology-tissue-background-segmentation-processor",
    "cite": "Bándi P, Balkenhol M, van Ginneken B, van der Laak J, Litjens G. 2019. Resolution-agnostic tissue segmentation in whole-slide histopathology images with convolutional neural networks. PeerJ 7:e8242",
    "citeUrl": "https://peerj.com/articles/8242/"
  },
  {
    "name": "totalsegmentator",
    "label": "TotalSegmentator",
    "description": "TotalSegmentator is an AI-based pipeline for the segmentation of 104 anatomical structures in CT images (with and without contrast).",
    "modalities": [
      "CT"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "WHOLEBODY"
    ],
    "segments": [
      "SPLEEN",
      "RIGHT_KIDNEY",
      "LEFT_KIDNEY",
      "GALLBLADDER",
      "LIVER",
      "STOMACH",
      "AORTA",
      "INFERIOR_VENA_CAVA",
      "PORTAL_AND_SPLENIC_VEIN",
      "PANCREAS",
      "RIGHT_ADRENAL_GLAND",
      "LEFT_ADRENAL_GLAND",
      "LEFT_UPPER_LUNG_LOBE",
      "LEFT_LOWER_LUNG_LOBE",
      "RIGHT_UPPER_LUNG_LOBE",
      "RIGHT_MIDDLE_LUNG_LOBE",
      "RIGHT_LOWER_LUNG_LOBE",
      "VERTEBRAE_L5",
      "VERTEBRAE_L4",
      "VERTEBRAE_L3",
      "VERTEBRAE_L2",
      "VERTEBRAE_L1",
      "VERTEBRAE_T12",
      "VERTEBRAE_T11",
      "VERTEBRAE_T10",
      "VERTEBRAE_T9",
      "VERTEBRAE_T8",
      "VERTEBRAE_T7",
      "VERTEBRAE_T6",
      "VERTEBRAE_T5",
      "VERTEBRAE_T4",
      "VERTEBRAE_T3",
      "VERTEBRAE_T2",
      "VERTEBRAE_T1",
      "VERTEBRAE_C7",
      "VERTEBRAE_C6",
      "VERTEBRAE_C5",
      "VERTEBRAE_C4",
      "VERTEBRAE_C3",
      "VERTEBRAE_C2",
      "VERTEBRAE_C1",
      "ESOPHAGUS",
      "TRACHEA",
      "MYOCARDIUM",
      "LEFT_ATRIUM",
      "LEFT_VENTRICLE",
      "RIGHT_ATRIUM",
      "RIGHT_VENTRICLE",
      "PULMONARY_ARTERY",
      "BRAIN",
      "LEFT_ILIAC_ARTERY",
      "RIGHT_ILIAC_ARTERY",
      "LEFT_ILIAC_VEIN",
      "RIGHT_ILIAC_VEIN",
      "SMALL_INTESTINE",
      "DUODENUM",
      "COLON",
      "LEFT_RIB_1",
      "LEFT_RIB_2",
      "LEFT_RIB_3",
      "LEFT_RIB_4",
      "LEFT_RIB_5",
      "LEFT_RIB_6",
      "LEFT_RIB_7",
      "LEFT_RIB_8",
      "LEFT_RIB_9",
      "LEFT_RIB_10",
      "LEFT_RIB_11",
      "LEFT_RIB_12",
      "RIGHT_RIB_1",
      "RIGHT_RIB_2",
      "RIGHT_RIB_3",
      "RIGHT_RIB_4",
      "RIGHT_RIB_5",
      "RIGHT_RIB_6",
      "RIGHT_RIB_7",
      "RIGHT_RIB_8",
      "RIGHT_RIB_9",
      "RIGHT_RIB_10",
      "RIGHT_RIB_11",
      "RIGHT_RIB_12",
      "LEFT_HUMERUS",
      "RIGHT_HUMERUS",
      "LEFT_SCAPULA",
      "RIGHT_SCAPULA",
      "LEFT_CLAVICLE",
      "RIGHT_CLAVICLE",
      "LEFT_FEMUR",
      "RIGHT_FEMUR",
      "LEFT_HIP",
      "RIGHT_HIP",
      "SACRUM",
      "FACE",
      "LEFT_GLUTEUS_MAXIMUS",
      "RIGHT_GLUTEUS_MAXIMUS",
      "LEFT_GLUTEUS_MEDIUS",
      "RIGHT_GLUTEUS_MEDIUS",
      "LEFT_GLUTEUS_MINIMUS",
      "RIGHT_GLUTEUS_MINIMUS",
      "LEFT_AUTOCHTHONOUS_BACK_MUSCLE",
      "RIGHT_AUTOCHTHONOUS_BACK_MUSCLE",
      "LEFT_ILIOPSOAS",
      "RIGHT_ILIOPSOAS",
      "URINARY_BLADDER"
    ],
    "websiteUrl": "https://mhub.ai/models/totalsegmentator",
    "githubUrl": "https://github.com/wasserth/TotalSegmentator",
    "cite": "Jakob Wasserthal, Hanns-Christian Breit, Manfred T. Meyer, Maurice Pradella, Daniel Hinck, Alexander W. Sauter, Tobias Heye, Daniel T. Boll, Joshy Cyriac, Shan Yang, Michael Bach, and Martin Segeroth (2023). TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology Artificial Intelligence, 5:5",
    "citeUrl": "https://pubs.rsna.org/doi/10.1148/ryai.230024"
  },
  {
    "name": "nnunet_prostate_task24",
    "label": "Whole Prostate Seg (nnU-Net)",
    "description": "nnU-Net's whole prostate segmentation model is a single-modality (i.e. T2) input AI-based pipeline for the automated segmentation of the whole prostate on MRI scans.",
    "modalities": [
      "MR"
    ],
    "category": "Segmentation",
    "bodyParts": [
      "Prostate"
    ],
    "segments": [
      "PROSTATE"
    ],
    "websiteUrl": "https://mhub.ai/models/nnunet_prostate_task24",
    "githubUrl": "https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1",
    "cite": "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.",
    "citeUrl": "https://www.nature.com/articles/s41592-020-01008-z"
  }
]);

function modalityTokens(entry: string): string[] {
  return String(entry || "")
    .split("|")
    .map((t) => t.trim().toUpperCase())
    .filter(Boolean)
    .map((t) => (t === "MRI" ? "MR" : t === "PET" ? "PT" : t));
}

function modelMatchesModality(model: MhubModelDef, modality: string): boolean {
  const want = modality.trim().toUpperCase();
  if (!want) return true;
  const normalizedWant = want === "MRI" ? "MR" : want === "PET" ? "PT" : want;
  return model.modalities.some((m) => modalityTokens(m).includes(normalizedWant));
}

/** Filter skill models by open-study modality. Null/empty → all models. */
export function listMhubModelsForModality(
  modality: string | null | undefined,
): MhubModelDef[] {
  const raw = String(modality || "").trim();
  if (!raw) return [...MHUB_MODELS];
  return MHUB_MODELS.filter((m) => modelMatchesModality(m, raw));
}

function canonBodyPartKey(raw: string): string {
  return String(raw || "")
    .trim()
    .toUpperCase()
    .replace(/[\s_-]+/g, "");
}

/** Unique body-part options for a model list (sorted, WHOLEBODY last). */
export function listMhubBodyPartsForModels(
  models: readonly MhubModelDef[],
): string[] {
  const parts = new Set<string>();
  for (const model of models) {
    for (const bp of model.bodyParts || []) {
      const t = String(bp || "").trim();
      if (t) parts.add(t);
    }
  }
  return [...parts].sort((a, b) => {
    if (a === "WHOLEBODY") return 1;
    if (b === "WHOLEBODY") return -1;
    return a.localeCompare(b);
  });
}

/** Filter models that declare the given body part examined. */
export function listMhubModelsForBodyPart(
  models: readonly MhubModelDef[],
  bodyPart: string | null | undefined,
): MhubModelDef[] {
  const want = String(bodyPart || "").trim();
  if (!want) return [...models];
  const wantKey = canonBodyPartKey(want);
  return models.filter((m) =>
    (m.bodyParts || []).some((bp) => canonBodyPartKey(bp) === wantKey),
  );
}

/** Normalize anatomy search terms like mhub_helper.find_models_for_anatomy. */
export function normalizeMhubAnatomyTerms(
  query: string | null | undefined,
): string[] {
  return String(query || "")
    .trim()
    .split(/[\s,;+/]+/)
    .map((t) => t.trim().toUpperCase().replace(/[\s-]+/g, "_"))
    .filter(Boolean);
}

/**
 * Find models whose segment IDs contain any anatomy term (skill `find`).
 * Sorted by number of matched segments (descending).
 */
export function findMhubModelsForAnatomy(
  query: string | null | undefined,
  models: readonly MhubModelDef[] = MHUB_MODELS,
): MhubModelDef[] {
  const terms = normalizeMhubAnatomyTerms(query);
  if (!terms.length) return [...models];
  const scored: { model: MhubModelDef; hits: number }[] = [];
  for (const model of models) {
    const segments = (model.segments || []).map((s) => String(s).toUpperCase());
    if (!segments.length) continue;
    let hits = 0;
    for (const seg of segments) {
      if (terms.some((t) => seg.includes(t))) hits += 1;
    }
    if (hits > 0) scored.push({ model, hits });
  }
  scored.sort((a, b) => b.hits - a.hits || a.model.label.localeCompare(b.model.label));
  return scored.map((s) => s.model);
}

/** Known body-part tokens for inferring from study description / series text. */
const BODY_PART_HINTS: { key: string; label: string; patterns: RegExp[] }[] = [
  { key: "KIDNEY", label: "Kidney", patterns: [/\bkidn(?:ey|eys)?\b/i, /\brenal\b/i] },
  { key: "LIVER", label: "Liver", patterns: [/\bliver\b/i, /\bhepatic\b/i] },
  { key: "LUNG", label: "Lung", patterns: [/\blung(?:s)?\b/i, /\bpulmon/i] },
  { key: "CHEST", label: "Chest", patterns: [/\bchest\b/i, /\bthorax\b/i, /\bthoracic\b/i] },
  { key: "ABDOMEN", label: "Abdomen", patterns: [/\babdomen\b/i, /\babdominal\b/i] },
  { key: "PROSTATE", label: "Prostate", patterns: [/\bprostate\b/i] },
  { key: "BREAST", label: "Breast", patterns: [/\bbreast\b/i, /\bmammo/i] },
  { key: "SPINE", label: "Spine", patterns: [/\bspine\b/i, /\bspinal\b/i, /\bvertebra/i] },
  { key: "WHOLEBODY", label: "WHOLEBODY", patterns: [/\bwhole\s*body\b/i, /\bwb\b/i] },
];

/**
 * Infer a canonical body-part label from free text (study description, etc.).
 * Returns "" when no confident match.
 */
export function inferMhubBodyPartFromText(
  text: string | null | undefined,
): string {
  const raw = String(text || "").trim();
  if (!raw) return "";
  for (const hint of BODY_PART_HINTS) {
    if (hint.patterns.some((re) => re.test(raw))) return hint.label;
  }
  return "";
}

/** Extract ImagingStudy description + modality coding from FHIRcast context. */
export function mhubContextHintsFromImagingStudy(
  context: unknown[] | null | undefined,
): { modality: string; description: string; bodyPart: string } {
  let modality = "";
  let description = "";
  if (!Array.isArray(context)) {
    return { modality, description, bodyPart: "" };
  }

  const normalizeMod = (raw: string): string => {
    let m = String(raw || "").trim().toUpperCase();
    if (m === "MRI") m = "MR";
    if (m === "PET") m = "PT";
    if (m === "CT" || m === "CTAC") return "CT";
    if (m === "MR" || m === "PT" || m === "NM") return m === "NM" ? "PT" : m;
    return "";
  };

  const modalityFromCoding = (value: unknown): string => {
    if (!value) return "";
    if (typeof value === "string") return normalizeMod(value);
    if (Array.isArray(value)) {
      for (const entry of value) {
        const hit = modalityFromCoding(entry);
        if (hit) return hit;
      }
      return "";
    }
    if (typeof value === "object") {
      const obj = value as { code?: unknown; coding?: unknown; text?: unknown };
      if (obj.coding != null) return modalityFromCoding(obj.coding);
      if (obj.code != null) return normalizeMod(String(obj.code));
      if (obj.text != null) return normalizeMod(String(obj.text));
    }
    return "";
  };

  for (const item of context) {
    if (!item || typeof item !== "object") continue;
    const key = String((item as { key?: unknown }).key || "");
    const resource = (item as { resource?: unknown }).resource;
    if (!resource || typeof resource !== "object") continue;
    const res = resource as {
      modality?: unknown;
      description?: unknown;
      series?: unknown;
      files?: unknown;
    };
    if (!description && res.description != null) {
      description = String(res.description).trim();
    }
    if (!modality && res.modality != null) {
      modality = modalityFromCoding(res.modality);
    }
    if (!modality && Array.isArray(res.series)) {
      for (const series of res.series) {
        if (!series || typeof series !== "object") continue;
        modality = modalityFromCoding(
          (series as { modality?: unknown }).modality,
        );
        if (modality) break;
      }
    }
    // IDC builders put modality on volume file labels (role: volume), not study.modality.
    if (!modality && (key === "files" || Array.isArray(res.files))) {
      const files = Array.isArray(res.files) ? res.files : [];
      for (const file of files) {
        if (!file || typeof file !== "object") continue;
        const row = file as { role?: unknown; label?: unknown };
        const role = String(row.role || "").trim().toLowerCase();
        if (role && role !== "volume") continue;
        modality = normalizeMod(String(row.label || ""));
        if (modality) break;
      }
    }
  }
  const bodyPart = inferMhubBodyPartFromText(description);
  return { modality, description, bodyPart };
}
