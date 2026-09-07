# MediFusion Dataset Inspection & Profiling Documentation

## 1. Dataset Overview
- **Dataset Name:** NIH ChestX-ray14 (NIH Clinical Center)
- **Local Directory Location:** `d:\MediFusion\DataSet` (and raw data archive)
- **Total Raw Storage Size:** ~45 GB
- **Domain:** Deep Learning-Based Chest X-Ray Disease/Abnormality Multi-Label Classification
- **Primary Function:** Diagnostic screening of 14 common chest pathology findings from frontal-view chest radiographs.

---

## 2. Directory Structure
```
d:\MediFusion\DataSet/
├── ARXIV_V5_CHESTXRAY.pdf       # NIH ChestX-ray14 research paper reference
├── FAQ_CHESTXRAY.pdf            # Frequently asked questions document
├── LOG_CHESTXRAY.pdf            # Data extraction log file
├── README_CHESTXRAY.pdf         # Official NIH release notes & license
├── Data_Entry_2017.csv          # Primary metadata file (112,120 rows)
├── BBox_List_2017.csv           # Ground-truth bounding box coordinates (984 rows)
├── train_val_list.txt           # Official train/validation image split list (86,524 images)
├── test_list.txt                # Official test image split list (25,596 images)
├── images_001/                  # Image batch directory 001 (subfolder `images/`)
├── images_002/                  # Image batch directory 002
├── images_003/                  # Image batch directory 003
├── images_004/                  # Image batch directory 004
├── images_005/                  # Image batch directory 005
├── images_006/                  # Image batch directory 006
├── images_007/                  # Image batch directory 007
├── images_008/                  # Image batch directory 008
├── images_009/                  # Image batch directory 009
├── images_010/                  # Image batch directory 010
├── images_011/                  # Image batch directory 011
└── images_012/                  # Image batch directory 012
```

---

## 3. Metadata Files Inspection

### A. `Data_Entry_2017.csv`
- **Total Image Records:** 112,120
- **Total Unique Patients:** 30,805
- **Columns (12 total):**
  1. `Image Index` (string): Image filename (e.g., `00000001_000.png`). Key join field for images.
  2. `Finding Labels` (string): Pipe-separated disease labels (e.g., `Cardiomegaly|Emphysema` or `No Finding`).
  3. `Follow-up #` (int64): Follow-up scan number for patient.
  4. `Patient ID` (int64): Unique patient identifier (e.g., `1`).
  5. `Patient Age` (int64/string): Age of patient at scan time (formatted as `058Y`, `060Y`).
  6. `Patient Gender` (string): Gender (`M` or `F`).
  7. `View Position` (string): Radiographic view (`PA` - Posterior-Anterior or `AP` - Anterior-Posterior).
  8. `OriginalImage[Width` (int64): Original scan width in pixels (e.g., `2500`).
  9. `Height]` (int64): Original scan height in pixels (e.g., `2048`).
  10. `OriginalImagePixelSpacing[x` (float64): Physical pixel spacing X-axis (mm).
  11. `y]` (float64): Physical pixel spacing Y-axis (mm).
  12. `Unnamed: 11` (float64): Unused empty column artifact.

### B. `BBox_List_2017.csv`
- **Total Ground-Truth Bounding Boxes:** 984
- **Columns (9 total):** `Image Index`, `Finding Label`, `Bbox [x`, `y`, `w`, `h]`, `Unnamed: 6`, `Unnamed: 7`, `Unnamed: 8`.
- **Pathologies Covered:** Atelectasis (180), Effusion (153), Cardiomegaly (146), Infiltrate (123), Pneumonia (120), Pneumothorax (98), Mass (85), Nodule (79).

---

## 4. Image Information & Quality Inspection
- **Total PNG Images on Disk:** 112,120 (100% matched with `Data_Entry_2017.csv`, 0 missing).
- **Filename Uniqueness:** 112,120 unique filenames (0 duplicate filenames across subdirectories).
- **Image Resolution:** Uniform $1024 \times 1024$ pixels across sampled images.
- **Aspect Ratio:** $1:1$ square aspect ratio.
- **Color Channels & Modes:**
  - 98.6% of sampled images are single-channel 8-bit grayscale (`Mode: L`).
  - 1.4% of sampled images carry RGBA metadata (`Mode: RGBA`).
  - *Recommendation:* Explicitly convert images to 1-channel grayscale (`.convert('L')`) or 3-channel RGB (`.convert('RGB')`) during PyTorch tensor dataset initialization.
- **Image Integrity:** 0 corrupted images detected out of 2,122 inspected sample files. All files are readable via PIL/OpenCV.

---

## 5. Label Structure & Class Distribution

### Multi-Label Problem Formulation
This is a **Multi-Label Binary Classification** task. An individual X-ray can present with:
- `No Finding` (normal scan)
- 1 single pathology
- Multiple co-occurring pathologies (e.g., `Cardiomegaly|Effusion|Atelectasis`)

#### Number of Labels Per Image Breakdown:
- **1 Label:** 91,324 images (includes 60,361 `No Finding` + 30,963 single-disease scans)
- **2 Labels:** 14,306 images
- **3 Labels:** 4,856 images
- **4 Labels:** 1,247 images
- **5 Labels:** 301 images
- **6 Labels:** 67 images
- **7 Labels:** 16 images
- **8 Labels:** 1 image
- **9 Labels:** 2 images

### Label Frequency Table (All 15 Classes)

| Finding / Label | Image Count | Percentage of Dataset | Category Type |
| :--- | :--- | :--- | :--- |
| **No Finding** | 60,361 | 53.84% | Normal / Non-Pathological |
| **Infiltration** | 19,894 | 17.74% | Pathology Class 1 |
| **Effusion** | 13,317 | 11.88% | Pathology Class 2 |
| **Atelectasis** | 11,559 | 10.31% | Pathology Class 3 |
| **Nodule** | 6,331 | 5.65% | Pathology Class 4 |
| **Mass** | 5,782 | 5.16% | Pathology Class 5 |
| **Pneumothorax** | 5,302 | 4.73% | Severe / Triage Target |
| **Consolidation** | 4,667 | 4.16% | Severe / Triage Target |
| **Pleural_Thickening** | 3,385 | 3.02% | Pathology Class 8 |
| **Cardiomegaly** | 2,776 | 2.48% | Pathology Class 9 |
| **Emphysema** | 2,516 | 2.24% | Pathology Class 10 |
| **Edema** | 2,303 | 2.05% | Severe / Triage Target |
| **Fibrosis** | 1,686 | 1.50% | Pathology Class 12 |
| **Pneumonia** | 1,431 | 1.28% | Severe / Triage Target |
| **Hernia** | 227 | 0.20% | Rare Minority Class |

### Class Imbalance Analysis
- **Largest Pathology Class:** `Infiltration` (19,894 images, 17.74%).
- **Smallest Pathology Class:** `Hernia` (227 images, 0.20%).
- **Disease Class Imbalance Ratio:** **87.64×** between largest and smallest pathology classes.
- **Normal vs. Disease Ratio:** 53.84% `No Finding` vs. 46.16% with at least one pathology.

---

## 6. Patient Information & Data Leakage Analysis

### Patient Distribution
- **Unique Patients:** 30,805
- **Total Scans:** 112,120
- **Scans per Patient Statistics:**
  - Minimum: 1 scan
  - Maximum: 184 scans
  - Mean: 3.64 scans per patient
  - Median: 1.00 scan
  - **Patients with Multiple Scans (>1 image):** 11,885 patients (~38.58% of all unique patients).

### Major Data Leakage Concern: Patient ID Overlap
If standard random row splitting (`sklearn.model_selection.train_test_split`) is used without grouping by `Patient ID`:
- Images from the **same patient** taken at different follow-ups will exist in both training and validation/test sets.
- The model will memorize patient-specific anatomical features (bone structure, implants, body habitus) rather than learning generalizable pathology representations, inflating validation AUC unnaturally.

### Verification of Provided Split Files (`train_val_list.txt` & `test_list.txt`)
- **`train_val_list.txt`:** 86,524 images across **28,008 unique patients**.
- **`test_list.txt`:** 25,596 images across **2,797 unique patients**.
- **Sum of Splits:** $86,524 + 25,596 = 112,120$ images (100% of dataset).
- **Patient Overlap Check:** **0 patient overlap**.
  - `set(train_val_patients).intersection(set(test_patients)) == empty_set`.
  - The official provided split list strictly enforces patient-level separation out-of-the-box.

---

## 7. Key Initial Observations & Questions for Next Phase

### Initial Observations
1. **Raw Storage & Read-Only Requirement:** The dataset inside `DataSet` / `archive` is complete, uncorrupted, and must be treated as strictly read-only raw data.
2. **Classification Task Formulation:** Multi-label binary classification across 14 pathology outputs using Sigmoid activation per output neuron.
3. **Loss Function Requirement:** Standard Binary Cross-Entropy (BCE) will suffer on rare classes like `Hernia` (0.20%) and `Pneumonia` (1.28%). Loss functions like Weighted BCE (`pos_weight`) or Asymmetric Loss (ASL) are strongly indicated.
4. **Resolution Strategy:** Original scans are $1024 \times 1024$. Downsampling to $224 \times 224$ or $512 \times 512$ is required for memory efficiency on local GPU hardware (RTX 2050 4GB).

### Key Decisions Needed Before Training Strategy Selection
1. **Model Architecture:** Training custom compact architecture (`MedCXRNet`, ~2.1M params) from scratch vs. fine-tuning pre-trained weights (subject to academic constraints).
2. **Validation Strategy:** Further splitting `train_val_list.txt` (86,524 images) into Train and Validation subsets using `GroupKFold` or `GroupShuffleSplit` on `Patient ID` to guarantee 0 patient leakage in validation.
3. **Evaluation Metric:** Use Macro ROC-AUC across all 14 pathology classes (standard NIH ChestX-ray14 benchmark metric) rather than accuracy.
