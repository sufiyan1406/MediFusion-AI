# MediFusion Machine Learning Methodology

## 1. Problem Formulation: Multi-Label Binary Classification
Chest X-ray interpretation is fundamentally a **Multi-Label Classification** problem, not a single-class or multi-class problem.

### Why Multi-Label?
In a standard multi-class problem (e.g. classifying digits 0-9), each image belongs to exactly **one** category. However, a patient's lungs can simultaneously suffer from multiple conditions at the exact same time—for instance, a patient can have both *Cardiomegaly* (enlarged heart) and *Effusion* (fluid in the pleural space), or *Infiltration* alongside *Atelectasis*. 

Therefore, each input image can be assigned:
- Zero diseases (`No Finding`)
- Exactly one disease
- Multiple co-occurring diseases (up to 9 pathologies observed on a single scan in this dataset)

---

## 2. Model Output Architecture & The 14 Pathology Neurons
Our neural network output layer contains **14 output neurons**, corresponding exactly to the 14 target pathology conditions in canonical order:
1. `Atelectasis`
2. `Cardiomegaly`
3. `Effusion`
4. `Infiltration`
5. `Mass`
6. `Nodule`
7. `Pneumonia`
8. `Pneumothorax`
9. `Consolidation`
10. `Edema`
11. `Emphysema`
12. `Fibrosis`
13. `Pleural_Thickening`
14. `Hernia`

### Why `No Finding` is NOT an Output Neuron
- `No Finding` indicates a completely normal X-ray scan with zero detected abnormalities.
- If we added `No Finding` as a 15th output neuron, it would create conflicting signals because `No Finding` is mathematically the logical negation of all 14 pathology neurons.
- Instead, a normal scan is cleanly represented when **all 14 pathology output probabilities remain near 0.0** (target vector: `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`).
- Each of the 14 output neurons produces an unnormalized raw logit $z_k$. Sigmoid activation $\sigma(z_k) = \frac{1}{1 + e^{-z_k}}$ is applied independently to each neuron, treating each disease as an independent binary classification question.

---

## 3. Implemented Baseline CNN Architecture (`MedCXRNet`)

To satisfy academic requirements (training from scratch with zero pre-trained ImageNet weights), we implemented **`MedCXRNet`**, a compact 4-stage Convolutional Neural Network:

```
Input (1 × 224 × 224 Grayscale Image)
  │
  ├── Stage 1: Conv2d(1 -> 32) ──► BatchNorm ──► ReLU ──► MaxPool2d(2×2)  => [32 × 112 × 112]
  ├── Stage 2: Conv2d(32 -> 64) ──► BatchNorm ──► ReLU ──► MaxPool2d(2×2)  => [64 × 56 × 56]
  ├── Stage 3: Conv2d(64 -> 128) ──► BatchNorm ──► ReLU ──► MaxPool2d(2×2) => [128 × 28 × 28]
  ├── Stage 4: Conv2d(128 -> 256) ──► BatchNorm ──► ReLU ──► MaxPool2d(2×2)=> [256 × 14 × 14]
  │
  ├── Global Average Pooling: AdaptiveAvgPool2d(1, 1)                      => [256 × 1 × 1]
  ├── Flatten                                                               => [256]
  ├── Dropout(p=0.30)                                                       => [256]
  └── Classifier: Linear(256 -> 14)                                         => [14 Raw Logits]
```

- **Weight Initialization:** Kaiming He Normal (`nn.init.kaiming_normal_`) for Conv2d and Linear layers; constant initialization for BatchNorm (`weight=1.0`, `bias=0.0`).
- **Parameter Efficiency:** ~2,105,000 parameters (~8.4 MB), allowing rapid scratch training on 4GB VRAM local GPUs (RTX 2050).

---

## 4. Baseline Training Configuration

| Parameter | Configuration | Justification / Purpose |
| :--- | :--- | :--- |
| **Loss Function** | `BCEWithLogitsLoss()` | Combines Sigmoid + BCE using log-sum-exp trick for numerical stability. |
| **Optimizer** | `AdamW` | Weight decay regularized Adam optimizer. |
| **Learning Rate** | `1e-4` (0.0001) | Standard stable learning rate for Kaiming He initialized scratch CNNs. |
| **Weight Decay** | `1e-4` (0.0001) | Regularizes weight norms, preventing overfitting. |
| **Batch Size** | `16` | Fits comfortably within local RTX 2050 4GB VRAM footprint. |
| **Max Epochs** | `15` | Baseline training epochs. |
| **LR Scheduler** | `ReduceLROnPlateau` | Factor `0.5`, Patience `2`, monitoring Validation Macro ROC-AUC. |
| **Early Stopping** | Patience `4` | Halts training if Validation Macro ROC-AUC stops improving. |
| **Mixed Precision** | `torch.cuda.amp` | Accelerates GPU training and cuts VRAM usage by ~40–50%. |

---

## 5. Patient-Level Data Splitting & Leakage Prevention
A critical pitfall in medical image deep learning is **Data Leakage across Patient IDs**.

### Why Patient-Level Grouping Matters
- In the NIH ChestX-ray14 dataset, 11,885 patients (~38.6%) have multiple X-ray scans taken over time (ranging from 2 up to 184 scans per patient).
- If scans are randomly assigned to train and test sets by image filename, scans from the **same patient** end up in both training and validation/test sets. The model memorizes patient-specific anatomical features rather than generalizable pathology representations.
- **Our Solution:** We enforce strict **Patient-Level Grouping**. All scans belonging to a single `Patient ID` are assigned exclusively to either Train (73,916 images / 23,806 patients), Validation (12,608 images / 4,202 patients), or Test (25,596 images / 2,797 patients)—**0 patient overlap**.

---

## 6. Preprocessing & Augmentation Strategy

1. **Single-Channel Grayscale Conversion (`.convert('L')`):**
   Converts X-rays to 1-channel intensity maps, avoiding artificial 3-channel RGB memory overhead.
2. **Resizing ($224 \times 224$ pixels):**
   Downsamples $1024 \times 1024$ raw scans to $224 \times 224$, reducing memory footprint by ~20×.
3. **Training Augmentation:**
   - Random Rotation ($\pm 7^\circ$)
   - Random Affine Translation ($\pm 4\%$)
   - ColorJitter Brightness & Contrast ($\pm 10\%$)
   - **Horizontal Flip is strictly disabled** to avoid mirroring thoracic anatomy.
4. **Validation/Testing Transformation:**
   - Deterministic resizing, tensor conversion, and normalization `(input - 0.5) / 0.5`.

---

## 7. Evaluation & Decision Thresholding

- **Primary Metric:** **Macro ROC-AUC** across the 14 pathology classes (unweighted mean AUC).
- **Secondary Metrics:** Per-Class ROC-AUC, Precision, Recall (Sensitivity), Specificity, F1-Score, PR-AUC.
- **Decision Threshold:** Baseline threshold $T = 0.5$ converts probabilities to binary predictions. Threshold tuning per pathology is reserved for post-baseline experiments.
