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
- Each of the 14 output neurons produces an unnormalized raw logit $z_k$. Sigmoid activation $\sigma(z_k) = \frac{1}{1 + e^{-z_k}}$ is applied independently to each neuron during evaluation, treating each disease as an independent binary classification question.

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
- **Parameter Efficiency:** 391,918 parameters (~0.39 MB), allowing rapid scratch training on 4GB VRAM local GPUs (RTX 2050) and Kaggle cloud GPUs.

---

## 4. Evaluation Workflow & Strict Scientific Isolation Rule

```
TRAIN (73,916 images / 23,806 patients)
   ↓
VALIDATION (12,608 images / 4,202 patients)  ──► Select Best Checkpoint (val_macro_auc)
   ↓
LOCK MODEL CHECKPOINT (medcxrnet_baseline_best.pth)
   ↓
OFFICIAL TEST EVALUATION (25,596 images / 2,797 patients)
   ↓
FINAL BASELINE TEST METRICS (experiment_1_test_results.json)
   ↓
EXPERIMENT 2 (Tuning & Architectural Iterations)
```

### Scientific Isolation Guidelines
1. **Validation Set Role:** Used exclusively for monitoring model convergence, hyperparameter selection, and saving the best checkpoint (`best_val_macro_auc = 0.7203` in Experiment 1).
2. **Official Test Set Role:** Reserved strictly for final evaluation of locked checkpoints against unseen patient cases (25,596 images / 2,797 unique patients from `test_list.txt`).
3. **Strict Non-Feedback Rule:** Official test-set metrics must **NEVER** be used to select models, adjust hyperparameters, or tune future experiments (such as Experiment 2). This prevents dataset contamination and preserves academic integrity.

---

## 5. Preprocessing & Augmentation Strategy

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
   - Deterministic resizing, tensor conversion, and normalization `(input - 0.5) / 0.5`. **NO random augmentations.**

---

## 6. Official Test-Set Evaluation Metrics

- **Primary Metric:** **Macro ROC-AUC** across the 14 pathology classes (unweighted arithmetic mean using `np.nanmean`).
- **Secondary Metrics:** Per-Class ROC-AUC, Macro & Per-Class PR-AUC, Precision, Recall (Sensitivity), Specificity, F1-Score.
- **Decision Threshold:** Baseline threshold $T = 0.5$ converts probabilities to binary predictions during test reporting.
