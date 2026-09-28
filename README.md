
<div align="center">

<img src="./medifusion-logo.svg" alt="MediFusion AI Logo" width="220"/>

<br/>

# MediFusion AI

### AI-Assisted Chest X-Ray Abnormality Analysis

<img src="https://readme-typing-svg.demolab.com?font=Inter&weight=600&size=20&duration=2800&pause=900&color=0E7490&center=true&vCenter=true&width=820&lines=Medical+AI+%7C+Computer+Vision+%7C+Deep+Learning;14-Class+Multi-Label+Chest+X-Ray+Analysis;Patient-Level+Evaluation+%7C+Threshold+Optimization;Research-Oriented+AI+Prototype" alt="MediFusion AI animated tagline" />

</div>
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B3C5D,45:0E7490,100:14B8A6&height=180&section=header&text=MediFusion%20AI&fontSize=52&fontColor=FFFFFF&fontAlignY=38&desc=AI-Assisted%20Chest%20X-Ray%20Abnormality%20Analysis&descAlignY=62&descSize=18&animation=fadeIn" width="100%" />

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Inter&weight=600&size=20&duration=2800&pause=900&color=0E7490&center=true&vCenter=true&width=760&lines=Medical+AI+%7C+Computer+Vision+%7C+Deep+Learning;14-Class+Multi-Label+Chest+X-Ray+Analysis;Patient-Level+Evaluation+%7C+Threshold+Optimization;Research-Oriented+AI+Prototype" alt="MediFusion animated tagline" />

<br/><br/>

<img src="https://img.shields.io/badge/Project-MediFusion%20AI-0E7490?style=for-the-badge" />
<img src="https://img.shields.io/badge/Domain-Medical%20AI-0B3C5D?style=for-the-badge" />
<img src="https://img.shields.io/badge/Task-Multi--Label%20Classification-14B8A6?style=for-the-badge" />
<img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />

<br/>

<img src="https://img.shields.io/badge/Dataset-NIH%20ChestX--ray14-2563EB?style=flat-square" />
<img src="https://img.shields.io/badge/Input-224%C3%97224%20Grayscale-64748B?style=flat-square" />
<img src="https://img.shields.io/badge/Classes-14-0E7490?style=flat-square" />
<img src="https://img.shields.io/badge/Model-MedCXRNet-0B3C5D?style=flat-square" />
<img src="https://img.shields.io/badge/Parameters-391%2C918-475569?style=flat-square" />
<img src="https://img.shields.io/badge/Tests-43%20Passed-16A34A?style=flat-square" />

<br/><br/>

MediFusion is a research and educational prototype for AI-assisted chest X-ray abnormality analysis.

It is not a medical device, does not provide medical diagnoses, and must not be used for clinical decision-making.

<br/>

</div>

🩻 About MediFusion

MediFusion AI is a deep-learning computer vision system designed to analyze chest X-ray images and estimate the probability of 14 thoracic abnormalities.

The project focuses on building an end-to-end medical AI pipeline rather than only training a neural network.

The complete workflow includes:

                    ┌───────────────────────┐
                    │    Chest X-Ray Image  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Image Preprocessing   │
                    │ Grayscale • 224×224   │
                    │ Normalization         │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      MedCXRNet        │
                    │   Deep Learning Model │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ 14 Probability Scores │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Validation-Derived    │
                    │ Frozen Thresholds     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Predicted Findings    │
                    │ + Probabilities       │
                    └───────────────────────┘

🧠 What Does MediFusion Do?

MediFusion performs multi-label classification.

This means one X-ray can receive multiple predicted findings simultaneously rather than being forced into a single class.

For every supported abnormality, the model produces:

Continuous probability

Validation-derived decision threshold

Threshold-based prediction

Per-class prediction information

The system therefore separates:

Model probability

from

Final threshold-based prediction

This is important because the decision threshold is not necessarily 0.50 for every pathology.

🧬 Supported Abnormalities

MediFusion currently works with the following 14 classes:

#

Abnormality

01

Atelectasis

02

Cardiomegaly

03

Effusion

04

Infiltration

05

Mass

06

Nodule

07

Pneumonia

08

Pneumothorax

09

Consolidation

10

Edema

11

Emphysema

12

Fibrosis

13

Pleural Thickening

14

Hernia

🔬 Model

MedCXRNet

MediFusion uses a custom neural network named MedCXRNet.

Property

Value

Model

MedCXRNet

Parameters

391,918

Input

1 × 224 × 224

Image type

Grayscale

Output

14 classes

Task

Multi-label classification

Training

From scratch

The architecture was intentionally kept unchanged between the major experiments so that the effect of the training strategy could be studied more cleanly.

🗂️ Dataset

MediFusion uses the NIH ChestX-ray14 dataset.

The project uses official patient-level train, validation and test splits.

Dataset Split

Split

Images

Patients

Training

73,916

23,806

Validation

12,608

4,202

Official Test

25,596

2,797

Patient-level separation is important because images from the same patient should not leak across training and evaluation sets.

⚙️ Preprocessing

Each chest X-ray goes through a consistent preprocessing pipeline.

Original X-Ray
      │
      ▼
Grayscale Conversion
      │
      ▼
Resize
224 × 224
      │
      ▼
Normalization
      │
      ▼
Tensor
1 × 224 × 224

The same preprocessing strategy is used during training and inference.

🧪 Experiments

MediFusion was developed through controlled experiments rather than immediately selecting a single training configuration.

Experiment 1 — Baseline

Objective

Establish a baseline model using standard unweighted binary cross-entropy loss.

Configuration

Parameter

Value

Architecture

MedCXRNet

Parameters

391,918

Loss

BCEWithLogitsLoss

Optimizer

AdamW

Learning Rate

1e-4

Batch Size

16

Maximum Epochs

15

Input Size

224 × 224

Input

Grayscale

Split

Patient-level

Best Checkpoint

Epoch 15

Validation

Validation Macro ROC-AUC:

0.7203

Official Test

Test Macro ROC-AUC:

0.6630

Test Macro Metrics at Threshold 0.50

Metric

Value

Macro ROC-AUC

0.6630

Macro PR-AUC

0.1258

Macro Precision

0.0803

Macro Recall

0.0035

Macro Specificity

0.9993

Macro F1

0.0065

The very high specificity together with extremely low recall at the default threshold demonstrates the effect of severe class imbalance on fixed-threshold predictions.

⚖️ Experiment 2 — Class-Imbalance-Aware Training

Objective

Experiment 2 investigates whether explicitly accounting for class imbalance during training improves the model's ability to identify positive findings.

The architecture and major training configuration were kept consistent with Experiment 1.

Weighted BCE

The positive class weight was calculated exclusively from the training split:

pos_weight[c] = negative_count[c] / positive_count[c]

Validation and test labels were not used to calculate these weights.

Training Configuration

Parameter

Value

Architecture

MedCXRNet

Loss

Weighted BCEWithLogitsLoss

Optimizer

AdamW

Learning Rate

1e-4

Batch Size

16

Maximum Epochs

15

Input Size

224 × 224

Split

Patient-level

Best Checkpoint

Epoch 15

Experiment 2 Class Weights

The weights were calculated from the training split only.

Pathology

Positive

Negative

Weight

Atelectasis

7,003

66,913

9.5549

Cardiomegaly

1,453

72,463

49.8713

Effusion

7,415

66,501

8.9684

Infiltration

11,853

62,063

5.2361

Mass

3,452

70,464

20.4125

Nodule

4,042

69,874

17.2870

Pneumonia

742

73,174

98.6172

Pneumothorax

2,287

71,629

31.3201

Consolidation

2,482

71,434

28.7808

Edema

1,201

72,715

60.5454

Emphysema

1,274

72,642

57.0188

Fibrosis

1,051

72,865

69.3292

Pleural Thickening

1,870

72,046

38.5273

Hernia

113

73,803

653.1239

Experiment 2 Official Test Results

The Experiment 2 checkpoint was evaluated on the official test split.

Metric

Experiment 2

Macro ROC-AUC

0.6569

Macro PR-AUC

0.1164

Macro Precision

0.1013

Macro Recall

0.7198

Macro Specificity

0.5044

Macro F1

0.1676

The results show a substantially different precision/recall/specificity balance compared with the unweighted baseline at threshold 0.50.

Importantly, ROC-AUC and PR-AUC are calculated from continuous model probabilities and do not depend on the classification threshold.

📊 Experiment 1 Threshold Optimization

Because the baseline model showed extremely low recall at the default 0.50 threshold, a separate validation-only threshold optimization procedure was performed.

Method

For each of the 14 classes:

Threshold candidates:
0.05 → 0.10 → 0.15 → ... → 0.95

The threshold producing the highest validation F1 score was selected.

The official test set was not used during threshold selection.

Frozen Validation Thresholds

Pathology

Threshold

Atelectasis

0.10

Cardiomegaly

0.05

Effusion

0.15

Infiltration

0.20

Mass

0.05

Nodule

0.10

Pneumonia

0.05

Pneumothorax

0.05

Consolidation

0.05

Edema

0.10

Emphysema

0.05

Fibrosis

0.05

Pleural Thickening

0.05

Hernia

0.05

These thresholds were then frozen before official test evaluation.

📈 Experiment 1 Test Results With Frozen Validation Thresholds

The following metrics were obtained after applying the validation-derived thresholds to the official test set.

Pathology

Precision

Recall

Specificity

F1

ROC-AUC

PR-AUC

Atelectasis

0.1589

0.7319

0.4308

0.2611

0.6174

0.1819

Cardiomegaly

0.0961

0.1628

0.9333

0.1209

0.6500

0.0706

Effusion

0.2818

0.7450

0.5777

0.4089

0.7196

0.3431

Infiltration

0.3132

0.7317

0.4967

0.4386

0.6533

0.3509

Mass

0.0871

0.6882

0.4716

0.1547

0.6159

0.0995

Nodule

0.0953

0.1935

0.8757

0.1277

0.5974

0.0853

Pneumonia

0.0000

0.0000

0.9996

0.0000

0.5942

0.0272

Pneumothorax

0.2101

0.5129

0.7759

0.2981

0.7135

0.2032

Consolidation

0.1093

0.7537

0.5311

0.1909

0.6786

0.1153

Edema

0.0883

0.2638

0.8979

0.1324

0.7503

0.0816

Emphysema

0.0899

0.0988

0.9554

0.0941

0.6688

0.0731

Fibrosis

0.0503

0.0690

0.9775

0.0582

0.7352

0.0412

Pleural Thickening

0.0976

0.2765

0.8805

0.1443

0.6707

0.0823

Hernia

0.0000

0.0000

1.0000

0.0000

0.6172

0.0054

Macro Average

Metric

Value

Precision

0.1199

Recall

0.3734

Specificity

0.7717

F1

0.1736

ROC-AUC

0.6630

PR-AUC

0.1258

🔍 Inference Pipeline

MediFusion includes a production-style single-image inference pipeline.

                    INPUT
                      │
                      ▼
             ┌─────────────────┐
             │ Chest X-Ray     │
             │ PNG / JPEG      │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Preprocessing   │
             │ Grayscale       │
             │ 224 × 224       │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │   MedCXRNet     │
             │   Checkpoint    │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ 14 Probabilities│
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Frozen Class    │
             │ Thresholds      │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Predicted       │
             │ Findings        │
             └─────────────────┘

🧪 Example Prediction Output

The inference system returns structured prediction information similar to:

{
  "predicted_findings": [
    "Atelectasis",
    "Effusion",
    "Mass"
  ],
  "model_info": {
    "model_name": "MedCXRNet",
    "device": "cpu"
  },
  "abnormality_predictions": [
    {
      "pathology": "Atelectasis",
      "probability": 0.141978,
      "threshold": 0.10,
      "predicted": true
    },
    {
      "pathology": "Effusion",
      "probability": 0.189253,
      "threshold": 0.15,
      "predicted": true
    }
  ]
}

The system deliberately uses terminology such as predicted findings and abnormality predictions rather than claiming to provide a medical diagnosis.

🧰 Technology Stack

<div align="center">

Technology

Purpose

Python

Core development

PyTorch

Deep learning

NumPy

Numerical computation

Pandas

Dataset processing

scikit-learn

Evaluation metrics

PIL

Image processing

pytest

Automated testing

CUDA

GPU acceleration

Kaggle

Model training environment

Git / GitHub

Version control

</div>

🏗️ Project Architecture

MediFusion/
│
├── DataSet/
│   └── NIH ChestX-ray14 data
│
├── models/
│   ├── medcxrnet_baseline_best.pth
│   ├── medcxrnet_baseline_latest.pth
│   ├── medcxrnet_exp2_best.pth
│   └── medcxrnet_exp2_latest.pth
│
├── reports/
│   └── results/
│       ├── experiment_1_baseline_results.json
│       ├── experiment_1_test_results.json
│       ├── experiment_1_validation_optimal_thresholds.json
│       ├── experiment_1_test_thresholded_results.json
│       ├── experiment_2_class_imbalance_results.json
│       └── experiment_2_test_results.json
│
├── scripts/
│   ├── train.py
│   └── ...
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   │   └── loss.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── metrics.py
│   └── inference/
│       ├── predictor.py
│       └── __init__.py
│
├── tests/
│   ├── test_model_pipeline.py
│   ├── test_threshold_optimizer.py
│   └── test_predictor.py
│
└── README.md

🚀 Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/MediFusion.git
cd MediFusion

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

▶️ Training

Check available training modes:

python scripts/train.py --help

Baseline Experiment

python scripts/train.py --mode train --epochs 15 --batch-size 16 --lr 1e-4

Experiment 2

python scripts/train.py --mode exp2 --epochs 15 --batch-size 16 --lr 1e-4

Experiment 2 calculates positive class weights exclusively from the training split.

🔬 Evaluation

The official test set is kept separate from training and model selection.

Evaluation can be performed using the project's evaluator:

python scripts/evaluate.py

The evaluator produces structured JSON results containing:

ROC-AUC

PR-AUC

Precision

Recall / Sensitivity

Specificity

F1

Per-class metrics

🩻 Single Image Inference

MediFusion provides a reusable inference interface.

Conceptually:

from src.inference import MediFusionPredictor

predictor = MediFusionPredictor()

result = predictor.predict("path/to/chest_xray.png")

print(result)

The predictor:

Loads the trained checkpoint.

Loads frozen validation-derived thresholds.

Preprocesses the X-ray.

Runs deterministic inference.

Calculates probabilities.

Applies per-class thresholds.

Returns structured predicted findings.

🧪 Testing

The project includes automated tests covering the training and inference pipeline.

Run:

python -m pytest

The verified test suite reached:

43 passed

Tests cover areas including:

Model pipeline

Weighted BCE loss

Class-weight calculation

Numerical stability

Threshold optimization

Predictor loading

Prediction formatting

Probability ranges

Frozen threshold behavior

Deterministic inference

Error handling

🛡️ Data Leakage Prevention

MediFusion explicitly separates training, validation and official test data.

Class weights

Experiment 2 class weights are calculated using:

TRAINING SPLIT ONLY

Threshold optimization

Decision thresholds are calculated using:

VALIDATION SPLIT ONLY

Final test evaluation

The official test set is used only for:

FINAL PERFORMANCE EVALUATION

This prevents test-set information from influencing training or threshold selection.

📐 Evaluation Methodology

MediFusion reports both threshold-independent and threshold-dependent metrics.

Threshold-independent

These metrics operate directly on continuous probabilities:

ROC-AUC

PR-AUC

Threshold-dependent

These metrics depend on the selected decision threshold:

Precision

Recall / Sensitivity

Specificity

F1

This distinction is important when interpreting model performance.

💡 Why Multi-Label Classification?

Chest X-rays can contain multiple findings simultaneously.

For example:

Chest X-Ray
     │
     ├── Effusion
     ├── Atelectasis
     └── Cardiomegaly

A multi-label model can represent these findings simultaneously instead of forcing the image into only one category.

🔐 Responsible AI

MediFusion is intended for research and educational purposes.

The model:

Is not clinically validated.

Is not a medical device.

Does not replace radiologists or physicians.

Does not establish a diagnosis.

Should not be used for clinical decision-making.

May produce false positives and false negatives.

Should not be interpreted as definitive medical advice.

Predictions should be treated as model outputs for research and demonstration purposes only.

⚠️ Limitations

The current system has several important limitations.

Dataset limitations

The model is trained on the NIH ChestX-ray14 dataset and may not generalize to other populations, hospitals, imaging equipment or acquisition protocols.

Class imbalance

Some abnormalities are substantially rarer than others.

For example:

Hernia:
113 positive training images

This makes reliable learning for rare classes difficult.

Model size

MedCXRNet is intentionally lightweight:

391,918 parameters

This makes the system easier to train and deploy but limits representational capacity compared with much larger modern architectures.

Threshold dependence

Different thresholds can substantially change precision, recall and specificity.

Therefore, a single metric should not be interpreted in isolation.

🔮 Future Work

Potential future development includes:

Larger and stronger backbone architectures

Transfer learning

Improved calibration

More sophisticated imbalance handling

Data augmentation studies

Explainability methods such as Grad-CAM

External dataset validation

Robustness testing

Confidence calibration

Better rare-class performance

Clinical expert evaluation

Model comparison experiments

Deployment optimization

📚 Research Workflow

The development process follows a reproducible experimental structure:

Dataset
   │
   ▼
Patient-Level Split
   │
   ├───────────────┐
   ▼               ▼
Training        Validation
   │               │
   ▼               │
Model Training     │
   │               │
   └───────┬───────┘
           ▼
     Model Selection
           │
           ▼
   Validation Analysis
           │
           ▼
   Threshold Optimization
           │
           ▼
   Thresholds Frozen
           │
           ▼
    Official Test Set
           │
           ▼
    Final Evaluation

📊 Key Results at a Glance

<div align="center">

Experiment

Test Macro ROC-AUC

Test Macro PR-AUC

Test Macro Recall

Test Macro F1

Experiment 1 — Baseline, threshold 0.50

0.6630

0.1258

0.0035

0.0065

Experiment 2 — Weighted BCE, threshold 0.50

0.6569

0.1164

0.7198

0.1676

Experiment 1 — Frozen validation thresholds

0.6630

0.1258

0.3734

0.1736

</div>

Important: These rows use different thresholding configurations. ROC-AUC and PR-AUC remain threshold-independent, while precision, recall, specificity and F1 depend on the decision thresholds used.

🧑‍💻 Development Philosophy

MediFusion was developed around several principles:

Reproducibility
      +
Patient-Level Evaluation
      +
Strict Data Separation
      +
Controlled Experiments
      +
Automated Testing
      +
Responsible Medical AI

The goal is not simply to achieve a high number on a metric.

The goal is to build an understandable, reproducible and testable medical-AI research pipeline.

📁 Important Outputs

The most important generated artifacts include:

models/
│
├── medcxrnet_baseline_best.pth
├── medcxrnet_baseline_latest.pth
├── medcxrnet_exp2_best.pth
└── medcxrnet_exp2_latest.pth

and:

reports/results/
│
├── experiment_1_baseline_results.json
├── experiment_1_test_results.json
├── experiment_1_validation_optimal_thresholds.json
├── experiment_1_test_thresholded_results.json
├── experiment_2_class_imbalance_results.json
└── experiment_2_test_results.json

🌐 Project Status

<div align="center">

🩻 MediFusion AI

Research Prototype

Dataset ──────── Complete
Baseline ─────── Complete
Experiment 2 ─── Complete
Evaluation ───── Complete
Thresholding ─── Complete
Inference ────── Complete
Testing ──────── Verified
UI ───────────── Prototype

<br/>

<img src="https://img.shields.io/badge/Research%20Prototype-0E7490?style=for-the-badge" />
<img src="https://img.shields.io/badge/Medical%20AI-Educational-0B3C5D?style=for-the-badge" />
<img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />

</div>

⚕️ Medical Disclaimer

<div align="center">

IMPORTANT

MediFusion AI is a research and educational prototype.

It is not a medical device, is not clinically validated, and does not provide medical diagnoses.

Model predictions may be incorrect and must not be used for diagnosis, treatment, triage, or other clinical decisions.

Always consult a qualified healthcare professional for medical interpretation.

</div>

⭐ Acknowledgements

This project uses the NIH ChestX-ray14 dataset for research and educational experimentation.

The project also builds upon the open-source Python and deep-learning ecosystem, including PyTorch, NumPy, Pandas, scikit-learn, PIL and pytest.

<div align="center">

🩻 MediFusion AI

Seeing patterns. Supporting research. Not replacing clinicians.

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B3C5D,45:0E7490,100:14B8A6&height=120&section=footer" width="100%" />

</div>
