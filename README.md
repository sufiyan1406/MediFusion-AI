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

<br/><br/>

> **MediFusion is a research and educational prototype for AI-assisted chest X-ray abnormality analysis.**
>
> It is **not a medical device**, does not provide medical diagnoses, and must not be used for clinical decision-making.

<br/>

</div>

---

# 🩻 About MediFusion

**MediFusion AI** is a deep-learning computer vision system designed to analyze chest X-ray images and estimate the probability of **14 thoracic abnormalities**.

The project focuses on building an end-to-end medical AI pipeline rather than only training a neural network.

The complete workflow includes:

```text
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
