<div align="center">

# 🩺 MediFusion

### AI-Assisted Chest X-Ray Abnormality Analysis

<p>
  <img src="https://readme-typing-svg.demolab.com?font=Inter&weight=600&size=20&duration=2800&pause=900&color=149EAD&center=true&vCenter=true&width=700&lines=Chest+X-Ray+Abnormality+Analysis;14-Class+Multi-Label+Prediction;Validation-Driven+Threshold+Optimization;Research+Prototype+for+Medical+AI" />
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/NIH-ChestX--ray14-149EAD?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Tests-43%20Passing-2EA44F?style=for-the-badge" />
</p>

<p>
  <strong>Research Prototype • Multi-Label Classification • Chest X-Ray Analysis</strong>
</p>

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B7285,50:149EAD,100:4CC9F0&height=130&section=header&text=MediFusion&fontSize=42&fontColor=FFFFFF&animation=fadeIn&fontAlignY=45" />

</div>

---

## ⚕️ What is MediFusion?

**MediFusion** is a research-oriented AI system designed to analyze chest X-ray images and generate **model-based abnormality predictions across 14 thoracic pathology classes**.

The system uses a custom lightweight deep-learning architecture, **MedCXRNet**, trained on the NIH ChestX-ray14 dataset.

Instead of treating the task as a single "normal vs abnormal" classification problem, MediFusion performs **multi-label classification**, allowing multiple abnormalities to be predicted for the same X-ray.

> ⚠️ **Medical Safety Notice**
>
> MediFusion is an educational/research prototype.
> Its outputs are model predictions and **must not be interpreted as medical diagnoses or used for clinical decision-making.**

---

# 🧠 The Problem

Chest X-rays can contain multiple abnormalities simultaneously.

A conventional binary classifier might answer:

```text
Abnormal: YES
