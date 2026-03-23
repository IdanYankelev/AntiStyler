# 🛡️ AntiStyler  
**Defending Object Detection Models Against Adversarial Patch Attacks Using Style Removal**

> Official implementation of the CVPR 2026 paper  
> *AntiStyler: Defending Object Detection Models Against Adversarial Patch Attacks Using Style Removal*

---

## 🚀 Overview

Adversarial patch attacks can severely degrade object detection (OD) performance by inserting localized, malicious patterns into images.

**AntiStyler** is a **fast, training-free, and fully agnostic defense** that:

- 🧠 Detects adversarial patches via “style inconsistency”
- 🎭 Removes adversarial “random style” using a modified style transfer process
- 🎯 Masks suspicious regions with minimal impact on benign images
- ⚡ Runs in real-time (~10–12 FPS)

Unlike many prior defenses, AntiStyler:
- ✅ Does **not require retraining**
- ✅ Is **model-, patch-, and attack-agnostic**
- ✅ Preserves **benign performance**
- ✅ Works on **both digital and physical attacks**

---

## 🖼️ AntiStyler Pipeline

![Pipeline](AntiStyler%20pipeline.png)

AntiStyler consists of **four sequential phases**:

### (A) Style Removal
- Apply **random padding** to ensure the presence of “random style”
- Use a modified style transfer model (**AntiStyle**) to remove style
- Output: *AntiStyled image*

### (B) Filter Phase
- Compute **absolute difference** between original and AntiStyled image  
- Extract pixels with the largest changes → **raw mask**

### (C) Enhancement Phase
Apply morphological operations:
- Dilation → connect regions  
- Erosion → remove noise  
- Smoothing + thresholding → refine mask  

### (D) Mask Phase
- Apply mask to input image  
- Output: **defended image** → fed to detector

---

## 📊 Results

![Results](AntiStyler%20results.png)

### Key Findings

- 📈 **+8–15 mAP improvement** under attacks  
- 🟰 **No degradation on benign images**  
- ⚡ **~80–90 ms per image (~10–12 FPS)**  
- 🥇 Best **speed–robustness tradeoff** among SOTA defenses  

---

## 🔬 Method Intuition

AntiStyler is based on a key observation:

> Adversarial patches introduce **high-frequency, random “style” patterns** that differ from natural image statistics.

Instead of reconstructing images, AntiStyler:
1. Removes style (not content) using a modified loss:
   - Minimize content loss  
   - **Maximize style loss**
2. Regions that change the most → likely adversarial
3. These regions are **masked instead of reconstructed**

This avoids:
- Object misalignment  
- Localization errors  
- Heavy computation  

---

## 🧪 Supported Attacks

AntiStyler is evaluated against:

- Google Adversarial Patch  
- M-PGD  
- DPatch  
- TSEA  
- Printable patches  
- Naturalistic patches  
- Physical attacks (APRICOT, Superstore)

---

## ▶️ Quick Start

### Run Demo (Recommended)

```bash
jupyter notebook AntiStyler_Demo.ipynb
```

This notebook demonstrates:
- Attack Generation
- AntiStyler Pipeline
- AntiStyler's Effect on Benign and Adversarial Images

 ---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
TBD
```

## 🙌 Acknowledgements

- Ben-Gurion University  
- Fujitsu Research  

---

## ⭐ If You Find This Useful

Please ⭐ the repo and cite the paper!
