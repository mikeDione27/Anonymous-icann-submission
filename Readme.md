# DASTMoE: A Spatio-Temporal Mixture-of-Experts Model for Distributed Acoustic Sensing

This repository contains the implementation of the models and experiments described in the anonymous submission to ICANN 2026.

The proposed framework focuses on spatio-temporal representation learning for Distributed Acoustic Sensing (DAS) data using transformer-based architectures, including a Mixture-of-Experts (MoE) model: **DASTMoE**.

---

## Repository Structure

.<br>
├── main.py<br>
├── models.py<br>
├── utils.py<br>
├── requirements.txt<br>
└── results/<br>



---

## Supported Models

The repository includes the following models:

- **CNNOnly**: Convolutional baseline model  
- **CNN1Transformer**: Hybrid CNN + Transformer architecture  
- **DASTMoE**: Proposed Mixture-of-Experts model including:
  - Temporal expert (Transformer)
  - Spatial expert (Transformer)
  - Fusion expert (cross-attention + Transformer)
  - Gating network for adaptive expert weighting

---

# Dataset Description

Below are the datasets used in this work, along with their descriptions and references.

---

## 1. φ-OTDR Laboratory Dataset

This dataset is widely used for benchmarking DAS-based event classification.

### Description

<p align="center">
  <img src="images/labo_scenario.png" width="70%">
</p>

- 6 classes:
  - (a) background  
  - (b) digging  
  - (c) knocking  
  - (d) watering  
  - (e) shaking  
  - (f) walking  

### Train/Test Split

| Event      | Number | Label | Train | Test |
|------------|--------|-------|-------|------|
| Background | 3094   | 0     | 2505  | 589  |
| Digging    | 2512   | 1     | 2018  | 494  |
| Knocking   | 2530   | 2     | 2025  | 505  |
| Watering   | 2298   | 3     | 1853  | 445  |
| Shaking    | 2728   | 4     | 2183  | 545  |
| Walking    | 2450   | 5     | 1969  | 481  |
| **Total**  | **15612** | -- | **12553** | **3059** |

### Access

The dataset is publicly available via the original publication.

### Reference

- Cao, X., et al.  
  *An open dataset of φ-OTDR events with two classification models as baselines*  
  IEEE Sensors Journal

---

## 2. Geophysical DAS Dataset (Rock Slope Failure Monitoring)

This dataset is used for geophysical monitoring tasks.

### Description

<p align="center">
  <img src="images/DAS_rockfall.png" width="60%">
</p>

- DAS signals recorded for rock slope failure detection  
- High spatial resolution along the fiber  
- 3 classes:
  - Vehicle noise  
  - Slope failure  
  - Narrow-band noise  

### Train/Test Split

| Event             | Number | Label | Train | Test |
|-------------------|--------|-------|-------|------|
| Vehicle noise     | 792    | 0     | 633   | 159  |
| Slope failure     | 384    | 1     | 307   | 77   |
| Narrow-band noise | 191    | 2     | 153   | 38   |
| **Total**         | **1367** | --  | **1093** | **274** |

### Access

https://www.envidat.ch/#/metadata/distributed-acoustic-sensing-brienz

### Reference

- Kang, J., et al. (2024)  
  *Automatic monitoring of rock-slope failures using Distributed Acoustic Sensing and semi-supervised learning*  
  Geophysical Research Letters  
  https://doi.org/10.1029/2024GL110672

---

## 3. Real-world DAS Dataset (Infrastructure Monitoring)

This dataset corresponds to a realistic deployment scenario with environmental variability.

### Description

<p align="center">
  <img src="images/Das_university.png" width="70%">
</p>

- Real-world DAS acquisition  
- Multiple event types under varying conditions  
- Used to evaluate robustness and generalization  

### Train/Test Split

| Event         | Number | Label | Train | Test |
|---------------|--------|-------|-------|------|
| Car           | 1085   | 0     | 757   | 217  |
| Construction  | 825    | 1     | 576   | 165  |
| Fence         | 326    | 2     | 228   | 65   |
| Longboard     | 609    | 3     | 426   | 122  |
| Manipulation  | 527    | 4     | 369   | 105  |
| Open/Close    | 124    | 5     | 87    | 25   |
| Regular       | 1780   | 6     | 1246  | 356  |
| Running       | 533    | 7     | 373   | 107  |
| Walking       | 1468   | 8     | 1031  | 294  |
| **Total**     | **6549** | --  | **5093** | **1456** |

### Access

https://doi.org/10.6084/m9.figshare.27004732

### Reference

- Tomasov, A., et al.  
  *Comprehensive Dataset for Event Classification Using Distributed Acoustic Sensing (DAS) Systems*  
  https://doi.org/10.6084/m9.figshare.27004732

---

## Notes

 
- GPU is automatically used if available  
- Paths can be configured via command-line arguments  

---

## Anonymity Statement

All identifying information has been removed to ensure a double-blind review process.

---