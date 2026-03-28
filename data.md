# Dataset Description

This repository relies on three datasets used for evaluating the proposed models.  
Due to anonymity and data sharing restrictions, the datasets are **not included** in this repository.

Below are the links and references to access each dataset.

---

## 1. φ-OTDR Laboratory Dataset

This dataset is widely used for benchmarking DAS-based event classification.

### Description
- 6 classes:
  - background
  - digging
  - knocking
  - watering
  - shaking
  - walking
- Sample shape: `(10000 × 12)` (time × channels)

### Access
The dataset is publicly available and can typically be accessed via the original publication or associated repositories.

### Reference

- Cao, X., et al.  
  *Distributed Acoustic Sensing for Event Recognition: A Deep Learning Approach*  
  IEEE Sensors Journal

---

## 2. Real-world DAS Dataset (Infrastructure Monitoring Scenario)

This dataset corresponds to a more realistic deployment scenario with environmental variability.

### Description
- Real-world DAS acquisition
- Multiple event types under varying conditions
- Used to evaluate robustness and generalization

### Access
- Adrian Tomasov et al.
  
*Comprehensive Dataset for Event Classification Using Distributed Acoustic Sensing (DAS) Systems*

- https://doi.org/10.6084/m9.figshare.27004732


### Reference

- [Anonymous reference for double-blind review]

---

## 3. Geophysical DAS Dataset (Rock Slope Failure Monitoring)

This dataset is used for geophysical monitoring tasks.

### Description
- DAS signals recorded for rock slope failure detection
- Binary or multi-class classification depending on setup
- High spatial resolution along fiber

### Access

The dataset is available through the following repository:

- https://www.envidat.ch/#/metadata/distributed-acoustic-sensing-brienz

(Search for the dataset associated with the reference below)

### Reference

- Jiahui Kang et al. 

    Kang, J., et al. (2024).
    *Automatic monitoring of rock‐slope failures using Distributed Acoustic Sensing and semi‐supervised learning*
    Geophysical Research Letters, 51,
    e2024GL110672. https://doi.org/10.1029/
    2024GL110672


