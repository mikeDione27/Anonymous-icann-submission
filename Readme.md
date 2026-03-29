# DASTMoE: A Spatio-Temporal Mixture-of-Experts Model for Distributed Acoustic Sensing

This repository contains the implementation of the models and experiments described in the anonymous submission to ICANN 2026.

The proposed framework focuses on spatio-temporal representation learning for Distributed Acoustic Sensing (DAS) data using transformer-based architectures, including a Mixture-of-Experts (MoE) model: DASTMoE

---

## Repository Structure
.<br>
├── main.py <br>
├── models.py <br>
├── utils.py <br>
├── requirements.txt <br>
├── data/ <br>
└── Results



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

## Dataset Format

The dataset (Laboratory dataset) must be structured as follows:

data/
<br>
├── train/ <br>
│ ├── label.txt<br>
│ └── *.mat files<br>
└── test/<br>
  ├── label.txt<br>
  └── *.mat files<br>


Each `.mat` file must contain a variable named:

data 

with shape:


(T, N)


Where:
- `T`: temporal dimension (e.g., 10000)
- `N`: number of DAS channels (e.g., 12)

---

## Label File Format

Each `label.txt` file must contain lines formatted as:


relative_path label


Example:


sample1.mat 0 <br>
sample2.mat 3


---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt 
Usage
Train the proposed MoE model
python main.py --model_name DASTMoE --dataset_name LaboDAS (ex: Laboratory dataset)
Train baseline models
python main.py --model_name CNNOnly --dataset_name LaboDAS
python main.py --model_name CNN1Transformer --dataset_name LaboDAS
Default Configuration
Input shape: (T, N) = (10000, 12)
Hidden dimension: 128
Batch size: 32
Learning rate: 1e-4
Optimizer: Adam



Reproducibility

The repository is self-contained and allows full reproducibility given access to the dataset.

