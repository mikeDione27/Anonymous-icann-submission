# Anonymous ICANN 2026 Submission Repository

This repository contains the implementation of the models and experiments described in the anonymous submission to ICANN 2026.

The proposed framework focuses on spatio-temporal representation learning for Distributed Acoustic Sensing (DAS) data using transformer-based architectures, including a Mixture-of-Experts (MoE) model: DASTMoE

---

## Repository Structure
.<br>
├── main.py # Training and evaluation script<br>
├── models.py # Model architectures<br>
├── utils.py # Dataset loading, training utilities, evaluation, plotting<br>
├── requirements.txt # Dependencies<br>
├── data/ # Expected dataset structure (view cited paper)<br>
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
python main.py --model_name SpatioTemporalTrueMoE --dataset_name LaboDAS
Train baseline models
python main.py --model_name CNNOnly --dataset_name LaboDAS
python main.py --model_name CNN1Transformer --dataset_name LaboDAS
Default Configuration
Input shape: (T, N) = (10000, 12)
Hidden dimension: 128
Batch size: 32
Learning rate: 1e-4
Optimizer: Adam
Outputs

All outputs are saved in:

outputs/

Including:

Trained model (.pth)
Training curves
Confusion matrix
Feature embeddings
Training history (.csv, .npy)
MoE gating statistics (if applicable)
Key Features
Spatio-temporal modeling of DAS signals
Transformer-based expert networks
Cross-attention fusion mechanism
Adaptive expert selection via gating
Load balancing regularization
End-to-end training and evaluation pipeline
Notes
The dataset is not included to ensure anonymity.
GPU is automatically used if available.
Paths and hyperparameters can be configured via command-line arguments.
Anonymity Statement

All identifying information has been removed to ensure a double-blind review process.

Reproducibility

The repository is self-contained and allows full reproducibility given access to the dataset.

