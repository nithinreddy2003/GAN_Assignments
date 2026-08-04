# 7COM1079 Coursework 2 (Set 3) — GANs for Retinal, Network-Traffic and Sketch Data

Building generative adversarial networks (GANs) from scratch and applying them to three real-world problems: medical imaging (OCTMNIST retinal scans), cybersecurity (CICIDS 2017 network flows) and creative AI (QuickDraw sketches).

## What is in here

**Part 1 — GANs from scratch (PyTorch).** Two small MLP-based GANs trained on synthetic 2D data: the sine-wave baseline from the tutorial, a chosen noisy parametric curve `y = sin(2x) + 0.3cos(5x) + ε`, and an architecture study (depth-2 ReLU baseline vs a depth-4 ELU generator).

**Part 2 — three real-world applications.** Each domain uses the architecture that fits its data, so the framework deliberately switches between parts:

| Section | Domain | Dataset | Model | Framework |
|---|---|---|---|---|
| 2.1 | Medicine | [OCTMNIST](https://medmnist.com/) (MedMNIST v2) | DCGAN + conditional GAN (extension) | TensorFlow/Keras |
| 2.2 | Cybersecurity | [CICIDS 2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/data) | Dense feature-vector GAN + full-day run (extension) | TensorFlow/Keras |
| 2.3 | Creative AI | [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) `birthday cake` + cat/house (extension) | DCGAN | PyTorch |

Evaluation uses generator/discriminator loss curves, real-vs-generated grids, Fréchet Inception Distance (FID) for the image models, and PCA / t-SNE plus per-feature alignment gaps for the tabular model.

## Files

```
Set 3/
├── GAN_Assignment_Set3_Final.ipynb   # full notebook: all modelling, training, figures and metrics
├── GAN_Assignment_Report_Set 3.docx  # written report (6–8 pages, with embedded figures)
├── Set 3 Images Results/             # exported figures used in the report
├── requirements.txt
└── README.md
```

The notebook also writes its figures to a local `results/` folder at run time (created automatically, not tracked here).

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is recommended — the Part 2 models train much faster on one. The very first notebook cell also installs these packages, so on Colab/SageMaker you can simply run the notebook top to bottom.

## Running it

Run the cells in order. A few practical notes:

- **OCTMNIST (2.1)** downloads automatically through the `medmnist` package on first use.
- **CICIDS 2017 (2.2)** is fetched with `kagglehub`, which needs Kaggle credentials the first time (a `kaggle.json` API token, or an interactive login on Colab). The eight daily CSVs are located and combined automatically — no manual folder setup.
- **QuickDraw (2.3)** bitmaps download directly from Google Cloud Storage.
- Seeds are fixed (`GLOBAL_SEED = 2025` for PyTorch/NumPy, `TF_SEED = 2026` for TensorFlow) so the figures and numbers reproduce.

## Headline results

| Model | Metric | Value |
|---|---|---|
| Part 1 sine-wave GAN | D / G loss at convergence | ≈ 1.38 / 0.70 (near the 2ln2 / ln2 equilibrium) |
| 2.1 OCTMNIST DCGAN | FID | 39.23 |
| 2.2 CICIDS (Wednesday) | mean / std alignment gap | 0.1576 / 0.2067 |
| 2.2 CICIDS (all days) | mean / std alignment gap | 0.1300 / 0.1483 |
| 2.3 QuickDraw | FID — house / cake / cat | 26.44 / 34.29 / 38.79 |

## Note on the CICIDS labels

The brief refers to "DDoS", but the Wednesday file's attacks are single-source **DoS** (Hulk, GoldenEye, slowloris, Slowhttptest); genuine DDoS traffic sits in the Friday file and is only included in the full-day extension. The notebook and report follow the named day and model Wednesday as BENIGN vs DoS.
