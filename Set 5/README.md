# GANs from Scratch and in the Wild (Set 5) — 7COM1079 Coursework 2

A single notebook (`GAN_Assignment_Set5_Final.ipynb`) that builds a GAN from scratch on synthetic 2D data and then applies GANs to three real-world domains: medical imaging (OCTMNIST retinal scans), cybersecurity (CICIDS 2017 network flows) and creative AI (QuickDraw sketches).

## What it covers

**Part 1 — GANs from scratch (PyTorch).** A `SpectralGanTrainer` pairs a LayerNorm generator with a spectral-normalised critic, trained with the non-saturating loss under a two-step critic schedule and one-sided label smoothing: the sine-wave baseline, a **nine-mode mixture of Gaussians on a 3×3 grid**, and an architecture study (2-layer LeakyReLU baseline vs a deeper 3-layer GELU generator).

**Part 2 — three real-world applications.** The framework is chosen to suit each data type:

| Section | Domain | Dataset | Model | Framework |
|---|---|---|---|---|
| 2.1 | Medicine | [OCTMNIST](https://medmnist.com/) (MedMNIST v2) | DCGAN + conditional GAN (extension) | TensorFlow/Keras |
| 2.2 | Cybersecurity | [CICIDS 2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/data) | Dense feature-vector GAN + full-day run (extension) | PyTorch |
| 2.3 | Creative AI | [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) `birthday cake` + cat/house (extension) | DCGAN | TensorFlow/Keras |

The two image GANs share one reusable `GanTrainer` (`keras.Model` subclass trained via `model.fit`); the conditional-GAN extension uses **late-fusion** conditioning (the label is mixed in only at the final dense layer) with instance noise and a two-time-scale rule to keep the critic from overpowering the generator. Evaluation uses loss curves, real-vs-generated grids, FID for the image models, and PCA / t-SNE plus per-feature alignment gaps for the tabular model.

## Files

```
Set 5/
├── GAN_Assignment_Set5_Final.ipynb   # full notebook: modelling, training, figures and metrics
├── Set 5 Result Images/              # exported figures used in the report
├── requirements.txt
└── README.md
```

The notebook also writes figures to a local `results/` folder at run time (created automatically, not tracked here).

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is recommended (the Part 2 models train much faster on one). The first notebook cell also installs these packages, so on Colab you can just run the notebook top to bottom.

## Running it

Run the cells in order (`Kernel → Restart & Run All`). Practical notes:

- **OCTMNIST (2.1)** downloads automatically through the `medmnist` package on first use.
- **CICIDS 2017 (2.2)** is fetched with `kagglehub` from the public `chethuhn/network-intrusion-dataset` (or picked up from a local folder if present); the eight daily CSVs are located and combined automatically.
- **QuickDraw (2.3)** bitmaps download directly from Google Cloud Storage.
- Seeds are fixed (`SEED = 5050` for PyTorch/NumPy, `KERAS_SEED = 6060` for TensorFlow) so the figures and numbers reproduce.

## Headline results

| Model | Metric | Value |
|---|---|---|
| 2.1 OCTMNIST DCGAN | FID (lower = better) | **17.89** |
| 2.1 (ext.) conditional GAN (late-fusion) | per-class scans on demand | one class per grid row |
| 2.2 CICIDS (Wednesday, BENIGN vs DoS) | mean / std alignment gap | **0.2435 / 0.2395** |
| 2.2 (ext.) CICIDS (all days) | mean / std alignment gap | **0.2955 / 0.5277** |
| 2.3 QuickDraw | FID — cake / cat / house | **25.56 / 28.51 / 41.98** |

All metrics are computed in the notebook and match the accompanying report.

## Note on the CICIDS labels

The brief refers to "DDoS", but the Wednesday file's attacks are single-source **DoS** (Hulk, GoldenEye, slowloris, Slowhttptest); genuine DDoS traffic sits in the Friday file and only enters through the full-day extension. The notebook follows the named day and models Wednesday as BENIGN vs DoS.

## References

Key methods: Goodfellow et al. (2014) — GAN objective; Radford et al. (2016) — DCGAN; Mirza & Osindero (2014) — conditional GAN; Miyato et al. (2018) — spectral normalisation; Kingma & Ba (2015) — Adam; Heusel et al. (2017) — FID and the two-time-scale update rule. Full references and dataset citations are in the report.
