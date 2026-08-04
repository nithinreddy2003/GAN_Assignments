# GAN Coursework (Set 4) — GANs from Scratch and Across Three Domains

A single notebook (`GAN_Assignment_Set4_Final.ipynb`) that builds a GAN from scratch on 2D data and then applies GANs to three real-world problems: medical imaging (OCTMNIST retinal scans), cybersecurity (CICIDS 2017 network flows) and creative AI (QuickDraw sketches).

## What it covers

**Part 1 — GANs from scratch (PyTorch).** Small configurable MLPs driven by a `Toy2DGAN` trainer, trained with the non-saturating loss: the sine-wave baseline, a 2D **spiral**, and an architecture study (depth-3 GELU generator vs a deeper depth-5 LeakyReLU generator).

**Part 2 — three real-world applications.** The framework is chosen to suit each data type:

| Section | Domain | Dataset | Model | Framework |
|---|---|---|---|---|
| 2.1 | Medicine | [OCTMNIST](https://medmnist.com/) (MedMNIST v2) | Resize-convolution DCGAN + conditional GAN (extension) | PyTorch |
| 2.2 | Cybersecurity | [CICIDS 2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/data) | Dense feature-vector GAN + full-day run (extension) | TensorFlow/Keras (`keras.Model` subclass) |
| 2.3 | Creative AI | [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) `birthday cake` + cat/house (extension) | Resize-convolution DCGAN | PyTorch |

The OCT generator uses nearest-neighbour **resize-convolution** (upsample + 3×3 conv) to avoid the checkerboard artefacts transposed convolutions leave on small scans, and the conditional-GAN extension uses a **spectral-normalised critic** with a wrong-label trick to keep training stable. Evaluation uses generator/discriminator loss curves, real-vs-generated grids, Fréchet Inception Distance (FID) for the image models, and PCA / t-SNE plus per-feature alignment gaps for the tabular model.

## Files

```
Set 4/
├── GAN_Assignment_Set4_Final.ipynb   # full notebook: modelling, training, figures and metrics
├── Set 4 Result Images/              # exported figures used in the report
├── requirements.txt
└── README.md
```

The notebook also writes figures to a local `gan_results/` folder at run time (created automatically, not tracked here).

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
- **CICIDS 2017 (2.2)** is fetched with `kagglehub` from the public `chethuhn/network-intrusion-dataset`; the eight daily CSVs are located and combined automatically.
- **QuickDraw (2.3)** bitmaps download directly from Google Cloud Storage.
- Seeds are fixed (`TORCH_SEED = 3030` for PyTorch/NumPy, `KERAS_SEED = 4040` for TensorFlow) so the figures and numbers reproduce.

## Headline results

| Model | Metric | Value |
|---|---|---|
| 2.1 OCTMNIST DCGAN | FID (lower = better) | **31.94** |
| 2.1 (ext.) conditional GAN | per-class scans on demand | one class per grid row |
| 2.2 CICIDS (Wednesday, BENIGN vs DoS) | mean / std alignment gap | **0.4792 / 0.5522** |
| 2.2 (ext.) CICIDS (all days) | mean / std alignment gap | **0.5675 / 0.6285** |
| 2.3 QuickDraw | FID — house / cake / cat | **28.76 / 41.01 / 67.36** |

All metrics are computed in the notebook and match the accompanying report.

## Note on the CICIDS labels

The brief refers to "DDoS", but the Wednesday file's attacks are single-source **DoS** (Hulk, GoldenEye, slowloris, Slowhttptest); genuine DDoS traffic sits in the Friday file and only enters through the full-day extension. The notebook follows the named day and models Wednesday as BENIGN vs DoS.

## References

Key methods: Goodfellow et al. (2014) — GAN objective; Radford et al. (2016) — DCGAN; Mirza & Osindero (2014) — conditional GAN; Odena et al. (2017) / Miyato et al. (2018) — spectral normalisation; Kingma & Ba (2015) — Adam; Heusel et al. (2017) — FID. Full references and dataset citations are in the report.
