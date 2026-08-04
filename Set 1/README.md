# Generative Adversarial Networks: From 2D Toy Data to Real-World Applications

Coursework implementing GANs from scratch and applying them across three real-world domains: medical imaging (OCTMNIST retinal scans), cybersecurity (CICIDS2017 network traffic), and creative AI (QuickDraw sketches).

## Overview

This project has two parts:

**Part 1 — GANs from scratch (PyTorch).** A vanilla GAN is built and trained on synthetic 2D data: a sine-wave baseline, a 2D Archimedean spiral, and an architecture variant (LeakyReLU + deeper generator) compared against the original.

**Part 2 — Real-world applications.**
| Section | Domain | Dataset | Model | Framework |
|---|---|---|---|---|
| 2.1 | Medicine | [OCTMNIST](https://medmnist.com/) (MedMNIST v2) | DCGAN (+ conditional GAN extension) | TensorFlow/Keras |
| 2.2 | Cybersecurity | [CICIDS2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) | Tabular MLP GAN | PyTorch |
| 2.3 | Creative AI | [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) `birthday cake` (+ cat/house extension) | DCGAN | TensorFlow/Keras |

All three real-world models share a common training recipe for comparability: BatchNorm in the generator, Adam (β₁=0.5, β₂=0.999), and one-sided label smoothing.

## Repository structure

```
.
├── GAN_Assignment_Set1_Final.ipynb   # full notebook — all modelling, training, and evaluation
├── GAN_Assignment_Report_Set_1.docx  # written report (6–8 pages)
├── requirements.txt
├── README.md
└── results/                          # generated on first run — figures, sample grids, zipped archive
    └── results_set1/                 # TensorFlow-section outputs (OCTMNIST, QuickDraw)
```

`results/` and `results_set1/` are created automatically by the notebook; they are not tracked in the repo.

## Setup

```bash
git clone <this-repo-url>
cd <this-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**GPU note:** `requirements.txt` installs `tensorflow[and-cuda]`, which assumes an NVIDIA GPU. On a CPU-only or non-NVIDIA machine, install plain `tensorflow` instead:
```bash
pip uninstall tensorflow-cuda  # if partially installed
pip install tensorflow
```

## Data access

All three datasets are fetched automatically by the notebook — no manual download required — but one needs a one-time credential setup:

- **OCTMNIST** — downloads automatically via the `medmnist` package on first run.
- **CICIDS2017** — downloads automatically via `kagglehub`, which requires a Kaggle account and API token. Place your `kaggle.json` at `~/.kaggle/kaggle.json` (or set the `KAGGLE_KEY`/`KAGGLE_USERNAME` environment variables) before running the Part 2.2 cells. See [Kaggle API docs](https://www.kaggle.com/docs/api) for details.
- **QuickDraw** — downloaded directly from Google's public dataset bucket via a plain URL request; no credentials needed.

## Running the notebook

Open `GAN_Assignment_Set1_Final.ipynb` and run all cells top to bottom (`Kernel → Restart & Run All`). The notebook is organised sequentially:

1. Environment setup
2. Part 1 — sine-wave GAN, spiral GAN, architecture-variant comparison
3. Part 2.1 — OCTMNIST DCGAN + conditional GAN extension
4. Part 2.2 — CICIDS2017 tabular GAN + all-days extension
5. Part 2.3 — QuickDraw DCGAN + cat/house category extension

Each section saves its figures to `results/` (Part 1) or `results_set1/` (Parts 2.1–2.3) as it runs. The final cell consolidates everything into a single `results.zip`.

## Key results

| Part | Model | Data | Metric | Value |
|---|---|---|---|---|
| 2.1 | DCGAN | OCTMNIST scans | FID (lower = better) | **66.83** |
| 2.1 (ext.) | Conditional DCGAN | OCTMNIST scans | final losses D / G | 0.35 / 4.11 |
| 2.3 | DCGAN | QuickDraw *birthday cake* | FID | **32.59** |
| 2.3 (ext.) | DCGAN | QuickDraw *cat* | FID | **41.09** |
| 2.3 (ext.) | DCGAN | QuickDraw *house* | FID | **24.75** |
| 2.2 | MLP GAN | CICIDS Wednesday (DoS) | mean-gap / std-gap | **0.0723 / 0.3179** |
| 2.2 (ext.) | MLP GAN | CICIDS all days | mean-gap / std-gap | **0.1141 / 0.3544** |

Full discussion, model justifications, and figures are in the accompanying report.

## Notes

- All metrics above are computed directly in the notebook and match the report exactly.
- The notebook runs top-to-bottom without errors; `FAST` mode (a reduced-epoch debug flag in the cybersecurity section) is set to `False` for the runs reported here.
- Package versions are not pinned in `requirements.txt`; if reproducing results exactly, consider freezing versions with `pip freeze > requirements-lock.txt` after a successful run.

## References

Key papers underpinning the training recipe and evaluation methodology: Goodfellow et al. (2014) for the GAN objective, Radford et al. (2016) for DCGAN, Mirza and Osindero (2014) for conditional GANs, Ioffe and Szegedy (2015) for BatchNorm, Kingma and Ba (2015) for Adam, Salimans et al. (2016) for label smoothing, and Heusel et al. (2017) for FID and the two-time-scale update rule. Full reference list with dataset citations (MedMNIST, CICIDS2017, QuickDraw) is in the report.

---
*Coursework submission — Unit 5, GAN assignment.*
