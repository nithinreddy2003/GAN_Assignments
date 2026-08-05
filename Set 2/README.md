# GAN Assignment

> **TL;DR** — one notebook, five GANs, two frameworks. Open
> `GAN_Assignment_Set2_Final.ipynb` on a GPU runtime with internet on and
> `Restart & Run All`. Every figure/metric lands in `results/` and is zipped at the end.

## Run it in three steps

1. `pip install -r requirements.txt`  *(or just run the first notebook cell — it self-installs)*
2. Set the runtime to **GPU** and make sure you have internet access.
3. `Restart & Run All`. Flip `QUICK_SMOKE_TEST = True` near the top for a fast, low-epoch dry run; leave it `False` to reproduce the reported figures.

## What's inside

- **Part 1 (PyTorch)** — GAN from scratch on 2D data: sine wave → **mixture of Gaussians** → architecture comparison (ReLU baseline vs a wider/deeper **SELU** generator).
- **Part 2.1 (PyTorch)** — OCTMNIST DCGAN + conditional-GAN extension.
- **Part 2.2 (PyTorch)** — CICIDS 2017 feature-vector GAN (Wednesday, BENIGN vs DoS) + all-days extension.
- **Part 2.3 (TensorFlow/Keras)** — QuickDraw *birthday cake* DCGAN + cat/house extension.

## Datasets (all auto-download, nothing to place by hand)

- **OCTMNIST** → via `medmnist`, first run.
- **QuickDraw** → Google's public bucket, cached locally.
- **CICIDS 2017** → via `kagglehub`; it's a **public** dataset so **no Kaggle token is needed** (~1 GB, cached after the first pull).

## Numbers you should see

- OCTMNIST DCGAN → **FID 43.70**
- QuickDraw → **FID** cake **42.13**, cat **39.53**, house **35.22**
- CICIDS alignment (mean / std gap) → Wednesday **0.1387 / 0.2480**, all days **0.1009 / 0.3004**

## Good to know

- A fixed seed is re-applied before every training run, so results are reproducible.
- The two FID scores use different backbones — **PyTorch** InceptionV3 for OCTMNIST, **Keras** InceptionV3 for QuickDraw — so compare them *within* a section, not across sections.
- Tested on Python 3.10 + a single CUDA GPU. It runs on CPU too, just slowly.

Full write-up, model justifications and references: see the report (`.docx`).
