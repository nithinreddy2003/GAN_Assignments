# Generative Adversarial Networks (GANs)

Single notebook (`GAN_Assignment_Set2_Final.ipynb`) covering a from-scratch GAN on
synthetic 2D data plus three real-world GAN applications.

| Part | Content | Framework |
|------|---------|-----------|
| Part 1   | GAN from scratch on 2D data (sine wave, mixture of Gaussians, architecture comparison) | PyTorch |
| Part 2.1 | Medical imaging DCGAN on OCTMNIST, plus a conditional-GAN (cGAN) extension | PyTorch |
| Part 2.2 | Cybersecurity feature-vector GAN on CICIDS 2017 (Wednesday), plus an all-days extension | PyTorch |
| Part 2.3 | Creative DCGAN on QuickDraw 'birthday cake', plus other-category extension | TensorFlow/Keras |

Every figure and metric (loss curves, sample grids, FID scores) is written to a
`results/` folder and zipped at the end for use in the report.

## Requirements

- Python 3.10
- A CUDA-capable GPU is strongly recommended (the notebook falls back to CPU, but the
  image GANs are slow without a GPU).
- Internet access (for the dataset downloads listed below).

Install dependencies:

```bash
pip install -r requirements.txt


The notebook's first cell also installs the packages automatically if they are missing.

## Datasets

- **OCTMNIST (Part 2.1)** - downloaded automatically by the `medmnist` package on first
  use. No manual step needed.
- **QuickDraw (Part 2.3)** - the 'birthday cake' (and extension) categories are fetched
  automatically from Google's public QuickDraw bucket and cached locally.
- **CICIDS 2017 (Part 2.2)** - downloaded automatically by the Part 2.2 "Get the CSV
  files" cell via `kagglehub` from https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset
  This is a public dataset, so no Kaggle account or API token is required; the files are
  fetched directly and cached locally (about 1 GB), so re-runs reuse the copy on disk.

## How to run

1. Install the requirements (or let the setup cell do it).
2. Run the notebook top to bottom on a GPU runtime with internet enabled (all three
   datasets download automatically on first use).

`QUICK_SMOKE_TEST` (near the top of the notebook) can be set to `True` for a fast
end-to-end check with fewer epochs; leave it `False` to reproduce the full report figures.

## Notes

- A fixed random seed is set and re-applied before each training run for reproducibility.
- The two FID scores (OCTMNIST via PyTorch InceptionV3, QuickDraw via Keras InceptionV3)
  are each valid within their own section but are computed with different backbones, so
  they are not directly comparable to one another.
