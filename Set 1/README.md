# Generative Adversarial Networks: From 2D Toy Data to Real-World Applications

## Abstract

This submission develops generative adversarial networks (GANs) from first principles and
then applies them across three contrasting domains — medical imaging, network-security
telemetry and freehand sketches. Everything lives in a single notebook,
`GAN_Assignment_Set1_Final.ipynb`, which runs top to bottom and produces every figure and
metric cited in the report.

## Part 1 — Foundations (PyTorch)

Part 1 implements a vanilla GAN by hand and studies its behaviour on three targets in turn:
the tutorial **sine wave**, a **2D Archimedean spiral**, and finally a **deeper LeakyReLU
generator** compared against the ReLU baseline on that same spiral. One training recipe is
held fixed throughout — BatchNorm in the generator, Adam (β₁ = 0.5, β₂ = 0.999) and
one-sided label smoothing — so that the effect of the architecture change can be read off
cleanly rather than confounded with optimiser or schedule differences.

## Part 2 — Applications

The three real-world problems are each tackled with the framework best suited to the data,
which is why the notebook deliberately moves between PyTorch and TensorFlow/Keras:

| Section | Domain | Dataset | Model | Framework |
|---|---|---|---|---|
| 2.1 | Medicine | [OCTMNIST](https://medmnist.com/) (MedMNIST v2) | DCGAN + conditional GAN | TensorFlow/Keras |
| 2.2 | Cybersecurity | [CICIDS 2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) | Fully-connected (tabular) GAN | PyTorch |
| 2.3 | Creative AI | [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) *birthday cake* (+ cat / house) | DCGAN | TensorFlow/Keras |

## Reproducing the work

Create an environment and install the pinned dependency set, then run the notebook end to
end (`Kernel → Restart & Run All`):

```bash
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pulls `tensorflow[and-cuda]`, which assumes an NVIDIA GPU; on CPU-only
hardware replace it with plain `tensorflow`. The datasets are obtained automatically at
run time: **OCTMNIST** via the `medmnist` package, **QuickDraw** via a direct request to
Google's public bucket, and **CICIDS 2017** via `kagglehub` (which needs a one-time Kaggle
API token at `~/.kaggle/kaggle.json`). Figures are written to `results/` and `results_set1/`
and consolidated into a `results.zip` by the final cell.

## Results obtained

The unconditional OCTMNIST DCGAN reaches an FID of **66.83**; the QuickDraw models score
**32.59** (birthday cake), **41.09** (cat) and **24.75** (house). On CICIDS 2017 the
tabular GAN achieves mean/standard-deviation alignment gaps of **0.0723 / 0.3179** on the
Wednesday split and **0.1141 / 0.3544** across all days, and the conditional retinal GAN
settles at final discriminator/generator losses of about **0.35 / 4.11** while producing a
distinct scan for each requested diagnostic class. All numbers are generated in the notebook
and mirrored in the report.

## References

The methodology rests on Goodfellow et al. (2014, adversarial objective), Radford et al.
(2016, DCGAN), Mirza & Osindero (2014, conditional GANs), Ioffe & Szegedy (2015, BatchNorm),
Kingma & Ba (2015, Adam), Salimans et al. (2016, label smoothing) and Heusel et al. (2017,
FID). Dataset citations (MedMNIST, CICIDS 2017, QuickDraw) and the full reference list appear
in the accompanying report.
