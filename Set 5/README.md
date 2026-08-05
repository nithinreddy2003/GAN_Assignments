# GAN Coursework FAQ

Everything you need to understand and re-run `GAN_Assignment_Set5_Final.ipynb`, written as
a set of questions.

### What is this, in one sentence?
One notebook that first builds a GAN from scratch on synthetic 2D data, then applies GANs to
three real-world domains — medical imaging (OCTMNIST), cybersecurity (CICIDS 2017) and
creative sketching (QuickDraw).

### What happens in Part 1?
A from-scratch PyTorch GAN driven by a `SpectralGanTrainer` — a **LayerNorm generator** paired
with a **spectral-normalised critic**, trained with the non-saturating loss, a two-step critic
schedule and one-sided label smoothing. It's run on three targets: the tutorial **sine wave**,
a **nine-mode mixture of Gaussians laid out on a 3×3 grid**, and an architecture study
comparing a 2-layer **LeakyReLU** generator against a deeper 3-layer **GELU** generator.

### Which framework does each Part 2 task use, and why?
Whichever suits the data type:

| Section | Domain | Dataset | Model | Framework |
|---|---|---|---|---|
| 2.1 | Medicine | [OCTMNIST](https://medmnist.com/) | DCGAN + conditional GAN | TensorFlow/Keras |
| 2.2 | Cybersecurity | [CICIDS 2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/data) | Dense feature-vector GAN | PyTorch |
| 2.3 | Creative AI | [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) (cake/cat/house) | DCGAN | TensorFlow/Keras |

The two image GANs share one reusable `GanTrainer` (a `keras.Model` subclass trained via
`model.fit`). Part 2.2 scales its features with a **RobustScaler** before training.

### How is the conditional-GAN extension (2.1) built?
It uses **late-fusion** conditioning — the class label is mixed in only at the final dense
layer — together with instance noise and a two-time-scale rule so the critic doesn't overpower
the generator. Result: you can ask it for a specific diagnostic class and get one class per grid row.

### How do I install it?
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
A CUDA GPU is recommended. The notebook's first cell also self-installs the packages, so on
Colab you can just open it and run.

### How do I run it, and where do the datasets come from?
Run all cells in order (`Kernel → Restart & Run All`).
- **OCTMNIST** downloads via the `medmnist` package on first use.
- **CICIDS 2017** is fetched with `kagglehub` from the public `chethuhn/network-intrusion-dataset` (or read from a local folder if present); the eight daily CSVs are located and combined automatically.
- **QuickDraw** bitmaps download directly from Google Cloud Storage.
- Seeds are fixed (`SEED = 5050`, `KERAS_SEED = 6060`) for reproducibility. Figures are written to an auto-created `results/` folder.

### What numbers should I expect?

| Model | Metric | Value |
|---|---|---|
| 2.1 OCTMNIST DCGAN | FID (lower = better) | **17.89** |
| 2.2 CICIDS Wednesday (BENIGN vs DoS) | mean / std alignment gap | **0.2435 / 0.2395** |
| 2.2 CICIDS all days (extension) | mean / std alignment gap | **0.2955 / 0.5277** |
| 2.3 QuickDraw | FID — cake / cat / house | **25.56 / 28.51 / 41.98** |

All values are computed in the notebook and match the report.

### Wait — the brief says "DDoS", but the labels say "DoS"?
Correct. The Wednesday file's attacks are single-source **DoS** (Hulk, GoldenEye, slowloris,
Slowhttptest). Genuine DDoS traffic lives in the Friday file and only enters through the
all-days extension, so Wednesday is modelled as BENIGN vs DoS as named.

### Where's the write-up?
In the accompanying report (`GAN_Assignment_Set5_Report.docx`), with exported figures in
`Set 5 Result Images/`.

### References?
Goodfellow et al. (2014) — GAN objective; Radford et al. (2016) — DCGAN; Mirza & Osindero
(2014) — conditional GAN; Miyato et al. (2018) — spectral normalisation; Kingma & Ba (2015) —
Adam; Heusel et al. (2017) — FID and the two-time-scale rule. Full list is in the report.
