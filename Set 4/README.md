# Set 4 — GANs from Scratch and Across Three Domains

This notebook (`GAN_Assignment_Set4_Final.ipynb`) works through the assignment task by task.
The checklists below map each requirement to what the notebook actually does.

## Part 1 — from scratch (PyTorch, `Toy2DGAN` trainer)
- [x] Reproduce the tutorial **sine-wave** GAN
- [x] New 2D distribution → a **2D spiral**
- [x] Architecture change → depth-3 **GELU** generator vs a deeper depth-5 **LeakyReLU** generator, compared side by side

## Part 2.1 — OCTMNIST retinal scans (PyTorch)
- [x] Load & explore the dataset (class balance + sample grid)
- [x] ConvNet DCGAN — **resize-convolution** generator (upsample + 3×3 conv, avoids checkerboard artefacts)
- [x] Track generator/discriminator losses
- [x] Generate fakes; compare vs real visually **and** by FID → **31.94**
- [x] *Extension:* conditional GAN with a **spectral-normalised critic** + wrong-label trick → one class per grid row

## Part 2.2 — CICIDS 2017 traffic (TensorFlow/Keras, `keras.Model` subclass)
- [x] Combine the eight daily CSVs; inspect features & class balance
- [x] Feature-vector GAN (not images)
- [x] Train on **Wednesday** (BENIGN vs DoS); monitor loss curves
- [x] Compare real vs synthetic with **PCA and t-SNE**
- [x] Alignment gaps → **0.4792 / 0.5522** (Wednesday)
- [x] *Extension:* all-days run → gaps **0.5675 / 0.6285** + generalisation discussion

## Part 2.3 — QuickDraw sketches (PyTorch)
- [x] Download & preview *birthday cake*
- [x] ConvNet DCGAN; track samples across epochs
- [x] Generate + compare vs real; FID → cake **41.01**
- [x] *Extension:* more categories → FID cat **67.36**, house **28.76** + complexity discussion

## How to run
```bash
pip install -r requirements.txt      # or let the notebook's first cell do it
```
Open the notebook on a GPU runtime with internet, then `Restart & Run All`.
- OCTMNIST → `medmnist` (auto). CICIDS 2017 → `kagglehub` (public `chethuhn/network-intrusion-dataset`). QuickDraw → Google Cloud Storage.
- Seeds: `TORCH_SEED = 3030`, `KERAS_SEED = 4040`. Figures are written to `gan_results/` (auto-created).
- Result figures used in the report are in `Set 4 Result Images/`.

## Label note
Wednesday's attacks are single-source **DoS** (the brief says "DDoS"); true DDoS is in the
Friday file and only appears in the all-days extension. Wednesday is modelled as BENIGN vs DoS.

## Key references
Goodfellow et al. (2014); Radford et al. (2016, DCGAN); Mirza & Osindero (2014, cGAN);
Miyato et al. (2018, spectral norm); Odena et al. (2016, resize-conv / checkerboard);
Kingma & Ba (2015, Adam); Heusel et al. (2017, FID). Full list in the report.
