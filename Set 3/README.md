# Set 3 — GAN Coursework (Retinal · Traffic · Sketch)

**Module:** 7COM1079 Coursework 2 · **Notebook:** `GAN_Assignment_Set3_Final.ipynb` · **Runtime:** GPU + internet

---

### Section index

| # | Task | Data | Model | Stack |
|---|------|------|-------|-------|
| 1 | GANs from scratch (2D) | sine · noisy parametric curve `sin(2x)+0.3cos(5x)+ε` · arch study | MLP GAN | PyTorch |
| 2.1 | Medical image synthesis | OCTMNIST | DCGAN + conditional GAN | TensorFlow/Keras |
| 2.2 | Network-traffic synthesis | CICIDS 2017 | Dense feature-vector GAN + full-day run | TensorFlow/Keras |
| 2.3 | Sketch synthesis | QuickDraw (cake · cat · house) | DCGAN | PyTorch |

*Part 1 architecture study:* depth-2 **ReLU** baseline vs depth-4 **ELU** generator, same target.

---

### Fixed parameters

| Field | Value |
|---|---|
| PyTorch/NumPy seed | `GLOBAL_SEED = 2025` |
| TensorFlow seed | `TF_SEED = 2026` |
| Image size | 32 × 32 |
| Evaluation | loss curves · real-vs-fake grids · FID (images) · PCA/t-SNE + moment gaps (tabular) |
| Runtime outputs | `results/` (auto-created, untracked) |

---

### Files

| Path | Purpose |
|---|---|
| `GAN_Assignment_Set3_Final.ipynb` | all modelling, training, figures, metrics |
| `GAN_Assignment_Report_Set 3.docx` | written report (6–8 pp.) — *not modified by the code* |
| `Set 3 Result Images/` | exported figures used in the report |
| `requirements.txt` | dependency list |

---

### Install & run

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Then run all cells in order (the first cell also self-installs the packages).

| Dataset | Retrieval |
|---|---|
| OCTMNIST | `medmnist`, auto on first use |
| CICIDS 2017 | `kagglehub` (Kaggle token needed once); 8 daily CSVs located + merged automatically |
| QuickDraw | direct download from Google Cloud Storage |

---

### Results

| Metric | Value |
|---|---|
| Part 1 sine — D / G at convergence | ≈ 1.38 / 0.70 (near 2ln2 / ln2) |
| 2.1 OCTMNIST DCGAN — FID | **39.23** |
| 2.2 CICIDS Wednesday — mean / std gap | **0.1576 / 0.2067** |
| 2.2 CICIDS all-days — mean / std gap | **0.1300 / 0.1483** |
| 2.3 QuickDraw — FID (house / cake / cat) | **26.44 / 34.29 / 38.79** |

---

### Label note

The brief says "DDoS", but Wednesday's attacks are single-source **DoS** (Hulk, GoldenEye,
slowloris, Slowhttptest). Real DDoS lives in the Friday file and only enters via the
all-days extension; Wednesday is modelled as BENIGN vs DoS as named.
