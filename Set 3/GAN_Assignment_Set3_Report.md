# 7COM1079 Coursework 2 — Building GANs and Applying Them to Retinal, Network-Traffic and Sketch Data

## 1. Overview and Aims

This report documents what was built across the notebook, why each design was chosen, and what the experiments showed. Part 1 builds two GANs from scratch on synthetic 2D data and studies an architecture change; Part 2 applies GANs to three contrasting domains — medical imaging, network-security traffic and hand-drawn sketches — each of which stresses the model differently and so justifies a different design. Two frameworks are used deliberately: PyTorch for the from-scratch and sketch work, TensorFlow/Keras for the medical and tabular models. Seeds are fixed (`GLOBAL_SEED = 2025`, `TF_SEED = 2026`) so every number below is reproducible.

Every model rests on the adversarial idea of Goodfellow *et al.* (2014): a generator maps noise to samples while a discriminator scores how real they look, and the two train in competition under the non-saturating binary cross-entropy objective. Image data uses the convolutional DCGAN of Radford *et al.* (2016); tabular data uses a fully-connected variant. Throughout, the emphasis is on interpreting results rather than restating code.

---

## 2. Part 1 — GANs Built From Scratch on 2D Data

Both networks are small MLPs trained with `BCEWithLogitsLoss`, one discriminator step per generator step, using Adam (2e-4, β = 0.5, 0.999) as recommended for GAN stability (Radford *et al.*, 2016; Kingma and Ba, 2015). One configurable pair of classes serves all three tasks, so the Task 3 comparison is fair — only the changed hyper-parameters differ.

### 2.1 Task 1: Reproducing the Sine-Wave GAN

The target is points on *y* = sin(*x*); training ran 6000 steps from a 32-D latent vector.

![Task 1: sine-wave real vs generated](Set%203%20Images%20Results/43d21613cb93b51d8770dbe7210f5bd53512b101.png)
*Figure 1: Real (blue) versus generated (orange) samples for the sine-wave GAN after 6000 steps.*

Figure 1 shows the generated cloud lying almost exactly on the curve, and the losses agree: the discriminator settles at 1.35–1.39 (final 1.3796) and the generator at 0.68–0.76 (final 0.6979), close to this objective's equilibrium (≈ 2ln2 = 1.386 for D, ln2 = 0.693 for G). Neither loss runs away, indicating genuine convergence rather than one network dominating.

### 2.2 Task 2: A Noisy Parametric Curve

The new target is *y* = sin(2*x*) + 0.3cos(5*x*) + ε (ε a small Gaussian noise term) — harder than the sine because of the higher-frequency term and the noise band; 8000 steps.

![Task 2: noisy parametric curve real vs generated](Set%203%20Images%20Results/6974007aaaaf458258f44b60a844f3ce0ff9eecb.png)
*Figure 2: Real versus generated samples for the noisy parametric curve y = sin(2x) + 0.3cos(5x) + ε.*

Figure 2 shows the generator tracing the double-bump shape and spreading its points into the same noisy band as the real data rather than collapsing to a thin line, so it learns the target's variance and not just its mean trend. Losses stay healthy (D 1.1652–1.3705, G 0.7229–0.8429), a little noisier than Task 1 as expected for a more oscillatory target.

### 2.3 Task 3: Architecture Modification and Comparison

The baseline generator (depth 2, ReLU) is compared against a deeper variant (depth 4, ELU) on the same target and budget, changing only the generator.

![Task 3: baseline versus modified architecture](Set%203%20Images%20Results/a7dce6cb8a6f1dc16ff028d424cf5e31df0895c3.png)
*Figure 3: Generated samples from the baseline (depth 2, ReLU) and modified (depth 4, ELU) generators against the real curve.*

Figure 3 shows a visible but modest gain: the deeper ELU generator sits slightly tighter around the curve with fewer stray points, and its losses stay stable (final D 1.3821, G 0.7089), so the added depth caused no optimisation trouble. That the gain is small is itself informative — a 2D curve is simple enough that the shallow baseline already captures most of the structure.

---

## 3. Part 2 — Applying GANs to Three Real-World Domains

### 3.1 Part 2.1: Retinal OCT Scans with OCTMNIST

**Dataset and approach.** OCTMNIST (Yang *et al.*, 2023) provides 97,477 training scans of 28×28 pixels in four diagnostic classes (choroidal neovascularisation, diabetic macular edema, drusen, normal), rescaled to 32×32 in [−1, 1] to match the generator's `tanh`. A DCGAN (Radford *et al.*, 2016) suits the strong local structure of a scan — a curved retinal band on a dark background — which convolutions capture far more efficiently than a dense network. The generator upsamples a 100-D latent vector via transposed convolutions; the discriminator mirrors it with strided convolutions and 0.3 dropout. Training used a manual `tf.GradientTape` loop for 35 epochs with one-sided label smoothing (real labels 0.9), which stops the discriminator over-confidently starving the generator of gradient (Salimans *et al.*, 2016).

![OCTMNIST class distribution](Set%203%20Images%20Results/b9fe7580670ba1e0790b7aa7d35d642c09f1bc68.png)
*Figure 4: Class distribution of the OCTMNIST training split.*

Figure 4 shows the four classes are unevenly represented. This is relevant later: an imbalanced training set is the condition under which a class-conditional generator gives uneven quality.

![OCT training losses](Set%203%20Images%20Results/a1aace33aaab2560b9a12a1df4301d1347543d6f.png)
*Figure 5: Generator and discriminator loss curves over the 35-epoch DCGAN run.*

Figure 5 shows neither network diverging — the discriminator holds around 1.1–1.34 and the generator around 0.75–1.28, oscillating within a stable band. That stability is what makes the sample quality below meaningful rather than accidental.

![OCT real versus generated scans](Set%203%20Images%20Results/f4f9c6377442ee619def914e84d4394b8e79cf14.png)
*Figure 6: Real (left) versus generated (right) OCT scans from the trained generator.*

Figure 6 shows the generator reproduces the main anatomy with real variation in band position and curvature, so it is not memorising one image; its weakness is fine detail — the fakes are softer and lose sub-layer texture and speckle. It scores **FID = 39.20** (Heusel *et al.*, 2017), a believable mid-range value for a 35-epoch DCGAN, though read cautiously: FID passes 28×28 grayscale (upscaled) through an ImageNet-trained InceptionV3 (Szegedy *et al.*, 2016), a domain it was never calibrated for, so it is indicative rather than absolute.

**Extension — conditional GAN.** The DCGAN was made conditional (Mirza and Osindero, 2014) by concatenating a class embedding onto the latent vector and feeding the label to the discriminator as an extra channel.

![OCT conditional GAN, one class per row](Set%203%20Images%20Results/c3d597b0e96011b073ebed14a9d482b9822d77d4.png)
*Figure 7: Class-conditional samples, one diagnostic class per row.*

Figure 7 shows visibly different scans per row, so the label steers the output rather than being ignored; better-represented classes look sharper — the direct effect of the imbalance in Figure 4.

**On GANs and "ducks".** The duck test — if it looks and quacks like a duck, assume it is one — captures a real risk here. The discriminator only judges whether an image *looks* real, never whether it is anatomically correct, so nothing stops the generator inventing convincing structure that matches no real scan (a bright region reading as pathology, or a smoothed-over feature). Synthetic OCT images are therefore fine for augmentation or teaching but never diagnostic ground truth, and any model trained on mixed data should be re-validated on real-only data.

### 3.2 Part 2.2: Network-Traffic Synthesis with CICIDS 2017

**Dataset and approach.** CICIDS 2017 (Sharafaldin *et al.*, 2018) was pulled via `kagglehub` and its eight daily files combined into **2,830,743 flows × 80 columns** (**78 numeric features**). Because each flow is a feature vector, a dense GAN replaces the DCGAN — there is no spatial axis for convolutions. The generator widens 128→256 (batch norm, ReLU) ending in a linear layer (standardised features are unbounded, so no `tanh`); the discriminator narrows 256→128 (LeakyReLU, dropout). Latent dimension 48, 60 epochs.

![CICIDS 2017 class distribution](Set%203%20Images%20Results/7346a93d7e2feb018a303868ac6d292479989d81.png)
*Figure 8: Class distribution across all days (log scale) and flows per day.*

Figure 8 shows severe imbalance: BENIGN dominates at 2,273,097 flows while Heartbleed (11), SQL Injection (21) and Infiltration (36) barely appear, hence the log scale. Per the brief, the model trained on the **Wednesday file only** (691,406 rows: 440,031 BENIGN plus DoS Hulk, GoldenEye, slowloris and Slowhttptest) using a sampled 20,000 rows, standardised and clipped to ±5σ to stop extreme outliers (and 4,376 infinite, 1,358 missing values) destabilising training.

![CICIDS feature-vector GAN losses](Set%203%20Images%20Results/a1a417695bf2338ac5ed485041aa5ddf73e9f08f.png)
*Figure 9: Discriminator and generator loss curves for the Wednesday feature-vector GAN.*

Figure 9 shows stable training: the discriminator drifts from about 1.13 to 0.87 while the generator rises from about 1.11 to 1.73 — a rising generator loss against a controlled discriminator, typical of a dense GAN settling.

![CICIDS PCA of real versus synthetic](Set%203%20Images%20Results/f16ce793d3f35c24631bcacee54c7c7545011b4c.png)
*Figure 10: PCA projection of real BENIGN, real DoS and synthetic feature vectors.*

![CICIDS t-SNE of real versus synthetic](Set%203%20Images%20Results/4d72fcc9500b3ca8262ceaf517837f081f0dd6b3.png)
*Figure 11: t-SNE projection of real versus synthetic feature vectors.*

Figures 10 and 11 agree: in both the linear PCA and non-linear t-SNE views (van der Maaten and Hinton, 2008) the synthetic points sit inside the main mass of real traffic rather than forming a separate blob, so the generator found the right region of feature space. However, the synthetic cloud is tighter than the real one and misses the tails. The alignment gaps quantify this — mean absolute per-feature error of **0.1227** for means and **0.2780** for standard deviations — so averages are matched well but spread is understated.

**Extension — full multi-day dataset.** Retraining on 30,000 rows from all eight days *improved* both gaps (**0.1059** / **0.2669** versus 0.1227 / 0.2780). A single unconditional generator would usually fit worse across many attack families, so the extra data volume appears to offset the added diversity; the model still leans towards BENIGN and the high-volume attacks, and the rare classes stay under-represented. Note the brief's "DDoS" label — Wednesday's attacks are single-source DoS, and true DDoS only appears in the Friday file, which is included here.

**On "ducks".** The duck test bites harder on tabular data: a generated flow only has to look statistically like traffic, not behave like a real attack. Used to augment an intrusion-detection set, a fake "attack" matching no real signature could teach the wrong pattern and miss real attacks, while a drifting "benign" row could raise false alarms. Since 78 numbers cannot be eyeballed like an image, the per-feature gaps — especially the standard-deviation gap where such drift shows first — are the main safeguard.

### 3.3 Part 2.3: Sketch Generation with QuickDraw 'birthday cake'

**Dataset and approach.** The QuickDraw *birthday cake* set (Ha and Eck, 2018) provides **144,982** 28×28 bitmaps. A PyTorch DCGAN with a 128-D latent vector trained for 30 epochs, using the DCGAN weight initialisation of Radford *et al.* (2016) and label smoothing; a convolutional model suits the local stroke structure.

![QuickDraw birthday cake real versus generated](Set%203%20Images%20Results/4d29b02fce9393f50c92f8f1ea474b7d36994357.png)
*Figure 12: Real (left) versus generated (right) 'birthday cake' sketches.*

Figure 12 shows the generated cakes are clearly recognisable — a body shape with candle-like marks on top, some with a plate or base line — though a few have broken strokes or speckle. The silhouette is consistent across the grid with no mode collapse. Training was stable apart from a brief epoch-13 spike (D 1.633, G 11.493) that recovered, and it scored **FID = 34.29**.

**Extension — other categories and complexity.** The same pipeline ran on *cat* and *house*, comparing FID against an ink-density complexity proxy.

![QuickDraw extension: FID versus ink-density complexity](Set%203%20Images%20Results/06c9fd4841a65a62cdc062a7fb955cb0d87b04ba.png)
*Figure 13: FID by category alongside the ink-density complexity proxy.*

Figure 13 orders ink density as house (0.1725) < cat (0.1986) < cake (0.1989) and FID as house (**26.44**) < cake (**34.29**) < cat (**38.79**). The sparsest category (house) is the easiest, supporting the idea that emptier sketches are simpler; but cat and cake have near-identical ink density while cat scores worse, so ink alone does not explain difficulty. Cat sketches vary far more in pose and detail (ears, whiskers, body angle), and it is that structural variety — not ink volume — that makes the category harder.

---

## 4. Reflections and Future Directions

The same non-saturating objective was applied throughout, but the architecture was matched to the data each time: dense MLPs for 2D points and tabular traffic, convolutional DCGANs for images and sketches, and a conditional variant for class control. The from-scratch models reached near-theoretical equilibria; the DCGANs produced recognisable OCT scans (FID 39.20) and cake sketches (FID 34.29); and the tabular GAN placed synthetic flows within the real distribution (gaps 0.1227 / 0.2780, improving to 0.1059 / 0.2669 across all days). The recurring limitation is under-dispersion — central tendency is reproduced more faithfully than variance and fine detail — together with uneven quality on rare classes.

Future work follows from these weaknesses: conditioning the CICIDS generator on attack family (as done for OCT) would help it serve rare classes rather than defaulting to BENIGN; a Wasserstein objective with gradient penalty should improve tail coverage and the std-gap; and a domain-appropriate feature extractor would give a fairer image score than ImageNet InceptionV3. Above all, the duck-test analysis means any downstream use of this synthetic data — medical augmentation or intrusion-detection training — must be validated against real-only data before it is trusted.

---

## References

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y. (2014) 'Generative adversarial nets', *Advances in Neural Information Processing Systems*, 27, pp. 2672–2680.

Ha, D. and Eck, D. (2018) 'A neural representation of sketch drawings', *International Conference on Learning Representations (ICLR)*.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B. and Hochreiter, S. (2017) 'GANs trained by a two time-scale update rule converge to a local Nash equilibrium', *Advances in Neural Information Processing Systems*, 30, pp. 6626–6637.

Kingma, D. P. and Ba, J. (2015) 'Adam: a method for stochastic optimization', *International Conference on Learning Representations (ICLR)*.

Mirza, M. and Osindero, S. (2014) 'Conditional generative adversarial nets', *arXiv preprint* arXiv:1411.1784.

Radford, A., Metz, L. and Chintala, S. (2016) 'Unsupervised representation learning with deep convolutional generative adversarial networks', *International Conference on Learning Representations (ICLR)*.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A. and Chen, X. (2016) 'Improved techniques for training GANs', *Advances in Neural Information Processing Systems*, 29, pp. 2234–2242.

Sharafaldin, I., Lashkari, A. H. and Ghorbani, A. A. (2018) 'Toward generating a new intrusion detection dataset and intrusion traffic characterization', *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, pp. 108–116.

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. and Wojna, Z. (2016) 'Rethinking the Inception architecture for computer vision', *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2818–2826.

van der Maaten, L. and Hinton, G. (2008) 'Visualizing data using t-SNE', *Journal of Machine Learning Research*, 9, pp. 2579–2605.

Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H. and Ni, B. (2023) 'MedMNIST v2 — a large-scale lightweight benchmark for 2D and 3D biomedical image classification', *Scientific Data*, 10(1), p. 41.
