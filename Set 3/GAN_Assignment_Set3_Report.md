# 7COM1079 Coursework 2 — Generative Adversarial Networks

## Introduction

This report accompanies the GAN notebook and documents what was built, why each design choice was made, and what the experiments actually showed. Part 1 builds two generative adversarial networks (GANs) from scratch on synthetic two-dimensional data and studies how an architecture change affects them. Part 2 applies GANs to three very different real-world problems — medical imaging, network-security traffic, and hand-drawn sketches — because each domain stresses the model in a different way and therefore justifies a different design. Two frameworks were used deliberately: PyTorch for the from-scratch and sketch work, and TensorFlow/Keras for the medical and tabular models, with fixed seeds (`GLOBAL_SEED = 2025`, `TF_SEED = 2026`) so every figure and number quoted below is reproducible.

The underlying method throughout is the adversarial game of Goodfellow *et al.* (2014): a generator learns to map noise to data while a discriminator learns to separate real from generated samples, and the two are trained against each other with the non-saturating binary cross-entropy objective. Where the data is imagery, the convolutional DCGAN formulation of Radford *et al.* (2016) is used; where it is tabular, a fully-connected variant is used instead. This report prioritises interpretation of the results over description of the code.

---

## Part 1: Building and Understanding GANs from Scratch

Part 1 was implemented entirely in PyTorch. Both networks are small multilayer perceptrons trained with `BCEWithLogitsLoss`, one discriminator update per generator update, using Adam (learning rate 2e-4, β = 0.5, 0.999) as recommended for GAN stability by Radford *et al.* (2016) and Kingma and Ba (2015). A single configurable pair of classes serves all three tasks, which keeps the comparison in Task 3 fair because only the changed hyper-parameters differ.

### Task 1: Reproduce the sine-wave GAN

The target is points sampled along *y* = sin(*x*). Training ran for 6000 steps from a 32-dimensional latent vector.

![Task 1: sine-wave real vs generated](Set%203%20Images%20Results/43d21613cb93b51d8770dbe7210f5bd53512b101.png)
*Figure 1: Real (blue) versus generated (orange) samples for the sine-wave GAN after 6000 steps.*

Figure 1 shows the generated cloud lying almost exactly on the sine curve, and the loss trace supports this rather than contradicting it. The discriminator settles at roughly 1.35–1.39 (final 1.3796) and the generator at roughly 0.68–0.76 (final 0.6979), which is close to the theoretical equilibrium for this objective — about 2ln2 ≈ 1.386 for D and ln2 ≈ 0.693 for G. Neither network runs away from the other, so the model has reached the balanced state that indicates genuine convergence rather than one side dominating.

### Task 2: Create and model a new 2D distribution

The chosen distribution is the noisy parametric curve *y* = sin(2*x*) + 0.3cos(5*x*) + ε, with ε a small Gaussian noise term. This is harder than the plain sine because of the higher-frequency second term and the noise band, and it ran for 8000 steps.

![Task 2: noisy parametric curve real vs generated](Set%203%20Images%20Results/6974007aaaaf458258f44b60a844f3ce0ff9eecb.png)
*Figure 2: Real versus generated samples for the noisy parametric curve y = sin(2x) + 0.3cos(5x) + ε.*

Figure 2 shows the generator tracing the full double-bump shape and, importantly, spreading its points into the same noisy band as the real data rather than collapsing onto a single thin line. This means it has learned the variance of the target and not merely its mean trend. The losses stay healthy across the run (D between 1.1652 and 1.3705, G between 0.7229 and 0.8429) and are slightly noisier than in Task 1, which is the expected consequence of a more oscillatory, higher-frequency target.

### Task 3: Modify the GAN architecture and compare

To test sensitivity to architecture, the baseline generator (depth 2, ReLU) was compared against a deeper variant (depth 4, ELU) on the same target and training budget, changing only the generator.

![Task 3: baseline versus modified architecture](Set%203%20Images%20Results/a7dce6cb8a6f1dc16ff028d424cf5e31df0895c3.png)
*Figure 3: Generated samples from the baseline (depth 2, ReLU) and modified (depth 4, ELU) generators against the real curve.*

Figure 3 shows a visible but modest improvement: the deeper ELU generator sits slightly tighter around the curve and leaves fewer stray points drifting off the shape. Its losses remained stable (final D 1.3821, G 0.7089), so the extra depth did not introduce optimisation problems. The improvement being incremental rather than dramatic is itself informative — a 2D curve is simple enough that the shallow baseline already captures most of the structure, so added capacity has limited room to help.

---

## Part 2: Real-World GAN Applications

### Part 2.1: Optical coherence tomography (OCT) retinal images with MedMNIST

**Dataset and approach.** The OCTMNIST subset of MedMNIST (Yang *et al.*, 2023) provides 97,477 training scans of 28×28 pixels across four diagnostic classes: choroidal neovascularisation, diabetic macular edema, drusen, and normal. Images were rescaled to 32×32 in the [−1, 1] range to match the generator's `tanh` output. A DCGAN (Radford *et al.*, 2016) was chosen because OCT scans have strong local spatial structure — a curved retinal band over a dark background — which convolutions capture far more efficiently than a dense network. The generator upsamples a 100-dimensional latent vector through transposed convolutions; the discriminator mirrors this with strided convolutions and dropout of 0.3. Training used a manual `tf.GradientTape` loop for 35 epochs with one-sided label smoothing (real labels set to 0.9), a technique from Salimans *et al.* (2016) that stops the discriminator becoming over-confident and starving the generator of gradient.

![OCTMNIST class distribution](Set%203%20Images%20Results/b9fe7580670ba1e0790b7aa7d35d642c09f1bc68.png)
*Figure 4: Class distribution of the OCTMNIST training split.*

Figure 4 shows the dataset is imbalanced, with the four classes represented very unequally. This matters later: an imbalanced training set is exactly the condition under which a class-conditional generator produces uneven quality, and it frames the interpretation of the conditional results below.

![OCT training losses](Set%203%20Images%20Results/a1aace33aaab2560b9a12a1df4301d1347543d6f.png)
*Figure 5: Generator and discriminator loss curves over the 35-epoch DCGAN run.*

Figure 5 shows neither network diverging. The discriminator stays around 1.1–1.34 and the generator around 0.75–1.28, oscillating within a stable band rather than collapsing — the healthy adversarial signature. This stability is what makes the sample quality in Figure 6 believable rather than accidental.

![OCT real versus generated scans](Set%203%20Images%20Results/f4f9c6377442ee619def914e84d4394b8e79cf14.png)
*Figure 6: Real (left) versus generated (right) OCT scans from the trained generator.*

Figure 6 shows the generator reproduces the main anatomy — a bright, curved retinal band on a darker background — with genuine variation in band position and curvature across the batch, so it is not memorising one image. The clear weakness is fine detail: the fakes look softer than the real scans and lose the sharp sub-layer texture and speckle. Quantitatively, the model scores **FID = 39.20** (Heusel *et al.*, 2017). This is a believable mid-range value for a 35-epoch DCGAN, but it should be read cautiously: FID here pushes 28×28 medical grayscale (upscaled to 32×32) through an InceptionV3 (Szegedy *et al.*, 2016) trained on natural ImageNet photographs, a domain the metric was never calibrated for, so it is best treated as an indicator rather than an absolute benchmark.

**Extension — conditional GAN.** For extra credit the DCGAN was made conditional (Mirza and Osindero, 2014) by concatenating a class embedding onto the latent vector and feeding the label to the discriminator as an extra channel.

![OCT conditional GAN, one class per row](Set%203%20Images%20Results/c3d597b0e96011b073ebed14a9d482b9822d77d4.png)
*Figure 7: Class-conditional samples, one diagnostic class per row.*

Figure 7 shows visibly different scans from row to row, confirming the generator uses the label rather than ignoring it. Quality is uneven across classes, and the better-represented classes look sharper — the direct consequence of the imbalance seen in Figure 4.

**On GANs and "ducks".** The brief asks for a note on GANs "generating ducks". The reference is the duck test: if something looks like a duck and quacks like a duck, it is assumed to be one. A GAN exploits exactly this shortcut, because the discriminator only ever judges whether an image *looks* real, never whether it is anatomically correct. Nothing in training prevents the generator from drawing a convincing structure that corresponds to no real scan — inventing a bright region that reads like pathology, or smoothing over a genuine feature. This is precisely why synthetic OCT images are useful for augmentation or teaching but must never be treated as diagnostic ground truth, and why any classifier trained on mixed real and synthetic data should be re-validated on real-only data.

### Part 2.2: Cybersecurity — Synthetic Traffic with CICIDS 2017

**Dataset and approach.** CICIDS 2017 (Sharafaldin *et al.*, 2018) was downloaded directly through `kagglehub` and the eight per-day capture files were combined into a single frame of **2,830,743 flows × 80 columns**, of which **78 are numeric features**. Because each flow is a feature vector rather than an image, a dense (fully-connected) GAN was used instead of a DCGAN — convolutions have no meaningful spatial axis to exploit here. The generator widens 128→256 with batch normalisation and ReLU and ends in a linear layer (the standardised features are unbounded, so no `tanh`), and the discriminator narrows 256→128 with LeakyReLU and dropout. The latent dimension is 48 and training ran for 60 epochs.

![CICIDS 2017 class distribution](Set%203%20Images%20Results/7346a93d7e2feb018a303868ac6d292479989d81.png)
*Figure 8: Class distribution across all days (log scale) and flows per day.*

Figure 8 shows severe class imbalance: BENIGN dominates at 2,273,097 flows, while rare attacks such as Heartbleed (11), SQL Injection (21) and Infiltration (36) barely appear. The log scale is necessary precisely because of this spread. Following the brief, the model was trained on the **Wednesday file only** (691,406 rows: 440,031 BENIGN plus DoS Hulk, GoldenEye, slowloris and Slowhttptest), on a sampled 20,000 rows. Features were standardised and clipped to ±5σ, which was essential to stop CICIDS's extreme outliers (and 4,376 infinite / 1,358 missing values) from destabilising training.

![CICIDS feature-vector GAN losses](Set%203%20Images%20Results/a1a417695bf2338ac5ed485041aa5ddf73e9f08f.png)
*Figure 9: Discriminator and generator loss curves for the Wednesday feature-vector GAN.*

Figure 9 shows stable training: the discriminator drifts down from about 1.13 to 0.87 while the generator rises from about 1.11 to 1.73. The steadily rising generator loss against a controlled discriminator is typical of a dense GAN settling, and matches the reasonable distributional overlap seen next.

![CICIDS PCA of real versus synthetic](Set%203%20Images%20Results/f16ce793d3f35c24631bcacee54c7c7545011b4c.png)
*Figure 10: PCA projection of real BENIGN, real DoS and synthetic feature vectors.*

![CICIDS t-SNE of real versus synthetic](Set%203%20Images%20Results/4d72fcc9500b3ca8262ceaf517837f081f0dd6b3.png)
*Figure 11: t-SNE projection of real versus synthetic feature vectors.*

Figures 10 and 11 tell a consistent story. In both the linear PCA view and the non-linear t-SNE view (van der Maaten and Hinton, 2008), the synthetic points fall inside the main mass of real traffic rather than forming a separate blob, so the generator has found the correct region of feature space. However, the synthetic cloud is noticeably tighter than the real one and does not reach into the tails. The alignment-gap metrics quantify this: on the Wednesday data the mean absolute error in per-feature means is **0.1227** while the error in per-feature standard deviations is **0.2780**, meaning the generator matches average feature values well but understates how far each feature actually spreads.

**Extension — full multi-day dataset.** Retraining the same model on 30,000 rows sampled from all eight days gave a mildly counter-intuitive result: the alignment gaps *improved* on both statistics (mean **0.1059**, std **0.2669** versus 0.1227 and 0.2780 for Wednesday only). A single unconditional generator would normally fit *worse* when asked to cover many attack families at once, so the extra data volume appears to offset the extra diversity here. The model almost certainly still leans towards BENIGN and the high-volume attacks; the rare classes (Heartbleed, Infiltration, SQL Injection) are too sparse to be represented well even after combining every day. It is worth noting the brief's "DDoS" label: the Wednesday attacks are single-source DoS, and genuine DDoS traffic only appears in the Friday file, which is included in this extension.

**On "ducks".** The duck test applies even more sharply to tabular traffic. A generated flow only needs to look statistically like real traffic to satisfy the discriminator, which is a far weaker bar than behaving like a real attack. If such data were used to augment an intrusion-detection dataset, a fake "attack" matching no real signature could teach a classifier the wrong pattern and let real attacks slip through, while a drifting "benign" row could raise false alarms. Because 78 numeric features cannot be sanity-checked by eye the way an image can, the per-feature alignment gaps — especially the standard-deviation gap where such drift first appears — are the main defence against this failure mode.

### Part 2.3: Creative AI — QuickDraw 'birthday cake' Subset

**Dataset and approach.** The QuickDraw *birthday cake* category (Ha and Eck, 2018) provides **144,982** 28×28 bitmaps. A PyTorch DCGAN with a 128-dimensional latent vector was trained for 30 epochs using the DCGAN weight initialisation of Radford *et al.* (2016) and one-sided label smoothing. A convolutional model suits sketches because, like the OCT scans, they carry local stroke structure.

![QuickDraw birthday cake real versus generated](Set%203%20Images%20Results/4d29b02fce9393f50c92f8f1ea474b7d36994357.png)
*Figure 12: Real (left) versus generated (right) 'birthday cake' sketches.*

Figure 12 shows the generated cakes are clearly recognisable — most have a body shape with candle-like marks on top, and some include a plate or base line. They are not all clean (a few broken or doubled strokes, some background speckle), but the core silhouette is consistent across the grid, with no sign of mode collapse. Training was stable overall despite a brief spike at epoch 13 (D 1.633, G 11.493) that quickly recovered, and the model scored **FID = 34.29**.

**Extension — other categories and sketch complexity.** The same pipeline was run on *cat* and *house* and the FID compared against an ink-density proxy for sketch complexity.

![QuickDraw extension: FID versus ink-density complexity](Set%203%20Images%20Results/06c9fd4841a65a62cdc062a7fb955cb0d87b04ba.png)
*Figure 13: FID by category alongside the ink-density complexity proxy.*

Figure 13 shows ink density ordered as house (0.1725) < cat (0.1986) < cake (0.1989), and FID as house (**26.44**) < cake (**34.29**) < cat (**38.79**). The sparsest category, house, is both the least inked and the easiest to model, supporting the intuition that emptier sketches are simpler. But cat and cake have almost identical ink density while cat scores clearly worse, so ink alone does not explain difficulty. The most plausible reading is that cat sketches vary far more in pose and internal detail (ears, whiskers, body angle) than cakes, and it is this structural variety, not the amount of ink, that makes the category harder.

---

## Conclusion and Future Work

Across all four experiments the same non-saturating adversarial objective was applied, but the architecture was matched to the data every time: dense MLPs for 2D points and tabular traffic, convolutional DCGANs for images and sketches, and a conditional variant where class control was needed. The from-scratch models converged to near-theoretical equilibria; the DCGANs produced recognisable OCT scans (FID 39.20) and birthday-cake sketches (FID 34.29); and the tabular GAN placed synthetic flows convincingly within the real distribution (alignment gaps 0.1227 / 0.2780, improving to 0.1059 / 0.2669 on the full dataset). The recurring limitation was under-dispersion — generators reproduce central tendency more faithfully than variance and fine detail — and, for imbalanced datasets, uneven quality across rare classes.

Future work follows directly from these weaknesses. Conditioning the CICIDS generator on attack family (as done for OCT) would let it allocate capacity to rare classes rather than defaulting to BENIGN; a Wasserstein objective with gradient penalty would likely improve tail coverage and the std-gap; and a domain-appropriate feature extractor for FID would give a fairer image score than ImageNet InceptionV3. Most importantly, the "duck test" analysis argues that any downstream use of this synthetic data — medical augmentation or IDS training — must be validated against real-only data before it is trusted.

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
