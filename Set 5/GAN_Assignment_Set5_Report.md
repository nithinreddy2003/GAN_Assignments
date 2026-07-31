# 7COM1079 Coursework 2 — Generative Adversarial Networks

## Introduction

A generative adversarial network (GAN) trains a generator against a discriminator; at the equilibrium where the discriminator can no longer separate real from synthetic samples, the generator has effectively learned the data distribution (Goodfellow et al., 2014). Part 1 builds this machinery from scratch on controllable 2D data so the mechanics stay visible, and Part 2 applies GANs to three contrasting problems — medical imaging, network security and creative sketching — each stressing a different aspect of generative modelling. PyTorch was used for the 2D and tabular models and TensorFlow/Keras for the two image problems, with all seeds fixed for reproducibility.

---

## Part 1: Building and Understanding GANs from Scratch

Part 1 was implemented directly in PyTorch. The generator is an MLP regularised with Layer Normalisation (Ba, Kiros & Hinton, 2016) and the critic is constrained with spectral normalisation (Miyato et al., 2018), which was chosen because on tiny 2D problems an unconstrained critic wins almost immediately; spectral normalisation is the cheapest principled way to keep the two networks balanced. Both optimise the non-saturating loss with one-sided label smoothing (Salimans et al., 2016) using the Adam optimiser.

### Task 1 — Reproducing the sine-wave GAN

![Figure 1: Real (blue) versus generated (orange) points for the sine-wave GAN.](Set%205%20Oputput%20Images/289ceaed53e8fb5e4f5453742e9a062466e64646.png)
*Figure 1 — Sine-wave GAN: real versus generated 2D points.*

Figure 1 shows the generated points falling along the sine curve, confirming the generator has captured a 1D manifold embedded in 2D. The critic loss settles near 1.37 and the generator near 0.80 and both hold there. A critic near 1.37 (≈ 2·ln2) is exactly the value reached when it can no longer separate the distributions, so the flatness signals a healthy equilibrium rather than a stalled optimiser (Goodfellow et al., 2014). The one-sided label smoothing contributes directly to this: capping the real target at 0.9 stops the critic pushing its logits to extremes, which keeps a usable gradient flowing to the generator and explains why the curves stay so flat rather than oscillating violently.

### Task 2 — Modelling a new 2D distribution (mixture of Gaussians)

A ring of eight Gaussians was chosen over the spiral or noisy curve because a set of disconnected modes is the classic stress test for mode collapse.

![Figure 2: Real versus generated samples for the ring of eight Gaussians.](Set%205%20Oputput%20Images/1c64cbb3e682421f3e9b9b9948a23cb30cafb064.png)
*Figure 2 — Mixture-of-Gaussians GAN: real versus generated samples.*

Figure 2 shows the generator reaching all eight modes rather than collapsing onto one or two, which is the success criterion here. Losses hold at the same levels as the sine case (critic ≈ 1.37, generator ≈ 0.80), so the setup is stable even on a multi-modal target. The clusters are not perfectly balanced — a mild form of mode-imbalance (Salimans et al., 2016) — and the faint scatter between modes is unavoidable, since a continuous generator must pass through the empty space separating disconnected blobs. Reaching all eight modes is the meaningful result — the usual failure here is collapse onto one blob — and the spectral-normalised critic largely prevents it, since a Lipschitz-bounded critic cannot hand the generator the sharp shortcut that triggers collapse.

### Task 3 — Modifying the architecture and comparing samples

The third task holds the target fixed and swaps only the generator: a two-layer LeakyReLU baseline versus a three-layer GELU version, on an identical budget.

![Figure 3: Baseline (left) versus deeper GELU generator (right).](Set%205%20Oputput%20Images/b9fc72ed60c571a8d73c2718a58b0cd0af37590a.png)
*Figure 3 — Architecture comparison: baseline versus modified generator.*

Numerically the two variants are almost identical (both critic ≈ 1.36–1.37, generator ≈ 0.80), so the loss curves alone suggest no difference. The value of Figure 3 is that it exposes what the metrics hide: the deeper GELU generator produces visibly tighter, better-separated clusters with less inter-mode bridging. The wider lesson, recurring throughout, is that loss values and sample quality are not the same thing. It also cautions against tuning GAN architectures on loss alone: judged only on final loss the depth-and-activation change would look pointless, when in fact it measurably improved mode separation.

---

## Part 2: Real-World GAN Applications

### Part 2.1: Optical coherence tomography (OCT) retinal images with MedMNIST

This application trains a deep convolutional GAN (DCGAN) on OCTMNIST (Yang et al., 2023). The DCGAN (Radford, Metz & Chintala, 2015) is the natural choice for images because its strided convolutions learn spatial structure directly. The training split holds 97,477 grayscale scans across four diagnostic classes, rescaled to 32×32 in [−1, 1] to match the generator's tanh output.

![Figure 4: OCTMNIST training-set class distribution.](Set%205%20Oputput%20Images/a85ed5788966f847f3fc368a0982924946dfb642.png)
*Figure 4 — Class distribution of the OCTMNIST training split.*

Figure 4 reveals a clearly imbalanced dataset, which matters later: the most frequent classes are the ones the conditional model reproduces best. The generator upsamples a size-100 latent vector to 32×32 with nearest-neighbour upsampling and 3×3 convolutions — transposed convolutions were avoided because they leave checkerboard artefacts (Odena, Dumoulin & Olah, 2016) — and the critic is a strided-convolution stack with Batch Normalisation, trained for 40 epochs.

![Figure 5: Generator and discriminator loss curves for the OCT DCGAN.](Set%205%20Oputput%20Images/8ed43e7f62aaca52119da778af68a559e593529e.png)
*Figure 5 — OCT DCGAN training losses.*

Figure 5 shows the two losses oscillating within a bounded band across 40 epochs rather than collapsing — the behaviour expected of a stable DCGAN.

![Figure 6: Real (left) versus generated (right) OCT scans.](Set%205%20Oputput%20Images/2df62b0fb6d47f7935adcd310cb8544c0810d5e8.png)
*Figure 6 — OCT DCGAN: real versus generated retinal scans.*

Figure 6 is backed quantitatively by a Fréchet Inception Distance (FID) of **32.74** (Heusel et al., 2017), the lowest of the three image models here. The generated scans capture the essential anatomy — a bright, curved retinal band on a dark background — and vary in the band's shape and position, so the generator is not memorising one template. The main weakness is fine texture: the fakes are slightly too smooth, consistent with a good but non-zero FID. This is the expected signature of a loss that rewards global plausibility over pixel detail; a higher-capacity generator or a perceptual-loss term would be the natural way to recover the missing speckle. In a medical setting this smoothing is not harmless, since the fine speckle can itself carry diagnostic signal.

**Extension — conditional GAN.** For extra credit the DCGAN was extended into a conditional GAN (Mirza & Osindero, 2014) so a class can be requested on demand.

![Figure 7: Conditional GAN output, one diagnostic class per row.](Set%205%20Oputput%20Images/b4690c79b95803c823b2ce8dd108f287271c8c91.png)
*Figure 7 — Conditional GAN: one OCT class per row.*

Figure 7 confirms conditioning works — each row differs, so the label genuinely steers the output, and the most frequent classes (Figure 4) come out cleanest. This model was also the hardest to stabilise. A first attempt that fed the label to the critic as a full 32×32 channel gave it an overwhelming cue: its loss collapsed while the generator's ran away. The fix combined late-fusion conditioning (the label embedding joins only at the critic's final dense layer), instance noise (Arjovsky & Bottou, 2017), a two-time-scale rule with a slower critic (Heusel et al., 2017), and two generator steps per critic step. The losses then trade off healthily — critic drifting from about 1.32 to 0.45 and generator from about 0.93 to 4.18 over 40 epochs. The broader lesson is that conditioning is less about the generator than about denying the critic an easy signal: the label that guides the generator becomes a liability once the critic can read it too directly.

**The "duck" reflection.** The duck test — looks and quacks like a duck, so call it one — captures the risk here. A low-FID GAN is only trained to pass that surface test; the critic never learns what is *medically* correct, so a convincing scan can still contain structure present in no real patient. This is why synthetic scans are defensible for augmentation or teaching but not diagnosis: a low FID says the population statistics align, not that any individual image is real.

### Part 2.2: Cybersecurity – Synthetic Traffic with CICIDS 2017

This task generates synthetic traffic feature vectors from CICIDS 2017 (Sharafaldin, Lashkari & Ghorbani, 2018). Because each flow is numeric rather than pixels, a convolutional model is inappropriate; a fully-connected GAN written in PyTorch was used, with a 56-dimensional noise vector. The eight daily CSVs were merged into **2,830,743** flows, from which **78** numeric features were retained after removing 4,376 infinite and 1,358 missing cells.

![Figure 8: CICIDS 2017 class distribution (log scale).](Set%205%20Oputput%20Images/5cd6e20f0519869b2289b76954766557f1b73c9f.png)
*Figure 8 — CICIDS 2017 class distribution (log scale).*

Figure 8 uses a log scale precisely because BENIGN traffic (2,273,097 flows) dwarfs every attack class, with rare families such as Heartbleed and Infiltration numbering only tens of flows — an imbalance that directly limits what the generator can learn. Following the brief, the GAN trained on the Wednesday file (BENIGN versus DoS), scaled with a RobustScaler and clipped to ±5 to tame outliers.

![Figure 9: PCA of real versus synthetic feature vectors.](Set%205%20Oputput%20Images/61dd21eac59b02bdc817d283bf42779d8db56dbc.png)
*Figure 9 — PCA of real (BENIGN, DoS) versus synthetic flows.*

![Figure 10: t-SNE of real versus synthetic feature vectors.](Set%205%20Oputput%20Images/fd3b1a5f935386cb780032842376028d4967bd0c.png)
*Figure 10 — t-SNE of real versus synthetic flows.*

Figures 9 and 10 compare the distributions with PCA and t-SNE (van der Maaten & Hinton, 2008). In both, the synthetic points sit inside the real cloud rather than off to one side, so the generator finds the right region of feature space; however, the synthetic cloud is slightly tighter, under-representing the spread. The alignment metrics quantify this — a mean per-feature difference of **0.2435** in means and **0.2395** in standard deviations — a moderate fit that captures location and scale but not the heavy tails. The tails matter operationally, because intrusion detectors often key on rare, extreme feature values, so a generator that compresses the spread risks producing traffic that looks normal precisely where the real discriminative signal lives.

**Extension — the full dataset.** Retraining on all eight days and every attack type gives Figure 11, with real points coloured by attack family.

![Figure 11: PCA of the full dataset, real by attack family versus synthetic.](Set%205%20Oputput%20Images/0c0de7cb6546dd7b536be9c17b3860087d5516c6.png)
*Figure 11 — Full-dataset PCA: real by attack family versus synthetic.*

On the full week the gaps widen to **0.2955** (means) and **0.5277** (standard deviations); the std gap roughly doubles. This follows directly from Figure 8: BENIGN and the high-volume attacks dominate the mixture, so a single unconditional generator tracks those modes while rare families stay under-served — an honest limit of unconditional generation on imbalanced data. In terms of generalisation across attack types, the model transfers well to the high-volume families it has seen in quantity (BENIGN, DoS, DDoS and PortScan) but poorly to the long tail, so a single unconditional generator is best understood as a model of *typical* traffic rather than of any specific attack. The duck point sharpens here: a fake flow only has to look statistically plausible to fool the critic, and a row of 78 numbers cannot be eyeballed for wrongness the way an image can, so the moment gaps above (the std gap especially) are the real check.

### Part 2.3: Creative AI – QuickDraw 'birthday cake' Subset

The final application trains a DCGAN on the QuickDraw 'birthday cake' category (Ha & Eck, 2017). To show a second standard construction, this generator uses transposed convolutions (latent size 64) and its critic uses dropout rather than Batch Normalisation.

![Figure 12: Generated 'birthday cake' samples across training epochs.](Set%205%20Oputput%20Images/755d0e764c751f4435c74faf31507c81e7fff03f.png)
*Figure 12 — 'Birthday cake' samples captured at successive epochs.*

Figure 12 tracks the visual outputs across epochs: the early grids are noisy blobs that progressively resolve into recognisable cake outlines with candle-like marks, showing the generator is genuinely learning rather than memorising and that training is stable to the final epoch.

![Figure 13: Real (left) versus generated (right) 'birthday cake' sketches.](Set%205%20Oputput%20Images/570ceebda29a55d0bfde53ec3f631ac8dc54bbaa.png)
*Figure 13 — QuickDraw 'birthday cake': real versus generated sketches.*

Figure 13 shows most generated sketches are recognisable cakes with strokes that hold together, reflected in a cake FID of **24.59** — the best sketch score. Occasional stroke artefacts appear, but the overall form is preserved.

**Extension — other categories and complexity.** The same model was trained on two more categories and their FIDs compared against an ink-density proxy for complexity.

![Figure 14: FID versus ink-density across QuickDraw categories.](Set%205%20Oputput%20Images/5428b43dfbb7cd5c352e9c9862feb41ef1633182.png)
*Figure 14 — QuickDraw extension: FID and ink-density by category.*

Figure 14 gives the most interesting result. The FIDs are cake **24.59**, cat **31.83** and house **42.16**, while ink density is 0.198, 0.1985 and 0.173 respectively. House is therefore the *sparsest* category yet scores the *worst* FID — the opposite of the initial "busier is harder" hypothesis. The interpretation is that ink coverage is a poor difficulty predictor: a house is a few long, straight lines meeting at clean right angles, and reproducing that geometry precisely is harder than the loose, forgiving strokes of a cat or cake. Structural regularity matters more than the amount of ink. This reframes the hypothesis as a point about inductive bias: convolutional generators reproduce local texture cheaply but struggle with long-range geometric constraints, so a drawing that is simple for a human is not necessarily easy for the model.

---

## Conclusion and Future Work

Across all four problems the models trained stably and matched the real data on the metrics used: the Part 1 GANs reach every mode, the OCT DCGAN reaches an FID of 32.74 with a working conditional extension, the CICIDS generator places synthetic flows inside the real distribution with moderate moment gaps, and the QuickDraw DCGAN reaches an FID of 24.59 on cakes. The analysis also surfaced clear limits: saturated losses can hide real quality differences (Task 3); conditional GANs are highly sensitive to how the label is injected (2.1); unconditional generators under-represent rare classes on imbalanced data (2.2); and drawing difficulty is driven by geometric regularity, not ink volume (2.3). Future work would address these with a Wasserstein GAN with gradient penalty to tighten the CICIDS gaps, class-conditional oversampling to serve rare attack families, and a downstream train-on-synthetic/test-on-real test to measure the *utility* of synthetic data rather than only its distributional match. For the medical task a domain-specific perceptual metric would be more trustworthy than the ImageNet-based FID used here, where the "duck" problem is most consequential. A complementary next step would be a small classifier-based sanity check on the conditional samples, verifying that a network trained on real OCT scans assigns generated images to the intended class before any downstream use.

---

## References

Arjovsky, M. and Bottou, L. (2017) 'Towards principled methods for training generative adversarial networks', *International Conference on Learning Representations (ICLR)*.

Ba, J.L., Kiros, J.R. and Hinton, G.E. (2016) 'Layer normalization', *arXiv preprint* arXiv:1607.06450.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y. (2014) 'Generative adversarial nets', *Advances in Neural Information Processing Systems (NeurIPS)*, 27, pp. 2672–2680.


Ha, D. and Eck, D. (2017) 'A neural representation of sketch drawings', *arXiv preprint* arXiv:1704.03477.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B. and Hochreiter, S. (2017) 'GANs trained by a two time-scale update rule converge to a local Nash equilibrium', *Advances in Neural Information Processing Systems (NeurIPS)*, 30, pp. 6626–6637.



Mirza, M. and Osindero, S. (2014) 'Conditional generative adversarial nets', *arXiv preprint* arXiv:1411.1784.

Miyato, T., Kataoka, T., Koyama, M. and Yoshida, Y. (2018) 'Spectral normalization for generative adversarial networks', *International Conference on Learning Representations (ICLR)*.

Odena, A., Dumoulin, V. and Olah, C. (2016) 'Deconvolution and checkerboard artifacts', *Distill*, 1(10).

Radford, A., Metz, L. and Chintala, S. (2015) 'Unsupervised representation learning with deep convolutional generative adversarial networks', *arXiv preprint* arXiv:1511.06434.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A. and Chen, X. (2016) 'Improved techniques for training GANs', *Advances in Neural Information Processing Systems (NeurIPS)*, 29, pp. 2234–2242.

Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A. (2018) 'Toward generating a new intrusion detection dataset and intrusion traffic characterization', *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, pp. 108–116.


van der Maaten, L. and Hinton, G. (2008) 'Visualizing data using t-SNE', *Journal of Machine Learning Research*, 9, pp. 2579–2605.

Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H. and Ni, B. (2023) 'MedMNIST v2 - a large-scale lightweight benchmark for 2D and 3D biomedical image classification', *Scientific Data*, 10(1), p. 41.
