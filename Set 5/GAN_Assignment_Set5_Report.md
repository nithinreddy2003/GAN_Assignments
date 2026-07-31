# 7COM1079 Coursework 2 — Generative Adversarial Networks

## Introduction

This report accompanies the submitted notebook and documents the design decisions, training behaviour and results of a set of generative adversarial networks (GANs). A GAN pits a generator against a discriminator in a minimax game, and at the point where the discriminator can no longer separate real from synthetic samples the generator has, in effect, learned the data distribution (Goodfellow et al., 2014). Part 1 builds this machinery from scratch on controllable 2D data so the mechanics stay visible; Part 2 then applies GANs to three contrasting real-world problems — medical imaging, network security and creative sketching — each of which stresses a different aspect of generative modelling.

A single framework was not imposed everywhere. The from-scratch 2D models and the tabular traffic model are written in PyTorch, while the two image problems use TensorFlow/Keras, where the convolutional and conditional components were quicker to assemble. All random seeds are fixed before execution so every figure and metric below is reproducible.

---

## Part 1: Building and Understanding GANs from Scratch

Part 1 was implemented directly in PyTorch without a GAN library, wrapping the generator and critic in one reusable class. The generator is a multilayer perceptron regularised with Layer Normalisation (Ba, Kiros & Hinton, 2016), and the critic is constrained with spectral normalisation (Miyato et al., 2018) so it cannot overwhelm the generator early in training. Both optimise the non-saturating binary cross-entropy loss with one-sided label smoothing (Salimans et al., 2016) using Adam (Kingma & Ba, 2015). This configuration was chosen deliberately: on tiny 2D problems an unconstrained critic tends to win immediately, and spectral normalisation is the cheapest principled way to keep the two networks balanced.

### Task 1 — Reproducing the sine-wave GAN

The first task reproduces the tutorial sine-wave GAN, sampling points from *y = sin(x)* and asking the generator to match them.

![Figure 1: Real (blue) versus generated (orange) points for the sine-wave GAN.](results/part1_task1_sine.png)
*Figure 1 — Sine-wave GAN: real versus generated 2D points.*

Figure 1 shows the generated points falling along the sine curve rather than scattering, confirming the generator has captured a one-dimensional manifold embedded in 2D. The training log supports this: the critic loss settles near 1.37 and the generator near 0.80 and both stay there without diverging. A critic loss close to 1.37 is meaningful — it is approximately 2·ln2, the value reached when the critic can no longer distinguish the two distributions — so the flatness is a sign of a healthy equilibrium rather than a stalled optimiser (Goodfellow et al., 2014).

### Task 2 — Modelling a new 2D distribution (mixture of Gaussians)

For the second task a mixture of eight Gaussians arranged in a ring was chosen over the spiral or noisy curve, because a set of disconnected modes is the classic stress test for mode collapse.

![Figure 2: Real versus generated samples for the ring of eight Gaussians.](results/part1_task2.png)
*Figure 2 — Mixture-of-Gaussians GAN: real versus generated samples.*

Figure 2 demonstrates that the generator reaches all eight modes rather than collapsing onto one or two, which is the main success criterion for this target. The losses hold at essentially the same levels as the sine case (critic ≈ 1.37, generator ≈ 0.80), showing the setup is stable even on a harder, multi-modal target. Closer inspection shows the clusters are not perfectly balanced — some modes are denser than others — a mild form of the mode-imbalance GANs are known to exhibit on multi-modal data (Salimans et al., 2016). The thin scatter of points between modes is not a defect: a continuous generator mapping a connected latent space onto disconnected blobs must pass through the empty space between them, so some bridging points are unavoidable.

### Task 3 — Modifying the architecture and comparing samples

The third task isolates the effect of architecture by holding the Gaussian-ring target fixed and swapping only the generator: a two-layer LeakyReLU baseline against a three-layer GELU version, trained on an identical budget.

![Figure 3: Baseline (left) versus deeper GELU generator (right) on the Gaussian ring.](results/part1_task3_comparison.png)
*Figure 3 — Architecture comparison: baseline versus modified generator.*

Figure 3 makes the comparison directly. Numerically the two variants are almost indistinguishable — both finish with a critic near 1.36–1.37 and a generator near 0.80 — so the loss curves alone would suggest no difference. The value of the visual comparison is precisely that it exposes what the metrics hide: the deeper GELU generator produces visibly tighter, better-separated clusters with less inter-mode bridging. The interpretation is that added depth and a smoother activation improve how cleanly the generator carves the latent space into eight regions, even though the adversarial loss has already saturated. This illustrates a wider point that recurs throughout the report — loss values and sample quality are not the same thing, and both must be inspected.

---

## Part 2: Real-World GAN Applications

### Part 2.1: Optical coherence tomography (OCT) retinal images with MedMNIST

The first application trains a deep convolutional GAN (DCGAN) on the OCTMNIST subset of MedMNIST (Yang et al., 2023). The DCGAN architecture (Radford, Metz & Chintala, 2015) is the natural choice for images because its strided convolutions let the networks learn spatial structure directly. The training split contains 97,477 grayscale scans across four diagnostic classes — choroidal neovascularisation, diabetic macular edema, drusen and normal — rescaled to 32×32 in the range [−1, 1] to match the generator's tanh output.

![Figure 4: OCTMNIST training-set class distribution.](results/oct_class_distribution.png)
*Figure 4 — Class distribution of the OCTMNIST training split.*

Figure 4 shows a clearly imbalanced dataset, which matters later: the classes with the most examples are the ones the conditional model reproduces most convincingly. The generator upsamples a latent vector of size 100 to 32×32 through nearest-neighbour upsampling followed by 3×3 convolutions; transposed convolutions were deliberately avoided because they tend to leave checkerboard artefacts (Odena, Dumoulin & Olah, 2016). The critic is a strided-convolution stack with Batch Normalisation, and training runs for 40 epochs through a shared trainer class.

![Figure 5: Generator and discriminator loss curves for the OCT DCGAN.](results/oct_losses.png)
*Figure 5 — OCT DCGAN training losses.*

Figure 5 shows the adversarial game trading off rather than collapsing, with the two losses oscillating within a bounded band across the 40 epochs — the behaviour expected of a stable DCGAN.

![Figure 6: Real (left) versus generated (right) OCT scans.](results/oct_real_vs_fake.png)
*Figure 6 — OCT DCGAN: real versus generated retinal scans.*

Figure 6 is the qualitative comparison, and it is backed quantitatively by a Fréchet Inception Distance (FID) of **32.74** (Heusel et al., 2017), the lowest of the three image models in this notebook. The generated scans capture the essential anatomy — a bright, curved retinal band over a darker background — and vary in the band's shape and position, so the generator is not memorising a single template. The main weakness is fine texture: the fakes are slightly too smooth and lose some of the speckle of genuine OCT, which is consistent with the FID being good but not near-zero.

**Extension — conditional GAN.** For extra credit the DCGAN was extended into a conditional GAN (Mirza & Osindero, 2014) so that a class can be requested on demand.

![Figure 7: Conditional GAN output, one diagnostic class per row.](results/oct_cgan_per_class.png)
*Figure 7 — Conditional GAN: one OCT class per row.*

Figure 7 confirms the conditioning works — each row differs, so the label genuinely steers the output, and the more frequent classes (Figure 4) come out cleanest. This model was also the hardest to stabilise and is worth analysing. A first attempt that fed the label to the critic as a full 32×32 channel gave the critic an overwhelming cue: its loss collapsed while the generator's ran away. The fix combined late-fusion conditioning (the label embedding is only concatenated at the critic's final dense layer), instance noise on the images (Arjovsky & Bottou, 2017), a two-time-scale update rule with a slower critic (Heusel et al., 2017), and two generator steps per critic step. With these changes the losses trade off healthily — the critic drifts from about 1.32 to 0.45 and the generator from about 0.93 to 4.18 over 40 epochs — and the class rows stay distinct.

**Reflection and the "duck" discussion.** The task asks for a reflection on synthetic-data quality, including the "duck" point. The name comes from the duck test: if it looks and quacks like a duck, we call it one. A GAN, even a low-FID one, is only ever trained to pass that surface test. The critic never learns what is *medically* correct, so a sharp, convincing scan can still contain structure present in no real patient — an invented bright region that reads as pathology, or a genuine feature quietly smoothed away. This is why synthetic scans like these are defensible for augmentation or teaching but must not be trusted for diagnosis: a low FID says the population statistics align, not that any individual image is real.

### Part 2.2: Cybersecurity – Synthetic Traffic with CICIDS 2017

The second application generates synthetic network-traffic feature vectors from CICIDS 2017 (Sharafaldin, Lashkari & Ghorbani, 2018). Because each flow is described by numeric features rather than pixels, a convolutional model is inappropriate; a fully-connected GAN written in PyTorch was used instead, with a 56-dimensional noise vector. The eight daily CSV files were merged into a single table of **2,830,743** flows, from which **78** numeric features were retained after removing 4,376 infinite and 1,358 missing cells.

![Figure 8: CICIDS 2017 class distribution (log scale).](results/cicids_class_distribution.png)
*Figure 8 — CICIDS 2017 class distribution across all days (log scale).*

Figure 8 is plotted on a log scale precisely because the data are so skewed: BENIGN traffic (2,273,097 flows) dwarfs every attack class, with DoS Hulk (231,073) and DDoS (128,027) the largest attack families and rare classes such as Heartbleed and Infiltration numbering only tens of flows. This imbalance directly shapes what the generator can learn. Following the brief, the GAN was trained on the Wednesday file only (BENIGN versus DoS), scaled with a RobustScaler and clipped to ±5 to tame outliers, using a 20,000-flow training sample.

![Figure 9: PCA projection of real versus synthetic feature vectors.](results/cicids_pca.png)
*Figure 9 — PCA of real (BENIGN, DoS) versus synthetic flows.*

![Figure 10: t-SNE projection of real versus synthetic feature vectors.](results/cicids_tsne.png)
*Figure 10 — t-SNE of real versus synthetic flows.*

Figures 9 and 10 compare the real and synthetic distributions using PCA (Jolliffe & Cadima, 2016) and t-SNE (van der Maaten & Hinton, 2008). In both projections the synthetic points sit inside the main body of real data rather than off to one side, so the generator has found roughly the right region of feature space; however, the synthetic cloud is slightly more tightly packed, indicating it under-represents the spread. The alignment metrics quantify this: the mean absolute per-feature difference in means is **0.2435** and in standard deviations **0.2395** on the Wednesday data — a moderate fit that captures location and scale but not the heavy tails of the more skewed features.

**Extension — the full dataset.** For extra credit the same model was retrained on all eight days and every attack type.

![Figure 11: PCA of the full dataset, real coloured by attack family versus synthetic.](results/cicids_full_pca.png)
*Figure 11 — Full-dataset PCA: real points by attack family versus synthetic.*

Figure 11 colours the real points by attack family so it is clear where the synthetic cloud settles. On the full week the gaps widen to **0.2955** (means) and **0.5277** (standard deviations); the standard-deviation gap roughly doubles relative to Wednesday. The interpretation follows directly from Figure 8: BENIGN and the high-volume attacks dominate the mixture, so a single unconditional generator tracks those dominant modes while the rare families remain under-served simply because there are so few examples of them. This is an honest limitation of unconditional generation on heavily imbalanced data.

The "duck" point transfers to tabular data with a sharper edge. A fake flow only has to look statistically like real traffic to satisfy the critic; it need not correspond to any genuine attack or benign session. Feeding such plausible-but-meaningless flows into intrusion-detection training could teach a detector to expect attacks that never occur, or to distrust normal traffic — and unlike an image, a row of 78 numbers cannot be eyeballed for wrongness, which is exactly why the moment gaps above (the standard-deviation gap in particular) are the real check.

### Part 2.3: Creative AI – QuickDraw 'birthday cake' Subset

The final application trains a DCGAN on the Google QuickDraw 'birthday cake' category (Ha & Eck, 2017), reusing the shared image trainer. To demonstrate a second standard construction, this generator was built with transposed convolutions (latent size 64) and its critic uses dropout rather than Batch Normalisation.

![Figure 12: Real (left) versus generated (right) 'birthday cake' sketches.](results/qd_cake_real_vs_fake.png)
*Figure 12 — QuickDraw 'birthday cake': real versus generated sketches.*

Figure 12 shows that most generated sketches are recognisable cakes — an oval body with candle-like marks on top — with strokes that hold together rather than fragmenting. This is reflected in a cake FID of **24.59**, the best score of the sketch categories. Occasional stroke artefacts appear, but the overall form is preserved across the grid, and the loss curves stayed level to the final epoch with neither network running away.

**Extension — other categories and complexity.** The same model was trained on two further categories, and their FIDs compared against an ink-density proxy for drawing complexity.

![Figure 13: FID versus ink-density across QuickDraw categories.](results/qd_extension_summary.png)
*Figure 13 — QuickDraw extension: FID and ink-density by category.*

Figure 13 gives the most interesting result in the notebook. The FIDs are cake **24.59**, cat **31.83** and house **42.16**, while ink density is 0.198, 0.1985 and 0.173 respectively. House is therefore the *sparsest* category yet scores the *worst* FID — the opposite of the initial hypothesis that busier drawings would be harder. The interpretation is that raw ink coverage is a poor predictor of difficulty: a house is a few long, straight lines meeting at clean right angles, and reproducing that geometry precisely is harder for the generator than the loose, forgiving strokes of a cat or cake. In short, structural regularity matters more than the amount of ink, which is a genuinely useful insight into where this class of model struggles.

---

## Conclusion and Future Work

Across all four problems the models trained stably and produced samples that match the real data on the metrics used: the Part 1 GANs reach every mode of both targets, the OCT DCGAN achieves an FID of 32.74 with a working conditional extension, the CICIDS generator places synthetic flows inside the real distribution with moderate moment gaps, and the QuickDraw DCGAN reaches an FID of 24.59 on cakes. Equally important are the limitations surfaced by the analysis: saturated losses can hide real quality differences (Part 1, Task 3); conditional GANs are highly sensitive to how the label is injected (Part 2.1); unconditional generators under-represent rare classes on imbalanced data (Part 2.2); and drawing difficulty is driven by geometric regularity rather than ink volume (Part 2.3).

Several alternative approaches would address these limitations in future work. A Wasserstein GAN with gradient penalty (Gulrajani et al., 2017) would likely tighten the CICIDS moment gaps and reduce the reliance on careful loss balancing. A class-conditional or minority-oversampling scheme would give the rare CICIDS attack families a fairer chance, and a downstream train-on-synthetic/test-on-real evaluation would measure the *utility* of the synthetic traffic rather than only its distributional match. For the imaging tasks, a domain-specific perceptual metric would be more trustworthy than an ImageNet-based FID (Szegedy et al., 2016), particularly for medical scans where the "duck" problem is most consequential.

---

## References

Arjovsky, M. and Bottou, L. (2017) 'Towards principled methods for training generative adversarial networks', *International Conference on Learning Representations (ICLR)*.

Ba, J.L., Kiros, J.R. and Hinton, G.E. (2016) 'Layer normalization', *arXiv preprint* arXiv:1607.06450.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y. (2014) 'Generative adversarial nets', *Advances in Neural Information Processing Systems (NeurIPS)*, 27, pp. 2672–2680.

Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V. and Courville, A. (2017) 'Improved training of Wasserstein GANs', *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

Ha, D. and Eck, D. (2017) 'A neural representation of sketch drawings', *arXiv preprint* arXiv:1704.03477.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B. and Hochreiter, S. (2017) 'GANs trained by a two time-scale update rule converge to a local Nash equilibrium', *Advances in Neural Information Processing Systems (NeurIPS)*, 30, pp. 6626–6637.

Jolliffe, I.T. and Cadima, J. (2016) 'Principal component analysis: a review and recent developments', *Philosophical Transactions of the Royal Society A*, 374(2065).

Kingma, D.P. and Ba, J. (2015) 'Adam: a method for stochastic optimization', *International Conference on Learning Representations (ICLR)*.

Mirza, M. and Osindero, S. (2014) 'Conditional generative adversarial nets', *arXiv preprint* arXiv:1411.1784.

Miyato, T., Kataoka, T., Koyama, M. and Yoshida, Y. (2018) 'Spectral normalization for generative adversarial networks', *International Conference on Learning Representations (ICLR)*.

Odena, A., Dumoulin, V. and Olah, C. (2016) 'Deconvolution and checkerboard artifacts', *Distill*, 1(10).

Radford, A., Metz, L. and Chintala, S. (2015) 'Unsupervised representation learning with deep convolutional generative adversarial networks', *arXiv preprint* arXiv:1511.06434.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A. and Chen, X. (2016) 'Improved techniques for training GANs', *Advances in Neural Information Processing Systems (NeurIPS)*, 29, pp. 2234–2242.

Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A. (2018) 'Toward generating a new intrusion detection dataset and intrusion traffic characterization', *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, pp. 108–116.

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. and Wojna, Z. (2016) 'Rethinking the Inception architecture for computer vision', *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2818–2826.

van der Maaten, L. and Hinton, G. (2008) 'Visualizing data using t-SNE', *Journal of Machine Learning Research*, 9, pp. 2579–2605.

Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H. and Ni, B. (2023) 'MedMNIST v2 - a large-scale lightweight benchmark for 2D and 3D biomedical image classification', *Scientific Data*, 10(1), p. 41.
