# 7COM1079 Coursework 2 — Generative Adversarial Networks

## Introduction

This report accompanies my notebook implementation of generative adversarial networks (GANs). A GAN pits a generator against a discriminator in a minimax game, and at equilibrium the generator's samples become statistically indistinguishable from real data (Goodfellow et al., 2014). Part 1 builds GANs from scratch on synthetic 2D data to make the training dynamics visible; Part 2 applies convolutional and tabular GANs to three real datasets — medical imaging, network-security traffic and hand-drawn sketches. Throughout, the focus is on *why* an architecture suits a dataset and on interpreting the training behaviour and metrics rather than describing them. Part 1 and the image problems use PyTorch, the tabular model TensorFlow/Keras; fixed seeds (`TORCH_SEED = 3030`, `KERAS_SEED = 4040`) make every figure reproducible.

## Part 1: Building and Understanding GANs from Scratch

Part 1 is written from scratch in PyTorch: the generator and discriminator are small multilayer perceptrons trained with the non-saturating binary cross-entropy loss (Goodfellow et al., 2014). One configurable `Toy2DGAN` trainer drives all three tasks. Decaying instance noise is added to the discriminator inputs — a standard stabiliser that stops it separating the point clouds too early.

### Task 1: Reproduce the sine-wave GAN

The tutorial task samples points along `y = sin(x)` and trains for 6000 steps. It is the cleanest correctness check because the non-saturating loss has a known equilibrium: the discriminator should sit near `2ln2 ≈ 1.386` and the generator near `ln2 ≈ 0.693`.

![Figure 1](Set%204%20Result%20Images/117d337be2a2754d900d881e21ea9198e40804e4.png)

**Figure 1.** Task 1 — real (blue) versus generated (orange) samples for the sine-wave target after 6000 steps.

Figure 1 shows the generated points lying almost exactly on the sine curve across the full `[-π, π]` domain, with no gaps or clumping. The loss trace confirms this: the discriminator settles around 1.39 and the generator around 0.69, essentially on the theoretical equilibrium. That agreement is the strongest single sign in Part 1 that the from-scratch implementation has genuinely converged rather than one network running away.

### Task 2: A new 2D distribution (2D spiral)

For the new distribution I chose a noisy 2D Archimedean spiral, trained for 8000 steps. It is harder than the sine curve because its arms curl back on themselves, so the generator must learn a self-overlapping structure rather than a single-valued function.

![Figure 2](Set%204%20Result%20Images/3070d1ab0ceb8e09ec41c8dda0a2c606f7bbd1d6.png)

**Figure 2.** Task 2 — real versus generated samples for the 2D spiral after 8000 steps.

Figure 2 shows the generator tracing the spiral's arms rather than collapsing onto a single loop, which would signal mode collapse. Losses stay near equilibrium (D ≈ 1.45, G ≈ 0.66), with a brief excursion to G ≈ 1.1 around step 6000 before settling — a self-correcting wobble, not divergence. The main weakness is that the outermost arm is slightly sparser than the real data, expected since that region carries the least probability mass and gives the weakest learning signal.

### Task 3: Modify the architecture and compare

Holding the target and budget fixed, only the network changed: the baseline generator is depth 3 with GELU, the modified one is deeper (depth 5) with LeakyReLU (0.3 slope). Isolating these factors keeps the comparison interpretable.

![Figure 3](Set%204%20Result%20Images/1b89c3e63e5e4a641ef13e5837833221de1a389f.png)

**Figure 3.** Task 3 — baseline (depth 3, GELU) versus modified (depth 5, LeakyReLU-0.3) generators against the real spiral.

Figure 3 shows the two point clouds are very similar in coverage and shape, and both hold losses near equilibrium throughout (D ≈ 1.37–1.46, G ≈ 0.66–0.76, ending D ≈ 1.46 / G ≈ 0.69 for the modified network). The deeper LeakyReLU model does not destabilise training; its loss trace is marginally smoother. This shows added depth and a leaky activation are safe here, motivating the deeper convolutional variants used in Part 2.

## Part 2: Real-World GAN Applications

### Part 2.1: Optical coherence tomography (OCT) retinal images with MedMNIST

The OCTMNIST subset provides 97,477 grayscale 28×28 scans across four diagnostic classes — choroidal neovascularization, diabetic macular edema, drusen and normal (Yang et al., 2023). Because the data are images with strong local structure, a deep convolutional GAN (DCGAN) is the natural choice: convolutions share weights and encode the translation-invariance fully-connected layers cannot (Radford, Metz and Chintala, 2016). The generator maps a latent vector (dimension 100) to a 32×32 scan via a linear projection, nearest-neighbour upsampling and 3×3 convolutions; the discriminator mirrors it with strided convolutions. I used resize-convolution (upsample-then-convolve) rather than transposed convolutions because the latter cause checkerboard artefacts on small grayscale images (Odena, Dumoulin and Olah, 2016); in practice this gave a lower FID.

![Figure 4](Set%204%20Result%20Images/56fc192fbd336d99f0349ee8477da692859693d4.png)

**Figure 4.** OCTMNIST class distribution across the four diagnostic labels.

Figure 4 shows the data are imbalanced, some classes far better represented than others. This matters for the unconditional DCGAN, which spends most capacity on the dominant modes, and foreshadows the conditional model, where better-represented classes come out most convincingly.

![Figure 5](Set%204%20Result%20Images/ba8e019a9fb23d9f9e6ecd43ee33af9b18e03d76.png)

**Figure 5.** OCT DCGAN generator and discriminator loss curves over 45 epochs.

Figure 5 shows the characteristic DCGAN pattern rather than a textbook equilibrium: the discriminator loss stays low (ending near 0.393) while the generator loss is higher and noisier (ending near 4.090). With one-sided label smoothing the discriminator has an easier task on 28×28 scans; the key point is that neither loss diverges and the generator keeps receiving usable gradients across all 45 epochs.

![Figure 6](Set%204%20Result%20Images/9e98b738bd8e579a1f7988a79e793c2e959c4ab9.png)

**Figure 6.** Real (left) versus generated (right) OCT scans from the trained DCGAN.

Figure 6 shows the generated scans reproduce the defining structure of a real OCT image — a bright, curved retinal band over a darker background — with real variation in band position and curvature rather than one repeated pattern. The Fréchet Inception Distance (Heusel et al., 2017), using an ImageNet-pretrained Inception-v3 extractor (Szegedy et al., 2016), is **31.94**, the strongest image score in this project. The main remaining weakness is fine sub-layer texture rather than gross anatomy.

**Reflection and the "duck test".** A discriminator only ever learns "real or fake", never "anatomically correct" — the GAN version of the duck test, where a sample that looks and "quacks" convincingly passes. Nothing stops the generator inventing a bright region or layer boundary no real scan had, provided it is statistically plausible. A good FID only tells us the *aggregate* statistics of a batch are close to real; it says nothing about whether a single image is trustworthy. In a retinal scan a hallucinated bright patch could mimic pathology that is not there, so I would restrict synthetic scans to augmentation or teaching and re-validate any classifier trained on them against real-only data.

**Extension — conditional GAN.** The extension conditions the generator on the label by concatenating a learned class embedding with the latent vector, while the discriminator receives the label as an extra channel (Mirza and Osindero, 2014).

![Figure 7](Set%204%20Result%20Images/1499c8335b3ec723f1bd6e9764632d3d642a40ff.png)

**Figure 7.** Conditional GAN output, one diagnostic class per row, generated on demand.

Figure 7 shows the rows differ visibly, confirming the generator responds to the label rather than ignoring it. As expected from the imbalance in Figure 4, the better-represented classes are most convincing — the standard outcome for a conditional GAN on imbalanced data — and the model now generates a chosen class on demand.

### Part 2.2: Cybersecurity — Synthetic Traffic with CICIDS 2017

CICIDS 2017 records each network flow as a numeric feature vector, not an image (Sharafaldin, Lashkari and Ghorbani, 2018). Combining the eight daily files gives 2,830,743 flows and 78 numeric features. With no spatial structure a convolutional model makes no sense; instead I use a fully-connected (dense) GAN as a `keras.Model` subclass with a custom `train_step`, latent dimension 64. The generator uses a linear output (the scaled features are unbounded) and the critic returns a single logit.

![Figure 8](Set%204%20Result%20Images/96c5fc13ac27f1b4c7c7ac54c2baee62d4e868b3.png)

**Figure 8.** CICIDS 2017 overall class distribution (log scale) and flows per day.

Figure 8 shows extreme imbalance: BENIGN dominates with 2,273,097 flows, then DoS Hulk (231,073) and DDoS (128,027), with several families in the thousands. This imbalance is the central challenge and directly explains the alignment results below. Features were first mapped to a standard normal with a rank-based quantile transformer, because CICIDS contains extreme outliers that otherwise destabilise training. Following the brief, the core model trains on the **Wednesday** file only (BENIGN versus DoS; 691,406 rows, sampled to 20,000). The brief names this "DDoS", but Wednesday's attacks are single-source DoS; true DDoS is in the Friday file and enters only via the extension.

![Figure 9](Set%204%20Result%20Images/41f018da649ab0b39d373bf47ca65bf24f9558b5.png)

**Figure 9.** PCA of real BENIGN, real DoS and synthetic feature vectors (Wednesday model).

Figure 9 projects the 78-dimensional vectors to two components. The synthetic points sit within the spread of the real ones rather than forming a separate island — the first sign the generator learned the broad distribution rather than collapsing.

![Figure 10](Set%204%20Result%20Images/9f2c0ecdefc01a04e2e11c73dbddea209bf7035d.png)

**Figure 10.** t-SNE of real versus synthetic feature vectors (Wednesday model).

Figure 10 gives the non-linear view (van der Maaten and Hinton, 2008), better at exposing local structure. Synthetic points overlap the real manifold across most regions but thin out in some pockets, matching the alignment gaps: the mean absolute per-feature difference is **0.4730** for means and **0.5293** for standard deviations. This is a moderate match — the generator captures the general shape but under-represents the spread of heavy-tailed or near-binary features (byte/packet counts, flag columns), the known hard case for tabular GANs.

**Extension — full multi-day dataset.** Retraining on a 30,000-row sample spanning all days tests whether one unconditional generator can cover many attack families.

![Figure 11](Set%204%20Result%20Images/6185fa52602d60257146e00448b17091fe09c63a.png)

**Figure 11.** Full-dataset PCA — real points coloured by attack family versus synthetic points.

Figure 11 shows the real data now spans many families and the synthetic points concentrate where the mass is. The gaps worsen slightly (mean **0.5454**, std **0.6094**) — the expected result of spreading one generator across more families: BENIGN and high-volume attacks dominate, while rare families such as Heartbleed and Infiltration are under-represented. The duck test bites harder here than for images, since there is no "just look at it" check for 78 numbers: a fake "attack" matching no real signature could teach a detector the wrong pattern, and a fake "benign" flow drifting toward attack-like statistics could inflate false positives — which is why the PCA/t-SNE overlap and per-feature gaps matter more here than a visual check.

### Part 2.3: Creative AI — QuickDraw 'birthday cake' Subset

The QuickDraw birthday-cake category provides 144,982 grayscale 28×28 doodles (Ha and Eck, 2017). These are sparse line drawings but still images with local structure, so the same resize-convolution DCGAN is appropriate; bitmaps are served through a custom `Dataset`. Training tracks losses, snapshots samples across epochs, and measures quality with FID.

![Figure 12](Set%204%20Result%20Images/81dcbf6f6c1199ca640664ec997fc53223a341d9.png)

**Figure 12.** Real (left) versus generated (right) 'birthday cake' sketches.

Figure 12 shows the generated cakes are recognisable — a cake-body silhouette with candle-like marks appears consistently across the batch — and training stayed stable across 45 epochs (final D ≈ 0.416 / G ≈ 4.214) without diverging. The FID is **41.01**, a solid score for sparse sketches.

**Extension — other categories and sketch complexity.** The pipeline was repeated on *cat* and *house*, with FID placed against an ink-density complexity proxy. The cat model initially collapsed under a one-to-one schedule, so — demonstrating an alternative training strategy rather than a new architecture — cat was trained with three generator updates per discriminator update, a lower discriminator learning rate (a two-time-scale update rule; Heusel et al., 2017) and 60 epochs, cutting its FID from a collapsed ~98 to 67.36.

![Figure 13](Set%204%20Result%20Images/3ef116d51773d161b1fa3ed9ff4919f03b09210f.png)

**Figure 13.** FID by category (left) versus ink-density complexity proxy (right).

Figure 13 reveals a clean relationship: the ink-density order house (0.1721) < cake (0.1967) < cat (0.1986) matches the FID order house (**28.76**) < cake (**41.01**) < cat (**67.36**) exactly. The sparsest, most regular category is easiest and the busiest is hardest. The deeper driver is intra-class variety: cat sketches differ far more from one another (pose, ears, whiskers, sitting versus standing) than houses do, and that variety is the harder thing for a fixed-capacity DCGAN to capture — which is why cat stays weakest even after the stronger schedule.

## Conclusion

Across all five experiments the implementations behaved as theory predicts and the metrics tell a consistent story. Part 1 confirmed correctness by landing the sine-wave GAN on its analytic equilibrium and showed a deeper, leaky-activated generator is a safe modification. Part 2 matched architecture to data: a resize-convolution DCGAN for OCT scans (FID 31.94) and QuickDraw sketches (cake 41.01, house 28.76, cat 67.36), and a dense feature-vector GAN for CICIDS traffic (gaps 0.4730/0.5293 on Wednesday). Recurring themes: class imbalance limits how faithfully minority modes are reproduced; FID and alignment gaps measure aggregate fidelity, not individual trustworthiness (the duck test); and intra-class variety predicts difficulty better than surface complexity. For future work I would add a conditional or Wasserstein objective to the tabular model for rare attack families, increase generator capacity for high-variety classes such as cat, and pair FID with a precision–recall metric.

## References

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y. (2014) 'Generative adversarial nets', *Advances in Neural Information Processing Systems*, 27, pp. 2672–2680.

Ha, D. and Eck, D. (2017) *A neural representation of sketch drawings*. arXiv:1704.03477.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B. and Hochreiter, S. (2017) 'GANs trained by a two time-scale update rule converge to a local Nash equilibrium', *Advances in Neural Information Processing Systems*, 30, pp. 6626–6637.

Mirza, M. and Osindero, S. (2014) *Conditional generative adversarial nets*. arXiv:1411.1784.

Odena, A., Dumoulin, V. and Olah, C. (2016) 'Deconvolution and checkerboard artifacts', *Distill*, 1(10).

Radford, A., Metz, L. and Chintala, S. (2016) *Unsupervised representation learning with deep convolutional generative adversarial networks*. arXiv:1511.06434.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A. and Chen, X. (2016) 'Improved techniques for training GANs', *Advances in Neural Information Processing Systems*, 29, pp. 2234–2242.

Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A. (2018) 'Toward generating a new intrusion detection dataset and intrusion traffic characterization', *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, pp. 108–116.

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. and Wojna, Z. (2016) 'Rethinking the Inception architecture for computer vision', *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2818–2826.

van der Maaten, L. and Hinton, G. (2008) 'Visualizing data using t-SNE', *Journal of Machine Learning Research*, 9, pp. 2579–2605.

Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H. and Ni, B. (2023) 'MedMNIST v2 — a large-scale lightweight benchmark for 2D and 3D biomedical image classification', *Scientific Data*, 10(1), 41.
