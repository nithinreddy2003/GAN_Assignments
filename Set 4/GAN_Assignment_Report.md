# 7COM1079 Coursework 2 — Generative Adversarial Networks

## Introduction

This report accompanies my notebook implementation of generative adversarial networks (GANs) for the Unit 5 coursework. A GAN pits a generator against a discriminator in a minimax game, and at equilibrium the generator's samples become statistically indistinguishable from real data (Goodfellow et al., 2014). Part 1 builds GANs from scratch on synthetic 2D data to make the training dynamics visible, and Part 2 applies convolutional and tabular GANs to three very different real datasets — medical imaging, network-security traffic and hand-drawn sketches. Throughout, the emphasis is on *why* a given architecture suits a given dataset and on interpreting the training behaviour and evaluation metrics rather than merely describing them. Part 1 and the two image problems are implemented in PyTorch; the tabular traffic model uses TensorFlow/Keras, demonstrating both frameworks. Seeds are fixed (`TORCH_SEED = 3030`, `KERAS_SEED = 4040`) so every figure and number below is reproducible.

## Part 1: Building and Understanding GANs from Scratch

Part 1 is written from scratch in PyTorch: the generator and discriminator are small multilayer perceptrons trained with the non-saturating binary cross-entropy loss (Goodfellow et al., 2014). A single configurable `Toy2DGAN` trainer drives all three tasks, so the only things that change between experiments are the target distribution and the network shape. A small amount of instance noise is added to the discriminator's inputs and decayed to zero over training; this is a standard stabiliser that stops the discriminator separating the real and fake point clouds too early and starving the generator of gradient.

### Task 1: Reproduce the sine-wave GAN

The tutorial task samples points along `y = sin(x)` and trains for 6000 steps. This is the cleanest possible check that the implementation is correct, because the non-saturating loss has a known equilibrium: the discriminator should sit near `2ln2 ≈ 1.386` and the generator near `ln2 ≈ 0.693` when neither network can improve.

![Figure 1](Set%204%20Result%20Images/117d337be2a2754d900d881e21ea9198e40804e4.png)

**Figure 1.** Task 1 — real (blue) versus generated (orange) samples for the sine-wave target after 6000 training steps.

Figure 1 shows the generated points lying almost exactly on the sine curve across the full `[-π, π]` domain, with no obvious gaps or clumping. This visual match is backed up by the loss trace: the discriminator settles around 1.39 and the generator around 0.69, essentially on top of the theoretical equilibrium. That agreement is the strongest single piece of evidence in Part 1 that the from-scratch implementation has genuinely converged rather than one network having run away — a common failure mode in GAN training.

### Task 2: A new 2D distribution (2D spiral)

For the required new distribution I chose a 2D spiral, generated as a noisy Archimedean spiral. The spiral is deliberately harder than the sine curve because its arms curl back on themselves, so the generator must learn a self-overlapping structure rather than a single-valued function; it was therefore given a longer budget of 8000 steps.

![Figure 2](Set%204%20Result%20Images/3070d1ab0ceb8e09ec41c8dda0a2c606f7bbd1d6.png)

**Figure 2.** Task 2 — real versus generated samples for the 2D spiral after 8000 steps.

Figure 2 shows the generator tracing the spiral's arms rather than collapsing onto a single loop or a subset of the structure, which would be the signature of mode collapse. The losses stay close to equilibrium (discriminator ≈ 1.45, generator ≈ 0.66), with a brief excursion to a generator loss near 1.1 around step 6000 before settling — a small, self-correcting wobble rather than divergence. The main visible weakness is that the outermost arm is a little sparser than the real data, which is expected since that region has the least probability mass and therefore the weakest learning signal.

### Task 3: Modify the architecture and compare

To study architectural sensitivity, the target and training budget were held fixed and only the network was changed. The baseline generator has depth 3 with a GELU activation; the modified generator is deeper (depth 5) and uses LeakyReLU with a 0.3 slope. Isolating these two factors makes the comparison interpretable.

![Figure 3](Set%204%20Result%20Images/1b89c3e63e5e4a641ef13e5837833221de1a389f.png)

**Figure 3.** Task 3 — generated spiral point clouds for the baseline (depth 3, GELU) and modified (depth 5, LeakyReLU-0.3) generators against the real data.

Figure 3 shows the two point clouds are very similar in coverage and shape, and both networks hold losses near equilibrium throughout (discriminator ≈ 1.37–1.46, generator ≈ 0.66–0.76, ending at D ≈ 1.46 / G ≈ 0.69 for the modified network). The deeper LeakyReLU model therefore does not destabilise training; if anything its loss trace is marginally smoother. On a target this simple I would not over-interpret the small difference — the useful conclusion is that added depth and a leaky activation are safe changes here, which motivates using deeper convolutional variants for the harder image problems in Part 2.

## Part 2: Real-World GAN Applications

### Part 2.1: Optical coherence tomography (OCT) retinal images with MedMNIST

The OCTMNIST subset of MedMNIST provides 97,477 grayscale 28×28 training scans across four diagnostic classes — choroidal neovascularization, diabetic macular edema, drusen and normal (Yang et al., 2023). Because the data are images with strong local spatial structure, a deep convolutional GAN (DCGAN) is the natural choice: convolutions share weights across the image and encode the translation-invariance that fully-connected layers cannot (Radford, Metz and Chintala, 2016). The generator maps a latent vector (dimension 100) to a 32×32 scan through a linear projection followed by nearest-neighbour upsampling and 3×3 convolutions, and the discriminator mirrors this with strided convolutions. I deliberately used resize-convolution (upsample-then-convolve) instead of transposed convolutions because the latter are prone to checkerboard artefacts on small grayscale images (Odena, Dumoulin and Olah, 2016); in practice this choice gave a cleaner, lower FID.

![Figure 4](Set%204%20Result%20Images/56fc192fbd336d99f0349ee8477da692859693d4.png)

**Figure 4.** OCTMNIST class distribution across the four diagnostic labels.

Figure 4 shows the dataset is imbalanced, with some diagnostic classes far better represented than others. This matters for the unconditional DCGAN, which sees the classes mixed together and will naturally spend most of its capacity on the dominant modes; it also foreshadows the behaviour of the conditional model below, where the better-represented classes are reproduced most convincingly.

![Figure 5](Set%204%20Result%20Images/ba8e019a9fb23d9f9e6ecd43ee33af9b18e03d76.png)

**Figure 5.** OCT DCGAN generator and discriminator loss curves over 45 epochs.

Figure 5 shows the characteristic DCGAN pattern rather than a textbook equilibrium: the discriminator loss stays low (ending near 0.393) while the generator loss is higher and noisier (ending near 4.090). This is expected for a convolutional GAN with one-sided label smoothing — the discriminator has an easier task on 28×28 scans — and the key point is that neither loss diverges and the generator continues to receive usable gradients across all 45 epochs, which is what allows the sample quality below.

![Figure 6](Set%204%20Result%20Images/9e98b738bd8e579a1f7988a79e793c2e959c4ab9.png)

**Figure 6.** Real (left) versus generated (right) OCT retinal scans from the trained DCGAN.

Figure 6 shows the generated scans reproduce the defining structure of a real OCT image — a bright, curved retinal band over a darker background — and the band's position and curvature vary across the batch rather than the model repeating one memorised image. Quantitatively, the Fréchet Inception Distance (Heusel et al., 2017), computed with an ImageNet-pretrained Inception-v3 feature extractor (Szegedy et al., 2016), is **31.94**, the strongest image score in this project. The main remaining weakness is the fine sub-layer texture rather than the gross anatomy, consistent with the low-but-not-tiny FID.

**Reflection and the "duck test".** The results are good, but they should be read with care. A discriminator only ever learns to answer "real or fake", never "anatomically correct". This is the GAN version of the duck test — if a sample looks and "quacks" convincingly, it passes — so nothing in the objective stops the generator inventing a bright region or a layer boundary that no real patient scan ever had, provided it is statistically plausible. A good FID only tells us the *aggregate* feature statistics of a batch are close to real; it says nothing about whether any single image is trustworthy. In a retinal scan a hallucinated bright patch could mimic pathology that is not there, which is exactly why I would restrict synthetic scans to augmentation or teaching and re-validate any classifier trained on them against real-only data.

**Extension — conditional GAN.** The extension conditions the generator on the diagnostic label by concatenating a learned class embedding with the latent vector, while the discriminator receives the label as an extra image channel (Mirza and Osindero, 2014).

![Figure 7](Set%204%20Result%20Images/1499c8335b3ec723f1bd6e9764632d3d642a40ff.png)

**Figure 7.** Conditional GAN output, one diagnostic class per row, generated on demand.

Figure 7 shows the rows differ visibly from one another, confirming the generator responds to the label rather than ignoring it. In line with the class imbalance in Figure 4, the better-represented classes come out the most convincingly — the standard outcome for a conditional GAN trained on imbalanced data — and the model can now generate a chosen class on demand, satisfying the extension goal.

### Part 2.2: Cybersecurity — Synthetic Traffic with CICIDS 2017

CICIDS 2017 records each network flow as a vector of numeric features rather than an image (Sharafaldin, Lashkari and Ghorbani, 2018). Combining the eight daily files gives 2,830,743 flows and 78 numeric features. Because the data are tabular with no spatial structure, a convolutional model makes no sense; instead I use a fully-connected (dense) GAN implemented as a `keras.Model` subclass with a custom `train_step`, latent dimension 64. The generator uses a linear output because the scaled features are unbounded, and the critic returns a single logit.

![Figure 8](Set%204%20Result%20Images/96c5fc13ac27f1b4c7c7ac54c2baee62d4e868b3.png)

**Figure 8.** CICIDS 2017 overall class distribution (log scale) and flows per day.

Figure 8 shows extreme class imbalance: BENIGN dominates with 2,273,097 flows, followed by DoS Hulk (231,073) and DDoS (128,027), with several attack families down in the thousands. The log scale is necessary just to keep the minority classes visible. This imbalance is the central challenge for the generator and directly explains the alignment results below. Before training, features were mapped to a standard normal with a rank-based quantile transformer, because CICIDS contains extreme outliers that otherwise destabilise GAN training.

Following the brief, the core model trains on the **Wednesday** file only (BENIGN versus DoS — Hulk, GoldenEye, slowloris and Slowhttptest; 691,406 rows after cleaning, sampled to 20,000 for training). It is worth noting that the brief names this the "DDoS" task, but Wednesday's attacks are single-source DoS; true DDoS appears in the Friday file and only enters through the all-days extension.

![Figure 9](Set%204%20Result%20Images/41f018da649ab0b39d373bf47ca65bf24f9558b5.png)

**Figure 9.** PCA of real BENIGN, real DoS and synthetic feature vectors (Wednesday model).

Figure 9 projects the 78-dimensional vectors to two principal components. The synthetic points (orange) sit within the spread of the real BENIGN and DoS points rather than forming a separate island, which is the first sign that the generator has learned the broad shape of the distribution rather than collapsing to a single point.

![Figure 10](Set%204%20Result%20Images/9f2c0ecdefc01a04e2e11c73dbddea209bf7035d.png)

**Figure 10.** t-SNE of real versus synthetic feature vectors (Wednesday model).

Figure 10 gives the non-linear view (van der Maaten and Hinton, 2008), which is better at exposing local cluster structure. The synthetic points overlap the real manifold across most regions but are visibly thinner in some pockets, matching the quantitative alignment gaps: the mean absolute per-feature difference is **0.4730** for the means and **0.5293** for the standard deviations. This is a moderate rather than tight match — the generator captures the general shape but under-represents the spread of some features, typically the heavy-tailed or near-binary ones such as byte/packet counts and flag columns, which are the known hard case for tabular GANs.

**Extension — full multi-day dataset.** Retraining the same dense GAN on a 30,000-row sample spanning all days tests whether one unconditional generator can cover many attack families at once.

![Figure 11](Set%204%20Result%20Images/6185fa52602d60257146e00448b17091fe09c63a.png)

**Figure 11.** Full-dataset PCA — real points coloured by attack family versus synthetic points.

Figure 11 shows the real data now spans many families, and the synthetic points concentrate where the mass is. The alignment gaps get slightly worse (mean **0.5454**, standard deviation **0.6094**) than the Wednesday-only run — the expected result of asking one generator to spread its capacity across many more families. BENIGN and the high-volume attacks (DoS Hulk, the true DDoS) dominate what it reproduces well, while rare families such as Heartbleed and Infiltration are under-represented. The duck-test caution applies even more forcefully here than for images: there is no "just look at it" check for a row of 78 numbers, so a fabricated flow only has to satisfy the critic's weak notion of "looks like traffic". A fake "attack" matching no real signature could teach a downstream intrusion detector the wrong pattern, and a fake "benign" flow drifting toward attack-like statistics could inflate false positives — which is precisely why the PCA/t-SNE overlap and the per-feature gaps carry more weight here than a visual check would for images.

### Part 2.3: Creative AI — QuickDraw 'birthday cake' Subset

The QuickDraw birthday-cake category provides 144,982 grayscale 28×28 doodles (Ha and Eck, 2017). These are sparse line drawings, but they are still images with local structure, so the same resize-convolution DCGAN used for OCT is appropriate; the bitmaps are served through a custom `torch.utils.data.Dataset`. Training tracks losses and snapshots samples across epochs, and quality is measured with FID.

![Figure 12](Set%204%20Result%20Images/81dcbf6f6c1199ca640664ec997fc53223a341d9.png)

**Figure 12.** Real (left) versus generated (right) 'birthday cake' sketches.

Figure 12 shows the generated cakes are recognisable — a cake-body silhouette with candle-like marks on top appears consistently across the batch — and training stayed stable across the 45 epochs (final losses D ≈ 0.416 / G ≈ 4.214) without diverging. The FID is **41.01**, a solid score for sparse sketches at this resolution, and the samples are clearly drawn from the learned category rather than being noise.

**Extension — other categories and sketch complexity.** The pipeline was repeated on two further categories, *cat* and *house*, and their FID scores placed against an ink-density complexity proxy. The cat model initially collapsed under a plain one-to-one generator/discriminator schedule, so — demonstrating an alternative training strategy rather than a new architecture — cat was trained with three generator updates per discriminator update, a lower discriminator learning rate (a two-time-scale update rule; Heusel et al., 2017) and a longer 60-epoch budget. This brought its FID down from a collapsed value near 98 to 67.36.

![Figure 13](Set%204%20Result%20Images/3ef116d51773d161b1fa3ed9ff4919f03b09210f.png)

**Figure 13.** FID by category (left) versus ink-density complexity proxy (right).

Figure 13 reveals a clean relationship. The ink-density order is house (0.1721) < cake (0.1967) < cat (0.1986), and the FID order matches it exactly: house (**28.76**) < cake (**41.01**) < cat (**67.36**). So the sparsest, most regular category is the easiest to model and the busiest, most varied one is the hardest. Beyond raw ink density, the deeper driver is intra-class variety: cat sketches differ far more from one another (pose, ear shape, whiskers, sitting versus standing) than houses do, and that variety is the harder thing for a fixed-capacity DCGAN to capture — which is why cat remains the weakest score even after the stronger training schedule.

## Conclusion

Across all five experiments the implementations behaved as the theory predicts and the metrics tell a consistent story. Part 1 confirmed correctness by landing the sine-wave GAN on its analytic equilibrium and showed that a deeper, leaky-activated generator is a safe modification. Part 2 then matched architecture to data: a resize-convolution DCGAN for the OCT scans (FID 31.94) and QuickDraw sketches (cake 41.01, house 28.76, cat 67.36), and a dense feature-vector GAN for CICIDS traffic (alignment gaps 0.4730/0.5293 on Wednesday). The recurring themes are that class imbalance limits how faithfully minority modes are reproduced, that FID and alignment gaps measure aggregate fidelity but not individual trustworthiness (the duck test), and that intra-class variety predicts difficulty better than surface complexity alone. For future work I would add a conditional or Wasserstein objective to the tabular model to better cover rare attack families, increase generator capacity for high-variety sketch classes such as cat, and pair FID with a precision–recall style metric to separate sample fidelity from diversity.

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
