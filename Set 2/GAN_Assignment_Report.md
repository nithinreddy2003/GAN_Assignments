# Generative Adversarial Networks: From Synthetic 2D Data to Medical, Cybersecurity and Creative Applications

## 1. Introduction

This report documents my work building generative adversarial networks (GANs) from scratch and then applying them to three real-world problems. A GAN pairs a generator that maps random noise to samples against a discriminator that tries to separate real data from generated data; the two are trained together in a minimax game so that, at convergence, the generated distribution should be indistinguishable from the real one (Goodfellow et al., 2014). Part 1 reinforces the mechanics on controllable 2D data, where success or failure can be judged by eye. Part 2 moves to harder, higher-dimensional data in medicine (OCTMNIST), cybersecurity (CICIDS 2017) and creative sketching (QuickDraw). Throughout, my aim was not just to make the models train, but to interpret *why* a particular architecture suits each dataset and what the loss curves, projections and Fréchet Inception Distance (FID) scores actually tell me about sample quality. Every figure and number quoted below is produced directly by the accompanying notebook, and a single fixed seed (42) is re-applied before each run so the results are reproducible.

## Part 1: Building and Understanding GANs from Scratch

For Part 1 the generator and discriminator are small multi-layer perceptrons (MLPs) built by one configurable factory, so the architecture change in Task 3 reuses exactly the same training loop. Training uses the non-saturating objective through `BCEWithLogitsLoss`, which avoids the vanishing gradients the generator suffers early on under the original minimax loss (Goodfellow et al., 2014), optimised with Adam (Kingma and Ba, 2015). MLPs are the right choice here because the data are unordered 2D coordinates with no spatial grid to exploit — there is nothing for a convolution to slide over.

### Task 1: Reproduce the sine-wave GAN

The baseline is a 16-dimensional latent vector feeding a width-128, depth-2 ReLU network, trained for 6,000 steps against points sampled as `y = sin(x)` on `[-π, π]`. This run is mainly a sanity check on the shared training code before it is reused on harder targets.

![Real (blue) versus generated (red) samples for the sine-wave target after 6,000 training steps.](report_images/fig01_sine_real_vs_generated.png)

**Figure 1.** Real versus generated samples for the sine-wave GAN.

Figure 1 shows the generated points (red) lying along the true `sin(x)` curve, with only minor scatter near the turning points where the curve is most nonlinear. This visual agreement is backed up by the loss trace, which settles close to the theoretical equilibrium and holds there: the discriminator hovers around 1.38 and the generator around 0.70 for the whole run (final step: `D=1.3778`, `G=0.6964`). A discriminator loss near `2·ln(2) ≈ 1.386` is the signature of a well-balanced game in which it can no longer reliably tell real from fake, so the baseline pipeline is behaving exactly as intended.

### Task 2 and Task 3: A new 2D distribution and an architecture change

For the new distribution I chose a **mixture of Gaussians**: eight isotropic modes evenly spaced on a ring of radius 2.0 with per-mode standard deviation 0.05, trained for 8,000 steps. I picked this deliberately because a ring of well-separated modes is the standard probe for *mode collapse* — a failure in which the generator ignores part of the data distribution. A collapsed generator would visibly cover only a subset of the eight blobs. Task 3 then holds everything else fixed and changes only the network, moving from the 2×128 ReLU baseline to a deeper, wider 4×256 network with the self-normalising SELU activation (Klambauer et al., 2017), which keeps activations better-scaled in deeper stacks.

![Baseline (2x128, ReLU) versus enhanced (4x256, SELU) generators on the eight-mode Gaussian ring; real points in blue, generated points in red.](report_images/fig02_mog_baseline_vs_enhanced.png)

**Figure 2.** Architecture comparison on the mixture of Gaussians.

Figure 2 is the most informative single plot in Part 1 because it captures both required tasks at once. The baseline (left) reaches all eight modes rather than parking on one or two, so there is *no hard mode collapse*; however, the red cloud is loose, filling the gaps between blobs instead of sitting tightly on the eight centres. This is the classic "covers the modes but not sharply" behaviour of a small vanilla GAN. The enhanced network (right) tightens the samples noticeably — the red points concentrate on the mode centres and trace a clean ring. The loss curves stay stable in both cases (the enhanced run ends at `D=1.4444`, `G=0.8657`), confirming that the extra depth, width and smoother SELU activation buy sharper, better-localised samples here rather than fixing a collapse that never actually occurred. The practical inference is that, for this target, capacity and activation choice mainly control *sample sharpness*, not *mode coverage*.

## Part 2: Real-World GAN Applications

### Part 2.1: Optical coherence tomography (OCT) retinal images with MedMNIST

OCTMNIST is a subset of MedMNIST containing 28×28 grayscale retinal OCT scans across four diagnostic classes (Yang et al., 2023). The goal is to train a DCGAN to synthesise plausible scans. A DCGAN is the natural choice for images because it replaces dense layers with strided and transposed convolutions, uses BatchNorm (Ioffe and Szegedy, 2015) and LeakyReLU/ReLU activations, and follows a weight-initialisation recipe shown to stabilise image GANs (Radford et al., 2015). I work at 32×32 (a clean power of two for the four conv/transpose stages) and scale pixels to `[-1, 1]` to match the generator's final Tanh.

![OCTMNIST training-set class distribution across the four diagnostic classes.](report_images/fig03_octmnist_class_distribution.png)

**Figure 3.** Class distribution of the OCTMNIST training split.

Loading the data confirms four classes (choroidal neovascularization, diabetic macular edema, drusen and normal) and a substantial training split of shape `(97477, 28, 28)`. Figure 3 shows the classes are markedly imbalanced — roughly a 6:1 ratio between the largest and smallest — which matters later because the conditional extension must correct for it or the minority classes will barely be learned. For the unconditional DCGAN I preprocess a random 40,000-image subset.

![Generator and discriminator loss curves over 40 epochs of OCTMNIST DCGAN training.](report_images/fig04_octmnist_loss_curves.png)

**Figure 4.** Generator/discriminator losses during OCTMNIST DCGAN training.

Figure 4 tracks the adversarial game over 40 epochs. Training begins highly unbalanced (`D=0.39`, `G=5.35` at epoch 1), meaning the discriminator initially dominates and the generator is heavily penalised. As training proceeds the two losses move into a much closer back-and-forth (roughly `D≈0.9–1.1`, `G≈1.3–2.6` by mid-training). This is the healthy shape for a DCGAN that is learning rather than collapsing; the occasional spike (for example the epoch-19 jump to `D=3.16`) is normal adversarial oscillation rather than divergence.

![Real OCTMNIST scans (left) versus DCGAN-generated scans (right).](report_images/fig05_octmnist_real_vs_fake.png)

**Figure 5.** Real versus generated OCT scans, with FID = 43.70.

Figure 5 places real scans beside generated ones for direct visual comparison, and I quantify the gap with FID, which compares the mean and covariance of InceptionV3 features between the two sets (Heusel et al., 2017; Szegedy et al., 2016). The trained generator scores **FID = 43.70** (lower is better). Visually the fakes capture the bright curved retinal band over a darker background and are varied rather than collapsed, but they are softer than the real scans with occasionally smeared sub-layers — consistent with a mid-40s FID for a 32×32 DCGAN after 40 epochs. I read the score as *directional* rather than absolute, because Inception was trained on natural RGB photographs, not grayscale medical images, so its feature space is an imperfect judge here.

**Extension — conditional GAN.** To generate a chosen class on demand I implemented a conditional DCGAN (Mirza and Osindero, 2014), embedding the label and concatenating it with the latent vector for the generator while adding it as an extra input channel for the discriminator.

![Conditional DCGAN samples generated per diagnostic class on demand.](report_images/fig06_octmnist_cgan_per_class.png)

**Figure 6.** Class-conditional samples from the cGAN extension.

A naïve conditional model collapsed for me — the discriminator loss fell to zero, the generator loss blew up, and the output became noise. Three standard fixes stabilised it: one-sided label smoothing (real target 0.9) to stop the discriminator becoming over-confident (Salimans et al., 2016), a two time-scale update rule with a slower discriminator (Heusel et al., 2017), and class-balanced sampling to counter the ~6:1 imbalance from Figure 3. Figure 6 shows the fixed model producing samples for each class on request; the losses recover into a stable range (from `D=0.66`, `G=3.03` at epoch 1 to a healthy balance by the end). The visual differences between classes remain subtle, which honestly reflects how similar these scans look even in the real data.

**Reflection, and the "generating ducks" question.** The brief asks us to reflect on quality and to comment on using GANs to "generate ducks". The serious point behind the joke is that this DCGAN has no idea it is looking at eyes — it only ever sees a pile of 32×32 arrays and learns their statistics. If I swapped OCTMNIST for a folder of duck photographs, the identical code would learn those instead. That content-agnosticism is exactly why the same recipe transfers across all of Part 2, but it is also the core limitation: the discriminator only learns *real versus fake*, never *anatomically plausible*, so the generator can render a convincing-looking structure with no basis in real anatomy. Synthetic scans are therefore fine for augmentation or teaching but should not feed a real diagnostic pipeline without expert review.

### Part 2.2: Cybersecurity – Synthetic Traffic with CICIDS 2017

CICIDS 2017 stores each network flow as roughly 78 numeric features rather than an image (Sharafaldin, Lashkari and Ghorbani, 2018). Because the data are tabular, a convolutional DCGAN is inappropriate — there is no spatial locality for a kernel to exploit — so I use a **fully-connected GAN** that keeps the DCGAN *training recipe* (adversarial `BCEWithLogitsLoss`, BatchNorm in the generator, a LeakyReLU discriminator with dropout, Adam, one-sided label smoothing) but replaces convolutions with dense layers and uses a linear output because the standardised features are unbounded. Following the brief I first combine all per-day CSVs, then focus training on the Wednesday file.

![Left: overall class distribution across all days (log scale). Right: flow counts per day.](report_images/fig07_cicids_class_balance.png)

**Figure 7.** Combined CICIDS 2017 class balance and per-day volumes.

Merging the eight day-files yields a combined frame of **2,830,743 flows × 80 columns** with **78 numeric features**, requiring cleaning of 4,376 infinite and 1,358 missing values. Figure 7 shows BENIGN dominating so heavily that a log scale is needed to see the attack classes at all — a severe imbalance that foreshadows the model's later difficulty with rare attacks. A brief label note: the brief says "DDoS", but the Wednesday file actually contains single-source DoS attacks (Hulk, GoldenEye, slowloris, Slowhttptest), so I treat it as BENIGN vs DoS and flag the mismatch deliberately. Wednesday alone gives BENIGN = 440,031 and DoS Hulk = 231,073 as the two largest labels. Features are standardised and clipped to ±5σ so CICIDS's extreme outliers cannot destabilise training, and the GAN is trained on a 20,000-row sample for 60 epochs (ending at `D=1.0799`, `G=1.3983`).

![t-SNE projection of real BENIGN, real DoS and synthetic Wednesday feature vectors.](report_images/fig08_cicids_wednesday_tsne.png)

**Figure 8.** t-SNE of real versus synthetic Wednesday traffic.

Because I cannot eyeball a feature vector the way I can an image, I evaluate alignment by projecting real and synthetic vectors to 2D with PCA (a linear view) and t-SNE (a nonlinear, local-structure view; van der Maaten and Hinton, 2008). Figure 8 shows the synthetic cloud sitting on top of the dense BENIGN region and pushing partway into the DoS clusters, without fully reproducing their separate structure. Quantitatively, the average absolute per-feature gap (in standardised units) is **0.1057 on the means and 0.1975 on the standard deviations**. Together these say the generator matches *where most traffic lives* and *typical feature values* well, but comes out smoother than the real data and does not reach the DoS tails — the usual outcome for a plain unconditional GAN on heavy-tailed, partly-binary network features.

**Extension — all days and generalisation.** Retraining the same model on the full dataset (2,827,876 rows, 30,000 sampled) tests how well one GAN spans many attack families.

![PCA of the full-dataset run: real points coloured by attack family versus synthetic samples.](report_images/fig09_cicids_alldays_pca_family.png)

**Figure 9.** All-days PCA, real points coloured by attack family.

Figure 9 colours the real points by family. The synthetic cloud leans toward BENIGN and the large attack blobs while barely touching rare families — the comparison sample contains only 2 Bot, 11 Brute Force and 2 Web Attack real points, so there is almost nothing for the model to learn from. Interestingly, adding data slightly *improves* the mean-gap (0.1017 vs 0.1057) but *worsens* the std-gap (0.2762 vs 0.1975): more data sharpens the averages, but the model still cannot stretch to cover the extra variance introduced by many attack types. My inference is that a single unconditional GAN does not generalise cleanly across attack families; a conditional GAN or one model per family would be the sensible next step. As with Part 2.1, the "duck" point applies with extra force here — a synthetic flow can be statistically plausible yet physically impossible as a real connection, which is precisely why the distribution-overlap and per-feature-gap checks matter more than any visual inspection could.

### Part 2.3: Creative AI – QuickDraw 'birthday cake' Subset

The final task trains a DCGAN on the QuickDraw 'birthday cake' category — 28×28 grayscale sketch bitmaps (Ha and Eck, 2018). Here I implement the model in TensorFlow/Keras (Abadi et al., 2016) in the object-oriented `model.fit` style: the generator and discriminator are built with the functional API, wrapped in a `keras.Model` subclass whose `train_step` performs the adversarial update. I deliberately used a different framework and coding style from Parts 1–2.2 to demonstrate both. The configuration is a 100-d latent vector, 48 base feature maps, Adam at `lr = 0.00015` (`β₁ = 0.5`) and 35 epochs at batch size 256; the generator has 677,952 parameters. The category provides 144,982 sketches, of which I train on a random 30,000 to keep epoch time reasonable.

![Real birthday-cake sketches (left) versus DCGAN-generated sketches (right).](report_images/fig10_quickdraw_cake_real_vs_fake.png)

**Figure 10.** Real versus generated birthday-cake sketches, FID = 41.32.

Figure 10 compares real and generated cakes; the model scores **FID = 41.32** using the same InceptionV3-feature approach as Part 2.1. The generated cakes are clearly recognisable — a rounded or rectangular body, candles on top and often a plate underneath. Line-art faults such as broken strokes and speckle are visible up close but do not hide the subject, which is consistent with a low-40s FID. The training losses stay in a healthy back-and-forth (from `d_loss=0.55`, `g_loss=2.84` at epoch 1), with only occasional spikes.

![Fixed-noise generator samples captured at five evenly-spaced epochs during training.](report_images/fig11_quickdraw_cake_epoch_progression.png)

**Figure 11.** Sample progression across training epochs.

Figure 11 shows fixed-noise samples logged at five epochs. Early snapshots are blurry blobs; later ones sharpen into recognisable cake outlines with candles, which is the qualitative counterpart to the loss curves stabilising. Tracking the same noise vectors over time is a useful diagnostic because it separates genuine learning from mode-hopping — the samples evolve smoothly toward structure rather than jumping between unrelated shapes.

**Extension — other categories and sketch complexity.** I retrained the identical model on two further categories and compared FID against an "ink fraction" proxy for sketch complexity.

![FID versus ink-fraction across the birthday cake, cat and house categories.](report_images/fig12_quickdraw_category_fid_vs_ink.png)

**Figure 12.** Cross-category FID versus ink-fraction complexity proxy.

Figure 12 reports **house = 37.39, cat = 41.22 and cake = 41.32**, while the ink-fraction values are almost flat (0.2368, 0.2655, 0.2708). Because ink density barely varies, it clearly does not explain the FID differences. House is the easiest category, which fits intuition — houses are structurally rigid (a square with a triangular roof), whereas cats and cakes vary far more in pose, tiers and candle count. My inference is that *shape variety*, not raw ink coverage, drives difficulty; I note, though, that the cat–cake gap (0.10 FID) is small enough to sit within run-to-run noise, so I would not read it as a firm ranking.

## Conclusions and Future Work

Across the four parts, the same adversarial recipe adapted cleanly to very different data by changing only the input representation and network family. Part 1 confirmed the mechanics: a small MLP learned a sine wave almost perfectly and reached every mode of a Gaussian ring, and adding depth/width with SELU sharpened the samples without altering mode coverage. Part 2.1 produced plausible OCT scans (FID 43.70) and, once stabilised with label smoothing, TTUR and balanced sampling, a conditional model that generates any diagnostic class on demand. Part 2.2 showed a dense feature-vector GAN matching the bulk of real network traffic (mean/std gaps of 0.1057/0.1975 on Wednesday) but missing the attack tails, with the all-days run confirming that one unconditional model cannot span many attack families. Part 2.3 generated recognisable cake sketches (FID 41.32) and showed that cross-category difficulty tracks shape variety rather than ink density.

Three themes for future work follow directly from these results. First, **coverage of rare classes**: both the OCT imbalance and the CICIDS rare-attack failure point to conditional generation, per-class models, or minority-aware sampling as the highest-value improvement. Second, **better-matched evaluation**: FID via ImageNet-trained Inception is only directional for grayscale medical scans and line-art, so a domain-specific feature extractor (or a downstream "train-on-synthetic, test-on-real" classifier) would give a more trustworthy quality signal. Third, **stronger objectives**: a Wasserstein loss with gradient penalty (Gulrajani et al., 2017) would likely tighten the loose Gaussian-mixture samples and the smoothed CICIDS distribution. Above all, the recurring lesson — framed by the "duck" question — is that a GAN only ever learns *statistical* resemblance, never *semantic correctness*, so synthetic medical or security data must be validated before it is trusted downstream.

## References

Abadi, M. et al. (2016) 'TensorFlow: A system for large-scale machine learning', in *Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*. Savannah, GA, pp. 265–283.

Goodfellow, I. et al. (2014) 'Generative adversarial nets', in *Advances in Neural Information Processing Systems (NeurIPS) 27*. Montreal, pp. 2672–2680.

Gulrajani, I. et al. (2017) 'Improved training of Wasserstein GANs', in *Advances in Neural Information Processing Systems (NeurIPS) 30*. Long Beach, CA, pp. 5767–5777.

Ha, D. and Eck, D. (2018) 'A neural representation of sketch drawings', in *International Conference on Learning Representations (ICLR)*. Vancouver.

Heusel, M. et al. (2017) 'GANs trained by a two time-scale update rule converge to a local Nash equilibrium', in *Advances in Neural Information Processing Systems (NeurIPS) 30*. Long Beach, CA, pp. 6626–6637.

Ioffe, S. and Szegedy, C. (2015) 'Batch normalization: accelerating deep network training by reducing internal covariate shift', in *Proceedings of the 32nd International Conference on Machine Learning (ICML)*. Lille, pp. 448–456.

Jolliffe, I.T. and Cadima, J. (2016) 'Principal component analysis: a review and recent developments', *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202.

Kingma, D.P. and Ba, J. (2015) 'Adam: a method for stochastic optimization', in *International Conference on Learning Representations (ICLR)*. San Diego, CA.

Klambauer, G. et al. (2017) 'Self-normalizing neural networks', in *Advances in Neural Information Processing Systems (NeurIPS) 30*. Long Beach, CA, pp. 971–980.

Mirza, M. and Osindero, S. (2014) 'Conditional generative adversarial nets', *arXiv preprint* arXiv:1411.1784.

Paszke, A. et al. (2019) 'PyTorch: an imperative style, high-performance deep learning library', in *Advances in Neural Information Processing Systems (NeurIPS) 32*. Vancouver, pp. 8024–8035.

Radford, A., Metz, L. and Chintala, S. (2015) 'Unsupervised representation learning with deep convolutional generative adversarial networks', *arXiv preprint* arXiv:1511.06434.

Salimans, T. et al. (2016) 'Improved techniques for training GANs', in *Advances in Neural Information Processing Systems (NeurIPS) 29*. Barcelona, pp. 2234–2242.

Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A. (2018) 'Toward generating a new intrusion detection dataset and intrusion traffic characterization', in *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*. Funchal, pp. 108–116.

Szegedy, C. et al. (2016) 'Rethinking the Inception architecture for computer vision', in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. Las Vegas, NV, pp. 2818–2826.

van der Maaten, L. and Hinton, G. (2008) 'Visualizing data using t-SNE', *Journal of Machine Learning Research*, 9, pp. 2579–2605.

Yang, J. et al. (2023) 'MedMNIST v2 — a large-scale lightweight benchmark for 2D and 3D biomedical image classification', *Scientific Data*, 10(1), 41.
