# Generative Adversarial Networks: From 2D Toy Data to Real-World Applications

## 1. Introduction

A Generative Adversarial Network (GAN) trains two networks against one another: a *generator* that maps random noise to samples, and a *discriminator* that tries to separate real data from generated data. The two are optimised in opposition, so that as the discriminator gets better at spotting fakes the generator is forced to produce increasingly convincing samples, and at the ideal equilibrium the discriminator can do no better than chance (Goodfellow et al., 2014). This report works through that idea in two stages. Part 1 builds a GAN from scratch in PyTorch and fits it to synthetic 2D distributions, where the learning behaviour is fully visible on a scatter plot. Part 2 then applies GANs to three real-world problems in different domains: generating optical coherence tomography (OCT) retinal scans (medicine), synthesising network-flow records for intrusion detection (cybersecurity), and drawing doodles (creative arts). Throughout, the emphasis is on *why* a particular architecture suits each problem and on interpreting what the trained models actually produce, rather than restating standard definitions.

---

## 2. Part 1: Building and Understanding GANs from Scratch

All three Part 1 tasks share a deliberately fixed training recipe so that later comparisons are controlled: a generator with BatchNorm (Ioffe and Szegedy, 2015) after each hidden layer, the Adam optimiser (Kingma and Ba, 2015) with `betas=(0.5, 0.999)`, and one-sided label smoothing with a real-label target of 0.9 (Salimans et al., 2016). The lower Adam `beta1` and the label smoothing are both standard GAN stabilisers: the first damps the oscillation that a momentum term of 0.9 tends to cause, and the second stops the discriminator becoming over-confident and starving the generator of gradient. The latent dimension is 16 and each hidden layer is 64 units wide.

### Task 1: Sine-wave GAN reproduction

The baseline reproduces the tutorial's sine-wave GAN: real points are drawn from `y = sin(x)` with small Gaussian noise, and a two-hidden-layer generator (ReLU) is trained for 5,000 epochs against a LeakyReLU discriminator. The losses settle into a stable equilibrium, ending at roughly D = 1.36 and G = 0.83, which is the balanced region you want to see, neither network dominating. Visually the generated points trace the sine curve across the full domain, with only slight thinning of coverage right at the crests of the peaks. This confirms the training loop works before moving to a harder target.

### Task 2: A new 2D distribution (spiral)

For the new distribution I chose a 2D Archimedean spiral, sampled with a square-root radius transform so points spread evenly rather than bunching near the centre, then scaled to roughly [-1.3, 1.3]. A spiral is a genuinely awkward target because it is a thin, multi-turn 1D manifold embedded in 2D, and a vanilla GAN tends to collapse it into a blob. Trained for 10,000 epochs with the shared recipe, the generator recovers both arms and the central curl, and the generated cloud lands on top of the real spiral. The discriminator loss drifts around 1.25-1.36 and the generator around 0.81-0.86, staying balanced throughout. It is worth being honest that this only works *because* of the stabilisers; BatchNorm in particular is what prevents the collapse.

### Task 3: Architecture modification and comparison

The architecture experiment changes exactly two things and holds everything else constant (latent dimension, width, learning rate, epochs, BatchNorm, label smoothing): the activation moves from **ReLU to LeakyReLU(0.2)** and the hidden depth from **2 to 4**. Isolating the change this way means any difference can be attributed to the architecture rather than to luck. LeakyReLU keeps a small gradient for negative inputs, so fewer units go permanently dead, and the extra depth gives the generator more capacity to bend a thin curve.

![Real spiral (left) versus the original ReLU/depth-2 generator (centre) and the modified LeakyReLU/depth-4 generator (right); the modified model traces both arms far more tightly with much less stray noise.](./images/task3_comparison.png)

The three-panel comparison makes the effect clear. The original generator scatters a visible number of stray points off the curve and partially fills the empty centre of the spiral, whereas the deeper LeakyReLU model draws a tighter, cleaner spiral along both arms with far less noise. The cost of that sharper fit shows up in the loss traces: the modified generator's loss runs higher and noisier, fluctuating up towards 1.5-1.8 (ending near G = 1.19) rather than sitting flat around 0.82 as the original does. That is consistent with a harder-working generator pushing against a discriminator it cannot easily satisfy, and it is the expected signature of increased generator capacity rather than of instability.

![Spiral GAN loss curves; discriminator and generator remain balanced across 10,000 epochs, which is what allows the spiral to be learned instead of collapsing.](./images/spiral_losses.png)

The take-away from Part 1 is that a modest, well-motivated architecture change (non-saturating activation plus depth) measurably improves how precisely the generator fits a difficult low-dimensional manifold.

---

## 3. Part 2.1: OCTMNIST Retinal Images (DCGAN, TensorFlow/Keras)

### Dataset exploration

OCTMNIST, from the MedMNIST v2 benchmark (Yang et al., 2023), contains 97,477 training scans of 28x28 grayscale OCT retinal images across four diagnostic classes: *choroidal neovascularization*, *diabetic macular edema*, *drusen*, and *normal*. The classes are imbalanced, which matters later for the conditional extension. A grid of real scans shows the common structure the model must learn: a bright, curved retinal band sitting over a darker background, with faint layering beneath it.

### Model description and justification

The useful structure in these scans is spatial and local, exactly what convolutional filters are built for, so a Deep Convolutional GAN (DCGAN) is the natural choice over a fully-connected one (Radford et al., 2016). The generator maps a 100-dimensional latent vector through a dense layer and three `Conv2DTranspose` blocks (each with BatchNorm and ReLU) up to a 32x32 `tanh` image; the discriminator mirrors this with strided `Conv2D` layers and LeakyReLU, plus 0.3 dropout for regularisation, and outputs a single logit. This follows the DCGAN recipe closely: strided convolutions instead of pooling, BatchNorm to stabilise the deep stack, and no fully-connected hidden layers. The generator has about 1.08M parameters. Training uses Adam (lr = 2e-4, `beta1` = 0.5) with one-sided label smoothing, on a 30,000-image subset for 60 epochs.

### Training and loss curves

Across the 60 epochs the discriminator loss stays low (roughly 0.4-0.6) while the generator loss bounces in the 2-5 range without ever collapsing to zero or exploding. That pattern, a discriminator that keeps a modest edge but never wins outright, is the healthy case: it means the generator kept receiving usable gradient throughout. Epoch snapshots taken at epochs 1, 16, 30, 45 and 60 show near-random texture at epoch 1, a recognisable layered band emerging by the middle snapshots, and only small refinements after that.

### Real vs fake comparison and FID

![OCTMNIST real scans (left) versus DCGAN-generated scans (right); the generator reproduces the curved retinal band and its variety, but the fakes are visibly softer than the real scans.](./images/octmnist_real_vs_fake.png)

Side by side, the generated scans clearly capture the dominant retinal-band structure and, importantly, show genuine variety across the batch rather than one repeated image, so there is no obvious mode collapse. The measured Fréchet Inception Distance (Heusel et al., 2017) is **79.02** (lower is better). That is on the high side and matches the visible blur: the fakes are softer and lose the fine sub-layers and speckle of the real scans. Three factors explain the elevated value beyond raw image quality: only 60 epochs of training on a subset, upscaling from 28 to 32 pixels, and the fact that FID is computed from an InceptionV3 network (Szegedy et al., 2016) trained on natural ImageNet photographs rather than single-channel medical images, so its feature space is an imperfect fit for this domain. FID also varies slightly between runs even with a fixed seed because not all GPU operations are deterministic, so the exact figure is best read as indicative.

### Reflection: quality, flaws, and the "looks real vs is true" problem

The central risk in medical image synthesis is captured by the duck test, if it looks like a duck and quacks like a duck, we call it a duck, except that fooling exactly that surface judgement is a GAN's entire objective. The discriminator only ever answers "real or fake"; nothing in the training objective checks that a generated structure corresponds to real physiology. The generator can therefore invent a bright band, a lesion-like patch, or a layer boundary that never occurred in any real patient, as long as it looks statistically plausible. This is the core hazard of treating GAN output as data: it is optimised for looking real, not for being true. In a retinal-scan context a hallucinated bright region could mimic disease that is not there, or smooth over a real anomaly, so a synthetic scan must never be used as a genuine diagnostic image. The defensible uses are data augmentation and teaching, and any classifier trained partly on synthetic scans should still be validated against a real-only holdout set to confirm it has not simply learned to recognise GAN artefacts.

### Extension: conditional GAN

The unconditional model cannot be asked for a specific diagnosis. The conditional GAN (Mirza and Osindero, 2014) fixes this by feeding the class label into both networks: on the generator side the label is embedded and concatenated with the latent vector, and on the discriminator side it is embedded, reshaped to an image-sized map, and added as an extra channel, which forces the discriminator to judge realism *for the requested class*. Two practical tweaks help: each class is oversampled to the same count so the rarer diagnoses are not ignored, and the discriminator is given a slower learning rate (1e-4 vs the generator's setup), a two-time-scale update rule that keeps training stable (Heusel et al., 2017).

![Conditional GAN output with one diagnostic class per row; the model produces class-consistent scans on demand, confirming the label conditioning is being used.](./images/octmnist_cgan_per_class.png)

Generating one class per row confirms the conditioning works: each row is visibly class-consistent, so the model can produce a chosen category on demand rather than sampling the dataset at random. The conditional losses end around D = 0.34, G = 4.90, similar in character to the unconditional run.

---

## 4. Part 2.2: Cybersecurity - CICIDS2017 Synthetic Traffic (Tabular GAN, PyTorch)

### Dataset exploration and class balance

CICIDS2017 (Sharafaldin et al., 2018) is split across eight day-files that must be combined. Reading only the label column across all days gives a heavily imbalanced picture: 2,273,097 BENIGN flows dwarf every attack type, the largest of which are DoS Hulk (231,073), PortScan (158,930) and DDoS (128,027), tailing off to rarities such as Infiltration (36), SQL Injection (21) and Heartbleed (11). Following the brief, the main model focuses on the **Wednesday file**, which after loading holds 692,703 rows and, once the ±inf and NaN values are dropped (these arise from features like *Flow Bytes/s* when duration is zero), 691,406 rows over 78 numeric features. Wednesday contains 439,683 BENIGN and 251,723 attack flows; I take a balanced subsample of 20,000 per class (40,000 rows total) and standardise the features with a z-score scaler.

### Model description and justification

Because these are tabular flow statistics with no spatial grid, a convolutional DCGAN makes no sense here; the right model is a plain multilayer-perceptron GAN that emits an entire 78-dimensional feature vector at once. The generator is three BatchNorm + LeakyReLU hidden blocks (256 units) with a *linear* output, deliberately not `tanh`, because the features are z-scored rather than bounded to [-1, 1], so a linear head fits them better. The discriminator is a 256-unit LeakyReLU MLP with 0.3 dropout. Training keeps the same discipline as the image parts, Adam (lr = 2e-4, `betas=(0.5, 0.999)`), `BCEWithLogitsLoss`, and one-sided label smoothing, so the three GANs differ only where they should.

### Training and loss curves

![CICIDS2017 tabular GAN loss curves; the discriminator loss drifts down from ~1.35 to ~1.02 while the generator rises to ~1.35, a slow, stable separation rather than a collapse.](./images/loss_curves.png)

Over 60 epochs the discriminator loss falls gradually from 1.3548 to 1.0248 and the generator rises from 0.8257 to 1.3455. There is no sudden divergence or mode-collapse spike; the curves separate slowly, which is the benign case for a tabular GAN.

### PCA/t-SNE comparison and per-feature alignment

![PCA of real (blue) versus generated (red) flow vectors; the synthetic cloud sits on the main body of the real data but is more concentrated and does not reach the tails.](./images/pca_real_vs_fake.png)

Projected with PCA (Jolliffe, 2002), the generated cloud sits squarely on top of the main body of real points but is noticeably more concentrated, it captures where most traffic lives without reaching into the tails. A t-SNE projection (van der Maaten and Hinton, 2008) tells the same story. The per-feature alignment check quantifies this: across all 78 features the mean absolute error on feature *means* is a good **0.0723** (in standardised units), but on *standard deviations* it is a weaker **0.3179**. So the GAN reproduces average feature values much more faithfully than it reproduces their spread. The worst-aligned features are almost all timing and flag columns, `FIN Flag Count` (mean error 0.326), the backward inter-arrival-time family (`Bwd IAT Max/Std/Total/Mean`), `Destination Port`, `Fwd IAT Mean`, `URG Flag Count` and `Fwd PSH Flags`. For the near-binary flags the generated standard deviation almost vanishes (`URG Flag Count` fake std 0.030, `Fwd PSH Flags` fake std 0.024, against real values near 0.9-1.0), meaning the generator effectively pins those features to a constant instead of reproducing their on/off behaviour, which is partial mode collapse on the discrete features.

### Reflection and reliability in a security context

The hallucination problem is arguably worse here than for the OCT scans. A generated flow only has to satisfy the discriminator's notion of "looks like real traffic", a far weaker bar than "matches a real attack technique". Nothing forces the generator to respect the actual mechanics of a DoS attack, its timing, packet structure and flag sequences, so it can be close in aggregate while getting operationally important detail wrong. If such data augmented an IDS training set, two failure modes follow: a fabricated "attack" matching no real signature could teach the classifier to expect the wrong thing and miss genuine attacks, while a fabricated "benign" flow drifting toward attack-like statistics could raise false positives. Crucially, unlike the OCT case there is no quick visual sanity check for a row of 78 numbers, so the burden falls entirely on quantitative checks like the alignment gap above, and even those can look broadly fine while hiding the flag-count collapse. One labelling clarification: the Wednesday file actually contains DoS variants (Hulk, GoldenEye, slowloris, Slowhttptest, Heartbleed), **not DDoS**, despite the brief's wording; genuine DDoS lives in the Friday file and is included in the extension.

### Extension: full multi-day dataset and generalisation

![PCA of the all-days data coloured by attack type (left) versus real-vs-generated (right); the generator covers the dense central mass but under-represents the smaller outlying attack clusters.](./images/pca_full_dataset_by_attack_type.png)

Combining all eight days (with per-file capping to stop BENIGN dominating) yields 125,310 rows spanning every attack family. Trained on this, the generator's PCA cloud covers the dense middle where BENIGN and the high-volume attacks (DoS Hulk, true DDoS) sit, but clearly under-represents the smaller outlying clusters, expected, since a single unconditional generator has no incentive to spend capacity on classes it barely sees. The quantitative result is the interesting part: adding diversity made alignment **worse**, not better. The mean |mean error| rose from 0.0723 (Wednesday) to **0.1300** (all days) and the mean |std error| from 0.3179 to **0.4670**, with the worst features now dominated by the timing family (`Flow Duration`, `Fwd IAT Total`, the `Idle` and `Flow IAT` statistics), whose spread the generator badly under-produces. In other words, forcing one generator to cover many modes cost more than the extra data gained, so a plain unconditional GAN does not generalise cleanly to the many-attack setting. The obvious next step is to give rare attacks their own capacity via a class-conditional GAN (as in Part 2.1) or a separate model per attack family.

---

## 5. Part 2.3: Creative AI - QuickDraw 'Birthday Cake' (DCGAN, TensorFlow/Keras)

### Dataset exploration

The QuickDraw dataset (Ha and Eck, 2018; Google Creative Lab) provides millions of 28x28 grayscale doodles as "numpy bitmap" files, high-contrast line art on a black background. The birthday-cake category holds 144,982 sketches. A sample grid shows the recurring elements the model must learn: a rounded cake body, candles on top, and often a plate line underneath.

### Model description and justification

A doodle is defined by *where the strokes go*, i.e. local spatial structure, so the same DCGAN builders from Part 2.1 apply directly (reusing them across such different domains is itself a useful check that the architecture generalises). The only change is a larger 128-dimensional latent vector, reflecting the greater shape variety of freehand sketches. Training is again Adam (lr = 2e-4, `beta1` = 0.5) with label smoothing on a 30,000-image subset for 50 epochs.

### Training and loss curves

The loss behaviour is typical DCGAN: the discriminator loss hovers low (mostly 0.34-0.9) while the generator oscillates through the 1.3-6.1 range as the two trade advantage, with no terminal collapse. Epoch snapshots at 1, 13, 26, 38 and 50 show the cake outline and candles appearing early and then sharpening over training.

### Real vs fake comparison and FID

![QuickDraw real birthday cakes (left) versus DCGAN-generated sketches (right); the fakes are readable as cakes with candles, with occasional broken strokes and background speckle.](./images/quickdraw_real_vs_fake.png)

The generated cakes are easy to read: a body, candles on top, often a plate line. The grid varies across samples (no mode collapse), and the flaws are the usual line-art ones, broken or doubled strokes and slight background speckle, but the overall shape holds. The FID is **42.67**, notably lower (better) than the OCT scans' 79.02. This fits intuition: clean, high-contrast line art on a plain black background sits more comfortably in InceptionV3's feature space than soft grayscale medical texture does.

### Extension: additional categories and complexity

![FID by category (left) versus mean ink density (right) for birthday cake, cat and house; the two rank in the same order, but house is far easier than the other two.](./images/quickdraw_extension_complexity.png)

Training the identical DCGAN on *cat* and *house* lets me compare FID against a cheap complexity proxy, mean ink density (the fraction of inked pixels). The ink-density order is house (0.2368) < cat (0.2655) < cake (0.2708), and the FID order comes out in the *same* direction: house (**22.65**) < cat (**41.60**) < cake (**42.67**). So on this run the busier categories were also the harder ones, a tidy result, but with two caveats. First, cat and cake are almost tied (about one FID point apart), well within FID's run-to-run noise at this sample/epoch budget, so their ordering should not be over-read. Second, the real signal is that house is clearly the easiest by a wide margin, roughly half the FID of the others. A house is largely a square plus a triangular roof, a strongly repeated template, whereas cats and cakes vary far more in pose and layout. The honest conclusion is therefore that **structural variety, more than raw ink coverage, drives modelling difficulty** here.

---

## 6. Conclusion

Across all four experiments the same picture emerges. GANs are very good at reproducing the *dominant* structure of a distribution, the sine curve and spiral in Part 1, the retinal band, the central mass of network traffic, and the cake-and-candles motif in Part 2, and reliably avoid mode collapse when stabilised with BatchNorm, tuned Adam betas, and label smoothing. Their consistent weakness is the *tails and fine detail*: thinning coverage at the spiral and sine peaks, blur in the OCT scans (FID 79.02), the collapsed variance of binary flag features in the tabular case (std error 0.3179 rising to 0.4670 on the full dataset), and broken strokes in the sketches. Two findings stand out as genuinely informative rather than confirmatory: the multi-day cybersecurity GAN *generalised worse* than the single-day model, showing that one unconditional generator cannot cover many modes at once; and in the creative task structural variety predicted difficulty better than ink density. The recurring "looks real vs is true" hazard means none of this synthetic data should be trusted downstream without validation.

Several alternative approaches would address these limitations in future work. Replacing the standard loss with a Wasserstein objective and gradient penalty would give smoother gradients and better tail coverage. Class-conditional or per-family models would fix the multi-attack generalisation failure by giving rare classes dedicated capacity. For the tabular case, architectures designed for mixed discrete/continuous columns (such as CTGAN-style conditional generators) would directly target the flag-feature collapse. Finally, using a domain-appropriate feature extractor for FID, rather than ImageNet-trained InceptionV3, would make the medical-image metric more meaningful. Longer training and larger samples would tighten every FID reported here.

---

## 7. References

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y. (2014) 'Generative adversarial nets', *Advances in Neural Information Processing Systems*, 27, pp. 2672-2680.

Ha, D. and Eck, D. (2018) 'A neural representation of sketch drawings', *International Conference on Learning Representations (ICLR)*. (The Quick, Draw! Dataset, Google Creative Lab: https://github.com/googlecreativelab/quickdraw-dataset)

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B. and Hochreiter, S. (2017) 'GANs trained by a two time-scale update rule converge to a local Nash equilibrium', *Advances in Neural Information Processing Systems*, 30, pp. 6626-6637.

Ioffe, S. and Szegedy, C. (2015) 'Batch normalization: accelerating deep network training by reducing internal covariate shift', *International Conference on Machine Learning (ICML)*, pp. 448-456.

Jolliffe, I.T. (2002) *Principal Component Analysis*. 2nd edn. New York: Springer.

Kingma, D.P. and Ba, J. (2015) 'Adam: a method for stochastic optimization', *International Conference on Learning Representations (ICLR)*.

Mirza, M. and Osindero, S. (2014) 'Conditional generative adversarial nets', *arXiv:1411.1784*.

Radford, A., Metz, L. and Chintala, S. (2016) 'Unsupervised representation learning with deep convolutional generative adversarial networks', *International Conference on Learning Representations (ICLR)*.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A. and Chen, X. (2016) 'Improved techniques for training GANs', *Advances in Neural Information Processing Systems*, 29, pp. 2234-2242.

Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A. (2018) 'Toward generating a new intrusion detection dataset and intrusion traffic characterization', *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, pp. 108-116. (CICIDS2017)

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. and Wojna, Z. (2016) 'Rethinking the Inception architecture for computer vision', *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2818-2826.

van der Maaten, L. and Hinton, G. (2008) 'Visualizing data using t-SNE', *Journal of Machine Learning Research*, 9, pp. 2579-2605.

Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H. and Ni, B. (2023) 'MedMNIST v2 - a large-scale lightweight benchmark for 2D and 3D biomedical image classification', *Scientific Data*, 10(1), 41. (OCTMNIST; https://medmnist.com/)
