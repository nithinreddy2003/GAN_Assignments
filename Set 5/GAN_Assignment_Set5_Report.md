# 7COM1079 Coursework 2: generative adversarial networks in medicine, cybersecurity and creative sketching

Student Name:

Student Id:

Github URL: https://github.com/nithinreddy2003/GAN_Assignments

## 1. Introduction

A generative adversarial network (GAN) trains a generator against a discriminator; at the equilibrium where the discriminator can no longer separate real from synthetic samples, the generator has effectively learned the data distribution (Goodfellow et al., 2014). Part 1 builds this machinery from scratch on controllable 2D data so the mechanics stay visible, and Part 2 applies GANs to three contrasting problems in medical imaging, network security and creative sketching, each stressing a different part of generative modelling. PyTorch was used for the 2D and tabular models and TensorFlow/Keras for the two image problems. All seeds are fixed (5050 for PyTorch and NumPy, 6060 for TensorFlow) so the numbers and figures reproduce, and evaluation is matched to the data type: the Frechet Inception Distance for the image and sketch models, and low-dimensional projections with per-feature alignment gaps for the tabular model.

## 2. Part 1: building and understanding GANs from scratch

Part 1 is implemented directly in PyTorch. The generator is an MLP regularised with layer normalisation (Ba, Kiros and Hinton, 2016) and the critic is constrained with spectral normalisation (Miyato et al., 2018), which was chosen because on tiny 2D problems an unconstrained critic wins almost immediately, and spectral normalisation is the cheapest principled way to keep the two networks balanced. Both optimise the non-saturating loss with one-sided label smoothing (Salimans et al., 2016) using Adam, and the latent vector is 24-dimensional.

### 2.1 Task 1: reproducing the sine-wave GAN

The first task reproduces the tutorial sine-wave GAN on points drawn from a sine curve, trained for 6,000 steps. Figure 1 shows the generated points falling along the curve, so the generator has captured a one-dimensional manifold embedded in 2D. The critic loss settles near 1.38 and the generator near 0.80 and both hold there (final critic 1.3765, generator 0.8009). A critic near 2ln2 = 1.386 is the value reached when it can no longer separate the two distributions, so the flat curves signal a healthy equilibrium and not a stalled optimiser. One-sided label smoothing helps here: capping the real target at 0.9 stops the critic pushing its logits to extremes, which keeps a usable gradient flowing to the generator.

![Figure 1](Set%205%20Result%20Images/dc9c9acab72bc0499c58f19d65d1c34275a2c18c.png)
*Figure 1. Sine-wave GAN: real versus generated 2D points.*

### 2.2 Task 2: a new 2D distribution, a mixture of Gaussians

For the new distribution I chose a mixture of Gaussians laid out on a 3x3 grid of nine modes, which is the classic stress test for mode collapse because the modes are disconnected. Figure 2 shows the generator reaching all nine modes and not collapsing onto one or two, which is the success criterion here. The losses hold at the sine levels (critic about 1.35, generator about 0.82, final critic 1.3458), so the setup stays stable on a multi-modal target. The clusters are not perfectly balanced, and the noticeable scatter between them is hard to avoid, because a continuous generator has to pass through the empty space separating disconnected blobs. Reaching every mode is the meaningful result, and the spectral-normalised critic is what prevents collapse, since a Lipschitz-bounded critic cannot hand the generator the sharp shortcut that triggers it.

![Figure 2](Set%205%20Result%20Images/0756a3e8f06ca44243647aeb7bd3aa4b880e04c2.png)
*Figure 2. Mixture-of-Gaussians GAN: real versus generated samples for the 3x3 grid of nine modes.*

### 2.3 Task 3: modifying the architecture and comparing samples

The third task holds the target fixed and changes only the generator: a two-layer LeakyReLU baseline against a three-layer GELU version on an identical budget. The losses are almost identical (both critic about 1.36 to 1.37, generator about 0.80, modified final critic 1.3671, generator 0.8025). Figure 3 shows that the samples are also broadly similar: both variants reach all nine modes with comparable inter-mode scatter, and the deeper GELU generator is at most marginally cleaner, with slightly less diffuse background noise around the lattice. On a target this simple the extra depth and the change of activation make little visible difference, so here the loss curves and the samples agree that the change is minor; a harder target would be needed to separate the two architectures clearly.

![Figure 3](Set%205%20Result%20Images/9b969a549f27e7913d61f11df3aadeb38639bfb8.png)
*Figure 3. Architecture comparison: baseline (two-layer LeakyReLU) versus modified (three-layer GELU) generator.*

## 3. Part 2: real-world GAN applications

### 3.1 Part 2.1: OCT retinal images with OCTMNIST

This application trains a deep convolutional GAN (DCGAN) on OCTMNIST (Yang et al., 2023), which provides 97,477 grayscale scans across four diagnostic categories. A DCGAN (Radford, Metz and Chintala, 2015) is the natural choice for images because its convolutions learn spatial structure directly. Each scan is resized to 32x32 and normalised to the [-1, 1] range expected by the tanh-terminated generator. Figure 4 shows the four categories are far from evenly represented, and that skew returns later, because the categories that dominate the training counts are the ones the conditional model renders most convincingly. The generator projects a 128-dimensional latent vector to a 4x4 feature map and doubles it three times up to 32x32 with strided transposed convolutions, the standard DCGAN upsampling layers, though these can leave faint checkerboard artefacts (Odena, Dumoulin and Olah, 2016); the critic is a strided-convolution stack with batch normalisation, trained for 40 epochs.

![Figure 4](Set%205%20Result%20Images/64512e296c65a309ac76262d8cb51e6052966d51.png)
*Figure 4. Number of OCTMNIST training scans in each diagnostic category.*

Figure 5 shows the two losses oscillating within a bounded band across the 40 epochs instead of collapsing, the behaviour expected of a stable DCGAN. Figure 6 compares real and generated scans and is backed by a Frechet Inception Distance (Heusel et al., 2017) of 17.89, the lowest of the image models here. The generated scans capture the essential anatomy, a bright curved retinal band on a dark background, and vary in the band's shape and position, so the generator is not memorising one template. The main weakness is fine texture: the fakes are a little too smooth, which is consistent with a good but non-zero FID and is the expected signature of a loss that rewards global plausibility over pixel detail. In a medical setting this smoothing is not harmless, since the fine speckle can itself carry diagnostic signal.

![Figure 5](Set%205%20Result%20Images/1e6acc6bf10f080b1e1fb280d7a982f00317942e.png)
*Figure 5. OCT DCGAN training losses over 40 epochs.*

![Figure 6](Set%205%20Result%20Images/62b2333a47906e2ac7dd64bcd172838cee07019f.png)
*Figure 6. OCT DCGAN: real (left) versus generated (right) retinal scans, FID 17.89.*

For extra credit the DCGAN was extended into a conditional GAN (Mirza and Osindero, 2014) so a class can be requested on demand. Figure 7 confirms conditioning works, because each row differs and the most frequent classes come out cleanest. This model was the hardest to stabilise. A first attempt that fed the label to the critic as a full 32x32 channel gave it an overwhelming cue, so its loss collapsed while the generator ran away. The fix combined late-fusion conditioning, where the label embedding joins only at the critic's final dense layer, with instance noise (Arjovsky and Bottou, 2017), a two-time-scale rule using a slower critic (Heusel et al., 2017), and two generator steps per critic step. The losses then trade off gently, the critic moving from about 1.29 to 1.05 and the generator from about 0.94 to 1.42 over 40 epochs. The broader lesson is that conditioning is mostly about denying the critic an easy signal: the label that guides the generator becomes a liability once the critic can read it too directly.

![Figure 7](Set%205%20Result%20Images/192caf2dc23c7a1a2bf580e1d1591896c774ce8f.png)
*Figure 7. Conditional GAN: one OCT diagnostic class per row.*

On the duck test, where an image that looks and quacks like a duck is called one, a low-FID GAN is only trained to pass that surface test. The critic never learns what is medically correct, so a convincing scan can still contain structure that no real patient produced. This is why synthetic scans are defensible for augmentation or teaching but not for diagnosis: a low FID says the population statistics align, not that any single image is genuine.

### 3.2 Part 2.2: cybersecurity, synthetic traffic with CICIDS 2017

This task generates synthetic traffic feature vectors from CICIDS 2017 (Sharafaldin, Lashkari and Ghorbani, 2018). Because each flow is numeric and not an image, a convolutional model is inappropriate, so I use a fully-connected GAN written in PyTorch with a 56-dimensional noise vector. The eight daily CSVs were merged into 2,830,743 flows, and 78 numeric features were kept after removing 4,376 infinite and 1,358 missing cells. Figure 8 uses a log scale because BENIGN traffic (2,273,097 flows) dwarfs every attack class, with rare families such as Heartbleed and Infiltration numbering only tens of flows, an imbalance that limits what the generator can learn. Following the brief, the GAN trained on the Wednesday file (BENIGN versus DoS), scaled with a RobustScaler and clipped to plus or minus 5 to tame outliers, on 20,000 flows.

![Figure 8](Set%205%20Result%20Images/5cd6e20f0519869b2289b76954766557f1b73c9f.png)
*Figure 8. CICIDS 2017 class distribution across all days (log scale).*

Figure 9 compares the real and synthetic flows with PCA; a t-SNE projection (van der Maaten and Hinton, 2008) examined alongside it tells the same story. The generated flows land within the body of the real data instead of forming a separate island, so the generator has located the correct part of the feature space, though the synthetic points are packed a little more tightly than the real ones, so their variability is understated. The alignment metrics quantify this, with a mean per-feature difference of 0.2435 on the means and 0.2395 on the standard deviations, a moderate fit that captures location and scale but not the heavy tails. The tails matter operationally, because intrusion detectors often key on rare, extreme feature values, so a generator that compresses the spread risks producing traffic that looks normal exactly where the real signal lives. Retraining on all eight days widens the gaps to 0.2955 on the means and 0.5277 on the standard deviations, the standard-deviation gap roughly doubling: a single unconditional generator tracks the high-volume families (BENIGN, DoS, DDoS and PortScan) well but under-serves the long tail, so it is best read as a model of typical traffic. The duck point sharpens here, because a row of 78 numbers cannot be eyeballed for wrongness the way an image can, so the moment gaps, and the standard-deviation gap especially, are the real check.

![Figure 9](Set%205%20Result%20Images/61dd21eac59b02bdc817d283bf42779d8db56dbc.png)
*Figure 9. PCA of real (BENIGN and DoS) versus synthetic flow vectors.*

### 3.3 Part 2.3: creative AI, QuickDraw sketches

The final application trains a DCGAN on the QuickDraw 'birthday cake' category (Ha and Eck, 2017), 144,982 bitmaps. To show a second standard construction, this generator uses transposed convolutions with a 64-dimensional latent vector and its critic uses dropout in place of batch normalisation. Figure 10 tracks the visual outputs across epochs: the early grids are noisy blobs that resolve into recognisable cake outlines with candle-like marks, which shows the generator is learning the cake shape and that training stays stable to the final epoch.

![Figure 10](Set%205%20Result%20Images/4bc80b4de62ba0fdd04811312182b559e3be4a93.png)
*Figure 10. Birthday-cake samples captured at successive training epochs.*

Figure 11 shows most generated sketches are recognisable cakes with strokes that hold together, reflected in a cake FID of 25.56, the best sketch score. Occasional stroke artefacts appear, but the overall form is preserved. Training the same model on two more categories gives Figure 12, which is the most interesting result: the FIDs are cake 25.56, cat 28.51 and house 41.98, while ink density is 0.198, 0.1985 and 0.173. House is the sparsest category yet scores the worst FID, the opposite of a simple busier-is-harder hypothesis. Ink coverage is therefore a poor difficulty predictor: a house is a few long straight lines meeting at clean right angles, and reproducing that geometry precisely is harder than the loose, forgiving strokes of a cat or cake. This is really a point about inductive bias, because convolutional generators reproduce local texture cheaply but struggle with long-range geometric constraints, so a drawing that is simple for a person is not necessarily easy for the model.

![Figure 11](Set%205%20Result%20Images/8060c0225797da399e8b2a5a3ae7912fce574541.png)
*Figure 11. QuickDraw 'birthday cake': real (left) versus generated (right) sketches.*

![Figure 12](Set%205%20Result%20Images/551b0745b385cfb98e2428a6614cdcb7d8c1d9e4.png)
*Figure 12. QuickDraw extension: FID and ink density by category.*

## 4. Conclusion and future work

Across the four problems the models trained stably and matched the real data on the metrics used: the Part 1 GANs reach every mode, the OCT DCGAN reaches an FID of 17.89 with a working conditional extension, the CICIDS generator places synthetic flows inside the real distribution with moderate moment gaps, and the QuickDraw DCGAN reaches an FID of 25.56 on cakes. The analysis also surfaced clear limits: saturated losses can hide real quality differences (Task 3); conditional GANs are highly sensitive to how the label is injected (Part 2.1); unconditional generators under-represent rare classes on imbalanced data (Part 2.2); and drawing difficulty is driven by geometric regularity more than ink volume (Part 2.3). Future work would use a Wasserstein GAN with a gradient penalty to tighten the CICIDS gaps, class-conditional oversampling to serve rare attack families, and a train-on-synthetic, test-on-real check to measure the downstream utility of the synthetic data beyond its distributional match. For the medical task a domain-specific perceptual metric would be more trustworthy than the ImageNet-based FID used here, and a small classifier-based check on the conditional samples would confirm that a network trained on real OCT scans assigns generated images to the intended class before any downstream use.

## References

Arjovsky, M. and Bottou, L. (2017) 'Towards principled methods for training generative adversarial networks', International Conference on Learning Representations (ICLR).

Ba, J.L., Kiros, J.R. and Hinton, G.E. (2016) 'Layer normalization', arXiv preprint arXiv:1607.06450.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y. (2014) 'Generative adversarial nets', Advances in Neural Information Processing Systems, 27, pp. 2672-2680.

Ha, D. and Eck, D. (2017) 'A neural representation of sketch drawings', arXiv preprint arXiv:1704.03477.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B. and Hochreiter, S. (2017) 'GANs trained by a two time-scale update rule converge to a local Nash equilibrium', Advances in Neural Information Processing Systems, 30, pp. 6626-6637.

Mirza, M. and Osindero, S. (2014) 'Conditional generative adversarial nets', arXiv preprint arXiv:1411.1784.

Miyato, T., Kataoka, T., Koyama, M. and Yoshida, Y. (2018) 'Spectral normalization for generative adversarial networks', International Conference on Learning Representations (ICLR).

Odena, A., Dumoulin, V. and Olah, C. (2016) 'Deconvolution and checkerboard artifacts', Distill, 1(10), e3.

Radford, A., Metz, L. and Chintala, S. (2015) 'Unsupervised representation learning with deep convolutional generative adversarial networks', arXiv preprint arXiv:1511.06434.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A. and Chen, X. (2016) 'Improved techniques for training GANs', Advances in Neural Information Processing Systems, 29, pp. 2234-2242.

Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A. (2018) 'Toward generating a new intrusion detection dataset and intrusion traffic characterization', Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP), pp. 108-116.

van der Maaten, L. and Hinton, G. (2008) 'Visualizing data using t-SNE', Journal of Machine Learning Research, 9, pp. 2579-2605.

Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H. and Ni, B. (2023) 'MedMNIST v2 - a large-scale lightweight benchmark for 2D and 3D biomedical image classification', Scientific Data, 10(1), 41.
