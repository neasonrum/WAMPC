# V3 ego12

This is a standalone ablation of full_diffusionv3 with the same frozen TF++,
PDM-Lite data split, pose-aligned three-frame BEV, behavior prediction, 10 spatial
path points, 6 temporal displacements, and AutoMoT controller. Its model is trained
from scratch. Each denoising branch reads the BEV and one 12 × 256 token sequence:
the 11 original TF++ joined path/target-speed features and the TF++ status token
encoding current speed and six-way route command. The 64 coarse scene tokens
remain internal to TF++ when it forms the joined queries; the denoiser has no
separate memory attention, including no direct memory pooling in its behavior
head. The explicit goal token is replaced to keep 12 slots; target XY remains in
the behavior/gate inputs. This tests the whole conditioning change.

Ten ranks process four batches of seven and six of six at each update. The
original batch size 64, epoch permutation, learning rate, loss weights, and
10-epoch schedule are retained. Use epoch_10.pt for 25-step DDIM after training.
For a like-for-like comparison run the unmodified V3 with 25 steps as well.
