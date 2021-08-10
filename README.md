# UNet2D-Segmentation-Model-Builder-KERAS
This repository contains 2D Signal (Image) Segmentation Model Builder for UNet and several of its variants developed in KERAS-Tensorflow. The code supports Deep Supervision, Autoencoder mode and other options explained in the DEMO.

# Models supported:
1. **UNet** [Reference: https://arxiv.org/abs/1505.04597]
2. **UNet Ensembled (UNetE)** [Reference: https://www.arxiv-vanity.com/papers/1912.05074/]
3. **UNet+ (UNetP)** [Reference: https://www.arxiv-vanity.com/papers/1912.05074/]
3. **UNet++ (UNetPP)** [Reference: https://arxiv.org/abs/1807.10165]
5. **MultiResUNet** [Reference: https://arxiv.org/abs/1902.04049]

Version 2 of all the networks (e.g., UNet_v2) uses Transposed Convolution instead of UpSampling in the Decoder section.
