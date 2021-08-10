# UNet2D-Segmentation-Model-Builder-KERAS
This repository contains 2D Signal (Image) Segmentation Model Builder for UNet and several of its variants developed in KERAS-Tensorflow. The code supports Deep Supervision, Autoencoder mode and other options explained in the DEMO.

# Models supported:
1. **UNet** [Reference: https://arxiv.org/abs/1505.04597]
2. **UNet Ensembled (UNetE)** [Reference: https://www.arxiv-vanity.com/papers/1912.05074/]
3. **UNet+ (UNetP)** [Reference: https://www.arxiv-vanity.com/papers/1912.05074/]
3. **UNet++ (UNetPP)** [Reference: https://arxiv.org/abs/1807.10165]
5. **MultiResUNet** [Reference: https://arxiv.org/abs/1902.04049]

# UNet Variants' Architectures
![UNet Architectures](https://github.com/Sakib1263/UNet2D-Segmentation-Model-Builder-KERAS/blob/main/Documents/Images/UNet.jpg "UNet Models") 

Version 2 of all the networks (e.g., UNet_v2) uses Transposed Convolution instead of UpSampling in the Decoder section. The models can accept image of various sizes and shapes, different model depth, filter or kernel number, kernel size, number of channels, etc. The models can optionally perform Deep Supervision on the inputs or can be used as an AutoEncoder to extract latent features which can be used for feature ranking, regression, etc. The models are flexible enough to be used for regression, or binary or multiclass segmentation.
