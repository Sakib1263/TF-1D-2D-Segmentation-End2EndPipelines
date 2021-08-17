# UNet-Segmentation-Model-Builder-KERAS
This repository contains 1D and 2D Signal Segmentation Model Builder for UNet and several of its variants developed in KERAS-Tensorflow. The code supports Deep Supervision, Autoencoder mode and other options explained in the DEMO. The segmentation models can be used for binary or multiclass segmentation, or for regression tasks.  

# Models supported
1. **UNet** [Reference: https://arxiv.org/abs/1505.04597]
2. **UNet Ensembled (UNetE)** [Reference: https://www.arxiv-vanity.com/papers/1912.05074/]
3. **UNet+ (UNetP)** [Reference: https://www.arxiv-vanity.com/papers/1912.05074/]
3. **UNet++ (UNetPP)** [Reference: https://arxiv.org/abs/1807.10165]
5. **MultiResUNet** [Reference: https://arxiv.org/abs/1902.04049]

# UNet Variants' Architectures
![UNet Architectures](https://github.com/Sakib1263/UNet2D-Segmentation-Model-Builder-KERAS/blob/main/Documents/Images/UNet.jpg "UNet Models") 

Version 2 of all the networks (e.g., UNet_v2) uses Transposed Convolution instead of UpSampling in the Decoder section. The models can accept image of various sizes and shapes, different model depth, filter or kernel number, kernel size, number of channels, etc. The models can optionally perform Deep Supervision on the inputs or can be used as an AutoEncoder to extract latent features which can be used for feature ranking, regression, etc. The models are flexible enough to be used for regression, or binary or multiclass segmentation.

MultiResUNet Architecture
![MultiResUNet Architecture](https://github.com/Sakib1263/UNet-2D-Segmentation-AutoEncoder-Model-Builder-KERAS/blob/main/Documents/Images/MultiResUNet.png "MultiResUNet Model")  

# Supported Features
The speciality about this model is its flexibility, such as:
1. The user can choose any of the 5 available UNet variants for either 1D or 2D Segmentation tasks.
2. The models can be used for Binary or Multi-Class Classification, or Regression type Segmentation tasks.
3. The models allow Deep Supervision with flexibility during Segmentation.
4. The segmentation models can also be used as Autoencoders for Feature Extraction.
5. Number of input kernel/filter, commonly known as Width of the model can be varied.
6. Number of classes for Classification tasks and number of extracted features for Regression tasks can be varied.
7. Number of Channels in the Input Dataset can be varied.  

Details of the process are available in the DEMO provided in the codes section. The datasets used in the DEMO as also available in the 'Documents' folder.  
