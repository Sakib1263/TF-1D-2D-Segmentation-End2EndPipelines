import configparser

# CREATE OBJECT
config_file = configparser.ConfigParser()

# ADD NEW SECTION AND SETTINGS
config_file["TRAIN"] = {
        # General Configurations
        "train_image_dir": "Data/Train/Fold_1/Images",
        "train_mask_dir": "Data/Train/Fold_1/Masks",
        "val_image_dir": "Data/Val/Fold_1/Images",
        "val_mask_dir": "Data/Val/Fold_1/Masks",
        "data_loading_mode": "TF_DataLoader",  # "TF_DataLoader" or "Legacy"
        "independent_val_set": False,
        "imlength": 512,  # Size of the Images being Trained, they will be resized in this shape: [imsize, imsize]
        "imwidth": 512,  # Size of the Images being Trained, they will be resized in this shape: [imsize, imsize]
        "color_mode": "rgb",  # Color Mode of the images
        "num_channel": 3,  # Number of Input Channels in the Model [rgb:3, rgba:4, grayscale:1]
        # Model Configurations
        "encoder_mode": "pretrained_encoder",  # Transfer Learning: "pretrained_encoder" | Train from scratch: "from_scratch"
        # Encoder Configurations: Configurations specific to the "pretrained_encoder" mode [None if encoder_mode: "from_scratch"]
        "encoder_name": "EfficientNetB0", # Select an Encoder from a pool of ImageNet trained Models available from TensorFlow, default: ResNet50
        "encoder_trainable": False,  # Fine Tuning ON/OFF [True/False] | Start with OFF, Fine Tune later in the 2nd stage, which is optional
        # Decoder Configurations: General Configurations for both encoder modes
        "decoder_name": "UNet3P",  # Select a Model from the list to train from scratch, ResNet50 is kept as default
        "model_width": 16,  # Width of the Initial Layer, subsequent layers start from here
        "model_depth": 5,  # Depth or Number of Layers in the Model [For the "pretrained_encoder" mode: Maximum 5, Minimum 1]
        "D_S": 0,  # Turn on Deep Supervision, does not work with TF DataLoader [Default: 1]
        "A_E": 0,  # Turn on AutoEncoder Mode for Feature Extraction [Default: 0]
        "A_G": 0,  # Turn on for Guided Attention [Default: 0]
        "LSTM": 0,  # Turn on for LSTM [Default: 0]
        "num_dense_loop": 2,  # Number of Densely Connected Residual Blocks in the BottleNeck Layer
        "problem_type": 'Regression',  # Problem Type: Classification (Multi-Class) or Regression (Single Class)
        "output_nums": 1,  # Number of Output Classes
        "is_transconv": True,  # Number of Output Classes
        "final_activation": "linear",  # Activation Function for the Final Layer: "Linear", "Sigmoid", "Softmax", etc. depending on the problem type
        "feature_number": 1024,  # Number of Features to be Extracted [Only required for the AutoEncoder Mode]
        "alpha": 1,  # Required for MultiResUNet models
        "ds_type": "UNet",  # "UNet" or "UNetPP"; only required when Deep Supervision (D_S) is on
        # Training Configurations
        "batch_size": 8,  # Batch Size of the Images bein loaded for training
        "validation_portion": 0.2,  # 0 to 1 [Default: 0; '0' means validation set is independent, otherwise it will created randomly while training based on "validation_portion"]
        "learning_rate": 0.0005,  # During Fine-Tuning the network, the Learning Rate should be very low (e.g., 1e-5), otherwise more (e.g., 1e-4, 1e-3)
        "normalizing_factor": 255.0,  # 255.0 for images with pixel values varying between 0 to 255. If it is between 0 to 1, change it to 1
        "start_fold": 1,  # Fold to Start Training, can be varied from 1 to the last fold
        "end_fold": 2,  # Fold to End Training, can be any value from the start_fold [Number of Folds + 1]
        "num_iter": 1,  # Number of Folds completed training
        "monitor": "val_loss",  # Monitoring parameter during training
        "patience": 15,  # Number of epochs to wait before training to stop
        "patience_mode": "min",  # patience mode: 'min', 'max' or 'auto'
        "epochs": 100,  # Number of epochs for training
        "loss_function_name": "MeanSquaredError", # Loss Functions
        "optimizer_function_name": "Adam",  # Optimization Algorithm
        "metric_list": "MeanAbsoluteError",  # Metric(s) being monitored
        "seed": 42  # SEED required for randomly split Validation set from the Training set, not used when "validation_portion"= 0.0
        }

# SAVE CONFIG FILE
with open(r"Train Configurations.ini", 'w') as configfileObj:
    config_file.write(configfileObj)
    configfileObj.flush()
    configfileObj.close()

print("Config file 'Train Configurations.ini' created")
