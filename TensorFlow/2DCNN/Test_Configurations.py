import configparser

# CREATE OBJECT
config_file = configparser.ConfigParser()

# ADD NEW SECTION AND SETTINGS
config_file["TEST"] = {
        # General Configurations
        "fold_num": 1,
        "test_image_dir": "Data/Test/Fold_1/Images",
        "test_mask_dir": "Data/Test/Fold_1/Masks",
        "imlength": 224,  # Size of the Images being Trained, they will be resized in this shape: [imsize, imsize]
        "imwidth": 224,  # Size of the Images being Trained, they will be resized in this shape: [imsize, imsize]
        "color_mode": "rgb",  # Color Mode of the images
        "num_channel": 3,  # Number of Input Channels in the Model [rgb:3, rgba:4, grayscale:1]
        "class_number": 3,  # Number of Output Classes in the Task
        # Model Configurations
        "encoder_name": "ResNet50", # Select an Encoder from a pool of ImageNet trained Models available from TensorFlow, default: ResNet50
        "decoder_name": "UNet",  # Select a Model from the list to train from scratch, ResNet50 is kept as default
        # Test Configurations
        "batch_size": 8,  # Batch Size of the Images bein loaded for training
        "normalizing_factor": 255.0,  # 255.0 for images with pixel values varying between 0 to 255. If it is between 0 to 1, change it to 1
        "start_fold": 1,  # Fold to Start Training, can be varied from 1 to the last fold
        "end_fold": 2,  # Fold to End Training, can be any value from the start_fold [Number of Folds + 1]
        "num_iter": 1,  # Number of Folds completed training
        "seed": 42  # SEED required for randomly split Validation set from the Training set, not used when "validation_portion"= 0.0
        }

# SAVE CONFIG FILE
with open(r"Test Configurations.ini", 'w') as configfileObj:
    config_file.write(configfileObj)
    configfileObj.flush()
    configfileObj.close()

print("Config file 'Test Configurations.ini' created")
