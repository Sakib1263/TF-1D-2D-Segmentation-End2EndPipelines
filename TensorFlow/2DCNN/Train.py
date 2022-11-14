import os
import gc
import configparser
import numpy as np
import tensorflow as tf
from TF_Losses import TFLosses
from TF_Metrics import TFMetrics
from TF_Optimizers import TFOptimizers
from TF_2D_Segmentation_Models_with_Pretrained_Encoders import UNetWithPretrainedEncoder
from Helper_Functions import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

"""[CONFIGURATIONS]"""
'''General Configurations'''
config_file = configparser.ConfigParser()
config_file.read('Train Configurations.ini')
train_dir = config_file["TRAIN"]["train_dir"]
val_dir = config_file["TRAIN"]["val_dir"]
data_loading_mode = config_file["TRAIN"]["data_loading_mode"]
independent_val_set = config_file["TRAIN"].getboolean("independent_val_set")
validation_portion = float(config_file["TRAIN"]["validation_portion"])
imlength = int(config_file["TRAIN"]["imlength"])
imwidth = int(config_file["TRAIN"]["imwidth"])
image_color_mode = config_file["TRAIN"]["image_color_mode"]
mask_color_mode = config_file["TRAIN"]["mask_color_mode"]
num_channel = int(config_file["TRAIN"]["num_channel"])
normalizing_factor_img = float(config_file["TRAIN"]["normalizing_factor_img"])
normalizing_factor_msk = float(config_file["TRAIN"]["normalizing_factor_msk"])
'''Model Configurations'''
encoder_mode = config_file["TRAIN"]["encoder_mode"]
'''Encoder Configurations'''
encoder_name = config_file["TRAIN"]["encoder_name"]
encoder_trainable = config_file["TRAIN"].getboolean("encoder_trainable")
'''Decoder Configurations'''
decoder_name = config_file["TRAIN"]["decoder_name"]
model_width = int(config_file["TRAIN"]["model_width"])
model_depth = int(config_file["TRAIN"]["model_depth"])
D_S = int(config_file["TRAIN"]["D_S"])
A_E = int(config_file["TRAIN"]["A_E"])
A_G = int(config_file["TRAIN"]["A_G"])
LSTM = int(config_file["TRAIN"]["LSTM"])
num_dense_loop = int(config_file["TRAIN"]["num_dense_loop"])
problem_type = config_file["TRAIN"]["problem_type"]
output_nums = int(config_file["TRAIN"]["output_nums"])
class_number = int(config_file["TRAIN"]["class_number"])
is_transconv = config_file["TRAIN"].getboolean("is_transconv")
final_activation = config_file["TRAIN"]["final_activation"]
feature_number = int(config_file["TRAIN"]["feature_number"])
ds_type = config_file["TRAIN"]["ds_type"]
alpha = float(config_file["TRAIN"]["alpha"])
'''Training Configurations'''
batch_size = int(config_file["TRAIN"]["batch_size"])
learning_rate = float(config_file["TRAIN"]["learning_rate"])
start_fold = int(config_file["TRAIN"]["start_fold"])
end_fold = int(config_file["TRAIN"]["end_fold"])
num_iter = int(config_file["TRAIN"]["num_iter"])
monitor_param = config_file["TRAIN"]["monitor_param"]
patience_amount = int(config_file["TRAIN"]["patience_amount"])
patience_mode = config_file["TRAIN"]["patience_mode"]
num_epochs = int(config_file["TRAIN"]["num_epochs"])
loss_function_name = config_file["TRAIN"]["loss_function_name"]
optimizer_function_name = config_file["TRAIN"]["optimizer_function_name"]
metric_list = config_file["TRAIN"]["metric_list"]
seed = int(config_file["TRAIN"]["seed"])

config_file = open("Train Configurations.ini", "r")
content = config_file.read()
print("Current Configurations:\n")
print(content)
config_file.flush()
config_file.close()


'''Conditional Configurations, Model Building and Model Compiling'''
if encoder_name == "ResNet50":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).ResNet50()
elif encoder_name == "ResNet50V2":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).ResNet50V2()
elif encoder_name == "ResNet101":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).ResNet101()
elif encoder_name == "ResNet101V2":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).ResNet101V2()
elif encoder_name == "ResNet152":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).ResNet152()
elif encoder_name == "ResNet152V2":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).ResNet152V2()
elif encoder_name == "VGG16":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).VGG16()
elif encoder_name == "VGG19":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).VGG19()
elif encoder_name == "DenseNet121":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).DenseNet121()
elif encoder_name == "DenseNet169":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).DenseNet169()
elif encoder_name == "DenseNet201":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).DenseNet201()
elif encoder_name == "MobileNet":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).MobileNet()
elif encoder_name == "MobileNetV2":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).MobileNetV2()
elif encoder_name == "MobileNetV3Small":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).MobileNetV3Small()
elif encoder_name == "MobileNetV3Large":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).MobileNetV3Large()
elif encoder_name == "InceptionV3":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).InceptionV3()
elif encoder_name == "InceptionResNetV2":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).InceptionResNetV2()
elif encoder_name == "EfficientNetB0":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetB0()
elif encoder_name == "EfficientNetB1":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetB1()
elif encoder_name == "EfficientNetB2":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetB2()
elif encoder_name == "EfficientNetB3":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetB3()
elif encoder_name == "EfficientNetB4":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetB4()
elif encoder_name == "EfficientNetB5":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetB5()
elif encoder_name == "EfficientNetB6":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetB6()
elif encoder_name == "EfficientNetB7":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetB7()
elif encoder_name == "EfficientNetV2B0":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetV2B0()
elif encoder_name == "EfficientNetV2B1":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetV2B1()
elif encoder_name == "EfficientNetV2B2":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetV2B2()
elif encoder_name == "EfficientNetV2B3":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetV2B3()
elif encoder_name == "EfficientNetV2S":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetV2S()
elif encoder_name == "EfficientNetV2M":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetV2M()
elif encoder_name == "EfficientNetV2L":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).EfficientNetV2L()
elif encoder_name == "CheXNet":
    Segmentation_Model = UNetWithPretrainedEncoder(decoder_name, imlength, imwidth, model_width, model_depth, problem_type=problem_type, num_channels = num_channel,
                                                   output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, is_transconv=is_transconv,
                                                   train_mode=encoder_mode, final_activation=final_activation, is_base_model_trainable=encoder_trainable).CheXNet()
else:
    raise ValueError("Please provide a valid/available Encoder Name. Check for spelling mistakes, capital/small letters, etc.")

# Declaring Data Generators for non Deep Supervision mode (D_S = 0)
image_datagen = []
mask_datagen = []
train_image_datagen = []
train_mask_datagen = []
val_image_datagen = []
val_mask_datagen = []
if (D_S == 0) and (data_loading_mode == "TF_DataLoader"):
    if not independent_val_set:
        image_datagen = ImageDataGenerator(rescale=1./normalizing_factor_img, featurewise_center=False, featurewise_std_normalization=False, validation_split=validation_portion)
        mask_datagen = ImageDataGenerator(rescale=1./normalizing_factor_msk, featurewise_center=False, featurewise_std_normalization=False, validation_split=validation_portion)
    elif independent_val_set:
        train_image_datagen = ImageDataGenerator(rescale=1./normalizing_factor_img,featurewise_center=False, featurewise_std_normalization=False, validation_split=0.0)
        train_mask_datagen = ImageDataGenerator(rescale=1./normalizing_factor_msk,featurewise_center=False, featurewise_std_normalization=False, validation_split=0.0)
        val_image_datagen = ImageDataGenerator(rescale=1./normalizing_factor_img,validation_split=0.0)
        val_mask_datagen = ImageDataGenerator(rescale=1./normalizing_factor_msk, validation_split=0.0)
# Main Training Loop
for i in range(start_fold, end_fold):
    # Import Train Dataset using Image Data Generator pipeline
    print(f'\nFold {i}\n')
    train_image_dir = train_dir + f'Fold_{i}/Images'
    train_mask_dir = train_dir + f'Fold_{i}/Masks'
    val_image_dir = val_dir + f'Fold_{i}/Images'
    val_mask_dir = val_dir + f'Fold_{i}/Masks'
    train_ds = []
    val_ds = []
    X_Train = []
    Y_Train = []
    X_Val = []
    Y_Val = []
    Y_Train_Dict = []
    Y_Val_Dict = []
    history = []
    loss_weights = []
    if (D_S == 0) and (data_loading_mode == "TF_DataLoader"):
        if not independent_val_set:
            print('Importing Train and (Non-independent) Validation Data using TF2 DataLoader. In the Train and Validation sets, respectively: ')
            # Create Train Data Generators (N.B.: It has Data Preprocessing Options)
            # Import Train Data in Batches - Condition for Data with/without a Ready Validation Set
            train_image_generator = image_datagen.flow_from_directory('Data/Train/Fold_' + str(i), color_mode=image_color_mode, target_size=(imlength, imwidth), classes=['Images'],
                                                              batch_size=batch_size, class_mode=None, subset="training", shuffle=False, seed=seed)
            train_mask_generator = mask_datagen.flow_from_directory('Data/Train/Fold_' + str(i), color_mode=mask_color_mode, target_size=(imlength, imwidth), classes=['Masks'],
                                                              batch_size=batch_size, class_mode=None, subset="training", shuffle=False, seed=seed)
            train_ds = zip(train_image_generator, train_mask_generator)
            # Import Validation Set - Randomly Splitted from the Training Set
            val_image_generator = image_datagen.flow_from_directory('Data/Train/Fold_' + str(i), color_mode=image_color_mode, target_size=(imlength, imwidth), classes=['Images'],
                                                              batch_size=batch_size, class_mode=None, subset="validation", shuffle=False, seed=seed)
            val_mask_generator = mask_datagen.flow_from_directory('Data/Train/Fold_' + str(i), color_mode=mask_color_mode, target_size=(imlength, imwidth), classes=['Masks'],
                                                              batch_size=batch_size, class_mode=None, subset="validation", shuffle=False, seed=seed)
            val_ds = zip(val_image_generator, val_mask_generator)
        elif independent_val_set:
            print('Importing Train and (Independent) Validation Data using TF2 DataLoader. In the Train and Validation sets, respectively: ')
            # Create Train Data Generators (N.B.: It has Data Preprocessing Options)
            # Import Train Data in Batches - Condition for Data with/without a Ready Validation Set
            train_image_generator = train_image_datagen.flow_from_directory('Data/Train/Fold_' + str(i), color_mode=image_color_mode, target_size=(imlength, imwidth), classes=['Images'],
                                                                      batch_size=batch_size, class_mode=None, subset=None, shuffle=False, seed=seed)
            train_mask_generator = train_mask_datagen.flow_from_directory('Data/Train/Fold_' + str(i), color_mode=mask_color_mode, target_size=(imlength, imwidth), classes=['Masks'],
                                                                    batch_size=batch_size, class_mode=None, subset=None, shuffle=False, seed=seed)
            train_ds = zip(train_image_generator, train_mask_generator)
            # Import Validation Set - Randomly Splitted from the Training Set
            val_image_generator = val_image_datagen.flow_from_directory('Data/Val/Fold_' + str(i), color_mode=image_color_mode, target_size=(imlength, imwidth), classes=['Images'],
                                                                    batch_size=batch_size, class_mode=None, subset=None, shuffle=False, seed=seed)
            val_mask_generator = val_mask_datagen.flow_from_directory('Data/Val/Fold_' + str(i), color_mode=mask_color_mode, target_size=(imlength, imwidth), classes=['Masks'],
                                                                  batch_size=batch_size, class_mode=None, subset=None, shuffle=False, seed=seed)
            val_ds = zip(val_image_generator, val_mask_generator)
    elif (D_S == 1) and (data_loading_mode == "Legacy"):
        # Get loss weights for Deep Supervision
        loss_weights = np.zeros(model_depth)
        for i in range(0, model_depth):
            loss_weights[i] = 1 - (i * 0.1)
        if independent_val_set:
            print('Importing Entire Train Data...')
            X_Train, Y_Train = get_datasets(train_image_dir, train_mask_dir, imlength, imwidth, num_channel, normalizing_factor_img, normalizing_factor_msk, class_number)
            X_Val, Y_Val = get_datasets(val_image_dir, val_mask_dir, imlength, imwidth, num_channel, normalizing_factor_img, normalizing_factor_msk, class_number)
            print('Preparing Data for Deep Supervision...')
            Y_Train_Dict = prepareTrainDict(Y_Train, model_depth, imlength, imwidth, ds_type)
            Y_Val_Dict = prepareTrainDict(Y_Val, model_depth, imlength, imwidth, ds_type)
        elif not independent_val_set:
            print('Importing Entire Train Data...')
            X_Train1, Y_Train1 = get_datasets(train_image_dir, train_mask_dir, imlength, imwidth, num_channel, normalizing_factor_img, normalizing_factor_msk, class_number)
            print('Randomly Splitting Data into Train and Validation Sets...')
            X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train1, Y_Train1, test_size=validation_portion, random_state=42)
            print('Preparing Data for Deep Supervision...')
            Y_Train_Dict = prepareTrainDict(Y_Train, model_depth, imlength, imwidth, ds_type)
            Y_Val_Dict = prepareTrainDict(Y_Val, model_depth, imlength, imwidth, ds_type)
    elif (D_S == 0) and (data_loading_mode == "Legacy"):
        if independent_val_set:
            print('Importing Entire Train Data...')
            X_Train, Y_Train = get_datasets(train_image_dir, train_mask_dir, imlength, imwidth, num_channel, normalizing_factor_img, normalizing_factor_msk, class_number)
            X_Val, Y_Val = get_datasets(val_image_dir, val_mask_dir, imlength, imwidth, num_channel, normalizing_factor_img, normalizing_factor_msk, class_number)
        elif not independent_val_set:
            print('Importing Entire Train Data...')
            X_Train1, Y_Train1 = get_datasets(train_image_dir, train_mask_dir, imlength, imwidth, num_channel, normalizing_factorImg, normalizing_factor_msk, class_number)
            print('Randomly Splitting Data into Train and Validation Sets...')
            X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train1, Y_Train1, test_size=validation_portion, random_state=42)
    else:
        raise ValueError("Please select a correct combinarion of DataLoader and Deep Supervision Mode")
    print(' ')
    # Reinitialize Model for the New Fold, Change Function depending on Cases
    model = Segmentation_Model
    # Compile Model
    model.compile(loss=TFLosses(loss_function_name).loss(), optimizer=TFOptimizers(optimizer_function_name, learning_rate).optimizer(),
                  metrics=TFMetrics(metric_list).metric())
    if encoder_mode == "pretrained_encoder":
        model_name = encoder_name + '_' + decoder_name
    elif encoder_mode == "from_scratch":
        model_name = 'Scratch_' + decoder_name
    if num_iter == 1:
        # Print Model Summary
        model.summary()
    # Load Pre-trained Weights to continue training; delete the trained model from the directory in case of any change causing mismatch
    if os.path.exists('Results/'+model_name+'/'+model_name+'_'+str(imwidth)+'_Fold_'+str(i)+'.h5'):
        print('Loading PreTrained Weights...')
        model.load_weights('Results/'+model_name+'/'+model_name+'_'+str(imwidth)+'_Fold_'+str(i)+'.h5')
    print(' ')
    # Declare Callbacks
    callbacks = [EarlyStopping(monitor=monitor_param, patience=patience_amount, mode=patience_mode),
                 ModelCheckpoint('Results/'+model_name+'/'+model_name+'_'+str(imwidth)+'_Fold_'+str(i)+'.h5', verbose=1,
                                 monitor=monitor_param, save_best_only=True, mode=patience_mode)]
    # Train and Record History
    if (D_S == 0) and (data_loading_mode == "TF_DataLoader"):
        num_train_imgs = np.round(len(os.listdir(train_image_dir))*(1-validation_portion))
        num_val_imgs = np.round((num_train_imgs*(1/(1-validation_portion)))*validation_portion)
        if independent_val_set:
          num_val_imgs = len(os.listdir(val_image_dir))
        history = model.fit(train_ds, epochs=num_epochs, steps_per_epoch=np.ceil(num_train_imgs/batch_size), verbose=1,
                            validation_data=val_ds, validation_steps=np.ceil(num_val_imgs/batch_size), callbacks=callbacks)
    elif (data_loading_mode == "Legacy"):
        if D_S == 0:
            if independent_val_set:
                history = model.fit(X_Train, Y_Train, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_data=(X_Val, Y_Val), callbacks=callbacks)
            elif not independent_val_set:
                if validation_portion == 0:
                    validation_portion = 0.2  # Default
                history = model.fit(X_Train, Y_Train, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_split=validation_portion, callbacks=callbacks)
        elif D_S == 1:
            if independent_val_set:
                history = model.fit(X_Train, Y_Train_Dict, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_data=(X_Val, Y_Val_Dict), callbacks=callbacks, loss_weights=loss_weights)
            elif not independent_val_set:
                if validation_portion == 0:
                    validation_portion = 0.2  # Default
                history = model.fit(X_Train, Y_Train_Dict, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_split=validation_portion, callbacks=callbacks, loss_weights=loss_weights)
    # Save History Plots based on the History Keys
    plot_history(history, model_name, i)
    num_iter = num_iter + 1
    print('=======================================================================================')
    # Garbage Collector
    del model  # Delect any existing Model from the Memory to avoid Reuse in the next iteration
    gc.collect()
