import os
import gc
import h5py
import configparser
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
# Local Libraries
from utils.helper_functions import *
from utils.tf_losses import TFLosses
from utils.tf_metrics import TFMetrics
from utils.tf_optimizers import TFOptimizers
from utils.DataGenerator import CustomDataGenerator
from models.model_selector import model_selector


'''CONFIGURATIONS'''
## Data Configurations
config_file = configparser.ConfigParser()
config_file.read('Train_Configs.ini')
train_dir = config_file["TRAIN"]["train_dir"]  # Train Directory
val_dir = config_file["TRAIN"]["val_dir"]  # Validation Directory
data_loading_mode = config_file["TRAIN"]["data_loading_mode"]  # TF_DataLoader: Use TensorFlow DataLoader module to load data | Custom_DataLoader: Load all train and validation data
independent_val_set = config_file["TRAIN"].getboolean("independent_val_set")  # True: Independent Validation Set | False: Validation Set randomly splitted from the Training Set
validation_portion = np.float_(config_file["TRAIN"]["validation_portion"])  # 0 to 1 [Default: 0; when validation set is independent, otherwise created randomly while training based on "validation_portion"]
imlength = np.int_(config_file["TRAIN"]["imlength"])  # Length or Height of the Image | Image Size: [imwidth, imlength]
imwidth = np.int_(config_file["TRAIN"]["imwidth"])  # Width of the Image
image_color_mode = config_file["TRAIN"]["image_color_mode"]  # Color Mode of the images [rgb, rgba (rgb with transparent alpha channel), grayscale (black and white single channel image)]
mask_color_mode = config_file["TRAIN"]["mask_color_mode"]  # Color Mode of the masks [rgb or grayscale (black and white single channel image)]
num_channels = np.int_(config_file["TRAIN"]["num_channels"])  # Number of Input Channels in the Model [rgb:3, rgba:4, grayscale:1]
normalizing_factor_img = np.int_(config_file["TRAIN"]["normalizing_factor_img"])  # 255.0 for images with pixel values varying between 0 to 255. If it is between 0 to 1, change it to 1
normalizing_factor_msk = np.int_(config_file["TRAIN"]["normalizing_factor_msk"])  # 255.0 for masks with pixel values varying between 0 to 255. If it is between 0 to 1, change it to 1
## Model Configurations
model_genre = config_file["TRAIN"]["model_genre"]  # model_genre: Generation or Genre of the Model: UNet, FPN, LinkNet, etc.
# Encoder
encoder_mode = config_file["TRAIN"]["encoder_mode"]  # Transfer Learning: "pretrained_encoder" | Train from scratch: "from_scratch"
encoder_name = config_file["TRAIN"]["encoder_name"]  # Select an Encoder from a pool of ImageNet trained Models available from TensorFlow, default: ResNet50
encoder_trainable = config_file["TRAIN"].getboolean("encoder_trainable")  # Fine Tuning ON/OFF [True/False] | Start with OFF, Fine Tune later in the 2nd stage, which is optional
# Decoder
decoder_name = config_file["TRAIN"]["decoder_name"]  # Select a Model from the list to train from scratch, UNet is kept as default
model_width = np.int_(config_file["TRAIN"]["model_width"])  # Number of Filters or Kernels of the Input Layer, subsequent layers start from here
model_depth = np.int_(config_file["TRAIN"]["model_depth"])  # Number of Layers in the Model [For the "pretrained_encoder" mode: Maximum 5, Minimum 1]
output_nums = np.int_(config_file["TRAIN"]["output_nums"])  # Number of Outputs for the model
A_E = np.int_(config_file["TRAIN"]["A_E"])  # Turn on AutoEncoder Mode for Feature Extraction [Default: 0]
A_G = np.int_(config_file["TRAIN"]["A_G"])  # Turn on for Guided Attention [Default: 0]
LSTM = np.int_(config_file["TRAIN"]["LSTM"])  # Turn on for LSTM [Default: 0]
dense_loop = np.int_(config_file["TRAIN"]["dense_loop"])  # Number of Densely Connected Residual Blocks in the BottleNeck Layer [Default: 2]
feature_number = np.int_(config_file["TRAIN"]["feature_number"])  # Number of Features to be Extracted [Only required for the AutoEncoder (A_E) Mode]
is_transconv = config_file["TRAIN"].getboolean("is_transconv")  # True: Transposed Convolution | False: UpSampling in the Decoder layer
alpha = np.float_(config_file["TRAIN"]["alpha"])  # Alpha parameter, required for MultiResUNet models [Default: 1]
q_onn = np.int_(config_file["TRAIN"]["q_onn"])  # 'q' for Self-ONN' [Default: 3, set 1 to get CNN]
final_activation = config_file["TRAIN"]["final_activation"]  # Activation Function for the Final Layer: "Linear", "Sigmoid", "Softmax", etc. depending on the problem type
class_number = np.int_(config_file["TRAIN"]["class_number"])  # Number of Output Classes [e.g., here for Kidney Tumor segmentation, Class 1: Kidney | Class 2: Tumor]
## Training Configurations
batch_size = np.int_(config_file["TRAIN"]["batch_size"])  # Batch Size of the Images being loaded for training
learning_rate = np.float_(config_file["TRAIN"]["learning_rate"])  # During Fine-Tuning, the Learning Rate should be very low (e.g., 1e-5), otherwise more (e.g., 1e-4, 1e-3)
start_fold = np.int_(config_file["TRAIN"]["start_fold"])  # Fold to Start Training, can be varied from 1 to the last fold
end_fold = np.int_(config_file["TRAIN"]["end_fold"])  # Fold to End Training, can be any value from the start_fold [Number of Folds + 1]
monitor_param = config_file["TRAIN"]["monitor_param"]
patience_amount = np.int_(config_file["TRAIN"]["patience_amount"])
patience_amount_RLROnP = np.int_(config_file["TRAIN"]["patience_amount_RLROnP"])
patience_mode = config_file["TRAIN"]["patience_mode"]
RLROnP_factor = np.float_(config_file["TRAIN"]["RLROnP_factor"])
num_epochs = np.int_(config_file["TRAIN"]["num_epochs"])
loss_function_name = config_file["TRAIN"]["loss_function"]
optimizer_function_name = config_file["TRAIN"]["optimizer_function"]
metric_list = config_file["TRAIN"]["metric_list"]
save_history = config_file["TRAIN"].getboolean("save_history")
load_weights = config_file["TRAIN"].getboolean("load_weights")
save_dir = config_file["TRAIN"]["save_dir"]
task_name = config_file["TRAIN"]["task_name"]
seed = np.int_(config_file["TRAIN"]["seed"])
# Patchify
ispatchify = config_file["TRAIN"].getboolean("patchify")
patch_width = np.int_(config_file["TRAIN"]["patch_width"])  # Length or Height of the Image | Image Size: [imwidth, imlength]
patch_height = np.int_(config_file["TRAIN"]["patch_height"])  # Width of the Image
overlap_ratio = np.float_(config_file["TRAIN"]["overlap_ratio"])
# Deep Supervision
D_S = np.int_(config_file["TRAIN"]["D_S"])  # Turn on Deep Supervision [Default: 0]
ds_type = config_file["TRAIN"]["ds_type"]  # "UNet" or "UNetPP"; only required when Deep Supervision (D_S) is on

config_file = open("Train_Configs.ini", "r")
content = config_file.read()
print("Current Configurations:\n")
print(content)
config_file.flush()
config_file.close()

## Set or Assert Default Conditions
# Validation Set
if independent_val_set == True:
    assert validation_portion == 0.0
# Image Color Mode and Input Channels
if image_color_mode == 'rgb':
    assert num_channels == 3
elif image_color_mode == 'rgba':
    assert num_channels == 4
elif image_color_mode == 'grayscale':
    assert num_channels == 1
# Error Metrics
if metric_list == ["MeanSquaredError"]:
    monitor_param == "val_mean_sqaured_error"
elif metric_list == ["MeanAbsoluteError"]:
    monitor_param == "val_mean_absolute_error"

# Load 2D Segmentation Model
if ispatchify == False:
    Segmentation_Model = model_selector(model_genre,          
                                    encoder_name,
                                    decoder_name, 
                                    imlength, 
                                    imwidth, 
                                    model_width, 
                                    model_depth, 
                                    num_channels=num_channels,
                                    output_nums=output_nums,
                                    ds=D_S, 
                                    ae=A_E, 
                                    ag=A_G, 
                                    lstm=LSTM, 
                                    dense_loop=dense_loop,
                                    feature_number=feature_number,  
                                    is_transconv=is_transconv,
                                    final_activation=final_activation, 
                                    train_mode=encoder_mode,
                                    is_base_model_trainable=encoder_trainable,
                                    alpha=alpha,
                                    q=q_onn).segmentation_model()
elif ispatchify == True:
    Segmentation_Model = model_selector(model_genre,          
                                    encoder_name,
                                    decoder_name, 
                                    patch_width, 
                                    patch_height, 
                                    model_width, 
                                    model_depth, 
                                    num_channels=num_channels,
                                    output_nums=output_nums,
                                    ds=D_S, 
                                    ae=A_E, 
                                    ag=A_G, 
                                    lstm=LSTM, 
                                    dense_loop=dense_loop,
                                    feature_number=feature_number,  
                                    is_transconv=is_transconv,
                                    final_activation=final_activation, 
                                    train_mode=encoder_mode,
                                    is_base_model_trainable=encoder_trainable,
                                    alpha=alpha,
                                    q=q_onn).segmentation_model()

# Declare Blank Arrays
image_datagen = []
mask_datagen = []
train_image_datagen = []
train_mask_datagen = []
val_image_datagen = []
val_mask_datagen = []
# Create TensorFlow Data Generator Instance
if data_loading_mode == "TF_DataLoader":
    if independent_val_set == False:
        image_datagen = ImageDataGenerator(rescale=1./normalizing_factor_img, 
                                           validation_split=validation_portion)
        mask_datagen = ImageDataGenerator(rescale=1./normalizing_factor_msk, 
                                          validation_split=validation_portion)
        # Provide the same seed and keyword arguments to the fit and flow methods
    elif independent_val_set == True:
        train_image_datagen = ImageDataGenerator(rescale=1./normalizing_factor_img)
        train_mask_datagen = ImageDataGenerator(rescale=1./normalizing_factor_msk)
        val_image_datagen = ImageDataGenerator(rescale=1./normalizing_factor_img)
        val_mask_datagen = ImageDataGenerator(rescale=1./normalizing_factor_msk)
# Main Training Loop
num_iter = 1
for i in range(start_fold, (end_fold + 1)):
    # Import Train Dataset using Image Data Generator pipeline
    print(f'Fold {i}\n')
    train_image_dir = train_dir + f'/Images/Fold_{i}'
    train_mask_dir = train_dir + f'/Masks/Fold_{i}'
    val_image_dir = val_dir + f'/Images/Fold_{i}'
    val_mask_dir = val_dir + f'/Masks/Fold_{i}'
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
    if data_loading_mode == "TF_DataLoader":
        if independent_val_set == False:
            print('Importing Train and (Non-independent) Validation Data using TF-DataLoader.')
            # Create Train Data Generators (N.B.: It has Data Preprocessing Options)
            # Import Train Data in Batches - Condition for Data with/without a Ready Validation Set
            train_image_generator = image_datagen.flow_from_directory(train_image_dir,
                                                                      color_mode=image_color_mode,
                                                                      target_size=(imlength, imwidth),
                                                                      classes=None,
                                                                      batch_size=batch_size,
                                                                      class_mode=None,
                                                                      subset="training",
                                                                      shuffle=False, 
                                                                      seed=seed)
            train_mask_generator = mask_datagen.flow_from_directory(train_mask_dir,
                                                                    color_mode=mask_color_mode,
                                                                    target_size=(imlength, imwidth),
                                                                    classes=None,
                                                                    batch_size=batch_size,
                                                                    class_mode=None,
                                                                    subset="training",
                                                                    shuffle=False,
                                                                    seed=seed)
            train_ds = zip(train_image_generator, train_mask_generator)
            # Import Validation Set - Randomly Splitted from the Training Set
            val_image_generator = image_datagen.flow_from_directory(train_image_dir,
                                                                    color_mode=image_color_mode,
                                                                    target_size=(imlength, imwidth),
                                                                    classes=None,
                                                                    batch_size=batch_size,
                                                                    class_mode=None,
                                                                    subset="validation",
                                                                    shuffle=False,
                                                                    seed=seed)
            val_mask_generator = mask_datagen.flow_from_directory(train_mask_dir,
                                                                  color_mode=mask_color_mode,
                                                                  target_size=(imlength, imwidth),
                                                                  classes=None,
                                                                  batch_size=batch_size,
                                                                  class_mode=None,
                                                                  subset="validation",
                                                                  shuffle=False,
                                                                  seed=seed)
            val_ds = zip(val_image_generator, val_mask_generator)
        elif independent_val_set == True:
            print('Importing Train and (Independent) Validation Data using TF2 DataLoader. In the Train and Validation sets, respectively: ')
            # Create Train Data Generators (N.B.: It has Data Preprocessing Options)
            # Import Train Data in Batches - Condition for Data with/without a Ready Validation Set
            train_image_generator = train_image_datagen.flow_from_directory(train_image_dir, 
                                                                            color_mode=image_color_mode, 
                                                                            target_size=(imlength, imwidth), 
                                                                            classes=None,
                                                                            batch_size=batch_size,
                                                                            class_mode=None, 
                                                                            subset=None, 
                                                                            shuffle=True, 
                                                                            seed=seed)
            train_mask_generator = train_mask_datagen.flow_from_directory(train_mask_dir, 
                                                                          color_mode=mask_color_mode, 
                                                                          target_size=(imlength, imwidth), 
                                                                          classes=None,
                                                                          batch_size=batch_size,
                                                                          class_mode=None, 
                                                                          subset=None, 
                                                                          shuffle=True, 
                                                                          seed=seed)
            train_ds = zip(train_image_generator, train_mask_generator)
            # Import Validation Set - Randomly Splitted from the Training Set
            val_image_generator = val_image_datagen.flow_from_directory(val_image_dir, 
                                                                        color_mode=image_color_mode, 
                                                                        target_size=(imlength, imwidth), 
                                                                        classes=None,
                                                                        batch_size=batch_size,
                                                                        class_mode=None, 
                                                                        subset=None, 
                                                                        shuffle=False, 
                                                                        seed=seed)
            val_mask_generator = val_mask_datagen.flow_from_directory(val_mask_dir, 
                                                                      color_mode=mask_color_mode, 
                                                                      target_size=(imlength, imwidth), 
                                                                      classes=None,
                                                                      batch_size=batch_size, 
                                                                      class_mode=None, 
                                                                      subset=None, 
                                                                      shuffle=False, 
                                                                      seed=seed)
            val_ds = zip(val_image_generator, val_mask_generator)
    elif data_loading_mode == "Custom_DataLoader":
        train_ds = CustomDataGenerator(img_dir=f'{train_image_dir}/Images',
                                               msk_dir=f'{train_mask_dir}/Kidney',
                                               img_size=(imlength,imwidth),
                                               batch_size=batch_size,
                                               num_img_channel=num_channels,
                                               num_msk_channel=1,
                                               norm_factor_img=normalizing_factor_img,
                                               norm_factor_msk=normalizing_factor_msk,
                                               num_class=class_number,
                                               is_train=True,
                                               patchify=ispatchify,
                                               patch_shape=(patch_height,patch_width),
                                               overlap_ratio=overlap_ratio,
                                               deep_supervision=D_S,
                                               model_depth=model_depth,
                                               ds_type=ds_type
                                               )
        if independent_val_set == True:
            val_ds = CustomDataGenerator(img_dir=f'{val_image_dir}/Images',
                                            msk_dir=f'{val_mask_dir}/Kidney',
                                            img_size=(imlength,imwidth),
                                            batch_size=batch_size,
                                            num_img_channel=num_channels,
                                            num_msk_channel=1,
                                            norm_factor_img=normalizing_factor_img,
                                            norm_factor_msk=normalizing_factor_msk,
                                            num_class=class_number,
                                            is_train=False,
                                            patchify=ispatchify,
                                            patch_shape=(patch_height,patch_width),
                                            overlap_ratio=overlap_ratio,
                                            deep_supervision=D_S,
                                            model_depth=model_depth,
                                            ds_type=ds_type
                                            )
    else:
        raise ValueError("Please select a valid DataLoader")
    print(' ')
    # Reinitialize Model for the New Fold, Change Function depending on Cases
    model = Segmentation_Model
    # Compile Model
    model.compile(loss=TFLosses(loss_function_name).loss(),
                  optimizer=TFOptimizers(optimizer_function_name, learning_rate).optimizer(),
                  metrics=TFMetrics(metric_list).metric()
                  )
    if encoder_mode == "pretrained_encoder":
        if D_S == 0:
            if ispatchify == False:
                model_name = encoder_name + '_' + decoder_name
            elif ispatchify == True:
                model_name = encoder_name + '_' + decoder_name + '_patched'
        elif D_S == 1:
            if ispatchify == False:
                model_name = encoder_name + '_' + decoder_name + '_DS'
            elif ispatchify == True:
                model_name = encoder_name + '_' + decoder_name + '_DS_patched'
    elif encoder_mode == "from_scratch":
        if D_S == 0:
            if ispatchify == False:
                model_name = decoder_name + '_from_scratch'
            elif ispatchify == True:
                model_name = decoder_name + '_from_scratch_patched'
        elif D_S == 1:
            if ispatchify == False:
                model_name = decoder_name + '_from_scratch_DS'
            elif ispatchify == True:
                model_name = decoder_name + '_from_scratch_DS_patched'
    if task_name == "None":
        task_name = model_name
    # Print Model Info
    if num_iter == 1:
        # Model Summary
        # print(model.summary())
        trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        print(f'Trainable Params: {trainable_params}')
        print(f'Non-trainable Params: {non_trainable_params}')
        print(f'Total Params: {total_params}\n')
    # Load Pre-trained Weights to continue training; delete the trained model from the directory in case of any change causing mismatch
    if os.path.exists(f'{save_dir}/{task_name}/Fold_{i}/'+model_name+'_'+str(imwidth)+'_Fold_'+str(i)+'.keras') and (load_weights == True) and (encoder_trainable == False):
        print('Loading PreTrained Weights...')
        model.load_weights(f'{save_dir}/{task_name}/Fold_{i}/'+model_name+'_'+str(imwidth)+'_Fold_'+str(i)+'.keras')
    if encoder_trainable == True:
        if os.path.exists(f'{save_dir}/{task_name}/Fold_{i}/'+model_name+'_'+str(imwidth)+'_Fold_'+str(i)+'.keras'):
            print('Loading PreTrained Weights for Finetuning...')
            model.load_weights(f'{save_dir}/{task_name}/Fold_{i}/'+model_name+'_'+str(imwidth)+'_Fold_'+str(i)+'.keras')
        else:
            print('Previously trained model is not available for finetuning. ImageNet weights need to be used.')
    print(' ')
    # Declare Callbacks
    callbacks = [EarlyStopping(monitor=monitor_param,
                               patience=patience_amount,
                               mode=patience_mode),
                 ModelCheckpoint(f'{save_dir}/{task_name}/Fold_{i}/'+model_name+'_'+str(imwidth)+'_Fold_'+str(i)+'.keras',
                                 verbose=1,
                                 monitor=monitor_param,
                                 save_best_only=True,
                                 mode=patience_mode),
                 ReduceLROnPlateau(monitor=monitor_param,
                                   factor=RLROnP_factor,
                                   patience=patience_amount_RLROnP,
                                   verbose=1,
                                   mode=patience_mode,
                                   min_delta=0.0001,
                                   cooldown=0,
                                   min_lr=0)]
    # Train, Check Validity, Save Model and Record History
    if data_loading_mode == "TF_DataLoader":
        num_train_imgs = np.round(len(os.listdir(train_image_dir + '/Images'))*(1-validation_portion))
        num_val_imgs = np.round(len(os.listdir(train_image_dir + '/Images'))*validation_portion)
        if independent_val_set == True:
            num_val_imgs = np.round(len(os.listdir(val_image_dir + '/Images')))
        model_history = model.fit(train_ds,
                            epochs=num_epochs,
                            steps_per_epoch=(num_train_imgs//batch_size),
                            verbose=1,
                            validation_data=val_ds,
                            validation_steps=num_val_imgs,
                            callbacks=callbacks)
    elif (data_loading_mode == "Custom_DataLoader"):
        if independent_val_set == True:
            model_history = model.fit(train_ds,
                                epochs=num_epochs, 
                                verbose=1, 
                                validation_data=val_ds, 
                                callbacks=callbacks)
        elif independent_val_set == False:
            if (validation_portion <= 0) or (validation_portion > 1):
                validation_portion = 0.2  # Default
            model_history = model.fit(train_ds, 
                                epochs=num_epochs, 
                                verbose=1, 
                                validation_split=validation_portion, 
                                callbacks=callbacks)
    # Save History Plots based on the History Keys
    if save_history:
        print('Saving History...')
        history_dict = model_history.history
        history_list = list(history_dict)
        history_len = len(history_list)
        history_path = f'{save_dir}/{task_name}/Fold_{i}'
        if not os.path.exists(history_path):
            os.makedirs(history_path)
        hf = h5py.File(f'{history_path}/{model_name}_Fold_{i}_History.h5', 'w')
        for j in range(0,history_len):
            history_item_name = history_list[j]
            history_item_val = history_dict[history_item_name]
            hf.create_dataset(f'{history_item_name}', data=history_item_val)
        hf.close()
        # Get the dictionary containing each metric and the loss for each epoch
        plot_history(history_dict, history_path, i)
    
    print('\n')
    num_iter = num_iter + 1
    print('=======================================================================================')
    # Garbage Collector
    del model  # Delect any existing Model from the Memory to avoid Reuse in the next iteration
    gc.collect()
