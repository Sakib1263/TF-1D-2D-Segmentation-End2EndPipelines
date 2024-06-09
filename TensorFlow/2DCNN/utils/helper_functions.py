import os
import cv2
import random
import shutil
import numpy as np
import seaborn as sns
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from patchify import patchify
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, auc, roc_curve
sns.set_theme(style='white')


def create_patches(input_image, patch_shape, overlap_ratio):
    img = np.asarray(input_image)
    patch_width = patch_shape[0]
    patch_height = patch_shape[1]
    if len(img.shape) == 3:
        num_channels = img.shape[2]
        patch_shape = (patch_width, patch_height, num_channels)
    assert patch_width == patch_height, 'The patches are required to be squared shape'
    patches = patchify(img, patch_shape, step=np.int_(patch_height*(1-overlap_ratio)))
    num_patches = patches.shape[0]*patches.shape[1]
    return patches, num_patches


def one_hot_encoding(data):
    if np.ndim(data) == 1:
        data = np.expand_dims(data, 1)
    OHE = OneHotEncoder(sparse_output=False, dtype=np.int8)
    OHE_data = OHE.fit_transform(data)
    return OHE_data


def reverse_one_hot_encoding(Predictions):
    prediction_shape = Predictions.shape
    prediction_length = prediction_shape[0]
    Y_Preds = np.zeros((prediction_length, 1), dtype=int)
    for i in range(0, prediction_length):
        prediction = Predictions[i]
        x = np.where(prediction == np.max(prediction))
        x = int(x[0])
        Y_Preds[i] = x
    return Y_Preds


def misclassifications(Y_Test_, Y_Preds, Predictions, class_names, File_Names):
    missed_cases = []
    for i in range(0, len(Y_Test_)):
        if Y_Test_[i] != Y_Preds[i]:
            missed_case = str(File_Names[i]) + ': ' + str(Predictions[i])
            missed_cases.append(missed_case)
            print(f'Image No. {i}: ' + File_Names[i])
            print('Prediction' + str(class_names) + ': ' + str(Predictions[i]))
            print('')
    return missed_cases


def plot_history(history, history_path, Fold_Num):
    # list all dictionaries in history
    print('')
    history_list = list(history.keys())
    print(history_list)
    # Parameter 1
    history_train_param_1_name = history_list[0].capitalize()
    history_train_param_1 = list(history[history_list[0]])
    history_val_param_1_name = history_list[3].capitalize()
    history_val_param_1 = list(history[history_list[3]])
    # Parameter 2
    history_train_param_2_name = history_list[2].capitalize()
    history_train_param_2 = list(history[history_list[2]])
    history_val_param_2_name = history_list[4].capitalize()
    history_val_param_2 = list(history[history_list[4]])
    # Save and plot for loss
    plt.figure(figsize=(12, 8))
    plt.plot(history_train_param_1, 'ro-', linewidth=3)
    plt.plot(history_val_param_1, 'go-', linewidth=3)
    plt.title(f'{history_train_param_1_name} Curve', fontsize=20)
    plt.ylabel(history_train_param_1_name, fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig(f'{history_path}/{history_train_param_1_name}_Plot_Fold_{Fold_Num}.png')
    plt.tight_layout()
    plt.show()
    plt.close()
    # Save and plot for the metric (error or accuracy)
    plt.figure(figsize=(12, 8))
    plt.plot(history_train_param_2, 'ro-', linewidth=3)
    plt.plot(history_val_param_2, 'go-', linewidth=3)
    plt.title(f'{history_train_param_2_name} Curve', fontsize=20)
    plt.ylabel(history_train_param_2_name, fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig(f'{history_path}/{history_train_param_2_name}_Plot_Fold_{Fold_Num}.png')
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_conf_mat(Ground_Truth_Labels, Predictions, labels):
    confusion_matrix_raw = confusion_matrix(Ground_Truth_Labels, Predictions, normalize=None)
    confusion_matrix_norm = confusion_matrix(Ground_Truth_Labels, Predictions, normalize='true')
    shape = confusion_matrix_raw.shape
    data = np.asarray(confusion_matrix_raw, dtype=int)
    text = np.asarray(confusion_matrix_norm, dtype=float)
    annots = (np.asarray(["{0:.2f} ({1:.0f})".format(text, data) for text, data in zip(text.flatten(), data.flatten())])).reshape(shape[0],shape[1])
    fig = plt.figure(figsize=(len(labels)*3, len(labels)*2))
    sns.heatmap(confusion_matrix_norm, cmap='Blues', annot=annots, fmt='', annot_kws={'fontsize': 14}, xticklabels=labels, yticklabels=labels, vmax=1)
    plt.title('Confusion Matrix', fontsize=24)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    return fig


def plot_multiclass_roc(Y_Test, Predictions, class_number, save_dir):
    # Compute ROC curve and Area Under Curve (AUC) for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    Y_Test_OHE = one_hot_encoding(Y_Test)
    Predictions_OHE = one_hot_encoding(Predictions)
    for i in range(class_number):
        fpr[i], tpr[i], _ = roc_curve(Y_Test_OHE[:, i], Predictions_OHE[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_number)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(class_number):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= class_number
    if class_number < 1:
        raise ValueError("Number of classes cannot be less than 1.")
    
    # Compute micro-average ROC curve and ROC area
    fpr_micro, tpr_micro, thresholds = roc_curve(Y_Test.ravel(), Predictions.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    roc_auc_macro = auc(fpr_macro, tpr_macro)
    print (f'Area under the ROC Curve(s): {roc_auc_macro:0.2f}')

    # Plot all ROC curves
    plt.figure(figsize=(20, 10))
    plt.plot(fpr_micro, tpr_micro, label='micro-average ROC Curve (area = {0:0.2f})'''.format(roc_auc_micro), color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr_macro, tpr_macro, label='macro-average ROC Curve (area = {0:0.2f})'''.format(roc_auc_macro), color='navy', linestyle=':', linewidth=4)

    if class_number < 2:
      plt.plot(fpr, tpr, lw=2, label=f'ROC Curve of Class 1 (area = {roc_auc:0.2f})')
    elif class_number >= 2:
      for i in range(class_number):
          plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC Curve of Class {i} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k-', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('MultiClass ROC Plot with Respective AUC', fontsize=25)
    plt.legend(loc="lower right")
    plt.savefig(save_dir)
    plt.show()
    plt.close()


def plot_multiclass_precision_recall_curves(Y_Test, Predictions, class_number, save_dir):
    # Compute ROC curve and Area Under Curve (AUC) for each class
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    Y_Test_OHE = one_hot_encoding(Y_Test)
    Predictions_OHE = one_hot_encoding(Predictions)
    for i in range(class_number-1):
        precision[i], recall[i], _ = precision_recall_curve(Y_Test_OHE[:, i], Predictions_OHE[:, i])
        average_precision[i] = average_precision_score(Y_Test_OHE[:, i], Predictions_OHE[:, i])
        
    if class_number < 1:
        raise ValueError("Number of classes cannot be less than 1.")

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_Test.ravel(), Predictions.ravel())
    average_precision["micro"] = average_precision_score(Y_Test, Predictions, average="micro")
    print('Area under Precision-Recall Curve(s): {0:0.2f}'.format(average_precision["micro"]))

    from itertools import cycle
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(20, 10))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    l = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'''.format(average_precision["micro"]))

    for i, color in zip(range(class_number-1), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    plt.title('MultiClass Precision-Recall Curves', fontsize=25)
    plt.legend(lines, labels, loc=(0, -.3), prop=dict(size=14))
    plt.savefig(save_dir)
    plt.show()
    plt.close()


def get_datasets(imgs_dir, groundTruth_dir, height, width, channels, normalizing_factor_img, normalizing_factor_msk):
    Nimgs = len(os.listdir(imgs_dir))  # List containing all images
    Nmsks = len(os.listdir(groundTruth_dir))  # List containing all images
    print(f"Number of Test Images: {Nimgs}")
    print(f"Number of Test Masks (GroundTruth): {Nmsks}")
    imgs = np.empty((Nimgs, height, width, channels))
    groundTruth = np.empty((Nimgs, height, width))
    for path, subdirs, files in os.walk(imgs_dir):  # List all files, directories in the path
        for i in range(len(files)):
            # Original
            # print("Original image: "+files[i])
            img = Image.open(imgs_dir + '/' + files[i])
            img = np.asarray(img)/normalizing_factor_img
            imgs[i,:,:,:] = img
            # Corresponding Ground Truth
            groundTruth_name = files[i]
            # print ("Ground Truth Name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + '/' + groundTruth_name)
            g_truth = np.asarray(g_truth/normalizing_factor_msk)
            groundTruth[i,:,:] = g_truth
    print("Max Pixel Value for Images: " + str(np.max(imgs)))
    print("Min Pixel Value for Images: " + str(np.min(imgs)))
    print("Max Pixel Value for Masks: " + str(np.max(groundTruth)))
    print("Min Pixel Value for Masks: " + str(np.min(groundTruth)))
    print(f"Shape of the Test Image Dataset: {imgs.shape}")
    print(f"Shape of the Test Mask Dataset: {groundTruth.shape}")
    assert(imgs.shape == (Nimgs, height, width, channels))
    groundTruth = np.reshape(groundTruth, (Nimgs, height, width, 1))
    assert(groundTruth.shape == (Nimgs, height, width, 1))
    return imgs, groundTruth


def prepareTrainDict_V1(image_batch, model_depth, model_type):
    def approximate(inp, w_len, length, width, num_channel):
        ops = np.zeros((len(inp), length // w_len, width // w_len, num_channel))
        for c in range(0, num_channel):
            op = np.zeros((len(inp), length//w_len, width // w_len))
            for i in range(0, length, w_len):
                for j in range(0, width, w_len):
                    try:
                        op[:, i//w_len, j//w_len] = np.mean(np.mean(inp[:, i:i+w_len, j:j+w_len, c], axis=2), axis=1)
                    except Exception as e:
                        print(e)
            ops[:,:,:,c] = op
        return ops

    # Prcoess Image
    image_batch = np.array(image_batch)
    image_batch_shape = image_batch.shape
    if len(image_batch_shape) == 3:
        image_batch = np.expand_dims(image_batch, axis=3)
    image_batch_shape = image_batch.shape
    batch_size = image_batch_shape[0]
    img_length = image_batch_shape[1]
    img_width = image_batch_shape[2]
    num_channels = image_batch_shape[3]
    out = {}
    Y_Train_dict = {}
    out['out'] = image_batch
    Y_Train_dict['out'] = out['out']
    for i in range(1, (model_depth+1)):
        name = f'level{i}'
        if model_type == 'UNet':
            out[name] = approximate(image_batch, 2**i, img_length, img_width, num_channels)
        elif model_type == 'UNetPP':
            out[name] = image_batch
        Y_Train_dict[f'level{i}'] = out[f'level{i}']

    return Y_Train_dict


def prepareTrainDict_V2(image_batch, model_depth, model_type):
    def approximate(inp, w_len, length, width, num_channel, batch_size):
        assert length//w_len == width//w_len, 'The patches are required to be squared shape'
        ops = np.empty((batch_size, length//w_len, width//w_len, num_channel), dtype=np.uint8)
        for i in range(0, batch_size):
            patches = patchify(inp[i,:,:,:], (length//w_len, width//w_len, num_channel), step=length//w_len)
            ops[i,:,:,:] = np.mean(np.reshape(np.squeeze(patches), (patches.shape[0]*patches.shape[1], length//w_len, width//w_len, num_channel)), axis=0)
        return ops
    
    # Prcoess Image
    image_batch = np.array(image_batch)
    if len(image_batch.shape) == 3:
        image_batch = np.expand_dims(image_batch, axis=3)
    out = {}
    Y_Train_dict = {}
    out['out'] = image_batch
    Y_Train_dict['out'] = out['out']
    for i in range(1, (model_depth+1)):
        name = f'level{i}'
        if model_type == 'UNet':
            out[name] = approximate(image_batch, 2**i, image_batch.shape[1], image_batch.shape[2], image_batch.shape[3], image_batch.shape[0])
        elif model_type == 'UNetPP':
            out[name] = image_batch
        Y_Train_dict[f'level{i}'] = out[f'level{i}']

    return Y_Train_dict


def prepareTrainDict_V3(image_batch, model_depth, model_type):
    def approximate(inp, w_len, length, width, num_channel, batch_size):
        ops = np.empty((batch_size, length//w_len, width//w_len, num_channel), dtype=np.uint8)
        for i in range(0, batch_size):
            for c in range(0, num_channel):
                img_temp = Image.fromarray(inp[i,:,:,c], mode='L')
                img_temp = np.asarray(img_temp.resize((length//w_len, width//w_len), Image.Resampling.NEAREST))
                ops[i,:,:,c] = img_temp
        return ops
    
    # Prcoess Image
    image_batch = np.array(image_batch)
    if len(image_batch.shape) == 3:
        image_batch = np.expand_dims(image_batch, axis=3)
    out = {}
    Y_Train_dict = {}
    out['out'] = image_batch
    Y_Train_dict['out'] = out['out']
    for i in range(1, (model_depth+1)):
        name = f'level{i}'
        if model_type == 'UNet':
            out[name] = approximate(image_batch, 2**i, image_batch.shape[1], image_batch.shape[2], image_batch.shape[3], image_batch.shape[0])
        elif model_type == 'UNetPP':
            out[name] = image_batch
        Y_Train_dict[f'level{i}'] = out[f'level{i}']

    return Y_Train_dict


def prepareTrainDict(image_batch, model_depth, model_type):
    def approximate(inp, w_len):
        ops = tf.keras.layers.MaxPooling2D(pool_size=(w_len, w_len))(inp)
        return ops
    
    # Prcoess Image
    image_batch = np.array(image_batch)
    if len(image_batch.shape) == 3:
        image_batch = np.expand_dims(image_batch, axis=3)
    out = {}
    Y_Train_dict = {}
    out['out'] = image_batch
    Y_Train_dict['out'] = out['out']
    for i in range(1, (model_depth+1)):
        name = f'level{i}'
        if model_type == 'UNet':
            out[name] = approximate(image_batch, 2**i)
        elif model_type == 'UNetPP':
            out[name] = image_batch
        Y_Train_dict[f'level{i}'] = out[f'level{i}']

    return Y_Train_dict


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum


def process_raw_data(data_path, process_list):
    Class_Names = os.listdir(f'{data_path}')
    transform = A.Compose(process_list)
    for ii in tqdm(range(0, len(Class_Names))):
        # print(f'Current Class: {Class_Names[ii]}')  # Print Current Class Name
        # List containing all images of a certain class in the Raw Dataset
        Image_List = os.listdir(f'{data_path}/{Class_Names[ii]}')
        for iii in range(0, len(Image_List)):
            current_image = os.path.splitext(Image_List[iii])[0]
            # Read an image with OpenCV and process
            org_image = cv2.imread(f'{data_path}/{Class_Names[ii]}/{Image_List[iii]}')  # Read Original Image
            img_nparray = np.asarray(org_image)
            if img_nparray.shape[2] == 4:
                org_image = org_image[:,:,:3]  # Remove Alpha (4th) Channel from the Image if required
            org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB Colorspace, cv2 by default reads in BGR format
            transformed = transform(image=org_image)  # Augment Image
            transformed_image = transformed["image"]
            # transformed_image = cv2.resize(transformed_image, (331, 331), interpolation=cv2.INTER_CUBIC)  # Resize Image
            cv2.imwrite(f'{data_path}/{Class_Names[ii]}/{current_image}.png', transformed_image)  # Replacing the Original Image with a Transformed Version


def copyimagefile(source_path, destination_path, files_list):
    for order in range(1, len(files_list)):
        files = files_list[order]
        shutil.copyfile(os.path.join(source_path, files), os.path.join(destination_path, files))


def create_folds(raw_data_path, num_folds, train_portion, validation_portion=False):
  random.seed(1)
  Class_Names = os.listdir(raw_data_path)
  for i in range(1, num_folds + 1):
      print(f'Creating Fold {i}')
      for ii in tqdm(range(0, len(Class_Names))):
          # Get Train and Test Image Indices Randomly for each Fold
          source_path = f'{raw_data_path}/{Class_Names[ii]}'
          if (ii == 0):
            X_Tot = os.listdir(source_path)  # List containing all images
            X_Train_Len = int(len(X_Tot) * train_portion)
            X_Train = random.sample(X_Tot, (X_Train_Len - 1))  # Randomly formed List containing images for training, can be stratified otherwise
            X_Test = [x for x in X_Tot if x not in X_Train]  # List containing images for testing
            X_Val = []
            if validation_portion:
                X_Val_Len = int(len(X_Train) * validation_portion)  # Size of the Validation Set, subset of the Training Set
                X_Val = random.sample(X_Train, (X_Val_Len - 1))  # Validation Set
                X_Train = [x for x in X_Train if x not in X_Val]  # Validation Set is deducted from the Training Set

          # Make Required Directories after Checking their Existence, sometimes delete old folders and run the code again
          train_dir = f'Data/Train/fold_{i}/{Class_Names[ii]}'
          test_dir = f'Data/Test/fold_{i}/{Class_Names[ii]}'
          val_dir = f'Data/Val/fold_{i}/{Class_Names[ii]}'
          if not os.path.isdir(train_dir):
              os.makedirs(train_dir)  # Train Directory for Fold ii
          if not os.path.isdir(test_dir):
              os.makedirs(test_dir)  # Test Directory for Fold ii
          if (not os.path.isdir(val_dir)) and (validation_portion != False):
              os.makedirs(val_dir)  # Validation Directory for Fold ii

          # Copy Image Files from the Source Folder to the Destination Folder
          copyimagefile(source_path, train_dir, X_Train)
          copyimagefile(source_path, test_dir, X_Test)
          if validation_portion:
              copyimagefile(source_path, val_dir, X_Val)  # True if validation set is created independently


def augment(data_path, num_folds, augmentation_list, augmentation_num):
    Class_Names = os.listdir(f'{data_path}/fold_1')
    # Declare an Augmentation Pipeline
    transform = A.Compose(augmentation_list)
    for i in range(1, num_folds + 1):
        print(f'\nCurrently Processing Fold {i}')
        for ii in tqdm(range(0, len(Class_Names))):
            # print(f'Current Class: {Class_Names[ii]}')
            # List containing all images of a certain class in a certain fold
            X_Train_List = os.listdir(f'{data_path}/fold_{i}/{Class_Names[ii]}')
            for iii in range(0, len(X_Train_List)):
                current_image = os.path.splitext(X_Train_List[iii])[0]
                # Read an image with OpenCV and convert it to the RGB colorspace
                org_image = cv2.imread(f'{data_path}/fold_{i}/{Class_Names[ii]}/{X_Train_List[iii]}')  # Read Original Image
                img_nparray = np.asarray(org_image)
                if img_nparray.shape[2] == 4:
                    org_image = org_image[:,:,:3]  # Remove Alpha (4th) Channel from the Image
                org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB Colorspace
                for iv in range(1, augmentation_num + 1):
                    transformed = transform(image=org_image)  # Augment Image
                    transformed_image = transformed["image"]
                    cv2.imwrite(f'{data_path}/fold_{i}/{Class_Names[ii]}/{current_image}_Augmented_{iv}.png', transformed_image)
