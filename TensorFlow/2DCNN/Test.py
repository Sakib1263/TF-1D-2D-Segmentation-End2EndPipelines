import os
import gc
import configparser
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from patchify import patchify, unpatchify
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from utils.helper_functions import *
np.seterr(invalid='ignore')


"""[CONFIGURATIONS]"""
## General Configurations
config_file = configparser.ConfigParser()
config_file.read('Test_Configs.ini')
test_dir = config_file["TEST"]["test_dir"]
imheight = np.int_(config_file["TEST"]["imheight"])
imwidth = np.int_(config_file["TEST"]["imwidth"])
image_color_mode = config_file["TEST"]["image_color_mode"]
mask_color_mode = config_file["TEST"]["mask_color_mode"]
num_channels = np.int_(config_file["TEST"]["num_channels"])
class_number = np.int_(config_file["TEST"]["class_number"])
labels = config_file["TEST"]["labels"]
## Model Configurations
encoder_mode = config_file["TEST"]["encoder_mode"]
encoder_name = config_file["TEST"]["encoder_name"]
decoder_name = config_file["TEST"]["decoder_name"]
## Test Configurations
batch_size = np.int_(config_file["TEST"]["batch_size"])
normalizing_factor_img = np.float_(config_file["TEST"]["normalizing_factor_img"])
normalizing_factor_msk = np.float_(config_file["TEST"]["normalizing_factor_msk"])
start_fold = np.int_(config_file["TEST"]["start_fold"])
end_fold = np.int_(config_file["TEST"]["end_fold"])
num_iter = np.int_(config_file["TEST"]["num_iter"])
threshold = np.float_(config_file["TEST"]["threshold"])
seed = np.int_(config_file["TEST"]["seed"])
save_dir = config_file["TEST"]["save_dir"]
# Patchify
ispatchify = config_file["TEST"].getboolean("patchify")
patch_width = np.int_(config_file["TEST"]["patch_width"])  # Length or Height of the Image | Image Size: [imwidth, imlength]
patch_height = np.int_(config_file["TEST"]["patch_height"])  # Width of the Image
overlap_ratio = np.float_(config_file["TEST"]["overlap_ratio"])
# Deep Supervision
D_S = np.int_(config_file["TEST"]["D_S"])  # Turn on Deep Supervision [Default: 0]

config_file = open("Test_Configs.ini", "r")
content = config_file.read()
print("Current Configurations:\n")
print(content)
config_file.flush()
config_file.close()

'''Custom Configs'''
# Labels or Pixel Categories
if (labels == "") or (labels == "[]") or (labels == []) or (labels == None):
  labels = []
  for i in range(0,(class_number+1)):
    labels.append(str(i))
# Set Model Name
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
# Declare Global Blank Arrays
test_image_dir = []
test_mask_dir = []

counter_pf = 0  # pf: Per Fold
# Main Testing Loop
for i in range(start_fold, (end_fold + 1)):
    # Directories (Loading and Saving)
    test_image_dir = test_dir + f'/Images/Fold_{i}/Images'
    test_mask_dir = test_dir + f'/Masks/Fold_{i}/Kidney'
    results_save_dir = save_dir + f'/{model_name}/Fold_{i}'
    if not os.path.exists(results_save_dir):
      os.makedirs(results_save_dir)
    predicted_mask_save_dir = results_save_dir + '/Predictions'
    if not os.path.exists(predicted_mask_save_dir):
      os.makedirs(predicted_mask_save_dir)
    # Read Images and Corresponding Masks
    imgs = os.listdir(test_image_dir)  # List containing all Images
    msks = os.listdir(test_mask_dir)  # List containing all Masks
    Nimgs = len(imgs)  # Number of Images
    Nmsks = len(msks)  # Number of Masks
    y_true_all_pf = np.zeros((imwidth*imheight,Nmsks), dtype=np.int8)
    y_pred_all_pf = np.zeros((imwidth*imheight,Nmsks), dtype=np.int8)
    assert Nimgs == Nmsks, 'Number of Images and corresponding Masks are not equal!'
    num_batches = np.int_(np.ceil(Nimgs/batch_size))
    # Load Pre-trained Weights
    if (os.path.exists(f'Results/{model_name}/Fold_{i}/'+model_name+'_'+str(imwidth)+f'_Fold_{i}.keras')):
        print('Loading Pretrained Model...')
        model = None
        gc.collect()
        model = tf.keras.saving.load_model(f'Results/{model_name}/Fold_{i}/'+model_name+'_'+str(imwidth)+f'_Fold_{i}.keras')
    else:
        raise ValueError("Requested pretrained model is not present in the provided directory")
    # Now delve into making predictions
    print('Making Predictions...')
    counter_pb = 0
    for ii in range(0, num_batches):
      print(f'Batch Number {ii+1} out of {num_batches}')
      counter = 0
      img_batch = np.zeros((batch_size, imheight, imwidth, num_channels), dtype=np.float64)
      msk_batch = np.zeros((batch_size, imheight, imwidth, 1), dtype=np.uint8)
      pred_batch = np.zeros((batch_size, imheight, imwidth, class_number), dtype=np.float64)
      for iii in range(ii*batch_size,(ii+1)*batch_size):
        if (iii >= Nimgs):
          continue
        img = Image.open(test_image_dir + '/' + imgs[counter])
        if (img.size[0] != imwidth) or (img.size[1] != imheight):
          img = Image.resize(img, [imwidth, imheight, num_channels], Image.Resampling.NEAREST)
        img = np.asarray(img)/normalizing_factor_img
        if len(img.shape) == 2:
          img = np.expand_dims(img, axis=2)
        if len(img.shape) == 4:
          img = img[:,:,:,0]
        img_batch[counter,:,:,:] = img
        # Corresponding Ground Truth Mask
        msk = Image.open(test_mask_dir + '/' + msks[counter])
        if (msk.size[0] != imwidth) or (msk.size[1] != imheight):
          msk = Image.resize(msk, [imwidth, imheight], Image.Resampling.NEAREST)
        msk = np.asarray(msk)/normalizing_factor_msk
        if len(msk.shape) == 3:
          msk = msk[:,:,0]
        elif len(msk.shape) == 4:
          msk = msk[:,:,0,0]
        msk_batch[counter,:,:,0] = msk
        # Predict
        if ispatchify == True:
          # Patchify Images
          img_patches, num_img_patches = create_patches(img, (patch_width, patch_height), overlap_ratio)
          img_patches_reshaped = np.reshape(np.squeeze(img_patches), (num_img_patches, patch_width, patch_height, num_channels))
          pred_patches = model.predict(img_patches_reshaped, verbose=0)
          if D_S == 1:
            pred_patches = pred_patches[0]
          msk_patches, num_msk_patches = create_patches(msk, (patch_width, patch_height), overlap_ratio)
          # Unpatchify Predicted Masks
          pred_patches_reshaped = np.reshape(np.squeeze(pred_patches), msk_patches.shape)
          pred = unpatchify(pred_patches_reshaped, msk.shape)
        else:
          pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
        if D_S == 1:
          pred = pred[0]
        pred_batch[counter,:,:,:] = pred
        counter = counter + 1
      img_batch = img_batch[0:counter,:,:,:]
      msk_batch = msk_batch[0:counter,:,:,:]
      pred_batch = pred_batch[0:counter,:,:,:]
      if class_number > 1:
        for j in range(0, class_number):
          pred_per_class_temp = pred_batch[:,:,:,j]
          pred_per_class_binarized_temp = np.where(pred_per_class_temp < threshold, 0, 1)
          pred_batch_new = pred_batch_new + pred_per_class_binarized_temp
        pred_batch = np.int_(np.expand_dims(pred_batch_new, axis=3))
      elif class_number == 1:
        pred_batch = np.int_(np.where(pred_batch < threshold, 0, 1))
      # Save Pred
      for k in range(0, batch_size):
        if (ii*batch_size + k) >= Nimgs:
          continue
        pred_data = pred_batch[k,:,:,0].astype(np.int8)
        pred_data_from_array = Image.fromarray(pred_data * np.round(normalizing_factor_msk / class_number))
        pred_data_from_array = pred_data_from_array.convert("L")
        pred_dir = predicted_mask_save_dir + '/' + imgs[counter_pb*batch_size + k]
        pred_data_from_array.save(pred_dir)
      # print(np.unique(msk_batch))
      # print(np.unique(pred_batch))
      y_true = np.asarray(msk_batch.ravel(), dtype=np.int8)
      y_pred = np.asarray(pred_batch.ravel(), dtype=np.int8)
      # Main confusion matrix 'cm' (per fold)
      cm = confusion_matrix(y_true, y_pred)
      # cm_per_class: it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
      cm_per_class = multilabel_confusion_matrix(y_true, y_pred)
      if counter_pb == 0:
        cm_pf = cm
        cm_per_class_pf = cm_per_class
      elif counter_pb > 0:
        cm_pf = cm_pf + cm
        cm_per_class_pf = cm_per_class_pf + cm_per_class
      for k in range(0, batch_size):
        if (ii*batch_size + k) >= Nimgs:
          continue
        y_true_all_pf[:,ii*batch_size + k] = y_true[(k*imwidth*imheight):((k+1)*imwidth*imheight)]
        y_pred_all_pf[:,ii*batch_size + k] = y_pred[(k*imwidth*imheight):((k+1)*imwidth*imheight)]
      # y_true_all_pf.extend(y_true)
      # y_pred_all_pf.extend(y_pred)
      # Garbage Collector
      y_true = None
      y_pred = None
      counter_pb = counter_pb + 1
    y_true_all_pf = np.asarray(y_true_all_pf[:,0:Nimgs].ravel(), dtype=np.uint8)
    y_pred_all_pf = np.asarray(y_pred_all_pf[:,0:Nimgs].ravel(), dtype=np.uint8)
    # Normalized Confusion Metrics
    cmn_pf = cm_pf.astype('float') / cm_pf.sum(axis=1)[:, np.newaxis]
    # Overall Accuracy
    Overall_Accuracy = np.sum(np.diagonal(cm_pf))/np.sum(cm_pf)
    Overall_Accuracy = round(Overall_Accuracy*100, 2)
    # Create confusion matrix table (pd.DataFrame)
    cm_table = pd.DataFrame(cm_pf, index=labels, columns=labels)
    cmn_table = pd.DataFrame(cmn_pf, index=labels, columns=labels)
    # Generate Confusion matrix figure
    cm_plot = plot_conf_mat(y_true_all_pf, y_pred_all_pf, labels)
    cm_plot.savefig(results_save_dir + f'/{model_name}_Confusion_Matrix_Fold_{i}.png', dpi=600)
    # Generate Multiclass ROC curve figure
    roc_plot = plot_multiclass_roc(y_true_all_pf, y_pred_all_pf, class_number + 1, results_save_dir + f'/{model_name}_Multiclass_ROC_plot_Fold_{i}.png')
    # roc_plot.savefig(results_save_dir + f'/{model_name}_ROC_plot_Fold_{i}.png', dpi=600)
    # Generate Multiclass precision-recall curve figure
    prc_plot = plot_multiclass_precision_recall_curves(y_true_all_pf, y_pred_all_pf, class_number + 1, results_save_dir + f'/{model_name}_Multiclass_PRC_plot_Fold_{i}.png')
    # prc_plot.savefig(results_save_dir + f'/{model_name}_PRC_plot_Fold_{i}.png', dpi=600)
    Eval_Mat = []
    # Per class metrics
    for k in range(len(labels)):
      TN = cm_per_class_pf[k][0][0]
      FP = cm_per_class_pf[k][0][1]
      FN = cm_per_class_pf[k][1][0]
      TP = cm_per_class_pf[k][1][1]
      Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
      Precision = round(100*(TP)/(TP+FP), 2)
      Sensitivity = round(100*(TP)/(TP+FN), 2)
      F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
      Specificity = round(100*(TN)/(TN+FP), 2)
      DSC = round(100*((2*TP)/((2*TP)+FP+FN)), 2)
      IOU = round(100*(TP/(TP+FP+FN)), 2)
      Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity, DSC, IOU])
    # Sizes of each class
    s = np.sum(cm_pf, axis=1) 
    # Create temp excel table 
    headers = ['Accuracy', 'Precision', 'Sensitivity', 'F1-score', 'Specificity', 'DSC', 'IOU']
    temp_table = pd.DataFrame(Eval_Mat, index=labels, columns=headers)
    # Weighted average of per class metricies
    Accuracy = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
    Precision = round(temp_table['Precision'].dot(s)/np.sum(s), 2)
    Sensitivity = round(temp_table['Sensitivity'].dot(s)/np.sum(s), 2)
    F1_score = round(temp_table['F1-score'].dot(s)/np.sum(s), 2)
    Specificity = round(temp_table['Specificity'].dot(s)/np.sum(s), 2)
    DSC = round(temp_table['DSC'].dot(s)/np.sum(s), 2)
    IOU = round(temp_table['IOU'].dot(s)/np.sum(s), 2)
    values = [Accuracy, Precision, Sensitivity, F1_score, Specificity, DSC, IOU]
    # Create per class metricies excel table with weighted average row
    Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity, DSC, IOU])
    labels_wa = labels + ['Weighted Average']
    Eval_table = pd.DataFrame(Eval_Mat, index=labels_wa, columns=headers)
    # Create confusion matrix table (pd.DataFrame)
    Overall_Acc = pd.DataFrame(Overall_Accuracy, index=['Overall_Accuracy'] , columns=[' '])
    print('\n')
    print(f'Confusion Matrix Fold {i}')
    print('----------------------------')
    print(cm_table)
    print('\n')
    print(f'Normalized Confusion Matrix Fold {i}')
    print('----------------------------')
    print(cmn_table)
    print('\n')
    print(f'Evaluation Matrices Fold {i}')
    print('-----------------------------')
    print(Eval_table)
    print(Overall_Acc)
    print('\n')
    # Save to excel file   
    new_savepath = results_save_dir + f'/{model_name}_fold_{i}.xlsx'  # file to save 
    writer = pd.ExcelWriter(new_savepath, engine='openpyxl')
    # Sheet 1 (Evaluation metricies) + (Commulative Confusion Matrix) 
    col = 1
    row = 1
    Eval_table.to_excel(writer, "Results", startcol=col, startrow=row)
    row = row + 2 + len(labels_wa)
    Overall_Acc.to_excel(writer, "Results", startcol=col, startrow=row, header=None)
    col = col + 9
    row = 2
    True_Class = pd.DataFrame(['True Class'])
    True_Class.to_excel(writer, "Results", startcol=col + 1, startrow=row, header=None, index=None)
    col = col + 1
    row = 1
    Predicted_Class = pd.DataFrame(['Predicted Class'])
    Predicted_Class.to_excel(writer, "Results", startcol=col + 1, startrow=row, header=None, index=None)
    row = 2     
    cm_table.to_excel(writer, "Results", startcol=col, startrow=row)
    # save 
    writer.close()
    counter_pf = counter_pf + 1
    if counter_pf == 1:
      cm_all = cm_pf
      cm_per_class_all = cm_per_class_pf
      y_true_all = y_true_all_pf
      y_pred_all = y_pred_all_pf
    elif counter_pf > 1:
      cm_all = cm_all + cm_pf
      cm_per_class_all = cm_per_class_all + cm_per_class_pf
      y_true_all = np.concatenate((y_true_all, y_true_all_pf))
      y_pred_all = np.concatenate((y_pred_all, y_pred_all_pf))
results_save_dir = save_dir + f'/{model_name}'
# Normalized Confusion Metrics
cmn_all = cm_all.astype('float') / cm_all.sum(axis=1)[:, np.newaxis]
# Overall Accuracy
Overall_Accuracy = np.sum(np.diagonal(cm_all))/np.sum(cm_all)
Overall_Accuracy = round(Overall_Accuracy*100, 2)
# Create confusion matrix table (pd.DataFrame)
cm_table = pd.DataFrame(cm_all, index=labels, columns=labels)
cmn_table = pd.DataFrame(cmn_all, index=labels, columns=labels)
# Generate Confusion matrix figure
cm_plot = plot_conf_mat(y_true_all, y_pred_all, labels)
cm_plot.savefig(results_save_dir + f'/{model_name}_Confusion_Matrix_Overall.png', dpi=600)
# Generate Multiclass ROC curve figure
roc_plot = plot_multiclass_roc(y_true_all, y_pred_all, class_number + 1, results_save_dir + f'/{model_name}_Multiclass_ROC_plot_Overall.png')
# Generate Multiclass precision-recall curve figure
prc_plot = plot_multiclass_precision_recall_curves(y_true_all, y_pred_all, class_number + 1, results_save_dir + f'/{model_name}_Multiclass_PRC_plot_Overall.png')
Eval_Mat = []
# Per class metrics
for k in range(len(labels)):
  TN = cm_per_class_all[k][0][0]
  FP = cm_per_class_all[k][0][1]
  FN = cm_per_class_all[k][1][0]
  TP = cm_per_class_all[k][1][1]
  Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
  Precision = round(100*(TP)/(TP+FP), 2)
  Sensitivity = round(100*(TP)/(TP+FN), 2)
  F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
  Specificity = round(100*(TN)/(TN+FP), 2)
  DSC = round(100*((2*TP)/((2*TP)+FP+FN)), 2)
  IOU = round(100*(TP/(TP+FP+FN)), 2)
  Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity, DSC, IOU])
# Sizes of each class
s = np.sum(cm_all, axis=1) 
# Create temp excel table 
headers = ['Accuracy', 'Precision', 'Sensitivity', 'F1-score', 'Specificity', 'DSC', 'IOU']
temp_table = pd.DataFrame(Eval_Mat, index=labels, columns=headers)
# Weighted average of per class metricies
Accuracy = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
Precision = round(temp_table['Precision'].dot(s)/np.sum(s), 2)
Sensitivity = round(temp_table['Sensitivity'].dot(s)/np.sum(s), 2)
F1_score = round(temp_table['F1-score'].dot(s)/np.sum(s), 2)
Specificity = round(temp_table['Specificity'].dot(s)/np.sum(s), 2)
DSC = round(temp_table['DSC'].dot(s)/np.sum(s), 2)
IOU = round(temp_table['IOU'].dot(s)/np.sum(s), 2)
values = [Accuracy, Precision, Sensitivity, F1_score, Specificity, DSC, IOU]
# Create per class metricies excel table with weighted average row
Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity, DSC, IOU])
labels_wa = labels + ['Weighted Average']
Eval_table = pd.DataFrame(Eval_Mat, index=labels_wa, columns=headers)
# Create confusion matrix table (pd.DataFrame)
Overall_Acc = pd.DataFrame(Overall_Accuracy, index=['Overall_Accuracy'] , columns=[' '])
print('\n')
print(f'Cumulative Confusion Matrix')
print('-------------------------------------')
print(cm_table)
print('\n')
print(f'Cumulative Confusion Matrix (Normalized)')
print('----------------------------')
print(cmn_table)
print('\n')
print(f'Evaluation Matrices (Overall)')
print('-----------------------------')
print(Eval_table)
print(Overall_Acc)
print('\n')
# Save to excel file   
new_savepath = results_save_dir + f'/{model_name}_Overall.xlsx'  # file to save 
writer = pd.ExcelWriter(new_savepath, engine='openpyxl')
# Sheet 1 (Evaluation metricies) + (Commulative Confusion Matrix) 
col = 1
row = 1
Eval_table.to_excel(writer, "Results", startcol=col, startrow=row)
row = row + 2 + len(labels_wa)
Overall_Acc.to_excel(writer, "Results", startcol=col, startrow=row, header=None)
col = col + 9
row = 2
True_Class = pd.DataFrame(['True Class'])
True_Class.to_excel(writer, "Results", startcol=col + 1, startrow=row, header=None, index=None)
col = col + 1
row = 1
Predicted_Class = pd.DataFrame(['Predicted Class'])
Predicted_Class.to_excel(writer, "Results", startcol=col + 1, startrow=row, header=None, index=None)
row = 2     
cm_table.to_excel(writer, "Results", startcol=col, startrow=row)
# save 
writer.close()
