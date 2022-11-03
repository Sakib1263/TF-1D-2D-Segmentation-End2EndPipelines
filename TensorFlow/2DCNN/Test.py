import os
import gc
import PIL
import cv2
import numpy as np
import configparser
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from Helper_Functions import *
from PIL import Image
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

np.seterr(invalid='ignore')

"""[CONFIGURATIONS]"""
'''General Configurations'''
config_file = configparser.ConfigParser()
config_file.read('Test Configurations.ini')
test_dir = config_file["TEST"]["test_dir"]
imheight = int(config_file["TEST"]["imheight"])
imwidth = int(config_file["TEST"]["imwidth"])
color_mode = config_file["TEST"]["color_mode"]
num_channel = int(config_file["TEST"]["num_channel"])
num_class = int(config_file["TEST"]["num_class"])
labels = config_file["TEST"]["labels"]
'''Model Configurations'''
encoder_name = config_file["TEST"]["encoder_name"]
decoder_name = config_file["TEST"]["decoder_name"]
'''Test Configurations'''
batch_size = int(config_file["TEST"]["batch_size"])
normalizing_factor = float(config_file["TEST"]["normalizing_factor"])
start_fold = int(config_file["TEST"]["start_fold"])
end_fold = int(config_file["TEST"]["end_fold"])
num_iter = int(config_file["TEST"]["num_iter"])
threshold = float(config_file["TEST"]["threshold"])
seed = int(config_file["TEST"]["seed"])

config_file = open("Test Configurations.ini", "r")
content = config_file.read()
print("Current Configurations:\n")
print(content)
config_file.flush()
config_file.close()

if (labels == "") or (labels == "[]") or (labels == []) or (labels == None):
  labels = []
  for i in range(0,(num_class+1)):
    labels.append(str(i))

print(f'Labels: {labels}')

test_image_dir = []
test_mask_dir = []
save_dir = []

# Main Testing Loop
for i in range(start_fold, end_fold):
    overall_accuracy = []
    overall_precision = []
    overall_recall = []
    overall_f1score = []
    overall_jaccardscore = []
    overall_dicescore = []
    y_true_all = []
    y_pred_all = []
    test_image_dir = test_dir + f'/Fold_{i}/Images'
    test_mask_dir = test_dir + f'/Fold_{i}/Masks'
    model_name = encoder_name + "_" + decoder_name
    results_save_dir = 'Results' + '/' + model_name
    if not os.path.exists(results_save_dir):
      os.makedirs(results_save_dir)
    predicted_mask_save_dir = f"Data/Test/Fold_{i}/Predictions"
    if not os.path.exists(predicted_mask_save_dir):
      os.makedirs(predicted_mask_save_dir)
    global_counter = 0
    #
    imgs = os.listdir(test_image_dir)  # List containing all Images
    msks = os.listdir(test_mask_dir)  # List containing all Masks
    Nimgs = len(imgs)  # Number of Images
    Nmsks = len(msks)  # Number of Masks
    assert Nimgs == Nmsks, 'Number of Images and corresponding Masks are not equal'
    num_batches = int(np.ceil(Nimgs/batch_size))
    # print(Nimgs)
    # Load Pre-trained Weights
    if (os.path.exists('Results/'+model_name+'/'+model_name+'_'+str(imwidth)+f'_Fold_{i}.h5')):
        print('Loading Trained Model...')
        model = None
        gc.collect()
        model = tf.keras.models.load_model('Results/'+model_name+'/'+model_name+'_'+str(imwidth)+f'_Fold_{i}.h5')
    else:
        raise ValueError("Requested trained model is not present in the provided directory")
    #
    print('Making Predictions...')
    for ii in range(0,num_batches):
      print(f'Batch Number {ii+1} out of {num_batches}')
      counter = 0
      img_batch = np.empty((batch_size, imheight, imwidth, num_channel))
      msk_batch = np.empty((batch_size, imheight, imwidth))
      pred_dir_all = []
      for iii in range(ii*batch_size,(ii+1)*batch_size):
        if (iii >= Nimgs):
          break
        counter = counter+1
        global_counter = global_counter+1
        img_dir = test_image_dir+'/'+imgs[iii]
        msk_dir = test_mask_dir+'/'+msks[iii]
        pred_dir = predicted_mask_save_dir+'/Predicted_'+imgs[iii]
        pred_dir_all.append(pred_dir)
        # print(img_dir)
        img = PIL.Image.open(img_dir)
        img = np.asarray(img)/normalizing_factor
        img_shape = img.shape
        if len(img_shape) < 3:
          img = np.expand_dims(img, axis=2)
        img_batch[counter-1,:,:,:] = img
        # Corresponding Ground Truth Mask
        msk = PIL.Image.open(msk_dir)
        msk = np.asarray(msk)/(normalizing_factor/num_class)
        msk_shape = msk.shape
        if len(msk_shape) == 3:
          msk = msk[:,:,0]
        elif len(msk_shape) == 4:
          msk = msk[:,:,0,0]
        msk_batch[counter-1,:,:] = msk
      msk_batch = np.reshape(msk_batch, (batch_size, imheight, imwidth, 1))
      img_batch = img_batch[0:counter,:,:,:]
      msk_batch = msk_batch[0:counter,:,:,:]
      Predictions = np.array(model.predict(img_batch, verbose=0))
      # msk_batch = np.where(msk_batch > threshold, 1, 0)
      # Predictions = np.where(Predictions > threshold, 1, 0)
      # Predictions_shape = Predictions.shape
      #
      for j in range(0,counter):
        predicted_image = np.squeeze(Predictions[j,:,:,:])
        predicted_image = predicted_image.astype(np.uint8)
        image_data_from_array = PIL.Image.fromarray(predicted_image*255)
        image_data_from_array.save(pred_dir_all[j])
      y_true = np.asarray(msk_batch.ravel())
      y_pred = np.asarray(Predictions.ravel())
      y_true = y_true.astype(int)
      y_pred = y_pred.astype(int)
      y_true_all.append(y_true)
      y_pred_all.append(y_pred)
      # Evaluation Metrics
      conf_mat = confusion_matrix(y_true, y_pred, sample_weight=None, normalize=None)
      if ii == 0:
        conf_mat_all = conf_mat
      else:
        conf_mat_all = conf_mat_all+conf_mat
      #
      FP = conf_mat.sum(axis=0) - np.diag(conf_mat)  # False Positive
      FN = conf_mat.sum(axis=1) - np.diag(conf_mat)  # False Negative
      TP = np.diag(conf_mat)  # True Positive
      TN = conf_mat.sum() - (FP + FN + TP)  # True Negative
      Accuracy = np.mean((TP+TN)/(TP+TN+FP+FN))
      overall_accuracy.append(Accuracy*counter)
      # Precision = np.mean(TP/(TP+FP))
      Precision = precision_score(y_true, y_pred, average='weighted')
      overall_precision.append(Precision*counter)
      # Recall = np.mean(TP/(TP+FN))
      Recall = recall_score(y_true, y_pred, average='weighted')
      overall_recall.append(Recall*counter)
      # F1_Score = np.mean(TP/(TP+0.5*(FP+FN)))
      F1_Score = f1_score(y_true, y_pred, average='weighted')
      overall_f1score.append(F1_Score*counter)
      JaccardScore = jaccard_score(y_true, y_pred, average='weighted')
      overall_jaccardscore.append(JaccardScore*counter)
      DiceSimilarityScore = dice(y_true, y_pred)
      overall_dicescore.append(DiceSimilarityScore*counter)
      # Garbage Collector
      Predictions = None
      y_true = None
      y_pred = None
      conf_mat = None
      gc.collect()
    #
    overall_accuracy = np.sum(overall_accuracy)/global_counter
    overall_precision = np.sum(overall_precision)/global_counter
    overall_recall = np.sum(overall_recall)/global_counter
    overall_f1score = np.sum(overall_f1score)/global_counter
    overall_jaccardscore = np.sum(overall_jaccardscore)/global_counter
    overall_dicescore = np.sum(overall_dicescore)/global_counter
    #
    print(f'Overall Accuracy: {overall_accuracy:.3f}')
    print(f'Weighted Precision: {overall_precision:.3f}')
    print(f'Weighted Sensitivity or Recall: {overall_recall:.3f}')
    print(f'Weighted F1-Score: {overall_f1score:.3f}')
    print(f'Weighted IoU or Jaccard-Score: {overall_jaccardscore:.3f}')
    print(f'Dice Similarity Score: {overall_dicescore:.3f}')
    print('------------------------------------------------')
    # Print and Save Confusion Matrix
    print('Pixel-wise Raw Confusion Matrix:')
    print(conf_mat_all)
    cmn = conf_mat_all.astype('float') / conf_mat_all.sum(axis=1)[:, np.newaxis]
    print('Pixel-wise Normalized Confusion Matrix:')
    print(cmn)
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_true_all = y_true_all.ravel()
    y_pred_all = y_pred_all.ravel()
    # y_true_all_OHE = one_hot_encoding(y_true_all)
    # y_pred_all_OHE = one_hot_encoding(y_pred_all)
    plot_conf_mat(y_true_all, y_pred_all, labels, results_save_dir)
    plot_multiclass_roc(y_true_all, y_pred_all, num_class, results_save_dir)
    plot_multiclass_precision_recall_curves(y_true_all, y_pred_all, num_class, results_save_dir)
    #