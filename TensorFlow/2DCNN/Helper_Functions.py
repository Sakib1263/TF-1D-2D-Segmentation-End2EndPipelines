import os
import PIL
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, auc, roc_curve


def one_hot_encoding(data):
    L_E = LabelEncoder()
    integer_encoded = L_E.fit_transform(data)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    one_hot_encoded_data = onehot_encoder.fit_transform(integer_encoded)

    return one_hot_encoded_data


def reverse_one_hot_encoding(Predictions):
    prediction_shape = Predictions.shape
    prediction_length = prediction_shape[0]
    Y_Preds = np.zeros((prediction_length, 1), dtype=int)
    #
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


def plot_history(history, model_name, Fold_Num):
    # list all dictionaries in history
    print('')
    print(f'History Keys: {history.history.keys()}')
    with open('Results/'+model_name+f'/History_Fold_{Fold_Num}.pickle', 'wb') as handle:  # saving the history of the model
        pickle.dump(history.history, handle)
    history_list = list(history.history.values())
    # summarize history for error
    plt.figure(figsize=(12, 10))
    plt.plot(history_list[1])
    plt.plot(history_list[3])
    plt.title('Model Error Plot')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.ylim([0.0, 2.0])
    plt.savefig('Results/'+model_name+f'/Error_History_Plot_Fold_{Fold_Num}.png')
    plt.close()
    # summarize history for loss
    plt.figure(figsize=(12, 10))
    plt.plot(history_list[0])
    plt.plot(history_list[2])
    plt.title('Model Loss Plot')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.ylim([0.0, 5.0])
    plt.savefig('Results/'+model_name+f'/Loss_History_Plot_Fold_{Fold_Num}.png')
    plt.close()


def plot_conf_mat(Ground_Truth_Labels, Predictions, labels, save_dir):
    confusion_matrix_raw = confusion_matrix(Ground_Truth_Labels, Predictions, normalize=None)
    confusion_matrix_norm = confusion_matrix(Ground_Truth_Labels, Predictions, normalize='true')
    shape = confusion_matrix_raw.shape
    data = np.asarray(confusion_matrix_raw, dtype=int)
    text = np.asarray(confusion_matrix_norm, dtype=float)
    annots = (np.asarray(["{0:.2f} ({1:.0f})".format(text, data) for text, data in zip(text.flatten(), data.flatten())])).reshape(shape[0], shape[1])
    plt.figure(figsize=(20, 10))
    sns.heatmap(confusion_matrix_norm, cmap='YlGnBu', annot=annots, fmt='', xticklabels=labels, yticklabels=labels, vmax=1)
    plt.title('Confusion Matrix', fontsize=25)
    plt.xlabel("Predicted", fontsize=15)
    plt.ylabel("Actual", fontsize=15)
    plt.savefig(save_dir+'/Confusion_Matrix.png')
    plt.show()
    plt.close()


def plot_multiclass_roc(Y_Test, Predictions, class_number, save_dir):
    # Compute ROC curve and Area Under Curve (AUC) for each class
    if class_number < 2:
      fpr, tpr, thresholds = roc_curve(Y_Test, Predictions)
      roc_auc = auc(fpr, tpr)
      mean_tpr = np.zeros_like(fpr)
      mean_tpr += np.interp(fpr, fpr, tpr)
      all_fpr = fpr
    elif class_number >= 2:
      fpr = dict()
      tpr = dict()
      roc_auc = dict()
      for i in range(class_number):
          fpr[i], tpr[i], _ = roc_curve(Y_Test[:, i], Predictions[:, i])
          roc_auc[i] = auc(fpr[i], tpr[i])

      # First aggregate all false positive rates
      all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_number)]))
      # Then interpolate all ROC curves at this points
      mean_tpr = np.zeros_like(all_fpr)
      for i in range(class_number):
          mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

      # Finally average it and compute AUC
      mean_tpr /= class_number

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
    plt.savefig(save_dir+'/Multiclass_ROC_Curve.png')
    plt.show()
    plt.close()


def plot_multiclass_precision_recall_curves(Y_Test, Predictions, class_number, save_dir):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(class_number-1):
        precision[i], recall[i], _ = precision_recall_curve(Y_Test[:, i], Predictions[:, i])
        average_precision[i] = average_precision_score(Y_Test[:, i], Predictions[:, i])

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
    plt.savefig(save_dir+'/Multiclass_Precision_Recall_Curve.png')
    plt.show()
    plt.close()


def get_datasets(imgs_dir, groundTruth_dir, height, width, channels, normalizing_factor_img, normalizing_factor_msk, class_number):
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
            img = PIL.Image.open(imgs_dir + '/' + files[i])
            img = np.asarray(img)/normalizing_factor_img
            imgs[i,:,:,:] = img
            # Corresponding Ground Truth
            groundTruth_name = files[i]
            # print ("Ground Truth Name: " + groundTruth_name)
            g_truth = PIL.Image.open(groundTruth_dir + '/' + groundTruth_name)
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


def prepareTrainDict(y, model_depth, length, width, model_name):
    def approximate(inp, w_len, length, width):
        op = np.zeros((len(inp), length // w_len, width // w_len))
        for i in range(0, length, w_len):
            for j in range(0, width, w_len):
                try:
                    op[:, i // w_len, j // w_len] = np.mean(inp[:, i:i + w_len, j:j + w_len])
                except Exception as e:
                    print(e)
                    print(i)

        return op

    out = {}
    Y_Train_dict = {}
    out['out'] = np.array(y)
    Y_Train_dict['out'] = out['out']
    for i in range(1, (model_depth + 1)):
        name = f'level{i}'
        if ((model_name == 'UNet') or (model_name == 'MultiResUNet') or (model_name == 'FPN')):
            out[name] = np.expand_dims(approximate(np.squeeze(y), 2 ** i, length, width), axis=3)
        elif ((model_name == 'UNetE') or (model_name == 'UNetP') or (model_name == 'UNetPP')):
            out[name] = np.expand_dims(approximate(np.squeeze(y), 2 ** 0, length, width), axis=3)
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