import gc
import pickle
import configparser
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from Helper_Functions import *

'''Basic Configurations'''
config_file = configparser.ConfigParser()
config_file.read('Inference Configurations.ini')
imlength = int(config_file["INFERENCE"]["imlength"])
imwidth = int(config_file["INFERENCE"]["imwidth"])
color_mode = config_file["INFERENCE"]["color_mode"]
class_number = int(config_file["INFERENCE"]["class_number"])
batch_size = int(config_file["INFERENCE"]["batch_size"])
learning_rate = float(config_file["INFERENCE"]["learning_rate"])
model_name = config_file["INFERENCE"]["model_name"]
normalizing_factor = float(config_file["INFERENCE"]["normalizing_factor"])
start_fold = int(config_file["INFERENCE"]["start_fold"])
end_fold = int(config_file["INFERENCE"]["end_fold"])
num_iter = int(config_file["INFERENCE"]["num_iter"])

# Initialize Variables
Y_Test_Total_ = []
Y_Test_Total = []
Predictions_Total = []
Y_Preds_Total = []
class_names = []
# Main Inference Loop
for i in range(start_fold, end_fold):
    # Import Test Dataset using Image Data Generator
    print(f'\nFold {i}\n')
    save_path = 'Results/' + model_name + '/' + model_name + '_' + str(imwidth) + '_Fold_' + str(i) + '.h5'
    print(f'Loading Pretrained Model...')
    Model = tf.keras.models.load_model(save_path)
    # Compile Model
    if class_number == 2:  # For fine-tuning, the learning rate should be very low (here '1e-5') for slow change
        Model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=[tf.keras.metrics.MeanSquaredError()])
    elif class_number > 2:
        Model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=[tf.keras.metrics.MeanSquaredError()])
    if num_iter == 1:
        # Print Model Summary
        Model.summary()
        print('\n')
    # Import Test Data
    print('In the Test Set: ')
    test_datagen = ImageDataGenerator(rescale=1./normalizing_factor)
    test_ds = test_datagen.flow_from_directory('Data/Test/fold_'+str(i), color_mode=color_mode, target_size=(imlength, imwidth), interpolation='bicubic',
                                                                    batch_size=batch_size, class_mode='categorical', shuffle=False) # Do not Shuffle while Testing
    Y_Test_ = np.expand_dims(test_ds.labels, axis=1) # Ground Truth (GT) Labels for the Test Set
    Y_Test = one_hot_encoding(Y_Test_.ravel()) # Perform One_Hot_Encoding on the GT Labels to get a Multi-Array
    file_names = test_ds.filenames # Filenames in the Test Directory
    # Class Names along with Respective Indices
    class_names = test_ds.class_indices
    Class_Number = len(class_names) # Number of Classes in the Dataset
    print(f'\nClass Names with Indices: {class_names}\n')
    # Test and Predict (Print Results)
    print('Predicting...')
    Predictions = np.array(Model.predict(test_ds, verbose=1))
    Y_Preds = reverse_one_hot_encoding(Predictions)
    Error = mean_absolute_error(Y_Test, Predictions)
    print(f"\nMean Absolute Error (MAE): {Error}")
    # Print and Save Raw and Normalized Confusion Matrices
    print('\nRaw Confusion Matrix:')
    print(confusion_matrix(Y_Test_, Y_Preds, normalize=None))
    print('\nNormalized Confusion Matrix:')
    print(confusion_matrix(Y_Test_, Y_Preds, normalize='true'))
    # Print and Save Evaluation Parameters
    print('\nEvaluation Metrics:')
    Accuracy = accuracy_score(Y_Test_, Y_Preds)
    print(f'Accuracy: {Accuracy:.3f}')
    Precision = precision_score(Y_Test_, Y_Preds, average= 'weighted')
    print(f'Precision: {Precision:.3f}')
    Recall = recall_score(Y_Test_, Y_Preds, average= 'weighted')
    print(f'Recall (Sensitivity): {Recall:.3f}')
    f1_Score = f1_score(Y_Test_, Y_Preds, average= 'weighted')
    print(f'F1-Score: {f1_Score:.3f}')
    # Print and Save Classification Report
    print('\nClassification Report:')
    print(classification_report(Y_Test_, Y_Preds, target_names=class_names, zero_division=0))
    # Print and Saved Misclassified Cases with respective Prediction Weights
    Missed_Cases = misclassifications(Y_Test_, Y_Preds, Predictions, class_names, file_names, i)
    with open('Results/'+model_name+'/'+model_name+'_'+f'Missed_Cases_Fold_{i}.txt', 'w') as txtfile:
        for ii in range(len(Missed_Cases)):
            txtfile.write(str(Missed_Cases[ii]))
            txtfile.write('\n')
    # Create Total Arrays
    if (num_iter == 1):
        Y_Test_Total_ = Y_Test_
        Y_Test_Total = Y_Test
        Predictions_Total = Predictions
        Y_Preds_Total = Y_Preds
    elif (num_iter > 1):
        Y_Test_Total_ = np.concatenate((Y_Test_Total_, Y_Test_), axis=0)
        Y_Test_Total = np.concatenate((Y_Test_Total, Y_Test), axis=0)
        Predictions_Total = np.concatenate((Predictions_Total, Predictions), axis=0)
        Y_Preds_Total = np.concatenate((Y_Preds_Total, Y_Preds), axis=0)
    history_save_path = 'Results/' + model_name + '/' + model_name + '_' + str(imwidth) + '_Fold_' + str(i) + '_history.pickle'
    #with open(history_save_path, 'rb') as file:
    #    history = pickle.load(file)
    num_iter = num_iter + 1
    # Garbage Collector
    Model = None  # Delect any existing Model from the Memory to avoid Reuse in the next iteration
    test_ds = None
    Predictions = None
    gc.collect()
    print('=======================================================================================')
# Combined Error from all Folds
Overall_Error = mean_absolute_error(Y_Test_Total, Predictions_Total)
print(f'Overall Error (MAE): {Overall_Error}')
Overall_Accuracy = accuracy_score(Y_Test_Total_, Y_Preds_Total)
print(f'Overall_Accuracy: {Overall_Accuracy:.3f}')
# Save Final ROC Curve
plot_multiclass_roc(Y_Test_Total, Predictions_Total, class_number, model_name)
print('\n=======================================================================================\n')
# Save Final Precision-Recall Curve
plot_multiclass_precision_recall_curves(Y_Test_Total, Predictions_Total, class_number, model_name)
print('\n=======================================================================================\n')
# Save Final Confusion Matrix
plot_conf_mat(Y_Test_Total_, Y_Preds_Total, class_names, model_name)
