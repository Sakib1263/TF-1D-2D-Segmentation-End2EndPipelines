import tensorflow as tf


class TFMetrics:
    def __init__(self, metrics_function_name, num_classes=2, target_class_ids=None, k=5, num_thresholds=200, at_param="recall"):
        self.metrics_function_name = metrics_function_name
        self.numclasses = num_classes
        if target_class_ids is None:
          target_class_ids = []
          for i in range(0,(self.numclasses+1)):
              target_class_ids.append(i)
        self.target_class_IDs = target_class_ids
        self.k = k
        self.num_thresholds = num_thresholds
        self.at_param = at_param

    def metric(self):
        if self.metrics_function_name == "AUC":
            metric = tf.keras.metrics.AUC(num_thresholds=self.num_thresholds, curve='ROC', summation_method='interpolation', name=None, dtype=None, thresholds=None, multi_label=False,
                                          num_labels=None, label_weights=None, from_logits=False)
        elif self.metrics_function_name == "Accuracy":
            metric = tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
        elif self.metrics_function_name == "BinaryAccuracy":
            metric = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5)
        elif self.metrics_function_name == "BinaryCrossentropy":
            metric = tf.keras.metrics.BinaryCrossentropy(name='binary_crossentropy', dtype=None, from_logits=False, label_smoothing=0)
        elif self.metrics_function_name == "BinaryIoU":
            metric = tf.keras.metrics.BinaryIoU(target_class_ids=self.target_class_IDs, threshold=0.5, name=None, dtype=None)
        elif self.metrics_function_name == "CategoricalAccuracy":
            metric = tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)
        elif self.metrics_function_name == "CategoricalCrossentropy":
            metric = tf.keras.metrics.CategoricalCrossentropy(name='categorical_crossentropy', dtype=None, from_logits=False, label_smoothing=0)
        elif self.metrics_function_name == "CategoricalHinge":
            metric = tf.keras.metrics.CategoricalHinge(name='categorical_hinge', dtype=None)
        elif self.metrics_function_name == "CosineSimilarity":
            metric = tf.keras.metrics.CosineSimilarity(name='cosine_similarity', dtype=None, axis=-1)
        elif self.metrics_function_name == "Hinge":
            metric = tf.keras.metrics.Hinge(name='hinge', dtype=None)
        elif self.metrics_function_name == "IoU":
            metric = tf.keras.metrics.IoU(num_classes=self.numclasses, target_class_ids=self.target_class_IDs)
        elif self.metrics_function_name == "KLDivergence":
            metric = tf.keras.metrics.KLDivergence(name='kullback_leibler_divergence', dtype=None)
        elif self.metrics_function_name == "LogCoshError":
            metric = tf.keras.metrics.LogCoshError(name='logcosh', dtype=None)
        elif self.metrics_function_name == "Mean":
            metric = tf.keras.metrics.Mean(name='mean', dtype=None)
        elif self.metrics_function_name == "MeanAbsoluteError":
            metric = tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error', dtype=None)
        elif self.metrics_function_name == "MeanAbsolutePercentageError":
            metric = tf.keras.metrics.MeanAbsolutePercentageError(name='mean_absolute_percentage_error', dtype=None)
        elif self.metrics_function_name == "MeanIoU":
            metric = tf.keras.metrics.MeanIoU(num_classes=self.numclasses, name=None, dtype=None)
        elif self.metrics_function_name == "MeanSquaredError":
            metric = tf.keras.metrics.MeanSquaredError(name='mean_squared_error', dtype=None)
        elif self.metrics_function_name == "MeanSquaredLogarithmicError":
            metric = tf.keras.metrics.MeanSquaredLogarithmicError(name='mean_squared_logarithmic_error', dtype=None)
        elif self.metrics_function_name == "OneHotIoU":
            metric = tf.keras.metrics.OneHotIoU(num_classes=self.numclasses, target_class_ids=self.target_class_IDs, name=None, dtype=None)
        elif self.metrics_function_name == "OneHotMeanIoU":
            metric = tf.keras.metrics.OneHotMeanIoU(num_classes=self.numclasses, name=None, dtype=None)
        elif self.metrics_function_name == "Poisson":
            metric = tf.keras.metrics.Poisson(name='poisson', dtype=None)
        elif self.metrics_function_name == "Precision":
            metric = tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)
        elif self.metrics_function_name == "Recall":
            metric = tf.keras.metrics.Recall(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)
        elif self.metrics_function_name == "RootMeanSquaredError":
            metric = tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error', dtype=None)
        elif self.metrics_function_name == "SparseCategoricalAccuracy":
            metric = tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy', dtype=None)
        elif self.metrics_function_name == "SparseCategoricalCrossentropy":
            metric = tf.keras.metrics.SparseCategoricalCrossentropy(name='sparse_categorical_crossentropy', dtype=None, from_logits=False, axis=-1)
        elif self.metrics_function_name == "SparseTopKCategoricalAccuracy":
            metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=self.k, name='sparse_top_k_categorical_accuracy', dtype=None)
        elif self.metrics_function_name == "SquaredHinge":
            metric = tf.keras.metrics.SquaredHinge(name='squared_hinge', dtype=None)
        elif self.metrics_function_name == "Sum":
            metric = tf.keras.metrics.Sum(name='sum', dtype=None)
        elif self.metrics_function_name == "TopKCategoricalAccuracy":
            metric = tf.keras.metrics.TopKCategoricalAccuracy(k=self.k, name='top_k_categorical_accuracy', dtype=None)
        elif self.metrics_function_name == "tf.keras.metrics.TrueNegatives":
            metric = tf.keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None)
        elif self.metrics_function_name == "tf.keras.metrics.TruePositives":
            metric = tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None)
        elif self.metrics_function_name == "tf.keras.metrics.FalseNegatives":
            metric = tf.keras.metrics.FalseNegatives(thresholds=None, name=None, dtype=None)
        elif self.metrics_function_name == "tf.keras.metrics.FalsePositives":
            metric = tf.keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None)
        elif self.metrics_function_name == "tf.keras.metrics.PrecisionAtRecall":
            metric = tf.keras.metrics.PrecisionAtRecall(self.at_param, num_thresholds=self.num_thresholds, class_id=None, name=None, dtype=None)
        elif self.metrics_function_name == "tf.keras.metrics.RecallAtPrecision":
            metric = tf.keras.metrics.RecallAtPrecision(self.at_param, num_thresholds=self.num_thresholds, class_id=None, name=None, dtype=None)
        elif self.metrics_function_name == "tf.keras.metrics.SensitivityAtSpecificity":
            metric = tf.keras.metrics.SensitivityAtSpecificity(self.at_param, num_thresholds=self.num_thresholds, class_id=None, name=None, dtype=None)
        elif self.metrics_function_name == "tf.keras.metrics.SpecificityAtSensitivity":
            metric = tf.keras.metrics.SpecificityAtSensitivity(self.at_param, num_thresholds=self.num_thresholds, class_id=None, name=None, dtype=None)
        else:
            raise ValueError("Please select a valid metric. Check for spelling mistakes, capital/small letters, etc.")

        return metric
