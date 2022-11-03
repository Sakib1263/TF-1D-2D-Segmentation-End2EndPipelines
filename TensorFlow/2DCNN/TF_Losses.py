import tensorflow as tf


class TFLosses:
    def __init__(self, loss_function_name):
        self.loss_function_name = loss_function_name

    def loss(self):
        if self.loss_function_name == "BinaryCrossentropy":
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0, axis=-1, name='binary_crossentropy')
        elif self.loss_function_name == "BinaryFocalCrossentropy":
            loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False, label_smoothing=0.0, axis=-1, name='binary_focal_crossentropy')
        elif self.loss_function_name == "CategoricalCrossentropy":
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.0, axis=-1, name='categorical_crossentropy')
        elif self.loss_function_name == "CategoricalHinge":
            loss = tf.keras.losses.CategoricalHinge(name='categorical_hinge')
        elif self.loss_function_name == "CosineSimilarity":
            loss = tf.keras.losses.CosineSimilarity(axis=-1, name='cosine_similarity')
        elif self.loss_function_name == "Hinge":
            loss = tf.keras.losses.Hinge(name='hinge')
        elif self.loss_function_name == "Huber":
            loss = tf.keras.losses.Huber(delta=1.0, name='huber_loss')
        elif self.loss_function_name == "KLDivergence":
            loss = tf.keras.losses.KLDivergence(name='kl_divergence')
        elif self.loss_function_name == "LogCosh":
            loss = tf.keras.losses.LogCosh(name='log_cosh')
        elif self.loss_function_name == "MeanAbsoluteError":
            loss = tf.keras.losses.MeanAbsoluteError(name='mean_absolute_error')
        elif self.loss_function_name == "MeanAbsolutePercentageError":
            loss = tf.keras.losses.MeanAbsolutePercentageError(name='mean_absolute_percentage_error')
        elif self.loss_function_name == "MeanSquaredError":
            loss = tf.keras.losses.MeanSquaredError(name='mean_squared_error')
        elif self.loss_function_name == "MeanSquaredLogarithmicError":
            loss = tf.keras.losses.MeanSquaredLogarithmicError(name='mean_squared_logarithmic_error')
        elif self.loss_function_name == "Poisson":
            loss = tf.keras.losses.Poisson(name='poisson')
        elif self.loss_function_name == "SparseCategoricalCrossentropy":
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, name='sparse_categorical_crossentropy')
        elif self.loss_function_name == "SquaredHinge":
            loss = tf.keras.losses.SquaredHinge(name='squared_hinge')
        else:
            raise ValueError("Please select a valid loss function. Check for spelling mistakes, capital/small letters, etc.")

        return loss
