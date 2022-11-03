import tensorflow as tf


class TFOptimizers:
    def __init__(self, optimizer_function_name, learning_rate):
        self.optimizer_function_name = optimizer_function_name
        self.learning_rate = learning_rate

    def optimizer(self):
        if self.optimizer_function_name == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
        elif self.optimizer_function_name == "Adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=self.learning_rate, rho=0.95, epsilon=1e-07, name='Adadelta')
        elif self.optimizer_function_name == "Adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate, initial_accumulator_value=0.1, epsilon=1e-07, name='Adagrad')
        elif self.optimizer_function_name == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax')
        elif self.optimizer_function_name == "FTRL":
            optimizer = tf.keras.optimizers.Ftrl(learning_rate=self.learning_rate, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0,
                                            l2_regularization_strength=0.0,name='Ftrl', l2_shrinkage_regularization_strength=0.0, beta=0.0)
        elif self.optimizer_function_name == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
        elif self.optimizer_function_name == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop')
        elif self.optimizer_function_name == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.0, nesterov=False, name='SGD')
        else:
            raise ValueError("Please select a valid optimizer. Check for spelling mistakes, capital/small letters, etc.")

        return optimizer
