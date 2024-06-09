import numpy as np
import tensorflow as tf


# Operational Layers.
class Oper2D(tf.keras.Model):
  def __init__(self, filters, kernel_size, padding='same', strides=(1, 1), activation=None, q=1):
    super(Oper2D, self).__init__()
    self.activation = activation
    self.q = q
    self.all_layers = []
    for i in range(0, q):  # q convolutional layers.
      self.all_layers.append(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=None, name=f'ONN_Conv_{i+1}'))
  @tf.function
  def call(self, input_tensor, training=False):
    x = self.all_layers[0](input_tensor)  # First convolutional layer.
    if self.q > 1:
      for i in range(1, self.q):
        x += self.all_layers[i](tf.math.pow(input_tensor, i+1, name=f"tf_math_pow{i}"))
    if self.activation is not None:
      # return eval('tf.nn.' + self.activation + '(x)')
      x = tf.keras.layers.Activation(self.activation, name=f"activation_func_{self.activation}")(x)
      return x
    else:
      return x


# Transposed Operational Layers.
class Oper2DTranspose(tf.keras.Model):
  def __init__(self, filters, kernel_size, strides=(1, 1), padding='same', activation=None, q=1):
    super(Oper2DTranspose, self).__init__()
    self.activation = activation
    self.q = q
    self.all_layers = []
    for i in range(0, q):  # q convolutional layers.
      self.all_layers.append(tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, activation=None, name=f'ONN_TransConv_{i+1}'))
  @tf.function
  def call(self, input_tensor, training=False):
    x = self.all_layers[0](input_tensor)  # First convolutional layer.
    if self.q > 1:
      for i in range(1, self.q):
        x += self.all_layers[i](tf.math.pow(input_tensor, i+1, name=f"tf_math_pow_transposed_{i}"))
    if self.activation is not None:
      # return eval('tf.nn.' + self.activation + '(x)')
      x = tf.keras.layers.Activation(self.activation, name=f"activation_func_{self.activation}")(x)
      return x
    else:
      return x
      