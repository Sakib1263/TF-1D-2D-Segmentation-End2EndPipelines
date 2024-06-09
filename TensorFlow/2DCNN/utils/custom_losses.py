import tensorflow as tf
from keras import backend as K

class DiceLoss(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
  def call(self, y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    dice = 1 - ((2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth))

    return dice


class BCEDiceLoss(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
  def call(self, y_true, y_pred, smooth=1e-6):
    BCE = tf.keras.metrics.binary_crossentropy(y_true, y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)   
    dice_loss = 1 - ((2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth))
    Dice_BCE = BCE + dice_loss

    return Dice_BCE


class IoULoss(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
  def call(self, y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1) 
    total = K.sum(y_true) + K.sum(y_pred)
    union = total - intersection
    
    IoU = 1 - ((intersection + smooth) / (union + smooth))

    return IoU


class FocalLoss(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
  def call(self, y_true, y_pred, alpha=0.8, gamma=2):
    BCE = tf.keras.metrics.binary_crossentropy(y_true, y_pred)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)

    return focal_loss
    