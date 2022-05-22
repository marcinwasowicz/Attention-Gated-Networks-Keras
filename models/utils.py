import tensorflow as tf
from tensorflow.keras import backend as K


def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true = K.cast(y_true, y_pred.dtype)
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    dice = (2 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return 1 - dice
