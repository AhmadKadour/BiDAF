# -*- coding: utf-8 -*-

from keras import backend as K
import tensorflow as tf


def accuracy(y_true, y_pred):

    def calculate_accuracy(true_and_pred):
        y_true, y_pred_start, y_pred_end = true_and_pred

        start_probability = y_pred_start[K.cast(y_true[0], dtype='int32')]
        end_probability = y_pred_end[K.cast(y_true[1], dtype='int32')]
        return (start_probability + end_probability) / 2.0

    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]
    accuracy = tf.map_fn(calculate_accuracy, (y_true, y_pred_start, y_pred_end), fn_output_signature='float32')
    return K.mean(accuracy, axis=0)