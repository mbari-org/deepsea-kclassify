#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Calculates accuracy, precision and recall *during* training.
This is not provided in the Keras framework.

@author: __author__
@status: __status__
@license: __license__
'''

import tensorflow.keras
import numpy as np
import sklearn.metrics
import threading
import wandb
import mlflow

class Metrics(tensorflow.keras.callbacks.Callback):

    def __init__(self, val_data, batch_size, labels):
        self.labels = labels
        self.epoch_count = 0
        self.validation_data = val_data
        self.batch_size = batch_size
        self.val_count = 0
        self.trim_start = 0
        self.trim_end = 0

    def on_epoch_end(self, epoch, logs={}):
        if 'categorical_accuracy' in logs.keys():
            mlflow.log_metric(key='categorical_accuracy', value=logs['categorical_accuracy'], step=epoch)
            mlflow.log_metric(key='val_categorical_accuracy', value=logs['val_categorical_accuracy'], step=epoch)
        else:
            mlflow.log_metric(key='binary_accuracy', value=logs['binary_accuracy'], step=epoch)
            mlflow.log_metric(key='val_binary_accuracy', value=logs['val_binary_accuracy'], step=epoch)
        mlflow.log_metric(key='loss',value=logs.get('loss'), step=epoch)
        mlflow.log_metric(key='val_loss',value=logs.get('val_loss'), step=epoch)
        self.epoch_count += 1
        batches = len(self.validation_data)
        total = batches * self.batch_size
        val_predict = np.zeros(total)
        val_true = np.zeros(total)
        class_map = {}
        label_index = 0
        for label in self.labels:
            class_map[label_index] = label
            label_index += 1
        re_map = {label:index for index, label in class_map.items()}
        for batch in range(batches):
            thread1 = threading.Thread(target=self.parse_batch(batch, val_true, val_predict))
            thread1.start()
            thread1.join()

        val_predict = np.delete(val_predict, np.s_[self.trim_start:self.trim_end])
        val_true = np.delete(val_true, np.s_[self.trim_start:self.trim_end])
        self.val_count = 0
        label_predict = [class_map[i] for i in val_predict]
        label_true = [class_map[i] for i in val_true]

        # possibly use different average param
        _val_f1 = sklearn.metrics.f1_score(label_true, label_predict, labels=self.labels, average=None)
        _val_recall = sklearn.metrics.recall_score(label_true, label_predict, labels=self.labels, average=None)
        _val_precision = sklearn.metrics.precision_score(label_true, label_predict, labels=self.labels, average=None)
        for label in self.labels:
            f1_log = {label+'_f1':_val_f1[re_map[label]]}
            precision_log = {label+'_precision':_val_precision[re_map[label]]}
            recall_log = {label+'_recall': _val_recall[re_map[label]]}
            wandb.log(f1_log, step=epoch, commit=False)
            wandb.log(precision_log, step=epoch, commit=False)
            wandb.log(recall_log, step=epoch, commit=False)
            mlflow.log_metric(key=label+'_f1', value=_val_f1[re_map[label]], step=epoch)
            mlflow.log_metric(key=label+'_precision', value=_val_precision[re_map[label]], step=epoch)
            mlflow.log_metric(key=label+'_recall', value=_val_recall[re_map[label]], step=epoch)
        return

    def parse_batch(self, batch, val_true, val_predict):
        try:
            xVal, yVal = next(self.validation_data)
        except StopIteration:
            return

        for i in range(self.batch_size):
            try:
                val_predict[batch * self.batch_size + i] = np.asarray(self.model.predict_classes(xVal))[i]
                val_true[batch * self.batch_size + i] = np.argmax(yVal, axis=1)[i]
                self.val_count += 1
            except IndexError:
                self.trim_start = self.val_count
                self.trim_end = self.trim_start + self.batch_size - i
                return
