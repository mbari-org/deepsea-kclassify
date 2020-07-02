#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

TensorFlow Keras classifier with integrated MLFlow and Wandb logging

@author: __author__
@status: __status__
@license: __license__
'''

import signal
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print('Adding {} to path'.format(parentdir))
import tensorflow as tf
import tempfile
import mlflow
import wandb
import numpy as np
from lookahead import Lookahead
from wandb.keras import WandbCallback
from metrics import Metrics
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from transfer_model import TransferModel
from sklearn.metrics import classification_report, confusion_matrix
from stopping import Stopping
from argparser import ArgParser
from radam_optimizer import RAdam
import plot
import utils
import time
import shutil
from focal_loss import focal_loss
from image_pyfunc import log_model
import glob
import uuid
from dotenv import load_dotenv

has_wandb = True


def sigterm_handler(signal, frame):
    print('Run got SIGTERM')


class TrainOutput():

    def __init__(self, model, train, image_size, labels, class_size, history, image_mean,
                 image_std, best_epoch, y_test, y_pred):
        self.model = model
        self.train = train
        self.image_size = image_size
        self.labels = labels
        self.class_size = class_size
        self.history = history
        self.image_mean = image_mean
        self.image_std = image_std
        self.best_epoch = best_epoch
        self.y_test = y_test
        self.y_pred = y_pred


class Train:

    def __init__(self, **kwargs):
        self.pyfunc_params = kwargs
        return

    def compile_and_fit_model(self, labels, model, fine_tune_at, train_generator, validation_generator,
                              epochs, batch_size, loss, optimizer, lr,
                              metric_type=tf.keras.metrics.categorical_accuracy,
                              early_stop=False, output_dir='/tmp'):

        print('Writing TensorFlow events locally to {}'.format(output_dir))
        tensorboard = TensorBoard(log_dir=output_dir)

        steps_per_epoch = train_generator.n // batch_size
        validation_steps = validation_generator.n // batch_size

        # Un-freeze the top layers of the model
        model.trainable = True

        # if fine tune at defined, freeze all the layers before the `fine_tune_at` layer
        if fine_tune_at > 0:
            for layer in model.layers[:fine_tune_at]:
                layer.trainable = False

        opt_loss = loss
        if loss == 'categorical_focal_loss':
            opt_loss = focal_loss(alpha=1)

        if optimizer == 'radam':
            model.compile(loss=opt_loss,
                          optimizer=RAdam(lr=lr),
                          metrics=[metric_type])
        elif optimizer == 'ranger':
            model.compile(loss=opt_loss,
                          optimizer=RAdam(lr=lr),
                          metrics=[metric_type])
            # Added lookahead optimizer to speed-up training and reduce
            # hyper parameter sweeps https://arxiv.org/abs/1907.08610
            lookahead = Lookahead(k=args.k, alpha=0.5)
            lookahead.inject(model)
        elif optimizer == 'adam':
            model.compile(loss=opt_loss,
                          optimizer=tf.keras.optimizers.Adam(lr=lr),
                          metrics=[metric_type])
        else:
            model.compile(loss=opt_loss,
                          optimizer=tf.keras.optimizers.SGD(lr=lr),
                          metrics=[metric_type])

        if loss == 'categorical_crossentropy' or loss == 'categorical_focal_loss':
            monitor = 'val_categorical_accuracy'
        else:
            monitor = 'val_binary_accuracy'

        early = Stopping(monitor=monitor, patience=2, verbose=1, restore_best_weights=True)
        checkpoint_path = '{}/checkpoints.best.h5'.format(output_dir)
        checkpoint = ModelCheckpoint(checkpoint_path, monitor=monitor, verbose=1, save_best_only=True, mode='max')
        callbacks = [tensorboard, checkpoint]
        if has_wandb:
            metrics = Metrics(labels=list(labels.keys()), val_data=validation_generator, batch_size=batch_size)
            wandb = WandbCallback(save_model=False, data_type="image", validation_data=validation_generator,
                                  labels=list(labels.keys()))
            callbacks += [metrics, wandb]
        if early_stop:
            callbacks += [early]

        if os.path.exists(checkpoint_path):
            print('Loading model weights from {}'.format(checkpoint_path))
            model.load_weights(checkpoint_path)
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      use_multiprocessing=True,
                                      validation_data=validation_generator,
                                      validation_steps=validation_steps,
                                      callbacks=callbacks)
        model.load_weights(checkpoint_path)
        if early_stop:
            best_epoch = early.best_epoch
        else:
            best_epoch = min(0, len(history.history) - 1)
        return history, model, best_epoch

    def get_binary_loss(self, hist):
        loss = hist.history['loss']
        loss_val = loss[len(loss) - 1]
        return loss_val

    def get_binary_acc(self, hist):
        acc = hist.history['binary_accuracy']
        acc_value = acc[len(acc) - 1]
        return acc_value

    def get_validation_loss(self, hist):
        val_loss = hist.history['val_loss']
        val_loss_value = val_loss[len(val_loss) - 1]
        return val_loss_value

    def get_validation_acc(self, hist):
        print("keys {}".format(hist.history.keys()))
        if 'val_binary_accuracy' in hist.history.keys():
            val_acc = hist.history['val_binary_accuracy']
        else:
            val_acc = hist.history['val_categorical_accuracy']
        val_acc_value = val_acc[len(val_acc) - 1]
        return val_acc_value

    def print_metrics(self, hist):
        if 'val_binary_accuracy' in hist.history.keys():
            acc_value = self.get_binary_acc(hist)
            loss_value = self.get_binary_loss(hist)
            print("Final metrics: binary_loss: {:6.4f}".format(loss_value))
            print("Final metrics: binary_accuracy= {:6.4f}".format(acc_value))

        val_acc_value = self.get_validation_acc(hist)
        val_loss_value = self.get_validation_loss(hist)

        print("Final metrics: validation_loss: {:6.4f}".format(val_loss_value))
        print("Final metrics: validation_accuracy: {:6.4f}".format(val_acc_value))

    def train_model(self, args, output_dir):
        """
        Train the model and log all the metrics to wandb
        :param args: command line argument object
        :param output_dir: directory to store model output to
        """
        # Rescale all images by 1./255 and apply image augmentation if requested
        if args.val_tar:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                            width_shift_range=args.augment_range,
                                                                            height_shift_range=args.augment_range,
                                                                            zoom_range=args.augment_range,
                                                                            horizontal_flip=args.horizontal_flip,
                                                                            vertical_flip=args.vertical_flip,
                                                                            shear_range=args.shear_range,
                                                                            featurewise_center=args.normalize,
                                                                            featurewise_std_normalization=args.normalize)

            val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                          width_shift_range=args.augment_range,
                                                                          height_shift_range=args.augment_range,
                                                                          zoom_range=args.augment_range,
                                                                          horizontal_flip=args.horizontal_flip,
                                                                          vertical_flip=args.vertical_flip,
                                                                          shear_range=args.shear_range,
                                                                          featurewise_center=args.normalize,
                                                                          featurewise_std_normalization=args.normalize)
        else:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                            width_shift_range=args.augment_range,
                                                                            height_shift_range=args.augment_range,
                                                                            zoom_range=args.augment_range,
                                                                            horizontal_flip=args.horizontal_flip,
                                                                            vertical_flip=args.vertical_flip,
                                                                            shear_range=args.shear_range,
                                                                            validation_split=0.2,
                                                                            featurewise_center=args.normalize,
                                                                            featurewise_std_normalization=args.normalize)
            val_datagen = train_datagen

        train_dir = os.path.join(output_dir, 'train')
        if args.val_tar:
            val_dir = os.path.join(output_dir, 'val')
        else:
            val_dir = train_dir
        image_dir = os.path.join(output_dir, 'images')
        os.makedirs(image_dir)
        try:
            utils.unpack(train_dir, args.train_tar)
            if args.val_tar:
                utils.unpack(val_dir, args.val_tar)
        except Exception as ex:
            raise (ex)

        # check if depth is only 1 - here we assume data is nested in subdir
        def check_depth(d):
            if len(os.listdir(d)) == 1:
                return os.path.join(d, os.listdir(d)[0])
            return d

        print('Checking training directory depth')
        train_dir = check_depth(train_dir)
        val_dir = check_depth(val_dir)
        print(os.listdir(train_dir))
        print(os.listdir(val_dir))

        # Get label size and calculate mean/std from data_gen
        print('Fetching class labels')
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        gen = data_gen.flow_from_directory(train_dir)
        labels = gen.class_indices
        for label in labels:
            print('found label ' + label)
        class_size = len(labels)
        train_x, _ = gen.next()

        model, image_size, fine_tune_at = TransferModel(args.base_model).build(class_size, args.l2_weight_decay_alpha)

        if args.val_tar:
            # train/val flow from separate directories
            print('Training data:')
            training_generator = train_datagen.flow_from_directory(train_dir,
                                                                   target_size=(image_size, image_size),
                                                                   batch_size=args.batch_size,
                                                                   class_mode='categorical')

            print('Validation data:')
            validation_generator = train_datagen.flow_from_directory(val_dir,
                                                                     target_size=(image_size, image_size),
                                                                     batch_size=args.batch_size,
                                                                     class_mode='categorical')
        else:
            # split data into train/val from same directory
            print('Training data:')
            training_generator = train_datagen.flow_from_directory(train_dir,
                                                                   target_size=(image_size, image_size),
                                                                   batch_size=args.batch_size,
                                                                   class_mode='categorical',
                                                                   subset='training')

            print('Validation data:')
            validation_generator = train_datagen.flow_from_directory(train_dir,
                                                                     target_size=(image_size, image_size),
                                                                     batch_size=args.batch_size,
                                                                     class_mode='categorical',
                                                                     subset='validation')

        # compute quantities required for featurewise normalize
        # (std, mean, and principal components if ZCA whitening is applied)
        print('Calculating normalization statistics')
        train_datagen.fit(train_x)
        print('Normalize mean= {}, std= {}'.format(train_datagen.mean(), train_datagen.std()))
        if args.val_tar:
            val_datagen.fit(train_x)

        train = Train()

        model.summary()
        history, model, best_epoch = train.compile_and_fit_model(labels=labels, model=model, fine_tune_at=fine_tune_at,
                                                                 train_generator=training_generator, lr=args.lr,
                                                                 validation_generator=validation_generator,
                                                                 epochs=args.epochs, batch_size=args.batch_size,
                                                                 loss=args.loss, output_dir=output_dir,
                                                                 optimizer=args.optimizer,
                                                                 metric_type=tf.keras.metrics.categorical_accuracy,
                                                                 early_stop=args.early_stop)

        # load model for best epoch and compute data for PR/CM
        checkpoint_path = '/{}/checkpoints.best.h5'.format(output_dir)
        if os.path.exists(checkpoint_path):
            print('Loading model weights from {}'.format(checkpoint_path))
            model.load_weights(checkpoint_path)

        print('Running prediction on validation data...')
        pred = model.predict_generator(validation_generator, args.batch_size)
        y_pred = np.argmax(pred, axis=1)
        print('===========Confusion Matrix========')
        print(confusion_matrix(validation_generator.classes, y_pred))
        print('===========Classification==========')
        print(classification_report(validation_generator.classes, y_pred, target_names=labels))
        return TrainOutput(model, train, image_size, labels, class_size, history, train_datagen.mean(),
                           train_datagen.std(), best_epoch, validation_generator.classes, pred)


def log_params(params):
    if has_wandb:
        wandb.log(dict(params))
    mlflow.log_params(params)


def log_metrics(train_output, image_dir):
    train_output.train.print_metrics(train_output.history)
    p = plot.Plot()

    # create plot of the loss/accuracy for quick reference
    graph_image_loss_png = os.path.join(image_dir, 'loss.png')
    graph_image_acc_png = os.path.join(image_dir, 'accuracy.png')
    figure_loss = p.plot_loss_graph(train_output.history, 'Training and Validation Loss')
    figure_loss.savefig(graph_image_loss_png)
    figure_acc = p.plot_accuracy_graph(train_output.history, 'Training and Validation Accuracy')
    figure_acc.savefig(graph_image_acc_png)

    if 'val_categorical_accuracy' in train_output.history.history.keys():
        acc = train_output.history.history['val_categorical_accuracy']
        mlflow.log_metric("best_val_categorical_accuracy", acc[train_output.best_epoch])
        if has_wandb:
            wandb.config.update({"best_val_categorical_accuracy": acc[train_output.best_epoch]})
    else:
        acc = train_output.history.history['val_binary_accuracy']
        mlflow.log_metric("best_val_binary_accuracy", acc[train_output.best_epoch])
        if has_wandb:
            wandb.config.update({"best_val_binary_accuracy": acc[train_output.best_epoch]})

    if has_wandb:
        labels = np.array(list(train_output.labels.items())) # convert dict to array
        wandb.log({'roc': wandb.plots.ROC(train_output.y_test, train_output.y_pred, labels)})
        wandb.log({'pr': wandb.plots.precision_recall(train_output.y_test, train_output.y_pred, labels)})
        wandb.sklearn.plot_confusion_matrix(train_output.y_test, np.argmax(train_output.y_pred, axis=1), labels)


def log_artifacts(normalize, model_path, image_dir, model_artifacts):
    """
    Logs artifacts to the MLflow server
    :param normalize: true if featurewise normalize on predict
    :param model_path:  full path to directory with training output artifacts
    :param image_dir:  full path to directory with any image artifacts, e.g. plots
    :param model_artifacts:  full path to output directory to save
    :return: 
    """
    # log generated plots to images directory
    for figure_path in glob.iglob(image_dir + '**/*', recursive=True):
        mlflow.log_artifact(local_path=figure_path, artifact_path="images")
    # log model
    log_model(normalize, model_path, artifact_path="model")
    # write out TensorFlow events as a run artifact in the events directory
    print("Uploading TensorFlow events as a run artifact.")
    for artifact in glob.iglob(model_artifacts + '**/*.*', recursive=True):
        mlflow.log_artifact(local_path=artifact, artifact_path="events")


def setup_wandb():
    """
    Checks if wandb is configured according to environment variable keys, and if so initializes run
    :return: wandb run object
    """
    keys = ['WANDB_ENTITY', 'WANDB_USERNAME', 'WANDB_API_KEY', 'WANDB_PROJECT',
            'WANDB_GROUP', ]
    has_wandb_keys = True
    for key in keys:
        if key not in env.keys():
            print('Need to set ' + key)
            has_wandb_keys = False

    wandb_run = None
    if has_wandb_keys:
        wandb_run = wandb.init(notes=parser.args.notes, job_type='training', entity=os.environ['WANDB_ENTITY'],
                               project=os.environ['WANDB_PROJECT'], group=os.environ['WANDB_GROUP'])

        # adds all of the arguments as config variables
        wandb.config.update(parser.args)

    return wandb_run


if __name__ == '__main__':

    signal.signal(signal.SIGTERM, sigterm_handler)
    load_dotenv(dotenv_path=os.environ['ENV'])

    output_dir = None
    parser = ArgParser()
    args = parser.parse_args()

    env = os.environ.copy()
    required_keys = ['MLFLOW_S3_ENDPOINT_URL', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    for k in required_keys:
        if k not in env.keys():
            print('Need to set ' + k)
            exit(-1)

    try:
        # first check connection to mlflow
        print('Connecting to MlflowClient {}'.format(os.environ['MLFLOW_TRACKING_URI']))
        tracking_client = mlflow.tracking.MlflowClient()
        print('Connection succeeded')

        start_time = time.time()
        print("Using parameters")
        parser.summary()

        run = setup_wandb()
        if run:
            run_id = run.id
            has_wandb = True
        else:
            run_id = uuid.uuid4().hex
            has_wandb = False

        with tf.Session():
            with mlflow.start_run(run_name=run_id):
                mlflow.log_param('run_name', run_id)
                output_dir = tempfile.mkdtemp()
                train_output = Train().train_model(args, output_dir)
                if args.normalize:
                    normalize_str = "True"
                else:
                    normalize_str = "False"
                # log model and normalize parameters needed for inference
                params = {'image_size': "{}x{}".format(train_output.image_size, train_output.image_size),
                          "image_mean": ','.join(map(str, train_output.image_mean.tolist())),
                          "image_std": ','.join(map(str, train_output.image_std.tolist())),
                          "normalize": normalize_str}

                log_params(params)
                log_metrics(train_output, os.path.join(output_dir, 'images'))
                log_artifacts(args.normalize, train_output, os.path.join(output_dir, 'images'), output_dir)
    except Exception as ex:
        print('Model training failed ' + str(ex))
        exit(-1)
    finally:
        if output_dir:
            shutil.rmtree(output_dir)

    runtime = time.time() - start_time
    print('Model complete. Total runtime {}'.format(runtime))
    exit(0)
