#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

MLFlow call back module. This implements a custom python function that includes image preprocessing then classification.
Any modifications to the inference use of the classifier should be done here.

@author: __author__
@status: __status__
@license: __license__
'''

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print('Adding {} to path'.format(parentdir))
import base64
import numpy as np
import os
import pandas as pd
import yaml
import tensorflow as tf
from radam_optimizer import RAdam
from lookahead import Lookahead
import mlflow
import mlflow.keras
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.file_utils import TempDir

MAX_BATCH = 100

class KerasImageClassifierPyfunc(object):
    """
    Image classification model with embedded pre-processing.

    This class is essentially an MLflow custom python function wrapper around a Tensorflow Keras model.
    The wrapper provides image preprocessing so that the model can be applied to images directly.
    The input to the model is base64 encoded image binary data (e.g. contents of a jpeg file).
    The output is the predicted class label, predicted class id followed by probabilities for each
    class.

    The model declares current local versions Tensorflow and pillow as dependencies in its
    conda environment file.
    """

    def __init__(self, graph, session, model, image_dims, image_mean, image_std, normalize, labels):
        self._graph = graph
        self._session = session
        self._model = model
        self._image_dims = image_dims
        self._image_mean = image_mean
        self._image_std = image_std
        self._normalize = normalize
        self._labels = labels
        probs_names = ["p({})".format(x) for x in labels]
        self._column_names = ["predicted_label", "predicted_label_id"] + probs_names

    def predict(self, input):
        """
        Generate predictions for the data.

        :param input: pandas.DataFrame with one column containing images to be scored. The image
                     column must contain base64 encoded binary content of the image files.

        :return: pandas.DataFrame containing predictions with the following schema:
                     Predicted class: string,
                     Predicted class index: int,
                     Probability(class==0): float,
                     ...,
                     Probability(class==N): float,
        """
        probs = self._predict_images(input)
        m, n = probs.shape
        label_idx = np.argmax(probs, axis=1)
        labels = np.array([self._labels[i] for i in label_idx], dtype=np.str).reshape(m, 1)
        output_data = np.concatenate((labels, label_idx.reshape(m, 1), probs), axis=1)
        res = pd.DataFrame(columns=self._column_names, data=output_data)
        res.index = input.index
        return res

    def _predict_images(self, input):
        """
        Generate predictions for input images.
        :param input: binary image data
        :return: predicted probabilities for each class
        """
        def decode_resize(bytes_data):
            image = tf.image.decode_jpeg(bytes_data[0], channels=3)
            image = tf.image.resize(image, self._image_dims)
            if self._normalize:
                image_mean = tf.reshape(self._image_mean, [1, 1, 3])
                image_std = tf.reshape(self._image_std, [1, 1, 3])
                image -= image_mean # normalize color with mean/std from training images
                image /= (image_std + 1.e-9)
                image /= 255.0  # normalize to [0,1] range
            return image

        def decode_img(x):
            p = None
            try:
                p = pd.Series(base64.decodebytes(bytearray(x[0])))
            except Exception as ex:
                p = pd.Series(base64.decodebytes(bytearray(x[0], encoding="utf8")))
            finally:
                return p

        images = input.apply(axis=1, func=decode_img)
        print('Predicting ' + str(len(images)) + ' images...')
        max_batch = min(images.shape[0], MAX_BATCH)

        with self._graph.as_default():
            with self._session.as_default():
                data = tf.constant(images.values)
                dataset = tf.data.Dataset.from_tensor_slices((data))
                dataset = dataset.map(decode_resize).batch(max_batch)
                iterator = dataset.make_one_shot_iterator()
                next_image_batch = iterator.get_next()
                return self._model.predict(next_image_batch, steps=1)


def log_model(normalize, train_output, artifact_path):
    """
    Log a KerasImageClassifierPyfunc model as an MLflow artifact for the current run.

    :param normalize: true if featurewise center and normalize on predict
    :param train_output: output from the training model
    :param artifact_path: Run-relative artifact path this model is to be saved to.
    """
    keras_model = train_output.model #Keras model to be saved.
    image_dims = (train_output.image_size, train_output.image_size)

    with TempDir() as tmp:
        data_path = tmp.path("image_model")
        os.mkdir(data_path)
        if normalize:
            normalize_str = "True"
        else:
            normalize_str = "False"
            train_output.image_mean = [-1, -1, -1]
            train_output.image_std = [-1, -1, -1]

        conf = {
            "image_dims": 'x'.join(map(str, image_dims)), #image dimensions the Keras model expects.
            "image_mean": ",".join(map(str, train_output.image_mean)),#image mean of training images
            "image_std": ",".join(map(str, train_output.image_std)),#image standard deviation of training images
            "normalize": normalize_str#true if featurewise centering and normalize
        }
        # labels for the classes this model can predict with integer id the model outputs
        df = pd.DataFrame.from_dict(train_output.labels, orient="index", columns=['id'])
        df.index.name = 'class_name'
        df.to_csv(os.path.join(data_path, "labels.csv"))

        with open(os.path.join(data_path, "conf.yaml"), "w") as f:
            yaml.safe_dump(conf, stream=f)

        conda_env = tmp.path("conda_env.yaml")
        with open(conda_env, "w") as f:
            f.write(conda_env_template.format(python_version=PYTHON_VERSION,
                                              tf_name=tf.__name__,  # can have optional -gpu suffix
                                              tf_version=tf.__version__,
                                              pillow_version=PIL.__version__))

        mlflow.keras.save_model(keras_model, path= os.path.join(data_path, "keras_model"),
                                custom_objects={'RAdam':RAdam, 'Lookahead': Lookahead})

        mlflow.pyfunc.log_model(artifact_path=artifact_path,
                                loader_module=__name__,
                                code_path=[__file__,
                                           os.path.join(currentdir, 'train.py'),
                                           os.path.join(currentdir, 'radam_optimizer.py'),
                                           os.path.join(currentdir, 'lookahead.py')],
                                data_path=data_path,
                                conda_env=conda_env)

def _load_pyfunc(path):
    """
    Load the KerasImageClassifierPyfunc model.
    """
    with open(os.path.join(path, "conf.yaml"), "r") as f:
        conf = yaml.safe_load(f)
    # Get list of class names sorted by their id
    with open(os.path.join(path, "labels.csv"), "r") as f:
        labels = pd.read_csv(f).sort_values(by='id')['class_name'].values.tolist()

    if 'False' in conf["normalize"]:
        normalize = False
    else:
        normalize = True
    keras_model_path = os.path.join(path, "keras_model")
    image_dims = np.array([np.int32(x) for x in conf["image_dims"].split("x")])
    image_mean = eval(conf["image_mean"].replace(' ', ','))[0]
    str = conf["image_std"].replace('  ', ',')
    image_std = eval(str.replace(' ', ','))[0]
    # NOTE: TensorFlow based models depend on global state (Graph and Session) given by the context.
    with tf.Graph().as_default() as g:
        with tf.Session().as_default() as sess:
            keras_model = mlflow.keras.load_model(keras_model_path)
    return KerasImageClassifierPyfunc(g, sess, keras_model, image_dims, image_mean, image_std, normalize,
                                      labels=labels)


conda_env_template = """        
name: kerasclassifier
channels:
  - defaults
  - anaconda
dependencies:
  - python=={python_version}
  - tensorflow-gpu=={tf_version} 
  - pip:    
    - pillow=={pillow_version}
"""
