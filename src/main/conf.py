#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Configuration file for model specifications. 
These models are the ones supported in this code.

@author: __author__
@status: __status__
@license: __license__
'''

MODEL_DICT = {}

densenet201 = dict(
    image_size = 224,
    model_instance = "tf.keras.applications.densenet.DenseNet201",
    fine_tune_at = -1,
    slim = False,
    has_depthwise_layers = False
)
inceptionv3 = dict(
    image_size = 299,
    model_instance = "tf.keras.applications.inception_v3.InceptionV3",
    fine_tune_at = -1,
    slim = False,
    has_depthwise_layers = False
)
inception_resnetv2 = dict(
    image_size = 299,
    model_instance = "tf.keras.applications.inception_resnet_v2.InceptionResNetV2",
    fine_tune_at = -1,
    slim = False,
    has_depthwise_layers = False
)
xception = dict(
    image_size = 299,
    model_instance="tf.keras.applications.xception.Xception",
    fine_tune_at = -1,
    slim = False,
    has_depthwise_layers = False
)
nasnetlarge = dict(
    image_size = 331,
    model_instance="tf.keras.applications.nasnet.NASNetLarge",
    fine_tune_at = -1,
    slim = False,
    has_depthwise_layers = False
)
resnet50 = dict(
    image_size = 224,
    model_instance="tf.keras.applications.ResNet50",
    fine_tune_at = -1, # freeze up to Block2 and slim the remainder of the network
    slim = False,
    has_depthwise_layers = False
)
vgg16slim = dict(
    image_size = 224,
    model_instance="tf.keras.applications.VGG16",
    fine_tune_at = 4,
    slim = True,
    has_depthwise_layers = False
)
vgg16 = dict(
    image_size = 224,
    model_instance="tf.keras.applications.VGG16",
    fine_tune_at = 4,
    slim = False,
    has_depthwise_layers = False
)
vgg19 = dict(
    image_size = 224,
    model_instance="tf.keras.applications.VGG19",
    fine_tune_at = 4,
    slim = False,
    has_depthwise_layers = False
)
mobilenetv2 = dict(
    image_size = 224,
    model_instance = "tf.keras.applications.MobileNetV2",
    model_url  = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2",
    feature_extractor_url="https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2",
    fine_tune_at = 100,
    slim = False,
    has_depthwise_layers = True
)
MODEL_DICT["densenet201"] = densenet201
MODEL_DICT["xception"] = xception
MODEL_DICT["inceptionv3"] = inceptionv3
MODEL_DICT["inception_resnetv2"] = inception_resnetv2
MODEL_DICT["nasnetlarge"] = nasnetlarge
MODEL_DICT["resnet50"] = resnet50
MODEL_DICT["mobilenetv2"] = mobilenetv2
MODEL_DICT["vgg16"] = vgg16
MODEL_DICT["vgg16slim"] = vgg16slim
MODEL_DICT["vgg19"] = vgg19
