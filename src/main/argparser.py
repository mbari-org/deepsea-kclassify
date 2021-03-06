#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Argument parser

@author: __author__
@status: __status__
@license: __license__
'''

import os
import argparse
import sys
import conf as model_conf
from argparse import RawTextHelpFormatter


class ArgParser:

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        examples = 'Examples:' + '\n\n'
        examples += sys.argv[0] + " --train_tar s3://127.0.0.1:9000/mydata/trainimages.tar.gz" \
                                  " --val_tar s3://127.0.0.1:9000/mydata/testimages.tar.gz"
        self.parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                              description='Run transfer learning on folder of images organized by label',
                                              epilog=examples)
        self.parser.add_argument("--train_tar", help="Path to training compressed data", required=True)
        self.parser.add_argument("--val_tar", help="Path to validation compressed data", required=False)
        self.parser.add_argument("--base_model", choices=model_conf.MODEL_DICT.keys(), default='vgg16',
                                 help='Enter the network you want as your base feature extractor')
        self.parser.add_argument("--batch_size", default=32, type=int,
                                 help='Enter the batch size that must be used to train')
        self.parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
        self.parser.add_argument('--l2_weight_decay_alpha', type=float, default=0.0, help='weight decay if using l2 '
                                                                                          'regularlization to reduce '
                                                                                          'overfitting (default: '
                                                                                          '0.0 which disabled)')
        self.parser.add_argument('--horizontal_flip', type=self.boolean_string, default=False,
                                 help='add horizontal flip augmentation')
        self.parser.add_argument('--vertical_flip', type=self.boolean_string, default=False,
                                 help='add vertical flip augmentation')
        self.parser.add_argument('--early_stop', type=self.boolean_string, default=False,
                                 help='apply early stopping to model')
        self.parser.add_argument('--normalize', type=self.boolean_string, default=True,
                                 help='apply featurewise normalize and center')
        self.parser.add_argument('--augment_range', type=float, default=0.0, help='range '
                                                                                  'between 0-1 to apply width, shift, and zoom augmentation during training')
        self.parser.add_argument('--k', type=int, default=5, help='1-5 batch interval for look-ahead')
        self.parser.add_argument('--shear_range', type=float, default=0.0, help='range '
                                                                                'between 0-1 to apply shear augmentation during training')
        self.parser.add_argument("--epochs", help="Number of epochs for training", nargs='?', action='store', default=1,
                                 type=int)
        self.parser.add_argument("--loss",
                                 help="Loss function for the gradients categorical_crossentropy, val_binary_accuracy, or categorical_focal_loss",
                                 nargs='?', action='store',
                                 default='categorical_crossentropy', type=str)
        self.parser.add_argument("--optimizer", help="optimizer: adam, radam, ranger", default='radam')
        self.parser.add_argument("--notes", help="Notes for the experiment", nargs='?', action='store', default='',
                                 type=str)
        self.parser.add_argument("--verbose", help="Verbose output", nargs='?', action='store', default=0, type=int)

    def parse_args(self):
        self.args = self.parser.parse_args()
        return self.args

    def boolean_string(self, s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    def summary(self):
        print("normalize:", self.args.normalize)
        print("early_stop:", self.args.early_stop)
        print("notes:", self.args.notes)
        print("base_model:", self.args.base_model)
        print("horizontal_flip:", self.args.horizontal_flip)
        print("vertical_flip:", self.args.vertical_flip)
        print("augment_range", self.args.augment_range)
        print("shear_range", self.args.shear_range)
        print("loss:", self.args.loss)
        print("l2_weight_decay_alpha:", self.args.l2_weight_decay_alpha)
        print("optimizer:", self.args.optimizer)
        print("look-ahead batch:", self.args.k)
        print("learning rate:", self.args.lr)
        print("batch_size:", self.args.batch_size)
        print("epochs:", self.args.epochs)
        print("train_tar:", self.args.train_tar)
        if self.args.val_tar:
            print("val_tar:", self.args.val_tar)


if __name__ == '__main__':
    parser = ArgParser()
    args = parser.parse_args()
    print(args)
