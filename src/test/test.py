#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Python test of Keras classifier using the nose python library

@author: __author__
@status: __status__
@license: __license__
'''
import subprocess
import docker
import os
import time
import boto3
from botocore.client import Config
import botocore
import util
from nose import with_setup
from subprocess import Popen
from dotenv import load_dotenv

print("")  # this is to get a newline after the dots
# directory this code is in
test_dir = os.path.dirname(os.path.abspath(__file__))
# this loads in all environmental and configuration variables
#load_dotenv(dotenv_path=os.path.join(test_dir,'test.env'))
load_dotenv(dotenv_path='/src/test/test.env')

print(os.environ)

def monitor(container):
    '''
    Monitor running container and print output
    :param container:
    :return:
    '''
    container.reload()
    l = ""
    while True:
        for line in container.logs(stream=True):
            l = line.strip().decode()
            print(l)
        else:
            break
    return l

def teardown_module(module):
    '''
    Run after everything in this file completes
    :param module:
    :return:
    '''
    print('teardown_module')

def custom_setup_function():
    endpoint_url = os.environ['MLFLOW_S3_ENDPOINT_URL']
    # util.check_s3('s3://test', endpoint_url)
    #
    # print('Uploading data to s3://test')
    # util.upload_s3('s3://test','/data/catsdogs.tar.gz',endpoint_url)
    # util.upload_s3('s3://test','/data/catsdogstrain.tar.gz',endpoint_url)
    # util.upload_s3('s3://test','/data/catsdogsval.tar.gz',endpoint_url)


def custom_teardown_function():
    print('custom_teardown_function')


@with_setup(custom_setup_function, custom_teardown_function)
def test_train():

    print('<============================ running test_train ============================ >')
    try:
        # Create experiment called test and put results in the bucket s3://test
        print('Creating experiment test and storing results in s3://test bucket')
        subprocess.check_output('mlflow experiments create -n test -l s3://test', shell=True)
        print('Running experiment...')
        subprocess.check_output('mlflow run --experiment-name test -P train_tar=s3://test/catsdogstrain.tar.gz '
                                '-P val_tar=s3://test/catsdogsval.tar.gz .',shell=True)
    except Exception as ex:
        raise(ex)

    # record = os.path.join(os.getcwd(), 'train.record')
    #exists = os.path.exists(record)
    #MLFLOW_TRACKING_URI = 'http: // 127.0.0.1: 5000'
    #MLFLOW_S3_ENDPOINT_URL = 'http://127.0.0.1: 9000'
    #s = 'foobar'
    #assert s == 'Image mean [122.73600502 140.90251093  95.58092464] normalized [0.48131767 0.55255887 0.37482716]'
    #assert exists
