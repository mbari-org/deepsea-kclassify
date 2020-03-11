#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Python test Keras classifier using the nose python framework

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
print('Setting up nose client')
client = docker.from_env()
mlflow_proc = None

# this loads in all environmental and configuration variables
load_dotenv(dotenv_path=os.path.join(os.getcwd(), 'src', 'test', 'test.env'))

def run_mlflowminio():
    '''
     Run mlflow/minio stack
    :return:
    '''
    mlflow_proc = subprocess.Popen(
         'cd src/test && nose-compose up -d ',
         shell=True)

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
    client.volumes.prune()
    if mlflow_proc:
        mlflow_proc.kill()


def custom_setup_function():
    run_mlflowminio()

    # delay for minio/mlflow stack startup
    #time.sleep(30)

    endpoint_url = os.environ['MLFLOW_S3_ENDPOINT_URL']
    print('Checking bucket to save to s3://testdata')
    # util.check_s3('s3://testdata', endpoint_url)
    #
    # print('Uploading data to s3://testdata')
    # util.upload_s3('s3://testdata', '/data/catsdogs.tar.gz', endpoint_url)
    # util.upload_s3('s3://testdata', '/data/catsdogstrain.tar.gz', endpoint_url)
    # util.upload_s3('s3://testdata', '/data/catsdogsval.tar.gz', endpoint_url)


def custom_teardown_function():
    print('custom_teardown_function')


@with_setup(custom_setup_function, custom_teardown_function)
def test_kclassify():

    print('<============================ running test detect ============================ >')
    # c = client.containers.run('tfdetect', command,
    #                           volumes={
    #                               os.path.join(os.getcwd(), 'data'): {'bind': '/', 'mode': 'rw'}
    #                           },
    #                           detach=True, auto_remove=False, stdout=True, stderr=True)
    # s = monitor(c)
    # record = os.path.join(os.getcwd(), 'train.record')
    #exists = os.path.exists(record)
    #MLFLOW_TRACKING_URI = 'http: // 127.0.0.1: 5000'
    #MLFLOW_S3_ENDPOINT_URL = 'http://127.0.0.1: 9000'
    s = 'foobar'
    assert s == 'Image mean [122.73600502 140.90251093  95.58092464] normalized [0.48131767 0.55255887 0.37482716]'
    #assert exists
