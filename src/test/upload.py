# !/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Setup bucket and upload data for training/testing in MLFlow Docker environment

@author: __author__
@status: __status__
@license: __license__
'''

import sys
import boto3
import os
import util
from botocore.client import Config
from dotenv import load_dotenv

test_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(test_dir, 'test.env'))

# setup client
endpoint_url = os.environ['MLFLOW_S3_ENDPOINT_URL']

s3 = boto3.client('s3',
                  endpoint_url=endpoint_url,
                  aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                  aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                  config=Config(signature_version='s3v4'),
                  region_name=os.environ['AWS_DEFAULT_REGION'])

print('Checking bucket to save to s3://test')
util.check_s3('s3://test', endpoint_url)

print('Uploading data to bucket')
util.upload_s3('s3://test', os.path.join(test_dir, '../../', 'data', 'catsdogs.tar.gz'), endpoint_url)
util.upload_s3('s3://test', os.path.join(test_dir, '../../', 'data', 'catsdogstrain.tar.gz'), endpoint_url)
util.upload_s3('s3://test', os.path.join(test_dir, '../../', 'data', 'catsdogsval.tar.gz'), endpoint_url)
