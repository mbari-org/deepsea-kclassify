# !/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2019, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Utilities for file upload/download

@author: __author__
@status: __status__
@license: __license__
'''
import tarfile
import shutil
import boto3
from botocore.client import Config
import botocore
import os
import time 
from urllib.parse import urlparse
import tempfile

def unpack(out_dir, tar_file):
    if os.path.isfile(tar_file) and 'tar.gz' in tar_file and 's3' not in tar_file:
        print('Unpacking {}'.format(tar_file))
        tar = tarfile.open(tar_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        tar.extractall(path=out_dir)
        tar.close()
    elif 'tar.gz' in tar_file and 's3' in tar_file:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # download first then untar
        target_dir = tempfile.mkdtemp()
        download_s3(endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'], source_bucket=tar_file,
                    target_dir=target_dir)
        os.listdir(target_dir)
        t = os.path.join(target_dir, os.path.basename(tar_file))
        print('Unpacking {}'.format(t))
        tar = tarfile.open(t)
        tar.extractall(path=out_dir)
        tar.close()
        shutil.rmtree(target_dir)
    else:
        raise ('{} invalid'.format(tar_file))

def check_s3(bucket_name, endpoint_url=None):
    '''
    Check bucket by creating the s3 bucket - this will either create or return the existing bucket
    :param endpoint_url: endpoint for the s3 service; for minio use only
    :param bucket_name: name of the bucket to check e.g. s3://test
    :return:
    '''
    # check and create bucket if it doesn't exist
    urlp = urlparse(bucket_name)
    bucket_name = urlp.netloc
    print('Creating bucket {}'.format(bucket_name))
    if endpoint_url:
        s3 = boto3.resource('s3', endpoint_url=endpoint_url)
    else:
        s3 = boto3.resource('s3')

    try:
        s3.create_bucket(Bucket=bucket_name)
    except botocore.exceptions.ClientError as e:
        print('Bucket already created or other error')

def download_s3(endpoint_url, source_bucket, target_dir):
    try:
        urlp = urlparse(source_bucket)
        bucket_name = urlp.netloc
        object_name = urlp.path.split('/')[-1]
        print(f'Downloading from bucket {source_bucket} endpoint_url {endpoint_url} to {target_dir}')
        s3 = boto3.client('s3', endpoint_url=endpoint_url, connect_timeout=60, read_timeout=120)
        ntrys = 0
        while ntrys < 3:
            try:
                s3.download_file(bucket_name, object_name, os.path.join(target_dir, object_name))
                break
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("The object does not exist.")
                print(e)
                ntrys += 1
                time.sleep(10)
    except Exception as e:
        raise (e)

def upload_s3(target_bucket, target_file, endpoint_url=None):
    urlp = urlparse(target_bucket)
    print(urlp)
    bucket_name = urlp.netloc
    print(f'Uploading {target_file} to bucket {target_bucket} using endpoint_url {endpoint_url}')
    if endpoint_url:
        s3 = boto3.resource('s3', config=Config(signature_version='s3v4'),endpoint_url=endpoint_url)
    else:
        s3 = boto3.resource('s3')
    try:
        with open(target_file, 'rb') as data:
              s3.Bucket(bucket_name).put_object(Key=os.path.basename(target_file), Body=data)
    except botocore.exceptions.ClientError as e:
        print('failed')#print(e)
