# !/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Utilities for bucket and file management

@author: __author__
@status: __status__
@license: __license__
'''

import os
import tarfile
import shutil
import tempfile
from urllib.parse import urlparse
import boto3
from botocore.client import Config
import botocore


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
                    target_dir=target_dir, object_name=os.path.basename(tar_file))
        t = os.path.join(target_dir, os.path.basename(tar_file))
        print('Unpacking {}'.format(t))
        tar = tarfile.open(t)
        tar.extractall(path=out_dir)
        tar.close()
        shutil.rmtree(target_dir)
    else:
        raise ('{} invalid'.format(tar_file))


def check_s3(bucket_name, endpoint_url=None):
    """
    Check bucket by creating the s3 bucket - this will either create or return the existing bucket
    :param endpoint_url: endpoint for the s3 service; for minio use only
    :param bucket_name: name of the bucket to check
    :return:
    """
    env = os.environ.copy()
    # check and create bucket if it doesn't exist
    bucket_bname = bucket_name.split('s3://')[-1]
    print('Creating bucket {}'.format(bucket_bname))
    if endpoint_url:
        s3 = boto3.resource('s3',
                            endpoint_url=endpoint_url,
                            aws_access_key_id=env['AWS_ACCESS_KEY_ID'],
                            aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'],
                            config=Config(signature_version='s3v4'),
                            region_name=env['AWS_DEFAULT_REGION'])
    else:
        s3 = boto3.resource('s3',
                            aws_access_key_id=env['AWS_ACCESS_KEY_ID'],
                            aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'],
                            config=Config(signature_version='s3v4'),
                            region_name=env['AWS_DEFAULT_REGION'])

    try:
        s3.create_bucket(Bucket=bucket_bname)
    except botocore.exceptions.ClientError as e:
        print(e)


def download_s3(endpoint_url, source_bucket, target_dir, object_name):
    try:
        env = os.environ.copy()
        urlp = urlparse(source_bucket)
        bucket_name = urlp.netloc
        print('Downloading {} bucket: {} using {} endpoint_url {}'.format(source_bucket, bucket_name, target_dir,
                                                                          endpoint_url))
        s3 = boto3.resource('s3',
                            endpoint_url=endpoint_url,
                            aws_access_key_id=env['AWS_ACCESS_KEY_ID'],
                            aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'],
                            config=Config(signature_version='s3v4', connect_timeout=5, read_timeout=5),
                            region_name='us-east-1')
        try:
            bucket = s3.Bucket(bucket_name)
            for s3_object in bucket.objects.all():
                path, filename = os.path.split(s3_object.key)
                if filename in urlp.path:
                    bucket.download_file(s3_object.key, os.path.join(target_dir, filename))
                    return
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            print(e)
    except Exception as e:
        raise (e)
