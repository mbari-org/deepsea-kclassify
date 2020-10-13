#!/usr/bin/env bash
if [ "$#" -ne 1 ] ; then
        echo "$0: exactly 1 argument expected CPU or GPU"
        exit 2
fi
if [ $1 != "CPU" ]; then
    # GPU recommended
    docker build --build-arg TF_VERSION=1.15.0rc1-gpu-py3 --build-arg DOCKER_GID=`id -u`  --build-arg DOCKER_UID=`id -g` -t mbari/deepsea-gpu-kclassify .
else
    # To build for CPU only
    docker build --build-arg TF_VERSION=1.15.0rc1-py3 --build-arg DOCKER_GID=`id -u`  --build-arg DOCKER_UID=`id -g`  -t mbari/deepsea-cpu-kclassify .
fi
