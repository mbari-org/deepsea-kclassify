#!/usr/bin/env bash
if [ "$#" -ne 1 ] ; then
        echo "$0: exactly 1 argument expected CPU or GPU"
        exit 2
fi
if [ $1 != "CPU" ]; then
    # GPU recommended
    docker build --build-arg TF_VERSION=2.3.1-gpu -t mbari/deepsea-gpu-kclassify .
else
    # To build for CPU only
    docker build --build-arg TF_VERSION=2.3.1 -t mbari/deepsea-cpu-kclassify .
fi
