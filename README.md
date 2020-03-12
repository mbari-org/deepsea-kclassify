# avedac-kclassify #
This is a dockerized implementation of the Keras (Keras Tensorflow, not pure Keras) 
image classifier for use in ML workflows with the MLFlow framework.
It trains images using Tensorflow Keras, with choices of

Optimizers

* Radam
* Adam
* Ranger

Models

* densenet201
* inceptionv3
* inception_resnetv2
* xception
* nasnetlarge
* resnet50
* vgg16
* vgg16slim (every other layer removed)
* vgg19
* mobilenetv2

Data balancing

* on/off

Augmentations

* width, shift, and zoom  
* horizontal/vertical flip

and all the typical hyperparameters needed for model training like 
learning rate,  batch size, etc.

Results are stored in a [MLFlow](https://github.com/mlflow/mlflow) tracking server and optionally in the
 [Wandb](http://wandb.com) service. 

Following training, you should see results in a locally running mlfow server at http://127.0.0.1:5001, e.g.

Organized by experiment
![ Image link ](/doc/img/mlflow_exp.jpg)

with runs
![ Image link ](/doc/img/mlflow_run.jpg)
## Prerequisites
 - Python version 3.6.1 
- minio/AWS storage
- (optional) W&B account 
## Running
Build docker image for GPU training.
```bash
./build.sh GPU
```
You can also build a CPU version for testing on your desktop but this is not recommended.
If using the CPU, replace gpu with cpu in toMLproject and src/nose/Dockerfile, e.g.
mbari/avedac-cpu-kclassify not mbari/avedac-gpu-kclassify 
```bash
    ./build.sh CPU
```
Start a local Minio and MLFlow server
```bash
cd src/test && docker-compose -f docker-compose.local.yml up --build
```
Set up a python virtual environment
```bash
virtualenv --python=/usr/bin/python3.6 .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Upload data into s3 buckets for testing training and inference, e.g.
```
python src/test/upload.py
```
Create .env file with test environment parameters e.g.
```
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=<your access key>
AWS_SECRET_ACCESS_KEY=<your secret key>
MLFLOW_TRACKING_URI=<your tracking URI, e.g. http://localhost:5000>
(for local testing only - not needed for AWS) MLFLOW_S3_ENDPOINT_URL=<your S3 endpoint for minio only, e.g. http://localhost:9000>
```
If logging to wandb also, add
```
WANDB_API_KEY=<your API key>
WANDB_USERNAME=<your username>
WANDB_ENTITY=mbari
WANDB_MODE=run
WANDB_GROUP=test
WANDB_PROJECT=test-project
```
Run training
```bash
 mlflow run .
```
Optionally, create and experiment called "test", saving the results to the bucket s3://test and log to the run to that
```bash
mlflow experiments create -n test -l s3://test
mlflow run --experiment-name test -P train_tar=s3://test/catsdogstrain.tar.gz -P val_tar=s3://test/catsdogsval.tar.gz .
```

## Testing

```bash
cd src/test && docker-compose  up --build --abort-on-container-exit
```
If successful should see something ending in
```bash
nosetests       |   inspect.getargspec(func)
nosetests       |
nosetests       | ----------------------------------------------------------------------
nosetests       | Ran 1 test in 1.453s
nosetests       |
nosetests       | OK
nosetests exited with code 0
```
Clean-up with
```bash
cd src/test && docker-compose down -v
```