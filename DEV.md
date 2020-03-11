# Developer notes
To test outside the nose testing framework, bring up the minio stack
with:
```bash
cd  src/test && docker-compose up -d
```
This will launch a minio (s3 like object  store)  on the [local server](http://localhost:9000/minio/login),
login using the credentials in test.env 
  
Train for 1 epoch on GPU device 3
TODO: add python notebook with example
```bash
docker run --env-file .env -e CUDA_VISIBLE_DEVICES=3 -e WANDB_RUN_GROUP=test \
	-it --rm  mbari/avedac-gpu-kclassify  python /app/train.py \
	--horizontal_flip True --augment_range 0.2 --experiment_bucket s3://kctrain-testmlflow \
	--train_tar s3://kctrain-test/catsdogstrain.tar.gz \
	--val_tar s3://kctrain-test/catsdogsval.tar.gz --lr 0.001  \
	--base_model vgg16  --batch_size 32 --epoch 1
```

#### Predict
Upload catsdog.tar.gz to a bucket test
TODO: add python notebook with example
```bash
docker run --env-file .env -e CUDA_VISIBLE_DEVICES=1 \
	-it --rm  mbari/avedac-gpu-kclassify  \
	--run_uuid s3://kctrain-testmlflow/efdd7390a6ba493cb9b78cae625b2b0f/artifacts/model \
	--s3_results_bucket s3://testdata \
	--image_path s3://testdata/catsdogs.tar.gz
```

## Learning curve 
TODO: add python notebook with example
Compute learning curve, compute 5 epochs for each iteration

```bash
docker run --env-file .env -e WANDB_RUN_GROUP=test \
	-it --rm  mbari/avedac-gpu-kclassify  python /app/learning_curve.py \
	--horizontal_flip True --augment_range 0.2 --experiment_bucket s3://testdata \
	--train_tar s3://testdata/catsdogstrain.tar.gz \
	--val_tar s3://testdata/catsdogsval.tar.gz --lr 0.001  \
	--base_model vgg16 --project catsdogs --batch_size 4 --epoch 5
