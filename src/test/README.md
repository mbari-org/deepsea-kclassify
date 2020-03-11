# Tests

Tests are run automatically upon check-in to the repository through the bitbucket pipeline.

To test manually

# Create virtual environment
```bash
$ virtualenv ../venv-mbari/avedac-kclalssify  --python=/usr/bin/python3.7
$ source ../venv-mbari/avedac-kclalssify/bin/activate
```

Install dependencies
```bash
$ pip3 install -r requirements.txt
```
 
### Create Docker image
 
1. Create GPU docker image with the code
```bash
    ./build.sh GPU
```

can also build a CPU version for testing on a non GPU machine

```bash
    ./build.sh CPU
```

## Run test
```bash
    docker-compose build && docker-compose up
```
