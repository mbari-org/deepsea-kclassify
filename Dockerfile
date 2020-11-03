ARG TF_VERSION
FROM tensorflow/tensorflow:${TF_VERSION}

RUN apt-get update && apt-get install -y git  && \
	apt-get install -y libsm6  && \
	apt-get install -y libxext6  && \
	apt-get install -y libxrender-dev

ENV APP_HOME /app
WORKDIR ${APP_HOME}
ENV PATH=${PATH}:${APP_HOME}

COPY requirements.txt ${APP_HOME}
RUN python3 -m pip install --upgrade pip &&  python3 -m pip install -r /app/requirements.txt

COPY src/main/ .
CMD ["python", "/app/train.py"]
