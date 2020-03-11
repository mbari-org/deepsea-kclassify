ARG TF_VERSION
FROM tensorflow/tensorflow:${TF_VERSION}

ENV APP_HOME /app
WORKDIR ${APP_HOME}
ENV PATH=${PATH}:${APP_HOME}

COPY requirements.txt ${APP_HOME}
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /app/requirements.txt

RUN apt-get update && apt-get install -y git  && \
	apt-get install -y libsm6  && \
	apt-get install -y libxext6  && \
	apt-get install -y libxrender-dev

# Add non-root user and fix permissions
COPY src/main/ ${APP_HOME}
ARG DOCKER_GID
ARG DOCKER_UID
RUN groupadd --gid $DOCKER_GID docker && adduser --uid $DOCKER_UID --gid $DOCKER_GID --disabled-password --quiet --gecos "" docker_user
RUN chown -Rf docker_user:docker ${APP_HOME}
USER docker_user

WORKDIR ${APP_HOME}
CMD ["python", "/app/train.py"]