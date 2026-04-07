# Final image (change image based on the version showed with $ nvidia-smi)
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV no_proxy=${NO_PROXY}

ARG HUGGING_FACE_HUB_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}

ENV USER=aikonapi
ARG USERID

ARG API_PORT

RUN useradd -u ${USERID} -m -d /home/${USER} ${USER}

## Silence error messages
ENV TERM=linux

## Bash instead of shell
SHELL ["/bin/bash", "-c"]

## Install utils
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10

RUN apt-get update && apt-get install -y \
    unzip \
    wget \
    zip \
    supervisor \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    python3.10-dev \
    python3.10-venv \
    redis-server \
    nginx \
    git \
    poppler-utils

WORKDIR /home/${USER}

## Set nvidia
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

## Copy requirements and install them
# (Before copying the rest so that this part is not rerun unless requirements change)
COPY --chown=${USER} ./requirements.txt ./requirements.txt
COPY --chown=${USER} ./requirements-prod.txt ./requirements-prod.txt
RUN python3.10 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements-prod.txt

# Copy the entire project code
COPY --chown=${USER} ./ ./api/

WORKDIR /home/${USER}

# Copy additional configurations
COPY --chown=${USER} ./.env.prod ./api/.env
COPY docker/nginx.conf /etc/nginx/conf.d/${USER}.conf
COPY docker/supervisord.conf ./supervisord.conf

# Expose the application port
EXPOSE ${API_PORT}

# Set matplotlib tmp dir
ENV MPLCONFIGDIR=/home/${USER}/.config/matplotlib

COPY docker/utils.sh /home/${USER}/utils.sh
COPY docker/start.sh /home/${USER}/start.sh
RUN chmod +x /home/${USER}/utils.sh
RUN chmod +x /home/${USER}/start.sh

CMD ["/home/aikonapi/start.sh"]
