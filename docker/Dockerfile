# NOTE : to use a GPU in docker, the nvidia container toolkit is needed on the host.
# see: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# see this if you have torch issues https://github.com/Aikon-platform/aikon-api/wiki/Torch-and-CUDA-Troubleshooting
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04 AS aikonapi

#RUN echo "195.221.193.4 apt-cache.enpc.fr" >> /etc/hosts

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV no_proxy=${NO_PROXY}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}
#ENV http_proxy=http://proxy.enpc.fr:3128
#ENV https_proxy=http://proxy.enpc.fr:3128
#ENV HTTP_PROXY=http://proxy.enpc.fr:3128
#ENV HTTPS_PROXY=http://proxy.enpc.fr:3128

ENV USER=aikonapi
ARG USERID

ARG API_PORT

# global python directory (user level)
ENV PYTHON_DIR=/home/${USER}/.local/share/python
# virutalenv directory (venv level)
ENV PYTHON_VENV=/home/${USER}/.venv

ENV HF_HOME=/home/${USER}/.cache/huggingface

RUN useradd -u ${USERID} -m -d /home/${USER} ${USER}

## Silence error messages
ENV TERM=linux

## Install utils
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# uv stuff
ENV PYTHONUNBUFFERED=1
ENV UV_NO_DEV=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_CACHE_DIR=/home/${USER}/.uv_cache
ENV UV_PROJECT_ENVIRONMENT=/home/${USER}/.venv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

## Set nvidia
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV N_PROCESS=8

## Set matplotlib tmp dir
ENV MPLCONFIGDIR=/home/${USER}/.config/matplotlib

## Bash instead of shell
SHELL ["/bin/bash", "-c"]

RUN echo 'Acquire::http::Proxy "http://195.221.193.4:3142";' > /etc/apt/apt.conf.d/proxy.conf && \
    echo 'Acquire::https::Proxy "http://195.221.193.4:3142";' >> /etc/apt/apt.conf.d/proxy.conf

    RUN apt-get update || true && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    curl \
    apt-transport-https \
    gnupg \
    dnsutils \
    iputils-ping \
    unzip \
    zip \
    supervisor \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    redis-server \
    nginx \
    git \
    poppler-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/${USER}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

## Copy lockfiles and install deps
# (Before copying the rest so that this part is not rerun unless the lock changes)
COPY --chown=${USER} ./pyproject.toml ./uv.lock ./

# Copy additional configurations
COPY --chown=${USER} ./.env ./api/.env
COPY docker/nginx.conf /etc/nginx/conf.d/${USER}.conf
COPY docker/supervisord.conf ./supervisord.conf

COPY docker/start.sh /home/${USER}/start.sh
RUN chmod +x /home/${USER}/start.sh

# install python deps (putting this to the bottom of the file can speedup iterative builds)
RUN uv python install 3.11
RUN --mount=type=cache,target=${UV_CACHE_DIR},id=uv-cache \
    uv sync --locked --group=prod

# Copy the api code (having this as close to the bottom of the
# file as possible may speed up successive docker builds)
COPY --chown=${USER} ./ ./api/

# Expose the application port
EXPOSE ${API_PORT}

CMD ["/home/aikonapi/start.sh"]
