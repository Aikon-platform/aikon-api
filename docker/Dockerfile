## STEP 1 - install python packages 
FROM ghcr.io/astral-sh/uv:debian-slim AS uvdeps

# ARG API_UV_CACHE_VOLUME
# ARG API_UV_VENV_VOLUME

# uv config: install python + venv under the user's home, compile bytecode
ENV UV_NO_DEV=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_CACHE_DIR=/root/.uv_cache
ENV UV_PROJECT_ENVIRONMENT=/root/.venv

COPY ./pyproject.toml ./uv.lock ./

RUN apt-get update && apt-get install -y git

# RUN --mount=type=cache,target=${UV_CACHE_DIR},id=uv-cache \
#     uv python install 3.11 
RUN --mount=type=cache,target=${UV_CACHE_DIR},id=uv-cache \
    uv sync --locked --group=prod

# ----------------------------------------------------------

## STEP 2 - final API image
# see this if you have torch issues https://github.com/Aikon-platform/aikon-api/wiki/Torch-and-CUDA-Troubleshooting
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV no_proxy=${NO_PROXY}

ENV USER=aikonapi
ARG USERID

ARG API_PORT

# # uv config: install python + venv under the user's home, compile bytecode
# ENV UV_NO_DEV=1
# ENV UV_COMPILE_BYTECODE=1
# ENV UV_CACHE_DIR=/home/${USER}/.uv_cache
# ENV UV_PROJECT_ENVIRONMENT=/home/${USER}/.venv
# ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

ENV PYTHON_VENV=/home/${USER}/.venv

ENV HF_HOME=/home/${USER}/.cache/huggingface

RUN useradd -u ${USERID} -m -d /home/${USER} ${USER}

## Silence error messages
ENV TERM=linux

## Install utils
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

## Set nvidia
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

## Set matplotlib tmp dir
ENV MPLCONFIGDIR=/home/${USER}/.config/matplotlib

## Bash instead of shell
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    unzip \
    wget \
    zip \
    supervisor \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    redis-server \
    nginx \
    git \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/${USER}

# # install uv
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ## Create and set permissions for local uv folders
# RUN mkdir -p ${UV_CACHE_DIR} ${UV_VENV_DIR} && \
#     chown -R ${USER}:${USER} ${UV_CACHE_DIR} ${UV_VENV_DIR}

## Copy lockfiles and install deps
# (Before copying the rest so that this part is not rerun unless the lock changes)
COPY --chown=${USER} ./pyproject.toml ./uv.lock ./
# RUN uv python install 3.11 
# RUN uv sync --locked --group=prod

# copy dependencies from build stage
COPY --chown=${USER} --from=uvdeps /root/.venv ${PYTHON_VENV}
RUN echo "1 *****" && echo "${PYTHON_VENV}"
RUN echo "2 *****" && ls -l ${PYTHON_VENV}

# Copy additional configurations
COPY --chown=${USER} ./.env ./api/.env
COPY docker/nginx.conf /etc/nginx/conf.d/${USER}.conf
COPY docker/supervisord.conf ./supervisord.conf

COPY docker/start.sh /home/${USER}/start.sh
RUN chmod +x /home/${USER}/start.sh

# Copy the api code (having this as close to the bottom of the 
# file as possible may speed up successive docker builds)
COPY --chown=${USER} ./ ./api/

# Expose the application port
EXPOSE ${API_PORT}

CMD ["/home/aikonapi/start.sh"]
