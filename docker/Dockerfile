## STEP 1 - install python packages 
FROM ghcr.io/astral-sh/uv:debian-slim AS uvdeps

# uv config: install python + venv under the user's home, compile bytecode
ENV UV_NO_DEV=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_CACHE_DIR=/root/.uv_cache
ENV UV_PROJECT_ENVIRONMENT=/root/.venv

COPY ./pyproject.toml ./uv.lock ./

RUN apt-get update && apt-get install -y git

RUN --mount=type=cache,target=${UV_CACHE_DIR},id=uv-cache \
    uv sync --locked --group=prod

# ----------------------------------------------------------

## STEP 2 - final API image
# see this if you have torch issues https://github.com/Aikon-platform/aikon-api/wiki/Torch-and-CUDA-Troubleshooting
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 AS aikonapi

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV no_proxy=${NO_PROXY}

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

## Copy lockfiles and install deps
# (Before copying the rest so that this part is not rerun unless the lock changes)
COPY --chown=${USER} ./pyproject.toml ./uv.lock ./
# RUN uv python install 3.11 
# RUN uv sync --locked --group=prod

# copy the python directory defined by uv at build stage to avoid symlinking 
# the venv python to a nonexistent file 
COPY --chown=${USER} --from=uvdeps /root/.local/share/uv/python ${PYTHON_DIR}
# copy dependencies from build stage and set properties
COPY --chown=${USER} --from=uvdeps /root/.venv ${PYTHON_VENV}
RUN \
    find ${PYTHON_VENV} -type d -print0 | xargs -0 -P ${N_PROCESS} chmod u+rwx & \
    find ${PYTHON_VENV}/bin -type f -print0 | xargs -0 -P ${N_PROCESS} chmod u+x & \
    find ${PYTHON_DIR} -type d -print0 | xargs -0 -P ${N_PROCESS} chmod u+rwx & \
    find ${PYTHON_DIR}/bin -type f -print0 | xargs -0 -P ${N_PROCESS} chmod u+x & \
    # update symlink so that the python in .venv properly links to the one in PYTHON_DIR
    find ${PYTHON_VENV}/bin -type l | while read link; do \
      target=$(readlink "$link"); \
      new_target="${PYTHON_DIR}${target#/root/.local/share/uv/python}"; \
      ln -sf "$new_target" "$link"; \
    done & \
    wait

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
