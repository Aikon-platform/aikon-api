#!/bin/bash

# HOW TO USE
# Inside the docker/ directory, run:
# bash docker.sh <start|stop|restart|pull|build>

set -e

DOCKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# initialize the .env files and data folder permissions on first initialization
bash "$DOCKER_DIR"/init.sh

# Load container variables from .env file
source "$DOCKER_DIR"/.env
source "$API_ROOT"/.env.prod

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -Eq "$CONTAINER_NAME"
}

image_exists() {
    docker image inspect "$CONTAINER_NAME" >/dev/null 2>&1
}

stop_container() {
    if container_exists; then
        colorEcho blue "\nStopping $CONTAINER_NAME"
        docker stop "$CONTAINER_NAME"
        docker rm "$CONTAINER_NAME"
    fi
}

build_image() {
    colorEcho blue "\nBuilding Docker image $CONTAINER_NAME"
    docker build --rm -t "$CONTAINER_NAME" -f Dockerfile .. \
        --build-arg USERID=$DEMO_UID \
        --build-arg HTTP_PROXY=${HTTP_PROXY} \
        --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
        --build-arg NO_PROXY=${NO_PROXY} \
        --build-arg HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} || {
            colorEcho red "\nDocker build failed"
            exit 1;
        }
}

pull_code() {
    colorEcho blue "\nPulling latest code and updating submodules"
    cd "$DOCKER_DIR/.."
    git pull
    git submodule update
    cd "$DOCKER_DIR"
}

# Function to start the container
start_container() {
    if image_exists; then
        colorEcho blue "\nStarting container $CONTAINER_NAME"
        docker run -d --gpus "$DEVICE_NB" --name "$CONTAINER_NAME" \
           -v "$DATA_FOLDER":/data/ -v "$CUDA_HOME":/cuda/ -p "$CONTAINER_HOST":"$API_PORT":"$API_PORT" \
           --restart unless-stopped --ipc=host "$CONTAINER_NAME"
    else
        colorEcho red "\nImage $CONTAINER_NAME does not exist. Build failed or not yet built."
        exit 1
    fi
}

case "$1" in
    start)
        stop_container
        start_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        stop_container
        start_container
        ;;
    pull)
        stop_container
        pull_code
        build_image
        start_container
        ;;
    build)
        stop_container
        build_image
        start_container
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|rebuild|pull|build}"
        exit 1
esac
