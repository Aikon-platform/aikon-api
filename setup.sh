#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

INSTALL_MODE=${INSTALL_MODE:-"full_install"}
source "$SCRIPT_DIR"/docker/utils.sh
source "../front"

color_echo cyan "Running a $INSTALL_MODE for the API! 🚀"

color_echo yellow "\nInstalling prompt utility fzy..."
if [ "$OS" = "Linux" ]; then
    sudo apt install fzy
elif [ "$OS" = "Mac" ]; then
    brew install fzy
else
    color_echo red "\nUnsupported OS: $OS"
    exit 1
fi

options=("yes" "no")

color_echo blue "\nDo you want to install system packages?"
answer=$(printf "%s\n" "${options[@]}" | fzy)
if [ "$answer" = "yes" ]; then
    color_echo yellow "\nSystem packages..."
    if [ "$OS" = "Linux" ]; then
        sudo apt-get install redis-server python3.10 python3.10-venv python3.10-dev curl
    elif [ "$OS" = "Mac" ]; then
        brew install redis python@3.10 curl
        brew services start redis
    fi
fi

color_echo blue "\nDo you want to install python virtual environment?"
answer=$(printf "%s\n" "${options[@]}" | fzy)
if [ "$answer" = "yes" ]; then
    color_echo yellow "\nAPI virtual env..."
    python3.10 -m venv venv
    venv/bin/pip install "wheel>=0.45.1"
    venv/bin/pip install -r requirements-dev.txt
    venv/bin/pip install python-dotenv
fi

color_echo blue "\nDo you want to setup environment variable?"
answer=$(printf "%s\n" "${options[@]}" | fzy)
if [ "$answer" = "yes" ]; then
    color_echo yellow "\nSetting up .env files"

    default_param=("API_PORT" "PROD_URL" "API_DATA_FOLDER" "TARGET" "DOCKER" "YOLO_CONFIG_DIR")
    setup_env "$SCRIPT_DIR"/.env "${default_params[@]}"
    setup_env "$SCRIPT_DIR"/.env.dev "${default_params[@]}"
fi

if [ "$TARGET" == "dev" ]; then
    color_echo yellow "\nPre-commit install"
    venv/bin/pip install pre-commit
    pre-commit install
fi

color_echo blue "\nDo you want to init submodules?"
answer=$(printf "%s\n" "${options[@]}" | fzy)
if [ "$answer" = "yes" ]; then
    color_echo yellow "\nSubmodules initialization"
    git submodule init
    git submodule update
fi

echo_title "🎉 API SET UP COMPLETED ! 🎉"
