#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$SCRIPT_DIR"/docker/utils.sh
source "../front/app/config/.env"

INSTALL_MODE=${INSTALL_MODE:-"full_install"}

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
        sudo apt-get install redis-server curl
        sudo systemctl start redis-server
    elif [ "$OS" = "Mac" ]; then
        brew install redis curl
        brew services start redis
    fi
    which uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh;
fi

color_echo blue "\nDo you want to setup python virtual environment?"
answer=$(printf "%s\n" "${options[@]}" | fzy)
if [ "$answer" = "yes" ]; then
    color_echo yellow "\nAPI virtual env..."
    uv sync --group=dev --directory="$SCRIPT_DIR"
fi

color_echo blue "\nDo you want to setup environment variables?"
answer=$(printf "%s\n" "${options[@]}" | fzy)
if [ "$answer" = "yes" ]; then
    color_echo yellow "\nSetting up .env files"
    default_params=("API_PORT" "PROD_URL" "API_DATA_FOLDER" "TARGET" "DOCKER" "YOLO_CONFIG_DIR")
    setup_env "$SCRIPT_DIR"/.env "${default_params[@]}"
    setup_env "$SCRIPT_DIR"/.env.dev "${default_params[@]}"
fi

source "$SCRIPT_DIR"/.env
source "$SCRIPT_DIR"/.env.dev
if [ "$TARGET" == "dev" ]; then
    color_echo yellow "\nPre-commit setup"
    uv tool install pre-commit --with pre-commit-uv
fi

color_echo blue "\nDo you want to init submodules?"
answer=$(printf "%s\n" "${options[@]}" | fzy)
if [ "$answer" = "yes" ]; then
    color_echo yellow "\nSubmodules initialization"
    git submodule init
    git submodule update
fi

echo_title "🎉 API SET UP COMPLETED ! 🎉"
