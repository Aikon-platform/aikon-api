#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$SCRIPT_DIR"/docker/utils.sh

export INSTALL_MODE=${$INSTALL_MODE:-"full_install"}

echo_title "REQUIREMENTS INSTALL"

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
    venv/bin/pip install wheel>=0.45.1
    venv/bin/pip install -r requirements-dev.txt
    venv/bin/pip install python-dotenv
fi

echo_title "ENVIRONMENT VARIABLES"

color_echo blue "\nDo you want to setup environment variable?"
answer=$(printf "%s\n" "${options[@]}" | fzy)
if [ "$answer" = "yes" ]; then
    color_echo yellow "\nSetting up .env files"
    setup_env "$SCRIPT_DIR"/.env
    setup_env "$SCRIPT_DIR"/.env.dev
fi

if [ "$TARGET" == "dev" ]; then
    echo_title "PRE-COMMIT INSTALL"
    venv/bin/pip install pre-commit
    pre-commit install
fi

set_redis() {
    redis_psw="$1"
    REDIS_CONF=$(redis-cli INFO | grep config_file | awk -F: '{print $2}' | tr -d '[:space:]')
    color_echo yellow "\nModifying Redis configuration file $REDIS_CONF ..."

    # use the same redis password for api and front
    $SED_CMD "s~^REDIS_PASSWORD=.*~REDIS_PASSWORD=\"$redis_psw\"~" "../front/app/config/.env"

    sudo $SED_CMD "s/\nrequirepass [^ ]*/requirepass $redis_psw/" "$REDIS_CONF"
    sudo $SED_CMD "s/# requirepass [^ ]*/requirepass $redis_psw/" "$REDIS_CONF"

    if [ "$OS" = "Linux" ]; then
        sudo systemctl restart redis-server
    elif [ "$OS" = "Mac" ]; then
        brew services restart redis
    fi
}
color_echo blue "\nDo you want to add a password to redis?"
answer=$(printf "%s\n" "${options[@]}" | fzy)
if [ "$answer" = "yes" ]; then
    set_redis $REDIS_PASSWORD
fi

color_echo blue "\nDo you want to init submodules?"
answer=$(printf "%s\n" "${options[@]}" | fzy)
if [ "$answer" = "yes" ]; then
    echo_title "DOWNLOADING SUBMODULES"
    git submodule init
    git submodule update
fi

echo_title "🎉 API SET UP COMPLETED ! 🎉"
