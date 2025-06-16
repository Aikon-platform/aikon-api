#!/bin/bash

set -e

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$ROOT_DIR"/utils.sh
source "$ROOT_DIR"/api/.env

# Create necessary directories at startup
mkdir -p "$ROOT_DIR"/var/dramatiq/
mkdir -p "$ROOT_DIR"/.config/matplotlib
chown -R $USER "$ROOT_DIR"/.config/matplotlib

source "$ROOT_DIR"/venv/bin/activate

is_build=0

if [[ "$INSTALLED_APPS" == *"vectorization"* ]]; then
    color_echo blue "Building operators for vectorization module..."
    cd "$ROOT_DIR"/api/app/vectorization/lib/
    python src/models/dino/ops/setup.py build install || {
        color_echo red "Failed to build vectorization operators"
    }
    # python src/models/dino/ops/test.py
    # pip install -e synthetic/
    is_build=1
fi


# if region in INSTALLED_APPS, and build is not already done
if [[ "$INSTALLED_APPS" == *"regions"* ]] && [[ $is_build -eq 0 ]]; then
    color_echo blue "Building operators for regions module..."
    cd "$ROOT_DIR"/api/app/regions/lib/line_predictor/dino/ops/
    python setup.py build install || {
        color_echo red "Failed to build regions operators"
    }
    # DTLR code should work without the need to build, and use line_predictor's build
fi

# Run command at each container launch
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
supervisord -c "$ROOT_DIR"/supervisord.conf
