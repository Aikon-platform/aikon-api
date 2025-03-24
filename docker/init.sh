DOCKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
API_ROOT="$(dirname "$DOCKER_DIR")"

source "$DOCKER_DIR"/utils.sh

# echo $CUDA_HOME  # if already defined, copy the path in the .env file
# Otherwise find CUDA version with (pay attention to version mismatches)
# nvcc --version
# nvidia-smi
# # set CUDA_HOME and make sure nvcc version matches selected CUDA_HOME
# export CUDA_HOME=<path/to/cuda>
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export INSTALL_MODE="full_install"

if [ ! -f "$API_ROOT"/.env.prod ]; then
    setup_env "$API_ROOT"/.env.prod
fi

# if docker/.env does not exist, create it
if [ ! -f "$DOCKER_DIR"/.env ]; then
    setup_env "$DOCKER_DIR"/.env
fi

source "$API_ROOT"/.env.prod
source "$DOCKER_DIR"/.env

if [ ! -f "$DOCKER_DIR"/nginx.conf ]; then
    cp "$DOCKER_DIR"/nginx.conf.template "$DOCKER_DIR"/nginx.conf
    sed_repl_inplace "s~API_PORT~${API_PORT:-8001}~" "$DOCKER_DIR"/nginx.conf
fi

if [ ! -f "$DOCKER_DIR"/supervisord.conf ]; then
    cp "$DOCKER_DIR"/supervisord.conf.template "$DOCKER_DIR"/supervisord.conf
#     sed_repl_inplace "s~CONTAINER_NAME~$CONTAINER_NAME~" "$DOCKER_DIR"/supervisord.conf
    NB_PROCS=$(echo "${INSTALLED_APPS:-}" | awk -F, '{print NF}')
    sed_repl_inplace "s~NB_PROCS~${NB_PROCS:-1}~" "$DOCKER_DIR"/supervisord.conf
fi

# if $DATA_FOLDER does not exist
if [ ! -d "$DATA_FOLDER" ]; then
    # Create $DATA_FOLDER folder with right permissions for user $USERID
    sudo mkdir -p "$DATA_FOLDER"
    sudo chown -R "$USERID:$USERID" "$DATA_FOLDER"
    sudo chmod -R 775 "$DATA_FOLDER"
fi
