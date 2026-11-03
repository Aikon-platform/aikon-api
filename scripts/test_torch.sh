#!/usr/bin env bash

USAGE=$(cat <<EOF

USAGE: bash test_torch [-d|--docker]?

	-d|--docker : test it in the dockerized API (named 'aikonapi')

test if a torch installation is successful: correctly installed, 
with CUDA available for GPU processing

EOF
)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
API_DIR="$SCRIPT_DIR/.."
CONTAINER_NAME="aikonapi"

# set IN_DOCKER
IN_DOCKER=false;
if [ -n "$1" ]; then
       	if [[ "$1" != "-d" ]] && [[ "$1" != "--docker" ]]; then 
		echo "$USAGE";
		exit 1;
	else
		IN_DOCKER=true;
	fi
fi

CMD_TORCH_VER="import torch; print(torch.__version__)"
CMD_TORCH_CUDA="import torch; print(torch.cuda.is_available())"
if [[ "$IN_DOCKER" = true ]]; then 
	TORCH_VERSION=$(docker exec -it "$CONTAINER_NAME" .venv/bin/python -c "$CMD_TORCH_VER")
	CUDA_OK=$(docker exec -it "$CONTAINER_NAME" .venv/bin/python -c "$CMD_TORCH_CUDA")
else
	TORCH_VERSION=$("$API_DIR"/.venv/bin/python -c "$CMD_TORCH_VER")
	CUDA_OK=$("$API_DIR"/.venv/bin/python -c "$CMD_TORCH_CUDA")
fi

echo "  * torch version: $TORCH_VERSION"
echo "  * cuda available: $CUDA_OK"
