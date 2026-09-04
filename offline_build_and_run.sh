#!/bin/bash

set -e

# === Configuration ===
IMAGE_NAME=cuda-agent-sim
DOCKERFILE_PATH=.

# === Parse args ===
seed=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            seed="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$SCRIPT_DIR"

# Define paths relative to the script's directory
STATE_ESTIMATIONS_HOST="${SCRIPT_DIR}/state_estimations"
SIMULATION_HOST="${SCRIPT_DIR}"

STATE_ESTIMATIONS_CONTAINER=/state_estimations
SIMULATION_CONTAINER=/sim

if [[ -z "$(docker images -q cuda-env:12.2 2> /dev/null)" ]]; then
    echo "Loading base image..."
    docker load -i cuda-env.tar
else
    echo "Base image already present."
fi
echo "Building Docker image: $IMAGE_NAME"
DOCKER_BUILDKIT=1 docker build --pull=false -t cuda-agent-sim .
#DOCKER_BUILDKIT=1 docker build --no-cache --pull=false -t cuda-agent-sim .
# Remove dangling images after every build
docker image prune -f

echo "Running container with GPU support..."
docker run --rm --privileged --gpus all \
    -v "$STATE_ESTIMATIONS_HOST":"$STATE_ESTIMATIONS_CONTAINER" \
    -v "$SIMULATION_HOST":"$SIMULATION_CONTAINER" \
    "$IMAGE_NAME" --seed "$seed"
#docker run --rm --privileged --gpus all \
#    -v "$STATE_ESTIMATIONS_HOST":"$STATE_ESTIMATIONS_CONTAINER" \
#    -v "$SIMULATION_HOST":"$SIMULATION_CONTAINER" \
#    -it --entrypoint /bin/bash \
#    "$IMAGE_NAME"

#docker run --rm --privileged --gpus all \
#    -v "$STATE_ESTIMATIONS_HOST":"$STATE_ESTIMATIONS_CONTAINER" \
#    -v "$SIMULATION_HOST":"$SIMULATION_CONTAINER" \
#    --entrypoint compute-sanitizer \
#    "$IMAGE_NAME" \
#    --tool memcheck \
#    --leak-check full \
#    --show-backtrace yes \
#    /app/build/untitled2 --seed "$seed"

echo "Exit code: $?"