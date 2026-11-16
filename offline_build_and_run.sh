#!/bin/bash

set -e

# === Configuration ===
IMAGE_NAME=cuda-agent-sim
DOCKERFILE_PATH=.

# === Parse args ===
seed=0
agent=""
run_idx=""
im=""
cond=""
worm_count=""
w_phero=""
w_food=""
out=""
skip_build=0
lambda=""
nsga=0
resume=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            seed="$2"
            shift 2
            ;;
        --agent)
            agent="$2"
            shift 2
            ;;
        --run-idx)
            run_idx="$2"
            shift 2
            ;;
        --im)
                    im="$2"
                    shift 2
                    ;;
        --cond)
              cond="$2"
              shift 2
              ;;
        --worm-count)
              worm_count="$2"
              shift 2
              ;;
        --w-phero)
              w_phero="$2"
              shift 2
              ;;
        --out)
            out="$2"
            shift 2
            ;;
        --skip-build)
                    skip_build=1
                    shift
                    ;;
        --nsga)
                    nsga=1
                    shift
                    ;;
        --resume)
                    resume=1
                    shift
                    ;;
        --lambda)
                    lambda="$2"
                    shift 2
                    ;;
        --w-food)
                    w_food="$2"
                    shift 2
                    ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Only pass --agent when given, so the default (no agent) mode still works
EXTRA_ARGS=()
if [[ -n "$agent" ]]; then
    EXTRA_ARGS+=(--agent "$agent")
fi
if [[ -n "$run_idx" ]]; then
    EXTRA_ARGS+=(--run-idx "$run_idx")
fi
if [[ -n "$im" ]]; then
    EXTRA_ARGS+=(--im "$im")
fi
if [[ -n "$cond" ]]; then
    EXTRA_ARGS+=(--cond "$cond")
fi
if [[ -n "$worm_count" ]]; then
    EXTRA_ARGS+=(--worm-count "$worm_count")
fi
if [[ -n "$w_phero" ]]; then
    EXTRA_ARGS+=(--w-phero "$w_phero")
fi
if [[ -n "$w_food" ]]; then
    EXTRA_ARGS+=(--w-food "$w_food")
fi
if [[ -n "$out" ]]; then
    EXTRA_ARGS+=(--out "$out")
fi
if [[ -n "$lambda" ]]; then
    EXTRA_ARGS+=(--lambda "$lambda")
fi
if [[ "$nsga" -eq 1 ]]; then
    EXTRA_ARGS+=(--nsga)
fi
if [[ "$resume" -eq 1 ]]; then
    EXTRA_ARGS+=(--resume)
fi

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

if [[ "$skip_build" -eq 0 ]]; then
    echo "Building Docker image: $IMAGE_NAME"
    DOCKER_BUILDKIT=1 docker build --pull=false -t cuda-agent-sim .
    docker image prune -f
fi

echo "Running container with GPU support..."
docker run --rm --privileged --gpus all \
    -v "$STATE_ESTIMATIONS_HOST":"$STATE_ESTIMATIONS_CONTAINER" \
    -v "$SIMULATION_HOST":"$SIMULATION_CONTAINER" \
    "$IMAGE_NAME" --seed "$seed" "${EXTRA_ARGS[@]}"
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