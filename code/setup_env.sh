#!/usr/bin/env bash
set -euo pipefail

# Lock to prevent concurrent installs in the same session
LOCK_FILE="/tmp/exampville_env.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another job is setting up env; waiting..."
    flock 9   # wait for lock to be released
fi

CONDA_ENV=${CONDA_ENV:-zigenv}
if [[ -n "$CONDA_DEFAULT_ENV" && "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]]; then
    conda activate "$CONDA_ENV"
fi

# Upgrade packages to latest within pinned bounds
pip install --quiet --upgrade -r "${PBS_O_WORKDIR:-$(pwd)}/requirements.txt"

# Detect GPU and configure JAX accordingly
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    echo "GPU detected, enabling JAX GPU backend"
    export JAX_PLATFORMS=gpu
    export JAX_PLATFORMS_ONLY=gpu
    export PIPELINE_ACCELERATOR=gpu
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
else
    echo "No GPU detected, forcing JAX CPU backend"
    export JAX_PLATFORMS=cpu
    export JAX_PLATFORMS_ONLY=cpu
    export PIPELINE_ACCELERATOR=cpu
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
fi

# Verify
python -c "
import larch, searchlibrium, metacountregressor
print('larch:', larch.__version__)
print('searchlibrium:', searchlibrium.__version__)
print('metacountregressor:', metacountregressor.__version__)
"

# Verify JAX backend
python -c "
import jax
print('JAX devices:', jax.devices())
print('JAX backend:', jax.lib.xla_bridge.get_backend().platform)
"