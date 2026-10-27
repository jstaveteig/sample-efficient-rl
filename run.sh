#!/usr/bin/env bash
set -euo pipefail
repoDir=$(cd -- "$(dirname -- "$0")" && pwd)
cacheDir=${XDG_CACHE_HOME:-"$HOME/.cache"}/sample-efficient-rl
envDir=${CRAFTAX_ENV:-"$cacheDir/env"}
pythonPath="$envDir/bin/python"
if [[ ! -x $pythonPath ]]; then
    printf 'set CRAFTAX_ENV to an external Python environment\n' >&2
    exit 1
fi
if [[ -d /usr/lib/wsl/drivers ]]; then
    driverLibrary=$(find /usr/lib/wsl/drivers -path '*/nv_dispi*/libcuda.so.1.1' -print -quit)
    test -n "$driverLibrary"
    driverDir=$(dirname -- "$driverLibrary")
    test -f "$driverDir/libnvidia-ptxjitcompiler.so.1"
    # wsl needs matching cuda driver libraries
    export LD_LIBRARY_PATH="$driverDir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PATH="/usr/lib/wsl/lib:$PATH"
fi
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export OMP_NUM_THREADS=4
entry=train.py
if [[ ${1:-} == evaluate ]]; then
    entry=evaluate.py
    shift
fi
cd -- "$repoDir"
exec "$pythonPath" -B "$entry" "$@"
