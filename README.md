# sample-efficient-rl

PPO, Dedieu M5, and ITC on Craftax-Classic.
Based on the [ITC reproduction](https://github.com/snu-mllab/Identifiable-Token-Correspondence).

## Setup

Python 3.12, Linux or WSL2, and an NVIDIA GPU.

```bash
envDir="$HOME/.cache/sample-efficient-rl/env"
python3.12 -m venv "$envDir"
"$envDir/bin/python" -m pip install -r requirements.txt
```

Set `CRAFTAX_ENV` to use another environment.

## Run

```bash
runDir="$HOME/.local/share/sample-efficient-rl/dedieu0"
bash run.sh dedieu --outputDir "$runDir" --execute
bash run.sh evaluate --runDir "$runDir"
```

Use `ppo` or `itc` with a separate output directory.
Drop `--execute` to print the configuration. Add `--resume` to continue a run.

W&B is off by default. Enable it in `configuration.py`.
