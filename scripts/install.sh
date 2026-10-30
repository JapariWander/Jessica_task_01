#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Jessica_task_01
python -m venv .venv
export PIP_CACHE_DIR="$PWD/.cache/pip"
.venv/bin/python -m pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
