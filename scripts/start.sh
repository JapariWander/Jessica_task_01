#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Jessica_task_01
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
exec .venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false
