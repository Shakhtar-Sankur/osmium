#!/bin/bash
# osmium M0 on Kaggle (GPU T4 x2, Internet on): reproduce Apple's MMLU numbers for open models.
#   !cd /tmp && rm -rf osmium && git clone -q --depth 1 https://github.com/Shakhtar-Sankur/osmium && bash osmium/scripts/kaggle_m0.sh
# QUICK=1: 500 questions per model (about 15 minutes); the full 14,042 take about 1-2 hours.
set -eo pipefail
cd "$(dirname "$0")/.."
echo "== osmium M0: osmium $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
pip install -q "vllm==0.31.0" 2>&1 | grep -v -i "dependency resolver\|incompatible\|^$" | tail -2 || true
python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| vllm', vllm.__version__, '| GPU usable:', torch.cuda.is_available())"
LIMIT=$( [ -n "$QUICK" ] && echo "--limit 500" )
mkdir -p runs
CUDA_VISIBLE_DEVICES=0 python scripts/eval_mmlu.py --model Qwen/Qwen3-4B $LIMIT --out runs/m0 > runs/gpu0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python scripts/eval_mmlu.py --model Qwen/Qwen2.5-3B-Instruct $LIMIT --out runs/m0 > runs/gpu1.log 2>&1 &
P1=$!
wait $P0 || { echo "GPU 0 failed:"; grep -E "Error|error|Traceback" runs/gpu0.log | tail -8; }
wait $P1 || { echo "GPU 1 failed:"; grep -E "Error|error|Traceback" runs/gpu1.log | tail -8; }
grep -h '^{' runs/gpu0.log runs/gpu1.log || true
[ -d /kaggle/working ] && mkdir -p /kaggle/working/osmium-m0 && cp -r runs/m0 /kaggle/working/osmium-m0/
echo "== summary"
python scripts/summarize_m0.py runs/m0
echo "== done: copy from '== osmium M0' to here and send it back"
