# osmium

**The densest language model per bit: a 2-bit open model against Apple Intelligence.**

Osmium is the densest element: the most mass in the least space. This project is about the most
intelligence in the fewest bits.

## The target

Apple's on-device model is about 3 billion parameters, shipped at **2 bits per weight**: 4-bit
embeddings, an 8-bit KV cache, and low-rank adapters to recover the quality compression loses
([Apple Intelligence Foundation Language Models, Tech Report 2025](https://arxiv.org/abs/2507.13575),
section 6). Apple reports, on MMLU as OpenAI's simple-evals runs it:

| model | bits per weight | MMLU | IFEval |
|---|---|---|---|
| AFM on-device (Table 3) | 16 | 67.8 | 85.1 |
| **AFM on-device, as shipped (Table 3)** | **2** | **64.4** | **82.3** |
| Qwen-3-4B (Table 1) | 16 | 75.10 | |
| Qwen-2.5-3B (Table 1) | 16 | 66.37 | |
| Gemma-3-4B (Table 1) | 16 | 62.81 | |

**The challenge:** compress an open model to 2 bits per weight, within the memory Apple's model
needs, and beat **64.4 on MMLU and 82.3 on IFEval**. Qwen-3-4B starts 7.3 points above Apple's
16-bit model, so it has 10.7 points to lose. At 2 bits, small models usually lose more than that,
which is what makes this a real test.

Apple's model cannot be downloaded, so the comparison is against Apple's published numbers. That
makes the evaluation method the whole game. Hence M0.

## M0: reproduce Apple's numbers first

Before any compressed model is compared with Apple's, the harness must reproduce what Apple
reported for the open models in its Table 1. Otherwise a win or a loss could just be a different
prompt.

- `osmium/simple_evals.py` copies MMLU's prompt, answer patterns, response normalisation and
  subject categories from [openai/simple-evals](https://github.com/openai/simple-evals)
  (commit 652c89d, MIT), the library Apple names. The 14,042 questions come from simple-evals'
  own file, checked against a pinned SHA-256.
- `scripts/eval_mmlu.py` serves each model with vLLM on one T4. Decoding is greedy, so a run
  repeats exactly, and instruct models answer in their non-thinking mode, as in Apple's report.
  Every response is kept.
- `scripts/kaggle_m0.sh` runs Qwen-3-4B and Qwen-2.5-3B-Instruct on Kaggle's two T4s and
  compares each with Apple's number for it. (Gemma-3-4B needs a Hugging Face login.)

## Plan

| | Milestone | State |
|---|---|---|
| M0 | Reproduce Apple's MMLU for open models with simple-evals; add IFEval | running |
| M1 | 2-bit baselines from published post-training methods on Qwen-3-4B, at an exact bits-per-weight budget that counts every scale and zero point | |
| M2 | Quantization-aware training block by block (so it fits on a T4), 4-bit embeddings, recovery adapters; the head-to-head with Apple | |
| M3 | Real 2-bit kernels: speed and memory on a GPU and on Apple silicon (Metal) | |

## Run it

```
pip install pytest
OSMIUM_DATA=1 python -m pytest -q tests      # downloads simple-evals' MMLU file
pip install vllm==0.31.0                     # on a GPU
python scripts/eval_mmlu.py --model Qwen/Qwen3-4B --limit 500
```
