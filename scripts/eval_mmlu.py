"""MMLU as simple-evals runs it (osmium/simple_evals.py), served by vLLM on one GPU.

M0 runs it on the open models in Apple's Table 1, to check that this harness reproduces the
numbers Apple reported for them before any compressed model is compared with Apple's.
Decoding is greedy (temperature 0) so a run repeats exactly; the instruct models answer in their
non-thinking mode, as in Apple's report.

    python scripts/eval_mmlu.py --model Qwen/Qwen3-4B [--limit 500] [--max-tokens 2048]
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from osmium import simple_evals  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# model -> (revision, MMLU Apple reported for it, arXiv 2507.13575 Table 1)
MODELS = {
    "Qwen/Qwen3-4B": ("1cfa9a7208912126459214e8b04321603b3df60c", 75.10),
    "Qwen/Qwen2.5-3B-Instruct": ("aa8e72537993ba99e69dfaafa59ed015b17504d1", 66.37),
    "google/gemma-3-4b-it": ("093f9f388b31de276ce2de164bdc2081324b9767", 62.81),
}
APPLE = {"AFM on-device, 16-bit": 67.85, "AFM on-device, 2-bit (shipped)": 64.4}


def chat(tok, text):
    messages = [{"role": "user", "content": text}]
    try:  # Qwen3: answer without a thinking block, as Apple evaluated it
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--limit", type=int, help="a seeded sample of questions, as simple-evals' num_examples")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "m0"))
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    revision = a.revision or MODELS.get(a.model, (None, None))[0]
    rows = simple_evals.questions()
    if a.limit:
        rows = random.Random(0).sample(rows, a.limit)  # simple-evals' own sampling
    llm = LLM(model=a.model, revision=revision, dtype="float16", max_model_len=a.max_model_len,
              gpu_memory_utilization=0.9, seed=0)
    tok = llm.get_tokenizer()
    prompts = [chat(tok, simple_evals.prompt(r)) for r in rows]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=a.max_tokens))
    seconds = time.time() - t0
    responses = [o.outputs[0].text for o in outs]
    result = simple_evals.score(rows, responses)
    generated = sum(len(o.outputs[0].token_ids) for o in outs)
    result.update({
        "model": a.model, "revision": revision, "sampling": "greedy", "max_tokens": a.max_tokens,
        "no_answer": sum(simple_evals.extract_answer(r) is None for r in responses),
        "truncated": sum(o.outputs[0].finish_reason == "length" for o in outs),
        "mean_output_tokens": round(generated / len(outs), 1), "seconds": round(seconds, 1),
        "apple_reported": MODELS.get(a.model, (None, None))[1],
    })
    name = a.model.replace("/", "__")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"mmlu_{name}.jsonl"), "w", encoding="utf-8") as f:
        for r, resp in zip(rows, responses):
            f.write(json.dumps({"subject": r["Subject"], "answer": r["Answer"], "extracted": simple_evals.extract_answer(resp),
                                "response": resp}, ensure_ascii=False) + "\n")
    json.dump(result, open(os.path.join(a.out, f"mmlu_{name}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "categories"}), flush=True)


if __name__ == "__main__":
    main()
