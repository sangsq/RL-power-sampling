"""Compute-matched self-consistency from repeated vLLM samples."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--limit", type=int, default=1319)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=1")

    from transformers import AutoTokenizer
    from vllm import LLM

    from power_sampling.grading import canonical_answer, grade, ground_truth, load_jsonl
    from power_sampling.sampler import PowerSampler

    examples = load_jsonl(ROOT / "data" / "gsm8k" / f"{args.split}.jsonl")[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = [tokenizer.encode(row["question"], add_special_tokens=False) for row in examples]
    flat_prompts = [prompt for prompt in prompt_ids for _ in range(args.samples)]
    started = time.time()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.25,
        max_model_len=1024,
        logprobs_mode="processed_logprobs",
        enable_prefix_caching=True,
    )
    chains = PowerSampler(llm, tokenizer).standard(
        flat_prompts, args.max_tokens, args.temperature, args.seed, independent_seeds=True
    )

    rows = []
    for index, example in enumerate(examples):
        group = chains[index * args.samples : (index + 1) * args.samples]
        responses = [tokenizer.decode(chain.tokens, skip_special_tokens=True) for chain in group]
        parsed = [canonical_answer(response) for response in responses]
        votes = Counter(answer for answer in parsed if answer is not None)
        winner = max(votes, key=lambda answer: (votes[answer], -parsed.index(answer))) if votes else None
        selected = parsed.index(winner) if winner is not None else 0
        correct, raw_answer = grade(responses[selected], ground_truth(example))
        rows.append(
            {
                "question": example["question"],
                "ground_truth": ground_truth(example),
                "responses": responses,
                "parsed_answers": parsed,
                "selected": selected,
                "parsed_answer": raw_answer,
                "correct": correct,
                "sampled_tokens": sum(chain.sampled_tokens for chain in group),
            }
        )

    elapsed = time.time() - started
    payload = {
        "config": {**vars(args), "model": MODEL, "prompt": "question_only", "physical_gpu": 1},
        "summary": {
            "n": len(rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "parse_rate": sum(row["parsed_answer"] is not None for row in rows) / len(rows),
            "mean_sampled_tokens": sum(row["sampled_tokens"] for row in rows) / len(rows),
            "elapsed_seconds": elapsed,
        },
        "rows": rows,
    }
    output = ROOT / "results" / f"{args.name}.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
