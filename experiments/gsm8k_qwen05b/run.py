"""Run one reproducible GSM8K sampling condition on physical GPU 1."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-0.5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--method", choices=("base", "low_temp", "power"), required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--proposal-temperature", type=float, default=0.25)
    parser.add_argument("--mcmc-steps", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument(
        "--suffix-policy", choices=("full", "official_overlap"), default="full"
    )
    parser.add_argument("--seed-policy", choices=("per_example", "batch"), default="per_example")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=1 so physical GPU 1 is the only visible device")

    from transformers import AutoTokenizer
    from vllm import LLM

    from power_sampling.grading import grade, ground_truth, load_jsonl
    from power_sampling.sampler import PowerSampler

    examples = load_jsonl(ROOT / "data" / "gsm8k" / f"{args.split}.jsonl")
    examples = examples[args.offset : args.offset + args.limit]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = [example["question"] for example in examples]
    prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]

    started = time.time()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=1024,
        logprobs_mode="processed_logprobs",
        enable_prefix_caching=True,
    )
    sampler = PowerSampler(llm, tokenizer)
    if args.method == "power":
        chains = sampler.power(
            prompt_ids,
            alpha=args.alpha,
            proposal_temperature=args.proposal_temperature,
            mcmc_steps=args.mcmc_steps,
            block_size=args.block_size,
            max_tokens=args.max_tokens,
            seed=args.seed,
            suffix_policy=args.suffix_policy,
        )
    else:
        temperature = args.temperature if args.method == "low_temp" else 1.0
        chains = sampler.standard(
            prompt_ids,
            args.max_tokens,
            temperature,
            args.seed,
            independent_seeds=args.seed_policy == "per_example",
        )

    rows = []
    for example, chain in zip(examples, chains):
        response = tokenizer.decode(chain.tokens, skip_special_tokens=True)
        correct, parsed = grade(response, ground_truth(example))
        rows.append(
            {
                "question": example["question"],
                "ground_truth": ground_truth(example),
                "response": response,
                "parsed_answer": parsed,
                "correct": correct,
                "response_tokens": len(chain.tokens),
                "sampled_tokens": chain.sampled_tokens,
                "acceptance_rate": chain.acceptance_rate,
                "attempts": chain.attempts,
                "mean_log_ratio": (
                    sum(chain.log_ratios) / len(chain.log_ratios) if chain.log_ratios else None
                ),
                "max_abs_log_ratio": max(map(abs, chain.log_ratios), default=None),
            }
        )

    elapsed = time.time() - started
    result = {
        "config": {**vars(args), "model": MODEL, "prompt": "question_only", "physical_gpu": 1},
        "summary": {
            "n": len(rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "parse_rate": sum(row["parsed_answer"] is not None for row in rows) / len(rows),
            "mean_response_tokens": sum(row["response_tokens"] for row in rows) / len(rows),
            "mean_sampled_tokens": sum(row["sampled_tokens"] for row in rows) / len(rows),
            "mean_acceptance_rate": sum(row["acceptance_rate"] for row in rows) / len(rows),
            "mean_log_ratio": (
                sum(sum(chain.log_ratios) for chain in chains)
                / sum(len(chain.log_ratios) for chain in chains)
                if any(chain.log_ratios for chain in chains)
                else None
            ),
            "max_abs_log_ratio": max(
                (abs(value) for chain in chains for value in chain.log_ratios), default=None
            ),
            "elapsed_seconds": elapsed,
        },
        "rows": rows,
    }
    output = ROOT / "results" / f"{args.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    (ROOT / "notes").mkdir(exist_ok=True)
    with (ROOT / "notes" / "ACTIVITY_LOG.md").open("a") as handle:
        handle.write(
            f"- {timestamp} — Finished {args.name}: accuracy={result['summary']['accuracy']:.4f}, "
            f"parse={result['summary']['parse_rate']:.4f}, elapsed={elapsed / 60:.1f} min.\n"
        )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
